import torch
import numpy as np
# Descriptor extractor
from zsp.method.zero_shot_pose import DescriptorExtractor, ZeroShotPoseMethod
from pytorch3d.renderer.cameras import get_world_to_view_transform

# Utils
from zsp.method.zero_shot_pose_utils import (
    scale_points_to_orig,
    get_structured_pcd,
    trans21_error,
)

def frames_to_relative_pose(ref_image: torch.Tensor, all_target_images: torch.Tensor,
                            ref_scalings: torch.Tensor, target_scalings: torch.Tensor,
                            ref_depth_map: torch.Tensor, target_depth_map: torch.Tensor,
                            take_best_view=True,
                            ransac_thresh=0.2, n_target=5, binning='log', patch_size=8, num_correspondences=50,
                            kmeans=True, best_frame_mode='corresponding_feats_similarity', device='cpu'):
    """
    Args:
        ref_image (torch.Tensor): Bx3xSxS
        all_target_images (torch.Tensor): N_TGTx3xSxS
        ref_scalings (torch.Tensor): Bx2 (H,W) rescaling from original resolution
        target_scalings (torch.Tensor): N_TGTx2 (H,W) rescaling from original resolution
        ref_depth_map (torch.Tensor): Bx1xHxW
        target_depth_map (torch.Tensor): N_TGTx1xHxW
        ref_camera (torch.Tensor): Bx4x4
        target_camera (torch.Tensor): N_TGTx4x4
    Returns:
        target_tform_ref (torch.Tensor): Bx4x4


    ##
        depth_map: A float Tensor of shape `(1, H, W)` holding the depth map
            of the frame; values correspond to distances from the camera;
            use `depth_mask` and `mask_crop` to filter for valid pixels.
        camera: A PyTorch3D camera object corresponding the frame's viewpoint,
            corrected for cropping if it happened.
    """

    # ---------------
    # SET UP DESCRIPTOR CLASS
    # ---------------
    desc = DescriptorExtractor(
        patch_size=patch_size,
        feat_layer=9,
        high_res=False,
        binning=binning,
        image_size=224,
        n_target=n_target,
        saliency_map_thresh=0.1,
        num_correspondences=num_correspondences,
        kmeans=kmeans,
        best_frame_mode=best_frame_mode
    )
    # ---------------
    # SET UP ZERO-SHOT POSE CLASS
    # ---------------
    pose = ZeroShotPoseMethod(
        batched_correspond=True,
        num_plot_examples_per_batch=1,
        saliency_map_thresh=0.1,
        ransac_thresh=ransac_thresh,
        n_target=n_target,
        num_correspondences=num_correspondences,
        take_best_view=take_best_view,
    )

    # ---------------
    # GET FEATURES
    # ---------------
    # Ref images shape: B x 3 x S x S (S = size, assumed square images)
    # Target images shape: B x N_TGT x 3 x S x S
    batch_size = ref_image.size(0)
    all_images = torch.cat([ref_image.unsqueeze(1), all_target_images], dim=1).to(device)  # B x (N_TGT + 1) x 3 x S x S
    # Extract features, attention maps, and cls_tokens
    features, attn, output_cls_tokens = desc.extract_features_and_attn(all_images)
    # Create descriptors from features, return descriptors and attn in appropriate shapes
    # attn shape Bx(n_tgt+1)xhxtxt, features shape Bx(n_tgt+1)x1x(t-1)xfeat_dim
    features, attn = desc.create_reshape_descriptors(features, attn, batch_size, device)
    # Split ref/target, repeat ref to match size of target, and flatten into batch dimension
    ref_feats, target_feats, ref_attn, target_attn = desc.split_ref_target(features, attn)

    # ----------------
    # GET CORRESPONDENCES
    # ----------------
    (selected_points_image_2,  # 10x50x2
     selected_points_image_1,  # 10x50x2
     cyclical_dists,  # 10x28x28
     sim_selected_12) = desc.get_correspondences(ref_feats, target_feats, ref_attn, target_attn, device)
    #  sim_selected_12 has shape 10x50
    # ----------------
    # FIND BEST IMAGE IN TARGET SEQ
    # ----------------
    _, _, _, t, t = attn.size()
    N = int(np.sqrt(t - 1))  # N is the height or width of the feature map
    similarities, best_idxs = desc.find_closest_match(attn, output_cls_tokens, sim_selected_12, batch_size)
    # -----------------
    # COMPUTE POSE OFFSET
    # -----------------
    all_trans21 = []
    all_errs = []
    all_points1 = []
    all_points2 = []
    for i in range(batch_size):
        # -----------------
        # PREPARE DATA
        # -----------------
        # Get other data required to compute pose offset
        target_frame = best_idxs[i]
        # cls = dataset.samples[uq_idx[i]]['class']

        ref_scaling = ref_meta_data['scalings'][i]
        ref_depth = ref_meta_data['depth_maps'][i]
        ref_camera = ref_meta_data['cameras'][i]
        #ref_pcd = ref_meta_data['pcd'][i]
        #ref_trans = ref_transform[i]

        target_scaling = target_meta_data['scalings'][i][target_frame]
        target_depth = target_meta_data['depth_maps'][i][target_frame]
        target_camera = target_meta_data['cameras'][i][target_frame]
        #target_pcd = target_meta_data['pcd'][i]
        #target_trans = target_transform[i]

        # NB: from here on, '1' <--> ref and '2' <--> target
        # Get points and if necessary scale them from patch to pixel space
        points1, points2 = (
            selected_points_image_1[i * n_target + target_frame],
            selected_points_image_2[i * n_target + target_frame]
        )
        points1_rescaled, points2_rescaled = desc.scale_patch_to_pix(
            points1, points2, N
        )
        all_points1.append(points1_rescaled.clone().int().long())
        all_points2.append(points2_rescaled.clone().int().long())
        # Now rescale to the *original* image pixel size, i.e. prior to resizing crop to square
        points1_rescaled, points2_rescaled = (scale_points_to_orig(p, s) for p, s in zip(
            (points1_rescaled, points2_rescaled), (ref_scaling, target_scaling)
        ))
        # --- If "take best view", simply return identity transform estimate ---
        if pose.take_best_view:
            trans21 = get_world_to_view_transform()
        # --- Otherwise, compute transform based on 3D point correspondences ---
        else:
            frame1 = {
                'shape': ref_depth.shape[1:],
                'scaling': ref_scaling,
                'depth_map': ref_depth,
                'camera': ref_camera
            }

            frame2 = {
                'shape': target_depth.shape[1:],
                'scaling': target_scaling,
                'depth_map': target_depth,
                'camera': target_camera
            }

            struct_pcd1 = get_structured_pcd(frame1, world_coordinates=False)
            struct_pcd2 = get_structured_pcd(frame2, world_coordinates=False)

            world_corr1 = struct_pcd1[points1_rescaled[:, 0], points1_rescaled[:, 1]].numpy()
            world_corr2 = struct_pcd2[points2_rescaled[:, 0], points2_rescaled[:, 1]].numpy()

            # -----------------
            # COMPUTE RELATIVE OFFSET
            # -----------------
            trans21 = pose.solve_umeyama_ransac(world_corr1, world_corr2)
        all_trans21.append(trans21)
    return trans21