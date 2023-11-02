import torch
import numpy as np

from pytorch3d.renderer.cameras import get_world_to_view_transform
from pytorch3d.renderer import PerspectiveCameras

# Utils
from zsp.method.zero_shot_pose_utils import (
    scale_points_to_orig,
    get_structured_pcd,
    trans21_error,
)

import os
import matplotlib.pyplot as plt
from zsp.utils.visuals import (
    plot_pcd,
    tile_ims_horizontal_highlight_best,
    draw_correspondences_lines
)

def frames_to_relative_pose(desc, pose, ref_image: torch.Tensor, all_target_images: torch.Tensor,
                            ref_scalings: torch.Tensor, target_scalings: torch.Tensor,
                            ref_depth_map: torch.Tensor, target_depth_map: torch.Tensor,
                            ref_cam_extr: torch.Tensor, target_cam_extr: torch.Tensor,
                            ref_cam_intr: torch.Tensor, target_cam_intr: torch.Tensor,
                            n_target=10, device='cpu'):
    """
    Args:
        ref_image (torch.Tensor): Bx3xSxS
        all_target_images (torch.Tensor): BxN_TGTx3xSxS
        ref_scalings (torch.Tensor): Bx2 (H,W) rescaling from original resolution
        target_scalings (torch.Tensor): BxN_TGTx2 (H,W) rescaling from original resolution
        ref_depth_map (torch.Tensor): Bx1xHxW
        target_depth_map (torch.Tensor): BxN_TGTx1xHxW
        ref_cam_extr (torch.Tensor): Bx4x4
        target_cam_extr (torch.Tensor): N_TGTx4x4
        ref_cam_intr (torch.Tensor): Bx4x4
        target_cam_intr (torch.Tensor): N_TGTx4x4
    Returns:
        target_tform_ref (torch.Tensor): Bx4x4


    ##
        depth_map: A float Tensor of shape `(1, H, W)` holding the depth map
            of the frame; values correspond to distances from the camera;
            use `depth_mask` and `mask_crop` to filter for valid pixels.
        camera: A PyTorch3D camera object corresponding the frame's viewpoint,
            corrected for cropping if it happened.
    """

    B = ref_image.size(0)
    N_TGT = all_target_images.size(1)
    img_size = torch.LongTensor(list(ref_image.shape[-2:]))
    print('B: ', B, ' N_TGT:', N_TGT)
    # 1. transform to PIL
    # 2. desc.transform
    import torchvision
    pil_transfrom = torchvision.transforms.ToPILImage()
    desc_transform = desc.get_transform()
    transform = torchvision.transforms.Compose([pil_transfrom, desc_transform])
    all_target_images = torch.stack([transform(all_target_images[b, n]) for b in range(B) for n in range(N_TGT)], dim=0).reshape(B, N_TGT, 3, *img_size).to(device=device)
    ref_image = torch.stack([transform(ref_image[b]) for b in range(B)], dim=0).to(device=device)

    #return Image.fromarray((image_rgb.permute(1, 2, 0) * 255).numpy().astype(np.uint8))
    # image_norm_mean = (0.485, 0.456, 0.406)
    # image_norm_std = (0.229, 0.224, 0.225)
    # image_size = 224  # Image size
    # image_transform = transforms.Compose([
    #     transforms.Resize((image_size, image_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=image_norm_mean, std=image_norm_std)
    # ])
    # ---------------
    # GET FEATURES
    # ---------------
    print('get feats....')
    # Ref images shape: B x 3 x S x S (S = size, assumed square images)
    # Target images shape: B x N_TGT x 3 x S x S
    all_images = torch.cat([ref_image.unsqueeze(1), all_target_images], dim=1).to(device)  # B x (N_TGT + 1) x 3 x S x S
    # Extract features, attention maps, and cls_tokens
    features, attn, output_cls_tokens = desc.extract_features_and_attn(all_images)
    # Create descriptors from features, return descriptors and attn in appropriate shapes
    # attn shape Bx(n_tgt+1)xhxtxt, features shape Bx(n_tgt+1)x1x(t-1)xfeat_dim
    features, attn = desc.create_reshape_descriptors(features, attn, B, device)
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
    similarities, best_idxs = desc.find_closest_match(attn, output_cls_tokens, sim_selected_12, B)
    # -----------------
    # COMPUTE POSE OFFSET
    # -----------------
    all_trans21 = []
    all_imgs = []
    all_errs = []
    all_points1 = []
    all_points2 = []
    for i in range(B):
        # -----------------
        # PREPARE DATA
        # -----------------
        # Get other data required to compute pose offset
        target_frame = best_idxs[i]
        # cls = dataset.samples[uq_idx[i]]['class']

        # ref_scaling = ref_scalings[i] #  ref_meta_data['scalings'][i]
        ref_depth = ref_depth_map[i] #  ref_meta_data['depth_maps'][i]
        ref_camera = get_perspective_camera(cam_tform_obj=ref_cam_extr[i], cam_intr=ref_cam_intr[i], img_size=img_size)  # ref_dep ref_meta_data['cameras'][i]
        #ref_pcd = ref_meta_data['pcd'][i]
        #ref_trans = ref_transform[i]

        #target_scaling = target_scalings[i] #  target_meta_data['scalings'][i][target_frame]
        target_depth = target_depth_map[i][target_frame]  # target_meta_data['depth_maps'][i][target_frame]
        target_camera = get_perspective_camera(cam_tform_obj=target_cam_extr[i][target_frame], cam_intr=target_cam_intr[i][target_frame], img_size=img_size)  #  ref_cam_extr[i][target_frame], ref_cam_intr[i][target_frame]  # target_meta_data['cameras'][i][target_frame]
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
            (points1_rescaled, points2_rescaled), (ref_scalings, target_scalings)
        ))
        # --- If "take best view", simply return identity transform estimate ---
        if pose.take_best_view:
            trans21 = get_world_to_view_transform()
        # --- Otherwise, compute transform based on 3D point correspondences ---
        else:
            frame1 = {
                'shape': ref_depth.shape[1:],
                'scaling': ref_scalings,
                'depth_map': ref_depth,
                'camera': ref_camera
            }

            frame2 = {
                'shape': target_depth.shape[1:],
                'scaling': target_scalings,
                'depth_map': target_depth,
                'camera': target_camera
            }

            struct_pcd1 = get_structured_pcd(frame1, world_coordinates=True)
            struct_pcd2 = get_structured_pcd(frame2, world_coordinates=True)

            world_corr1 = struct_pcd1[points1_rescaled[:, 0], points1_rescaled[:, 1]].numpy()
            world_corr2 = struct_pcd2[points2_rescaled[:, 0], points2_rescaled[:, 1]].numpy()

            # -----------------
            # COMPUTE RELATIVE OFFSET
            # -----------------
            trans21 = pose.solve_umeyama_ransac(world_corr1, world_corr2)
            #target_obj_tform_ref_obj = target_camera.get_world_to_view_transform().inverse().compose(trans21).compose(ref_camera.get_world_to_view_transform())
            #trans21 = target_obj_tform_ref_obj.get_matrix()[0].T
            trans21 = trans21.get_matrix()[0].T
            print(trans21.shape)
        all_trans21.append(trans21)

        #save_name = f'sample_{i}.png'
        #fig_dir = 'plots'
        # save_name = os.path.join(fig_dir, save_name)

        fig, axs = plt.subplot_mosaic([['A', 'B', 'B'],
                                       ['C', 'C', 'D']],
                                      figsize=(10, 5))
        for ax in axs.values():
            ax.axis('off')
        axs['A'].set_title('Reference image')
        axs['B'].set_title('Query images')
        axs['C'].set_title('Correspondences')
        axs['D'].set_title('Reference object in query pose')
        fig.suptitle(f'Error: ', fontsize=6) # {all_errs[i]:.2f}
        axs['A'].imshow(desc.denorm_torch_to_pil(ref_image[i].detach().cpu()))
        # ax[1].plot(similarities[i].cpu().numpy())
        tgt_pils = [desc.denorm_torch_to_pil(
            all_target_images[i][j].detach().cpu()) for j in range(N_TGT)]
        tgt_pils = tile_ims_horizontal_highlight_best(tgt_pils, highlight_idx=best_idxs[i].detach().cpu())
        axs['B'].imshow(tgt_pils)

        draw_correspondences_lines(all_points1[i].detach().cpu(), all_points2[i].detach().cpu(),
                                   desc.denorm_torch_to_pil(ref_image[i].detach().cpu()),
                                   desc.denorm_torch_to_pil(all_target_images[i][best_idxs[i]].detach().cpu()),
                                   axs['C'])

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        all_imgs.append(torch.from_numpy(data))

        #pcd1 = ref_meta_data['pcd'][i]
        #pcd2 = target_meta_data['pcd'][i]
        #trans21 = all_trans21[i]
        #cam1 = ref_meta_data['cameras'][i]
        #trans = cam1.get_world_to_view_transform().compose(trans21)
        #X1, numpoints1 = oputil.convert_pointclouds_to_tensor(pcd1)
        #trans_X1 = trans.transform_points(X1)
        ## Project the cam2points to NDC+then into screen
        #P_im = transform_cameraframe_to_screen(cam1, trans_X1, image_size=(500, 500))
        #P_im = P_im.squeeze()
        #plot_pcd(P_im, pcd1, axs['D'])

        #plt.tight_layout()
        #plt.savefig(save_name + '.png', dpi=150)
        plt.close('all')

    all_trans21 = torch.stack(all_trans21, dim=0)
    all_imgs = torch.stack(all_imgs, dim=0)
    return all_trans21, all_imgs

def get_perspective_camera(cam_tform_obj: torch.Tensor, cam_intr: torch.Tensor, img_size: torch.Tensor):
    """
    Args:
        cam_tform_obj (torch.Tensor): 4x4
        cam_intr (torch.Tensor): 4x4
        img_size (torch.Tensor): 2,

    Returns:
        cameras (PerspectiveCameras): B,...
    """
    dtype = cam_tform_obj.dtype
    device = cam_tform_obj.device

    focal_length = torch.Tensor([cam_intr[0, 0], cam_intr[1, 1]]).to(device=device, dtype=dtype)
    principal_point = torch.Tensor([cam_intr[0, 2], cam_intr[1, 2]]).to(device=device, dtype=dtype)

    # scale=1.
    # half_image_size_output = torch.tensor(img_size, dtype=torch.float) / 2.0
    # half_min_image_size_output = half_image_size_output.min()
    # # rescaled principal point and focal length in ndc
    # principal_point = (half_image_size_output - principal_point * scale) / half_min_image_size_output
    # focal_length = focal_length * scale / half_min_image_size_output

    t3d_tform_pscl3d = torch.Tensor([[-1., 0., 0., 0.],
                                     [0., -1., 0., 0.],
                                     [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]]).to(device=device, dtype=dtype)
    t3d_cam_tform_obj = torch.matmul(t3d_tform_pscl3d, cam_tform_obj)

    R = t3d_cam_tform_obj[:3, :3].T[None,]  # .T[None,]
    t = t3d_cam_tform_obj[:3, 3][None,]

    cameras = PerspectiveCameras(device=device, R=R, T=t, focal_length=focal_length[None,],
                                 principal_point=principal_point[None,], in_ndc=False,
                                 image_size=img_size[None,])
    return cameras
