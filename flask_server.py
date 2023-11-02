# pip install Flask
# FLASK_ENV=development FLASK_APP=flask_server.py flask run

from flask import Flask, jsonify, request
import pickle
app = Flask(__name__)
import io
import torch
from zsp.inference import frames_to_relative_pose
# Descriptor extractor
from zsp.method.zero_shot_pose import DescriptorExtractor, ZeroShotPoseMethod

take_best_view=False
ransac_thresh=0.2
n_target=10
binning='log'
patch_size=8
num_correspondences=50
kmeans=True
best_frame_mode='corresponding_feats_similarity'
device = 'cuda:0'
# ---------------
# SET UP DESCRIPTOR CLASS
# ---------------
print('init model...')
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
print('loading model...')
#  'dino_vitbase16_pretrain.pth', 'dino_deitsmall8_pretrain.pth'
fpath_model = 'dino_deitsmall8_pretrain.pth'
from pathlib import Path
print(f'fpath to model exists: {Path(fpath_model).exists()}')
desc.load_model(fpath_model, device)
# ---------------
# SET UP ZERO-SHOT POSE CLASS
# ---------------

print('init method...')
pose = ZeroShotPoseMethod(
    batched_correspond=True,
    num_plot_examples_per_batch=1,
    saliency_map_thresh=0.1,
    ransac_thresh=ransac_thresh,
    n_target=n_target,
    num_correspondences=num_correspondences,
    take_best_view=take_best_view,
)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if not file:
            return "No selected file"

        file_bytes = request.files['file'].read()
        bytes_io = io.BytesIO(file_bytes)
        data = pickle.load(bytes_io)
        print(data)

        obj2_tform_obj1, all_imgs = frames_to_relative_pose(
            desc=desc, pose=pose,
            ref_image = data['ref_image'],
            all_target_images = data['all_target_images'],
            ref_scalings = data['ref_scalings'],
            target_scalings = data['target_scalings'],
            ref_depth_map = data['ref_depth_map'],
            target_depth_map = data['target_depth_map'],
            ref_cam_extr = data['ref_cam_extr'],
            target_cam_extr = data['target_cam_extr'],
            ref_cam_intr = data['ref_cam_intr'],
            target_cam_intr = data['target_cam_intr'],
            n_target=n_target, device = device)

        data_return = {'obj2_tform_obj1': obj2_tform_obj1, 'all_imgs': all_imgs}
        data_return_json = {}
        for key, value in data_return.items():
            if isinstance(value, torch.Tensor):
                data_return_json[key] = value.tolist()
            else:
                data_return_json[key] = value
        #data_return_json['status_code'] = 200
        #data_return_json['message'] = 'OK'
        return jsonify(data_return_json)
