# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np

__all__ = [
    "FaceEncoderArcFace",
    "get_landmarks_from_image",
]


detector = None

def get_landmarks_from_image(image):
    """
    Detect landmarks with insightface.

    Args:
        image (np.ndarray or PIL.Image):
            The input image in RGB format.

    Returns:
        5 2D keypoints, only one face will be returned.
    """
    from insightface.app import FaceAnalysis
    global detector
    if detector is None:
        detector = FaceAnalysis()
        detector.prepare(ctx_id=0, det_size=(640, 640))

    in_image = np.array(image).copy()
    
    faces = detector.get(in_image)
    if len(faces) == 0:
        raise ValueError("No face detected in the image")
    
    # Get the largest face
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    
    # Return the 5 keypoints directly
    keypoints = face.kps  # 5 x 2

    return keypoints

from facexlib.utils import load_file_from_url
from facexlib.recognition.arcface_arch import Backbone

def init_recognition_model(model_name, half=False, device='cuda', model_rootpath=None):
    print("Initializing recognition model:", model_name)
    if model_name == 'arcface':
        model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se').to('cuda').eval()
        model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/recognition_arcface_ir_se50.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)
    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model

class FaceEncoderArcFace():
    """ Official ArcFace, no_grad-only """

    def __repr__(self):
        return "ArcFace"

    def init_encoder_model(self, device, eval_mode=True):
        self.device = device
        self.encoder_model = init_recognition_model('arcface', device=device)

        if eval_mode:
            self.encoder_model.eval()

    def __call__(self, in_image):
        return self.encoder_model(in_image[:, [2, 1, 0], :, :].contiguous()) # [B, 512], normalized