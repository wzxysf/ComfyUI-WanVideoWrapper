import os
import torch
import gc
from ..utils import log, dict_to_device
import numpy as np
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import comfy.model_management as mm
from comfy.utils import load_torch_file
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__)) 
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

from .resampler import Resampler

class LoadLynxResampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from 'ComfyUI/models/diffusion_models'"}),
                "precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("LYNXRESAMPLER",)
    RETURN_NAMES = ("resampler", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoWrapper"

    def loadmodel(self, model_name, precision):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_path = folder_paths.get_full_path("diffusion_models", model_name)
        resampler_sd = load_torch_file(model_path, safe_load=True)

        output_dim = resampler_sd["proj_out.weight"].shape[0]

        resampler = Resampler(
            depth=4,
            dim=1280,
            dim_head=64,
            embedding_dim=512,
            ff_mult=4,
            heads=20,
            num_queries=16,
            output_dim=output_dim,
            dtype=dtype,
        ).eval()
        resampler.to(offload_device, dtype)
        resampler.load_state_dict(resampler_sd, strict=True)
        #for name, param in resampler.named_parameters():
        #    print(f"{name}: {param.shape} {param.dtype}")

        return resampler,


class VideoStyleInfo:  # key names should match those used in style.yaml file
    style_name: str = 'none'
    num_frames: int = 81
    seed: int = -1
    guidance_scale: float = 5.0
    guidance_scale_i: float = 2.0
    num_inference_steps: int = 50
    width: int = 832
    height: int = 480
    prompt: str = ''
    negative_prompt: str = ''

class LynxEncodeFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resampler": ("LYNXRESAMPLER", {"tooltip": "lynx resampler model"}),
                "image": ("IMAGE", {"tooltip": "Input images for the model"}),
            },
        }

    RETURN_TYPES = ("LYNXFACE", "IMAGE",)
    RETURN_NAMES = ("lynx_face_embeds", "processed_image")
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"

    def encode(self, resampler, image):
        from .face.face_encoder import FaceEncoderArcFace, get_landmarks_from_image

        image_np = (image[0].numpy() * 255).astype(np.uint8)
        # Landmarks
        
        landmarks = np.array([
            [200.509, 213.98592],
            [297.2495, 212.67685],
            [245.74419, 272.85718],
            [212.2043, 331.09564],
            [288.75986, 330.27188]
        ])

        # landmarks = np.array([[379.35895, 381.4803],
        #                       [542.8676, 362.75436],
        #                       [451.62717, 467.9116],
        #                       [407.03708, 555.56305],
        #                       [554.57874, 539.22833]])

        #landmarks = get_landmarks_from_image(image_np)

        print(landmarks)

        # Face embedding via ArcFace
        face_encoder = FaceEncoderArcFace()
        face_encoder.init_encoder_model(device)
        arcface_embed, processed_image = face_encoder(image_np, need_proc=True, landmarks=landmarks)

        arcface_embed = arcface_embed.to(device, resampler.dtype)
        arcface_embed = arcface_embed.reshape([1, -1, 512])

        

        resampler.to(device)
        ip_x = resampler(arcface_embed)
        ip_x_uncond = resampler(arcface_embed * 0)
        resampler.to(offload_device)

        #ip_x = torch.load(os.path.join(script_directory, "debug_face_embeds.pt"))
        #ip_x = ip_x[1].unsqueeze(0)
        ip_x= ip_x.to(resampler.dtype)

        out_dict = {
            'ip_x': ip_x,
            'ip_x_uncond': ip_x_uncond,
            "landmarks": landmarks,
        }

        print("processed_image.shape", processed_image.min(), processed_image.max())
        processed_image = (processed_image - processed_image.min()) / (processed_image.max() - processed_image.min())

        processed_image = processed_image.permute(0, 2, 3, 1)
       
        return out_dict, processed_image

class DrawArcFaceLandmarks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lynx_face_embeds": ("LYNXFACE", {"tooltip": "lynx resampler model"}),
                "image": ("IMAGE", {"tooltip": "Input images for the model"}),
            },
            "optional": {
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("landmarked_image", )
    FUNCTION = "draw"
    CATEGORY = "WanVideoWrapper"

    def draw(self, lynx_face_embeds, image):
        import cv2
        landmarks = lynx_face_embeds['landmarks']
        image_np = image[0].numpy() * 255
        print("image_np.shape", image_np.shape) #image_np.shape (3, 512, 512)
        print(type(landmarks))
        print(landmarks)

        for (x, y) in landmarks:
            cv2.circle(image_np, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

        image_out = torch.from_numpy(image_np / 255).unsqueeze(0).float()
        print(image_out.shape) #torch.Size([1, 3, 512, 512])

        return image_out,

class WanVideoAddLynxEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds": ("WANVIDIMAGE_EMBEDS",),
                    "lynx_embeds": ("LYNXFACE", {"tooltip": "lynx face embeddings"}),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Strength of the MTV motion"}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percent to apply the ref "}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percent to apply the ref "}),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "WanVideoWrapper"

    def add(self, embeds, lynx_embeds, strength, start_percent, end_percent):
        new_entry = {
            "ip_x": lynx_embeds["ip_x"],
            "ip_x_uncond": lynx_embeds["ip_x_uncond"],
            "strength": strength,
            "start_percent": start_percent,
            "end_percent": end_percent,
        }

        updated = dict(embeds)
        updated["lynx_embeds"] = new_entry
        return (updated,)

NODE_CLASS_MAPPINGS = {
    "LoadLynxResampler": LoadLynxResampler,
    "LynxEncodeFace": LynxEncodeFace,
    "DrawArcFaceLandmarks": DrawArcFaceLandmarks,
    "WanVideoAddLynxEmbeds": WanVideoAddLynxEmbeds,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLynxResampler": "Load Lynx Resampler",
    "LynxEncodeFace": "Lynx Encode Face",
    "DrawArcFaceLandmarks": "Draw ArcFace Landmarks",
    "WanVideoAddLynxEmbeds": "WanVideo Add Lynx Embeds",
}
