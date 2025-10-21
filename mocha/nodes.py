import torch
from comfy import model_management as mm
import gc

def rope_params_mocha(max_seq_len, dim, theta=10000, L_test=25, k=0, start=0):
    assert dim % 2 == 0
    exponents = torch.arange(0, dim, 2, dtype=torch.float64).div(dim)
    inv_theta_pow = 1.0 / torch.pow(theta, exponents)
    
    if k > 0:
        print(f"RifleX: Using {k}th freq")
        inv_theta_pow[k-1] = 0.9 * 2 * torch.pi / L_test
        
    freqs = torch.outer(torch.arange(start, max_seq_len), inv_theta_pow)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@torch.autocast(device_type=mm.get_autocast_device(mm.get_torch_device()), enabled=False)
def rope_apply_mocha(x, grid_sizes, freqs):
    batch_size, _, n, c_doubled = x.shape
    c = c_doubled // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    f_tensor = grid_sizes[0, 0]
    h_tensor = grid_sizes[0, 1] 
    w_tensor = grid_sizes[0, 2]
    
    seq_len_tensor = f_tensor * h_tensor * w_tensor
    sf_tensor = (f_tensor - 2) // 2

    sf_range = torch.arange(1, sf_tensor + 1, device=freqs[0].device)
    h_range = torch.arange(1, h_tensor + 1, device=freqs[1].device)
    w_range = torch.arange(1, w_tensor + 1, device=freqs[2].device)
    
    repeat_freqs = torch.cat([
        freqs[0][sf_range].view(sf_tensor, 1, 1, -1).expand(sf_tensor, h_tensor, w_tensor, -1),
        freqs[1][h_range].view(1, h_tensor, 1, -1).expand(sf_tensor, h_tensor, w_tensor, -1),
        freqs[2][w_range].view(1, 1, w_tensor, -1).expand(sf_tensor, h_tensor, w_tensor, -1)
    ], dim=-1)

    mask_freqs = torch.cat([
        freqs[0][1:2].view(1, 1, 1, -1).expand(1, h_tensor, w_tensor, -1),
        freqs[1][h_range].view(1, h_tensor, 1, -1).expand(1, h_tensor, w_tensor, -1),
        freqs[2][w_range].view(1, 1, w_tensor, -1).expand(1, h_tensor, w_tensor, -1)
    ], dim=-1)

    img_freqs = torch.cat([
        freqs[0][0:1].view(1, 1, 1, -1).expand(1, h_tensor, w_tensor, -1),
        freqs[1][h_range].view(1, h_tensor, 1, -1).expand(1, h_tensor, w_tensor, -1),
        freqs[2][w_range].view(1, 1, w_tensor, -1).expand(1, h_tensor, w_tensor, -1)
    ], dim=-1)

    condition = (f_tensor == 2 * sf_tensor + 2)
    
    bias_h_range = torch.arange(h_tensor + 1, 2 * h_tensor + 1, device=freqs[1].device)
    bias_w_range = torch.arange(w_tensor + 1, 2 * w_tensor + 1, device=freqs[2].device)
    
    bias_freqs = torch.cat([
        freqs[0][0:1].view(1, 1, 1, -1).expand(1, h_tensor, w_tensor, -1),
        freqs[1][bias_h_range].view(1, h_tensor, 1, -1).expand(1, h_tensor, w_tensor, -1),
        freqs[2][bias_w_range].view(1, 1, w_tensor, -1).expand(1, h_tensor, w_tensor, -1)
    ], dim=-1)
    
    freqs_without_bias = torch.cat([repeat_freqs, repeat_freqs, mask_freqs, img_freqs], dim=0)
    freqs_with_bias = torch.cat([repeat_freqs, repeat_freqs, mask_freqs, img_freqs, bias_freqs], dim=0)
    
    pad_size = freqs_with_bias.size(0) - freqs_without_bias.size(0)
    padding = torch.zeros(
        pad_size, h_tensor, w_tensor, freqs_without_bias.size(-1),
        dtype=freqs_without_bias.dtype, device=freqs_without_bias.device
    )
    freqs_without_bias = torch.cat([freqs_without_bias, padding], dim=0)
    
    freqs_i = torch.where(condition.unsqueeze(0).unsqueeze(0).unsqueeze(0), freqs_without_bias, freqs_with_bias)
    freqs_i = freqs_i.reshape(seq_len_tensor, 1, -1).to(x.device)

    x_seq = x[:, :seq_len_tensor].to(torch.float64).reshape(batch_size, seq_len_tensor, n, -1, 2)
    x_complex = torch.view_as_complex(x_seq)
    
    x_rotated = torch.view_as_real(x_complex * freqs_i.unsqueeze(0)).flatten(3)
    
    x_remaining = x[:, seq_len_tensor:]
    output = torch.cat([x_rotated, x_remaining], dim=1)
    
    return output.to(x.dtype)


device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

class MochaEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("WANVAE",),
                "force_offload": ("BOOLEAN", {"default": True}),
                "input_video": ("IMAGE", {"tooltip": "Input video to encode"}),
                "mask": ("MASK", {"tooltip": "mask"}),
                "ref1": ("IMAGE", {"tooltip": "Image to encode"}),
            },
            "optional": {
                "ref2": ("IMAGE", {"tooltip": "Image to encode"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
            }
        }
    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)

    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Input for MoCha model: https://github.com/Orange-3DV-Team/MoCha"

    def process(self, vae, force_offload, input_video, mask, ref1, ref2=None, tiled_vae=False):
        W = input_video.shape[2]
        H = input_video.shape[1]
        F = input_video.shape[0]

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        F = (F - 1) // 4 * 4 + 1
        input_video = input_video.clone()[: F]

        mm.soft_empty_cache()
        gc.collect()
        vae.to(device)

        input_video = input_video.to(device, vae.dtype).unsqueeze(0).permute(0, 4, 1, 2, 3)
        ref1 = ref1.clone().to(device, vae.dtype).unsqueeze(0).permute(0, 4, 1, 2, 3)
        if ref2 is not None:
            ref2 = ref2.clone().to(device, vae.dtype).unsqueeze(0).permute(0, 4, 1, 2, 3)


        latents = vae.encode(input_video * 2.0 - 1.0, device, tiled=tiled_vae)
        
        ref_latents = vae.encode(ref1 * 2.0 - 1.0, device, tiled=tiled_vae)
        num_refs = 1
        if ref2 is not None:
            ref2_latents = vae.encode(ref2 * 2.0 - 1.0, device, tiled=tiled_vae)
            ref_latents = torch.cat([ref_latents, ref2_latents], dim=2)
            num_refs = 2
        

        input_latent_mask = torch.nn.functional.interpolate(mask.unsqueeze(1).to(vae.dtype), size=(lat_h, lat_w), mode='nearest').unsqueeze(1)
        input_latent_mask = input_latent_mask.repeat(1, 16, 1, 1, 1).to(device, vae.dtype)

        input_latent_mask[input_latent_mask <= 0.5] = 0
        input_latent_mask[input_latent_mask > 0.5] = 1
        input_latent_mask[input_latent_mask == 0] = -1

        mocha_embeds = torch.cat([latents, input_latent_mask, ref_latents], dim=2)
        mocha_embeds = mocha_embeds[0]

        target_shape = (16, (F - 1) // 4 + 1, lat_h, lat_w)

        seq_len = (target_shape[1] * 2 + 1 + num_refs) * (target_shape[2] * target_shape[3] // 4)

        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        image_embeds = {
            "seq_len": seq_len,
            "mocha_embeds": mocha_embeds,
            "num_frames": F,
            "target_shape": target_shape,
            "num_refs": num_refs,
        }
        
        return (image_embeds,)
        

NODE_CLASS_MAPPINGS = {
    "MochaEmbeds": MochaEmbeds,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "MochaEmbeds": "Mocha Embeds",
    }