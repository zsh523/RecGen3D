# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.
# Modified by the RecGen3D authors, 2026.

import copy
import importlib
import inspect
import os
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
import yaml
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_accelerate_version, is_accelerate_available
from tqdm import tqdm

from .models.autoencoders import ShapeVAE
from .utils import logger, synchronize_timer, smart_load_model
from .utils.sampling import sample_vggt_pcl
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map_torch

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@synchronize_timer('Export to trimesh')
def export_to_trimesh(mesh_output):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                outputs.append(mesh_output)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
        return mesh_output



class Hunyuan3DDiTPipeline:
    model_cpu_offload_seq = "conditioner->model->vae"
    _exclude_from_cpu_offload = []

    def __init__(
        self,
        vae,
        model,
        scheduler,
        conditioner,
        image_processor,
        device='cuda',
        dtype=torch.float16,
        vggt_model=None,
        vggt_blocks=None,
        vggt_blocks_linear=None,
        vggt_concat_linear=None,
        vggt_replace_linear=None,
        vggt_mode=None,
        vggt_cross_attn_enabled=False,
        intermediate_layer_enabled=False,
        use_dpt_features=False,
        vggt_dpt_compressor=None,
        cond_point_encoder=None,
        **kwargs
    ):
        self.vae = vae
        self.model = model
        self.scheduler = scheduler
        self.conditioner = conditioner
        self.image_processor = image_processor
        self.vggt_model = vggt_model
        self.vggt_blocks = vggt_blocks
        self.vggt_blocks_linear = vggt_blocks_linear
        self.vggt_concat_linear = vggt_concat_linear
        self.vggt_replace_linear = vggt_replace_linear
        self.vggt_mode = vggt_mode
        self.vggt_cross_attn_enabled = vggt_cross_attn_enabled
        self.intermediate_layer_enabled = intermediate_layer_enabled
        self.use_dpt_features = use_dpt_features
        self.vggt_dpt_compressor = vggt_dpt_compressor
        self.cond_point_encoder = cond_point_encoder
        self.kwargs = kwargs
        self.to(device, dtype)

    def compile(self):
        self.vae = torch.compile(self.vae)
        self.model = torch.compile(self.model)
        self.conditioner = torch.compile(self.conditioner)

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.vae.to(dtype=dtype)
            self.model.to(dtype=dtype)
            self.conditioner.to(dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
            self.vae.to(device)
            self.model.to(device)
            self.conditioner.to(device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module) or name in self._exclude_from_cpu_offload:
                continue

            if not hasattr(model, "_hf_hook"):
                return self.device
            for module in model.modules():
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    return torch.device(module._hf_hook.execution_device)
        return self.device

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
        if self.model_cpu_offload_seq is None:
            raise ValueError(
                "Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set."
            )

        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of "
                f"the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`)
        # or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu")
            device_mod = getattr(torch, self.device.type, None)
            if hasattr(device_mod, "empty_cache") and device_mod.is_available():
                device_mod.empty_cache()  
                # otherwise we don't see the memory savings (but they probably exist)

        all_model_components = {k: v for k, v in self.components.items() if isinstance(v, torch.nn.Module)}

        self._all_hooks = []
        hook = None
        for model_str in self.model_cpu_offload_seq.split("->"):
            model = all_model_components.pop(model_str, None)
            if not isinstance(model, torch.nn.Module):
                continue

            _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
            self._all_hooks.append(hook)

        # CPU offload models that are not in the seq chain unless they are explicitly excluded
        # these models will stay on CPU until maybe_free_model_hooks is called
        # some models cannot be in the seq chain because they are iteratively called, 
        # such as controlnet
        for name, model in all_model_components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                _, hook = cpu_offload_with_hook(model, device)
                self._all_hooks.append(hook)

    def maybe_free_model_hooks(self):
        r"""
        Function that offloads all components, removes all model hooks that were added when using
        `enable_model_cpu_offload` and then applies them again. In case the model has not been offloaded this function
        is a no-op. Make sure to add this function to the end of the `__call__` function of your pipeline so that it
        functions correctly when applying enable_model_cpu_offload.
        """
        if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:
            # `enable_model_cpu_offload` has not be called, so silently do nothing
            return

        for hook in self._all_hooks:
            # offload model and remove hook from model
            hook.offload()
            hook.remove()

        # make sure the model is in the same state as before calling it
        self.enable_model_cpu_offload()

    @synchronize_timer('Encode cond')
    def encode_cond(self, image, do_classifier_free_guidance, dual_guidance, point_masks=None, point=None, images=None, vggt_tokens=None, 
                    dpt_tokens=None, pos=None, pc_infos=None, cond_conditioner=None, latents_gt=None, predictions=None, aggregated_tokens_list=None, world_points=None):
        bsz = image.shape[0]
        if cond_conditioner is None:
            raise RuntimeError("Please check encode process - cond_conditioner cannot be None")
        else:
            cond = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in cond_conditioner.items()}

        if self.vggt_mode == 'encoder-vggtpcl-multiview':
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], image.shape[-2:])
            depth_map, depth_conf = predictions['depth'], predictions['depth_conf']
            point_map, point_conf = predictions['world_points'], predictions['world_points_conf']

            # sample vggt pcl
            depth_points = unproject_depth_map_to_point_map_torch(depth_map, extrinsic, intrinsic)
            point_map, conf, T, confidence, T_pa, (sampled_patch_indices, sampled_s_indices) = sample_vggt_pcl(point_map, depth_points, world_points, point_masks, point_conf, depth_conf, return_procrustes=True, return_indices=True)
            predictions['T'] = T
            predictions['confidence'] = confidence
            
            # apply T_pa SE(4) transformation to predictions
            predictions['T_pa'] = T_pa

            # get corresponding dino features for the sampled pcl
            dino_features = self.conditioner.images_dino_encoder(images)[:,:,1:,:] # (B, S, P, D)
            # get corresponding camera tokens for the sampled pcl
            camera_tokens = aggregated_tokens_list[-1][:, :, 0:1, :].repeat(1, 1, dino_features.shape[2], 1) # (B, S, P, 2D)
            # get vggt tokens
            vggt_tokens = vggt_tokens

            # reshape
            dino_features = dino_features.reshape(dino_features.shape[0], -1, dino_features.shape[-1]) # (B, S*N, D)
            camera_tokens = camera_tokens.reshape(camera_tokens.shape[0], -1, camera_tokens.shape[-1]) # (B, S*N, 2D)
            vggt_tokens = vggt_tokens.reshape(vggt_tokens.shape[0], -1, vggt_tokens.shape[-1]) # (B, S*P, 4 * 2D)

            cond_point, sampled_point = self.cond_point_encoder(point_map, conf, dino_features, camera_tokens, vggt_tokens)
            cond['main'] = cond_point
            cond['cond_point'] = sampled_point
        else:
            raise ValueError(f"Invalid vggt_mode: {self.vggt_mode}")

        if do_classifier_free_guidance:
            img_uncond = self.conditioner.unconditional_embedding(bsz)["dino"]["last_hidden_state"]
            un_cond = {'main': torch.cat([img_uncond, cond['main'][:, img_uncond.shape[1]:, :]], dim=1), 'cond_point': cond['cond_point'].detach().clone()}
            if self.vggt_mode == 'encoder-vggtpcl-multiview':
                un_cond = {'cond_point': torch.zeros_like(cond['cond_point']), 'main': torch.zeros_like(cond['main'])}

            if dual_guidance:
                un_cond_drop_main = copy.deepcopy(un_cond)
                un_cond_drop_main['additional'] = cond['additional']

                def cat_recursive(a, b, c):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b, c], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k], c[k])
                    return out

                cond = cat_recursive(cond, un_cond_drop_main, un_cond)
            else:
                def cat_recursive(a, b):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k])
                    return out

                cond = cat_recursive(cond, un_cond)

        return cond

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, dtype, device, generator, latents=None):
        shape = (batch_size, *self.vae.latent_shape)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)
        return latents

    def prepare_image(self, image, mask=None) -> dict:
        if isinstance(image, torch.Tensor): #and isinstance(mask, torch.Tensor):
            outputs = {
                'image': image,
                'mask': mask
            }
            return outputs
            
        if isinstance(image, str) and not os.path.exists(image):
            raise FileNotFoundError(f"Couldn't find image at path {image}")

        if not isinstance(image, list):
            image = [image]

        outputs = []
        for img in image:
            output = self.image_processor(img)
            outputs.append(output)

        cond_input = {k: [] for k in outputs[0].keys()}
        for output in outputs:
            for key, value in output.items():
                cond_input[key].append(value)
        for key, value in cond_input.items():
            if isinstance(value[0], torch.Tensor):
                cond_input[key] = torch.cat(value, dim=0)

        return cond_input

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

        #if mc_algo is None:
        #logger.info('The parameters `mc_algo` is deprecated, and will be removed in future versions.\n'
                    #'Please use: \n'
                    #'from hy3dshape_omni.models.autoencoders import SurfaceExtractors\n'
                    #'pipeline.vae.surface_extractor = SurfaceExtractors[mc_algo]() instead\n')


    def _export(
        self,
        latents,
        output_type='trimesh',
        octree_depth=8,
        box_v=1.01,
        mc_level=0.0,
        num_chunks=20000,
        octree_resolution=512,
        mc_mode='mc',
        sigmoid=False
    ):
        if not output_type == "latent":
            latents = 1. / self.vae.scale_factor * latents
            latents = self.vae(latents)
            outputs = self.vae.latents2mesh(
                latents,
                octree_depth=octree_depth,
                bounds=box_v,
                mc_level=mc_level,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                mc_mode=mc_mode,
                sigmoid=sigmoid,
            )
        else:
            outputs = latents

        if output_type == 'trimesh':
            outputs = export_to_trimesh(outputs)

        return outputs


class Hunyuan3DDiTFlowMatchingPipeline(Hunyuan3DDiTPipeline):

    def _process_vggt_tokens(self, image, images=None, vggt_vis=False, xt=None, t=None):
        """
        Process VGGT tokens from input images.
        
        Args:
            image: Input image tensor
            images: Optional pre-processed images
            vggt_vis: Whether to compute additional visual predictions
            
        Returns:
            tuple: (vggt_tokens, vggt_tokens_pos, predictions)
        """
        vggt_tokens = None
        dpt_tokens = None
        vggt_tokens_pos = None
        predictions = {}
        
        if self.vggt_model is not None:
            device = self.device
            dtype = self.dtype
            
            if images is None:
                images = image.detach().clone()
                images = (images + 1) / 2 * 255.0
                images = images.to(device, dtype=dtype)
                # Resize images to 518x518
                images = torch.nn.functional.interpolate(images, size=(518, 518), mode='bilinear', align_corners=False)
                images = images.unsqueeze(0)

            if self.vggt_cross_attn_enabled:
                # be careful, here we use some modules in the model
                t = t.to(xt.dtype).unsqueeze(0).repeat(xt.shape[0]) 
                t = t / self.scheduler.config.num_train_timesteps
                t_xt, _, _ = self.model.get_t_xt(xt, t)


                aggregated_tokens_list, patch_start_idx, pos, dino_tokens = self.vggt_model.aggregator(images, t_xt=t_xt, t=t)
            else:
                aggregated_tokens_list, patch_start_idx, pos, dino_tokens = self.vggt_model.aggregator(images)
            
            intermediate_layer_idx = [0, 1, 2, 3]

            if self.use_dpt_features:
                _, _, dpt_features = self.vggt_model.point_head(aggregated_tokens_list, images, patch_start_idx, return_scratch_features=True)
                dpt_tokens = self.vggt_dpt_compressor(dpt_features)
            else:
                if not self.intermediate_layer_enabled:
                    vggt_tokens = aggregated_tokens_list[-1][:, :, patch_start_idx:, :] #.detach()  # Shape: [B, S, P, 2C] where 2C = 2048 for VGGT-1B

                    # add dino tokens
                else:
                    vggt_tokens_list = [aggregated_tokens_list[i][:, :, patch_start_idx:, :] for i in intermediate_layer_idx]

                    ## add dino tokens

                    vggt_tokens = torch.cat(vggt_tokens_list, dim=-1)
            vggt_tokens_pos = pos[:, :, patch_start_idx:, :] - 1


            if vggt_vis:
                predictions["images"] = images
                with torch.cuda.amp.autocast(enabled=False):
                    if self.vggt_model.camera_head is not None:
                        pose_enc_list = self.vggt_model.camera_head(aggregated_tokens_list)
                        predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                    if self.vggt_model.camera_head_HY is not None:
                        pred_pose_enc_list_HY, pred_scale_HY = self.vggt_model.camera_head_HY(aggregated_tokens_list, token_idx = 1)
                        predictions["pose_enc_HY"] = pred_pose_enc_list_HY[-1]
                        predictions["scale_HY"] = pred_scale_HY
                    if self.vggt_model.depth_head is not None:
                        depth, depth_conf = self.vggt_model.depth_head(
                            aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                        )
                        predictions["depth"] = depth
                        predictions["depth_conf"] = depth_conf
                    if self.vggt_model.point_head is not None:
                        pts3d, pts3d_conf = self.vggt_model.point_head(
                            aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                        )
                        predictions["world_points"] = pts3d
                        predictions["world_points_conf"] = pts3d_conf

        return vggt_tokens, dpt_tokens, vggt_tokens_pos, predictions, aggregated_tokens_list

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        mask: torch.Tensor = None,
        surface: Union[str, List[str], torch.Tensor] = None,
        pose: Union[str, List[str], torch.Tensor] = None,
        bbox: Union[str, List[str], torch.Tensor] = None,
        point: Union[str, List[str], torch.Tensor] = None,
        voxel: Union[str, List[str], torch.Tensor] = None,
        point_masks: torch.Tensor = None,
        world_points: torch.Tensor = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_depth=8,
        octree_resolution=512,
        mc_level=0.0,
        mc_mode='mc',
        num_chunks=8000,
        sigmoid=False,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        images = None,
        vggt_vis = False,
        pc_infos=None,
        latents_gt=None,
        use_latents_gt=False,
        transport=None,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)


        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )

        image = self.prepare_image(image, mask).pop('image')
        vggt_encoder_tokens = 1
        cond_conditioner = self.conditioner(image=image, surface=surface, pose=pose, bbox=bbox, point=point, voxel=voxel, vggt_encoder_tokens=vggt_encoder_tokens)

        ## Process VGGT tokens
        #vggt_tokens, vggt_tokens_pos, predictions = self._process_vggt_tokens(
        #)

        ## Encode cond
        #cond = self.encode_cond(
        #)

        batch_size = image.shape[0]

        # 5. Prepare timesteps
        # NOTE: this is slightly different from common usage, we start from 0.
        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        latents = self.prepare_latents(batch_size, dtype, device, generator)

        guidance = None
        if hasattr(self.model, 'guidance_embed') and \
            self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)

        # if cross-attention is enabled, we need to store the predictions for each timestep
        if self.vggt_cross_attn_enabled:
            predictions_list = []
        else:
            if self.vggt_model is not None:
                # Process VGGT tokens
                vggt_tokens, dpt_tokens, vggt_tokens_pos, predictions, aggregated_tokens_list = self._process_vggt_tokens(
                    image=image, 
                    images=images, 
                    vggt_vis=vggt_vis,
                )
            else:
                # no vggt model, no predictions
                vggt_tokens, dpt_tokens, vggt_tokens_pos, predictions = None, None, None, {}
            predictions_list = [predictions]

            # Encode cond
            cond = self.encode_cond(
                image=image,
                point_masks=point_masks,
                do_classifier_free_guidance=do_classifier_free_guidance,
                dual_guidance=False,
                vggt_tokens=vggt_tokens,
                dpt_tokens=dpt_tokens,
                pos=vggt_tokens_pos,
                pc_infos=pc_infos,
                cond_conditioner=cond_conditioner,
                latents_gt = latents_gt,
                predictions=predictions,
                # point
                point=point,
                images=images,
                world_points=world_points,
                aggregated_tokens_list=aggregated_tokens_list,
            )
            if 'cond_point' in cond:
                predictions['cond_point'] = cond['cond_point'][::2] # remove unconditional cond_point
        
        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:")):
                # The VGGT branch only has to run once: its tokens condition every
                # denoising step, so compute them at i == 0 and reuse them.
                if self.vggt_cross_attn_enabled and i == 0:
                    vggt_tokens, dpt_tokens, vggt_tokens_pos, predictions, aggregated_tokens_list = self._process_vggt_tokens(
                        image=image, 
                        images=images, 
                        vggt_vis=vggt_vis,
                        xt=latents,
                        t=t,
                    )
                    predictions_list.append(predictions)

                    # Encode cond
                    cond = self.encode_cond(
                        image=image,
                        point_masks=point_masks,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        dual_guidance=False,
                        vggt_tokens=vggt_tokens,
                        dpt_tokens=dpt_tokens,
                        pos=vggt_tokens_pos,
                        pc_infos=pc_infos,
                        cond_conditioner=cond_conditioner,
                        latents_gt = latents_gt,
                        predictions = predictions,
                        # point
                        point=point,
                        images=images,
                        world_points=world_points,
                        aggregated_tokens_list=aggregated_tokens_list,
                    )
                    if 'cond_point' in cond:
                        predictions['cond_point'] = cond['cond_point'][::2] # remove unconditional cond_point

                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                # NOTE: we assume model get timesteps ranged from 0 to 1
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / self.scheduler.config.num_train_timesteps
                noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample
                if use_latents_gt:
                    t = timestep[::2] + 20 / self.scheduler.config.num_train_timesteps
                    t.clamp_(min=0, max=1)
                    x0 = torch.randn_like(latents_gt)
                    x1 = latents_gt
                    t, xt, _ = transport.path_sampler.plan(t, x0, x1)
                    latents = xt

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        return self._export(
            latents,
            octree_depth=octree_depth, output_type=output_type, box_v=box_v, mc_level=mc_level, 
            num_chunks=num_chunks, octree_resolution=octree_resolution, mc_mode=mc_mode, sigmoid=sigmoid,
        ), predictions_list
