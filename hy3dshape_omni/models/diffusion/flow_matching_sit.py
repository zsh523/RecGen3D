import os
from contextlib import contextmanager
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities import rank_zero_only, grad_norm
from deepspeed.utils import safe_get_full_grad

# Add visualization imports
import os

from ...utils.ema import LitEma
from ...utils.misc import instantiate_from_config, instantiate_non_trainable_model
from ...models.conditioners.omni_encoder import CondPointEncoder_MultiView
from ...utils.misc import get_parameter_groups, freeze_modules
from ...utils.sampling import sample_vggt_pcl, apply_se4_transform_to_surface
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map_torch

class Diffuser(pl.LightningModule):
    def __init__(
        self,
        *,
        first_stage_config,
        cond_stage_config,
        denoiser_cfg,
        scheduler_cfg,
        optimizer_cfg=None,
        pipeline_cfg=None,
        image_processor_cfg=None,
        lora_config=None,
        ema_config=None,
        first_stage_key: str = "surface",
        cond_stage_key: str = "image",
        scale_by_std: bool = False,
        z_scale_factor: float = 1.0,
        ckpt_path: Optional[str] = None,
        ignore_keys: Union[Tuple[str], List[str]] = (),
        torch_compile: bool = False,
        vggt_enable: bool = False,
        vggt_loss_enable: bool = False,
        vggt_mode: str = 'none',
        intermediate_layer_enabled: bool = False,
        use_dpt_features: bool = False,
        use_gradient_checkpointing: bool = False,
        normalize_batch: bool =True,
        vggt_train_only: bool = False,
        vggt_config: dict = None,
        vggt_loss: dict = None,
        deepspeed3: bool = False,
        conf_inject: bool = False,
    ):
        super().__init__()
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.normalize_batch = normalize_batch

        # ========= init deepspeed config ========= #
        if deepspeed3:
            use_deepspeed_ckpt = True
        else:
            use_deepspeed_ckpt = False

        # ========= init optimizer config ========= #
        self.optimizer_cfg = optimizer_cfg

        # ========= init diffusion scheduler ========= #
        self.scheduler_cfg = scheduler_cfg
        self.sampler = None
        if 'transport' in scheduler_cfg:
            self.transport = instantiate_from_config(scheduler_cfg.transport)
            self.sampler = instantiate_from_config(scheduler_cfg.sampler, transport=self.transport)
            self.sample_fn = self.sampler.sample_ode(**scheduler_cfg.sampler.ode_params)

        # ========= init the model ========= #
        self.denoiser_cfg = denoiser_cfg
        self.model, model_kwargs = instantiate_from_config(
            denoiser_cfg, device=None, dtype=None, 
            vggt_mode=vggt_mode, 
            intermediate_layer_enabled=intermediate_layer_enabled, 
            use_dpt_features=use_dpt_features,
            use_gradient_checkpointing=use_gradient_checkpointing, 
            use_deepspeed_ckpt=use_deepspeed_ckpt)
        self.cond_stage_model = instantiate_from_config(cond_stage_config)

        # freeze the cond stage model
        self.cond_stage_model = freeze_modules(self.cond_stage_model, patterns=["*"])

        # freeze the denoiser model
        self.model = freeze_modules(self.model, patterns=denoiser_cfg.get('frozen_module_names', None), keep_patterns=denoiser_cfg.get('keep_patterns', None))

        # ========= init VGGT model and additional layers ========= #
        self.vggt_model = None
        self.vggt_blocks = None
        self.vggt_blocks_linear = None
        self.vggt_concat_linear = None
        self.vggt_replace_linear = None
        self.vggt_criterion = None
        self.vggt_mode = vggt_mode if vggt_enable else 'none'
        self.vggt_train_only = vggt_train_only
        self.vggt_cross_attn_enabled = vggt_config['params'].get('cross_attn_enabled', False)
        self.intermediate_layer_enabled = intermediate_layer_enabled
        self.use_dpt_features = use_dpt_features
        self.vggt_dpt_compressor = None
        self.cond_point_encoder = None
        if vggt_enable:
            self.vggt_model = instantiate_from_config(
                vggt_config, 
                use_deepspeed_ckpt=use_deepspeed_ckpt)

            # freeze modules
            self.vggt_model = freeze_modules(
                self.vggt_model,
                patterns=vggt_config.frozen_module_names,
                keep_patterns=vggt_config.keep_patterns,
            )

            # ========================= VGGT tokens layer =========================
            # custom cross-attention layer
            if self.vggt_mode == 'encoder-vggtpcl-multiview':
                self.cond_point_encoder = CondPointEncoder_MultiView(self.cond_stage_model, conf_inject=conf_inject)
            else:
                raise ValueError(f"Invalid vggt_mode: {vggt_mode}")
            
            if vggt_loss_enable:
                camera_loss = instantiate_from_config(vggt_loss.camera)
                depth_loss = instantiate_from_config(vggt_loss.depth)
                point_loss = instantiate_from_config(vggt_loss.point)
                self.vggt_criterion = camera_loss + depth_loss + point_loss


        # ========= config lora model ========= #
        if lora_config is not None:
            from peft import LoraConfig, get_peft_model
            loraconfig = LoraConfig(
                r=lora_config.rank,
                lora_alpha=lora_config.rank,
                target_modules=lora_config.get('target_modules')
            )
            self.model = get_peft_model(self.model, loraconfig)

        
        if self.vggt_train_only:
            for param in self.model.parameters():
                param.requires_grad = False

        # ========= config ema model ========= #
        self.ema_config = ema_config
        if self.ema_config is not None:
            raise NotImplementedError("EMA is not supported for OMNI")
            if self.ema_config.ema_model == 'DSEma':
                from ..utils.ema_deepspeed import DSEma
                self.model_ema = DSEma(self.model, decay=self.ema_config.ema_decay)
            else:
                self.model_ema = LitEma(self.model, decay=self.ema_config.ema_decay)
            #do not initilize EMA weight from ckpt path, since I need to change moe layers
            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # ========= init vae at last to prevent it is overridden by loaded ckpt ========= #
        self.first_stage_model = instantiate_non_trainable_model(first_stage_config)

        self.scale_by_std = scale_by_std
        if scale_by_std:
            self.register_buffer("z_scale_factor", torch.tensor(z_scale_factor))
        else:
            self.z_scale_factor = z_scale_factor

        # ========= init pipeline for inference ========= #
        self.image_processor_cfg = image_processor_cfg
        self.image_processor = None
        if self.image_processor_cfg is not None:
            self.image_processor = instantiate_from_config(self.image_processor_cfg)
        self.pipeline_cfg = pipeline_cfg
        from ...schedulers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.pipeline = instantiate_from_config(
            pipeline_cfg,
            vae=self.first_stage_model,
            model=self.model,
            scheduler=scheduler, # self.sampler,
            conditioner=self.cond_stage_model,
            image_processor=self.image_processor,
            vggt_model=self.vggt_model,
            vggt_blocks=self.vggt_blocks,
            vggt_blocks_linear=self.vggt_blocks_linear,
            vggt_concat_linear=self.vggt_concat_linear,
            vggt_replace_linear=self.vggt_replace_linear,
            vggt_mode=self.vggt_mode,
            vggt_cross_attn_enabled=self.vggt_cross_attn_enabled,
            intermediate_layer_enabled=self.intermediate_layer_enabled,
            use_dpt_features=self.use_dpt_features,
            vggt_dpt_compressor=self.vggt_dpt_compressor,
            cond_point_encoder=self.cond_point_encoder,
        )

        # ========= torch compile to accelerate ========= #
        self.torch_compile = torch_compile
        if self.torch_compile:
            torch.nn.Module.compile(self.model)
            torch.nn.Module.compile(self.first_stage_model)
            torch.nn.Module.compile(self.cond_stage_model)
            print(f'*' * 100)
            print(f'Compile model for acceleration')
            print(f'*' * 100)

        # Set to False because we only care about the trainable parameters
        self.strict_loading = False

    @contextmanager
    def ema_scope(self, context=None):
        if self.ema_config is not None and self.ema_config.get('ema_inference', False):
            self.model_ema.store(self.model)
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.ema_config is not None and self.ema_config.get('ema_inference', False):
                self.model_ema.restore(self.model)
                if context is not None:
                    print(f"{context}: Restored training weights")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Save only trainable parameters"""
        if destination is None:
            destination = {}
        
        # Add only trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                key = prefix + name
                if keep_vars:
                    destination[key] = param
                else:
                    destination[key] = param.detach()
        
        return destination

    def init_from_ckpt(self, path, ignore_keys=()):
        ckpt = torch.load(path, map_location="cpu")
        if 'state_dict' not in ckpt:
            # deepspeed ckpt
            state_dict = {}
            for k in ckpt.keys():
                new_k = k.replace('_forward_module.', '')
                state_dict[new_k] = ckpt[k]
        else:
            state_dict = ckpt["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")



    def on_load_checkpoint(self, checkpoint):
        """
        The pt_model is trained separately, so we already have access to its
        checkpoint and load it separately with `self.set_pt_model`.

        However, the PL Trainer is strict about
        checkpoint loading (not configurable), so it expects the loaded state_dict
        to match exactly the keys in the model state_dict.

        So, when loading the checkpoint, before matching keys, we add all pt_model keys
        from self.state_dict() to the checkpoint state dict, so that they match
        """
        for key in self.state_dict().keys():
            if key.startswith("model_ema") and key not in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = self.state_dict()[key]

    def configure_optimizers(self) -> Tuple[List, List]:

        # for debug
        def print_optimizer_param_summary(param_groups):
            tot_params = 0
            trainable_params = 0

            for idx, pg in enumerate(param_groups):
                n_pg = sum(p.numel() for p in pg["params"])
                n_pg_train = sum(p.numel() for p in pg["params"] if p.requires_grad)
                tot_params += n_pg
                trainable_params += n_pg_train

                print(f"Group {idx:<2} │ {n_pg/1e6:>.2f} M params "
                    f"({n_pg_train/1e6:>.2f} M trainable)")

            print(f"─────────┼──────────────────────────")
            print(f"Total     {tot_params/1e6:.2f} M params "
                f"({trainable_params/1e6:.2f} M trainable)")
            print(f"≈{tot_params*2/1024**2:.0f} MB in bf16  "
                f"({tot_params*4/1024**2:.0f} MB in fp32)")

        lr = self.learning_rate
        lr_vggt = self.learning_rate_vggt

        params_list = []
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        # Print names of all trainable parameters
        print("Trainable parameter names:")
        print("-" * 50)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"  {name}")
        print("-" * 50)


        if not self.vggt_train_only:
            trainable_parameters.extend([p for p in self.cond_stage_model.parameters() if p.requires_grad])
            print("Cond Stage Model Trainable parameter names:")
            print("-" * 50)
            for name, param in self.cond_stage_model.named_parameters():
                if param.requires_grad:
                    print(f"  {name}")
            print("-" * 50)

        # Add VGGT blocks and linear layer parameters if they exist
        if self.vggt_blocks is not None:
            trainable_parameters.extend([p for p in self.vggt_blocks.parameters() if p.requires_grad])
        if self.vggt_blocks_linear is not None:
            trainable_parameters.extend([p for p in self.vggt_blocks_linear.parameters() if p.requires_grad])
        if self.vggt_concat_linear is not None:
            trainable_parameters.extend([p for p in self.vggt_concat_linear.parameters() if p.requires_grad])
        if self.vggt_replace_linear is not None:
            trainable_parameters_replace = [p for p in self.vggt_replace_linear.parameters() if p.requires_grad]
            params_list.append({'params': trainable_parameters_replace, 'lr': 1e-5})
        if self.vggt_dpt_compressor is not None:
            trainable_parameters.extend([p for p in self.vggt_dpt_compressor.parameters() if p.requires_grad])
        if self.cond_point_encoder is not None:
            trainable_parameters_cond_point = [p for p in self.cond_point_encoder.parameters() if p.requires_grad]
            params_list.append({'params': trainable_parameters_cond_point, 'lr': 1e-2, 'name': 'cond_point_encoder'})
            print("Cond Point Encoder Trainable parameter names:")
            print("-" * 50)
            for name, param in self.cond_point_encoder.named_parameters():
                if param.requires_grad:
                    print(f"  {name}")
            print("-" * 50)
        if self.vggt_criterion is not None:
            param_groups_vggt = get_parameter_groups(self.vggt_model, self.optimizer_cfg.optimizer_vggt.weight_decay, base_lr= lr_vggt)
            params_list.extend(param_groups_vggt)
            print('VGGT Params:')
            print_optimizer_param_summary(param_groups_vggt)

        if not self.vggt_train_only:
            params_list.append({'params': trainable_parameters, 'lr': lr, 'name': 'HY'})

        no_decay = ['bias', 'norm.weight', 'norm.bias', 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias']

        if self.optimizer_cfg.get('train_image_encoder', False):
            image_encoder_parameters = list(self.cond_stage_model.named_parameters())
            image_encoder_parameters_decay = [param for name, param in image_encoder_parameters if
                                              not any((no_decay_name in name) for no_decay_name in no_decay)]
            image_encoder_parameters_nodecay = [param for name, param in image_encoder_parameters if
                                                any((no_decay_name in name) for no_decay_name in no_decay)]
            # filter trainable params
            image_encoder_parameters_decay = [param for param in image_encoder_parameters_decay if
                                              param.requires_grad]
            image_encoder_parameters_nodecay = [param for param in image_encoder_parameters_nodecay if
                                                param.requires_grad]

            print(f"Image Encoder Params: {len(image_encoder_parameters_decay)} decay, ")
            print(f"Image Encoder Params: {len(image_encoder_parameters_nodecay)} nodecay, ")

            image_encoder_lr = self.optimizer_cfg['image_encoder_lr']
            image_encoder_lr_multiply = self.optimizer_cfg.get('image_encoder_lr_multiply', 1.0)
            image_encoder_lr = image_encoder_lr if image_encoder_lr is not None else lr * image_encoder_lr_multiply
            params_list.append(
                {'params': image_encoder_parameters_decay, 'lr': image_encoder_lr,
                 'weight_decay': 0.05})
            params_list.append(
                {'params': image_encoder_parameters_nodecay, 'lr': image_encoder_lr,
                 'weight_decay': 0.})

        optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=params_list, lr=lr)

        print('HY Params:')
        print_optimizer_param_summary([{'params': trainable_parameters, 'lr': lr}])


        if hasattr(self.optimizer_cfg, 'scheduler'):
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                "interval": "step",
                "frequency": 1
            }
            schedulers = [scheduler]
        else:
            schedulers = []
        optimizers = [optimizer]

        for pg in params_list:
            print(pg.get('name',None), pg['lr'])
        return optimizers, schedulers

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 \
            and batch_idx == 0 and self.ckpt_path is None:
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")

            z_q = self.encode_first_stage(batch[self.first_stage_key])
            z = z_q.detach()

            del self.z_scale_factor
            self.register_buffer("z_scale_factor", 1. / z.flatten().std())
            print(f"setting self.z_scale_factor to {self.z_scale_factor}")

            print("### USING STD-RESCALING ###")

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema_config is not None:
            self.model_ema(self.model)


    def rotate_y(self, surface: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Rotate the [B, N, 6] tensor along y-axis by angle theta (in radians).
        First 3 dims are xyz, last 3 are normals.
        
        Args:
            surface: (B, N, 6) tensor where first 3 dims are xyz, last 3 are normals
            theta: (B,) tensor of rotation angles in radians
        Returns:
            Rotated surface tensor of shape (B, N, 6)
        """
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        zeros = torch.zeros_like(cos_theta)
        ones = torch.ones_like(cos_theta)
        
        # Build (B, 3, 3) rotation matrices
        rotation_matrix = torch.stack([
            torch.stack([cos_theta, zeros, sin_theta], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sin_theta, zeros, cos_theta], dim=-1)
        ], dim=1)  # (B, 3, 3)
        
        # Split and rotate using batched matmul
        xyz = torch.bmm(surface[..., :3], rotation_matrix.transpose(-1, -2))  # (B, N, 3)
        nrm = torch.bmm(surface[..., 3:6], rotation_matrix.transpose(-1, -2))  # (B, N, 3)
        
        return torch.cat([xyz, nrm, surface[:, :, 6:]], dim=-1)  # (B, N, 7)

    def forward(self, batch):
        # Apply HY view_mode transformation to the surface
        first_stage_rotated = self.rotate_y(batch[self.first_stage_key], torch.deg2rad(batch.get('rot_y_deg')))

        # compute contexts (conditioning for the denoiser)
        if not self.vggt_train_only:
            with torch.profiler.record_function("1_cond_stage_encoding"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): #float32 for text
                    pose = batch.get("pose", None)
                    bbox = batch.get("bbox", None)
                    point = batch.get("point", None)
                    voxel = batch.get("voxel", None)
                    contexts = self.cond_stage_model(
                        image=batch.get('image'), 
                        surface=batch.get('surface'), 
                        pose=pose, 
                        bbox=bbox,
                        vggt_encoder_tokens=1,
                        point=point,
                        voxel=voxel
                    )

        # compute latents
        if not self.vggt_train_only or self.vggt_cross_attn_enabled:
            # compute latents
            with torch.profiler.record_function("2_vae_encoding"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    with torch.no_grad():
                        latents, pc_infos = self.first_stage_model.encode(first_stage_rotated, sample_posterior=True, return_pc_infos=True)
                        latents = self.z_scale_factor * latents

                    # check vae encode and decode is ok? answer is ok!
                    #outputs = self.first_stage_model.latents2mesh(
                        #latents_,
                    #)
                    #else:

        # Add VGGT tokens as additional conditioning
        vggt_tokens = None
        x0, t, xt, ut = None, None, None, None
        if self.vggt_model is not None:
            with torch.profiler.record_function("3_vggt_processing"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
                    # Get aggregated tokens from VGGT
                    with torch.profiler.record_function("3.1_vggt_aggregator"):
                        if self.vggt_cross_attn_enabled:
                            t, x0, x1 = self.transport.sample(latents)

                            t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)
                            # be careful, here we use some modules in the model

                            t_xt, _, _ = self.model.get_t_xt(xt, t)

                            aggregated_tokens_list, ps_idx, pos, dino_tokens = self.vggt_model.aggregator(batch['images'], t_xt=t_xt, t=t)

                        else:
                            aggregated_tokens_list, ps_idx, pos, dino_tokens = self.vggt_model.aggregator(batch['images'])
                    
                    if self.vggt_criterion is not None:
                        with torch.cuda.amp.autocast(enabled=False):
                            with torch.profiler.record_function("3.2_vggt_heads"):
                                ## Predict Cameras
                                pred_pose_enc_list = self.vggt_model.camera_head(aggregated_tokens_list)

                                if self.vggt_model.camera_head_HY is not None:
                                    pred_pose_enc_list_HY, pred_scale_HY = self.vggt_model.camera_head_HY(aggregated_tokens_list, token_idx = 1)
                                else:
                                    pred_pose_enc_list_HY, pred_scale_HY = None, None
                                # Predict Depth Maps
                                depth_map, depth_conf = self.vggt_model.depth_head(aggregated_tokens_list, batch['images'], ps_idx)
                                # Predict Point Maps
                                point_map, point_conf = self.vggt_model.point_head(aggregated_tokens_list, batch['images'], ps_idx)

                            with torch.profiler.record_function("3.3_vggt_loss"):
                                vggt_loss, vggt_loss_details = self.vggt_criterion(
                                    pred_pose_enc_list = pred_pose_enc_list,
                                    pred_pose_enc_list_HY = pred_pose_enc_list_HY,
                                    pred_scale_HY = pred_scale_HY,
                                    batch = batch,
                                    depth_map = depth_map,
                                    depth_conf = depth_conf,
                                    point_map = point_map,
                                    point_conf = point_conf,
                                )
                

                if self.vggt_train_only:
                    loss ={}
                    loss['total_loss'] = vggt_loss
                    loss.update(vggt_loss_details)
                    return loss

            # compute vggt tokens & dpt tokens
            with torch.profiler.record_function("3.4_token_preparation"):
                if self.use_dpt_features:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
                        _, _, dpt_features = self.vggt_model.point_head(aggregated_tokens_list, batch['images'], ps_idx, return_scratch_features=True)
                        dpt_tokens = self.vggt_dpt_compressor(dpt_features)
                else:
                    # Use the last layer tokens (most refined representation)
                    intermediate_layer_idx = [0, 1, 2, 3]
                    if not self.intermediate_layer_enabled:
                        vggt_tokens = aggregated_tokens_list[-1][:, :, ps_idx:, :] #.detach()  # Shape: [B, S, P, 2C] where 2C = 2048 for VGGT-1B

                        # add dino tokens
                    else:
                        vggt_tokens_list = [aggregated_tokens_list[i][:, :, ps_idx:, :] for i in intermediate_layer_idx]

                        # add dino tokens

                        vggt_tokens = torch.cat(vggt_tokens_list, dim=-1)
                pos = pos[:, :, ps_idx:, :] - 1


            # Update main cond with VGGT tokens
            with torch.profiler.record_function("3.5_token_integration"):
                if self.vggt_mode == 'encoder-vggtpcl-multiview':
                    with torch.cuda.amp.autocast(enabled=False): 
                        aggregated_tokens_list = [agg.float() for agg in aggregated_tokens_list]
                        pred_pose_enc_list = self.vggt_model.camera_head(aggregated_tokens_list)
                        extrinsic, intrinsic = pose_encoding_to_extri_intri(pred_pose_enc_list[-1], batch["images"].shape[-2:])
                        depth_map, depth_conf = self.vggt_model.depth_head(aggregated_tokens_list, batch['images'], ps_idx)
                        point_map, point_conf = self.vggt_model.point_head(aggregated_tokens_list, batch['images'], ps_idx)

                    # sample vggt pcl
                    depth_points = unproject_depth_map_to_point_map_torch(depth_map, extrinsic, intrinsic)
                    point_map, conf, T, confidence, T_pa, (sampled_patch_indices, sampled_s_indices) = sample_vggt_pcl(point_map, depth_points, batch['world_points'], batch['point_masks'], point_conf, depth_conf, return_procrustes=True, return_indices=True)

                    # Apply T_pa SE(4) transformation to surface
                    surface = apply_se4_transform_to_surface(first_stage_rotated, T_pa)
                    # apply procrustes to the latents (do vae encoding again)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        with torch.no_grad():
                            latents, pc_infos = self.first_stage_model.encode(surface, sample_posterior=True, return_pc_infos=True)
                            latents = self.z_scale_factor * latents

                    # get dino features
                    dino_features = self.cond_stage_model.images_dino_encoder(batch['images'])[:,:,1:,:] # (B, S, P, D)
                    # get camera tokens
                    camera_tokens = aggregated_tokens_list[-1][:, :, 0:1, :].repeat(1, 1, dino_features.shape[2], 1) # (B, S, P, 2D)
                    # get vggt tokens
                    vggt_tokens = torch.cat([aggregated_tokens[:, :, ps_idx:, :] for aggregated_tokens in aggregated_tokens_list], dim=-1) # (B, S, P, 4 * 2D)
                    
                    # reshape
                    dino_features = dino_features.reshape(dino_features.shape[0], -1, dino_features.shape[-1]) # (B, S*N, D)
                    camera_tokens = camera_tokens.reshape(camera_tokens.shape[0], -1, camera_tokens.shape[-1]) # (B, S*N, 2D)
                    vggt_tokens = vggt_tokens.reshape(vggt_tokens.shape[0], -1, vggt_tokens.shape[-1]) # (B, S*P, 4 * 2D)

                    cond_point, sampled_point = self.cond_point_encoder(point_map, conf, dino_features, camera_tokens, vggt_tokens)
                    contexts['main'] = cond_point
                    contexts['cond_point'] = sampled_point
                else:
                    raise ValueError(f"Invalid vggt_mode: {self.vggt_mode}")


                ##check vae encode and decode is ok? answer is ok!
                #outputs = self.first_stage_model.latents2mesh(
                    #latents_,
                #)
                #else:

        with torch.profiler.record_function("4_diffusion_training_loss"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): 
                loss = self.transport.training_losses(self.model, latents, dict(contexts=contexts),
                    x0 = x0,
                    t = t,
                    xt = xt,
                    ut = ut,
                )
        
        #try:
                ## visualize pred
                #outputs = self.first_stage_model.latents2mesh(
                    #latents_pred,
                #)
                #else:
               
        #except Exception as e:
        ## del the model pred from loss


        if self.vggt_criterion is not None:
            loss['vggt_loss'] = vggt_loss
            loss['total_loss'] = loss['diffusion_loss'] + vggt_loss
            loss.update(vggt_loss_details)
        else:
            loss['total_loss'] = loss['diffusion_loss']

        return loss



    def on_before_batch_transfer(self, batch, dataloader_idx):
        # NOTE: Normalization is now handled in the dataset processing stage
        # The batch already contains normalized extrinsics, cam_points, world_points, depths,
        # as well as extrinsics_HY, scale_HY, and point fields required for training.
        return batch

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        split = 'train'
        loss_dict = {
            f"{split}/total_loss": loss['total_loss'].detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        
        
        # Add sub-loss terms if they exist
        for key, value in loss.items():
            if key != 'total_loss':
                loss_dict[f"{split}/{key}"] = value.detach()
        
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss['total_loss']

    #@rank_zero_only
    def on_before_optimizer_step(self, optimizer):
        if self.global_step % 200 == 0:
            print("Logging gradients on rank 0")

            # Define list of substrings to match
            log_layers = ["blocks.1.", "blocks.10.", "blocks.20.", "aggregator", "transformer", "cross_attn_time_embedding", "cross_attn_time_projection"]

            for name, param in self.model.named_parameters():
                if any(layer in name for layer in log_layers) and param.requires_grad:
                    param_grad = safe_get_full_grad(param)
                    if param_grad is not None:
                        self.logger.experiment.add_histogram(f"{name}.grad", param_grad, self.global_step)
                        self.logger.experiment.add_scalar(f"{name}.grad_norm", param_grad.norm().item(), self.global_step)
            for name, param in self.first_stage_model.named_parameters():
                if any(layer in name for layer in log_layers) and param.requires_grad:
                    param_grad = safe_get_full_grad(param)
                    if param_grad is not None:
                        self.logger.experiment.add_histogram(f"{name}.grad", param_grad, self.global_step)
                        self.logger.experiment.add_scalar(f"{name}.grad_norm", param_grad.norm().item(), self.global_step)
            for name, param in self.vggt_model.named_parameters():
                if any(layer in name for layer in log_layers) and param.requires_grad:
                    param_grad = safe_get_full_grad(param)
                    if param_grad is not None:
                        self.logger.experiment.add_histogram(f"{name}.grad", param_grad, self.global_step)
                        self.logger.experiment.add_scalar(f"{name}.grad_norm", param_grad.norm().item(), self.global_step)
            if self.cond_point_encoder is not None:
                for name, param in self.cond_point_encoder.named_parameters():
                    if param.requires_grad:
                        param_grad = safe_get_full_grad(param)
                        if param_grad is not None:
                            self.logger.experiment.add_histogram(f"cond_point_encoder.{name}.grad", param_grad, self.global_step)
                            self.logger.experiment.add_scalar(f"cond_point_encoder.{name}.grad_norm", param_grad.norm().item(), self.global_step)



    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        split = 'val'
        loss_dict = {
            f"{split}/total_loss": loss['total_loss'].detach(),
            #f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        
        # Add sub-loss terms if they exist
        for key, value in loss.items():
            if key != 'total_loss':
                loss_dict[f"{split}/{key}"] = value.detach()
        
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss['total_loss']
    
    def log_rms_norm_scale(self, phase=None):
        assert phase in ['train', 'val'], "phase should be 'train' or 'val'"
        max_query_scale, max_key_scale = float("-inf"), float("-inf")
        for name, module in self.model.named_modules():
            if "query_norm" in name and max_query_scale < module.scale.max().item():
                max_query_scale = module.scale.max().item()
            if "key_norm" in name and max_key_scale < module.scale.max().item():
                max_key_scale = module.scale.max().item()
        if max_query_scale > float("-inf") and max_key_scale > float("-inf"):
            self.log_dict(
                {f"{phase}/query_scale": max_query_scale, f"{phase}/key_scale": max_key_scale}, 
                prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True
            )
        rank_zero_info(f"RMS Norm Scale: {phase} query_scale: {max_query_scale} key_scale: {max_key_scale}")

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch)
        split = 'test'
        loss_dict = {
            f"{split}/total_loss": loss['total_loss'].detach(),
        }
        
        # Add sub-loss terms if they exist
        for key, value in loss.items():
            if key != 'total_loss':
                loss_dict[f"{split}/{key}"] = value.detach()
        
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

    @torch.no_grad()
    def sample(self, batch, output_type='trimesh', seed: int = 0, **kwargs):
        self.cond_stage_model.disable_drop = True

        generator = torch.Generator().manual_seed(seed)

        with self.ema_scope("Sample"):
            with torch.amp.autocast(device_type='cuda'):
                self.pipeline.device = self.device
                self.pipeline.dtype = self.dtype
                # forward any sampling knobs (num_inference_steps, guidance_scale,
                # octree_resolution, mc_level, num_chunks, ...) on to the pipeline
                additional_params = {'output_type': output_type}
                additional_params.update(kwargs)

                image = batch.get("image", None)
                mask = batch.get('mask', None)

                surface = batch.get(self.first_stage_key, None)
                rot_y_degs = batch.get('rot_y_degs', None)
                if surface is not None and rot_y_degs is not None:
                    surface = self.rotate_y(surface, torch.deg2rad(rot_y_degs))

                if surface is not None:
                    latents, pc_infos = self.first_stage_model.encode(surface, sample_posterior=True, return_pc_infos=True)
                    latents = self.z_scale_factor * latents
                else:
                    latents = None

                if self.vggt_train_only and latents is not None:
                    use_latents_gt = True
                else:
                    use_latents_gt = False
                    
                outputs, outputs_vggt = self.pipeline(image=image, 
                                        mask=mask,
                                        surface=batch.get('surface', None),
                                        pose=batch.get('pose', None),
                                        bbox=batch.get('bbox', None),
                                        point=batch.get('point', None),
                                        voxel=batch.get('voxel', None),
                                        point_masks=batch.get('point_masks', None),
                                        world_points=batch.get('world_points', None),
                                        generator=generator,
                                        images=batch.get('images', None),
                                        vggt_vis=True,
                                        latents_gt=latents,
                                        use_latents_gt=use_latents_gt,
                                        transport=self.transport,
                                        **additional_params)

        self.cond_stage_model.disable_drop = False
        return [outputs], outputs_vggt
