# -*- coding: utf-8 -*-

import importlib
from omegaconf import OmegaConf, DictConfig, ListConfig
from collections import OrderedDict

import torch
import torch.distributed as dist
from typing import Union, List
from peft import LoraConfig, get_peft_model

from wcmatch import fnmatch
from functools import wraps
import torch.nn as nn
import json
import os
import safetensors

def get_config_from_file(config_file: str) -> Union[DictConfig, ListConfig]:
    config_file = OmegaConf.load(config_file)

    if 'base_config' in config_file.keys():
        if config_file['base_config'] == "default_base":
            base_config = OmegaConf.create()
            # base_config = get_default_config()
        elif config_file['base_config'].endswith(".yaml"):
            base_config = get_config_from_file(config_file['base_config'])
        else:
            raise ValueError(f"{config_file} must be `.yaml` file or it contains `base_config` key.")

        config_file = {key: value for key, value in config_file if key != "base_config"}

        return OmegaConf.merge(base_config, config_file)

    return config_file


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_obj_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    return get_obj_from_str(config["target"])


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    cls = get_obj_from_str(config["target"])

    if config.get("from_pretrained", None):
        if "HunYuanDiTPlain" in config["target"]:
            if kwargs.get('use_dpt_features', False):
                vggt_text_states_dim = 2048
            else:
                vggt_text_states_dim = 2048 * 4 + 1024 if kwargs.get('intermediate_layer_enabled', False) else 2048
            return cls.from_pretrained(
                        config["from_pretrained"], 
                        use_safetensors=config.get('use_safetensors', False),
                        variant=config.get('variant', 'fp16'),
                        wan_style_timestep_modulation=config['params'].get('wan_style_timestep_modulation', False),
                        vggt_feature_fusion_enabled=config['params'].get('vggt_feature_fusion_enabled', False),
                        vggt_text_states_dim=vggt_text_states_dim,
                        vggt_mode=kwargs.get('vggt_mode', 'none'),
                        use_gradient_checkpointing=kwargs.get('use_gradient_checkpointing', False),
                        use_deepspeed_ckpt=kwargs.get('use_deepspeed_ckpt', False),
                        normalize_batch=kwargs.get('normalize_batch', False),
                        )
        else:
            return cls.from_pretrained(
                        config["from_pretrained"]) 
                        #variant=config.get('variant', 'fp16'))
    
    # VGGT model
    if "vggt.models.vggt.VGGT" in config["target"]:
        model =  cls(**config['params'], use_deepspeed_ckpt=kwargs.get('use_deepspeed_ckpt', False))

        # ---- clone camera_head -> camera_head_canon BEFORE loading any pretrained weights ----
        if model.camera_head_HY is not None:
            # clone params + buffers without tying storage
            src_sd = {k: v.clone().detach() for k, v in model.camera_head.state_dict().items()}
            # load into target; strict=False in case target has extra/missing keys
            missing, unexpected = model.camera_head_HY.load_state_dict(src_sd, strict=False)
            if missing or unexpected:
                print(f"[warn] camera_head_HY load_state_dict: missing={missing}, unexpected={unexpected}")

        # Load the upstream VGGT-1B weights first; RecGen3D's own checkpoint is
        # layered on top of them below.
        base_weights = resolve_vggt_base_weights(config.get('vggt_base_pretrained', None))
        if not load_pretrained_weights(model, base_weights):
            raise RuntimeError(
                f"Could not load the base VGGT weights from {base_weights}. "
                f"Without them the reconstruction backbone would stay randomly "
                f"initialised. Run scripts/download_weights.sh, or set "
                f"vggt_base_pretrained in the config / $VGGT_BASE_WEIGHTS."
            )

        # A checkpoint that is itself a plain VGGT model is loaded here, before
        # any LoRA wrapping renames the modules.
        pretrained = config.get('pretrained', None)
        is_base_vggt = config.get('vggt_pretrained_is_base', None)
        if is_base_vggt is None:
            is_base_vggt = pretrained is not None and os.path.basename(str(pretrained)) == 'model.safetensors'
        if pretrained is not None and is_base_vggt:
            load_pretrained_weights(model, pretrained)


        # ========= config lora model ========= #
        # which freeze modules not in target_modules
        if config.get('lora_config', None) is not None:
            loraconfig = LoraConfig(
                r=config.lora_config.rank,
                lora_alpha=config.lora_config.rank,
                target_modules=config.lora_config.get('target_modules')
            )
            model = get_peft_model(model, loraconfig)
            model.print_trainable_parameters()


        # A RecGen3D Stage-1 checkpoint stores VGGT under `vggt_model.`, and is
        # loaded after LoRA wrapping so the module names line up.
        if pretrained is not None and not is_base_vggt:
            print(f"Loading RecGen3D Stage-1 VGGT weights: {pretrained}")
            if not load_pretrained_weights(model, pretrained, strip_prefix="vggt_model."):
                raise RuntimeError(f"Could not load the Stage-1 VGGT checkpoint: {pretrained}")
            print(f"Loaded RecGen3D Stage-1 VGGT weights: {pretrained}")


        ## unfreeze camera and register tokens
            #if any(substr in name for substr in ["camera_token_canon", 
                #"register_token_canon"]):


        if config.get('vggt_loss_enable') == False:
            # freeze the whole model
            for param in model.parameters():
                param.requires_grad = False
            print("Freezing the whole VGGT model (no VGGT loss)")
        else:
            # unfreeze cross-attention layers
            # careful: make sure the added params not wrapped by peft
            if config['params'].get('cross_attn_enabled', False):
                for name, param in model.named_parameters():
                    if any(substr in name for substr in ["cross_attn"]):
                        if 'aggregator' in name:
                            param.requires_grad = True
                            print(f"Unfreezing VGGT: {name}")
    
            # unfreeze modules
            modules_to_unfreeze = [
            "camera_head", 
            #"camera_head_canon",
            "depth_head",
            "point_head",
            ]

            for module_name, module in model.named_modules():
                if any(key in module_name for key in modules_to_unfreeze):
                    for param in module.parameters():
                        param.requires_grad = True
                    print(f"Unfreezing VGGT module: {module_name}")

        return model



    params = config.get("params", dict())
    # params.update(kwargs)
    # instance = cls(**params)
    kwargs.update(params)
    instance = cls(**kwargs)

    return instance


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def instantiate_non_trainable_model(config):
    model = instantiate_from_config(config)
    model = model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = False

    return model

def instantiate_non_trainable_model_w_unfreeze(config, unfreeze_module_names=None):
    """
    Instantiate a model from config and freeze all parameters,
    except for those in modules whose names contain any substring in `unfreeze_module_names`.

    Args:
        config: The configuration for instantiating the model.
        unfreeze_module_names: List of substrings. If a module name contains any of them, 
                               its parameters will be unfrozen.

    Returns:
        The model with selected modules unfrozen and training disabled elsewhere.
    """
    model = instantiate_from_config(config)

    if unfreeze_module_names is None:
        unfreeze_module_names = []

    # Handle top-level parameters (not part of named_modules)
    for param in model.parameters(recurse=False):
        param.requires_grad = False

    # Freeze or unfreeze submodules
    for name, module in model.named_modules():
        requires_grad = any(unfreeze_name in name for unfreeze_name in unfreeze_module_names)

        for param in module.parameters(recurse=False):
            param.requires_grad = requires_grad

        if not requires_grad:
            module.eval()
            module.train = disabled_train  # Disable .train() on frozen modules
        
        if requires_grad:
            print(f"Unfreezing {name}")

    return model



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()



# ------------------------------------------------------------
# Glob-matching flags (behave like the Unix shell)
# ------------------------------------------------------------
GLOB_FLAGS = (
    fnmatch.CASE       # case‑sensitive
    | fnmatch.DOTMATCH # '*' also matches '.'
    | fnmatch.EXTMATCH # extended patterns like *(foo|bar)
    | fnmatch.SPLIT    # "pat1|pat2" works out‑of‑the‑box
)


def freeze_modules(model: nn.Module, patterns: List[str] = None, keep_patterns: List[str] = None, recursive: bool = True) -> nn.Module:
    matched: set[str] = set()
    patterns = patterns or []
    keep_patterns = keep_patterns or []

    # Collect all keep-module and keep-param names
    keep_module_names = {
        name for name, _ in model.named_modules()
        if any(fnmatch.fnmatch(name, kp, flags=GLOB_FLAGS) for kp in keep_patterns)
    }
    keep_param_names = {
        name for name, _ in model.named_parameters()
        if any(fnmatch.fnmatch(name, kp, flags=GLOB_FLAGS) for kp in keep_patterns)
    }

    # Freeze modules
    for name, mod in model.named_modules():
        if any(fnmatch.fnmatch(name, p, flags=GLOB_FLAGS) for p in patterns):
            # Skip if this module or any of its children are in keep_module_names
            if any(k.startswith(name) for k in keep_module_names if k != ""):
                continue
            matched.add(name)
            _freeze(mod, recursive)

    # Freeze parameters (fine-grained)
    for name, param in model.named_parameters():
        if any(fnmatch.fnmatch(name, p, flags=GLOB_FLAGS) for p in patterns):
            if name in keep_param_names:
                continue
            param.requires_grad = False

    _check_every_pattern_used(matched, patterns)
    return model

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _freeze(mod: nn.Module, recursive: bool) -> None:
    """Put *mod* in eval mode and lock its parameters."""

    if recursive:
        mod.eval()            # affects the whole subtree
    else:
        mod.training = False  # only this exact module

    original_train = mod.train

    @wraps(original_train)
    def locked_train(mode: bool = True):
        if recursive:
            return original_train(False)  # ignore user's *mode*
        out = original_train(mode)        # children follow user's choice
        out.training = False              # but this module stays frozen
        return out

    mod.train = locked_train  # type: ignore[attr-defined]

    param_iter = (
        mod.parameters()              # default recurse=True
        if recursive
        else mod.parameters(recurse=False)
    )
    for p in param_iter:
        p.requires_grad = False


def _check_every_pattern_used(matched_names: set[str], patterns: List[str]):
    unused = [p for p in patterns if not any(fnmatch.fnmatch(n, p, flags=GLOB_FLAGS)
                                             for n in matched_names)]
    if unused:
        raise ValueError(f"These patterns matched nothing: {unused}")

def get_parameter_groups(
    model, weight_decay, layer_decay=1.0, skip_list=(), no_lr_scale_list=[], base_lr=1.0
):
    parameter_group_names = {}
    parameter_group_vars = {}
    enc_depth, dec_depth = None, None
    # prepare layer decay values
    assert layer_decay == 1.0 or 0.0 < layer_decay < 1.0
    if layer_decay < 1.0:
        enc_depth = model.enc_depth
        dec_depth = model.dec_depth if hasattr(model, "dec_blocks") else 0
        num_layers = enc_depth + dec_depth
        layer_decay_values = list(
            layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
        )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        
        # Check if this is a cross-attention parameter
        is_cross_attn = "cross_attn" in name
        
        # Assign weight decay values
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            if "enc_blocks" in name:
                group_name = "no_decay_enc_blocks"
            else:
                group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            if "enc_blocks" in name:
                group_name = "decay_enc_blocks"
            else:
                group_name = "decay"
            this_weight_decay = weight_decay
        
        # Assign layer ID for LR scaling
        if layer_decay < 1.0:
            skip_scale = False
            layer_id = _get_num_layer_for_vit(name, enc_depth, dec_depth)
            group_name = "layer_%d_%s" % (layer_id, group_name)
            if name in no_lr_scale_list:
                skip_scale = True
                group_name = f"{group_name}_no_lr_scale"
        else:
            layer_id = 0
            skip_scale = True
        
        # Add cross_attn suffix to group name if applicable
        if is_cross_attn:
            group_name = f"{group_name}_cross_attn"
        
        if group_name not in parameter_group_names:
            if not skip_scale:
                scale = layer_decay_values[layer_id]
            else:
                scale = 1.0
            if "enc_blocks" in group_name:
                scale *= 1.0
            
            # Apply 10x multiplier for cross-attention parameters
            if is_cross_attn:
                scale *= 1.0
            
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": scale * base_lr,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": scale * base_lr,
            }
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())

def _get_num_layer_for_vit(var_name, enc_depth, dec_depth):
    if var_name in ("cls_token", "mask_token", "pos_embed", "global_tokens"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("enc_blocks"):
        layer_id = int(var_name.split(".")[1])
        return layer_id + 1
    elif var_name.startswith("decoder_embed") or var_name.startswith(
        "enc_norm"
    ):  # part of the last black
        return enc_depth
    elif var_name.startswith("dec_blocks"):
        layer_id = int(var_name.split(".")[1])
        return enc_depth + layer_id + 1
    elif var_name.startswith("dec_norm"):  # part of the last block
        return enc_depth + dec_depth
    elif any(var_name.startswith(k) for k in ["head", "prediction_head"]):
        return enc_depth + dec_depth + 1
    else:
        raise NotImplementedError(var_name)
    
def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


VGGT_BASE_REPO = "facebook/VGGT-1B"
VGGT_BASE_FILENAME = "model.safetensors"


def resolve_vggt_base_weights(path=None):
    """
    Locate the upstream VGGT-1B weights that the reconstruction backbone starts from.

    Searched in order: the explicit `path`, $VGGT_BASE_WEIGHTS, the weights directory
    ($HY3DGEN_MODELS or ./pretrained_weights), and finally `model.safetensors` in the
    working directory. If none exist the file is fetched from the HuggingFace Hub.
    """
    weights_dir = os.environ.get("HY3DGEN_MODELS", "pretrained_weights")
    candidates = [
        path,
        os.environ.get("VGGT_BASE_WEIGHTS", None),
        os.path.join(weights_dir, "vggt", VGGT_BASE_FILENAME),
        os.path.join("pretrained_weights", "vggt", VGGT_BASE_FILENAME),
        VGGT_BASE_FILENAME,
    ]
    for candidate in candidates:
        if candidate and os.path.exists(os.path.expanduser(candidate)):
            return os.path.expanduser(candidate)

    target_dir = os.path.join(os.path.expanduser(weights_dir), "vggt")
    print(f"Base VGGT weights not found locally; downloading {VGGT_BASE_REPO} -> {target_dir}")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=VGGT_BASE_REPO, filename=VGGT_BASE_FILENAME, local_dir=target_dir
    )


def load_pretrained_weights(model, pretrained_path, device='cpu', strip_prefix=None):
    """
    Load pretrained weights into a model with detailed logging of missing and unexpected keys.
    
    Args:
        model (torch.nn.Module): The model to load weights into
        pretrained_path (str): Path to the pretrained checkpoint
        device (torch.device): Device to load the checkpoint on
        strip_prefix (str or list): Optional prefix(es) to strip from checkpoint keys.
                                    Can be a single string or list of strings to try.
                                    E.g., 'vggt_model.' or ['vggt_model.', 'model.']
    
    Returns:
        bool: True if loading was successful, False otherwise
    """
    try:
        print(f"Loading pretrained: {pretrained_path}")
        
        pretrained_state = load_checkpoint(pretrained_path, device)
        pretrained_state = strip_module(pretrained_state)
        
        # Apply custom prefix stripping if specified
        if strip_prefix is not None:
            if isinstance(strip_prefix, str):
                strip_prefix = [strip_prefix]
            
            new_state = {}
            for key, value in pretrained_state.items():
                new_key = key
                for prefix in strip_prefix:
                    if key.startswith(prefix):
                        new_key = key[len(prefix):]
                        break
                new_state[new_key] = value
            pretrained_state = new_state
            print(f"Applied custom prefix stripping: {strip_prefix}")
        
        model_state = model.state_dict()

        # Load with strict=False to allow partial loading
        load_result = model.load_state_dict(pretrained_state, strict=False)
        print("Loaded pretrained weights.")

        # Report missing keys (parameters not found in checkpoint)
        if load_result.missing_keys:
            print(f"Missing keys (new modules not in checkpoint):")
            for key in load_result.missing_keys:
                print(f"  [NEW] {key}")

        # Report unexpected keys (parameters in checkpoint but not in model)
        if load_result.unexpected_keys:
            print(f"Unexpected keys (present in checkpoint but not used):")
            for key in load_result.unexpected_keys:
                print(f"  [UNUSED] {key}")

        del pretrained_state  # free memory
        return True
        
    except Exception as e:
        print(f"Failed to load pretrained weights: {e}")
        return False


def load_checkpoint(checkpoint_path, device='cpu'):
    """
    Load checkpoint from either safetensors or PyTorch format.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (torch.device): Device to load the checkpoint on
    
    Returns:
        dict: The loaded state dictionary
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    if checkpoint_path.endswith('.safetensors'):
        print(f"Loading safetensors checkpoint: {checkpoint_path}")
        # Load to CPU first, then move to target device
        state_dict = safetensors.torch.load_file(checkpoint_path)
        # Move tensors to the target device
        state_dict = {k: v.to(device) for k, v in state_dict.items()}
        return state_dict
    else:
        print(f"Loading PyTorch checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)['state_dict']
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                return checkpoint["model"]
            elif "state_dict" in checkpoint:
                return checkpoint["state_dict"]
            else:
                # Assume the entire dict is the state dict
                return checkpoint
        else:
            # Assume it's directly a state dict
            return checkpoint