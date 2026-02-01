import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # Collect all weight names for error reporting
    all_weight_names = []
    safetensor_files = glob(os.path.join(path, "*.safetensors"))
    
    if not safetensor_files:
        raise FileNotFoundError(f"No .safetensors files found in {path}")
    
    for file in safetensor_files:
        with safe_open(file, "pt", "cpu") as f:
            all_weight_names.extend(f.keys())
    
    # Get model's available parameters for error reporting
    model_params = dict(model.named_parameters())
    
    for file in safetensor_files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                try:
                    for k in packed_modules_mapping:
                        if k in weight_name:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            break
                    else:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
                except AttributeError as e:
                    # Detailed error message for debugging
                    print(f"\n{'='*60}")
                    print(f"[nano-vllm] Weight loading error!")
                    print(f"{'='*60}")
                    print(f"Failed to load weight: {weight_name}")
                    print(f"Error: {e}")
                    print(f"\nWeight file: {file}")
                    print(f"\n--- Weights in safetensors file (first 20) ---")
                    for i, name in enumerate(sorted(all_weight_names)[:20]):
                        print(f"  {name}")
                    if len(all_weight_names) > 20:
                        print(f"  ... and {len(all_weight_names) - 20} more")
                    print(f"\n--- Model parameters (first 20) ---")
                    for i, name in enumerate(sorted(model_params.keys())[:20]):
                        print(f"  {name}")
                    if len(model_params) > 20:
                        print(f"  ... and {len(model_params) - 20} more")
                    print(f"{'='*60}\n")
                    raise
