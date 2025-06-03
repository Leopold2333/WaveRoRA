import torch
import os

for dataset in ["traffic"]:
    for prediction_length in [96, 192, 336, 720]:
        print(f"{dataset}_{prediction_length}")
        ckpt_path = "./checkpoints00/2025/"
        dataset_str = f"{dataset}(M)_96_{prediction_length}_loss(mse)"
        file_list = os.listdir(ckpt_path+dataset_str)
        for fl in file_list:
            state_dict = torch.load(ckpt_path+dataset_str+"./"+fl+"/checkpoint.pth", map_location="cuda:0", weights_only=True)
            new_state_dict = {}
            for k, v in state_dict.items():
                # Loading model from DataParallel?
                if k.startswith('module.'):
                    name = k[len('module.'):]  # 去掉开头的 'module.'
                else:
                    name = k
                new_state_dict[name] = v
            for layer in [0, 1, 2, 3]:
                key = f'encoder.attn_layers.{layer}.attention1.router_proj'
                new_key = f'encoder.attn_layers.{layer}.attention1.inner_attention.router_proj'
                if key in new_state_dict:
                    new_state_dict[new_key] = new_state_dict[key]
                    del new_state_dict[key]
                key = f'encoder.attn_layers.{layer}.attention1.rope.rotations'
                new_key = f'encoder.attn_layers.{layer}.attention1.inner_attention.rope.rotations'
                if key in new_state_dict:
                    new_state_dict[new_key] = new_state_dict[key]
                    del new_state_dict[key]
            new_path = "./checkpoints/2025/"+dataset_str+"./"+fl+"/checkpoint.pth"
            new_dir = os.path.dirname(new_path)
            os.makedirs(new_dir, exist_ok=True)
            torch.save(new_state_dict, new_path)