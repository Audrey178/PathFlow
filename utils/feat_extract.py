from dotenv import load_dotenv
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import os
import huggingface_hub
import torch
import argparse
from datasets_utils import FrameDataset
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms as T

# bật SDPA backend (FlashAttention, mem-efficient, math)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
huggingface_hub.login(token=hf_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNKSIZE = 50  # số frame xử lý mỗi lần

class TimmViTEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'hf-hub:MahmoodLab/UNI2-h', 
                 kwargs:dict = None):
        super().__init__()
        assert kwargs.get('pretrained', False), 'only pretrained models are supported'
        self.model = timm.create_model(model_name, **kwargs)
        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg))
        self.model_name = model_name
    
    def forward(self, x):
        try: 
            x_transformed = self.transform(x)
            feats = self.model(x_transformed)
            return feats
        except Exception as e:
            raise ValueError(f"Error at ViT: {e}")


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--split', required=True)
   
    
    args = parser.parse_args()
    data_dir = args.data_dir
    split = args.split
    
    timm_kwargs = {
        'pretrained': True,
        'img_size': 224, 
        'patch_size': 16, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
    }
    model = TimmViTEncoder(model_name='hf-hub:MahmoodLab/UNI2-h', kwargs=timm_kwargs)
    model.eval()
    model.to(device)
    
    print(f'Reading csv from {data_dir}/csv/{split}.csv')
    csv_path = f'{data_dir}/csv/{split}.csv'
    frames_dir = f"{data_dir}/frames"
    feats_dir = f"{data_dir}/feats"
    df = pd.read_csv(csv_path)
    slide_ids = df['slide_id'].tolist()
    labels_idx = df['label_idx'].tolist()
    labels = df['label'].tolist()
    print(f"Done!")
    print(f"Initializing dataset and dataloader...")
    dataset = FrameDataset(slide_ids, labels, csv_path=csv_path, root_dir=frames_dir)
    
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Done!")
    os.makedirs(feats_dir, exist_ok=True)
    transform = T.Compose([
        T.ToTensor(),
    ])
    
    print(f"Extracting features and saving to {feats_dir}...")
    with torch.no_grad():
        for i, (img_paths, slide_id) in enumerate(dataloader):
            img_paths = img_paths
            slide_id = slide_id[0]

            feat_file = Path(feats_dir) / f'{slide_id}.pt'
            
            if feat_file.exists():
                print(f'Features for slide {slide_id} already exist. Skipping...')
                continue
            all_feats = []
            chunk_size = min(CHUNKSIZE, len(img_paths))
            for paths in chunked(img_paths, chunk_size):
                imgs = []
                for p in paths:
                    p = p[0]  # unwrap from tensor
                    with Image.open(p) as img:
                        imgs.append(transform(img.convert("RGB")))

                imgs = torch.stack(imgs)
                imgs = imgs.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last
                )

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    feats = model(imgs)

                all_feats.append(feats.cpu())

            feats = torch.cat(all_feats, dim=0)  # [T, D]
            torch.save(feats, Path(feats_dir) / f"{slide_id}.pt")
            
            if (i + 1) % 10 == 0:
                print(f'Processed {i + 1}/{len(dataloader)} slides.')
                
if __name__ == "__main__":
    main()
    
