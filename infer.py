#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Run PathFlow inference on a single video.")
    parser.add_argument("--video_path", required=True, help="Path to input video file.")
    parser.add_argument("--model_path", required=True, help="Path to trained classifier checkpoint.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>. Default: auto")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of output classes.")
    parser.add_argument(
        "--class_names",
        default="Normal,Adenoma,Malignant",
        help="Comma-separated class names in label_idx order.",
    )
    parser.add_argument("--ratio", type=float, default=0.1, help="Adaptive token pooling ratio.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Classifier dropout.")
    parser.add_argument("--hidden_size", type=int, default=1536, help="UNI2-h feature dimension.")
    parser.add_argument("--chunk_size", type=int, default=50, help="Frames encoded per forward pass.")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch value passed to VTransAdaptive.")
    parser.add_argument("--stride", type=int, default=1, help="Temporal stride passed to VTransAdaptive.")
    return parser.parse_args()


def require_runtime_imports():
    try:
        import cv2
        import timm
        import torch
        import huggingface_hub
        from dotenv import load_dotenv
        from PIL import Image
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from torchvision import transforms as T
        from models.vit_transformer_model import VTransAdaptive
        from utils.find_frames_idx import get_smart_indices_from_frames
    except ImportError as exc:
        raise SystemExit(f"\n[Error] Missing runtime dependency: {exc}\n") from exc

    return {
        "cv2": cv2,
        "timm": timm,
        "torch": torch,
        "huggingface_hub": huggingface_hub,
        "load_dotenv": load_dotenv,
        "Image": Image,
        "resolve_data_config": resolve_data_config,
        "create_transform": create_transform,
        "T": T,
        "VTransAdaptive": VTransAdaptive,
        "get_smart_indices_from_frames": get_smart_indices_from_frames,
    }


def resolve_device(torch, requested):
    if requested == "auto":
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but CUDA is not available.")
    return device


def build_uni2_encoder(rt, device):
    torch = rt["torch"]
    timm = rt["timm"]
    huggingface_hub = rt["huggingface_hub"]
    load_dotenv = rt["load_dotenv"]
    resolve_data_config = rt["resolve_data_config"]
    create_transform = rt["create_transform"]

    load_dotenv(REPO_ROOT / ".env")
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        huggingface_hub.login(token=hf_token)
    else:
        print("Warning: HF_TOKEN is not set. HuggingFace model loading may fail.")

    timm_kwargs = {
        "pretrained": True,
        "img_size": 224,
        "patch_size": 16,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", **timm_kwargs)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg))
    model.eval().to(device)
    return model, transform


def read_video_frames(rt, video_path):
    cv2 = rt["cv2"]
    Image = rt["Image"]

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames were read from video: {video_path}")
    return frames


def select_frames(rt, frames):
    get_smart_indices_from_frames = rt["get_smart_indices_from_frames"]
    indices = get_smart_indices_from_frames(frames)
    if not indices:
        indices = [0]
    return [frames[i] for i in indices], indices


def extract_features(rt, encoder, encoder_transform, frames, device, chunk_size):
    torch = rt["torch"]
    T = rt["T"]

    to_tensor = T.ToTensor()
    all_features = []

    with torch.no_grad():
        for start in range(0, len(frames), chunk_size):
            batch_frames = frames[start:start + chunk_size]
            batch = torch.stack([to_tensor(img.convert("RGB")) for img in batch_frames]).to(device)

            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    transformed = encoder_transform(batch)
                    features = encoder(transformed)
            else:
                transformed = encoder_transform(batch)
                features = encoder(transformed)

            all_features.append(features.float().cpu())

    return torch.cat(all_features, dim=0).to(device)


def load_classifier(rt, args, device):
    torch = rt["torch"]
    VTransAdaptive = rt["VTransAdaptive"]

    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = VTransAdaptive(
        num_classes=args.num_classes,
        ratio=args.ratio,
        dropout=args.dropout,
        hidden_dim=args.hidden_size,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: checkpoint missing {len(missing)} model keys.")
    if unexpected:
        print(f"Warning: checkpoint has {len(unexpected)} unexpected keys.")

    model.eval()
    return model


def predict(rt, classifier, features, device, epoch, stride):
    torch = rt["torch"]
    inputs = features.unsqueeze(0)
    attn_mask = torch.zeros((1, inputs.size(1)), dtype=torch.bool, device=device)

    with torch.no_grad():
        logits = classifier(inputs, attn_mask, epoch=epoch, stride=stride)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
    return pred_idx, probs.detach().cpu()


def main():
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be a positive integer.")
    if args.stride <= 0:
        raise ValueError("--stride must be a positive integer.")

    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    if len(class_names) != args.num_classes:
        raise ValueError("--class_names must contain exactly --num_classes names.")

    rt = require_runtime_imports()
    torch = rt["torch"]
    device = resolve_device(torch, args.device)

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    print(f"Running inference on {device}")
    print(f"1. Reading video: {args.video_path}")
    frames = read_video_frames(rt, args.video_path)

    selected_frames, selected_indices = select_frames(rt, frames)
    print(f"   Original frames: {len(frames)} | Selected frames: {len(selected_frames)}")

    print("2. Loading UNI2-h encoder...")
    encoder, encoder_transform = build_uni2_encoder(rt, device)

    print("3. Extracting features...")
    features = extract_features(rt, encoder, encoder_transform, selected_frames, device, args.chunk_size)
    print(f"   Features shape: {tuple(features.unsqueeze(0).shape)}")

    print("4. Loading classifier...")
    classifier = load_classifier(rt, args, device)

    print("5. Predicting...")
    pred_idx, probs = predict(rt, classifier, features, device, args.epoch, args.stride)

    video_name = Path(args.video_path).name
    print("\n" + "=" * 40)
    print(f"VIDEO: {video_name}")
    print(f"SELECTED_FRAME_INDICES: {selected_indices}")
    print(f"PREDICTION: {class_names[pred_idx]} (class {pred_idx})")
    print(f"CONFIDENCE: {float(probs[pred_idx]):.2%}")
    print("PROBABILITIES:")
    for idx, prob in enumerate(probs.tolist()):
        print(f"  {class_names[idx]}: {prob:.2%}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
