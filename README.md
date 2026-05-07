# PathFlow: A Motion-Aware Prototyping Framework for Microscopic Pathology Video Classification

## Overview

This repository contains the implementation of PathFlow, a framework for video analysis with adaptive token selection and masking strategies. The codebase includes data preparation, feature extraction, model training, and evaluation pipelines.

We provide a subset of videos to facilitate reproducibility of our results: [Download](https://dataverse.harvard.edu/previewurl.xhtml?token=b328291f-18d1-4911-83e4-371880d281db). For your own dataset, follow the steps below to prepare data.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Inference](#inference-on-single-video)
6. [Video demo](#video-demo)

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- FFmpeg (for video processing)

### Setup

1. Clone the repository and navigate to the project directory:

```bash
cd PathFlow
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables (for HuggingFace token, WanDB token):

Create a `.env` file in the project root (.env.example) and add your HuggingFace token and Weights & Biases token:

```
HF_TOKEN=your_huggingface_token_here
WANDB_TOKEN=your_wandb_token_here
```

## Data Preparation

### Overview

For your own dataset, you can modify and run following this steps:

### Step 1: Prepare Dataset CSV Files and Data Directory

Organize your dataset directory as follows:

```datasets/
├── videos/
│   ├── Label_1/
│   │   ├── video_001.mp4
│   │   ├── video_002.mp4
│   ├── Label_2/
│   │   ├── video_003.mp4
│   │   ├── video_004.mp4
│   └── ...
├── csv/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── feats/  # Directory for extracted features
│
└── frames/ # Directory for extracted frames

```

Create CSV files for train/val/test splits in `datasets/csv/`:

**Format:** `{split}.csv`

```csv
slide_id,label,label_idx,path
video_001,Normal,0,datasets/videos/Normal/video_001.mp4
video_002,Adenoma,1,datasets/videos/Adenoma/video_002.mp4
...
```

### Step 2: Run the Data Preparation Pipeline

Run the preparation script for each split you want to process:

```bash
bash scripts/prepare_data.sh --split train
bash scripts/prepare_data.sh --split val
bash scripts/prepare_data.sh --split test
```

The script runs the full preparation pipeline:

1. Extract frames from videos with `preprocessing_video.py`
2. Select representative frames with `utils/finding_frame_idx_OF.py`
3. Extract UNI2-h features with `utils/feat_extract.py`

Full usage:

```bash
bash scripts/prepare_data.sh \
    --data_dir datasets \
    --split train \
    --max_workers 4
```

The script creates a backup of the split CSV before optical-flow processing:

```bash
datasets/csv/train.csv.bak
```

Before running the script, make sure:

- FFmpeg is installed and available in your system `PATH`
- Python dependencies are installed from `requirements.txt`
- `HF_TOKEN` is set in the environment or in `.env`
- You have access to the HuggingFace model `MahmoodLab/UNI2-h`

#### Optical Flow-based Frame Selection

This stage is already included in `scripts/prepare_data.sh`. It analyzes extracted frames and writes selected frame indices back to the split CSV.

The optical-flow stage selects representative frames based on:

- **Motion detection**: Tracks pixel-level motion between consecutive frames
- **Motion patterns**: Classifies scanning modes (SLOW_SWEEP, NORMAL_RASTER, FAST_SWEEP)
- **Adaptive sampling**: Selects frames based on accumulated motion distance
- **Turn detection**: Detects direction changes in raster scanning patterns

The algorithm detects:

- **MIN_FEATURES**: Minimum texture features required (50)
- **MIN_MOTION_PX**: Minimum motion threshold in pixels (2)
- **MAX_JITTER_PX**: Maximum jitter tolerance (3)
- **OVERLAP_RATIO**: Overlap ratio for stitching (0.2)

The updated CSV will include a `selected_frames` column containing the indices of keyframes:

```csv
slide_id,label,label_idx,path,selected_frames
video_001,Normal,0,datasets/videos/Normal/video_001.mp4,"[0, 5, 12, 28, 45, ...]"
video_002,Adenoma,1,datasets/videos/Adenoma/video_002.mp4,"[0, 8, 19, 31, 52, ...]"
...
```

### Optional: Manual Commands

If you need to debug one stage at a time, you can run the commands manually:

```bash
python preprocessing_video.py \
    --video_dir datasets/videos \
    --frame_dir datasets/frames \
    --max_workers 4

python utils/finding_frame_idx_OF.py \
    --csv_path datasets/csv/train.csv

python utils/feat_extract.py \
    --data_dir datasets \
    --split train
```

Feature extraction uses the **UNI2-h** checkpoint provided by MahmoodLab. If you do not already have access, follow the HuggingFace repository instructions before running this step.

## Training

### Baseline Model

To train the baseline:

```bash
bash scripts/run.sh
```

Or manually:

```bash
python main.py --config-name baseline
```

### Configuration

Edit `configs/baseline.yaml` to modify training parameters:

Example config:

```yaml
strategy: "baseline"
seed: 512
batch_size: 8
num_classes: 3
max_epochs: 100
lr: 2e-4
dropout: 0.2
ratio: 0.1
```

### Training with Custom Parameters

Override config parameters from command line:

```bash
python main.py \
    --config-name baseline \
    seed=1024 \
    batch_size=16 \
    lr=1e-4 \
    top_k=512 \
    save_name="custom_experiment"
```

## Evaluation

Evaluate trained models:

```bash
python eval.py --config-name baseline model_path results/baseline/best_model.pt
```

Or using the provided evaluation script:

```bash
bash scripts/eval.sh
```

### Notes

- Ensure `HF_TOKEN` is set in your `.env` file for HuggingFace model access
- The inference uses the same optical flow-based frame selection as training

## Key Features

- **Adaptive Token Selection**: Dynamically select relevant tokens based on optical flow analysis
- **Masking Strategies**: Evaluate different masking approaches for robustness
- **Multi-GPU Support**: Efficient training with distributed data parallel
- **Weights & Biases Integration**: Experiment tracking via W&B

## Performance Monitoring

Training metrics are logged to Weights & Biases (W&B). Configure in `baseline.yaml`:

```yaml
wandb: True # Set to False to disable W&B logging
```

View experiments at: https://wandb.ai/

## Setting up Web Demo

### 1. Prerequisites

Ensure your host machine has the following installed:

- **Docker** and **Docker Compose**
- **NVIDIA Container Toolkit** (for GPU support)
- **NVIDIA Drivers**

### 2. Build and Run

Navigate to the project directory and launch the system using Docker Compose:

```bash
cd Pathflow
docker compose up -d --build
```

### 3. Usage

Open your browser to http://localhost:8502.

Input: Upload a video study.

Output: The predicted class Normal, Adenoma or Malignant and the model's confidence score.

## Video demo

![PathFlow Demo](demo.gif)

▶ Full video: [Download clean_demo.mp4](clean_demo.mp4)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{PathFlow2026,
  title={PathFlow:  A Motion-Aware Prototyping Framework for Microscopic Pathology Video Classification},
  author={Your Name},
  journal={Your Journal},
  year={2026}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please contact: [your-email@example.com]
