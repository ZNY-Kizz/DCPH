# DCPH: Deep Contrastive Hashing with Proxy Guidance

Implementation of DCPH for video-text cross-modal retrieval using deep contrastive hashing with proxy guidance.

## Requirements

Python 3.7+
PyTorch 1.8+
CUDA (for GPU training)

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

Prepare your dataset in the following format:
Each NPZ file should contain video_feature, text_feature, and label.
JSON files should contain video IDs and corresponding label information.

## Usage

### Training

```bash
python train.py
```

Specify binary dimension (16, 32, 64, 128, or 256):

```bash
python train.py --binary_dim 64
```

Specify dataset paths:

```bash
python train.py --train_json path/to/train.json --val_json path/to/val.json --test_json path/to/test.json --npz_dir path/to/npz_features
```

### Testing

```bash
python test.py --model_path work_dirs/64/TIMESTAMP
```

## Configuration

Edit config.py to adjust hyperparameters:
binary_dim: Hash code dimension
num_classes: Number of classes
batch_size: Batch size
lr: Learning rate
device: cuda:0 or cpu

## Notes

Make sure codetable.xlsx is in the project root directory.
Dataset paths can be specified via command line arguments. If not provided, default paths will be used.
CLIP model will be downloaded automatically on first run.


