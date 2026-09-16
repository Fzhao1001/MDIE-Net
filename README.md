# MDIE-Net

## Requirements

- Python >= 3.7
- PyTorch >= 1.7
- torchvision
- basicsr
- numpy
- Pillow
- tqdm

## Dataset Preparation

The dataset directory structure should be:

```
data/
├── train/
│   ├── raw/    # training input images (hazy / low-light)
│   └── gt/     # training ground-truth images (clear)
└── test/
    ├── raw/    # test input images
    └── gt/     # test ground-truth images
```

Set the `path` variable in `data_utils.py` to your dataset root directory:

```python
path = '/path/to/your/data'
```

## Training

### Basic Usage

```bash
python main.py
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 200 | Number of training epochs |
| `--lr` | 0.0001 | Initial learning rate |
| `--net` | MFEdiff | Network name |
| `--bs` | 2 | Batch size |
| `--eval_epoch` | 5 | Evaluate every N epochs |
| `--no_lr_sche` | False | Disable cosine learning rate decay |
| `--resume` | False | Resume from checkpoint |
| `--model_dir` | ./trained_models/ | Model save directory |
| `--trainset` | its_train | Training set name |
| `--testset` | its_test | Test set name |
| `--crop` | False | Enable random cropping |
| `--crop_size` | 256 | Crop size (requires `--crop`) |

### Examples

Train with default settings:

```bash
python main.py
```

Train with custom parameters:

```bash
python main.py --epochs 200 --lr 0.0002 --bs 2 --eval_epoch 10
```

### Training Outputs

- **Model weights**: saved to `trained_models/<trainset>_<net>_<gps>_<blocks>.pk`
- **Loss/metrics logs**: saved as `.npy` files under `numpy_files/`
- **Logs**: saved under `logs/<model_name>/`

### Learning Rate Schedule

Default cosine decay:

```
lr = 0.5 × (1 + cos(epoch × π / T)) × init_lr
```

## Testing

### Basic Usage

```bash
python dehaze.py
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-d` / `--dataset_name` | LOL | Dataset name |
| `--save_dir` | '' | Output directory for dehazed images |
| `--save` | False | Save dehazed images |

### Configuration

Set the following variables in `dehaze.py` before running:

```python
haze_dir = '/path/to/your/hazy/images'
clear_dir = '/path/to/your/clear/images'
model_dir = '/path/to/your/model.ckp'
```

### Examples

```bash
python dehaze.py -d LOL --save_dir ./results/ --save
```

### Testing Outputs

- Dehazed images saved to `<save_dir>/<dataset_name>/`
- Average PSNR and SSIM printed to console