# CLI Reference

Command-line tools for training, evaluation, and benchmarking.

## Training

The unified `tinify` CLI provides a standardized interface for training all compression models.

### Basic Usage

```bash
# Train with config file
tinify train image --config configs/mbt2018-mean.yaml

# Train with CLI arguments
tinify train image -m mbt2018-mean -d /path/to/dataset --epochs 100

# Train video model
tinify train video --config configs/ssf2020-video.yaml

# List available models
tinify train list-models --domain image
```

### Training Commands

| Command | Description |
|---------|-------------|
| `tinify train image` | Train image compression model |
| `tinify train video` | Train video compression model |
| `tinify train pointcloud` | Train point cloud compression model |
| `tinify train list-models` | List available models |

### Training Arguments

| Argument | Description |
|----------|-------------|
| `-c, --config` | Path to config file (YAML/JSON/TOML) |
| `-m, --model` | Model architecture name |
| `-q, --quality` | Quality level (1-8) |
| `-d, --dataset` | Path to training dataset |
| `-e, --epochs` | Number of training epochs |
| `--batch-size` | Training batch size |
| `--lambda` | Rate-distortion trade-off parameter |
| `-lr, --learning-rate` | Learning rate |
| `--cuda` / `--no-cuda` | Enable/disable CUDA |
| `--checkpoint` | Resume from checkpoint |
| `--save-dir` | Directory to save checkpoints |
| `--seed` | Random seed |

### Config File Format

```yaml
# configs/mbt2018-mean.yaml
domain: image

model:
  name: mbt2018-mean
  quality: 3

dataset:
  path: /path/to/vimeo90k
  patch_size: [256, 256]
  num_workers: 4

training:
  epochs: 300
  batch_size: 16
  lmbda: 0.0067
  metric: mse  # or ms-ssim
  cuda: true
  save_dir: ./checkpoints

optimizer:
  net: {type: Adam, lr: 0.0001}
  aux: {type: Adam, lr: 0.001}

scheduler:
  type: ReduceLROnPlateau
  patience: 20
```

### Available Models

**Image Compression:**
- `bmshj2018-factorized` - Factorized prior (Ballé 2018)
- `bmshj2018-hyperprior` - Scale hyperprior (Ballé 2018)
- `mbt2018-mean` - Mean-scale hyperprior (Minnen 2018)
- `mbt2018` - Joint autoregressive (Minnen 2018)
- `cheng2020-anchor` - Attention-based (Cheng 2020)
- `cheng2020-attn` - Attention with GMM (Cheng 2020)
- `*-vbr` - Variable bitrate variants

**Video Compression:**
- `ssf2020` - Scale-space flow (Agustsson 2020)

### Training Examples

```bash
# Train mbt2018-mean on Vimeo90K
tinify train image \
    -m mbt2018-mean \
    -d /data/vimeo90k \
    --epochs 300 \
    --batch-size 16 \
    --lambda 0.0067 \
    --cuda

# Resume training from checkpoint
tinify train image \
    --config configs/mbt2018-mean.yaml \
    --checkpoint checkpoints/checkpoint.pth.tar

# Train with MS-SSIM metric
tinify train image \
    -m mbt2018-mean \
    -d /data/vimeo90k \
    --lambda 8.73 \
    --config configs/mbt2018-mean.yaml
```

---

## Model Evaluation

### Evaluate Trained Checkpoints

```bash
python -m tinify.utils.eval_model checkpoint /path/to/images/ \
    -a ARCHITECTURE \
    -p /path/to/checkpoint.pth.tar
```

### Evaluate Pre-trained Models

```bash
python -m tinify.utils.eval_model pretrained /path/to/images/ \
    -a ARCHITECTURE \
    -q QUALITY_LEVELS
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `-a, --architecture` | Model architecture name |
| `-p, --path` | Path to checkpoint file |
| `-q, --quality` | Quality level(s) to evaluate |
| `--cuda` | Use GPU acceleration |
| `--half` | Use half precision (FP16) |

**Example:**

```bash
python -m tinify.utils.eval_model pretrained /path/to/kodak/ \
    -a mbt2018-mean -q 1 2 3 4 5 6 7 8 --cuda
```

## Codec Benchmarking

### BPG Codec

```bash
python -m tinify.utils.bench bpg /path/to/images/ [OPTIONS]
```

### VTM (VVC Reference)

```bash
python -m tinify.utils.bench vtm /path/to/images/ [OPTIONS]
```

### General Options

```bash
python -m tinify.utils.bench --help
```

## Plotting Results

Generate rate-distortion plots from evaluation results:

```bash
python -m tinify.utils.plot results.json [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--show` | Display plot interactively |
| `--output` | Save plot to file |
| `--metric` | Metric to plot (psnr, ms-ssim) |

## Video Compression

### Evaluate Video Models

```bash
# Trained checkpoint
python -m tinify.utils.video.eval_model checkpoint /path/to/videos/ \
    -a ssf2020 -p /path/to/checkpoint.pth.tar

# Pre-trained model
python -m tinify.utils.video.eval_model pretrained /path/to/videos/ \
    -a ssf2020 -q QUALITY_LEVELS
```

### Video Codec Benchmarks

```bash
# x265/HEVC
python -m tinify.utils.video.bench x265 /path/to/videos/

# VTM (VVC)
python -m tinify.utils.video.bench VTM /path/to/videos/
```

### Video Plot

```bash
python -m tinify.utils.video.plot results.json --show
```

## Model Update

Update entropy bottleneck parameters after training:

```bash
python -m tinify.utils.update_model \
    --architecture ARCHITECTURE \
    checkpoint.pth.tar
```

This updates the learned CDFs required for entropy coding.

## Example Workflow

1. **Train a model:**

    ```bash
    python examples/train.py -d /path/to/dataset/ \
        -a mbt2018-mean --epochs 300 --cuda --save
    ```

2. **Update the model:**

    ```bash
    python -m tinify.utils.update_model \
        --architecture mbt2018-mean \
        checkpoint_best_loss.pth.tar
    ```

3. **Evaluate:**

    ```bash
    python -m tinify.utils.eval_model checkpoint /path/to/kodak/ \
        -a mbt2018-mean -p checkpoint_best_loss.pth.tar --cuda
    ```

4. **Plot results:**

    ```bash
    python -m tinify.utils.plot results.json --show
    ```
