# Datasets

Dataset loading utilities for training and evaluation.

## Image Datasets

### ImageFolder

Custom ImageFolder dataset for training compression models.

::: tinify.datasets.ImageFolder
    options:
      show_source: false

## Video Datasets

::: tinify.datasets.VideoFolder
    options:
      show_source: false

## Dataset Structure

Tinify datasets follow this structure:

```
dataset/
├── train/
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── test/
    ├── image001.png
    ├── image002.png
    └── ...
```

Images should be in PNG or JPEG format with RGB channels.
