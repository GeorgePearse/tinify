# Layers

Neural network layers for compression models.

## GDN - Generalized Divisive Normalization

The GDN layer is commonly used in learned image compression for its effectiveness at decorrelating features.

::: tinify.layers.GDN
    options:
      members:
        - forward

## Attention Modules

::: tinify.layers.AttentionBlock
    options:
      show_source: false

## Convolutional Layers

::: tinify.layers.conv3x3

::: tinify.layers.subpel_conv3x3

## Residual Blocks

::: tinify.layers.ResidualBlock
    options:
      show_source: false

::: tinify.layers.ResidualBlockUpsample
    options:
      show_source: false

::: tinify.layers.ResidualBlockWithStride
    options:
      show_source: false
