# Models

Compression model architectures.

## Base Classes

::: tinify.models.CompressionModel
    options:
      members:
        - forward
        - aux_loss
        - update
        - load_state_dict

## Image Compression Models

### Factorized Prior

::: tinify.models.FactorizedPrior
    options:
      show_source: false

### Scale Hyperprior

::: tinify.models.ScaleHyperprior
    options:
      show_source: false

### Mean-Scale Hyperprior

::: tinify.models.MeanScaleHyperprior
    options:
      show_source: false

### Joint Autoregressive Hierarchical Priors

::: tinify.models.JointAutoregressiveHierarchicalPriors
    options:
      show_source: false

## Attention-based Models

::: tinify.models.Cheng2020Anchor
    options:
      show_source: false

::: tinify.models.Cheng2020Attention
    options:
      show_source: false

## Utility Functions

::: tinify.models.utils.conv

::: tinify.models.utils.deconv
