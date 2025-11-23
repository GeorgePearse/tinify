# Metrics

Tinify provides several metrics to evaluate the quality and performance of compression models. These metrics are commonly used in image and video compression research.

## Image Metrics

### Quality Metrics

#### PSNR (Peak Signal-to-Noise Ratio)
The Peak Signal-to-Noise Ratio (PSNR) is a widely used metric to measure the quality of reconstruction of lossy compression codecs. It is defined as the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation.

- **psnr-rgb**: PSNR calculated on the RGB channels.

#### MS-SSIM (Multi-Scale Structural Similarity Index)
The Multi-Scale Structural Similarity Index (MS-SSIM) provides a quality measure that is often more consistent with human visual perception than PSNR. It is calculated over multiple scales.

- **ms-ssim-rgb**: MS-SSIM calculated on the RGB channels.

### Compression Metrics

#### BPP (Bits Per Pixel)
Bits Per Pixel (bpp) measures the average number of bits used to store one pixel of the image. It is calculated as:

$$ \text{bpp} = \frac{\text{Total Bits}}{\text{Total Pixels}} $$

### Performance Metrics

- **encoding_time**: Time taken to encode the image (in seconds).
- **decoding_time**: Time taken to decode the image (in seconds).

## Video Metrics

### Quality Metrics

Video quality is often evaluated on both RGB and YCbCr color spaces.

#### YCbCr Metrics
- **psnr-y**: PSNR calculated on the Luma (Y) channel.
- **psnr-u**: PSNR calculated on the Cb (U) channel.
- **psnr-v**: PSNR calculated on the Cr (V) channel.
- **psnr-yuv**: A weighted average of the Y, U, and V PSNRs, typically calculated as:
  $$ \text{PSNR}_{YUV} = \frac{4 \cdot \text{PSNR}_Y + \text{PSNR}_U + \text{PSNR}_V}{6} $$

#### RGB Metrics
- **psnr-rgb**: PSNR calculated on the RGB channels.
- **ms-ssim-rgb**: MS-SSIM calculated on the RGB channels.
- **mse-rgb**: Mean Squared Error (MSE) calculated on the RGB channels.

### Compression Metrics

#### Bitrate
For video, compression efficiency is often measured in bitrate (e.g., kbps) or bits per pixel.

- **bitrate**: The average bitrate of the video stream.
- **bpp**: Bits Per Pixel, averaged over all frames.

### Performance Metrics

- **encoding_time**: Total time taken to encode the video sequence.
- **decoding_time**: Total time taken to decode the video sequence.

## Loss Metrics

During training, the following metrics are often used as components of the loss function:

- **mse_loss**: Mean Squared Error between the original and reconstructed images/frames.
- **ms_ssim_loss**: $1 - \text{MS-SSIM}$.
- **bpp_loss**: The estimated bitrate (entropy) of the latent representation.
