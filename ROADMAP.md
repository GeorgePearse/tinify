# Tinify Roadmap

## Completed

- [x] **Rust Extensions (PyO3)** - Ported C++ pybind11 extensions to Rust
  - rANS entropy encoder/decoder
  - pmf_to_quantized_cdf operation
  - Python wrappers with automatic backend selection

- [x] **MkDocs Documentation** - Set up documentation site with GitHub Actions deployment

- [x] **CLI for Training** - Command-line interface for model training

## In Progress

- [ ] **Code Cleanup** - Remove unused legacy code, standardize imports

## Planned

### Triton Kernels (triton-lang)

Custom GPU kernels using [Triton](https://github.com/triton-lang/triton) for performance-critical operations:

- [ ] **Fused Entropy Coding** - GPU-accelerated rANS encode/decode
- [ ] **Fused GDN/IGDN** - Generalized Divisive Normalization as single kernel
- [ ] **Attention Kernels** - Optimized attention for transformer-based models
- [ ] **Quantization Kernels** - Fused quantize + noise injection for training
- [ ] **Convolution Fusions** - Conv + activation + normalization fused ops

Benefits:
- Eliminate CPU-GPU memory transfers for entropy coding
- Reduce kernel launch overhead
- Custom memory layouts optimized for compression workloads

### Lightning Fabric Integration

Migrate training infrastructure to [Lightning Fabric](https://lightning.ai/docs/fabric/stable/):

- [ ] **Multi-GPU Training** - Simplified DDP/FSDP support
- [ ] **Mixed Precision** - Native FP16/BF16 training with automatic scaling
- [ ] **Checkpointing** - Unified checkpoint format with automatic resume
- [ ] **Logging Integration** - TensorBoard, W&B, MLflow support
- [ ] **Distributed Inference** - Model parallel inference for large models

Benefits:
- Cleaner training code without boilerplate
- Easy switching between single-GPU, DDP, FSDP
- Built-in gradient accumulation and clipping
- Hardware-agnostic code (CUDA, MPS, TPU)

### Model Improvements

- [ ] **Variable Rate Models** - Single model for multiple bitrates
- [ ] **Lightweight Decoders** - Asymmetric encoder-decoder for mobile deployment
- [ ] **Neural Codec Layers** - Pluggable compression layers for other tasks

### Infrastructure

- [ ] **ONNX Export** - Export trained models to ONNX format
- [ ] **TensorRT Integration** - Optimized inference with TensorRT
- [ ] **Benchmark Suite** - Standardized benchmarks (Kodak, CLIC, Tecnick)
- [ ] **Pre-trained Model Hub** - Easy access to pre-trained checkpoints

## Future Considerations

- **Video Compression** - Extend to learned video codecs
- **Point Cloud Compression** - 3D data compression
- **Neural Audio Compression** - Audio codec integration
- **Codec Compliance** - JPEG AI / VVC integration testing

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this roadmap.
