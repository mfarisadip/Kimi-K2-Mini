# Kimi-K2-Mini 🚀

A miniaturized version of the Kimi-K2 model optimized for deployment on single H100 GPUs.

## Overview

Kimi-K2-Mini is an experimental compressed version of the 1.07T parameter Kimi-K2 model, targeting ~32.5B parameters for more accessible deployment. This project explores several optimization strategies including architecture reduction, expert pruning, and quantization techniques.

**⚠️ Experimental Project**: This is an active research project exploring model compression techniques. Please see [STATUS.md](STATUS.md) for current implementation status.

## Target Specifications

| Parameter | Original K2 | K2-Mini Target |
|-----------|------------|---------|
| Total Parameters | 1.07T | ~32.5B |
| Layers | 61 | 24 |
| Experts per Layer | 384 | 16 |
| Memory (BF16) | ~2TB | ~60GB |
| Hidden Size | 7168 | 7168 |
| Vocab Size | 163,840 | 163,840 |

## Optimization Strategies

- **Architecture reduction**: Intelligently selecting 24 most important layers from 61
- **Expert pruning**: Reducing MoE experts from 384 to 16 per layer  
- **Quantization support**: Exploring INT8/INT4 for further memory reduction
- **FP8 compatibility**: Handling FP8 model weights and conversions
- **Dynamic loading**: Smart expert caching and swapping concepts

## Research Goals

- 🎯 Enable deployment on single H100 (80GB) GPU
- 📉 Reduce memory footprint to ~60GB (bfloat16)
- 🎭 Preserve core model capabilities where possible
- ⚡ Achieve meaningful inference speedup
- 🔧 Develop reusable compression techniques

## Installation

```bash
git clone https://github.com/peteryuqin/Kimi-K2-Mini.git
cd Kimi-K2-Mini
pip install -r requirements.txt
```

## Model Creation (Experimental)

### Intelligent Layer Selection

```bash
# Analyze and convert with intelligent layer/expert selection
python scripts/convert_to_mini.py \
    --source-model /path/to/kimi-k2-instruct \
    --output-path ./k2-mini \
    --num-layers 24 \
    --experts-per-layer 16
```

### Fast Conversion

```bash
# Quick conversion with uniform layer selection
python scripts/convert_to_mini_fast.py \
    --source-model /path/to/kimi-k2-instruct \
    --output-path ./k2-mini
```

## Project Structure

```
Kimi-K2-Mini/
├── src/
│   ├── expert_selector.py    # Expert selection algorithms
│   ├── quantization.py       # Quantization utilities
│   └── inference.py          # Optimized inference
├── scripts/
│   ├── analyze_layers.py     # Layer importance analysis
│   ├── convert_to_mini.py    # Intelligent conversion
│   └── convert_to_mini_fast.py # Fast conversion
├── configs/
│   └── k2_mini_config.json   # Model configuration
├── test_*.py                 # Testing scripts
├── fix_*.py                  # Utility scripts
└── utils/
    └── memory_utils.py       # Memory optimization tools
```

## Testing & Validation

The project includes various testing scripts for different scenarios:

```bash
# Basic model loading test
python test_k2mini_simple.py

# CloudExe GPU testing
python test_k2mini_cloudexe.py

# Inference validation
python test_k2mini_inference.py
```

## Technical Approach

### Layer Selection Strategy
- Analyze layer importance using gradient-based metrics
- Preserve critical layers for reasoning and generation
- Maintain model coherence across selected layers

### Expert Compression
- Identify most activated experts per layer
- Merge similar expert patterns where possible
- Optimize routing efficiency for reduced expert count

### Memory Optimization
- FP8 to FP16 conversion handling
- Dynamic expert loading strategies
- Efficient weight storage and retrieval

## Research Status

This project is actively exploring model compression techniques. Current development focuses on:

- Resolving weight compatibility issues
- Optimizing expert selection algorithms  
- Improving inference pipeline stability
- Validating compression effectiveness

For detailed status updates, see [STATUS.md](STATUS.md).

## Contributing

This is an experimental research project. Contributions are welcome in the form of:

- Compression algorithm improvements
- Testing and validation scripts
- Documentation and examples
- Performance optimizations

## Citation

If you find this research useful, please cite:

```bibtex
@software{kimi-k2-mini,
  title = {Kimi-K2-Mini: Experimental Model Compression Research},
  author = {Peter Yu Qin},
  year = {2025},
  url = {https://github.com/peteryuqin/Kimi-K2-Mini}
}
```

## License

This project is licensed under the Apache 2.0 License.

## Acknowledgments

- Original Kimi-K2 model by Moonshot AI
- CloudExe team for GPU infrastructure support
- Open source community for inspiration and tools

---

**Note**: This is experimental research into model compression techniques. The goal is advancing understanding of efficient large model deployment rather than producing production-ready software.

## Latest Update (2025-07-14)

### Current Status: Model loads but inference fails ⚠️

We successfully compressed Kimi-K2 from 1.07T to 32.5B parameters and the model loads on H100 GPU. However, inference fails due to numerical instability issues. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed analysis.

### Key Findings
- ✅ Model compression: 1.07T → 32.5B (successful)
- ✅ Memory usage: ~40GB (fits in single H100)
- ❌ Inference: Fails with CUDA assertion errors
- 📊 Root cause: 98% parameter reduction too aggressive

### Recommended Next Steps
Instead of extreme compression, we recommend **Dynamic Expert Loading**:
- Keep all 384 experts but load on-demand
- Use 2TB CPU memory for caching
- Trade inference speed for model quality

See [FUTURE_WORK.md](FUTURE_WORK.md) for the proposed architecture.

---

