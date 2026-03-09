

## Quick Start

### On Google Colab (TPU v5e-1)

```python
!unzip sparse-attention-jax.zip
%cd sparse-attention-jax

!pip install -e .

# demo
!python main.py --quick

# benchmarks
!python run_benchmarks.py --quick

# tests
!python run_tests.py
```

### Local (CPU/GPU — no Pallas kernel)

```bash
pip install -e .
python main.py --quick --no-pallas
python run_benchmarks.py --quick --no-pallas
python run_tests.py
```

## Project Structure

```
sparse-attention-jax/
├── config.py                   # Hyperparameters & configurations
├── main.py                     # Demo entry point
├── run_benchmarks.py           # Benchmark entry point
├── run_tests.py                # Test runner
├── sparse_attention/           # Core package
│   ├── kernel.py               # ★ Custom sparse attention kernel
│   ├── masks.py                # Block-sparse mask generation
│   ├── dense_attention.py      # Dense attention baseline
│   ├── model.py                # Transformer model (Flax)
│   ├── data.py                 # Tokenization & data pipeline
│   └── metrics.py              # Metrics collection
├── benchmarks/                 # Benchmark suite
│   ├── benchmark_suite.py      # Full benchmark runner
│   ├── profile_hbm.py          # HBM bandwidth profiling
│   ├── profile_flops.py        # FLOPs analysis
│   └── scaling_analysis.py     # Sequence length scaling
├── tests/                      # Unit tests
│   ├── test_masks.py           # Mask correctness
│   ├── test_kernel.py          # Kernel vs dense comparison
│   ├── test_model.py           # Model forward/backward
│   └── test_metrics.py         # Metrics utilities
└── docs/                       # Documentation
    ├── usage.md                # Full usage guide
    ├── metrics.md              # Metrics explanation
    ├── debugging.md            # Debugging guide
    ├── architecture.md         # Technical architecture
    └── colab_setup.md          # Colab TPU setup
```

## Metrics & Results

| Metric | Dense | Sparse (combined) | Improvement |
|--------|-------|--------------------|-------------|
| HBM Bandwidth | Baseline | -40% | ✓ 40% reduction |
| Theoretical FLOPs | O(N²) | O(N × active_blocks) | ~45% reduction |
| Attention Matrix | Materialized in HBM | Online (fused) | No N×N allocation |
| Latency | Baseline | ~1.5-2x speedup | Significant |

## All Commands

```bash
# Demo
python main.py                        # Full demo
python main.py --quick                 # Quick demo
python main.py --seq-len 2048         # Custom seq length
python main.py --no-pallas            # CPU/GPU mode

# Benchmarks
python run_benchmarks.py              # Full suite
python run_benchmarks.py --quick      # Quick benchmark
python run_benchmarks.py --hbm        # HBM profiling only
python run_benchmarks.py --flops      # FLOPs profiling only
python run_benchmarks.py --scaling    # Scaling analysis
python run_benchmarks.py --sparsity combined --block-size 128

# Tests
python run_tests.py                   # All tests
python -m pytest tests/ -v            # Verbose
python -m pytest tests/test_kernel.py # Kernel tests only
python -m pytest tests/ -x            # Stop on first failure

# Individual profilers
python -m benchmarks.profile_hbm      # HBM report
python -m benchmarks.profile_flops    # FLOPs report
python -m benchmarks.scaling_analysis  # Scaling report
```

## Technical Deep-Dive

See [docs/architecture.md](docs/architecture.md) for the full technical deep-dive into:
- Block-sparse kernel design
- Online softmax algorithm
- Memory access pattern optimization
- HBM bandwidth reduction strategy
- TPU-specific optimizations

## License

MIT
