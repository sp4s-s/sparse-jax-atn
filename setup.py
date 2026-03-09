from setuptools import setup, find_packages

setup(
    name="sparse-attention-jax",
    version="1.0.0",
    description="Custom Sparse Attention Transformer with TPU Kernel in JAX/XLA",
    author="Portfolio Project",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "jax[tpu]>=0.4.30",
        "flax>=0.8.0",
        "optax>=0.2.0",
        "tiktoken>=0.5.0",
        "numpy>=1.24.0",
        "tabulate>=0.9.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "test": ["pytest>=7.0.0"],
    },
)
