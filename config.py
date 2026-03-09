from dataclasses import dataclass, field
from typing import List, Literal

@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    d_head: int = 32
    n_layers: int = 4
    d_ff: int = 1024
    vocab_size: int = 50257
    max_seq_len: int = 2048
    dropout_rate: float = 0.1
    dtype: str = "bfloat16"

    def __post_init__(self):
        self.d_head = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0

@dataclass
class SparseConfig:
    block_size: int = 128
    sparsity_type: Literal["causal", "strided", "fixed", "random", "combined"] = "combined"
    local_window_blocks: int = 3
    global_stride: int = 4
    random_density: float = 0.5
    use_pallas_kernel: bool = True

@dataclass
class BenchmarkConfig:
    seq_lengths: List[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    n_warmup: int = 5
    n_iterations: int = 20
    quick_seq_lengths: List[int] = field(default_factory=lambda: [1024, 2048])
    quick_batch_sizes: List[int] = field(default_factory=lambda: [1, 4])

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 4
    seq_len: int = 1024
    n_steps: int = 100
    max_steps: int = 500
    warmup_steps: int = 10
    seed: int = 42

@dataclass
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sparse: SparseConfig = field(default_factory=SparseConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def for_quick_test(cls):
        return cls(
            model=ModelConfig(d_model=128, n_heads=4, n_layers=2, d_ff=512, max_seq_len=1024),
            sparse=SparseConfig(block_size=64),
            benchmark=BenchmarkConfig(seq_lengths=[512, 1024], batch_sizes=[1, 2], n_warmup=2, n_iterations=3),
            training=TrainingConfig(n_steps=10, seq_len=512)
        )

    @classmethod
    def for_full_benchmark(cls):
        return cls(
            model=ModelConfig(d_model=512, n_heads=8, n_layers=6, d_ff=2048, max_seq_len=4096),
            sparse=SparseConfig(block_size=128),
            benchmark=BenchmarkConfig(seq_lengths=[512, 1024, 2048, 4096], batch_sizes=[1, 2, 4, 8], n_warmup=5, n_iterations=20)
        )
