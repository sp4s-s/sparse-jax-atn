from __future__ import annotations

from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import os

def load_corpus():
    corpus_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "friday_corpus.txt")
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return """
[SYSTEM_LOG]: Initializing F.R.I.D.A.Y. (Female Replacement Intelligent Digital Assistant Youth).
[SYSTEM_LOG]: Core subsystems coming online. Neural lattices stabilized at 104%.
[SYSTEM_LOG]: Interface protocols active. Multi-spectral sensors calibrated.
[FRIDAY]: Good morning, Boss. All Mark LXXXV systems are reporting nominal performance.
[FRIDAY]: Atmospheric conditions over Malibu are currently optimal for high-altitude flight testing.
[TONY]: Friday, run a full diagnostic on the repulsor stabilization units. I felt a slight shimmy during the last supersonic transition.
[FRIDAY]: Scanning now, Boss... I've detected a 0.04% variance in the lateral thrust vectors of the left boot assembly.
[FRIDAY]: It appears the vibration was caused by a microscopic stress fracture in the primary nozzle.
[TONY]: Can we compensate with the backup actuators or do I need to ground the suit?
[FRIDAY]: I can reroute 15% of the auxiliary power to the secondary stabilization grid. 
[FRIDAY]: That should stabilize flight for another 4 hours of mission time, though I recommend a full replacement once we return to the lab.
[TONY]: Great. How's the Arc Reactor holding up? Give me a power-to-thermal ratio.
[FRIDAY]: The core temperature is holding at a steady 1,200 Kelvin. Energy consumption is currently at 12% of peak output.
[FRIDAY]: However, if you initiate the 'House Party Protocol', the draw will increase by several orders of magnitude.
[SYSTEM_ALERT]: Warning—External integrity check failed. Plasma leakage detected in the primary cooling loop.
[FRIDAY]: Boss, we have a situation. The cooling system is losing pressure. At this rate, the core will reach critical temperature in under sixty seconds.
[TONY]: Dump the heat sink and vent the excess plasma into the vacuum. 
[FRIDAY]: Venting initiated. Emergency protocols engaged. Core temperature dropping... 1,100 Kelvin... 900 Kelvin.
[FRIDAY]: Crisis averted. Though I must say, Boss, your flair for the dramatic is becoming a significant variable in my predictive models.
[TONY]: That's what keeps life interesting, Friday. Set a course for the tower.
[FRIDAY]: Course plotted. Estimated time of arrival: 4 minutes and 12 seconds.
[FRIDAY]: Would you like me to prepare the cleaning droids for your arrival? The lab is currently at 85% capacity for clutter.
[TONY]: Just order some pizza, Friday. Extra pepperoni.
[FRIDAY]: On it, Boss. I've already anticipated your order and placed it three minutes ago based on your current blood sugar levels.
[SYSTEM_LOG]: Navigation locked. Propulsion system engaged. Destination: Stark Tower.
""".strip()

DEMO_CORPUS = load_corpus()

DEMO_PROMPT = "[FRIDAY]: Sir, I've detected a significant energy signature approaching from the East. "




def get_tokenizer():
    try:
        import tiktoken
        return tiktoken.get_encoding("gpt2")
    except ImportError:
        raise ImportError(
            "tiktoken is required for tokenization. "
            "Install with: pip install tiktoken"
        )


def tokenize_text(
    text: str,
    max_length: int = 1024,
    pad_to_max: bool = True,
) -> np.ndarray:
    enc = get_tokenizer()
    tokens = enc.encode(text)

    if len(tokens) > max_length:
        tokens = tokens[:max_length]

    tokens = np.array(tokens, dtype=np.int32)

    if pad_to_max and len(tokens) < max_length:
        padding = np.zeros(max_length - len(tokens), dtype=np.int32)
        tokens = np.concatenate([tokens, padding])

    return tokens


def decode_tokens(token_ids: np.ndarray) -> str:
    enc = get_tokenizer()
    valid_ids = [int(t) for t in token_ids if t > 0]
    return enc.decode(valid_ids)


def create_demo_batch(
    batch_size: int = 4,
    seq_len: int = 1024,
    seed: int = 42,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    tokens = tokenize_text(DEMO_CORPUS, max_length=seq_len + 1, pad_to_max=False)

    if len(tokens) < seq_len + 1:
        repeats = (seq_len + 1) // len(tokens) + 1
        tokens = np.tile(tokens, repeats)[:seq_len + 1]

    rng = np.random.RandomState(seed)
    max_start = max(1, len(tokens) - seq_len - 1)
    batch_inputs = []
    batch_targets = []

    for _ in range(batch_size):
        start = rng.randint(0, max_start)
        input_ids = tokens[start:start + seq_len]
        target_ids = tokens[start + 1:start + seq_len + 1]

        if len(input_ids) < seq_len:
            input_ids = np.pad(input_ids, (0, seq_len - len(input_ids)))
            target_ids = np.pad(target_ids, (0, seq_len - len(target_ids)))

        batch_inputs.append(input_ids)
        batch_targets.append(target_ids)

    input_ids = jnp.array(np.stack(batch_inputs), dtype=jnp.int32)
    target_ids = jnp.array(np.stack(batch_targets), dtype=jnp.int32)
    return input_ids, target_ids


def create_dummy_inputs(
    batch_size: int = 4,
    seq_len: int = 1024,
    n_heads: int = 8,
    d_head: int = 32,
    seed: int = 42,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rng = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(rng, 3)

    shape = (batch_size, seq_len, n_heads, d_head)
    query = jax.random.normal(k1, shape, dtype=dtype)
    key = jax.random.normal(k2, shape, dtype=dtype)
    value = jax.random.normal(k3, shape, dtype=dtype)

    return query, key, value


def create_random_token_batch(
    batch_size: int = 4,
    seq_len: int = 1024,
    vocab_size: int = 50257,
    seed: int = 42,
) -> jnp.ndarray:
    if isinstance(seed, (int, np.integer)):
        rng = jax.random.PRNGKey(int(seed))
    else:
        rng = seed
    return jax.random.randint(rng, (batch_size, seq_len), 0, vocab_size)
