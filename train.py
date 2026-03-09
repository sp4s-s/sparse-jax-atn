from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tensorboardX import SummaryWriter
from tqdm import tqdm

from config import ProjectConfig
from sparse_attention.model import create_model, init_model, count_parameters
from sparse_attention.masks import create_block_mask
from sparse_attention.data import DEMO_CORPUS, DEMO_PROMPT, decode_tokens, tokenize_text
from sparse_attention.live_viz import LiveNotebookDisplay, render_training_live
from sparse_attention.runtime_backend import resolve_training_backend


class TrainState(train_state.TrainState):
    pass


def create_train_state(rng, config, model, batch_size, seq_len):
    variables = init_model(model, rng, batch_size=1, seq_len=seq_len)
    params = variables["params"]

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,
        peak_value=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        decay_steps=config.training.max_steps,
        end_value=1e-6,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=config.training.weight_decay)
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


@jax.jit
def train_step(state: TrainState, batch: jnp.ndarray, dropout_rng: jnp.ndarray):
    def loss_fn(params):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = state.apply_fn(
            {"params": params}, 
            inputs, 
            deterministic=False, 
            rngs={"dropout": dropout_rng}
        )
        vocab_size = logits.shape[-1]
        one_hot_targets = jax.nn.one_hot(targets, vocab_size)
        log_probs = jax.nn.log_softmax(logits)
        per_token_loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
        loss = jnp.mean(per_token_loss)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
    
    metrics = {
        "loss": loss,
        "grad_norm": grad_norm,
    }
    return state, metrics


@jax.jit
def eval_step(state: TrainState, batch: jnp.ndarray):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = state.apply_fn({"params": state.params}, inputs, deterministic=True)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    preds = jnp.argmax(logits, axis=-1)
    one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])
    per_token_loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
    loss = jnp.mean(per_token_loss)
    accuracy = jnp.mean((preds == targets).astype(jnp.float32))
    return {
        "loss": loss,
        "accuracy": accuracy,
    }


def write_generation_snapshot(
    state: TrainState,
    generate_token_fn,
    output_path: str,
    prompt_text: str,
    seq_len: int,
    total_new_tokens: int = 100,
) -> None:
    prompt_tokens = tokenize_text(prompt_text, max_length=min(seq_len, 128), pad_to_max=False)
    generated = [int(token) for token in prompt_tokens]
    context = list(generated[-seq_len:]) or [0]
    target_total_tokens = len(prompt_tokens) + total_new_tokens
    prompt_text_decoded = decode_tokens(np.array(prompt_tokens, dtype=np.int32))

    while len(generated) < target_total_tokens:
        window = np.array(context[-seq_len:], dtype=np.int32)
        input_ids = jnp.array(window[None, :], dtype=jnp.int32)
        next_token = int(generate_token_fn(state.params, input_ids)[0])
        generated.append(next_token)
        context.append(next_token)

    full_text = decode_tokens(np.array(generated, dtype=np.int32))
    continuation_text = full_text[len(prompt_text_decoded):] if full_text.startswith(prompt_text_decoded) else full_text
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("=== PROMPT ===\n")
        handle.write(prompt_text_decoded)
        handle.write("\n\n=== GENERATED CONTINUATION ===\n")
        handle.write(continuation_text)
        handle.write("\n")


def ngram_repetition_score(text: str, n: int = 4) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    unique = len(set(ngrams))
    total = len(ngrams)
    return 1.0 - (unique / total if total else 1.0)


def evaluate_state(
    state: TrainState,
    eval_batch: jnp.ndarray,
    generate_token_fn,
    generation_path: str,
    prompt_text: str,
    seq_len: int,
) -> Dict[str, float]:
    metrics = eval_step(state, eval_batch)
    loss = float(metrics["loss"])
    accuracy = float(metrics["accuracy"])
    perplexity = float(np.exp(min(loss, 20.0)))
    write_generation_snapshot(state, generate_token_fn, generation_path, prompt_text, seq_len)
    with open(generation_path, "r", encoding="utf-8") as handle:
        generated_text = handle.read()
    repetition = ngram_repetition_score(generated_text, n=4)
    return {
        "validation_loss": loss,
        "perplexity": perplexity,
        "token_accuracy": accuracy,
        "repetition_4gram": repetition,
    }


def build_corpus_batches(
    text: str,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> tuple[jnp.ndarray, callable]:
    tokens = tokenize_text(text, max_length=max(seq_len * 64, seq_len + 2), pad_to_max=False)
    if len(tokens) < seq_len + 1:
        repeats = (seq_len + 1) // len(tokens) + 2
        tokens = np.tile(tokens, repeats)
    rng = np.random.RandomState(seed)

    def next_batch() -> jnp.ndarray:
        max_start = max(1, len(tokens) - seq_len - 1)
        starts = rng.randint(0, max_start, size=batch_size)
        rows = [tokens[start:start + seq_len + 1] for start in starts]
        batch = np.stack(rows).astype(np.int32)
        return jnp.array(batch, dtype=jnp.int32)

    fixed_start = min(max(0, len(tokens) - (batch_size * (seq_len + 1))), seq_len)
    eval_rows = []
    for idx in range(batch_size):
        start = (fixed_start + idx * (seq_len // 2 + 1)) % max(1, len(tokens) - seq_len - 1)
        eval_rows.append(tokens[start:start + seq_len + 1])
    eval_batch = jnp.array(np.stack(eval_rows).astype(np.int32), dtype=jnp.int32)
    return eval_batch, next_batch


def train(
    attention_type: str = "sparse",
    sparsity_type: str = "combined",
    block_size: int = 128,
    use_pallas: bool = True,
    max_steps: int = 500,
    log_dir: str = "tensorboard_logs",
    quick: bool = False,
):
    cfg = ProjectConfig.for_quick_test() if quick else ProjectConfig()
    cfg.training.max_steps = max_steps
    batch_size = cfg.training.batch_size
    seq_len = cfg.training.seq_len

    backend = resolve_training_backend(prefer_tpu=True)
    print(f"training {attention_type} | backend={backend} | B={batch_size} N={seq_len} steps={max_steps}")
    print(f"Log: {log_dir}")
    
    os.makedirs(log_dir, exist_ok=True)
    run_dir = os.path.join(log_dir, f"run_{attention_type}_{int(time.time())}")
    writer = SummaryWriter(logdir=run_dir)
    
    rng = jax.random.PRNGKey(cfg.training.seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    
    if attention_type == "sparse":
        block_mask = create_block_mask(seq_len, block_size, sparsity_type)
        print(f"Using Mask: {block_mask.summary()}")
    else:
        block_mask = None
        
    model = create_model(
        attention_type=attention_type,
        block_mask=block_mask,
        use_pallas=use_pallas,
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        max_seq_len=cfg.model.max_seq_len,
    )
    
    print("Initializing model...")
    state = create_train_state(init_rng, cfg, model, batch_size, seq_len)
    generate_token_fn = jax.jit(
        lambda params, input_ids: jnp.argmax(
            model.apply({"params": params}, input_ids, deterministic=True)[:, -1, :],
            axis=-1,
        )
    )
    
    print(f"Trainable Parameters: {count_parameters(state.params):,}\n")
    pretrain_sample_path = os.path.join(run_dir, "generation_before_training.txt")
    posttrain_sample_path = os.path.join(run_dir, "generation_after_training.txt")
    live = LiveNotebookDisplay("training", run_dir, "live_training.html", min_interval_steps=20)
    eval_batch, next_batch = build_corpus_batches(DEMO_CORPUS, batch_size, seq_len, cfg.training.seed)
    print(f"Writing pre-training sample to {pretrain_sample_path}")
    baseline_eval = evaluate_state(
        state,
        eval_batch,
        generate_token_fn,
        pretrain_sample_path,
        DEMO_PROMPT,
        seq_len,
    )
    print("Pre-training sample saved.")
    print(
        "Baseline metrics | "
        f"val_loss={baseline_eval['validation_loss']:.4f} "
        f"ppl={baseline_eval['perplexity']:.2f} "
        f"acc={baseline_eval['token_accuracy']:.4f} "
        f"rep4={baseline_eval['repetition_4gram']:.4f}"
    )

    pbar = tqdm(range(max_steps), desc="Training")
    step = -1
    step_times = []
    steps_hist = []
    loss_hist = []
    grad_hist = []
    lr_hist = []
    tok_hist = []
    try:
        dummy_batch = next_batch()
        print("Compiling training step...")
        train_step(state, dummy_batch, dropout_rng)[0].params
        print("Done compiling.\n")
        for step in pbar:
            batch = next_batch()
            rng, dropout_rng = jax.random.split(rng)
            step_start = time.perf_counter()
            state, metrics = train_step(state, batch, dropout_rng)
            jax.block_until_ready(metrics["loss"])
            step_duration = time.perf_counter() - step_start
            step_times.append(step_duration)
            loss_val = float(metrics["loss"])
            grad_norm_val = float(metrics["grad_norm"])
            tokens_per_sec = (batch_size * seq_len) / step_duration

            writer.add_scalar("Loss/train", loss_val, step)
            writer.add_scalar("Optimization/grad_norm", grad_norm_val, step)
            writer.add_scalar("Performance/step_time_ms", step_duration * 1000, step)
            writer.add_scalar("Performance/tokens_per_sec", tokens_per_sec, step)
            lr_val = float(1e-4)
            writer.add_scalar("Optimization/learning_rate", lr_val, step)
            steps_hist.append(step)
            loss_hist.append(loss_val)
            grad_hist.append(grad_norm_val)
            lr_hist.append(lr_val)
            tok_hist.append(tokens_per_sec)
            render_training_live(
                steps_hist,
                loss_hist,
                grad_hist,
                lr_hist,
                tok_hist,
                baseline_eval,
                run_dir,
                step,
                live,
            )

            if step % 10 == 0:
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "tok/s": f"{tokens_per_sec:.0f}"})
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nCRITICAL FAILURE during training step {step}: {e}")
        writer.add_text("Errors/fatals", str(e), step)
    finally:
        print(f"Writing post-training sample to {posttrain_sample_path}")
        final_eval = evaluate_state(
            state,
            eval_batch,
            generate_token_fn,
            posttrain_sample_path,
            DEMO_PROMPT,
            seq_len,
        )
        print("Post-training sample saved.")
        writer.add_scalar("Eval/validation_loss", final_eval["validation_loss"], max(step, 0))
        writer.add_scalar("Eval/perplexity", final_eval["perplexity"], max(step, 0))
        writer.add_scalar("Eval/token_accuracy", final_eval["token_accuracy"], max(step, 0))
        writer.add_scalar("Eval/repetition_4gram", final_eval["repetition_4gram"], max(step, 0))
        writer.close()
        if step_times:
            avg_dur = sum(step_times) / len(step_times)
            print(f"\nFinal: {avg_dur*1000:.1f}ms ({ (batch_size*seq_len)/avg_dur:.0f} tok/s)")
        print(
            "Post-training metrics | "
            f"val_loss={final_eval['validation_loss']:.4f} "
            f"ppl={final_eval['perplexity']:.2f} "
            f"acc={final_eval['token_accuracy']:.4f} "
            f"rep4={final_eval['repetition_4gram']:.4f}"
        )
        print(
            "Improvement | "
            f"delta_loss={baseline_eval['validation_loss'] - final_eval['validation_loss']:.4f} "
            f"delta_ppl={baseline_eval['perplexity'] - final_eval['perplexity']:.2f} "
            f"delta_acc={final_eval['token_accuracy'] - baseline_eval['token_accuracy']:.4f} "
            f"delta_rep4={baseline_eval['repetition_4gram'] - final_eval['repetition_4gram']:.4f}"
        )
        render_training_live(
            steps_hist or [0],
            loss_hist or [baseline_eval["validation_loss"]],
            grad_hist or [0.0],
            lr_hist or [0.0],
            tok_hist or [0.0],
            final_eval,
            run_dir,
            max(step, 0) + 1000,
            live,
        )

        try:
            print("Exporting plots...")
            from sparse_attention.viz_training import generate_training_viz
            out_path = f"results/train_{attention_type}_{int(time.time())}"
            generate_training_viz(log_dir, out_path)
            print(f"Results saved to: {out_path}")
        except Exception as e:
            print(f"Viz error: {e}")
            
        return state


def run_inference(state: TrainState, prompt: str, seq_len: int = 1024, max_new_tokens: int = 100):
    print(f"\nEvaluating prompt: '{prompt}'")
    generate_token_fn = jax.jit(
        lambda params, input_ids: jnp.argmax(
            state.apply_fn({"params": params}, input_ids, deterministic=True)[:, -1, :],
            axis=-1,
        )
    )
    
    prompt_tokens = tokenize_text(prompt, max_length=seq_len, pad_to_max=False)
    generated = [int(token) for token in prompt_tokens]
    context = list(generated[-seq_len:]) or [0]
    target_total_tokens = len(prompt_tokens) + max_new_tokens
    
    # We output tokens one by one
    print("Generation:", end=" ", flush=True)
    
    while len(generated) < target_total_tokens:
        window = np.array(context[-seq_len:], dtype=np.int32)
        input_ids = jnp.array(window[None, :], dtype=jnp.int32)
        next_token = int(generate_token_fn(state.params, input_ids)[0])
        generated.append(next_token)
        context.append(next_token)
        
    full_text = decode_tokens(np.array(generated, dtype=np.int32))
    print("\n\n" + full_text + "\n")
    return full_text



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", action="store_true", help="Run dense baseline")
    parser.add_argument("--quick", action="store_true", help="Use small model config")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--logdir", type=str, default="tensorboard_logs")
    parser.add_argument("--no-pallas", action="store_true")
    
    args = parser.parse_args()
    attn_type = "dense" if args.dense else "sparse"
    
    train(
        attention_type=attn_type,
        use_pallas=not args.no_pallas,
        max_steps=args.steps,
        log_dir=args.logdir,
        quick=args.quick,
    )
