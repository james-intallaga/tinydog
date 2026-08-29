"""Train MLM Arithmetic on an infinite verified curriculum."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .data import ArithmeticCurriculum, curriculum_level, make_batch
from .model import create_model
from .tokenizer import ArithmeticTokenizer


def loss_fn(model, inputs: mx.array, targets: mx.array) -> mx.array:
    logits = model(inputs)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def learning_rate(step: int, total_steps: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def evaluate_loss(
    model, tokenizer, seed: int, batches: int, batch_size: int, context: int
) -> float:
    curriculum = ArithmeticCurriculum(seed=seed, split="validation")
    losses = []
    model.eval()
    for _ in range(batches):
        packed = mx.array(make_batch(curriculum, tokenizer, batch_size, context, 6))
        losses.append(loss_fn(model, packed[:, :-1], packed[:, 1:]))
    mx.eval(losses)
    model.train()
    return sum(float(value.item()) for value in losses) / len(losses)


def save_checkpoint(
    model, checkpoint_dir: Path, step: int, args: argparse.Namespace
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights = checkpoint_dir / f"step-{step:07d}.safetensors"
    model.save_weights(str(weights))
    (checkpoint_dir / "latest.json").write_text(
        json.dumps(
            {"step": step, "weights": weights.name, "args": vars(args)}, indent=2
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/100m.json")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--resume", help="Path to model weights; optimizer state is restarted"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mx.random.seed(args.seed)
    tokenizer = ArithmeticTokenizer()
    model = create_model(args.config, args.resume)
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Config vocab is {model.config.vocab_size}, tokenizer vocab is {tokenizer.vocab_size}"
        )
    print(
        f"Model parameters: {model.parameter_count():,} ({model.parameter_count()/1e6:.2f}M)"
    )

    optimizer = optim.AdamW(
        learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    curriculum = ArithmeticCurriculum(seed=args.seed, split="train")
    context = model.config.context_length
    tokens_seen = 0
    interval_tokens = 0
    interval_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        level = curriculum_level(step - 1, args.steps)
        packed = mx.array(
            make_batch(curriculum, tokenizer, args.batch_size, context, level)
        )
        inputs, targets = packed[:, :-1], packed[:, 1:]
        loss, grads = loss_and_grad(model, inputs, targets)
        grads, grad_norm = optim.clip_grad_norm(grads, args.grad_clip)
        lr = learning_rate(step - 1, args.steps, args.learning_rate, args.warmup_steps)
        optimizer.learning_rate = mx.array(lr)
        optimizer.update(model, grads)
        mx.eval(loss, grad_norm, model.parameters(), optimizer.state)

        batch_tokens = args.batch_size * context
        tokens_seen += batch_tokens
        interval_tokens += batch_tokens
        if step == 1 or step % args.report_every == 0:
            elapsed = max(time.perf_counter() - interval_start, 1e-9)
            print(
                f"step={step} level={level} loss={float(loss.item()):.4f} grad={float(grad_norm.item()):.3f} lr={lr:.2e} tokens={tokens_seen:,} tok/s={interval_tokens/elapsed:,.0f}"
            )
            interval_start, interval_tokens = time.perf_counter(), 0
        if step % args.eval_every == 0:
            val_loss = evaluate_loss(
                model,
                tokenizer,
                args.seed + 10_000,
                args.eval_batches,
                args.batch_size,
                context,
            )
            print(
                f"validation step={step} loss={val_loss:.4f} perplexity={math.exp(val_loss):.3f}"
            )
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(model, Path(args.checkpoint_dir), step, args)


if __name__ == "__main__":
    main()
