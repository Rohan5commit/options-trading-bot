"""
Curate a high-quality 20K subset from the 160K dataset.
Strategy: regime-balanced + action-diverse sampling.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def extract_regime(instruction: str) -> str:
    """Extract market regime from instruction text."""
    for regime in [
        "strong_uptrend", "weak_uptrend", "strong_downtrend", "weak_downtrend",
        "ranging_low_vol", "ranging_high_vol", "pre_earnings", "post_earnings_crush",
        "squeeze_setup", "breakout", "breakdown", "mean_reversion",
    ]:
        if regime in instruction.lower().replace(" ", "_"):
            return regime
    return "unknown"


def extract_action(output: str) -> str:
    """Extract action from output JSON."""
    try:
        data = json.loads(output)
        return data.get("action", "unknown")
    except (json.JSONDecodeError, TypeError):
        return "unknown"


def extract_strategy(output: str) -> str:
    """Extract strategy from output JSON."""
    try:
        data = json.loads(output)
        return data.get("strategy", "unknown")
    except (json.JSONDecodeError, TypeError):
        return "unknown"


def curate(input_path: str, output_dir: str, target_total: int = 20000, seed: int = 42):
    """Curate dataset with regime balancing and action diversity."""
    rng = random.Random(seed)

    # Load all examples
    examples = []
    with open(input_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            examples.append(entry)

    print(f"Loaded {len(examples)} examples")

    # Group by regime
    by_regime = defaultdict(list)
    for ex in examples:
        regime = extract_regime(ex.get("instruction", ""))
        by_regime[regime].append(ex)

    print(f"Regimes found: {list(by_regime.keys())}")
    for r, exs in sorted(by_regime.items()):
        print(f"  {r}: {len(exs)}")

    # Target per regime (balanced)
    regimes = list(by_regime.keys())
    target_per_regime = target_total // len(regimes)
    extra = target_total - target_per_regime * len(regimes)

    curated = []
    for i, regime in enumerate(regimes):
        pool = by_regime[regime]
        count = target_per_regime + (1 if i < extra else 0)
        count = min(count, len(pool))

        # Within each regime, balance by strategy
        by_strategy = defaultdict(list)
        for ex in pool:
            strat = extract_strategy(ex.get("output", ""))
            by_strategy[strat].append(ex)

        # Round-robin across strategies
        selected = []
        strats = list(by_strategy.keys())
        strat_idx = {s: 0 for s in strats}
        while len(selected) < count:
            added = False
            for strat in strats:
                if len(selected) >= count:
                    break
                if strat_idx[strat] < len(by_strategy[strat]):
                    selected.append(by_strategy[strat][strat_idx[strat]])
                    strat_idx[strat] += 1
                    added = True
            if not added:
                break

        rng.shuffle(selected)
        curated.extend(selected[:count])
        print(f"  Curated {len(selected[:count])} from {regime} (had {len(pool)} candidates)")

    rng.shuffle(curated)

    # Split train/test
    split_idx = int(len(curated) * 0.95)
    train = curated[:split_idx]
    test = curated[split_idx:]

    # Write
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train_curated.jsonl"
    test_path = out / "test_curated.jsonl"

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(test_path, "w") as f:
        for ex in test:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(train)} train, {len(test)} test to {out}")

    # Stats
    action_counts = defaultdict(int)
    strategy_counts = defaultdict(int)
    for ex in curated:
        action_counts[extract_action(ex.get("output", ""))] += 1
        strategy_counts[extract_strategy(ex.get("output", ""))] += 1

    print("\nAction distribution:")
    for a, c in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {a}: {c} ({c/len(curated)*100:.1f}%)")

    print("\nStrategy distribution:")
    for s, c in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c} ({c/len(curated)*100:.1f}%)")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "./training_data/train.jsonl"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./training_data"
    curate(input_file, output_dir)
