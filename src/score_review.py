"""2026 portfolio reconstruction of the visible six-term sum; fictional data only."""
from math import isfinite
from random import Random

WEIGHTS = {
    "high_collateral_share": 0.15,
    "mean_deposit_price_ratio": 0.15,
    "insurance_rejection_share": 0.12,
    "new_villa_share": 0.10,
    "auction_count_scaled": 0.08,
    "failed_auction_count_scaled": 0.05,
}


def final_v4(normalized):
    """Six visible weights, without silently renormalizing their sum of 0.65."""
    values = [float(normalized[key]) for key in WEIGHTS]
    if not all(isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("Expected normalized finite values within [0, 1]")
    return sum(v * w for v, w in zip(values, WEIGHTS.values()))


def top_k_overlap(score_sets, k=10):
    """Count shared IDs in top-k sets over the same universe; ties by ID."""
    if not score_sets or not isinstance(k, int) or k < 1:
        raise ValueError("Nonempty score sets and positive integer k required")
    universe = set(next(iter(score_sets.values())))
    if len(universe) < k:
        raise ValueError("Insufficient IDs for k")
    top = {}
    for name, scores in score_sets.items():
        if set(scores) != universe or not all(isfinite(v) for v in scores.values()):
            raise ValueError("All score sets must cover the same IDs with finite scores")
        top[name] = set(sorted(scores, key=lambda x: (-scores[x], x))[:k])
    return {a: {b: len(top[a] & top[b]) for b in top} for a in top}


def main():
    rng = Random(17)
    rows = {f"fictional-{i:02d}": {key: rng.random() for key in WEIGHTS}
            for i in range(30)}
    scores = {
        "six_term_sum": {i: final_v4(row) for i, row in rows.items()},
        "equal_weight": {i: sum(row.values()) / len(row) for i, row in rows.items()},
        "single_signal": {i: row["high_collateral_share"] for i, row in rows.items()},
    }
    matrix = top_k_overlap(scores)
    print("FICTIONAL IDS ONLY — overlap is not accuracy")
    print(f"Visible weight sum: {sum(WEIGHTS.values()):.2f}")
    print("method," + ",".join(matrix))
    for name, counts in matrix.items():
        print(name + "," + ",".join(str(x) for x in counts.values()))


if __name__ == "__main__":
    main()
