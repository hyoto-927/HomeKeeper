"""New fictional contract workflow; no source transactions or original model.

Illustrates distribution mapping, assumed label generation, an imbalanced
independent synthetic test, and aggregation. Never use as a decision tool.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from math import isfinite
from random import Random


def quantile_map(values, source_reference, target_reference):
    """Empirical mid-rank then linear interpolation on the target reference.

    References must be chosen without access to evaluation outcomes.
    """
    source, target = sorted(source_reference), sorted(target_reference)
    if not source or not target or not all(isfinite(v) for v in source + target):
        raise ValueError("Nonempty finite reference arrays required")
    output = []
    for v in values:
        if not isfinite(v):
            raise ValueError("Non-finite input")
        p = (bisect_left(source, v) + bisect_right(source, v)) / (2 * len(source))
        index = p * (len(target) - 1)
        lo = int(index)
        hi = min(lo + 1, len(target) - 1)
        output.append(target[lo] + (index - lo) * (target[hi] - target[lo]))
    return output


def simulate(seed, count, prefix, shifted=False):
    """Invented correlated features and stochastic labels, not observations."""
    rng = Random(seed)
    rows = []
    for i in range(count):
        latent = rng.random()
        price_ratio = min(1, max(0, .3 + .6 * latent + rng.gauss(0, .12)))
        collateral = min(1, max(0, .15 + .7 * latent + rng.gauss(0, .15)))
        # Arbitrary hypothetical rule, not original coefficients or actual risks.
        if shifted:
            probability = .008 + .025 * (1 - latent)
        else:
            probability = .015 + .19 * (price_ratio * collateral)
        rows.append({"contract": f"{prefix}-{i}", "zone": f"{prefix}-{i % 24:02d}",
                     "price_ratio": price_ratio, "collateral": collateral,
                     "fictional_label": int(rng.random() < probability)})
    return rows


def aggregate(rows):
    totals = defaultdict(lambda: [0, 0])
    for row in rows:
        cell = totals[row["zone"]]
        if row["fictional_label"] not in (0, 1):
            raise ValueError("Binary fictional label required")
        cell[0] += 1
        cell[1] += row["fictional_label"]
    return {zone: {"contracts": n, "fictional_positive": p, "fictional_rate": p / n}
            for zone, (n, p) in totals.items()}


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 confusion_matrix, precision_score, recall_score)

    train = simulate(12, 6000, "TRAIN")
    test = simulate(57, 3000, "TEST", shifted=True)
    scoring = simulate(81, 1200, "SCORE")
    x = lambda rows: [[r["price_ratio"], r["collateral"]] for r in rows]
    y = lambda rows: [r["fictional_label"] for r in rows]
    print("FICTIONAL ONLY - not original B_train, B_test, A_realistic")
    print(f"Train positives: {sum(y(train))}/{len(train)}; test positives: {sum(y(test))}/{len(test)}")
    for label, class_weight in [("default", None), ("balanced", "balanced")]:
        model = LogisticRegression(class_weight=class_weight, random_state=42).fit(x(train), y(train))
        pred = model.predict(x(test))
        proba = model.predict_proba(x(test))[:, 1]
        print(f"{label}: accuracy={accuracy_score(y(test), pred):.4f}, "
              f"precision={precision_score(y(test), pred, zero_division=0):.4f}, "
              f"recall={recall_score(y(test), pred, zero_division=0):.4f}, "
              f"AP={average_precision_score(y(test), proba):.4f}, "
              f"TN FP FN TP={confusion_matrix(y(test), pred, labels=[0, 1]).ravel().tolist()}")
    print("Illustrative quantile mapping:", quantile_map([2, 5, 9], [0, 2, 5, 9, 10], [10, 20, 30, 40, 50]))
    print(f"Fictional scoring population: {len(scoring)} contracts, {len(aggregate(scoring))} zones")


if __name__ == "__main__":
    main()
