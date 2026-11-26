"""Demonstrate evaluation on independent fictional data only."""
from statistics import mean
from .synthetic import FEATURES, TARGET, generate

def partition(rows):
    train = [r for r in rows if int(r["zone"][1:]) < 30 and r["month"] <= 15]
    test = [r for r in rows if int(r["zone"][1:]) >= 30 and r["month"] >= 17]
    if not train or not test:
        raise ValueError("Empty train or test")
    if {r["zone"] for r in train} & {r["zone"] for r in test}:
        raise ValueError("Geographical overlap")
    if max(r["month"] + 1 for r in train) >= min(r["month"] for r in test):
        raise ValueError("Training outcome extends into evaluation")
    return train, test

def validate_features(rows, feature_names=FEATURES):
    # An explicit allowlist is safer than silently dropping only known bad labels.
    names = tuple(feature_names)
    if len(names) != len(FEATURES) or set(names) != set(FEATURES):
        raise ValueError("Only fictional, pre-outcome features are allowed")
    labels = [r[TARGET] for r in rows]
    for name in names:
        values = [r[name] for r in rows]
        if all(v == t for v, t in zip(values, labels)):
            raise ValueError(f"Target duplicate: {name}")
        if not all(0 <= v <= 1 for v in values):
            raise ValueError(f"Out of range: {name}")

def design(rows):
    validate_features(rows)
    return [[r[f] for f in FEATURES] for r in rows]

def weighted_signal(rows):
    # Arbitrary illustrative weights; not copied from the original team.
    validate_features(rows)
    return [[0.30*r["price_pressure"] + 0.35*r["collateral_pressure"]
             + 0.20*(1-r["insurance_access"]) + 0.15*r["housing_proxy"]] for r in rows]

def main():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt

    train, test = partition(generate())
    X, Xtest = design(train), design(test)
    y, ytest = [r[TARGET] for r in train], [r[TARGET] for r in test]
    print("SYNTHETIC ONLY — not real-world or original-team performance")
    print(f"Train {len(train)} rows; test {len(test)} rows; spatial and temporal holdout")
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8}")

    def report(name, prediction):
        print(f"{name:<25} {mean_absolute_error(ytest, prediction):8.4f} "
              f"{sqrt(mean_squared_error(ytest, prediction)):8.4f} "
              f"{r2_score(ytest, prediction):8.4f}")

    report("Train mean", [mean(y)] * len(ytest))
    weighted_model = LinearRegression().fit(weighted_signal(train), y)
    report("Weighted demo", weighted_model.predict(weighted_signal(test)))
    for name, model in [
        ("Linear regression", make_pipeline(StandardScaler(), LinearRegression())),
        ("Random forest", RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)),
        ("Gradient boosting", GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=42)),
    ]:
        model.fit(X, y)
        report(name, model.predict(Xtest))

if __name__ == "__main__":
    main()

