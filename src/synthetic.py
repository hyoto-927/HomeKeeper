"""Generate fictional data without reading any external dataset."""
import random

FEATURES = ("price_pressure", "collateral_pressure", "insurance_access", "housing_proxy")
TARGET = "future_rate"

def generate(seed=42):
    rng = random.Random(seed)
    rows = []
    for zone in range(40):
        region_effect = rng.uniform(-0.035, 0.035)
        for month in range(1, 25):
            price, collateral, insurance, housing = (rng.random() for _ in FEATURES)
            # The future label has independent noise; no actual risk data are used.
            signal = (0.05 + 0.15 * price + 0.17 * collateral
                      + 0.09 * (1 - insurance) + 0.08 * price * housing
                      + region_effect + 0.002 * month)
            future = min(1.0, max(0.0, signal + rng.gauss(0, 0.05)))
            rows.append(dict(zone=f"Z{zone:02}", month=month,
                             price_pressure=price, collateral_pressure=collateral,
                             insurance_access=insurance, housing_proxy=housing,
                             future_rate=future))
    return rows

if __name__ == "__main__":
    rows = generate()
    print(f"Synthetic rows: {len(rows)}")
    print("Columns:", ", ".join(rows[0]))

