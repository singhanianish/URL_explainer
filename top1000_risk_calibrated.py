import pandas as pd

INPUT_CSV = "distinct_1000_urls_simplified.csv"
OUTPUT_CSV = "distinct_1000_urls_risk_calibrated.csv"

df = pd.read_csv(INPUT_CSV)

# =====================================================
# SIGNAL PRIORITY
# =====================================================

HIGH_PRIORITY = [
    "raw ip",
    "nested subdomains",
    "deeply nested",
    "many special characters",
    "url and page content mismatch"
]

MEDIUM_PRIORITY = [
    "uncommon top-level domain",
    "very long url structure",
    "unusually long domain name",
    "missing https"
]

LOW_PRIORITY = [
    "repeated character patterns",
    "many numeric tokens",
    "excessive numeric patterns"
]

# =====================================================
# RISK CALCULATION
# =====================================================

risk_levels = []
top_indicators = []

for _, row in df.iterrows():

    signals = str(row["simplified_signals"]).lower()

    score = 0
    selected = []

    # HIGH
    for s in HIGH_PRIORITY:
        if s in signals:
            score += 3
            selected.append(s)

    # MEDIUM
    for s in MEDIUM_PRIORITY:
        if s in signals:
            score += 2
            selected.append(s)

    # LOW
    for s in LOW_PRIORITY:
        if s in signals:
            score += 1
            selected.append(s)

    # =================================================
    # RISK LEVEL
    # =================================================

    if score >= 7:
        risk = "HIGH"

    elif score >= 4:
        risk = "MEDIUM"

    else:
        risk = "LOW"

    # Keep strongest indicators only
    selected = selected[:4]

    risk_levels.append(risk)

    top_indicators.append(" | ".join(selected))

df["risk_level"] = risk_levels
df["top_indicators"] = top_indicators

df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved to: {OUTPUT_CSV}")