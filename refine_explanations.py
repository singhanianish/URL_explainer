import pandas as pd
import requests
from tqdm import tqdm
import random
import re

# =====================================================
# CONFIG
# =====================================================

INPUT_CSV = "distinct_1000_urls_simplified.csv"
OUTPUT_CSV = "refined_explanations_balanced_v1.csv"

MODEL_NAME = "gemma3:4b"

# =====================================================
# LOAD DATASET
# =====================================================

print("[+] Loading CSV...")

df = pd.read_csv(INPUT_CSV)

print(f"[+] Loaded {len(df)} rows")

# =====================================================
# OPTIONAL TEST MODE
# =====================================================

# Uncomment for testing
# df = df.head(100)

# =====================================================
# TOP BRAND TOKENS
# (small representative list; can expand later)
# =====================================================

BRAND_TOKENS = {
    "google", "microsoft", "apple", "amazon", "meta",
    "paypal", "netflix", "adobe", "oracle", "intel",
    "cisco", "samsung", "lenovo", "dell", "hp",
    "hsbc", "icici", "hdfc", "axisbank", "sbi",
    "github", "linkedin", "zoom", "dropbox", "slack",
    "spotify", "airbnb", "uber", "ola", "flipkart",
    "loginradius", "authorea", "cloudflare", "akamai",
    "salesforce", "sap", "ibm", "vmware", "mongodb",
    "notion", "atlassian", "stripe", "razorpay",
    "phonepe", "paytm", "telegram", "whatsapp",
    "instagram", "facebook", "twitter", "x", "reddit"
}

# =====================================================
# EXPLANATION ARCHETYPES
# =====================================================

EXPLANATION_STYLES = [
    "STRUCTURAL",
    "REPUTATION",
    "BEHAVIORAL",
    "CAUTIOUS"
]

# =====================================================
# POSITIVE SIGNAL EXTRACTION
# =====================================================

def extract_positive_signals(url):

    url_lower = str(url).lower()

    positive = []

    # -------------------------------------------------
    # BRAND-LIKE TOKENS
    # -------------------------------------------------

    for brand in BRAND_TOKENS:

        if brand in url_lower:
            positive.append(
                f"contains recognizable brand-style naming"
            )
            break

    # -------------------------------------------------
    # HTTPS
    # -------------------------------------------------

    if url_lower.startswith("https://"):
        positive.append(
            "uses secure HTTPS protocol"
        )

    # -------------------------------------------------
    # READABLE WORD STRUCTURE
    # -------------------------------------------------

    tokens = re.findall(r"[a-zA-Z]{4,}", url_lower)

    readable_ratio = 0

    if len(tokens) > 0:

        readable_tokens = [
            t for t in tokens
            if len(set(t)) > 2
        ]

        readable_ratio = len(readable_tokens) / len(tokens)

    if readable_ratio >= 0.7:

        positive.append(
            "contains readable domain structure"
        )

    # -------------------------------------------------
    # ENTERPRISE-STYLE SUBDOMAINS
    # -------------------------------------------------

    enterprise_patterns = [
        ".about.",
        ".support.",
        ".docs.",
        ".help.",
        ".api.",
        ".mail."
    ]

    if any(p in url_lower for p in enterprise_patterns):

        positive.append(
            "uses enterprise-style subdomain organization"
        )

    # -------------------------------------------------
    # COMMON TRUSTED TLDS
    # -------------------------------------------------

    trusted_tlds = [
        ".com",
        ".org",
        ".edu",
        ".gov",
        ".net"
    ]

    if any(url_lower.endswith(tld) for tld in trusted_tlds):

        positive.append(
            "uses common domain extension"
        )

    return positive

# =====================================================
# NEGATIVE SIGNAL FILTERING
# =====================================================

def process_negative_signals(signal_text):

    if pd.isna(signal_text):
        return [], 0

    signals = [
        s.strip().lower()
        for s in str(signal_text).split("|")
    ]

    filtered = []

    risk_score = 0

    for s in signals:

        # =============================================
        # REMOVE COMPLETELY
        # =============================================

        if "repeated character patterns" in s:
            continue

        # =============================================
        # HIGH RISK
        # =============================================

        if any(x in s for x in [
            "raw ip",
            "nested subdomains",
            "deeply nested",
            "url and page content mismatch"
        ]):

            filtered.append(s)
            risk_score += 3
            continue

        # =============================================
        # MEDIUM RISK
        # =============================================

        if any(x in s for x in [
            "many special characters",
            "missing https",
            "very long url structure"
        ]):

            filtered.append(s)
            risk_score += 2
            continue

        # =============================================
        # LOW RISK
        # =============================================

        if any(x in s for x in [
            "uncommon top-level domain",
            "many numeric tokens",
            "excessive numeric patterns",
            "unusually long domain name"
        ]):

            filtered.append(s)
            risk_score += 1
            continue

    return filtered[:4], risk_score

# =====================================================
# RISK CALIBRATION
# =====================================================

def determine_risk_level(risk_score, positive_signals):

    # Positive evidence reduces risk
    adjusted_score = risk_score - min(2, len(positive_signals))

    if adjusted_score >= 6:
        return "HIGH"

    elif adjusted_score >= 3:
        return "MEDIUM"

    else:
        return "LOW"

# =====================================================
# PROMPT
# =====================================================

def build_prompt(
    url,
    risk_level,
    negative_indicators,
    positive_signals,
    style
):

    return f"""
Explain why this URL may appear suspicious.

Risk Level:
{risk_level}

Observed Suspicious Indicators:
{negative_indicators}

Observed Legitimacy Indicators:
{positive_signals}

Explanation Style:
{style}

Write a concise cybersecurity explanation that:
- explains the overall URL behavior naturally
- balances suspicious and legitimate characteristics
- avoids exaggerated claims
- avoids repetitive phrasing
- sounds analytical and human-written
- stays grounded in the indicators
- explains WHY the URL may or may not appear suspicious

Rules:
- Do not invent facts.
- Do not assume phishing unless strongly supported.
- Low risk explanations should sound cautious.
- Mention positive legitimacy indicators when relevant.
- Avoid mechanically listing indicators.
- Keep explanation between 45 and 85 words.

Return ONLY the explanation.

URL:
{url}
"""

# =====================================================
# OLLAMA
# =====================================================

def refine_text(prompt):

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.45,
                "top_p": 0.92,
                "repeat_penalty": 1.15,
                "presence_penalty": 0.35,
                "num_predict": 90
            }
        },
        timeout=120
    )

    data = response.json()

    if "response" not in data:
        return ""

    result = data["response"].strip()

    result = result.replace("\n", " ").strip()

    result = result.strip('"')

    return result

# =====================================================
# PROCESS
# =====================================================

print("[+] Starting refinement...")

outputs = []
risk_levels = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    url = row["url"]

    raw_signals = row["simplified_signals"]

    # -------------------------------------------------
    # NEGATIVE SIGNALS
    # -------------------------------------------------

    negative_indicators, risk_score = process_negative_signals(
        raw_signals
    )

    # -------------------------------------------------
    # POSITIVE SIGNALS
    # -------------------------------------------------

    positive_signals = extract_positive_signals(url)

    # -------------------------------------------------
    # FINAL RISK
    # -------------------------------------------------

    risk_level = determine_risk_level(
        risk_score,
        positive_signals
    )

    risk_levels.append(risk_level)

    # -------------------------------------------------
    # STYLE RANDOMIZATION
    # -------------------------------------------------

    style = random.choice(EXPLANATION_STYLES)

    prompt = build_prompt(
        url,
        risk_level,
        negative_indicators,
        positive_signals,
        style
    )

    try:

        refined = refine_text(prompt)

    except Exception as e:

        print("ERROR:", e)

        refined = ""

    outputs.append(refined)

# =====================================================
# SAVE
# =====================================================

df["risk_level"] = risk_levels

df["refined_explanation"] = outputs

df.to_csv(OUTPUT_CSV, index=False)

print(f"[+] Saved to: {OUTPUT_CSV}")