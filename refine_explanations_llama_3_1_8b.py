import pandas as pd
import requests
from tqdm import tqdm
import random
import re
from collections import Counter

# =====================================================
# CONFIG
# =====================================================

INPUT_CSV = "distinct_1000_urls_simplified.csv"
OUTPUT_CSV = "llama31_8b_final_1000.csv"

MODEL_NAME = "llama3.1:8b"

# =====================================================
# LOAD DATASET
# =====================================================

print("[+] Loading CSV...")

df = pd.read_csv(INPUT_CSV)

print(f"[+] Loaded {len(df)} rows")

# =====================================================
# OPTIONAL TEST MODE
# =====================================================

# Uncomment for quick testing
# df = df.head(20)

# =====================================================
# BRAND TOKENS
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
    "instagram", "facebook", "twitter", "reddit",
    "securepay", "fvcbank", "iseepassword"
}

# =====================================================
# LOW-RISK GENERIC KEYWORDS
# =====================================================

LOW_RISK_KEYWORDS = {
    "login",
    "secure",
    "auth",
    "password",
    "bank",
    "confirm",
    "paypal"
}

# =====================================================
# HIGH-RISK KEYWORD COMBINATIONS
# =====================================================

HIGH_RISK_PATTERNS = [
    "login-secure",
    "verify-account",
    "auth-update",
    "confirm-payment",
    "wallet-verify",
    "secure-login",
    "account-verification",
    "signin-update",
    "payment-confirm"
]

# =====================================================
# EXPLANATION STYLES
# =====================================================

EXPLANATION_STYLES = [
    "STRUCTURE_FOCUSED",
    "BALANCED_ANALYSIS",
    "BEHAVIOR_FOCUSED",
    "OBSERVATIONAL",
    "REPUTATION_FOCUSED",
    "LEGITIMACY_BALANCED"
]

# =====================================================
# DOMAIN READABILITY SCORE
# =====================================================

def calculate_readability_score(url):

    url_lower = str(url).lower()

    score = 0

    # -------------------------------------------------
    # READABLE TOKENS
    # -------------------------------------------------

    tokens = re.findall(r"[a-zA-Z]{4,}", url_lower)

    readable_tokens = [
        t for t in tokens
        if len(set(t)) > 2
    ]

    if len(tokens) > 0:

        readability_ratio = (
            len(readable_tokens) / len(tokens)
        )

        if readability_ratio >= 0.8:
            score += 2

        elif readability_ratio >= 0.6:
            score += 1

    # -------------------------------------------------
    # BRAND-LIKE STRUCTURE
    # -------------------------------------------------

    for brand in BRAND_TOKENS:

        if brand in url_lower:
            score += 2
            break

    # -------------------------------------------------
    # EXCESSIVE SYMBOLS
    # -------------------------------------------------

    special_count = len(
        re.findall(r"[-_=+@]", url_lower)
    )

    if special_count >= 4:
        score -= 2

    elif special_count >= 2:
        score -= 1

    return score

# =====================================================
# POSITIVE SIGNALS
# =====================================================

def extract_positive_signals(url):

    url_lower = str(url).lower()

    positive = []

    # -------------------------------------------------
    # HTTPS
    # -------------------------------------------------

    if url_lower.startswith("https://"):

        positive.append(
            "secure HTTPS usage"
        )

    # -------------------------------------------------
    # BRAND-LIKE TOKENS
    # -------------------------------------------------

    for brand in BRAND_TOKENS:

        if brand in url_lower:

            positive.append(
                "recognizable brand-style naming"
            )

            break

    # -------------------------------------------------
    # READABLE STRUCTURE
    # -------------------------------------------------

    readability = calculate_readability_score(url)

    if readability >= 2:

        positive.append(
            "readable and structured domain naming"
        )

    # -------------------------------------------------
    # ENTERPRISE STRUCTURE
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
            "enterprise-style subdomain organization"
        )

    # -------------------------------------------------
    # TRUSTED TLDS
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
            "common domain extension"
        )

    return positive

# =====================================================
# NEGATIVE SIGNALS
# =====================================================

def process_negative_signals(url, signal_text):

    if pd.isna(signal_text):
        return [], 0

    url_lower = str(url).lower()

    signals = [
        s.strip().lower()
        for s in str(signal_text).split("|")
    ]

    filtered = []

    risk_score = 0

    # -------------------------------------------------
    # HIGH-RISK COMBINATIONS
    # -------------------------------------------------

    for pattern in HIGH_RISK_PATTERNS:

        if pattern in url_lower:

            filtered.append(
                "credential-themed impersonation pattern"
            )

            risk_score += 4

    # -------------------------------------------------
    # SIGNAL PROCESSING
    # -------------------------------------------------

    for s in signals:

        # REMOVE COMPLETELY
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

    # -------------------------------------------------
    # REDUCE GENERIC KEYWORD OVERREACTION
    # -------------------------------------------------

    for keyword in LOW_RISK_KEYWORDS:

        if keyword in url_lower:

            risk_score -= 1

    # -------------------------------------------------
    # REMOVE DUPLICATES
    # -------------------------------------------------

    filtered = list(dict.fromkeys(filtered))

    return filtered[:3], max(0, risk_score)

# =====================================================
# CONFIDENCE LEVEL
# =====================================================

def determine_confidence_level(
    risk_score,
    positive_signals,
    readability_score
):

    adjusted_score = (
        risk_score
        - min(2, len(positive_signals))
        - max(0, readability_score - 1)
    )

    if adjusted_score >= 6:
        return "STRONGLY SUSPICIOUS"

    elif adjusted_score >= 3:
        return "MODERATELY SUSPICIOUS"

    else:
        return "OBSERVATIONAL"

# =====================================================
# PROMPT
# Optimized for Llama 3.1 8B
# =====================================================

def build_prompt(
    url,
    confidence_level,
    negative_indicators,
    positive_signals,
    style
):

    return f"""
Explain why this URL may appear suspicious.

Confidence:
{confidence_level}

Suspicious Indicators:
{negative_indicators}

Legitimacy Indicators:
{positive_signals}

Explanation Style:
{style}

Write a concise human-written cybersecurity explanation.

Requirements:
- stay grounded in the indicators
- balance suspicious and legitimate characteristics naturally
- explain why the URL may OR may not appear suspicious
- avoid exaggerated phishing narratives
- avoid generic warning phrases
- avoid repetitive wording patterns
- avoid repeatedly using phrases related to:
  - obscuring destination
  - difficult to interpret
  - harder to interpret
- use realistic analyst-style reasoning
- observational explanations should sound cautious
- stronger language should only appear when clearly justified
- legitimacy indicators should be incorporated naturally
- keep explanation between 45 and 80 words

URL:
{url}
"""

# =====================================================
# OLLAMA API
# =====================================================

def refine_text(prompt):

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {

                # =====================================
                # LLAMA 3.1 8B OPTIMIZED SETTINGS
                # =====================================

                "temperature": 0.38,
                "top_p": 0.88,
                "repeat_penalty": 1.12,
                "presence_penalty": 0.28,
                "num_predict": 78
            }
        },
        timeout=180
    )

    data = response.json()

    if "response" not in data:
        return ""

    result = data["response"].strip()

    # -------------------------------------------------
    # CLEANUP
    # -------------------------------------------------

    result = result.replace("\n", " ").strip()

    result = result.strip('"')

    unwanted_prefixes = [
        "Explanation:",
        "Refined Explanation:",
        "The explanation is:"
    ]

    for prefix in unwanted_prefixes:

        if result.startswith(prefix):
            result = result[len(prefix):].strip()

    return result

# =====================================================
# PROCESS
# =====================================================

print("[+] Starting refinement using Llama 3.1 8B...")

outputs = []
confidence_levels = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    url = row["url"]

    raw_signals = row["simplified_signals"]

    # -------------------------------------------------
    # NEGATIVE SIGNALS
    # -------------------------------------------------

    negative_indicators, risk_score = process_negative_signals(
        url,
        raw_signals
    )

    # -------------------------------------------------
    # POSITIVE SIGNALS
    # -------------------------------------------------

    positive_signals = extract_positive_signals(url)

    # -------------------------------------------------
    # READABILITY
    # -------------------------------------------------

    readability_score = calculate_readability_score(
        url
    )

    # -------------------------------------------------
    # CONFIDENCE
    # -------------------------------------------------

    confidence_level = determine_confidence_level(
        risk_score,
        positive_signals,
        readability_score
    )

    confidence_levels.append(
        confidence_level
    )

    # -------------------------------------------------
    # STYLE
    # -------------------------------------------------

    style = random.choice(
        EXPLANATION_STYLES
    )

    # -------------------------------------------------
    # PROMPT
    # -------------------------------------------------

    prompt = build_prompt(
        url,
        confidence_level,
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
# SAVE OUTPUT
# =====================================================

print("[+] Saving output...")

df["confidence_level"] = confidence_levels

df["refined_explanation"] = outputs

df.to_csv(OUTPUT_CSV, index=False)

print("[+] Done.")
print(f"[+] Saved to: {OUTPUT_CSV}")