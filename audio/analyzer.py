"""Light analysis of transcribed text: rules/heuristics (anxiety, hesitation,
discomfort, etc.)"""

TERMS_ANXIETY = [
    "anxious",
    "anxiety",
    "nervous",
    "worried",
    "fear",
    "scared",
    "stress",
    "stressed",
    "tension",
    "restless",
]

TERMS_HESITATION = [
    "um",
    "uh",
    "eh",
    "so",
    "like",
    "well",
    "anyway",
    "basically",
    "actually",
]

TERMS_DISCOMFORT = [
    "discomfort",
    "pain",
    "uncomfortable",
    "difficult",
    "hard",
    "sad",
    "tired",
    "fatigue",
]

TERMS_POSTPARTUM_DEPRESSION = [
    "postpartum",
    "depression",
    "depressed",
    "hopeless",
    "crying",
    "overwhelmed",
    "baby",
    "blues",
    "mood",
    "swings",
    "bonding",
]

TERMS_DOMESTIC_VIOLENCE = [
    "abuse",
    "abused",
    "hit",
    "hurt",
    "threatened",
    "controlling",
    "afraid",
    "violence",
    "violent",
]

TERMS_HORMONAL_FATIGUE = [
    "exhausted",
    "hormonal",
    "mood",
    "irritable",
    "foggy",
    "drained",
    "fogginess",
]

TERM_LABELS = {
    "anxiety": TERMS_ANXIETY,
    "hesitation": TERMS_HESITATION,
    "discomfort": TERMS_DISCOMFORT,
    "postpartum_depression": TERMS_POSTPARTUM_DEPRESSION,
    "domestic_violence": TERMS_DOMESTIC_VIOLENCE,
    "hormonal_fatigue": TERMS_HORMONAL_FATIGUE,
}

SUMMARY_MESSAGES = {
    "anxiety": "Possible anxiety indicators ({} mentions).",
    "hesitation": "Hesitation or fillers ({} occurrences).",
    "discomfort": "Discomfort indicators ({} mentions).",
    "postpartum_depression": "Postpartum depression indicators ({} mentions).",
    "domestic_violence": "Domestic violence indicators ({} mentions).",
    "hormonal_fatigue": "Hormonal fatigue indicators ({} mentions).",
}


def analyze_transcript_text(text: str) -> dict:
    """Analyze transcript with term-list rules.
    Return dict with indicators and summary."""
    if not text or not text.strip():
        return {
            "anxiety_indicators": 0,
            "hesitation_indicators": 0,
            "discomfort_indicators": 0,
            "postpartum_depression_indicators": 0,
            "domestic_violence_indicators": 0,
            "hormonal_fatigue_indicators": 0,
            "matched_terms": [],
            "summary": "Empty text or no speech detected.",
        }
    lower = text.lower().strip()
    words = lower.split()
    matched = []
    counts = {label: 0 for label in TERM_LABELS}
    for w in words:
        w_clean = "".join(c for c in w if c.isalpha())
        if not w_clean:
            continue
        for label, terms in TERM_LABELS.items():
            if w_clean in terms:
                counts[label] += 1
                matched.append((label, w_clean))
    summary_parts = [
        SUMMARY_MESSAGES[label].format(counts[label])
        for label in TERM_LABELS
        if counts[label] > 0
    ]
    summary = (
        " ".join(summary_parts)
        if summary_parts
        else "No strong indicators detected in term lists."
    )
    return {
        "anxiety_indicators": counts["anxiety"],
        "hesitation_indicators": counts["hesitation"],
        "discomfort_indicators": counts["discomfort"],
        "postpartum_depression_indicators": counts["postpartum_depression"],
        "domestic_violence_indicators": counts["domestic_violence"],
        "hormonal_fatigue_indicators": counts["hormonal_fatigue"],
        "matched_terms": matched,
        "summary": summary,
    }
