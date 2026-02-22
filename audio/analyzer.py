"""Análise leve do texto transcrito: regras/heurísticas (ansiedade, hesitação)."""

# Listas de termos indicativos (contexto saúde da mulher / consultas)
TERMS_ANXIETY = [
    "ansiosa",
    "ansiedade",
    "nervosa",
    "preocupada",
    "medo",
    "assustada",
    "estresse",
    "estressada",
    "tensão",
    "inquieta",
]

TERMS_HESITATION = [
    "hum",
    "ah",
    "eh",
    "então",
    "tipo",
    "assim",
    "né",
    "bom",
    "olha",
    "sei lá",
]

TERMS_DISCOMFORT = [
    "desconforto",
    "dor",
    "incômodo",
    "difícil",
    "complicado",
    "triste",
    "cansada",
    "fadiga",
]

TERM_LABELS = {
    "anxiety": TERMS_ANXIETY,
    "hesitation": TERMS_HESITATION,
    "discomfort": TERMS_DISCOMFORT,
}

SUMMARY_MESSAGES = {
    "anxiety": "Possíveis indicadores de ansiedade ({} menções).",
    "hesitation": "Hesitação ou preenchimentos ({} ocorrências).",
    "discomfort": "Indicadores de desconforto ({} menções).",
}


def analyze_transcript_text(text: str) -> dict:
    """
    Analisa o texto transcrito com regras simples (listas de termos).
    Retorna dict com: anxiety_indicators, hesitation_indicators, discomfort_indicators,
    matched_terms, summary.
    """
    if not text or not text.strip():
        return {
            "anxiety_indicators": 0,
            "hesitation_indicators": 0,
            "discomfort_indicators": 0,
            "matched_terms": [],
            "summary": "Texto vazio ou sem fala detectada.",
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
        else "Nenhum indicador forte detectado nas listas de termos."
    )
    return {
        "anxiety_indicators": counts["anxiety"],
        "hesitation_indicators": counts["hesitation"],
        "discomfort_indicators": counts["discomfort"],
        "matched_terms": matched,
        "summary": summary,
    }
