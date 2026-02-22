"""Heurísticas de linguagem corporal a partir dos keypoints YOLOv8-pose (COCO 17)."""

# Índices COCO: 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

# Confiança mínima do keypoint para considerar
MIN_CONF = 0.3


def _get_pt(keypoints, idx: int):
    """Retorna (x, y) do keypoint ou None se confiança baixa."""
    if keypoints is None or idx >= keypoints.shape[0]:
        return None
    row = keypoints[idx]
    if len(row) >= 3 and row[2] >= MIN_CONF:
        return (float(row[0]), float(row[1]))
    return None


def _dist(p1, p2):
    if p1 is None or p2 is None:
        return float("inf")
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def head_lowered(keypoints) -> bool:
    """Cabeça baixa (nose abaixo dos ombros) — possível desconforto/medo."""
    nose = _get_pt(keypoints, NOSE)
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    if nose is None or (ls is None and rs is None):
        return False
    shoulder_y = ls[1] if rs is None else (rs[1] if ls is None else (ls[1] + rs[1]) / 2)
    return bool(nose[1] > shoulder_y)  # y aumenta para baixo


def hands_near_face(keypoints) -> bool:
    """Mãos próximas ao rosto (pulsos perto de nariz/olhos)."""
    nose = _get_pt(keypoints, NOSE)
    lw = _get_pt(keypoints, L_WRIST)
    rw = _get_pt(keypoints, R_WRIST)
    if nose is None:
        return False
    for w in (lw, rw):
        if w is not None and _dist(w, nose) < 80:  # threshold em pixels
            return True
    return False


def arms_defensive(keypoints) -> bool:
    """Braços em postura defensiva: pulsos próximos ao tronco/ombros."""
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    le = _get_pt(keypoints, L_ELBOW)
    re = _get_pt(keypoints, R_ELBOW)
    lw = _get_pt(keypoints, L_WRIST)
    rw = _get_pt(keypoints, R_WRIST)
    if not all([ls, rs, le, re, lw, rw]):
        return False
    # Cruzados: pulsos perto da linha central (entre os ombros)
    mid_x = (ls[0] + rs[0]) / 2
    lw_near_center = abs(lw[0] - mid_x) < 60
    rw_near_center = abs(rw[0] - mid_x) < 60
    # Cotovelos mais para fora que pulsos
    elbows_out = le[0] < lw[0] and re[0] > rw[0]  # left elbow left of left wrist, etc
    return bool((lw_near_center or rw_near_center) and elbows_out)


def closed_posture(keypoints) -> bool:
    """Postura fechada: ombros para frente, braços próximos ao corpo."""
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    le = _get_pt(keypoints, L_ELBOW)
    re = _get_pt(keypoints, R_ELBOW)
    lh = _get_pt(keypoints, L_HIP)
    rh = _get_pt(keypoints, R_HIP)
    if not all([ls, rs, le, re, lh, rh]):
        return False
    # Cotovelos mais próximos (braços “fechados”) que a largura dos ombros
    shoulder_span = abs(ls[0] - rs[0])
    if shoulder_span < 1:
        return False
    elbow_span = abs(le[0] - re[0])
    return bool(elbow_span < shoulder_span * 0.85)


def compute_frame_indicators(keypoints):
    """
    Retorna um dict com indicadores de desconforto/medo/defensivo para um frame.
    keypoints: array de shape (17, 3) com (x, y, conf) por keypoint.
    """
    if keypoints is None or keypoints.shape[0] < 17:
        return {"discomfort": False, "reasons": []}
    reasons = []
    if head_lowered(keypoints):
        reasons.append("head_lowered")
    if hands_near_face(keypoints):
        reasons.append("hands_near_face")
    if arms_defensive(keypoints):
        reasons.append("arms_defensive")
    if closed_posture(keypoints):
        reasons.append("closed_posture")
    return {
        "discomfort": len(reasons) > 0,
        "reasons": reasons,
    }
