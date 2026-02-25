"""Body language heuristics from YOLOv8-pose keypoints (COCO 17)."""

from config import (
    HEURISTIC_BODY_TURNED_SHOULDER_DIFF_PX,
    HEURISTIC_HAND_ON_CHEST_MAX_DIST_PX,
    HEURISTIC_HAND_TO_NECK_MAX_DIST_PX,
    HEURISTIC_HANDS_CLASPED_MAX_DIST_PX,
    HEURISTIC_HANDS_NEAR_FACE_MAX_DIST_PX,
    HEURISTIC_HANDS_ON_HIPS_MAX_DIST_PX,
    HEURISTIC_LEGS_CLOSED_MAX_DIST_PX,
)

# COCO keypoint indices: 0 nose, 1-4 eyes/ears, 5-6 shoulders, 7-8 elbows,
# 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

MIN_CONF = 0.3


def _get_pt(keypoints, idx: int):
    """Return (x, y) for keypoint or None if confidence is low."""
    if keypoints is None or idx >= keypoints.shape[0]:
        return None
    row = keypoints[idx]
    if len(row) >= 3 and row[2] >= MIN_CONF:
        return float(row[0]), float(row[1])
    return None


def _dist(p1, p2):
    if p1 is None or p2 is None:
        return float("inf")
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def head_lowered(keypoints) -> bool:
    """Nose below shoulder line."""
    nose = _get_pt(keypoints, NOSE)
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    if nose is None or (ls is None and rs is None):
        return False
    shoulder_y = ls[1] if rs is None else (rs[1] if ls is None else (ls[1] + rs[1]) / 2)
    return bool(nose[1] > shoulder_y)


def hands_near_face(keypoints) -> bool:
    """Wrists near nose/eyes."""
    nose = _get_pt(keypoints, NOSE)
    lw = _get_pt(keypoints, L_WRIST)
    rw = _get_pt(keypoints, R_WRIST)
    if nose is None:
        return False
    max_distance = HEURISTIC_HANDS_NEAR_FACE_MAX_DIST_PX
    for wrist in (lw, rw):
        if wrist is not None and _dist(wrist, nose) < max_distance:
            return True
    return False


def arms_defensive(keypoints) -> bool:
    """Wrists near torso, elbows outward."""
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    le = _get_pt(keypoints, L_ELBOW)
    re = _get_pt(keypoints, R_ELBOW)
    lw = _get_pt(keypoints, L_WRIST)
    rw = _get_pt(keypoints, R_WRIST)
    pts = [ls, rs, le, re, lw, rw]
    if sum(1 for p in pts if p is not None) < 5:
        return False
    if ls is None or rs is None:
        return False
    midpoint_x = (ls[0] + rs[0]) / 2
    lw_near_center = lw is not None and abs(lw[0] - midpoint_x) < 60
    rw_near_center = rw is not None and abs(rw[0] - midpoint_x) < 60
    left_elbow_out = le is not None and lw is not None and le[0] < lw[0]
    right_elbow_out = re is not None and rw is not None and re[0] > rw[0]
    elbows_out = left_elbow_out or right_elbow_out
    return bool((lw_near_center or rw_near_center) and elbows_out)


def closed_posture(keypoints) -> bool:
    """Elbows closer than shoulders (closed arms)."""
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    le = _get_pt(keypoints, L_ELBOW)
    re = _get_pt(keypoints, R_ELBOW)
    lh = _get_pt(keypoints, L_HIP)
    rh = _get_pt(keypoints, R_HIP)
    pts = [ls, rs, le, re, lh, rh]
    if sum(1 for p in pts if p is not None) < 5:
        return False
    if not (ls and rs and le and re):
        return False
    shoulder_span = abs(ls[0] - rs[0])
    if shoulder_span < 1:
        return False
    elbow_span = abs(le[0] - re[0])
    return bool(elbow_span < shoulder_span * 0.85)


def arms_raised(keypoints) -> bool:
    """At least one wrist above shoulder line."""
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    lw = _get_pt(keypoints, L_WRIST)
    rw = _get_pt(keypoints, R_WRIST)
    if lw is not None and ls is not None and lw[1] < ls[1]:
        return True
    if rw is not None and rs is not None and rw[1] < rs[1]:
        return True
    return False


def shoulders_contracted(keypoints) -> bool:
    """Shoulder span small relative to torso."""
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    lh = _get_pt(keypoints, L_HIP)
    rh = _get_pt(keypoints, R_HIP)
    if not all([ls, rs, lh, rh]):
        return False
    shoulder_span = abs(ls[0] - rs[0])
    mid_shoulder_y = (ls[1] + rs[1]) / 2
    mid_hip_y = (lh[1] + rh[1]) / 2
    torso_height = abs(mid_shoulder_y - mid_hip_y)
    if torso_height < 10:
        return False
    return bool(shoulder_span < torso_height * 0.45)


def arms_crossed(keypoints) -> bool:
    """Arms crossed: both wrists near torso center, elbows out (defensive, closed)."""
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    le = _get_pt(keypoints, L_ELBOW)
    re = _get_pt(keypoints, R_ELBOW)
    lw = _get_pt(keypoints, L_WRIST)
    rw = _get_pt(keypoints, R_WRIST)
    if not all([ls, rs, le, re, lw, rw]):
        return False
    midpoint_x = (ls[0] + rs[0]) / 2
    lw_near = abs(lw[0] - midpoint_x) < 70
    rw_near = abs(rw[0] - midpoint_x) < 70
    left_elbow_out = le[0] < lw[0]
    right_elbow_out = re[0] > rw[0]
    return bool(lw_near and rw_near and (left_elbow_out or right_elbow_out))


def hands_on_hips(keypoints) -> bool:
    """Hands on hips: wrists near hips (assertive, impatient)."""
    lh = _get_pt(keypoints, L_HIP)
    rh = _get_pt(keypoints, R_HIP)
    lw = _get_pt(keypoints, L_WRIST)
    rw = _get_pt(keypoints, R_WRIST)
    max_distance = HEURISTIC_HANDS_ON_HIPS_MAX_DIST_PX
    if lh and lw and _dist(lw, lh) < max_distance:
        return True
    if rh and rw and _dist(rw, rh) < max_distance:
        return True
    return False


def shoulders_raised(keypoints) -> bool:
    """Shoulders raised above ear line (tension, stress)."""
    le = _get_pt(keypoints, L_EAR)
    re = _get_pt(keypoints, R_EAR)
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    if not all([ls, rs]):
        return False
    ear_y = (le[1] + re[1]) / 2 if (le and re) else (ls[1] + rs[1]) / 2 - 30
    shoulder_y = (ls[1] + rs[1]) / 2
    return bool(shoulder_y < ear_y)


def hand_on_chest(keypoints) -> bool:
    """Wrist near chest center (anxiety, self-soothing)."""
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    lh = _get_pt(keypoints, L_HIP)
    rh = _get_pt(keypoints, R_HIP)
    lw = _get_pt(keypoints, L_WRIST)
    rw = _get_pt(keypoints, R_WRIST)
    if not all([ls, rs, lh, rh]) or (lw is None and rw is None):
        return False
    midpoint_x = (ls[0] + rs[0]) / 2
    midpoint_y = ((ls[1] + rs[1]) / 2 + (lh[1] + rh[1]) / 2) / 2
    chest = (midpoint_x, midpoint_y)
    max_distance = HEURISTIC_HAND_ON_CHEST_MAX_DIST_PX
    if lw and _dist(lw, chest) < max_distance:
        return True
    if rw and _dist(rw, chest) < max_distance:
        return True
    return False


def hand_to_neck(keypoints) -> bool:
    """Wrist near neck/shoulder line (insecurity, nervousness)."""
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    le = _get_pt(keypoints, L_EAR)
    re = _get_pt(keypoints, R_EAR)
    lw = _get_pt(keypoints, L_WRIST)
    rw = _get_pt(keypoints, R_WRIST)
    if not (ls and rs) or (lw is None and rw is None):
        return False
    shoulder_y = (ls[1] + rs[1]) / 2
    ear_y = (le[1] + re[1]) / 2 if (le and re) else shoulder_y - 25
    neck_y = (shoulder_y + ear_y) / 2
    max_distance = HEURISTIC_HAND_TO_NECK_MAX_DIST_PX
    for wrist, shoulder in [(lw, ls), (lw, rs), (rw, ls), (rw, rs)]:
        if wrist and shoulder and _dist(wrist, (shoulder[0], neck_y)) < max_distance:
            return True
    return False


def leaning_back(keypoints) -> bool:
    """Torso tilted backward (distancing, caution)."""
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    lh = _get_pt(keypoints, L_HIP)
    rh = _get_pt(keypoints, R_HIP)
    nose = _get_pt(keypoints, NOSE)
    if not all([ls, rs, lh, rh]) or nose is None:
        return False
    mid_shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    mid_hip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    shoulder_span = abs(ls[0] - rs[0])
    if shoulder_span < 5:
        return False
    torso_height = mid_hip[1] - mid_shoulder[1]
    if torso_height < 20:
        return False
    ratio = torso_height / shoulder_span
    return bool(ratio < 1.2)


def leaning_forward(keypoints) -> bool:
    """Torso tilted forward (engagement, interest)."""
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    lh = _get_pt(keypoints, L_HIP)
    rh = _get_pt(keypoints, R_HIP)
    nose = _get_pt(keypoints, NOSE)
    if not all([ls, rs, lh, rh]) or nose is None:
        return False
    mid_shoulder_y = (ls[1] + rs[1]) / 2
    mid_hip_y = (lh[1] + rh[1]) / 2
    shoulder_span = abs(ls[0] - rs[0])
    if shoulder_span < 5:
        return False
    torso_height = mid_hip_y - mid_shoulder_y
    if torso_height < 20:
        return False
    nose_forward = nose[1] > mid_shoulder_y + torso_height * 0.3
    return bool(nose_forward and torso_height / shoulder_span > 2.0)


def legs_closed(keypoints) -> bool:
    """Knees close together (closed posture, protection)."""
    lk = _get_pt(keypoints, L_KNEE)
    rk = _get_pt(keypoints, R_KNEE)
    if not (lk and rk):
        return False
    max_distance = HEURISTIC_LEGS_CLOSED_MAX_DIST_PX
    return bool(_dist(lk, rk) < max_distance)


def body_turned_away(keypoints) -> bool:
    """One shoulder significantly forward (avoidance, discomfort)."""
    ls = _get_pt(keypoints, L_SHOULDER)
    rs = _get_pt(keypoints, R_SHOULDER)
    if not (ls and rs):
        return False
    diff = abs(ls[1] - rs[1])
    threshold = HEURISTIC_BODY_TURNED_SHOULDER_DIFF_PX
    return bool(diff > threshold)


def hands_clasped(keypoints) -> bool:
    """Wrists very close together (nervousness, self-control)."""
    lw = _get_pt(keypoints, L_WRIST)
    rw = _get_pt(keypoints, R_WRIST)
    if not (lw and rw):
        return False
    max_distance = HEURISTIC_HANDS_CLASPED_MAX_DIST_PX
    return bool(_dist(lw, rw) < max_distance)


def compute_frame_indicators(keypoints):
    """Return discomfort/defensive indicators for a frame."""
    if keypoints is None or keypoints.shape[0] < 17:
        return {"discomfort": False, "reasons": []}

    reasons_map = {
        "head_lowered": head_lowered,
        "hands_near_face": hands_near_face,
        "arms_defensive": arms_defensive,
        "closed_posture": closed_posture,
        "arms_raised": arms_raised,
        "shoulders_contracted": shoulders_contracted,
        "arms_crossed": arms_crossed,
        "hands_on_hips": hands_on_hips,
        "shoulders_raised": shoulders_raised,
        "hand_on_chest": hand_on_chest,
        "hand_to_neck": hand_to_neck,
        "leaning_back": leaning_back,
        "leaning_forward": leaning_forward,
        "legs_closed": legs_closed,
        "body_turned_away": body_turned_away,
        "hands_clasped": hands_clasped,
    }

    reasons = []
    for reason, reason_callable in reasons_map.items():
        if reason_callable(keypoints):
            reasons.append(reason)

    return {
        "discomfort": len(reasons) > 0,
        "reasons": reasons,
    }
