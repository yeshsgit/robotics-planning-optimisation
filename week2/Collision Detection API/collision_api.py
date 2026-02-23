# collision_api.py
# 2D 2R robot collision checker against global 4-point rectangles.
# API: collision(theta1, L1, theta2, L2) -> bool
#
# Conventions:
# - Angles in radians
# - Robot base is BASE = (x0, y0) 
# - RECTS is a global list of rectangles, each rectangle is 4 points [(x,y),...]
# - Touching boundary counts as collision (safer for planning)

# =========================
# How to call?

##from collision_api import collision, RECTS, preprocess_rects

### If you want to set obstacles in your script:
##RECTS[:] = [
##    [(0.3,0.2),(0.5,0.2),(0.5,0.4),(0.3,0.4)],
##    [(0.6,0.6),(0.8,0.6),(0.8,0.8),(0.6,0.8)],
##]
##preprocess_rects(RECTS)

##ok = not collision(theta1, L1, theta2, L2)

# =========================




from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Sequence
import math

# =========================
# User-editable globals
# =========================

BASE: Tuple[float, float] = (0.0, 0.0)

# Each rectangle: 4 corner points (can be CW or CCW).
# Replace these with your assignment obstacles.
RECTS: List[List[Tuple[float, float]]] = [
    # Example rectangle (a box):
    [(0.3, 0.2), (0.5, 0.2), (0.5, 0.4), (0.3, 0.4)],
]

# Optional: link thickness (radius). If 0.0, links are line segments.
# If >0, we approximate thickness by inflating rectangle AABB in the broad-phase
# and doing additional distance checks would be needed for exact capsule-vs-rect;
# for now keep 0.0 unless you want the simple conservative inflation behavior.
LINK_RADIUS: float = 0.0

EPS: float = 1e-12

# =========================
# Internal preprocessed rects
# =========================

@dataclass(frozen=True)
class RectCache:
    pts: Tuple[Tuple[float, float], ...]        # ordered corners
    edges: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]  # 4 edges
    xmin: float
    xmax: float
    ymin: float
    ymax: float

_RECT_CACHE: List[RectCache] = []  # filled by preprocess_rects()


# =========================
# Geometry helpers
# =========================

def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])

def _cross(a, b) -> float:
    return a[0] * b[1] - a[1] * b[0]

def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1]

def _orient(p, q, r) -> float:
    # cross((q-p),(r-p))
    return _cross(_sub(q, p), _sub(r, p))

def _on_segment(p, q, r) -> bool:
    # q on segment pr (collinear assumed)
    return (min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12 and
            min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12)

def _segments_intersect(a, b, c, d) -> bool:
    # Includes collinear overlap; boundary touch => True (collision)
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    # Proper intersection
    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True

    # Collinear / touching
    if abs(o1) < 1e-12 and _on_segment(a, c, b): return True
    if abs(o2) < 1e-12 and _on_segment(a, d, b): return True
    if abs(o3) < 1e-12 and _on_segment(c, a, d): return True
    if abs(o4) < 1e-12 and _on_segment(c, b, d): return True

    return False

def _segment_aabb(a, b):
    return (min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1]))

def _aabb_overlap(axmin, axmax, aymin, aymax, bxmin, bxmax, bymin, bymax) -> bool:
    return not (axmax < bxmin or axmin > bxmax or aymax < bymin or aymin > bymax)

def _order_rect_points(pts: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    """
    Ensure 4 points are ordered around centroid (CW/CCW).
    Works for rectangles even if input is shuffled.
    """
    if len(pts) != 4:
        raise ValueError("Rectangle must have exactly 4 points")
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    pts_sorted = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return tuple((float(x), float(y)) for (x, y) in pts_sorted)

def _point_in_convex_quad(p: Tuple[float, float], quad: Tuple[Tuple[float, float], ...]) -> bool:
    """
    Point inside/on boundary of convex quadrilateral (rectangle).
    Uses consistent orientation sign.
    """
    # Check sign of cross products for consecutive edges
    # Accept boundary as inside (collision-safe)
    s = None
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        val = _orient(a, b, p)
        if abs(val) < 1e-12:
            # On edge line; still need ensure within edge bounds -> treat as inside for safety
            # (Rect convex, so near enough)
            continue
        sign = val > 0
        if s is None:
            s = sign
        elif s != sign:
            return False
    return True

def preprocess_rects(rects: List[List[Tuple[float, float]]]) -> None:
    """
    Precompute ordered points, edges, and AABB for all rectangles.
    Call this once after setting RECTS. Automatically called at import.
    """
    global _RECT_CACHE
    cache: List[RectCache] = []
    for rect in rects:
        ordered = _order_rect_points(rect)
        edges = tuple((ordered[i], ordered[(i + 1) % 4]) for i in range(4))
        xs = [p[0] for p in ordered]
        ys = [p[1] for p in ordered]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        cache.append(RectCache(pts=ordered, edges=edges, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax))
    _RECT_CACHE = cache


# =========================
# Segment vs rect collision
# =========================

def _segment_hits_rect(a: Tuple[float, float], b: Tuple[float, float], rect: RectCache) -> bool:
    # Broad-phase AABB overlap (inflate rect AABB by LINK_RADIUS as conservative thickness)
    sa_xmin, sa_xmax, sa_ymin, sa_ymax = _segment_aabb(a, b)
    pad = float(LINK_RADIUS)
    if not _aabb_overlap(
        sa_xmin, sa_xmax, sa_ymin, sa_ymax,
        rect.xmin - pad, rect.xmax + pad, rect.ymin - pad, rect.ymax + pad
    ):
        return False

    # If either endpoint is inside/on boundary -> collision
    if _point_in_convex_quad(a, rect.pts): return True
    if _point_in_convex_quad(b, rect.pts): return True

    # Edge intersection test
    for (c, d) in rect.edges:
        if _segments_intersect(a, b, c, d):
            return True

    return False


# =========================
# Forward kinematics (2R)
# =========================

def _fk(theta1: float, L1: float, theta2: float, L2: float):
    x0, y0 = BASE
    x1 = x0 + L1 * math.cos(theta1)
    y1 = y0 + L1 * math.sin(theta1)

    x2 = x1 + L2 * math.cos(theta1 + theta2)
    y2 = y1 + L2 * math.sin(theta1 + theta2)

    p0 = (x0, y0)
    p1 = (x1, y1)
    p2 = (x2, y2)
    return p0, p1, p2


# =========================
# Public API
# =========================

def collision(theta1: float, L1: float, theta2: float, L2: float) -> bool:
    """
    Collision API for planners.
    Returns:
        True  -> in collision
        False -> collision-free
    """
    # Optional sanity checks
    if L1 < 0 or L2 < 0:
        return True  # treat invalid lengths as collision

    p0, p1, p2 = _fk(theta1, L1, theta2, L2)

    # Check each link segment against each rectangle
    seg1 = (p0, p1)
    seg2 = (p1, p2)

    for rect in _RECT_CACHE:
        if _segment_hits_rect(seg1[0], seg1[1], rect):
            return True
        if _segment_hits_rect(seg2[0], seg2[1], rect):
            return True

    return False


# =========================
# Optional: simple debug viz + self-test
# =========================

def debug_plot(theta1: float, L1: float, theta2: float, L2: float, title: str = "") -> None:
    """
    Quick visualization to sanity-check geometry. Requires matplotlib.
    Not used by planners; just for your development.
    """
    import matplotlib.pyplot as plt

    p0, p1, p2 = _fk(theta1, L1, theta2, L2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal", adjustable="box")

    # Draw rectangles
    for rect in _RECT_CACHE:
        xs = [p[0] for p in rect.pts] + [rect.pts[0][0]]
        ys = [p[1] for p in rect.pts] + [rect.pts[0][1]]
        ax.plot(xs, ys)

    # Draw robot
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=3)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=3)
    ax.scatter([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]])

    col = collision(theta1, L1, theta2, L2)
    ax.set_title(f"{title}  collision={col}")
    plt.show()


def _self_test():
    # Minimal sanity tests (adjust RECTS for meaningful tests)
    if not _RECT_CACHE:
        print("[collision_api] No RECTS defined; self-test skipped.")
        return
    # Pick a random configuration and print collision result
    t1, t2 = 0.3, -0.7
    L1, L2 = 0.6, 0.4
    print("[collision_api] sample collision:", collision(t1, L1, t2, L2))
    debug_plot(t1, L1, t2, L2, title="Test Pose")


# Preprocess at import so planners can immediately call collision()
preprocess_rects(RECTS)

if __name__ == "__main__":
    _self_test()