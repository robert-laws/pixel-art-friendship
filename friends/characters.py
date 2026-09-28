"""BYTE (blue hero) and NULL (crimson rival), drawn procedurally into small RGBA sprites.

Local coordinates: x=0 is the body centre, y=0 is the ground line (rows above are negative).
Sprites face RIGHT; flip when blitting. Everything is cached, so poses are cheap."""
import math
from functools import lru_cache

import numpy as np

from .gfx import Art, hexc

CW, CH, OX, OY = 56, 56, 26, 52   # canvas size + local origin (ground line = row OY)
OUT = hexc("#0d0f22")

# ---------------------------------------------------------------- palettes
BYTE = dict(
    base=hexc("#2f74f0"), dark=hexc("#1a44b0"), hi=hexc("#79b0ff"),
    accent=hexc("#5ce3ff"), accent_d=hexc("#26a3d6"),
    skin=hexc("#ffd2ac"), skin_d=hexc("#e9a37f"),
)
NULL = dict(
    base=hexc("#d92f4a"), dark=hexc("#8a1734"), hi=hexc("#ff7a86"),
    suit=hexc("#41304f"), suit_d=hexc("#271a36"),
    gold=hexc("#ffc93c"), gold_d=hexc("#c98a1c"),
    visor=hexc("#160c25"), eye=hexc("#fff27a"), scarf=hexc("#ffe08a"), scarf_d=hexc("#e0a63c"),
)
WHITE, BLACK, RED = (255, 255, 255), (12, 12, 24), hexc("#e0344c")

HIP = dict(stand=9, run=9, jump=10, fall=10, hurt=9, sit=3, crouch=6)


def _shaded_ellipse(A, cx, cy, rx, ry, base, hi, lo):
    ys, xs = np.mgrid[0:A.h, 0:A.w]
    X, Y = xs - A.ox, ys - A.oy
    nx, ny = (X - cx) / rx, (Y - cy) / ry
    d = nx * nx + ny * ny
    m = d <= 1.0
    lit = nx * 0.6 - ny * 0.8
    A.a[m] = (*base, 255)
    A.a[m & (lit > 0.55) & (d > 0.30)] = (*hi, 255)
    A.a[m & (lit < -0.40) & (d > 0.30)] = (*lo, 255)
    return m


def _leg(A, xt, xf, lift, hip, col, col_d, boot, boot_d, w=4):
    y0, y1 = -hip, -1 - lift
    n = max(1, y1 - y0)
    for y in range(y0, y1 + 1):
        t = (y - y0) / n
        x = round(xt + (xf - xt) * t)
        A.rect(x - 2, y, w, 1, col)
        A.px(x - 2, y, col_d)
    for k in range(4):                       # boot
        y = y1 - 3 + k
        A.rect(xf - 2, y, 6, 1, boot)
    A.rect(xf - 2, y1, 6, 1, boot_d)
    A.rect(xf - 2, y1 - 3, 4, 1, col_d)


def _legs(A, P, stance, phase, hip, kind):
    if kind == "byte":
        col, col_d, boot, boot_d = P["base"], P["dark"], P["accent"], P["accent_d"]
    else:
        col, col_d, boot, boot_d = P["suit"], P["suit_d"], P["base"], P["dark"]
    if stance == "sit":
        # far leg then near leg, stretched forward along the floor
        A.rect(-2, -5, 12, 4, col_d); A.rect(9, -6, 5, 5, boot_d)
        A.rect(-3, -4, 12, 4, col); A.rect(8, -5, 6, 5, boot)
        A.rect(8, -1, 6, 1, boot_d)
        A.rect(-3, -4, 12, 1, P["hi"] if kind == "byte" else P["suit"])
        return
    if stance == "run":
        a, b = ((-3, -9, 2), (2, 7, 0)) if phase % 2 == 0 else ((-3, 6, 3), (2, -8, 1))
    elif stance == "jump":
        a, b = (-3, -7, 4), (2, 6, 2)
    elif stance == "fall":
        a, b = (-3, -5, 1), (2, 5, 3)
    elif stance == "hurt":
        a, b = (-3, -7, 0), (2, 6, 1)
    else:
        a, b = (-3, -4, 0), (2, 3, 0)
    _leg(A, a[0], a[1], a[2], hip, col_d, col_d, boot_d, boot_d)
    _leg(A, b[0], b[1], b[2], hip, col, col_d, boot, boot_d)


def _torso(A, P, top, kind, lean):
    x = lean
    if kind == "byte":
        A.shaded(x - 6, top, 12, 9, P["base"], P["hi"], P["dark"])
        A.rect(x + 0, top + 2, 5, 3, P["accent"]); A.rect(x + 0, top + 4, 5, 1, P["accent_d"])
        A.rect(x - 6, top + 8, 12, 1, P["dark"])
        A.rect(x - 1, top + 7, 3, 2, P["accent_d"])
    else:
        A.shaded(x - 6, top, 12, 9, P["base"], P["hi"], P["dark"])
        A.rect(x - 6, top + 5, 12, 4, P["suit"]); A.rect(x - 6, top + 8, 12, 1, P["suit_d"])
        A.rect(x - 1, top + 6, 4, 2, P["gold"]); A.px(x + 1, top + 6, P["gold_d"])   # buckle
        A.rect(x + 1, top + 1, 3, 3, P["gold"]); A.px(x + 3, top + 3, P["gold_d"])   # chest gem
        A.rect(x - 6, top, 12, 1, P["gold"])                                          # collar trim


def _scarf(A, P, top, lean, phase, wind):
    x0, y0 = lean - 5, top + 1
    n = 16 + int(7 * wind)
    for i in range(n):
        u = i / (n - 1)
        x = x0 - 2 - i
        y = y0 + u * (2 + 4 * (1 - wind)) - u * wind * 6 + math.sin(i * 0.42 - phase * math.pi / 2) * (1.2 + 1.6 * wind) * min(1, u * 3)
        th = 4 if i < 8 else (3 if i < 14 else 2)
        yy = round(y)
        A.rect(x, yy, 1, th, P["scarf"])
        A.px(x, yy + th - 1, P["scarf_d"])
        if i % 5 == 2:
            A.px(x, yy + 1, P["scarf_d"])
    A.rect(lean - 6, top - 1, 4, 4, P["scarf"]); A.rect(lean - 6, top + 2, 4, 1, P["scarf_d"])   # knot


def _arm(A, P, top, kind, arm, lean, back=False):
    """Draw one arm. Front arm carries the buster in 'aim'."""
    x = lean
    if kind == "byte":
        s, sd, sh, acc, accd = P["base"], P["dark"], P["hi"], P["accent"], P["accent_d"]
    else:
        s, sd, sh, acc, accd = P["suit"], P["suit_d"], P["base"], P["gold"], P["gold_d"]
    if back:
        s, sd = sd, sd
        acc = accd
    if arm == "down":
        A.shaded(x + 2 - (10 if back else 0), top + 1, 4, 8, s, sh, sd)
        A.rect(x + 2 - (10 if back else 0), top + 8, 4, 3, acc)
    elif arm == "aim":
        A.shaded(x + 1, top + 2, 6, 4, s, sh, sd)
        A.shaded(x + 6, top + 1, 11, 6, P["base"] if kind == "byte" else P["dark"], P["hi"], P["dark"] if kind == "byte" else P["suit_d"])
        A.rect(x + 10, top + 1, 2, 6, acc); A.rect(x + 10, top + 6, 2, 1, accd)
        A.rect(x + 15, top + 2, 2, 4, acc); A.rect(x + 16, top + 3, 1, 2, BLACK)
    elif arm == "brace":            # off-hand supporting the buster
        A.rect(x - 3, top + 3, 9, 3, sd); A.rect(x + 5, top + 2, 3, 4, acc)
    elif arm == "raise":
        if back:
            A.line(x - 4, top + 3, x - 9, top - 6, s, 4)
            A.rect(x - 12, top - 10, 5, 5, acc); A.rect(x - 12, top - 6, 5, 1, accd)
        else:
            A.line(x + 6, top + 3, x + 13, top - 6, s, 4)
            A.rect(x + 12, top - 11, 5, 5, acc); A.rect(x + 12, top - 7, 5, 1, accd)
    elif arm == "reach":
        A.shaded(x + 1, top + 3, 8, 3, s, sh, sd)
        A.rect(x + 8, top + 2, 4, 5, acc); A.rect(x + 8, top + 6, 4, 1, accd)
    elif arm == "point":
        A.shaded(x + 1, top + 2, 9, 3, s, sh, sd)
        A.rect(x + 9, top + 2, 3, 3, acc); A.rect(x + 12, top + 2, 4, 1, acc)
    elif arm == "cross":
        A.shaded(x - 5, top + 3, 11, 4, s, sh, sd)
        A.rect(x - 6, top + 3, 3, 4, acc); A.rect(x + 4, top + 3, 3, 3, acc)
    elif arm == "chin":
        A.shaded(x + 2, top + 4, 6, 3, s, sh, sd)
        A.shaded(x + 6, top - 1, 3, 8, s, sh, sd)
        A.rect(x + 5, top - 3, 4, 3, acc)
    elif arm == "shrug":
        A.shaded(x + 3, top + 4, 5, 3, s, sh, sd)
        A.rect(x + 6, top + 0, 4, 5, acc); A.rect(x + 6, top + 4, 4, 1, accd)
    elif arm == "fist":
        A.shaded(x + 1, top + 2, 7, 4, s, sh, sd)
        A.rect(x + 7, top + 1, 5, 6, acc); A.rect(x + 7, top + 6, 5, 1, accd)
    elif arm == "hit":
        if back:
            A.line(x - 4, top + 3, x - 11, top + 8, s, 4)
            A.rect(x - 14, top + 7, 4, 4, acc)
        else:
            A.line(x + 5, top + 3, x + 11, top + 9, s, 4)
            A.rect(x + 10, top + 8, 5, 4, acc)


# ---------------------------------------------------------------- heads
def _head_byte(A, P, cy, lean, expr, mouth):
    x = lean
    _shaded_ellipse(A, x, cy, 8.4, 7.4, P["base"], P["hi"], P["dark"])
    # crown stripe + ear
    A.rect(x - 4, cy - 7, 6, 1, P["accent"]); A.rect(x - 5, cy - 6, 3, 1, P["accent"])
    A.rect(x - 6, cy, 4, 4, P["accent"]); A.rect(x - 5, cy + 1, 2, 2, P["accent_d"])
    # face window (skin) inside the front of the helmet
    for yy in range(cy - 1, cy + 7):
        for xx in range(x + 1, x + 9):
            dx, dy = (xx - x) / 8.6, (yy - cy) / 7.6
            if dx * dx + dy * dy <= 1.0:
                A.px(xx, yy, P["skin"])
    A.rect(x + 1, cy - 2, 8, 1, P["dark"])                 # brim shadow
    for xx in range(x + 1, x + 9):
        A.px(xx, cy + 6, P["skin_d"]) if ((xx - x) / 8.6) ** 2 + (6 / 7.6) ** 2 <= 1.0 else None
    ex, ey = x + 4, cy
    if expr == "neutral":
        A.rect(ex, ey, 3, 3, WHITE); A.rect(ex + 2, ey + 1, 1, 2, BLACK)
    elif expr == "angry":
        A.rect(ex, ey + 1, 3, 2, WHITE); A.rect(ex + 2, ey + 1, 1, 2, BLACK)
        A.rect(ex - 1, ey - 1, 2, 1, BLACK); A.rect(ex + 1, ey, 3, 1, BLACK)
    elif expr == "wide":
        A.rect(ex - 1, ey - 1, 4, 4, WHITE); A.px(ex + 1, ey + 1, BLACK); A.px(ex + 1, ey, BLACK)
    elif expr == "happy":
        A.px(ex - 1, ey + 2, BLACK); A.rect(ex, ey + 1, 3, 1, BLACK); A.px(ex + 3, ey + 2, BLACK)
    elif expr == "blink":
        A.rect(ex, ey + 1, 3, 1, BLACK)
    elif expr == "worried":
        A.rect(ex, ey + 1, 3, 3, WHITE); A.rect(ex + 1, ey + 2, 2, 2, BLACK)
        A.rect(ex - 1, ey, 2, 1, BLACK); A.px(ex + 1, ey - 1, BLACK); A.rect(ex + 2, ey - 1, 2, 1, BLACK)
    elif expr == "ko":
        for (dx, dy) in ((0, 0), (2, 0), (1, 1), (0, 2), (2, 2)):
            A.px(ex + dx, ey + dy, BLACK)
    elif expr == "smug":
        A.rect(ex, ey + 1, 3, 2, WHITE); A.rect(ex + 2, ey + 1, 1, 2, BLACK); A.rect(ex, ey, 3, 1, P["dark"])
    if mouth is None:
        mouth = dict(neutral="flat", angry="flat", wide="open", happy="smile", blink="flat",
                     worried="frown", ko="open", smug="smile").get(expr, "flat")
    my = cy + 4
    if mouth == "flat":
        A.rect(x + 4, my, 3, 1, BLACK)
    elif mouth == "smile":
        A.px(x + 3, my - 1, BLACK); A.rect(x + 4, my, 3, 1, BLACK); A.px(x + 7, my - 1, BLACK)
    elif mouth == "frown":
        A.px(x + 3, my + 1, BLACK); A.rect(x + 4, my, 3, 1, BLACK); A.px(x + 7, my + 1, BLACK)
    elif mouth == "open":
        A.rect(x + 4, my - 1, 3, 3, BLACK); A.px(x + 5, my + 1, RED)
    elif mouth == "shout":
        A.rect(x + 3, my - 1, 5, 4, BLACK); A.rect(x + 4, my + 1, 3, 2, RED)


def _head_null(A, P, cy, lean, expr, mouth):
    x = lean
    # horn sweeping back
    A.line(x + 0, cy - 6, x - 8, cy - 13, P["gold"], 3)
    A.line(x + 1, cy - 5, x - 7, cy - 11, P["gold_d"], 1)
    _shaded_ellipse(A, x, cy, 8.4, 7.4, P["base"], P["hi"], P["dark"])
    A.rect(x - 3, cy - 7, 8, 1, P["gold"])                                   # crest trim
    A.rect(x - 7, cy - 1, 4, 5, P["suit"]); A.rect(x - 6, cy, 2, 3, P["gold"])   # ear disc
    # visor + chin guard
    for yy in range(cy - 2, cy + 4):
        for xx in range(x - 1, x + 10):
            dx, dy = (xx - x) / 9.0, (yy - cy) / 7.9
            if dx * dx + dy * dy <= 1.0:
                A.px(xx, yy, P["visor"])
    for yy in range(cy + 4, cy + 7):
        for xx in range(x + 1, x + 9):
            dx, dy = (xx - x) / 8.6, (yy - cy) / 7.6
            if dx * dx + dy * dy <= 1.0:
                A.px(xx, yy, P["suit"])
    A.rect(x + 1, cy + 6, 6, 1, P["suit_d"])
    ex, ey, E = x + 4, cy, P["eye"]
    if expr in ("neutral", "deadpan"):
        if expr == "deadpan":
            A.rect(ex, ey + 1, 5, 1, E); A.rect(ex, ey, 5, 1, P["visor"])
        else:
            A.rect(ex, ey, 5, 2, E); A.px(ex + 4, ey, WHITE)
    elif expr == "angry":
        A.rect(ex, ey + 1, 5, 2, E); A.rect(ex - 1, ey - 1, 3, 1, P["visor"]); A.rect(ex, ey, 2, 1, P["visor"])
        A.px(ex + 4, ey, E)
    elif expr == "wide":
        A.rect(ex, ey - 1, 4, 4, E); A.rect(ex + 1, ey, 2, 2, WHITE)
    elif expr == "happy":
        A.px(ex, ey + 2, E); A.rect(ex + 1, ey + 1, 3, 1, E); A.px(ex + 4, ey + 2, E)
    elif expr == "blink":
        A.rect(ex, ey + 1, 5, 1, P["suit"])
    elif expr == "ko":
        for (dx, dy) in ((0, -1), (3, -1), (1, 0), (2, 0), (1, 1), (0, 2), (3, 2)):
            A.px(ex + dx, ey + dy + 1, E)
    elif expr == "smug":
        A.rect(ex, ey + 1, 5, 1, E); A.rect(ex, ey - 1, 5, 2, P["visor"]); A.px(ex + 4, ey, E)
    if mouth == "open":
        A.rect(x + 3, cy + 4, 5, 2, hexc("#c9c4dd")); A.rect(x + 3, cy + 5, 5, 1, hexc("#8f89a6"))
    elif mouth == "shout":
        A.rect(x + 2, cy + 4, 6, 3, hexc("#e9e5ff")); A.rect(x + 3, cy + 6, 4, 1, RED)
    else:
        A.rect(x + 3, cy + 5, 5, 1, hexc("#5a4a6e"))


def _make(kind, stance, arm, back, expr, phase, mouth, scarf, wind, lean):
    A = Art(CW, CH, OX, OY)
    P = BYTE if kind == "byte" else NULL
    hip = HIP[stance]
    top = -hip - 9
    cy = top - 5
    if stance == "sit":
        pass
    if kind == "null":
        _scarf(A, P, top, lean, scarf, wind)
    # back arm, legs, torso, head, front arm
    _arm(A, P, top, kind, back if back else "down", lean, back=True)
    _legs(A, P, stance, phase, hip, kind)
    _torso(A, P, top, kind, lean)
    behind = arm in ("raise", "hit")
    if behind:
        _arm(A, P, top, kind, arm, lean)
    (_head_byte if kind == "byte" else _head_null)(A, P, cy, lean, expr, mouth)
    if not behind:
        _arm(A, P, top, kind, arm, lean)
    A.outline(OUT)
    return A.a


@lru_cache(maxsize=4096)
def sprite(kind, stance="stand", arm="down", back=None, expr="neutral", phase=0, mouth=None,
           scarf=0, wind=0.0, lean=0):
    return _make(kind, stance, arm, back, expr, phase, mouth, scarf, wind, lean)


def muzzle(kind, stance="stand", lean=0):
    """Local (x, y) of the buster muzzle when arm='aim'."""
    top = -HIP[stance] - 9
    return lean + 17, top + 4


def head_top(stance="stand"):
    return -HIP[stance] - 9 - 5 - 8


def head_center(stance="stand"):
    return -HIP[stance] - 9 - 5


def blit_char(frame, kind, x, ground_y, flip=False, **kw):
    """Place a character with its body centre at x and feet on ground_y."""
    from .gfx import blit
    spr = sprite(kind, **kw)
    ox = OX if not flip else CW - 1 - OX
    blit(frame, spr, x - ox, ground_y - OY, flip=flip)
