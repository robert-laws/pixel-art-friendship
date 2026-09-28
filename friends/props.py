"""Set dressing for the debug scene: chair, desk, CRT monitor, keyboard, mug and the hologram
that projects the code up from the monitor so the audience can read it."""
import math

import numpy as np

from .gfx import BAYER4, H, W, blend_rect, fill_rect, hline, vline

WOOD, WOOD_HI, WOOD_LO = (156, 100, 62), (204, 146, 92), (108, 68, 44)
DESK_DK, DESK_MD, DESK_HI = (44, 34, 72), (68, 54, 110), (96, 80, 146)
SEAT, SEAT_HI, SEAT_LO = (58, 66, 132), (104, 118, 200), (34, 40, 86)
BEZEL, BEZEL_HI, BEZEL_LO = (208, 200, 182), (240, 234, 220), (150, 142, 126)


def chair(f, x, gy=140):
    """Office chair, seat centre x. Seat top is 4px above the floor line so a seated fighter (y=4) rests on it."""
    fill_rect(f, x - 11, 114, 5, 23, SEAT)                     # backrest (behind the sitter's back)
    fill_rect(f, x - 11, 114, 1, 23, SEAT_HI)
    fill_rect(f, x - 7, 114, 1, 23, SEAT_LO)
    fill_rect(f, x - 10, 113, 3, 1, SEAT)
    fill_rect(f, x - 10, gy - 4, 21, 3, SEAT)                  # seat
    hline(f, x - 10, gy - 4, 21, SEAT_HI)
    hline(f, x - 10, gy - 2, 21, SEAT_LO)
    fill_rect(f, x - 1, gy - 1, 3, 3, (92, 98, 142))           # pole
    fill_rect(f, x - 9, gy + 2, 19, 2, (30, 30, 54))           # base
    for dx in (-9, -1, 7):
        fill_rect(f, x + dx, gy + 4, 3, 2, (18, 18, 34))       # wheels


def stool(f, x, gy=140, h=5):
    """Little hazard-striped step stool - Byte needs it to see over Null's shoulder."""
    top = gy - h
    fill_rect(f, x - 9, top, 19, 3, (255, 200, 60))
    hline(f, x - 9, top, 19, (255, 232, 130))
    hline(f, x - 9, top + 2, 19, (196, 140, 30))
    for i in range(x - 9, x + 10, 4):
        fill_rect(f, i, top + 1, 2, 1, (40, 30, 50))
    fill_rect(f, x - 8, top + 3, 2, h - 3, (150, 150, 172))
    fill_rect(f, x + 7, top + 3, 2, h - 3, (150, 150, 172))


def desk(f, x0, x1, top, gy=141):
    w = x1 - x0
    fill_rect(f, x0, top, w, 4, WOOD)
    hline(f, x0, top, w, WOOD_HI)
    hline(f, x0, top + 3, w, WOOD_LO)
    fill_rect(f, x0 + 2, top + 4, w - 4, gy - top - 4, (20, 16, 36))       # knee space
    for (bx, bw, n) in ((x0 + 3, 32, 1), (x1 - 3 - 32, 32, 2)):            # drawer stacks
        fill_rect(f, bx, top + 4, bw, gy - top - 4, DESK_DK)
        hline(f, bx, top + 4, bw, DESK_HI)
        dh = (gy - top - 6) // n
        for k in range(n):
            y = top + 5 + k * dh
            fill_rect(f, bx + 2, y + 1, bw - 4, dh - 2, DESK_MD)
            hline(f, bx + 2, y + 1, bw - 4, DESK_HI)
            fill_rect(f, bx + bw // 2 - 3, y + dh // 2 - 1, 6, 2, (206, 200, 220))   # handle
    fill_rect(f, x0, top + 4, 2, gy - top - 4, DESK_DK)
    fill_rect(f, x1 - 2, top + 4, 2, gy - top - 4, DESK_DK)


def keyboard(f, x, y, w=34):
    fill_rect(f, x, y, w, 3, (176, 170, 156))
    hline(f, x, y, w, (214, 208, 194))
    for i in range(x + 2, x + w - 1, 3):
        fill_rect(f, i, y + 1, 2, 1, (98, 92, 82))


def monitor(f, x, y, rows, t, state="normal", hl=None, cursor=None, w=46, h=27):
    """Front-facing CRT. `rows` = [(text, colour)]; drawn as tiny coloured bars (1px per character)."""
    fill_rect(f, x, y, w, h, BEZEL)
    hline(f, x, y, w, BEZEL_HI); vline(f, x, y, h, BEZEL_HI)
    hline(f, x, y + h - 1, w, BEZEL_LO); vline(f, x + w - 1, y, h, BEZEL_LO)
    sx, sy, sw, sh = x + 3, y + 3, w - 6, h - 9
    border = {"normal": (30, 44, 80), "error": (255, 60, 80), "ok": (90, 255, 140)}[state]
    if state == "error" and int(t * 10) % 2:
        border = (120, 20, 40)
    fill_rect(f, sx - 1, sy - 1, sw + 2, sh + 2, border)
    fill_rect(f, sx, sy, sw, sh, (8, 16, 34))
    for i, item in enumerate(rows[:7]):
        text, col = item if isinstance(item, tuple) else (item, (190, 255, 200))
        yy = sy + 1 + i * 2
        if hl == i:
            fill_rect(f, sx, yy - 1, sw, 2, (96, 74, 22))
        n = min(len(text.rstrip()) - 1 if text.strip() else 0, sw - 3)
        if n > 0:
            fill_rect(f, sx + 1, yy, n, 1, col)
        if cursor and cursor[0] == i and int(t * 2.5) % 2 == 0:
            fill_rect(f, sx + 1 + min(cursor[1] - 1, sw - 4), yy - 1, 2, 2, (255, 255, 255))
    if state == "ok":
        fill_rect(f, sx, sy, sw, sh, (120, 255, 160)) if int(t * 14) % 3 == 0 else None
    fill_rect(f, x + w - 8, y + h - 4, 3, 1, (90, 255, 130) if state != "error" else (255, 70, 90))   # power LED
    fill_rect(f, x + w // 2 - 6, y + h, 12, 2, (168, 160, 144))                                       # stand
    # sticky note on the bezel (of course it says TODO)
    fill_rect(f, x + w - 1, y + 5, 8, 8, (255, 226, 92))
    fill_rect(f, x + w - 1, y + 12, 8, 1, (214, 184, 60))
    for k, ww in enumerate((5, 4, 5)):
        fill_rect(f, x + w + 1, y + 6 + k * 2, ww, 1, (150, 70, 50))


def mug(f, x, y, t):
    fill_rect(f, x, y, 6, 6, (240, 240, 250)); fill_rect(f, x, y + 5, 6, 1, (170, 170, 200))
    fill_rect(f, x + 6, y + 1, 2, 3, (240, 240, 250)); fill_rect(f, x + 7, y + 2, 1, 1, (20, 16, 36))
    fill_rect(f, x + 1, y + 1, 4, 1, (110, 64, 40))
    for k in range(3):                                       # steam
        p = (t * 1.6 + k * 0.33) % 1.0
        fill_rect(f, x + 1 + k * 2 + int(math.sin(p * 7 + k) * 1.2), y - 2 - int(p * 8), 1, 1, (210, 220, 240))


def hologram(f, mx0, mx1, my, wx0, wx1, wy, t):
    """Light cone from the monitor up to the floating code window."""
    span = max(1, my - wy)
    for y in range(wy, my):
        u = (y - wy) / span
        xl, xr = wx0 + (mx0 - wx0) * u, wx1 + (mx1 - wx1) * u
        blend_rect(f, xl, y, xr - xl, 1, (110, 190, 255), 0.10 + 0.10 * u)
        if y % 2 == 0:
            for xx in (int(xl), int(xr)):
                if 0 <= xx < W:
                    f[y, xx] = (170, 226, 255)
    sy = my - int((t * 26) % span)                           # scan-line rising through the beam
    u = (sy - wy) / span
    blend_rect(f, wx0 + (mx0 - wx0) * u, sy, (wx1 - wx0) + ((mx1 - mx0) - (wx1 - wx0)) * u, 1, (220, 245, 255), 0.35)


def reveal(f, layer, p):
    """Materialise the pixels that `layer` added on top of `f` using ordered dithering (p: 0..1)."""
    changed = np.any(layer != f, axis=2)
    if p >= 1.0:
        f[changed] = layer[changed]
        return
    th = np.tile(BAYER4, (H // 4 + 1, W // 4 + 1))[:H, :W]
    show = changed & (th < p)
    edge = changed & ~show & (th < p + 0.14)
    f[show] = layer[show]
    f[edge] = (170, 240, 255)
