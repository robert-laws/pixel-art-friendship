"""HUD, dialogue box, banners and the terminal window."""
import math

import numpy as np

from .characters import CH, CW, OX, OY, sprite
from .font import ADV, draw_text, text_width
from .gfx import H, W, blend_rect, blit, fill_rect, hexc, hline, rect_outline, vline

BYTE_C, NULL_C = hexc("#4aa8ff"), hexc("#ff5a72")
GOLD, INK, PAPER = hexc("#ffd84a"), hexc("#0d0f22"), hexc("#eef0ff")


def text_c(f, cx, y, s, color, scale=1, shadow=(0, 0, 0), **kw):
    w = text_width(s, scale)
    return draw_text(f, cx - w // 2, y, s, color, scale, shadow, **kw)


def hp_bar(f, x, y, hp, color, flip=False, blink=False):
    n = 25
    fill_rect(f, x - 2, y - 2, 4 * n + 3, 9, INK)
    rect_outline(f, x - 2, y - 2, 4 * n + 3, 9, (220, 225, 255))
    lit = int(round(hp * n))
    for i in range(n):
        j = (n - 1 - i) if flip else i
        on = i < lit
        c = color if on else (30, 34, 66)
        if on and blink and (i % 2):
            c = (255, 255, 255)
        fill_rect(f, x + j * 4, y, 3, 5, c)
        if on:
            fill_rect(f, x + j * 4, y, 3, 1, tuple(min(255, int(v * 1.35) + 20) for v in c))


def hud(f, byte_hp, null_hp, t, stage="STAGE 8"):
    draw_text(f, 10, 5, "BYTE", (255, 255, 255), shadow=(0, 0, 0))
    hp_bar(f, 12, 17, byte_hp, BYTE_C, blink=byte_hp < 0.3 and int(t * 8) % 2 == 0)
    w = text_width("NULL")
    draw_text(f, W - 10 - w, 5, "NULL", (255, 255, 255), shadow=(0, 0, 0))
    hp_bar(f, W - 12 - 100 - 3 + 3 - 0, 17, null_hp, NULL_C, flip=True, blink=null_hp < 0.3 and int(t * 8) % 2 == 0)
    text_c(f, W // 2, 5, stage, GOLD)


def portrait(kind, expr="neutral", mouth=None):
    """Crop the head of a character sprite for the dialogue box."""
    spr = sprite(kind, stance="stand", arm="down", expr=expr, mouth=mouth)
    cy = OY - 23
    return spr[cy - 15:cy + 10, OX - 13:OX + 12]


def wrap(s, width):
    words, lines, cur = s.split(" "), [], ""
    for w in words:
        plain = (cur + " " + w).replace("*", "")
        if len(plain.strip()) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return lines


def caption(f, speaker, text, prog, speaking, t, expr="neutral"):
    """Bottom dialogue box. prog 0..1 = typewriter progress."""
    bx, by, bw, bh = 6, 146, W - 12, 30
    blend_rect(f, bx, by, bw, bh, (8, 10, 30), 0.82)
    col = BYTE_C if speaker == "byte" else NULL_C
    rect_outline(f, bx, by, bw, bh, col)
    rect_outline(f, bx + 1, by + 1, bw - 2, bh - 2, tuple(int(c * 0.45) for c in col))
    name = speaker.upper()
    fill_rect(f, bx + 6, by - 5, text_width(name) + 8, 10, col)
    draw_text(f, bx + 10, by - 3, name, (10, 12, 30))
    # portrait
    px, py = bx + 5, by + 3
    fill_rect(f, px - 1, py - 1, 27, 27, INK)
    rect_outline(f, px - 1, py - 1, 27, 27, col)
    mouth = ("open" if speaker == "null" else "open") if (speaking and int(t * 9) % 2 == 0) else None
    pr = portrait(speaker, expr, mouth or ("flat" if speaker == "byte" else None))
    blit(f, pr, px, py)
    lines = wrap(text, 41)
    total = sum(len(l.replace("*", "")) for l in lines)
    shown = int(prog * total + 0.999)
    y = by + 6 + (5 if len(lines) == 1 else 0)
    for l in lines[:2]:
        n = len(l.replace("*", ""))
        draw_text(f, bx + 36, y, l, PAPER, shadow=(0, 0, 20), limit=max(0, min(n, shown)))
        shown -= n
        y += 11
    if prog >= 1 and int(t * 3) % 2 == 0:            # 'more' arrow
        draw_text(f, bx + bw - 12, by + bh - 10, "▶", col)


def terminal(f, x, y, w, h, title, lines, t, cursor=None, hl_row=None, hl_color=(90, 70, 20), flash_row=None,
             pitch=10, top=16):
    """Retro terminal window. lines = [(text, color)], cursor = (row, col) or None."""
    blend_rect(f, x + 3, y + 3, w, h, (0, 0, 0), 0.5)
    fill_rect(f, x, y, w, h, (10, 14, 34))
    rect_outline(f, x, y, w, h, (120, 200, 255))
    fill_rect(f, x, y, w, 11, (36, 70, 150))
    draw_text(f, x + 5, y + 2, title, (255, 255, 255))
    for i, cc in enumerate(((255, 96, 96), (255, 210, 90), (96, 230, 130))):
        fill_rect(f, x + w - 30 + i * 9, y + 3, 5, 5, cc)
    ly = y + top
    for row, item in enumerate(lines):
        text, col = item if isinstance(item, tuple) else (item, (190, 255, 200))
        if hl_row == row:
            fill_rect(f, x + 2, ly - 1, w - 4, pitch - 1, hl_color)
        if flash_row == row and int(t * 10) % 2 == 0:
            fill_rect(f, x + 2, ly - 1, w - 4, pitch - 1, (150, 30, 50))
        draw_text(f, x + 6, ly, text, col)
        if cursor and cursor[0] == row and int(t * 2.5) % 2 == 0:
            fill_rect(f, x + 6 + cursor[1] * ADV, ly - 1, 5, 9, (255, 255, 255))
        ly += pitch
    # subtle scanlines inside the window
    for yy in range(y + 12, y + h - 1, 2):
        blend_rect(f, x + 1, yy, w - 2, 1, (0, 0, 0), 0.18)


def bulb(f, cx, cy, t):
    """Pixel light-bulb 'idea!' icon."""
    on = int(t * 8) % 2 == 0
    c = (255, 240, 120) if on else (255, 214, 60)
    fill_rect(f, cx - 5, cy - 8, 11, 10, c)
    fill_rect(f, cx - 3, cy - 10, 7, 2, c)
    fill_rect(f, cx - 3, cy + 2, 7, 3, (170, 170, 190))
    fill_rect(f, cx - 2, cy + 5, 5, 2, (120, 120, 150))
    fill_rect(f, cx - 3, cy - 6, 2, 4, (255, 255, 255))
    if on:
        for (dx, dy) in ((-10, -9), (10, -9), (-12, -1), (12, -1), (0, -15)):
            fill_rect(f, cx + dx, cy + dy, 3, 3, (255, 240, 120))


def bang(f, cx, cy, t):
    if int(t * 10) % 2:
        fill_rect(f, cx - 1, cy - 8, 3, 8, (255, 220, 60)); fill_rect(f, cx - 1, cy + 2, 3, 3, (255, 220, 60))
    else:
        fill_rect(f, cx - 2, cy - 9, 5, 9, (255, 255, 120)); fill_rect(f, cx - 2, cy + 2, 5, 4, (255, 255, 120))


def sweat(f, cx, cy, t):
    y = cy + int(t * 8) % 4
    fill_rect(f, cx, y, 2, 3, (140, 210, 255)); fill_rect(f, cx - 1, y + 2, 4, 2, (170, 230, 255))
