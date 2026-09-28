"""Stateless effects: projectiles, bursts, glitches, CRT. Everything is a pure function of time
so any frame can be rendered on its own."""
import math

import numpy as np

from .gfx import SCALE, blend_rect, fill_rect, hexc, hline, vline

WHITE = (255, 255, 255)


def shake(f, dx, dy):
    if dx or dy:
        f[:] = np.roll(np.roll(f, dy, axis=0), dx, axis=1)


def flash(f, a, color=WHITE):
    if a > 0:
        f[:] = (f.astype(np.float32) * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


def circle(f, cx, cy, r, c, fill=False):
    H, W = f.shape[:2]
    y0, y1, x0, x1 = max(0, int(cy - r - 1)), min(H, int(cy + r + 2)), max(0, int(cx - r - 1)), min(W, int(cx + r + 2))
    if y1 <= y0 or x1 <= x0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = (xs - cx) ** 2 + (ys - cy) ** 2
    m = d <= r * r if fill else (d <= r * r) & (d >= (r - 1.2) ** 2)
    f[ys[m], xs[m]] = c


def starburst(f, cx, cy, r, c, c2=WHITE, spikes=8, rot=0.0):
    H, W = f.shape[:2]
    for i in range(spikes):
        a = rot + i * math.tau / spikes
        L = r if i % 2 == 0 else r * 0.55
        for k in range(int(L)):
            x, y = int(cx + math.cos(a) * k), int(cy + math.sin(a) * k)
            if 0 <= x < W and 0 <= y < H:
                f[y, x] = c2 if k < L * 0.4 else c
    circle(f, cx, cy, max(2, r * 0.25), c2, fill=True)


def burst(f, cx, cy, tau, n=18, speed=90, colors=(WHITE,), life=0.7, size=2, grav=120, seed=1):
    """Particles radiating from a point, `tau` seconds after the event."""
    if tau < 0 or tau > life:
        return
    rng = np.random.default_rng(seed)
    for i in range(n):
        a = rng.uniform(0, math.tau)
        v = speed * rng.uniform(0.35, 1.0)
        x = cx + math.cos(a) * v * tau
        y = cy + math.sin(a) * v * tau + 0.5 * grav * tau * tau
        col = colors[i % len(colors)]
        sz = size if tau < life * 0.6 else max(1, size - 1)
        fill_rect(f, x, y, sz, sz, col)


def pellet(f, x, y, kind, t, size=1):
    """Buster shot. kind 'byte' (cyan) or 'null' (red). size 1 = small, 2 = big charged shot."""
    if kind == "byte":
        core, mid, edge = WHITE, hexc("#7cf0ff"), hexc("#1e8fe0")
    else:
        core, mid, edge = hexc("#ffe08a"), hexc("#ff5a5a"), hexc("#a11a3c")
    r = 3 if size == 1 else 7
    circle(f, x, y, r + 1, edge, fill=True)
    circle(f, x, y, r, mid, fill=True)
    circle(f, x, y, max(1, r - 2), core, fill=True)
    if size > 1:
        for k in range(3):       # orbiting sparkles
            a = t * 14 + k * 2.1
            fill_rect(f, x + math.cos(a) * (r + 3), y + math.sin(a) * (r + 3), 2, 2, WHITE)


def trail(f, x, y, dirx, kind, length=16):
    col = hexc("#7cf0ff") if kind == "byte" else hexc("#ff5a5a")
    for i in range(1, length, 2):
        fill_rect(f, x - dirx * (i + 4), y - 1 + (i % 3 == 0), 3 if i < 8 else 2, 2, col)


def aura(f, cx, cy, t, color, power=1.0, r=16):
    """Charging aura: pulsing rings + rising sparks."""
    for k in range(2):
        rr = r * ((t * 1.8 + k * 0.5) % 1.0)
        circle(f, cx, cy, rr, color)
    rng = np.random.default_rng(int(t * 12) + 5)
    for i in range(int(8 * power)):
        a = rng.uniform(0, math.tau)
        d = rng.uniform(6, r)
        fill_rect(f, cx + math.cos(a) * d, cy + math.sin(a) * d - (t * 30) % 10, 2, 2, color if i % 2 else WHITE)


def speedlines(f, t, color=(255, 255, 255), n=14, alpha=0.5):
    H, W = f.shape[:2]
    rng = np.random.default_rng(int(t * 30))
    for _ in range(n):
        y = int(rng.integers(6, 136))
        x = int(rng.integers(0, W))
        ln = int(rng.integers(20, 70))
        blend_rect(f, x, y, ln, 1, color, alpha)


def crt_on(f, p):
    """p 0..1: white line expands to fill the screen."""
    H, W = f.shape[:2]
    if p >= 1:
        return
    out = np.zeros_like(f)
    h = max(2, int(H * (p ** 2.2)))
    y0 = H // 2 - h // 2
    src = f[y0:y0 + h].astype(np.float32)
    glow = max(0.0, 1.0 - p * 1.6)
    out[y0:y0 + h] = (src * (1 - glow) + 255 * glow).astype(np.uint8)
    f[:] = out


def crt_off(f, p):
    """p 0..1: picture collapses to a horizontal line, then a dot."""
    H, W = f.shape[:2]
    if p <= 0:
        return
    if p >= 1:
        f[:] = 0
        return
    if p < 0.7:
        q = p / 0.7
        h = max(2, int(H * (1 - q) ** 2.0))
        y0 = H // 2 - h // 2
        band = f[y0:y0 + h].astype(np.float32)
        glow = q ** 2
        f[:] = 0
        f[y0:y0 + h] = (band * (1 - glow) + 255 * glow).astype(np.uint8)
    else:
        q = (p - 0.7) / 0.3
        f[:] = 0
        wdt = max(2, int(W * (1 - q)))
        f[H // 2 - 1:H // 2 + 1, W // 2 - wdt // 2:W // 2 + wdt // 2] = 255


# ---------------------------------------------------------------- glitch
def glitch(f, k, seed, keep_rows=None):
    """Corrupt a frame. k in 0..1 = intensity. Deterministic for a given seed."""
    H, W = f.shape[:2]
    rng = np.random.default_rng(seed)
    src = f.copy()
    # horizontal slice displacement
    for _ in range(int(3 + 22 * k)):
        y = int(rng.integers(0, H - 4))
        h = int(rng.integers(2, 6 + int(24 * k)))
        dx = int(rng.integers(-int(8 + 60 * k), int(8 + 60 * k) + 1))
        f[y:y + h] = np.roll(src[y:y + h], dx, axis=1)
    # RGB split
    s = int(1 + 5 * k)
    f[..., 0] = np.roll(f[..., 0], s, axis=1)
    f[..., 2] = np.roll(f[..., 2], -s, axis=1)
    # copy random blocks from elsewhere (data corruption)
    for _ in range(int(2 + 16 * k)):
        bw, bh = int(rng.integers(8, 56)), int(rng.integers(4, 26))
        sx, sy = int(rng.integers(0, W - bw)), int(rng.integers(0, H - bh))
        dx, dy = int(rng.integers(0, W - bw)), int(rng.integers(0, H - bh))
        f[dy:dy + bh, dx:dx + bw] = src[sy:sy + bh, sx:sx + bw]
    # solid colour blocks (missing data)
    for _ in range(int(1 + 8 * k)):
        bw, bh = int(rng.integers(6, 40)), int(rng.integers(2, 10))
        x, y = int(rng.integers(0, W - bw)), int(rng.integers(0, H - bh))
        c = [(255, 0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0), (255, 255, 0)][int(rng.integers(0, 5))]
        f[y:y + bh, x:x + bw] = c
    # invert a band
    if rng.random() < 0.5 * k:
        y = int(rng.integers(0, H - 20)); h = int(rng.integers(6, 30))
        f[y:y + h] = 255 - f[y:y + h]
    # static noise
    n = rng.random((H, W)) < 0.03 * k
    f[n] = 255


def static(f, amount, seed):
    H, W = f.shape[:2]
    rng = np.random.default_rng(seed)
    n = rng.random((H, W))
    v = (rng.random((H, W)) * 255).astype(np.uint8)
    m = n < amount
    f[m] = np.stack([v[m]] * 3, axis=1)


def smear(f, prev, k):
    """'Hall of mirrors' - blend in the previous frame with an offset (framebuffer not cleared)."""
    if prev is None:
        return
    f[:] = (f.astype(np.float32) * (1 - k) + np.roll(prev, 3, axis=1).astype(np.float32) * k).astype(np.uint8)


# ---------------------------------------------------------------- final look (applied at 1280x720)
_crt_mask = None
_crt_key = None


def crt_look(up, scale=SCALE):
    """Scanlines + soft vignette on the upscaled frame."""
    global _crt_mask, _crt_key
    if _crt_mask is None or _crt_key != (up.shape[:2], scale):
        _crt_key = (up.shape[:2], scale)
        h, w = up.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w]
        scan = np.where((ys % scale) == scale - 1, 0.86, 1.0)
        vx = (xs - w / 2) / (w / 2)
        vy = (ys - h / 2) / (h / 2)
        vig = 1.0 - 0.28 * np.clip(vx ** 2 * 0.6 + vy ** 2 * 0.8, 0, 1.3)
        _crt_mask = (scan * vig * 256).astype(np.uint16)[..., None]
    return ((up.astype(np.uint16) * _crt_mask) >> 8).astype(np.uint8)
