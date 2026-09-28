"""Low-res pixel drawing helpers (numpy). The whole film is rendered at 320x180
and then integer-upscaled 4x to 1280x720 with nearest-neighbour."""
import numpy as np

W, H, SCALE, FPS = 320, 180, 4, 30

# 4x4 Bayer matrix for ordered dithering (retro gradients)
BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16.0


def hexc(s):
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def new_frame(color=(0, 0, 0), size=None):
    w, h = size if size else (W, H)
    f = np.empty((h, w, 3), np.uint8)
    f[:] = color
    return f


def fill_rect(f, x, y, w, h, c):
    x0, y0, x1, y1 = max(0, int(x)), max(0, int(y)), min(f.shape[1], int(x + w)), min(f.shape[0], int(y + h))
    if x1 > x0 and y1 > y0:
        f[y0:y1, x0:x1] = c


def hline(f, x, y, w, c):
    fill_rect(f, x, y, w, 1, c)


def vline(f, x, y, h, c):
    fill_rect(f, x, y, 1, h, c)


def rect_outline(f, x, y, w, h, c):
    hline(f, x, y, w, c); hline(f, x, y + h - 1, w, c)
    vline(f, x, y, h, c); vline(f, x + w - 1, y, h, c)


def blend_rect(f, x, y, w, h, c, a):
    x0, y0, x1, y1 = max(0, int(x)), max(0, int(y)), min(f.shape[1], int(x + w)), min(f.shape[0], int(y + h))
    if x1 > x0 and y1 > y0:
        reg = f[y0:y1, x0:x1].astype(np.float32)
        f[y0:y1, x0:x1] = (reg * (1 - a) + np.array(c, np.float32) * a).astype(np.uint8)


def dither_gradient(f, y0, y1, stops):
    """Vertical dithered gradient between rows y0..y1. stops = [(t, (r,g,b)), ...] t in 0..1.
    Uses a Bayer matrix to pick between the two bracketing colours -> classic banded 16-bit sky."""
    h = y1 - y0
    ys = np.arange(h)[:, None] / max(1, h - 1)
    xs = np.arange(f.shape[1])[None, :]
    th = BAYER4[(np.arange(y0, y1)[:, None]) % 4, xs % 4]
    out = np.zeros((h, f.shape[1], 3), np.uint8)
    for (ta, ca), (tb, cb) in zip(stops[:-1], stops[1:]):
        m = (ys >= ta) & (ys <= tb) & np.ones_like(th, bool)
        local = np.clip((ys - ta) / max(1e-6, tb - ta), 0, 1)
        pick = (local > th)
        col = np.where(pick[..., None], np.array(cb, np.uint8), np.array(ca, np.uint8))
        out = np.where(m[..., None], col, out)
    f[y0:y1] = out


def blit(f, spr, x, y, flip=False, alpha=1.0):
    """Blit an RGBA uint8 sprite with its top-left at (x,y). Binary alpha (pixel-art) unless alpha<1."""
    if flip:
        spr = spr[:, ::-1]
    sh, sw = spr.shape[:2]
    x, y = int(round(x)), int(round(y))
    fx0, fy0 = max(0, x), max(0, y)
    fx1, fy1 = min(f.shape[1], x + sw), min(f.shape[0], y + sh)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    s = spr[fy0 - y:fy1 - y, fx0 - x:fx1 - x]
    m = (s[..., 3] > 127)
    reg = f[fy0:fy1, fx0:fx1]
    if alpha >= 1.0:
        reg[m] = s[..., :3][m]
    else:
        reg[m] = (reg[m] * (1 - alpha) + s[..., :3][m] * alpha).astype(np.uint8)


def sprite_from_ascii(rows, pal):
    h, w = len(rows), max(len(r) for r in rows)
    a = np.zeros((h, w, 4), np.uint8)
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            if ch in pal and pal[ch] is not None:
                a[y, x] = (*pal[ch], 255)
    return a


def tint(spr, c, amt):
    s = spr.copy()
    m = s[..., 3] > 0
    s[..., :3][m] = (s[..., :3][m] * (1 - amt) + np.array(c) * amt).astype(np.uint8)
    return s


def silhouette(spr, c):
    s = spr.copy()
    m = s[..., 3] > 0
    s[..., :3][m] = c
    return s


def upscale(f, scale=SCALE):
    return np.repeat(np.repeat(f, scale, axis=0), scale, axis=1)


class Art:
    """Tiny RGBA canvas with a local origin, used to build characters procedurally."""

    def __init__(self, w, h, ox, oy):
        self.a = np.zeros((h, w, 4), np.uint8)
        self.w, self.h, self.ox, self.oy = w, h, ox, oy

    def px(self, x, y, c):
        X, Y = self.ox + x, self.oy + y
        if 0 <= X < self.w and 0 <= Y < self.h:
            self.a[Y, X] = (*c, 255)

    def erase(self, x, y):
        X, Y = self.ox + x, self.oy + y
        if 0 <= X < self.w and 0 <= Y < self.h:
            self.a[Y, X] = 0

    def rect(self, x, y, w, h, c):
        X0, Y0 = self.ox + x, self.oy + y
        X0c, Y0c = max(0, X0), max(0, Y0)
        X1, Y1 = min(self.w, X0 + w), min(self.h, Y0 + h)
        if X1 > X0c and Y1 > Y0c:
            self.a[Y0c:Y1, X0c:X1] = (*c, 255)

    def shaded(self, x, y, w, h, base, hi, lo):
        self.rect(x, y, w, h, base)
        self.rect(x, y, w, 1, hi)
        self.rect(x, y + h - 1, w, 1, lo)
        self.rect(x + w - 1, y, 1, h, lo)

    def ellipse(self, cx, cy, rx, ry, c):
        ys, xs = np.mgrid[0:self.h, 0:self.w]
        X, Y = xs - self.ox, ys - self.oy
        m = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0
        self.a[m] = (*c, 255)

    def line(self, x0, y0, x1, y1, c, t=1):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for i in range(n + 1):
            u = i / max(1, n)
            x, y = round(x0 + (x1 - x0) * u), round(y0 + (y1 - y0) * u)
            self.rect(x, y, t, t, c)

    def outline(self, c):
        m = self.a[..., 3] > 0
        n = np.zeros_like(m)
        n[1:, :] |= m[:-1, :]; n[:-1, :] |= m[1:, :]; n[:, 1:] |= m[:, :-1]; n[:, :-1] |= m[:, 1:]
        edge = n & ~m
        self.a[edge] = (*c, 255)
        return self
