"""Environments: a parallax cyber-city bridge (dusk + dawn) and the 'debug void'."""
import math

import numpy as np

from .gfx import (BAYER4, H, W, blit, blend_rect, dither_gradient, fill_rect, hexc, hline, lerp,
                  new_frame, sprite_from_ascii, vline)

GY = 140  # ground line (feet baseline)


class Theme:
    def __init__(self, **kw):
        self.__dict__.update({k: (hexc(v) if isinstance(v, str) else v) for k, v in kw.items()})


DUSK = Theme(
    sky=[(0.0, hexc("#120a2b")), (0.34, hexc("#2a1550")), (0.62, hexc("#7b2b7a")), (0.82, hexc("#e2587b")), (1.0, hexc("#ffb374"))],
    far="#2b1b52", far_hi="#3b2768", mid="#1b1f4f", mid_hi="#2b3170", win_a="#ffd36b", win_b="#6ee7ff",
    steel="#3a4772", steel_hi="#6b80b8", steel_lo="#1a2141", floor="#252b52", floor_lo="#171a38", edge="#8595dc",
    neon="#ff5da2", neon2="#59e6ff", pole="#2c3663", lamp="#fff3b0", sun_a="#ffd06a", sun_b="#ff5f8e",
    cloud="#4a2a70", star=True, dim=1.0)
DAWN = Theme(
    sky=[(0.0, hexc("#4f9df5")), (0.38, hexc("#8fcbff")), (0.68, hexc("#ffd0e0")), (0.88, hexc("#ffe3a0")), (1.0, hexc("#fff3c4"))],
    far="#94aade", far_hi="#a9bdec", mid="#6f86cc", mid_hi="#8ba2e0", win_a="#ffffff", win_b="#ffe9a8",
    steel="#8396c8", steel_hi="#c3d0f2", steel_lo="#5567a0", floor="#aab7ec", floor_lo="#8593cc", edge="#ffffff",
    neon="#ffcf5a", neon2="#ff7fb0", pole="#7385bb", lamp="#ffffff", sun_a="#fff2a8", sun_b="#ffb35a",
    cloud="#ffffff", star=False, dim=1.0)


def _tile_blit(f, tile, off, y):
    """Composite an RGBA tile scrolled horizontally by `off` px (wraps) with its top at row y."""
    h, wt = tile.shape[:2]
    Hf, Wf = f.shape[:2]
    xs = (np.arange(Wf) + int(off)) % wt
    sub = tile[:, xs]
    y0, y1 = max(0, y), min(Hf, y + h)
    if y1 <= y0:
        return
    sub = sub[y0 - y:y1 - y]
    m = sub[..., 3] > 127
    reg = f[y0:y1]
    reg[m] = sub[..., :3][m]


class Arena:
    def __init__(self, theme, seed=11, size=(W, H), gy=GY, far=(24, 58), mid=(40, 84), sun=(196, 92, 32),
                 star_h=82, n_stars=46, girders=(320, (30, 190)), cloud_y=26, missing_y=(60, 40)):
        """`size`/`gy` = frame size and ground line; `far`/`mid` = skyline building height ranges."""
        self.t = theme
        self.w, self.h, self.gy = size[0], size[1], gy
        self.mid_hmax, self.sun_cfg, self.cloud_y, self.missing_y = mid[1], sun, cloud_y, missing_y
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.sky = self._sky()
        self.far = self._skyline(640, far[0], far[1], theme.far, theme.far_hi, windows=False, rng=rng)
        self.mid = self._skyline(640, mid[0], mid[1], theme.mid, theme.mid_hi, windows=True, rng=rng)
        self.near = self._girders(girders[0], girders[1])
        self.floor = self._floor(320)
        self.clouds = self._clouds(640, rng)
        self.stars = [(int(rng.integers(0, self.w)), int(rng.integers(2, star_h)), float(rng.random())) for _ in range(n_stars)]

    # ------------------------------------------------------------ static pieces
    def _sky(self):
        f = new_frame(size=(self.w, self.h))
        dither_gradient(f, 0, self.gy + 4, self.t.sky)
        f[self.gy + 4:] = self.t.floor_lo
        return f

    def _skyline(self, width, hmin, hmax, col, hi, windows, rng):
        tile = np.zeros((hmax + 14, width, 4), np.uint8)
        H_t = tile.shape[0]
        x = 0
        self_lights = []
        while x < width:
            bw = int(rng.integers(20, 44))
            if x + bw > width:
                bw = width - x
            bh = int(rng.integers(hmin, hmax))
            top = H_t - 2 - bh
            tile[top:H_t, x:x + bw] = (*col, 255)
            tile[top:top + 1, x:x + bw] = (*hi, 255)
            tile[top:H_t, x + bw - 3:x + bw] = (*[int(c * 0.78) for c in col], 255)
            if windows:
                for wy in range(top + 5, H_t - 4, 5):
                    for wx in range(x + 3, x + bw - 6, 5):
                        if rng.random() < 0.34:
                            c = self.t.win_a if rng.random() < 0.7 else self.t.win_b
                            tile[wy:wy + 2, wx:wx + 3] = (*c, 255)
                if rng.random() < 0.45:      # neon sign
                    c = self.t.neon if rng.random() < 0.5 else self.t.neon2
                    sy = top + int(rng.integers(6, max(8, bh // 2)))
                    tile[sy:sy + 4, x + 3:x + bw - 5] = (*c, 255)
                    tile[sy:sy + 1, x + 3:x + bw - 5] = (255, 255, 255, 255)
                if rng.random() < 0.5:       # antenna
                    ax = x + bw // 2
                    tile[max(0, top - 9):top, ax] = (*hi, 255)
            x += bw
        return tile

    def _girders(self, width, cols):
        h = self.gy - 10
        tile = np.zeros((h, width, 4), np.uint8)
        st, hi, lo = self.t.steel, self.t.steel_hi, self.t.steel_lo
        for cx in cols:
            tile[:, cx:cx + 12] = (*st, 255)
            tile[:, cx:cx + 2] = (*hi, 255)
            tile[:, cx + 10:cx + 12] = (*lo, 255)
            for k in range(6, h - 8, 22):                     # X braces
                for i in range(22):
                    xx = cx + 12 + i * 0 
                # cross plates
                tile[k:k + 3, cx - 2:cx + 14] = (*lo, 255)
                tile[k:k + 1, cx - 2:cx + 14] = (*hi, 255)
        tile[0:9, :] = (*st, 255)                               # cross beam across the top
        tile[0:2, :] = (*hi, 255)
        tile[7:9, :] = (*lo, 255)
        for x in range(4, width, 16):
            tile[9:11, x:x + 8] = (*lo, 255)                    # rivet strip shadow
        return tile

    def _floor(self, width):
        h = self.h - self.gy
        tile = np.zeros((h, width, 4), np.uint8)
        tile[..., 3] = 255
        tile[..., :3] = self.t.floor
        tile[0:2] = (*self.t.edge, 255)
        tile[2:4, :, :3] = self.t.neon
        tile[4:6, :, :3] = self.t.floor_lo
        for x in range(0, width, 32):                           # panel seams
            tile[6:, x:x + 1, :3] = self.t.floor_lo
            tile[6:, x + 1:x + 2, :3] = tuple(min(255, int(c * 1.12)) for c in self.t.floor)
        for y in range(22, 27):                                 # hazard stripe band
            for x in range(width):
                if ((x + y * 2) // 6) % 2 == 0:
                    tile[y, x, :3] = hexc("#ffc93c")
                else:
                    tile[y, x, :3] = hexc("#231a34")
        for yy in range(26, h):                                 # darker toward the bottom (dither)
            k = (yy - 26) / max(1, h - 27)
            for xx in range(width):
                if BAYER4[yy % 4, xx % 4] < k * 0.55:
                    tile[yy, xx, :3] = self.t.floor_lo
        return tile

    def _clouds(self, width, rng):
        h = 30
        tile = np.zeros((h, width, 4), np.uint8)
        x = 0
        while x < width:
            cw = int(rng.integers(40, 110))
            cy = int(rng.integers(4, h - 8))
            for i in range(cw):
                th = int(3 + 2 * math.sin(i / cw * math.pi))
                tile[cy:cy + th, (x + i) % width] = (*self.t.cloud, 255)
            x += cw + int(rng.integers(60, 160))
        return tile

    # ------------------------------------------------------------ dynamic render
    def render(self, f, scroll, t, offs=None, vshift=None, missing=(), sun=0.0, static=0.0, static_seed=0):
        """Draw the whole arena into f. `offs`/`vshift` override the parallax (used for the glitch)."""
        th = self.t
        f[:] = self.sky
        if static > 0:                        # the sky turns into TV static
            r = np.random.default_rng(static_seed)
            n = r.random((self.gy, self.w))
            v = (r.random((self.gy, self.w)) * 200).astype(np.uint8)
            m = n < static
            f[:self.gy][m] = np.stack([v[m]] * 3, axis=1)
        # sun
        sx, sy, sr = self.sun_cfg[0], self.sun_cfg[1] - int(sun * 30), self.sun_cfg[2] + int(sun * 8)
        ys, xs = np.mgrid[max(0, sy - sr):min(self.gy, sy + sr), max(0, sx - sr):sx + sr]
        m = (xs - sx) ** 2 + (ys - sy) ** 2 <= sr * sr
        tt = np.clip((ys - (sy - sr)) / (2 * sr), 0, 1)
        col = np.zeros(ys.shape + (3,), np.uint8)
        for c in range(3):
            col[..., c] = (th.sun_a[c] * (1 - tt) + th.sun_b[c] * tt).astype(np.uint8)
        if not th.star:
            pass
        slit = (ys > sy + 2) & (((ys - sy) // 3) % 2 == 0) & (ys - sy > 0) & (th.star)
        m2 = m & ~slit
        f[ys[m2], xs[m2]] = col[m2]
        if th.star:
            for (x, y, ph) in self.stars:
                if (math.sin(t * 3.0 + ph * 40) > -0.3):
                    f[y, x] = (255, 255, 255) if ph > 0.5 else (200, 190, 255)
        o = offs or [scroll * 0.05, scroll * 0.12, scroll * 0.3, scroll * 0.75, scroll * 1.0]
        v = vshift or [0, 0, 0, 0, 0]
        _tile_blit(f, self.clouds, o[0] + t * 3, self.cloud_y + v[0])
        if 1 in missing:
            _missing(f, self.missing_y[0], self.gy - 8, 16)
        else:
            _tile_blit(f, self.far, o[1], self.gy - 2 - self.far.shape[0] + 2 + v[1])
        if 2 in missing:
            _missing(f, self.missing_y[1], self.gy - 8, 8)
        else:
            _tile_blit(f, self.mid, o[2], self.gy - 2 - self.mid.shape[0] + 2 + v[2])
        # antenna beacons on the mid layer
        for k in range(6):
            bx = int((k * 107 - o[2]) % (self.w + 20)) - 10
            if math.sin(t * 5 + k * 1.7) > 0.4:
                f[self.gy - (self.mid_hmax + 24) + (k * 13) % 30, min(self.w - 1, max(0, bx))] = (255, 60, 60)
        _tile_blit(f, self.near, o[3], 6 + v[3])
        # lamp posts (behind the fighters, move with the floor)
        for k in range(3):
            px = int((k * 160 + 60 - o[4]) % 480) - 80
            if -12 < px < self.w + 12:
                fill_rect(f, px, self.gy - 44, 2, 44, th.pole)
                fill_rect(f, px - 4, self.gy - 47, 10, 3, th.pole)
                fill_rect(f, px - 3, self.gy - 44, 8, 1, th.lamp)
                _glow(f, px + 1, self.gy - 42, 11, th.lamp)
        _tile_blit(f, self.floor, o[4], self.gy + v[4])

    def foreground(self, f, scroll, t):
        """Cables drooping across the top, drawn in front of the fighters (fast parallax)."""
        o = scroll * 1.6
        for seg in range(4):
            x0 = int((seg * 190 - o) % 760) - 200
            for i in range(0, 190):
                x = x0 + i
                if 0 <= x < self.w:
                    y = 2 + int(14 * (1 - ((i - 95) / 95) ** 2)) * -1 + 16
                    f[y, x] = self.t.pole
                    if i % 6 == 0:
                        f[y + 1, x] = self.t.steel_lo


def _glow(f, cx, cy, r, c):
    H, W = f.shape[:2]
    y0, y1, x0, x1 = max(0, cy - r), min(H, cy + r), max(0, cx - r), min(W, cx + r)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = (xs - cx) ** 2 + (ys - cy) ** 2
    m = (d < r * r) & (((xs + ys) % 2) == 0) & (d > (r * 0.35) ** 2)
    f[ys[m], xs[m]] = c


def _missing(f, y0, y1, size):
    """Source-engine 'missing texture' look: magenta/black checker."""
    ys, xs = np.mgrid[y0:y1, 0:f.shape[1]]
    chk = ((xs // size) + (ys // size)) % 2 == 0
    f[y0:y1][chk] = (214, 32, 214)
    f[y0:y1][~chk] = (14, 12, 20)


# ==================================================================== debug void
BUG = ["..o.o..",
       ".o...o.",
       "..rrr..",
       ".rkrkr.",
       ".rrkrr.",
       "..rrr.."]
BUG_PAL = {"o": (250, 250, 250), "r": (140, 255, 80), "k": (20, 40, 20)}


def _butterfly(frame, col):
    a = np.zeros((6, 9, 4), np.uint8)
    if frame == 0:   # wings up
        pts = [(0, 1), (1, 0), (1, 1), (2, 1), (2, 2), (0, 7), (1, 8), (1, 7), (2, 7), (2, 6), (3, 3), (3, 5)]
    else:            # wings flat
        pts = [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 1), (2, 8), (2, 7), (3, 8), (3, 7), (3, 6), (4, 7)]
    for (y, x) in pts:
        a[y, x] = (*col, 255)
    for y in range(2, 6):
        a[y, 4] = (40, 30, 60, 255)
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)     # 2x so the payoff is readable


class Void:
    def __init__(self, seed=5, size=(W, H), gy=GY):
        self.w, self.h, self.gy = size[0], size[1], gy
        rng = np.random.default_rng(seed)
        self.cols = [(int(x), float(rng.uniform(0.6, 1.6)), float(rng.random()) * 300) for x in range(6, self.w, 11)]
        self.bug = sprite_from_ascii(BUG, BUG_PAL)
        self.bugs = [(float(rng.uniform(30, self.w - 30)), int(rng.integers(gy + 10, gy + 32)), float(rng.uniform(-14, 14)))
                     for _ in range(6)]
        self.bf = [_butterfly(0, (89, 230, 255)), _butterfly(1, (89, 230, 255)),
                   _butterfly(0, (255, 125, 176)), _butterfly(1, (255, 125, 176)),
                   _butterfly(0, (255, 210, 90)), _butterfly(1, (255, 210, 90))]

    def render(self, f, t, fixed=0.0):
        f[:] = (10, 7, 22)
        # faint grid
        for x in range(0, self.w, 16):
            vline(f, x, 0, self.gy, (19, 15, 42))
        for y in range(0, self.gy, 16):
            hline(f, 0, y, self.w, (19, 15, 42))
        # code rain
        for (x, sp, off) in self.cols:
            y = int((t * 38 * sp + off) % (self.gy + 60)) - 30
            for k in range(8):
                yy = y - k * 4
                if 0 <= yy < self.gy:
                    g = max(0, 200 - k * 26)
                    green = (30, g, 90) if fixed < 0.5 else (g, 120 + g // 3, 90)
                    fill_rect(f, x, yy, 2, 3, (green[0] // 3, green[1] // 3, green[2] // 3) if k else (green[0] // 2, green[1] // 2, green[2] // 2))
        # missing-texture floor
        ys, xs = np.mgrid[self.gy:self.h, 0:self.w]
        chk = ((xs // 20) + ((ys - self.gy) // 10)) % 2 == 0
        f[self.gy:self.h][chk] = (170, 28, 170) if fixed < 0.5 else (120, 90, 200)
        f[self.gy:self.h][~chk] = (14, 12, 22)
        hline(f, 0, self.gy, self.w, (255, 120, 255) if fixed < 0.5 else (200, 230, 255))
        hline(f, 0, self.gy - 1, self.w, (80, 30, 100))

    def draw_bugs(self, f, t, scatter=None, butter=None):
        """Crawling bugs; after the fix (`butter` = seconds since compile OK) they become butterflies."""
        for i, (x0, y0, v) in enumerate(self.bugs):
            if butter is None:
                x = (x0 + v * t) % (self.w + 30) - 10
                y = y0
                spr = self.bug
                if int(t * 6 + i) % 2:
                    spr = spr.copy(); spr[0, 2] = 0
                blit(f, spr, x, y, flip=v < 0)
            else:
                x = (x0 + v * (t - butter)) % (self.w + 30) - 10
                y = y0 - butter * (60 + i * 14) + math.sin(butter * 6 + i) * 7
                frame = int(butter * 10 + i) % 2
                blit(f, self.bf[(i % 3) * 2 + frame], x, y)
