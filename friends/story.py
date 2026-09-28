"""Instagram-Story cut: 1080x1920 (9:16), under 60 s.

This is re-staged for portrait rather than cropped: a 180x320 pixel canvas upscaled 6x, taller
parallax city, stacked title cards, a narrower code window, and every caption/HUD element kept
inside Instagram's UI safe zone (250 px at the top, 340 px at the bottom => art rows 42..263).
"""
import math

import numpy as np

from . import fx, props, ui
from .characters import CW, OX, OY, muzzle, sprite
from .font import ADV, draw_text, text_width
from .gfx import blend_rect, blit, fill_rect, hline, new_frame, silhouette
from .scenes import (BYTE_C, CYAN, DIM, GOLD, NULL_C, SHOUT, Film, clamp01, ease_in, ease_out, lerp, seg, smooth)
from .world import DAWN, DUSK, Arena, Void

SW, SH, SSCALE = 180, 320, 6
GY = 200                       # feet baseline
BX, NX = 52, 128               # fighters' standing spots
CX = SW // 2
SAFE_TOP, SAFE_BOTTOM = 42, 263
SPEED = 130.0                  # Byte's shot, px/s
CAPTION_BOX = (6, 212, 168, 46)

ARENA_KW = dict(size=(SW, SH), gy=GY, far=(40, 100), mid=(70, 150), sun=(112, 92, 34), star_h=150, n_stars=64,
                girders=(240, (24, 150)), cloud_y=60, missing_y=(120, 100))


def tc(f, y, s, color, scale=1, shadow=(8, 8, 30), **kw):
    """Centred text."""
    return draw_text(f, (SW - text_width(s, scale)) // 2, y, s, color, scale, shadow, **kw)


def fit_lines(s, max_w, scales=(3, 2, 1), max_lines=2):
    """Largest font scale at which `s` fits in <= max_lines lines; two-line splits are balanced."""
    words = s.split()
    for sc in scales:
        if text_width(s, sc) <= max_w:
            return sc, [s]
        if max_lines >= 2:
            best = None
            for i in range(1, len(words)):
                l, r = " ".join(words[:i]), " ".join(words[i:])
                w = max(text_width(l, sc), text_width(r, sc))
                if best is None or w < best[0]:
                    best = (w, [l, r])
            if best and best[0] <= max_w:
                return sc, best[1]
    return scales[-1], [s]


class StoryFilm(Film):
    def __init__(self, lines, C, D, credit="CLAUDE"):          # noqa - deliberately not calling super().__init__
        self.L, self.C, self.D, self.credit = lines, C, D, credit.upper()
        self.dusk = Arena(DUSK, **ARENA_KW)
        self.dawn = Arena(DAWN, seed=21, **ARENA_KW)
        self.void = Void(size=(SW, SH), gy=GY)
        self.prev = None
        self.expr = {"byte": "neutral", "null": "deadpan"}
        self.end = C["end"]

    # ------------------------------------------------------------------ helpers
    def speaking(self, lid, t):
        if lid not in self.D:
            return False
        s = self.C["L" + lid]
        return s <= t < s + self.D[lid]

    def draw_actor(self, f, kind, S, t):
        x, y = S["x"], S.get("y", 0)
        ground = GY - y
        sw = max(6, 16 - int(y * 0.25))
        blend_rect(f, x - sw // 2, GY - 1, sw, 2, (0, 0, 0), 0.38)
        spr = sprite(kind, stance=S.get("stance", "stand"), arm=S.get("arm", "down"), back=S.get("back"),
                     expr=S.get("expr", "neutral"), phase=S.get("phase", 0), mouth=S.get("mouth", self.mouth(kind, t)),
                     scarf=int(t * 8) % 4, wind=S.get("wind", 0.3), lean=S.get("lean", 0))
        if S.get("white"):
            spr = silhouette(spr, (255, 255, 255))
        flip = S.get("flip", False)
        ox = OX if not flip else CW - 1 - OX
        blit(f, spr, x - ox, ground - OY, flip=flip)
        self.expr[kind] = S.get("expr", "neutral")

    def caption(self, f, t):
        cur = None
        for lid in sorted(self.L):
            s, d = self.C["L" + lid], self.D[lid]
            if s <= t < s + d + 0.30:
                cur = lid
        if cur:
            ln = self.L[cur]
            s, d = self.C["L" + cur], self.D[cur]
            ui.caption(f, ln["speaker"], ln["text"], clamp01((t - s) / (0.85 * d)), self.speaking(cur, t), t,
                       self.expr[ln["speaker"]], box=CAPTION_BOX, cols=21, max_lines=3)

    # ------------------------------------------------------------------ dispatch
    def frame(self, t):
        C = self.C
        if t < C["title_end"]:
            return self._intro(t)
        if t < C["void_off"] + 0.55:
            f = self._arena(t)
            if t >= C["void_off"]:
                fx.crt_off(f, (t - C["void_off"]) / 0.55)
            return f
        if t < C["void_on"]:
            return new_frame(size=(SW, SH))
        if t < C["reboot"]:
            f = self._debug(t)
            if t < C["void_on"] + 0.35:
                fx.crt_on(f, (t - C["void_on"]) / 0.35)
            return f
        if t < C["credits"]:
            return self._ending(t)
        return self._credits(t)

    # ------------------------------------------------------------------ title card
    def _intro(self, t):
        C = self.C
        TE = C["title_end"]
        f = new_frame(size=(SW, SH))
        self.dusk.render(f, 18 * t, t)
        fx.flash(f, 0.52 * (1 - seg(t, TE - 0.3, TE)), (8, 6, 22))
        exit_ = ease_in(seg(t, TE - 0.22, TE))
        yo = int(-110 * exit_)

        def runner(kind, x0, x1, t0, t1, flip):
            u = seg(t, t0, t1)
            x = lerp(x0, x1, ease_out(u))
            expr = "angry" if kind == "byte" else "deadpan"
            if u < 1:
                return dict(x=x, stance="run", phase=int(t * 12) % 2, arm="aim", expr=expr, flip=flip, lean=1, wind=1.0, mouth="flat")
            return dict(x=x, stance="stand", arm="aim", expr=expr, flip=flip, wind=0.5, mouth="flat")

        self.draw_actor(f, "byte", runner("byte", -30, 44, 0.35, 0.95, False), t)
        self.draw_actor(f, "null", runner("null", 210, 136, 0.6, 1.2, True), t)

        def word(s, y, scale, col, t0, step=0.06):
            w = text_width(s, scale)
            x0 = (SW - w) // 2
            for i, ch in enumerate(s):
                u = seg(t, t0 + i * step, t0 + i * step + 0.28)
                if u <= 0:
                    continue
                dy = -90 * (1 - ease_out(u)) ** 2 + yo
                draw_text(f, x0 + i * ADV * scale, int(y + dy), ch, col, scale, (8, 8, 40))

        word("BYTE", 50, 4, BYTE_C, 0.30)
        word("NULL", 104, 4, NULL_C, 0.62)
        if t > 1.0:
            draw_text(f, (SW - text_width("VS", 3)) // 2, 82 + yo, "VS", GOLD, 3, (8, 8, 40))
            fx.flash(f, 0.55 * (1 - seg(t, 1.0, 1.2)))
            if t < 1.25:
                fx.shake(f, int(3 * (1 - seg(t, 1.0, 1.25))) * (1 if int(t * 40) % 2 else -1), 0)
        tag = [("TWO RIVALS. ONE BUG.", (255, 255, 255), 1.2, 140), ("CAN THEY FIX IT...", GOLD, 1.65, 151),
               ("TOGETHER?", GOLD, 2.0, 161)]
        for (s, col, t0, y) in tag:
            draw_text(f, (SW - text_width(s)) // 2, y + yo, s, col, 1, (8, 8, 40), limit=int(max(0, t - t0) * 34))
        # progress row + credits, on the (free) floor area
        if t > 1.7:
            rows = [[("FIGHT", 1.75), (">", None), ("CRASH", 1.95)], [("DEBUG", 2.15), (">", None), ("BEFRIEND", 2.35)]]
            for r, parts in enumerate(rows):
                tot = sum(text_width(w) for w, _ in parts) + ADV * 2
                x = (SW - tot) // 2
                yy = 212 + r * 11
                for (w, ta) in parts:
                    if ta is None:
                        draw_text(f, x, yy, w, DIM, 1, (8, 8, 40)); x += 2 * ADV
                        continue
                    lit = t > ta
                    draw_text(f, x, yy, w, (255, 255, 255) if lit else (90, 98, 140), 1, (8, 8, 40))
                    if lit:
                        hline(f, x, yy + 9, text_width(w), GOLD)
                    x += text_width(w) + ADV
            a = seg(t, 1.9, 2.2)
            tc(f, 240, "MADE WITH " + self.credit, tuple(int(v * a) for v in (200, 210, 255)))
            tc(f, 251, "VOICES: ELEVENLABS ELEVEN V4", tuple(int(v * a) for v in (150, 160, 215)))
        fx.crt_on(f, t / 0.3)
        if t > TE - 0.05:
            fx.flash(f, 0.8)
        return f

    # ------------------------------------------------------------------ boss fight + crash
    def _byte_state(self, t):
        C = self.C
        S = dict(x=BX, y=0, stance="stand", arm="aim", expr="angry", flip=False, wind=0.3)
        CL, FZ = C["clash"], C["freeze"]
        if t < C["L05"]:
            if C["fire"] <= t < C["fire"] + 0.16:
                S["lean"] = -2
            if t >= C["L04"] + 0.5:
                S["expr"] = "worried"
        elif t < C["leap"]:
            S.update(arm="raise", back="raise", expr="angry", lean=0, y=int(1.5 * math.sin(t * 40)))
        elif t < CL:
            u = (t - C["leap"]) / (CL - C["leap"])
            S.update(x=lerp(BX, 78, smooth(u)), y=24 * math.sin(math.pi * u * 0.75), stance="jump", arm="aim", expr="angry")
        else:
            v = seg(t, CL, FZ)
            S.update(x=lerp(78, 30, ease_out(v)), y=lerp(17, 30, ease_out(v)), stance="hurt", arm="hit", back="hit", lean=-2,
                     expr="ko", white=(t - CL) < 0.07)
            if t >= CL + 1.0:
                S["expr"] = "worried"
            if t >= C["L08"] + 1.4:
                S["expr"] = "wide"
        return S

    def _null_state(self, t):
        C = self.C
        S = dict(x=NX, y=0, stance="stand", arm="cross", expr="deadpan", flip=True, wind=0.3)
        CL, FZ = C["clash"], C["freeze"]
        arrive = C["fire"] + (NX - (BX + 17)) / SPEED
        if C["L02"] + 0.25 <= t < C["L03"]:
            S["arm"] = "shrug"
        if arrive - 0.28 <= t < arrive + 0.28:
            u = (t - (arrive - 0.28)) / 0.56
            S.update(y=22 * 4 * u * (1 - u), stance="jump", arm="cross", wind=0.6)
        if C["L06"] - 0.15 <= t < C["leap"]:
            S["arm"] = "aim"
        if C["leap"] <= t < CL:
            u = (t - C["leap"]) / (CL - C["leap"])
            S.update(x=lerp(NX, 102, smooth(u)), y=24 * math.sin(math.pi * u * 0.75), stance="jump", arm="aim", wind=1.0)
        if t >= CL:
            v = seg(t, CL, FZ)
            S.update(x=lerp(102, 150, ease_out(v)), y=lerp(17, 30, ease_out(v)), stance="hurt", arm="hit", back="hit", lean=-2,
                     expr="ko", white=(t - CL) < 0.07, wind=0.8)
            if t >= CL + 1.0:
                S.update(stance="fall", arm="cross", back=None, expr="deadpan", lean=0)
        return S

    def _arena(self, t):
        C = self.C
        CL, FZ, TE = C["clash"], C["freeze"], C["title_end"]
        f = new_frame(size=(SW, SH))
        scroll = 46.0 * (min(t, CL) - 1.0)
        tt = min(t, FZ)
        k = clamp01((t - FZ) / 1.8) if t >= FZ else 0.0
        offs = vs = None
        missing, static = (), 0.0
        if t >= FZ:
            ph = int(t * 9)
            r = np.random.default_rng(ph + 7)
            base = [scroll * m for m in (0.05, 0.12, 0.3, 0.75, 1.0)]
            offs = [b + float(r.integers(-60, 60)) * k for b in base]
            vs = [int(r.integers(-9, 10) * k) for _ in range(5)]
            if k > 0.3 and ph % 4 == 0:
                missing = (1,)
            elif k > 0.45 and ph % 7 == 3:
                missing = (2,)
            static = clamp01((t - (FZ + 0.6)) / 1.1) * 0.7
        self.dusk.render(f, scroll, tt, offs, vs, missing, static=static, static_seed=int(t * 30))
        if k > 0:
            fx.smear(f, self.prev, 0.28 * k)
            self.prev = f.copy()
            fx.glitch(f, min(0.8, 0.2 + 0.6 * k), seed=int(t * 24))
        else:
            self.prev = None

        B, N = self._byte_state(t), self._null_state(t)
        if t < C["fight_start"]:                        # beam-in
            appear = C["beam_in"] + 0.32
            for (x, col) in ((BX, (120, 235, 255)), (NX, (255, 110, 130))):
                u = seg(t, C["beam_in"], C["beam_in"] + 0.28)
                if 0 < u < 1:
                    y1 = int(GY * u)
                    fill_rect(f, x - 3, max(0, y1 - 40), 7, min(40, y1), col)
                    fill_rect(f, x - 1, max(0, y1 - 40), 3, min(40, y1), (255, 255, 255))
            if t >= appear:
                B.update(white=(t - appear) < 0.09)
                N.update(white=(t - appear) < 0.09)
                self.draw_actor(f, "byte", B, t)
                self.draw_actor(f, "null", N, t)
                fx.flash(f, 0.6 * (1 - seg(t, appear, appear + 0.16)))
        else:
            self._battle_fx(f, t, B, N)
            self.draw_actor(f, "null", N, t)
            self.draw_actor(f, "byte", B, t)
            self._battle_front(f, t, B, N)
        self.dusk.foreground(f, scroll, tt)

        if TE <= t < C["fight_start"] - 0.05:            # STAGE 8 / FINAL BOSS / READY
            xo1 = int((1 - ease_out(seg(t, TE, TE + 0.25))) * -220)
            xo2 = int((1 - ease_out(seg(t, TE + 0.1, TE + 0.35))) * 220)
            draw_text(f, (SW - text_width("STAGE 8", 3)) // 2 + xo1, 78, "STAGE 8", (255, 255, 255), 3, (8, 8, 40))
            draw_text(f, (SW - text_width("FINAL BOSS", 2)) // 2 + xo2, 106, "FINAL BOSS", (255, 90, 110), 2, (8, 8, 40))
            if t >= C["beam_in"] + 0.3 and int((t - C["beam_in"] - 0.3) * 8) % 2 == 0:
                draw_text(f, (SW - text_width("READY", 2)) // 2, 132, "READY", GOLD, 2, (8, 8, 40))
        if t < TE + 0.2:
            fx.flash(f, 0.5 * (1 - seg(t, TE, TE + 0.2)))

        if CL <= t < CL + 0.55:                          # clash shake / white-out
            u = (t - CL) / 0.55
            fx.shake(f, int(5 * (1 - u)) * (1 if int(t * 60) % 2 else -1), int(2 * (1 - u)) * (-1 if int(t * 45) % 2 else 1))
            fx.flash(f, max(0.0, 1.0 - u * 2.4))
        if k > 0:
            fx.glitch(f, 0.10 + 0.10 * k, seed=int(t * 24) + 1)
        if t >= C["fight_start"] - 0.4 and t < C["void_off"]:
            self._hud_and_text(f, t)
            self.caption(f, t)
        return f

    def _battle_fx(self, f, t, B, N):
        C = self.C
        CL, fire = C["clash"], C["fire"]
        if C["L03"] + 0.3 <= t < fire:                   # charging orb
            r = 1 + 6 * seg(t, C["L03"] + 0.3, fire)
            mx, my = muzzle("byte", "stand", 0)
            fx.circle(f, B["x"] + mx, GY + my, r, (120, 240, 255), fill=True)
            fx.circle(f, B["x"] + mx, GY + my, max(1, r - 3), (255, 255, 255), fill=True)
        if fire <= t < C["L05"] + 0.9:                   # the shot
            px = BX + 17 + SPEED * (t - fire)
            if px < SW + 12:
                fx.trail(f, px, GY - 14, 1, "byte")
                fx.pellet(f, px, GY - 14, "byte", t, 1)
                fx.circle(f, px, GY - 14, 6, (120, 240, 255))
        if fire <= t < fire + 0.2:
            mx, my = muzzle("byte", "stand", 0)
            fx.starburst(f, BX + 19, GY + my, 8, (120, 240, 255), spikes=6)
        if C["L05"] <= t < C["leap"]:
            fx.aura(f, B["x"], GY - 16, t, (90, 220, 255), 1.0, 24)
            fx.speedlines(f, t, (170, 230, 255), 10, 0.4)
            fx.shake(f, int(math.sin(t * 90)), 0)
        if C["L06"] - 0.1 <= t < C["leap"]:
            fx.aura(f, N["x"], GY - 16, t, (170, 40, 70), 0.35, 14)
        if C["leap"] <= t < CL:
            fx.speedlines(f, t, (255, 255, 255), 8, 0.5)

    def _battle_front(self, f, t, B, N):
        C, CL, fire = self.C, self.C["clash"], self.C["fire"]
        if C["L03"] + 0.2 <= t < fire + 1.0:             # attack-name callout (two lines)
            col = GOLD if int(t * 12) % 2 else (255, 255, 255)
            tc(f, 66, "MEGA-BYTE", col, 2, (60, 20, 10))
            tc(f, 84, "BUSTER!", col, 2, (60, 20, 10))
        if C["L04"] + 0.6 <= t < C["L05"]:
            ui.sweat(f, B["x"] + 8, GY - 40, t)
        if CL <= t < CL + 0.7:
            u = (t - CL) / 0.7
            fx.starburst(f, CX, GY - 34 - 10 * u, 10 + 56 * ease_out(u), (255, 214, 90), (255, 255, 255), 10, u * 0.8)
            fx.burst(f, CX, GY - 36, t - CL, 34, 120, [(255, 255, 255), (120, 240, 255), (255, 90, 110), (255, 220, 90)], 0.8, 2, 80, 3)

    def _hud_and_text(self, f, t):
        C = self.C
        CL = C["clash"]
        fill = seg(t, C["beam_in"] + 0.5, C["beam_in"] + 1.0)
        hb, hn = (0.36 * fill, 0.30 * fill) if t < CL else (0.0, 0.0)
        blinkb = hb < 0.3 and int(t * 8) % 2 == 0
        blinkn = hn < 0.3 and int(t * 8) % 2 == 0
        draw_text(f, 8, 46, "BYTE", (255, 255, 255), shadow=(0, 0, 0))
        ui.hp_bar(f, 10, 58, hb, BYTE_C, blink=blinkb, n=15, step=5)
        draw_text(f, SW - 8 - text_width("NULL"), 46, "NULL", (255, 255, 255), shadow=(0, 0, 0))
        ui.hp_bar(f, SW - 10 - 15 * 5 + 1, 58, hn, NULL_C, flip=True, blink=blinkn, n=15, step=5)
        if t >= CL + 0.35:                                # who won?
            u = t - (CL + 0.35)
            if u < 0.6:
                if int(t * 12) % 2 == 0:
                    tc(f, 96, "K.O.!", (255, 255, 255), 4, (200, 30, 60))
            else:
                per = max(0.06, 0.26 - 0.07 * (u - 0.6))
                which = int((u - 0.6) / per) % 2
                blend_rect(f, 10, 88, SW - 20, 26, (0, 0, 0), 0.45)
                if u > 2.6 and int(t * 20) % 3 == 0:
                    tc(f, 94, "BYTE WINS!", BYTE_C, 2)
                    draw_text(f, (SW - text_width("NULL WINS!", 2)) // 2 + 2, 96, "NULL WINS!", NULL_C, 2, (8, 8, 40))
                elif which == 0:
                    tc(f, 94, "BYTE WINS!", BYTE_C, 2)
                else:
                    tc(f, 94, "NULL WINS!", NULL_C, 2)

    # ------------------------------------------------------------------ the desk (debug void)
    def _terminal_state(self, t):
        """Rows: 0 run, 1 hang, 2 line 420, 3 line 421 (the fix), 4 line 422, 5 build, 6 result (<= 26 chars each)."""
        C = self.C
        rows = [("> RUN STAGE_8.GAME", (190, 255, 200)),
                ("!! HANG AT LINE 420 !!", (255, 110, 120)),
                ("420 IF (BOTH_HP == 0) {", (190, 255, 200)),
                ["421   ", (150, 170, 215)],
                ("422 }", (190, 255, 200)),
                ("", (0, 0, 0)),
                ("", (0, 0, 0))]
        todo = "// TODO: WHO WINS?"
        typed = C["type_chars"]
        hl, flash = None, None
        if t < C["type0"]:
            rows[3][0] = "421   " + todo
            if C["L13"] <= t < C["L15"]:
                hl = 3
        elif t < C["type0"] + C["erase_dur"]:
            u = seg(t, C["type0"], C["type0"] + C["erase_dur"])
            rows[3][0] = "421   " + todo[: int(len(todo) * (1 - u))]
            hl = 3
        else:
            n = min(len(typed), int((t - C["type0"] - C["erase_dur"]) * C["cps"]))
            semi = ";" if t >= C["semi"] else ""
            rows[3][0] = "421   " + typed[:n] + (semi if n == len(typed) else "")
            hl = 3 if t < C["ok"] + 0.2 else None
        rows[3] = (rows[3][0], (255, 255, 255) if t >= C["type0"] else (150, 170, 215))
        if C["type_end"] + 0.08 <= t:
            rows[5] = ("> BUILD...", (190, 255, 200))
        if C["err"] <= t < C["semi"]:
            rows[6] = ("ERROR: EXPECTED ';' AT 421", (255, 100, 110))
            flash = 6 if t < C["err"] + 0.7 else None
        if t >= C["ok"]:
            rows[6] = ("BUILD OK! 0 ERRORS 0 BUGS", (120, 255, 150))
            rows[1] = ("OK: NO MORE HANGS", (120, 255, 150))
        cursor = (3, len(rows[3][0])) if (C["L13"] <= t < C["ok"] + 0.4) else None
        return rows, cursor, hl, flash

    def _debug(self, t):
        C = self.C
        f = new_frame(size=(SW, SH))
        fixed = seg(t, C["ok"], C["ok"] + 0.8)
        self.void.render(f, t, fixed)
        rows, cursor, hl, flash = self._terminal_state(t)

        NXS, BXS, STOOL = 50, 22, 5
        DESK_X0, DESK_X1, DESK_TOP = 56, 176, GY - 11
        MON_X, MON_Y = 108, DESK_TOP - 29
        land_n = C["land_n"]
        seat = 4
        stool_p = seg(t, C["land_b"] - 0.12, C["land_b"] + 0.20)
        chair_p = seg(t, land_n - 0.15, land_n + 0.25)
        desk_p = seg(t, land_n + 0.10, land_n + 0.60)
        win_on = t >= land_n + 0.55
        state = "ok" if t >= C["ok"] else ("error" if C["err"] <= t < C["semi"] else "normal")
        jolt = int(2 * math.sin(t * 70)) if self.speaking("14", t) else 0

        if win_on:
            props.hologram(f, MON_X, MON_X + 46, MON_Y, 4 + jolt, 176 + jolt, 130, t)
            ui.terminal(f, 4 + jolt, 50, 172, 80, "DEBUG.EXE - STAGE_8", rows, t, cursor, hl, flash_row=flash,
                        pitch=9, top=14)
        self.void.draw_bugs(f, t, butter=(t - C["ok"]) if t >= C["ok"] else None)

        if stool_p > 0:
            g = f.copy(); props.stool(g, BXS, GY, STOOL); props.reveal(f, g, stool_p)
        if chair_p > 0:
            g = f.copy(); props.chair(g, NXS, GY); props.reveal(f, g, chair_p)

        B = dict(x=BXS, y=STOOL, stance="stand", arm="down", expr="neutral", flip=False)
        N = dict(x=NXS, y=seat, stance="sit", arm="reach", expr="deadpan", flip=False, wind=0.15)
        for (S, land, base) in ((B, C["land_b"], STOOL), (N, land_n, seat)):
            if t < land:
                u = seg(t, C["void_on"] - 0.05, land)
                S.update(y=base + 196 * (1 - u * u), stance="fall", arm="raise", back="hit", expr="wide", mouth="shout")
            elif t < land + 0.22:
                S["y"] = base + 4 * (1 - (t - land) / 0.22) * abs(math.sin((t - land) * 22))
        for (land, x) in ((C["land_b"], BXS), (land_n, NXS)):
            fx.burst(f, x, GY - 2, t - land, 12, 40, [(190, 170, 210), (120, 100, 150)], 0.5, 2, 30, 9)

        typing = C["type0"] <= t < C["type_end"] or C["semi"] <= t < C["semi"] + 0.25
        if t >= land_n + 0.25:
            B.update(arm="down", expr="worried" if int(t * 2) % 5 else "blink")
            if self.speaking("11", t):
                B["expr"] = "neutral"
            if C["L13"] - 0.2 <= t < C["L14"]:
                B.update(arm="chin", expr="neutral", lean=3)
                N["lean"] = 1
            if self.speaking("14", t):
                B.update(arm="raise", back="raise", expr="angry", lean=0, y=STOOL + (1 if int(t * 12) % 2 else 0))
                N.update(expr="wide", y=seat + 1)
                ui.bang(f, BXS + 4, GY - STOOL - 46, t)
            elif C["L15"] - 0.02 <= t < C["L16"]:
                B.update(arm="chin", expr="worried", lean=3)
                N["y"] = seat + (1 if int(t * 6) % 2 else 0)
            if C["L16"] <= t < C["type0"]:
                B.update(arm="point", lean=3, expr="wide" if t < C["L16"] + 0.7 else "happy")
                ui.bulb(f, BXS + 2, GY - STOOL - 44, t)
            if C["L17"] <= t < C["type0"]:
                N.update(expr="wide" if t < C["L17"] + 0.7 else "happy", arm="cross")
            if C["type0"] <= t < C["ok"] + 0.3:
                N.update(arm="reach", expr="deadpan", lean=1)
                B.update(arm="chin", expr="happy" if t < C["err"] else "wide", lean=3)
            if typing:
                N["y"] = seat + (1 if int(t * C["cps"]) % 2 else 0)
            if C["err"] <= t < C["L19"]:
                B.update(arm="chin", expr="wide", lean=3)
            if self.speaking("19", t):
                B.update(expr="happy", arm="down", y=STOOL + (1 if int(t * 9) % 2 else 0), lean=2)
                N.update(expr="happy", arm="cross")
            if C["semi"] <= t < C["ok"] + 0.3:
                B.update(arm="chin", expr="happy", lean=3)
            if t >= C["ok"] + 0.3:
                B.update(arm="raise", back="raise", expr="happy", y=STOOL + (1 if int(t * 7) % 2 else 0), lean=0)
                N.update(arm="raise", back=None, expr="happy", lean=0)

        self.draw_actor(f, "null", N, t)
        self.draw_actor(f, "byte", B, t)

        if desk_p > 0:
            g = f.copy()
            props.desk(g, DESK_X0, DESK_X1, DESK_TOP)
            props.keyboard(g, 62, DESK_TOP - 3)
            props.monitor(g, MON_X, MON_Y, rows, t, state, hl, cursor)
            props.mug(g, 164, DESK_TOP - 6, t)
            props.reveal(f, g, desk_p)

        if C["ok"] <= t < C["ok"] + 0.5:
            fx.burst(f, MON_X + 23, MON_Y, t - C["ok"], 40, 110, [(120, 255, 150), (255, 255, 255), (255, 216, 74)], 0.8, 2, -20, 4)
        if t >= C["ok"]:
            fx.flash(f, 0.5 * (1 - seg(t, C["ok"], C["ok"] + 0.25)), (200, 255, 210))
        if t >= C["reboot"] - 0.3:
            fx.flash(f, seg(t, C["reboot"] - 0.3, C["reboot"]))
        self.caption(f, t)
        return f

    # ------------------------------------------------------------------ repaired world + credits
    def _ending(self, t):
        C = self.C
        f = new_frame(size=(SW, SH))
        sun = seg(t, C["reboot"], C["credits"])
        scroll = 24.0 * (t - C["reboot"])
        self.dawn.render(f, scroll, t, sun=sun * 0.7)
        bx0, nx0 = 58, 122
        u = ease_out(seg(t, C["L21"] + self.D["21"] + 0.05, C["bump"]))
        B = dict(x=lerp(bx0, 78, u), stance="stand", arm="down", expr="happy", flip=False, wind=0.3)
        N = dict(x=lerp(nx0, 102, u), stance="stand", arm="cross", expr="deadpan", flip=True, wind=0.4)
        if self.speaking("21", t):
            N["expr"] = "happy" if t < C["L21"] + 0.6 else "smug"
            B["expr"] = "happy"
        if t >= C["L21"] + self.D["21"]:
            N["expr"] = "happy"
        if t >= C["bump"] - 0.3:
            B.update(arm="fist"); N.update(arm="fist", back=None)
        self.draw_actor(f, "null", N, t)
        self.draw_actor(f, "byte", B, t)
        self.dawn.foreground(f, scroll, t)
        if t >= C["bump"]:
            tau = t - C["bump"]
            if tau < 0.45:
                fx.starburst(f, CX, GY - 24, 4 + 26 * ease_out(tau / 0.35), (255, 240, 150), (255, 255, 255), 8, tau * 2)
            fx.burst(f, CX, GY - 24, tau, 30, 80, [(255, 240, 150), (255, 255, 255), (255, 150, 200), (130, 240, 255)], 1.1, 2, -10, 6)
            for i in range(5):
                hx = CX + math.sin(tau * 3 + i * 1.3) * 22 + (i - 2) * 7
                hy = GY - 40 - tau * (18 + i * 5)
                if 0 < tau < 1.5 and hy > 50:
                    draw_text(f, int(hx), int(hy), "♥", (255, 110, 150), 1, (60, 10, 30))
        if t < C["reboot"] + 0.45:
            fx.flash(f, 1 - seg(t, C["reboot"], C["reboot"] + 0.45))
        if C["join"] <= t < C["credits"] - 0.12:
            u = ease_out(seg(t, C["join"], C["join"] + 0.25))
            xo = int((1 - u) * -SW)
            blend_rect(f, 0, 96, SW, 44, (10, 12, 40), 0.74)
            hline(f, 0, 96, SW, GOLD); hline(f, 0, 139, SW, GOLD)
            col = GOLD if int(t * 6) % 2 == 0 else (255, 255, 255)
            draw_text(f, (SW - text_width("PLAYER 2", 2)) // 2 + xo, 104, "PLAYER 2", col, 2, (60, 30, 0))
            draw_text(f, (SW - text_width("HAS JOINED!", 2)) // 2 + xo, 121, "HAS JOINED!", col, 2, (60, 30, 0))
        self.caption(f, t)
        return f

    def _credits(self, t):
        C = self.C
        f = self._ending(t)
        u = seg(t, C["credits"], C["credits"] + 0.35)
        if u > 0:
            blend_rect(f, 0, SAFE_TOP, SW, 108, (8, 10, 34), 0.72 * u)       # title band
            blend_rect(f, 0, 204, SW, 58, (8, 10, 34), 0.72 * u)             # credits band (on the floor)
            tc(f, 50, "BYTE", GOLD, 4)
            tc(f, 82, "& NULL", GOLD, 4)
            tc(f, 118, "2-PLAYER CO-OP", (255, 255, 255), 2)
            if int(t * 2.4) % 2 == 0:
                tc(f, 138, "PRESS START", CYAN)
            tc(f, 208, "MADE WITH", (170, 180, 230))
            sc, lines = fit_lines(self.credit, SW - 12, scales=(3, 2), max_lines=2)
            y = 218
            for l in lines:
                tc(f, y, l, GOLD, sc)
                y += 7 * sc + 3
            tc(f, 250, "VOICES: ELEVENLABS ELEVEN V4", (200, 210, 255))
        if t > C["credits"] + 1.4:
            tc(f, 158, "(IT WAS A SEMICOLON.)", (170, 180, 225))
        fade = seg(t, self.end - 0.5, self.end)
        if fade > 0:
            fx.flash(f, fade, (0, 0, 0))
        return f

    # ------------------------------------------------------------------ sound cues
    def events(self):
        C = self.C
        ev = [(C["beam_in"], "dash", 0.7), (C["beam_in"] + 0.32, "land", 0.9), (C["title_end"], "confirm", 0.6),
              (0.35, "dash", 0.5), (0.6, "dash", 0.5), (1.0, "clash", 0.55)]
        ev += [(C["fire"], "shoot_byte", 0.9), (C["fire"] + (NX - BX - 17) / SPEED - 0.28, "jump", 0.7),
               (C["fire"] + (NX - BX - 17) / SPEED + 0.28, "land", 0.6),
               (C["L05"], "charge", 0.9), (C["leap"], "jump", 0.8), (C["leap"] + 0.02, "jump", 0.8),
               (C["clash"], "clash", 1.0), (C["clash"], "hit", 0.9), (C["clash"] + 0.02, "crash", 1.0)]
        for dt in (1.0, 1.75, 2.4):
            ev.append((C["clash"] + dt, "glitch", 0.55))
        ev += [(C["land_b"], "land", 0.9), (C["land_n"], "land", 0.9),
               (C["land_n"] + 0.12, "bleep", 0.55), (C["land_n"] + 0.34, "bleep", 0.5), (C["land_n"] + 0.56, "confirm", 0.35),
               (C["void_off"], "dash", 0.6)]
        n = len(C["type_chars"])
        for i in range(int(C["erase_dur"] * 30)):
            ev.append((C["type0"] + i / 30.0, "click", 0.35))
        for i in range(n):
            ev.append((C["type0"] + C["erase_dur"] + i / C["cps"], "click", 0.5))
        ev += [(C["err"], "error", 0.8), (C["semi"], "click", 0.6), (C["ok"], "confirm", 0.9),
               (C["reboot"] - 0.1, "dash", 0.6), (C["join"], "join", 0.9)]
        return sorted(ev)
