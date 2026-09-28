"""The film itself: intro -> boss fight -> crash -> debug void -> repaired world -> credits.
`Film.frame(t)` renders any frame on its own (320x180); `Film.events()` lists the sound cues."""
import math

import numpy as np

from . import fx, props, ui
from .characters import CW, OX, OY, muzzle, sprite
from .font import ADV, draw_text, text_width
from .gfx import H, W, blend_rect, blit, fill_rect, hexc, hline, new_frame, silhouette
from .timeline import BEAM_IN, FIGHT_START, TITLE_END
from .world import DAWN, DUSK, GY, Arena, Void

BX, NX = 110, 210                                   # fighters' standing positions
SHOUT = {"01", "03", "05", "14"}
GOLD, CYAN, PINK = (255, 216, 74), (110, 232, 255), (255, 93, 162)
BYTE_C, NULL_C = (74, 168, 255), (255, 90, 114)
DIM = (120, 128, 170)


def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def lerp(a, b, u):
    return a + (b - a) * u


def ease_out(u):
    u = clamp01(u)
    return 1 - (1 - u) ** 3


def ease_in(u):
    u = clamp01(u)
    return u * u * u


def smooth(u):
    u = clamp01(u)
    return u * u * (3 - 2 * u)


def seg(t, a, b):
    return clamp01((t - a) / (b - a))


def text_c(f, y, s, color, scale=1, **kw):
    x = (W - text_width(s, scale)) // 2
    return draw_text(f, x, y, s, color, scale, kw.pop("shadow", (8, 8, 30)), **kw)


class Film:
    def __init__(self, lines, C, D, credit="CLAUDE"):
        self.L, self.C, self.D, self.credit = lines, C, D, credit.upper()
        self.dusk, self.dawn, self.void = Arena(DUSK), Arena(DAWN, seed=21), Void()
        self.prev = None
        self.expr = {"byte": "neutral", "null": "deadpan"}
        self.end = C["end"]

    # ------------------------------------------------------------------ helpers
    def speaking(self, lid, t):
        s = self.C["L" + lid]
        return s <= t < s + self.D[lid]

    def mouth(self, kind, t):
        for lid, ln in self.L.items():
            if ln["speaker"] == kind and self.speaking(lid, t):
                on = int(t * 11) % 2 == 0
                m = "shout" if lid in SHOUT else "open"
                if kind == "byte":
                    return m if on else "flat"
                return m if on else None
        return None

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
            prog = clamp01((t - s) / (0.85 * d))
            ui.caption(f, ln["speaker"], ln["text"], prog, self.speaking(cur, t), t, self.expr[ln["speaker"]])

    # ------------------------------------------------------------------ dispatch
    def frame(self, t):
        C = self.C
        if t < TITLE_END:
            f = self._intro(t)
        elif t < C["void_off"] + 0.55:
            f = self._arena(t)
            if t >= C["void_off"]:
                fx.crt_off(f, (t - C["void_off"]) / 0.55)
        elif t < C["void_on"]:
            f = new_frame()
        elif t < C["reboot"]:
            f = self._debug(t)
            if t < C["void_on"] + 0.35:
                fx.crt_on(f, (t - C["void_on"]) / 0.35)
        elif t < C["credits"]:
            f = self._ending(t)
        else:
            f = self._credits(t)
        return f

    # ------------------------------------------------------------------ intro / title card
    def _intro(self, t):
        f = new_frame()
        self.dusk.render(f, 22 * t, t)
        fx.flash(f, 0.52 * (1 - seg(t, TITLE_END - 0.3, TITLE_END)), (8, 6, 22))
        exit_ = ease_in(seg(t, TITLE_END - 0.22, TITLE_END))         # title slides away before the cut

        def runner(kind, x0, x1, t0, t1, flip):
            u = seg(t, t0, t1)
            x = lerp(x0, x1, ease_out(u))
            expr = "angry" if kind == "byte" else "deadpan"
            if u < 1:
                return dict(x=x, stance="run", phase=int(t * 12) % 2, arm="aim", expr=expr, flip=flip, lean=1, wind=1.0, mouth="flat")
            return dict(x=x, stance="stand", arm="aim", expr=expr, flip=flip, wind=0.5, mouth="flat")

        self.draw_actor(f, "byte", runner("byte", -30, 78, 0.55, 1.15, False), t)
        self.draw_actor(f, "null", runner("null", 350, 242, 0.85, 1.45, True), t)

        def word(s, x, y, scale, col, t0, drop=80, step=0.07):
            for i, ch in enumerate(s):
                u = seg(t, t0 + i * step, t0 + i * step + 0.30)
                if u <= 0:
                    continue
                yo = -drop * (1 - ease_out(u)) ** 2 - 90 * exit_
                draw_text(f, x + i * ADV * scale, int(y + yo), ch, col, scale, (8, 8, 40))

        word("BYTE", 41, 24, 4, BYTE_C, 0.55)
        word("NULL", 41 + 92 + 10 + 33 + 10, 24, 4, NULL_C, 0.95)
        if t > 1.35:
            yo = int(-90 * exit_)
            draw_text(f, 41 + 92 + 10, 27 + yo, "VS", GOLD, 3, (8, 8, 40))
            fx.flash(f, 0.55 * (1 - seg(t, 1.35, 1.55)), (255, 255, 255))
        fx.shake(f, int(3 * (1 - seg(t, 1.35, 1.6)) * (1 if int(t * 40) % 2 else -1)) if 1.35 < t < 1.6 else 0, 0)

        yo = int(-90 * exit_)
        draw_text(f, (W - text_width("TWO RIVALS. ONE GAME-BREAKING BUG.")) // 2, 66 + yo, "TWO RIVALS. ONE GAME-BREAKING BUG.",
                  (255, 255, 255), 1, (8, 8, 40), limit=int(max(0, t - 1.55) * 34))
        draw_text(f, (W - text_width("CAN THEY FIX IT... TOGETHER?")) // 2, 78 + yo, "CAN THEY FIX IT... TOGETHER?",
                  GOLD, 1, (8, 8, 40), limit=int(max(0, t - 2.25) * 30))
        # progress row: FIGHT > CRASH > DEBUG > BEFRIEND
        words = ["FIGHT", "CRASH", "DEBUG", "BEFRIEND"]
        total = text_width("FIGHT > CRASH > DEBUG > BEFRIEND")
        x = (W - total) // 2
        for i, wd in enumerate(words):
            if t < 2.35:
                break
            lit = t > 2.55 + i * 0.22
            draw_text(f, x, 98 + yo, wd, (255, 255, 255) if lit else (90, 98, 140), 1, (8, 8, 40))
            if lit:
                hline(f, x, 107 + yo, text_width(wd), GOLD)
            x += text_width(wd) + ADV
            if i < 3:
                draw_text(f, x, 98 + yo, ">", DIM, 1, (8, 8, 40))
                x += 2 * ADV
        if t > 1.7:
            a = seg(t, 1.7, 2.0)
            c1 = tuple(int(v * a) for v in (200, 210, 255))
            text_c(f, 150, "MADE WITH " + self.credit, c1)
            text_c(f, 161, "VOICES: ELEVENLABS ELEVEN V4", tuple(int(v * a) for v in (150, 160, 215)))
        fx.crt_on(f, t / 0.35)
        if t > TITLE_END - 0.05:
            fx.flash(f, 0.8, (255, 255, 255))
        return f

    # ------------------------------------------------------------------ boss fight + crash
    def _byte_state(self, t):
        C, D = self.C, self.D
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
            S.update(x=lerp(BX, 148, smooth(u)), y=24 * math.sin(math.pi * u * 0.75), stance="jump", arm="aim", expr="angry")
        else:
            v = seg(t, CL, FZ)
            S.update(x=lerp(148, 98, ease_out(v)), y=lerp(17, 30, ease_out(v)), stance="hurt", arm="hit", back="hit", lean=-2,
                     expr="ko", white=(t - CL) < 0.07)
            if t >= CL + 1.15:
                S["expr"] = "worried"
            if t >= C["L07"] - 0.05:
                S["expr"] = "worried"
            if self.speaking("09", t):
                S.update(expr="wide", arm="hit" if int(t * 8) % 2 else "raise", back="raise" if int(t * 8) % 2 else "hit")
            if t >= C["L10"]:
                S["expr"] = "worried"
        return S

    def _null_state(self, t):
        C, D = self.C, self.D
        S = dict(x=NX, y=0, stance="stand", arm="cross", expr="deadpan", flip=True, wind=0.3)
        CL, FZ = C["clash"], C["freeze"]
        arrive = C["fire"] + 0.49
        if C["L02"] + 0.25 <= t < C["L03"]:
            S["arm"] = "shrug"
        if arrive - 0.28 <= t < arrive + 0.28:
            u = (t - (arrive - 0.28)) / 0.56
            S.update(y=22 * 4 * u * (1 - u), stance="jump", arm="cross", wind=0.6)
        if C["L06"] - 0.15 <= t < C["leap"]:
            S["arm"] = "aim"
        if C["leap"] <= t < CL:
            u = (t - C["leap"]) / (CL - C["leap"])
            S.update(x=lerp(NX, 172, smooth(u)), y=24 * math.sin(math.pi * u * 0.75), stance="jump", arm="aim", expr="deadpan", wind=1.0)
        if t >= CL:
            v = seg(t, CL, FZ)
            S.update(x=lerp(172, 222, ease_out(v)), y=lerp(17, 30, ease_out(v)), stance="hurt", arm="hit", back="hit", lean=-2,
                     expr="ko", white=(t - CL) < 0.07, wind=0.8)
            if t >= CL + 1.15:
                S.update(stance="fall", arm="cross", back=None, expr="deadpan", lean=0)
        return S

    def _arena(self, t):
        C, D = self.C, self.D
        CL, FZ = C["clash"], C["freeze"]
        f = new_frame()
        scroll = 55.0 * (min(t, CL) - 1.0)
        tt = min(t, FZ)
        k = clamp01((t - FZ) / 2.0) if t >= FZ else 0.0
        offs = vs = None
        missing, static = (), 0.0
        if t >= FZ:
            ph = int(t * 9)
            r = np.random.default_rng(ph + 7)
            base = [scroll * m for m in (0.05, 0.12, 0.3, 0.75, 1.0)]
            offs = [b + float(r.integers(-70, 70)) * k for b in base]
            vs = [int(r.integers(-9, 10) * k) for _ in range(5)]
            if k > 0.3 and ph % 4 == 0:
                missing = (1,)
            elif k > 0.45 and ph % 7 == 3:
                missing = (2,)
            static = clamp01((t - (FZ + 0.7)) / 1.3) * 0.7
        self.dusk.render(f, scroll, tt, offs, vs, missing, static=static, static_seed=int(t * 30))
        if k > 0:                                   # corrupt the WORLD, not the fighters
            fx.smear(f, self.prev, 0.28 * k)
            self.prev = f.copy()
            fx.glitch(f, min(0.8, 0.2 + 0.6 * k), seed=int(t * 24))
        else:
            self.prev = None

        B, N = self._byte_state(t), self._null_state(t)
        if t < FIGHT_START:                       # beam-in
            appear = BEAM_IN + 0.32
            for (x, col) in ((BX, (120, 235, 255)), (NX, (255, 110, 130))):
                u = seg(t, BEAM_IN, BEAM_IN + 0.28)
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

        # banner during beam-in / READY
        if TITLE_END <= t < FIGHT_START - 0.05:
            xo1 = int((1 - ease_out(seg(t, TITLE_END, TITLE_END + 0.25))) * -220)
            xo2 = int((1 - ease_out(seg(t, TITLE_END + 0.1, TITLE_END + 0.35))) * 220)
            s1, s2 = "STAGE 8", "FINAL BOSS"
            draw_text(f, (W - text_width(s1, 3)) // 2 + xo1, 34, s1, (255, 255, 255), 3, (8, 8, 40))
            draw_text(f, (W - text_width(s2, 2)) // 2 + xo2, 62, s2, (255, 90, 110), 2, (8, 8, 40))
            if t >= BEAM_IN + 0.3 and int((t - BEAM_IN - 0.3) * 8) % 2 == 0:
                draw_text(f, (W - text_width("READY", 2)) // 2, 92, "READY", GOLD, 2, (8, 8, 40))
        fx.flash(f, 0.5 * (1 - seg(t, TITLE_END, TITLE_END + 0.2)), (255, 255, 255)) if t < TITLE_END + 0.2 else None

        # shake / flash on the clash
        if CL <= t < CL + 0.55:
            u = (t - CL) / 0.55
            fx.shake(f, int(5 * (1 - u)) * (1 if int(t * 60) % 2 else -1), int(2 * (1 - u)) * (-1 if int(t * 45) % 2 else 1))
            fx.flash(f, max(0.0, 1.0 - u * 2.4))
        # glitching
        if k > 0:
            fx.glitch(f, 0.10 + 0.10 * k, seed=int(t * 24) + 1)
        if t >= FIGHT_START - 0.4 and t < C["void_off"]:
            self._hud_and_text(f, t)
            self.caption(f, t)
        return f

    def _battle_fx(self, f, t, B, N):
        C, D = self.C, self.D
        CL, fire = C["clash"], C["fire"]
        # charging orb on Byte's muzzle
        if C["L03"] + 0.3 <= t < fire:
            r = 1 + 6 * seg(t, C["L03"] + 0.3, fire)
            mx, my = muzzle("byte", "stand", 0)
            fx.circle(f, B["x"] + mx, GY + my, r, (120, 240, 255), fill=True)
            fx.circle(f, B["x"] + mx, GY + my, max(1, r - 3), (255, 255, 255), fill=True)
        if fire <= t < C["L05"] + 0.9:               # the shot
            px = 127 + 170 * (t - fire)
            if px < W + 12:
                fx.trail(f, px, GY - 14, 1, "byte")
                fx.pellet(f, px, GY - 14, "byte", t, 1)
                fx.circle(f, px, GY - 14, 6, (120, 240, 255))
        if fire <= t < fire + 0.2:
            mx, my = muzzle("byte", "stand", 0)
            fx.starburst(f, BX + 19, GY + my, 8, (120, 240, 255), spikes=6)
        # charge auras + speed lines
        if C["L05"] <= t < C["leap"]:
            fx.aura(f, B["x"], GY - 16, t, (90, 220, 255), 1.0, 24)
            fx.speedlines(f, t, (170, 230, 255), 10, 0.4)
            fx.shake(f, int(math.sin(t * 90)), 0)
        if C["L06"] - 0.1 <= t < C["leap"]:
            fx.aura(f, N["x"], GY - 16, t, (170, 40, 70), 0.35, 14)
        if C["leap"] <= t < CL:
            u = (t - C["leap"]) / (CL - C["leap"])
            fx.speedlines(f, t, (255, 255, 255), 8, 0.5)

    def _battle_front(self, f, t, B, N):
        C, CL, fire = self.C, self.C["clash"], self.C["fire"]
        if C["L03"] + 0.2 <= t < fire + 1.0:                       # attack-name callout
            col = GOLD if int(t * 12) % 2 else (255, 255, 255)
            s = "MEGA-BYTE BUSTER!"
            draw_text(f, (W - text_width(s, 2)) // 2, 27, s, col, 2, (60, 20, 10))
        if C["L04"] + 0.6 <= t < C["L05"]:
            ui.sweat(f, B["x"] + 8, GY - 40, t)
        if CL <= t < CL + 0.7:
            u = (t - CL) / 0.7
            fx.starburst(f, 160, GY - 34 - 10 * u, 10 + 62 * ease_out(u), (255, 214, 90), (255, 255, 255), 10, u * 0.8)
            fx.burst(f, 160, GY - 36, t - CL, 34, 130, [(255, 255, 255), (120, 240, 255), (255, 90, 110), (255, 220, 90)], 0.8, 2, 80, 3)

    def _hud_and_text(self, f, t):
        C = self.C
        CL = C["clash"]
        base_b, base_n = 0.36, 0.30
        fill = seg(t, BEAM_IN + 0.5, BEAM_IN + 1.0)
        if t < CL:
            hb, hn = base_b * fill, base_n * fill
        else:
            hb = hn = 0.0
        ui.hud(f, hb, hn, t)
        # who won?
        if t >= CL + 0.35:
            u = t - (CL + 0.35)
            if u < 0.6:
                if int(t * 12) % 2 == 0:
                    ui.text_c(f, W // 2, 50, "K.O.!", (255, 255, 255), 4, shadow=(200, 30, 60))
            else:
                per = max(0.06, 0.26 - 0.07 * (u - 0.6))
                which = int((u - 0.6) / per) % 2
                blend_rect(f, 40, 44, 240, 24, (0, 0, 0), 0.45)
                if u > 2.6 and int(t * 20) % 3 == 0:
                    ui.text_c(f, W // 2, 50, "BYTE WINS!", BYTE_C, 2, shadow=(8, 8, 40))
                    ui.text_c(f, W // 2 + 2, 52, "NULL WINS!", NULL_C, 2, shadow=(8, 8, 40))
                elif which == 0:
                    ui.text_c(f, W // 2, 50, "BYTE WINS!", BYTE_C, 2, shadow=(8, 8, 40))
                else:
                    ui.text_c(f, W // 2, 50, "NULL WINS!", NULL_C, 2, shadow=(8, 8, 40))

    # ------------------------------------------------------------------ the debug void
    def _terminal_state(self, t):
        """Rows of the code window: 0 run, 1 hang, 2 line 420, 3 line 421 (the fix), 4 line 422, 5 build, 6 result."""
        C = self.C
        rows = [("> RUN STAGE_8.GAME", (190, 255, 200)),
                ("!! HANG AT LINE 420 !!", (255, 110, 120)),
                ("420 IF (P1.HP == 0 && P2.HP == 0) {", (190, 255, 200)),
                ["421   ", (150, 170, 215)],
                ("422 }", (190, 255, 200)),
                ("", (0, 0, 0)),
                ("", (0, 0, 0))]
        todo = "// TODO: FIGURE OUT WHO WINS"
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
            rows[6] = ("ERROR: EXPECTED ';' AT LINE 421", (255, 100, 110))
            flash = 6 if t < C["err"] + 0.7 else None
        if t >= C["ok"]:
            rows[6] = ("BUILD OK!  0 ERRORS  0 BUGS", (120, 255, 150))
            rows[1] = ("OK: NO MORE HANGS", (120, 255, 150))
        cursor = (3, len(rows[3][0])) if (C["L13"] <= t < C["ok"] + 0.4) else None
        return rows, cursor, hl, flash

    def _debug(self, t):
        """Null sits at a desk typing on the terminal; Byte stands behind the chair, watching over his shoulder."""
        C, D = self.C, self.D
        f = new_frame()
        fixed = seg(t, C["ok"], C["ok"] + 0.8)
        self.void.render(f, t, fixed)
        rows, cursor, hl, flash = self._terminal_state(t)

        NXS, BXS, STOOL = 110, 88, 5                                  # chair spot / Byte's spot / step-stool height
        DESK_X0, DESK_X1, DESK_TOP = 116, 262, 129
        MON_X, MON_Y = 168, 100
        land_n = C["land_n"]
        seat = 4                                                      # seated fighters rest 4px above the floor line

        # the desk, chair and monitor materialise around them as they land
        stool_p = seg(t, C["land_b"] - 0.12, C["land_b"] + 0.20)
        chair_p = seg(t, land_n - 0.15, land_n + 0.25)
        desk_p = seg(t, land_n + 0.10, land_n + 0.60)
        win_on = t >= land_n + 0.55
        state = "ok" if t >= C["ok"] else ("error" if C["err"] <= t < C["semi"] else "normal")
        jolt = int(2 * math.sin(t * 70)) if self.speaking("14", t) else 0

        if win_on:
            props.hologram(f, MON_X, MON_X + 46, MON_Y, 44 + jolt, 276 + jolt, 80, t)
            ui.terminal(f, 44 + jolt, 2, 232, 78, "DEBUG.EXE - STAGE_8.GAME", rows, t, cursor, hl, flash_row=flash,
                        pitch=9, top=14)
        butter = (t - C["ok"]) if t >= C["ok"] else None
        self.void.draw_bugs(f, t, butter=butter)

        if stool_p > 0:
            g = f.copy()
            props.stool(g, BXS, GY, STOOL)
            props.reveal(f, g, stool_p)
        if chair_p > 0:
            g = f.copy()
            props.chair(g, NXS, GY)
            props.reveal(f, g, chair_p)

        B = dict(x=BXS, y=STOOL, stance="stand", arm="down", expr="neutral", flip=False)
        N = dict(x=NXS, y=seat, stance="sit", arm="reach", expr="deadpan", flip=False, wind=0.15)
        for (S, land, base) in ((B, C["land_b"], STOOL), (N, land_n, seat)):
            if t < land:                                              # falling in
                u = seg(t, C["void_on"] - 0.05, land)
                S.update(y=base + 172 * (1 - u * u), stance="fall", arm="raise", back="hit", expr="wide", mouth="shout")
            elif t < land + 0.22:                                     # bounce
                S["y"] = base + 4 * (1 - (t - land) / 0.22) * abs(math.sin((t - land) * 22))
        for (land, x) in ((C["land_b"], BXS), (land_n, NXS)):
            fx.burst(f, x, GY - 2, t - land, 12, 40, [(190, 170, 210), (120, 100, 150)], 0.5, 2, 30, 9)

        typing = C["type0"] <= t < C["type_end"] or C["semi"] <= t < C["semi"] + 0.25
        if t >= land_n + 0.25:
            B.update(arm="down", expr="worried" if int(t * 2) % 5 else "blink")
            if self.speaking("11", t):
                B["expr"] = "neutral"
            if self.speaking("12", t):
                N["expr"] = "smug"
                B.update(expr="angry", lean=1)
                ui.sweat(f, BXS + 9, GY - STOOL - 38, t)
            if C["L13"] - 0.2 <= t < C["L14"]:                        # leans in to read over Null's shoulder
                B.update(arm="chin", expr="neutral", lean=3)
                N["lean"] = 1
            if self.speaking("14", t):
                B.update(arm="raise", back="raise", expr="angry", lean=0, y=STOOL + (1 if int(t * 12) % 2 else 0))
                N.update(expr="wide", y=seat + 1)
                ui.bang(f, BXS + 4, GY - STOOL - 46, t)
            elif C["L15"] - 0.02 <= t < C["L16"]:
                B.update(arm="chin", expr="worried", lean=3)
                N["y"] = seat + (1 if int(t * 6) % 2 else 0)          # nods
            if C["L16"] <= t < C["type0"]:
                B.update(arm="point", lean=3, expr="wide" if t < C["L16"] + 0.7 else "happy")
                ui.bulb(f, BXS + 2, GY - STOOL - 44, t)
            if C["L17"] <= t < C["type0"]:
                N.update(expr="wide" if t < C["L17"] + 0.7 else "happy", arm="cross")
            if C["type0"] <= t < C["ok"] + 0.3:
                N.update(arm="reach", expr="deadpan", lean=1)
                B.update(arm="chin", expr="happy" if t < C["err"] else "wide", lean=3)
            if typing:
                N["y"] = seat + (1 if int(t * C["cps"]) % 2 else 0)   # keystroke bounce
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

        self.draw_actor(f, "null", N, t)                              # Null first, Byte in front of the chair back
        self.draw_actor(f, "byte", B, t)

        if desk_p > 0:
            g = f.copy()
            props.desk(g, DESK_X0, DESK_X1, DESK_TOP)
            props.keyboard(g, 122, DESK_TOP - 3)
            props.monitor(g, MON_X, MON_Y, rows, t, state, hl, cursor)
            props.mug(g, 236, DESK_TOP - 6, t)
            props.reveal(f, g, desk_p)

        if t >= C["ok"] and t < C["ok"] + 0.5:
            fx.burst(f, 190, 100, t - C["ok"], 40, 110, [(120, 255, 150), (255, 255, 255), (255, 216, 74)], 0.8, 2, -20, 4)
        if t >= C["ok"]:
            fx.flash(f, 0.5 * (1 - seg(t, C["ok"], C["ok"] + 0.25)), (200, 255, 210))
        if t >= C["reboot"] - 0.3:
            fx.flash(f, seg(t, C["reboot"] - 0.3, C["reboot"]))
        self.caption(f, t)
        return f

    # ------------------------------------------------------------------ repaired world
    def _ending(self, t):
        C = self.C
        f = new_frame()
        sun = seg(t, C["reboot"], C["credits"])
        scroll = 26.0 * (t - C["reboot"])
        self.dawn.render(f, scroll, t, sun=sun * 0.7)
        bx0, nx0 = 128, 192
        u = ease_out(seg(t, C["L21"] + D_TAIL(self), C["bump"]))
        BXp, NXp = lerp(bx0, 148, u), lerp(nx0, 172, u)
        B = dict(x=BXp, stance="stand", arm="down", expr="happy", flip=False, wind=0.3)
        N = dict(x=NXp, stance="stand", arm="cross", expr="deadpan", flip=True, wind=0.4)
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
            fx.starburst(f, 160, GY - 24, 4 + 26 * ease_out(tau / 0.35), (255, 240, 150), (255, 255, 255), 8, tau * 2) if tau < 0.45 else None
            fx.burst(f, 160, GY - 24, tau, 30, 80, [(255, 240, 150), (255, 255, 255), (255, 150, 200), (130, 240, 255)], 1.1, 2, -10, 6)
            for i in range(5):          # rising hearts
                hx = 160 + math.sin(tau * 3 + i * 1.3) * 26 + (i - 2) * 8
                hy = GY - 40 - tau * (18 + i * 5)
                if 0 < tau < 1.5 and hy > 8:
                    draw_text(f, int(hx), int(hy), "♥", (255, 110, 150), 1, (60, 10, 30))
        if t < C["reboot"] + 0.45:
            fx.flash(f, 1 - seg(t, C["reboot"], C["reboot"] + 0.45))
        if C["join"] <= t < C["credits"] - 0.12:
            u = ease_out(seg(t, C["join"], C["join"] + 0.25))
            xo = int((1 - u) * -320)
            blend_rect(f, 0, 44, W, 30, (10, 12, 40), 0.72)
            hline(f, 0, 44, W, GOLD); hline(f, 0, 73, W, GOLD)
            col = GOLD if int(t * 6) % 2 == 0 else (255, 255, 255)
            draw_text(f, (W - text_width("PLAYER 2 HAS JOINED!", 2)) // 2 + xo, 53, "PLAYER 2 HAS JOINED!", col, 2, (60, 30, 0))
        self.caption(f, t)
        return f

    # ------------------------------------------------------------------ credits card
    def _credits(self, t):
        C = self.C
        f = self._ending(t)
        u = seg(t, C["credits"], C["credits"] + 0.35)
        blend_rect(f, 0, 0, W, 118, (8, 10, 34), 0.72 * u)
        if u > 0:
            text_c(f, 8, "BYTE & NULL", GOLD, 3)
            text_c(f, 34, "2-PLAYER CO-OP", (255, 255, 255), 2)
            if int(t * 2.4) % 2 == 0:
                text_c(f, 54, "PRESS START", CYAN)
            text_c(f, 72, "MADE WITH", (170, 180, 230))
            text_c(f, 83, self.credit, GOLD, 2)
            text_c(f, 104, "VOICES: ELEVENLABS ELEVEN V4", (200, 210, 255))
        if t > C["credits"] + 1.6:
            text_c(f, 168, "(IT WAS A SEMICOLON.)", (150, 160, 210))
        fade = seg(t, self.end - 0.5, self.end)
        if fade > 0:
            fx.flash(f, fade, (0, 0, 0))
        return f

    # ------------------------------------------------------------------ sound cues
    def events(self):
        """[(time, name, gain)] - names map to chiptune SFX in mix.py."""
        C, D = self.C, self.D
        ev = [(BEAM_IN, "dash", 0.7), (BEAM_IN + 0.32, "land", 0.9), (TITLE_END, "confirm", 0.6),
              (0.55, "dash", 0.5), (0.85, "dash", 0.5), (1.35, "clash", 0.55)]
        for i in range(4):
            ev.append((BEAM_IN + 0.32 + 0.16 * i + 0.1, "bleep", 0.5)) if False else None
        ev += [(C["fire"], "shoot_byte", 0.9), (C["fire"] + 0.49 - 0.28, "jump", 0.7), (C["fire"] + 0.49 + 0.28, "land", 0.6),
               (C["L05"], "charge", 0.9), (C["leap"], "jump", 0.8), (C["leap"] + 0.02, "jump", 0.8),
               (C["clash"], "clash", 1.0), (C["clash"], "hit", 0.9), (C["clash"] + 0.02, "crash", 1.0)]
        for dt in (1.0, 1.75, 2.4, 3.1, 3.7):
            ev.append((C["clash"] + dt, "glitch", 0.55))
        ev += [(C["land_b"], "land", 0.9), (C["land_n"], "land", 0.9),
               (C["land_n"] + 0.12, "bleep", 0.55), (C["land_n"] + 0.34, "bleep", 0.5), (C["land_n"] + 0.56, "confirm", 0.35)]
        ev += [(C["void_off"], "dash", 0.6)]
        # typing
        n = len(C["type_chars"])
        for i in range(int(C["erase_dur"] * 30)):
            ev.append((C["type0"] + i / 30.0, "click", 0.35))
        for i in range(n):
            ev.append((C["type0"] + C["erase_dur"] + i / C["cps"], "click", 0.5))
        ev += [(C["err"], "error", 0.8), (C["semi"], "click", 0.6), (C["ok"], "confirm", 0.9),
               (C["reboot"] - 0.1, "dash", 0.6), (C["join"], "join", 0.9)]
        return sorted(ev)


def D_TAIL(film):
    return film.D["21"] + 0.05
