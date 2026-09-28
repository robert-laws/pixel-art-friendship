"""Audio mix: chiptune music sections + SFX cues + Eleven v4 dialogue, music ducked under speech."""
import wave

import numpy as np

from . import chiptune as ct
from .timeline import BEAM_IN, TITLE_END

SR = ct.SR
PAN = {"byte": -0.28, "null": 0.28}          # dialogue placed slightly left / right
MUSIC_GAIN, SFX_GAIN, DUCK = 0.34, 0.40, 0.27


def _put(buf, x, start, gain=1.0, pan=None):
    i = int(round(start * SR))
    n = buf.shape[-1]                       # NB: len(buf) would be the channel count for stereo
    if i >= n or i + len(x) <= 0:
        return
    a = max(0, -i)
    end = min(n, i + len(x))
    seg = x[a:end - i] * gain
    if buf.ndim == 1:
        buf[i + a:end] += seg
    else:
        th = np.pi / 4 if pan is None else (pan + 1.0) * np.pi / 4
        buf[0, i + a:end] += seg * np.cos(th) * np.sqrt(2)
        buf[1, i + a:end] += seg * np.sin(th) * np.sqrt(2)


def _fadeio(x, fi, fo):
    x = x.copy()
    a, b = int(fi * SR), int(fo * SR)
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if b:
        x[-b:] *= np.linspace(1, 0, b)
    return x


def build_music(C, total):
    m = np.zeros(int(total * SR) + SR, np.float32)
    title_end = C.get("title_end", TITLE_END)
    _put(m, ct.title_ready(title_end - 0.30), 0.30)
    battle_len = (C["clash"] + 0.95) - title_end
    _put(m, ct.tape_stop(ct.battle_theme(battle_len, fade=0.0), tail=1.0), title_end)
    _put(m, ct.crash_sting(4.0), C["clash"] + 0.02)
    bed_start = C["clash"] + 3.4
    bed_len = (C["void_off"] + 0.55) - bed_start
    _put(m, _fadeio(ct.sfx_static(bed_len, level=0.10), 0.8, 0.5), bed_start)
    _put(m, ct.debug_theme(C["reboot"] - C["void_on"], fade=0.30), C["void_on"], 1.25)
    _put(m, ct.friendship_theme(total - C["reboot"] - 0.25), C["reboot"], 1.15)
    return m[: int(total * SR)]


SFX = {
    "dash": ct.sfx_dash, "land": ct.sfx_land, "confirm": ct.sfx_confirm, "clash": ct.sfx_clash,
    "shoot_byte": ct.sfx_shoot_byte, "jump": ct.sfx_jump, "hit": ct.sfx_hit, "glitch": ct.sfx_glitch_burst,
    "click": ct.sfx_type_click, "error": ct.sfx_error_beep, "join": ct.sfx_powerup_join, "bleep": ct.sfx_ui_bleep,
}


def build_sfx(events, C, total):
    buf = np.zeros(int(total * SR) + SR, np.float32)
    cache = {}
    for (t, name, gain) in events:
        if name == "crash":
            continue                                 # handled inside the music (crash_sting)
        if name == "charge":
            x = ct.sfx_charge(C["leap"] - C["L05"] + 0.35)
        else:
            if name not in cache:
                cache[name] = SFX[name]()
            x = cache[name]
        _put(buf, x, t, SFX_GAIN * gain)
    return buf[: int(total * SR)]


def duck_curve(lines, C, D, total):
    g = np.ones(int(total * SR), np.float32)
    att, rel = int(0.05 * SR), int(0.35 * SR)
    for lid in lines:
        s, e = int(C["L" + lid] * SR), int((C["L" + lid] + D[lid]) * SR)
        lo = max(0, s - att)
        hi = min(len(g), e + rel)
        env = np.full(hi - lo, DUCK, np.float32)
        env[: s - lo] = np.linspace(1, DUCK, s - lo, endpoint=False) if s > lo else env[: s - lo]
        tail = hi - min(len(g), e)
        if tail > 0:
            env[-tail:] = np.linspace(DUCK, 1, tail)
        g[lo:hi] = np.minimum(g[lo:hi], env)
    return g


def render(lines, C, D, events, total, path):
    music = build_music(C, total) * MUSIC_GAIN * duck_curve(lines, C, D, total)
    sfx = build_sfx(events, C, total)
    out = np.zeros((2, int(total * SR)), np.float32)
    out[0] += music; out[1] += music
    out[0] += sfx; out[1] += sfx
    for lid, ln in lines.items():
        _put(out, ln["audio"], C["L" + lid], 1.0, PAN[ln["speaker"]])
    peak = float(np.abs(out).max())
    out = np.tanh(out * 1.05) / np.tanh(1.05)      # soft limiter, then trim to -1 dBFS
    out *= 0.89 / max(1e-6, float(np.abs(out).max()))
    pcm = (out.T * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    rms = float(np.sqrt(np.mean(out ** 2)))
    return dict(peak_before_limit=peak, rms_db=20 * np.log10(rms + 1e-9), seconds=out.shape[1] / SR)
