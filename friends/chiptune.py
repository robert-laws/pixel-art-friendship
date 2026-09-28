"""
chiptune.py - numpy-only NES/SNES flavoured music + SFX synthesiser for "BYTE vs NULL".

Everything here returns a 1-D ``np.float32`` mono array at ``SR = 44100`` Hz,
peak-normalised to ~0.8 (never clips, no NaN/inf) with tiny fades at both ends.
All output is deterministic (fixed ``np.random.default_rng`` seeds) and fully
vectorised (per-note loops only), so a 20 s track renders in well under 2 s.

Public API (durations in seconds)
---------------------------------
Music (each returns EXACTLY ``round(duration * SR)`` samples)::

    title_ready(duration=2.4)              STAGE START / READY! fanfare, ends on an
                                           E-power-chord accent -> battle theme.
    battle_theme(duration=25.0, fade=0.35) boss fight, E natural minor, 152 BPM, 8-bar
                                           loop.  Pass ``fade=0`` if you will apply
                                           tape_stop() to its end (keeps full level).
    crash_sting(duration=4.0)              boom + glitch stutter + corrupted arpeggio
                                           -> quiet digital static / hum "void".
    debug_theme(duration=24.0, fade=0.8)   calm lo-fi A-minor -> C-major, 88 BPM,
                                           8-bar loop, EQ'd to leave room for speech.
    friendship_theme(duration=20.0)        battle motif re-harmonised in E major,
                                           ~112 BPM (tempo is nudged +-10 % so that a
                                           whole number of bars fits ``duration``),
                                           builds and lands on a big E major chord with
                                           a bell sparkle exactly at the very end
                                           (use >= 6 s for a proper build).

Effect::

    tape_stop(audio, tail=1.2)             pitch-dive / slow-down + low-pass darkening
                                           on the last ``tail`` seconds.  Same length
                                           as input; the input level is preserved
                                           (no re-normalisation).

SFX (short arrays)::

    sfx_shoot_byte()  ~0.18s      sfx_shoot_null()  ~0.22s
    sfx_hit()         ~0.20s      sfx_block_clang() ~0.20s
    sfx_jump()        ~0.20s      sfx_land()        ~0.10s      sfx_dash() ~0.25s
    sfx_charge(duration=1.6)      sfx_clash()       ~0.50s
    sfx_type_click()  ~0.03s      sfx_type_burst(n=8, cps=14)
    sfx_error_beep()  ~0.40s      sfx_confirm()     ~0.35s
    sfx_powerup_join() ~1.0s      sfx_text_blip(pitch_hz=440, duration=0.045)
    sfx_glitch_burst(duration=0.6)
    sfx_static(duration, level=0.12, loop=True)
        quiet filtered static bed.  ``level`` is its PEAK (default 0.12, deliberately
        NOT 0.8).  With ``loop=True`` it tiles seamlessly (no edge fades).
    sfx_ui_bleep()    ~0.10s

Shared musical material
-----------------------
``MOTIF`` - the 4-bar (16 beat) lead motif of the battle theme as a list of
``(scale_degree, length_in_beats)``.  Degrees are 1-based relative to the tonic
(1 = tonic E5, 2 = F#5, ... 7 = D6, 8 = E6 (octave), 9 = F#6, 10 = G6; 0 = D5,
-1 = C5 ...).  ``None`` is a rest.  ``MOTIF_B`` is the varied answer used in bars
5-8 of the battle theme.  ``degree_to_midi(deg, tonic_midi, scale)`` converts;
the friendship theme uses the very same rhythm/contour with ``MAJOR`` instead of
``MINOR``.

Toolkit (all public, all vectorised)
------------------------------------
midi_to_freq, note, pulse (duties 0.125/0.25/0.5/0.75, poly-BLEP band limited),
triangle (optionally 16-step NES quantised), saw, sine, noise (white, NES
15-bit 'long' LFSR and 93-step 'short' periodic LFSR), adsr, vibrato,
pitch_slide, lowpass/highpass (one-pole), lowpass_sweep (time varying),
bitcrush (sample-rate + bit-depth reduce, scalar or time-varying), echo,
chorus, finalize.

Run ``python -m friends.chiptune`` (or this file) to render every asset as a
16-bit WAV into the demo directory and print duration / peak / RMS / sanity.
"""
from __future__ import annotations

import os
import sys
import wave
from functools import lru_cache

import numpy as np

SR = 44100

__all__ = [
    "SR", "MOTIF", "MOTIF_B", "MINOR", "MAJOR", "degree_to_midi", "LOOP_SECONDS",
    "midi_to_freq", "note", "pulse", "triangle", "saw", "sine", "noise", "adsr",
    "vibrato", "pitch_slide", "lowpass", "highpass", "lowpass_sweep", "bitcrush",
    "echo", "chorus", "finalize",
    "title_ready", "battle_theme", "crash_sting", "tape_stop", "debug_theme",
    "friendship_theme",
    "sfx_shoot_byte", "sfx_shoot_null", "sfx_hit", "sfx_block_clang", "sfx_jump",
    "sfx_land", "sfx_dash", "sfx_charge", "sfx_clash", "sfx_type_click",
    "sfx_type_burst", "sfx_error_beep", "sfx_confirm", "sfx_powerup_join",
    "sfx_text_blip", "sfx_glitch_burst", "sfx_static", "sfx_ui_bleep",
]

TWO_PI = 2.0 * np.pi


# ======================================================================
# TOOLKIT
# ======================================================================
def _ns(seconds) -> int:
    """seconds -> integer sample count."""
    return int(round(float(seconds) * SR))


def _t(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / SR


def midi_to_freq(m):
    """MIDI note number (scalar or array) -> frequency in Hz (A4 = 69 = 440 Hz)."""
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


_NOTE_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note(name: str) -> int:
    """'E5' -> 76, 'F#4' -> 66, 'Bb3' -> 58."""
    s = name.strip()
    v = _NOTE_BASE[s[0].upper()]
    i = 1
    while i < len(s) and s[i] in "#b":
        v += 1 if s[i] == "#" else -1
        i += 1
    return 12 * (int(s[i:]) + 1) + v


MINOR = (0, 2, 3, 5, 7, 8, 10)   # natural minor
MAJOR = (0, 2, 4, 5, 7, 9, 11)


def degree_to_midi(deg: int, tonic: int, scale=MINOR) -> int:
    """1-based scale degree (may exceed 7 / go below 1) -> MIDI note."""
    i = int(deg) - 1
    return int(tonic + 12 * (i // 7) + scale[i % 7])


# ---- oscillators -----------------------------------------------------
def _as_f(freq, n: int) -> np.ndarray:
    f = np.asarray(freq, dtype=np.float64)
    if f.ndim == 0:
        return np.full(n, float(f))
    if len(f) != n:
        raise ValueError("freq array length %d != n %d" % (len(f), n))
    return f


def _phase(freq, n: int, phase0: float = 0.0):
    f = np.clip(_as_f(freq, n), 1.0, SR * 0.45)
    dt = f / SR
    ph = phase0 + np.cumsum(dt) - dt
    return ph % 1.0, dt


def _blep(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Poly-BLEP residual for a rising unit step at t = 0 (t, dt arrays)."""
    out = np.zeros_like(t)
    m = t < dt
    if m.any():
        x = t[m] / dt[m]
        out[m] = x + x - x * x - 1.0
    m = t > 1.0 - dt
    if m.any():
        x = (t[m] - 1.0) / dt[m]
        out[m] = x * x + x + x + 1.0
    return out


def _len_of(freq, n):
    if n is None:
        f = np.asarray(freq)
        if f.ndim == 0:
            raise ValueError("n required for scalar frequency")
        return len(f)
    return int(n)


def pulse(freq, n=None, duty=0.5, phase0=0.0) -> np.ndarray:
    """Band-limited pulse wave, DC removed.  duty: 0.125 / 0.25 / 0.5 / 0.75 (or any
    value in 0.02..0.98, scalar or per-sample array).  freq scalar or array."""
    n = _len_of(freq, n)
    t, dt = _phase(freq, n, phase0)
    d = np.clip(np.asarray(duty, dtype=np.float64), 0.02, 0.98)
    y = np.where(t < d, 1.0, -1.0)
    y = y + _blep(t, dt) - _blep((t - d) % 1.0, dt)
    return y - (2.0 * d - 1.0)


def triangle(freq, n=None, steps: int = 0, phase0=0.0) -> np.ndarray:
    """Triangle wave.  steps=16 gives the NES 4-bit stair-stepped triangle."""
    n = _len_of(freq, n)
    t, _ = _phase(freq, n, phase0)
    y = 4.0 * np.abs(t - 0.5) - 1.0
    if steps and steps > 1:
        q = steps - 1
        y = np.round((y + 1.0) * 0.5 * q) / q * 2.0 - 1.0
    return y


def saw(freq, n=None, phase0=0.0) -> np.ndarray:
    """Band-limited (poly-BLEP) rising sawtooth."""
    n = _len_of(freq, n)
    t, dt = _phase(freq, n, phase0)
    return 2.0 * t - 1.0 - _blep(t, dt)


def sine(freq, n=None, phase0=0.0) -> np.ndarray:
    n = _len_of(freq, n)
    t, _ = _phase(freq, n, phase0)
    return np.sin(TWO_PI * t)


# ---- noise ------------------------------------------------------------
@lru_cache(maxsize=None)
def _lfsr_table(kind: str) -> np.ndarray:
    """NES 15-bit LFSR output (+-1).  'long' = period 32767, 'short' = tap-6 mode
    (period 93 or 31 depending on where the register lands)."""
    tap = 6 if kind == "short" else 1
    reg = 1
    seen = {}
    out = []
    for i in range(40000):
        if reg in seen:
            start = seen[reg]
            out = out[start:]
            break
        seen[reg] = i
        out.append(reg & 1)
        fb = (reg & 1) ^ ((reg >> tap) & 1)
        reg = (reg >> 1) | (fb << 14)
    return 1.0 - 2.0 * np.asarray(out, dtype=np.float64)


def noise(n: int, kind: str = "white", rate=None, seed: int = 0, offset: int = 0) -> np.ndarray:
    """Noise source.  kind: 'white' | 'long' (NES 32767-step LFSR) | 'short'
    (NES periodic/metallic 93-step LFSR).  For LFSR noise ``rate`` is the LFSR clock
    in Hz (scalar or per-sample array; lower = crunchier / darker)."""
    if kind == "white":
        return np.random.default_rng(seed).uniform(-1.0, 1.0, n)
    tab = _lfsr_table("short" if kind == "short" else "long")
    r = np.asarray(SR if rate is None else rate, dtype=np.float64)
    if r.ndim == 0:
        pos = np.arange(n) * (float(r) / SR)
    else:
        pos = np.cumsum(r) / SR
    idx = (np.floor(pos).astype(np.int64) + int(offset)) % len(tab)
    return tab[idx]


# ---- envelopes / modulation ------------------------------------------
def adsr(n: int, a=0.005, d=0.08, s=0.7, r=0.03, exp: bool = True) -> np.ndarray:
    """ADSR over ``n`` samples; the release occupies the last ``r`` seconds.
    Attack linear, decay exponential (or linear if exp=False)."""
    n = int(n)
    if n <= 0:
        return np.zeros(0)
    t = _t(n)
    T = n / SR
    r = min(max(r, 1e-4), T)
    env = np.minimum(t / a, 1.0) if a > 0 else np.ones(n)
    td = np.maximum(t - a, 0.0)
    if d > 0:
        if exp:
            dec = s + (1.0 - s) * np.exp(-5.0 * td / d)
        else:
            dec = np.maximum(1.0 - (1.0 - s) * td / d, s)
    else:
        dec = np.full(n, s)
    rel = np.clip((T - t) / r, 0.0, 1.0)
    return env * dec * rel


def vibrato(freq, n: int, rate=5.5, depth_cents=15.0, delay=0.1, fade=0.15) -> np.ndarray:
    """Frequency array with delayed-onset sine vibrato."""
    t = _t(n)
    ramp = np.clip((t - delay) / max(fade, 1e-6), 0.0, 1.0)
    lfo = np.sin(TWO_PI * rate * t) * ramp
    return _as_f(freq, n) * 2.0 ** (depth_cents * lfo / 1200.0)


def pitch_slide(f0, f1, n: int, curve: str = "exp", power: float = 1.0) -> np.ndarray:
    """Frequency array sliding f0 -> f1 ('exp' = constant semitones/sec, or 'lin')."""
    u = (np.arange(n) / max(n, 1)) ** power
    if curve == "exp":
        return f0 * (f1 / f0) ** u
    return f0 + (f1 - f0) * u


# ---- filters -----------------------------------------------------------
def _fftconv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Causal convolution truncated to len(x); FFT (overlap-add for long inputs)."""
    n, m = len(x), len(h)
    if n == 0:
        return x.copy()
    if m <= 48 or n <= 48:
        return np.convolve(x, h)[:n]
    B = 1 << int(np.ceil(np.log2(max(4 * m, 4096))))
    if n <= 4 * B:
        nfft = 1 << int(np.ceil(np.log2(n + m - 1)))
        return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)[:n]
    nfft = 1 << int(np.ceil(np.log2(B + m - 1)))
    H = np.fft.rfft(h, nfft)
    nb = -(-n // B)
    xp = np.zeros(nb * B)
    xp[:n] = x
    Y = np.fft.irfft(np.fft.rfft(xp.reshape(nb, B), nfft, axis=1) * H, nfft, axis=1)
    out = np.zeros((nb + 1) * B)
    out[:nb * B] += Y[:, :B].reshape(-1)
    tail = np.zeros((nb, B))
    tail[:, :nfft - B] = Y[:, B:nfft]
    out[B:] += tail.reshape(-1)
    return out[:n]


def lowpass(x, fc: float, order: int = 1) -> np.ndarray:
    """One-pole low-pass (6 dB/oct per order), FFT-convolved (vectorised)."""
    x = np.asarray(x, dtype=np.float64)
    if fc >= SR * 0.49 or len(x) == 0:
        return x.copy()
    a = float(np.exp(-TWO_PI * max(fc, 1.0) / SR))
    L = int(min(max(len(x), 8), max(8, np.ceil(np.log(1e-4) / np.log(a)))))
    h = (1.0 - a) * a ** np.arange(L)
    y = x
    for _ in range(max(1, order)):
        y = _fftconv(y, h)
    return y


def highpass(x, fc: float, order: int = 1) -> np.ndarray:
    """One-pole high-pass (input minus its low-passed self)."""
    y = np.asarray(x, dtype=np.float64)
    for _ in range(max(1, order)):
        y = y - lowpass(y, fc)
    return y


def lowpass_sweep(x, fc_curve, nbank: int = 10, order: int = 1) -> np.ndarray:
    """Time-varying low-pass: cross-fades between a bank of fixed one-pole filters
    (``order`` cascaded poles each)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    fc = np.clip(_as_f(fc_curve, n), 30.0, SR * 0.45)
    lo, hi = float(fc.min()), float(fc.max())
    if hi / lo < 1.05:
        return lowpass(x, float(np.sqrt(lo * hi)), order)
    cuts = np.geomspace(lo, hi, nbank)
    bank = np.stack([lowpass(x, c, order) for c in cuts])
    p = np.interp(np.log(fc), np.log(cuts), np.arange(nbank))
    i0 = np.clip(np.floor(p).astype(int), 0, nbank - 2)
    fr = p - i0
    ar = np.arange(n)
    return bank[i0, ar] * (1.0 - fr) + bank[i0 + 1, ar] * fr


def bitcrush(x, bits=8, downsample=1.0) -> np.ndarray:
    """Bit-crusher: sample-and-hold rate reduction (``downsample`` = hold factor,
    scalar or per-sample array) + bit-depth reduction (``bits``, scalar or array)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    ds = np.asarray(downsample, dtype=np.float64)
    if ds.ndim == 0:
        if ds > 1.0:
            idx = np.floor(np.floor(np.arange(n) / ds) * ds + 1e-9).astype(np.int64)
            x = x[np.minimum(idx, n - 1)]
    else:
        c = np.cumsum(1.0 / np.maximum(ds, 1.0))
        chg = np.diff(np.floor(c), prepend=-1.0) != 0
        idx = np.maximum.accumulate(np.where(chg, np.arange(n), 0))
        x = x[idx]
    levels = 2.0 ** (np.asarray(bits, dtype=np.float64) - 1.0)
    return np.round(x * levels) / levels


def echo(x, delay: float, feedback: float = 0.35, mix: float = 0.3, taps: int = 4,
         circular: bool = False) -> np.ndarray:
    """Simple multi-tap SNES-style echo (circular=True keeps a loop seamless)."""
    x = np.asarray(x, dtype=np.float64)
    d = _ns(delay)
    y = x.copy()
    for k in range(1, taps + 1):
        g = mix * feedback ** (k - 1)
        sh = d * k
        if circular:
            y += g * np.roll(x, sh)
        elif sh < len(x):
            y[sh:] += g * x[:-sh]
    return y


def chorus(x, depth_ms: float = 2.5, rate: float = 0.7, mix: float = 0.35,
           base_ms: float = 11.0) -> np.ndarray:
    """Light modulated-delay chorus."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    delay = (base_ms + depth_ms * np.sin(TWO_PI * rate * _t(n))) * SR / 1000.0
    wet = np.interp(np.arange(n) - delay, np.arange(n), x, left=0.0, right=0.0)
    return x * (1.0 - 0.5 * mix) + wet * mix


def _fade(x: np.ndarray, fade_in: float, fade_out: float) -> np.ndarray:
    x = x.copy()
    n = len(x)
    ni = min(_ns(fade_in), n // 2)
    no = min(_ns(fade_out), n // 2)
    if ni > 1:
        x[:ni] *= np.sin(0.5 * np.pi * np.arange(ni) / ni) ** 2
    if no > 1:
        x[n - no:] *= np.cos(0.5 * np.pi * np.arange(no) / no) ** 2
    return x


def finalize(x, peak: float = 0.8, fade_in: float = 0.003, fade_out: float = 0.005) -> np.ndarray:
    """NaN-safe, tiny edge fades, peak-normalise, cast to float32."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = _fade(x, fade_in, fade_out)
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    if m > 1e-12:
        x = x * (peak / m)
    return x.astype(np.float32)


def _softclip(x, drive: float = 1.3) -> np.ndarray:
    """NES-mixer-ish soft saturation (raises RMS relative to peak)."""
    return np.tanh(drive * np.asarray(x, dtype=np.float64))


def _circ(fn, x: np.ndarray, pad: int = 16384) -> np.ndarray:
    """Apply a (short-impulse-response) linear filter ``fn`` to a loop buffer
    *circularly*, so the loop seam stays seamless."""
    n = len(x)
    pad = min(pad, n)
    ext = np.concatenate([x[n - pad:], x, x[:pad]])
    return fn(ext)[pad:pad + n]


def _resize(x: np.ndarray, n: int) -> np.ndarray:
    """Tile / truncate to exactly n samples."""
    if len(x) == n:
        return x.copy()
    if len(x) > n:
        return x[:n].copy()
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


class _Buf:
    """Mix buffer with optional wrap-around (for seamless loops)."""

    def __init__(self, n: int, wrap: bool = False):
        self.n = int(n)
        self.wrap = wrap
        self.a = np.zeros(self.n)

    def add(self, x, start: int, gain: float = 1.0):
        L = len(x)
        s = int(start)
        if L == 0:
            return
        if not self.wrap:
            if s >= self.n or s + L <= 0:
                return
            lo = max(0, -s)
            hi = min(L, self.n - s)
            self.a[s + lo:s + hi] += gain * x[lo:hi]
        else:
            if L > self.n:
                x = x[:self.n]
                L = self.n
            s %= self.n
            first = min(L, self.n - s)
            self.a[s:s + first] += gain * x[:first]
            if first < L:
                self.a[:L - first] += gain * x[first:]


# ---- note voices -----------------------------------------------------
def _pnote(midi, gate, duty=0.5, vol=1.0, a=0.002, d=0.08, s=0.6, r=0.03, vib=None,
           f_end=None):
    """One pulse-wave note (gate seconds + release r).  vib=(rate, cents, delay)."""
    f0 = float(midi_to_freq(midi))
    n = _ns(gate + r)
    if f_end is not None:
        f = pitch_slide(f0, f_end, n)
    else:
        f = f0
    if vib:
        f = vibrato(f, n, *vib)
    return pulse(f, n, duty) * adsr(n, a, d, s, r) * vol


def _tnote(midi, gate, vol=1.0, a=0.002, d=0.08, s=0.8, r=0.01, steps=16, vib=None):
    """One triangle note (NES stair-stepped by default)."""
    f0 = float(midi_to_freq(midi))
    n = _ns(gate + r)
    f = vibrato(f0, n, *vib) if vib else f0
    return triangle(f, n, steps=steps) * adsr(n, a, d, s, r) * vol


def _bell(midi, dur, vol=1.0, harm=(1.0, 0.28, 0.10, 0.04), tau=0.9):
    """Bell-ish pluck: triangle fundamental + decaying harmonic (2f, 3f, 4f) partials.
    Only harmonic ratios, so the chord stays in tune."""
    f = float(midi_to_freq(midi))
    n = _ns(dur)
    t = _t(n)
    y = triangle(f, n, steps=32) * harm[0] * np.exp(-t / tau)
    for k in range(1, len(harm)):
        if f * (k + 1) < SR * 0.45:
            y += harm[k] * np.sin(TWO_PI * f * (k + 1) * t) * np.exp(-t / (tau / (1.7 * (k + 1) ** 0.8)))
    att = np.minimum(t / 0.002, 1.0)
    rel = np.clip((dur - t) / 0.04, 0.0, 1.0)
    return y * att * rel * vol


def _sparkle(midis, t0s, dur, vol=1.0):
    """Sum of pure-harmonic bell pings placed at offsets; returns array of length
    max(t0 + dur)."""
    n = _ns(max(t0s) + dur)
    y = np.zeros(n)
    for m, t0 in zip(midis, t0s):
        f = float(midi_to_freq(m))
        nn = _ns(dur)
        t = _t(nn)
        p = (np.sin(TWO_PI * f * t) + 0.35 * np.sin(TWO_PI * 2 * f * t) * np.exp(-t / 0.12)) \
            * np.exp(-t / 0.22) * np.minimum(t / 0.001, 1.0) * np.clip((dur - t) / 0.03, 0.0, 1.0)
        s0 = _ns(t0)
        y[s0:s0 + nn] += p[:max(0, n - s0)][:nn] * vol
    return y


# ---- drums (cached, NES-style LFSR noise) ------------------------------
@lru_cache(maxsize=None)
def _drum(kind: str) -> np.ndarray:
    if kind in ("kick", "kick_soft"):
        soft = kind == "kick_soft"
        n = _ns(0.15 if soft else 0.17)
        t = _t(n)
        f = (42.0 if soft else 45.0) + (85.0 if soft else 120.0) * np.exp(-t / 0.028)
        ph = np.cumsum(f) / SR
        body = 0.8 * np.sin(TWO_PI * ph) + 0.2 * (4.0 * np.abs(ph % 1.0 - 0.5) - 1.0)
        click = noise(n, "long", rate=22050, offset=100) * np.exp(-t / 0.003) * (0.15 if soft else 0.5)
        y = body * np.exp(-t / (0.06 if soft else 0.07)) + click
    elif kind in ("snare", "snare_soft"):
        soft = kind == "snare_soft"
        n = _ns(0.11 if soft else 0.16)
        t = _t(n)
        nz = noise(n, "long", rate=(9000 if soft else 13000), offset=700)
        nz = highpass(nz, 500 if soft else 350)
        if soft:
            nz = lowpass(nz, 5500)
        y = nz * np.exp(-t / (0.030 if soft else 0.045)) * (0.55 if soft else 1.0)
        y += 0.5 * np.sin(TWO_PI * (170.0 if soft else 190.0) * t) * np.exp(-t / 0.022)
    elif kind in ("hat", "hat_soft"):
        soft = kind == "hat_soft"
        n = _ns(0.04)
        t = _t(n)
        nz = highpass(noise(n, "short", rate=30000, offset=5), 5500 if not soft else 4500)
        y = nz * np.exp(-t / 0.008) * (0.6 if soft else 1.0)
    elif kind == "open_hat":
        n = _ns(0.16)
        t = _t(n)
        nz = highpass(noise(n, "long", rate=40000, offset=333), 5000)
        y = nz * np.exp(-t / 0.05)
    elif kind in ("crash", "crash_soft"):
        n = _ns(0.9)
        t = _t(n)
        nz = highpass(noise(n, "long", rate=44100, offset=901), 3000)
        y = nz * np.exp(-t / (0.22 if kind == "crash" else 0.32))
        if kind == "crash_soft":
            y = lowpass(y, 9000) * 0.7
    else:
        raise KeyError(kind)
    ramp = np.minimum(1.0, (len(y) - 1 - np.arange(len(y))) / (0.008 * SR))
    return y * ramp


# ======================================================================
# MUSICAL MATERIAL
# ======================================================================
# Lead motif of the battle theme (E natural minor, 1 = E5).  4 bars x 4 beats.
#  bar1 (Em):  B  B  E' D' B  G        bar2 (C):   G  G  C' B  G  E
#  bar3 (D9):  F# A  C'. E' F#' E' D' C'   bar4 (B):  B  A  F# A  B---
MOTIF = [
    (5, 0.75), (5, 0.25), (8, 1.0), (7, 0.5), (5, 0.5), (3, 1.0),
    (3, 0.75), (3, 0.25), (6, 1.0), (5, 0.5), (3, 0.5), (1, 1.0),
    (2, 0.5), (4, 0.5), (6, 0.75), (8, 0.25), (9, 1.0), (8, 0.5), (7, 0.25), (6, 0.25),
    (5, 0.5), (4, 0.5), (2, 0.5), (4, 0.5), (5, 1.5), (None, 0.5),
]
# Varied answer (bars 5-8 of the battle theme): higher, busier, same DNA.
MOTIF_B = [
    (8, 0.75), (8, 0.25), (10, 1.0), (9, 0.5), (8, 0.5), (7, 0.5), (5, 0.5),
    (6, 0.75), (6, 0.25), (8, 1.0), (7, 0.5), (5, 0.5), (3, 1.0),
    (2, 0.25), (4, 0.25), (6, 0.25), (8, 0.25), (9, 1.0), (8, 0.5), (7, 0.5), (6, 1.0),
    (5, 0.5), (4, 0.5), (2, 0.5), (4, 0.5), (5, 0.5), (7, 0.5), (5, 1.0),
]


def _motif_bars(motif, bpb: float = 4.0):
    """Split a motif into per-bar lists of (beat_in_bar, degree_or_None, length)."""
    bars = []
    pos = 0.0
    for deg, ln in motif:
        b = int(pos // bpb + 1e-9)
        while len(bars) <= b:
            bars.append([])
        bars[b].append((pos - b * bpb, deg, ln))
        pos += ln
    return bars


_FORM_MEL = _motif_bars(MOTIF) + _motif_bars(MOTIF_B)     # 8 bars
assert len(_FORM_MEL) == 8


# ======================================================================
# BATTLE THEME  (E natural minor, 152 BPM, 8-bar loop)
# ======================================================================
_BATTLE_BPM = 152.0
# per bar: (bass root MIDI, arp pool)   Em | C | D | Bm || Em | C | D | B7
_B_CHORDS = [
    (40, (55, 59, 64, 67)),
    (36, (55, 60, 64, 67)),
    (38, (57, 62, 66, 69)),
    (35, (59, 62, 66, 71)),
    (40, (55, 59, 64, 67)),
    (36, (55, 60, 64, 67)),
    (38, (57, 62, 66, 69)),
    (35, (59, 63, 66, 69)),
]
_B_ARP_A = [0, 1, 2, 3, 2, 1, 2, 3, 0, 1, 2, 3, 2, 1, 2, 3]
_B_ARP_B = [0, 2, 1, 3, 2, 3, 1, 3, 0, 2, 1, 3, 2, 3, 1, 3]
# drum steps (16ths)
_B_KICK_A = (0, 6, 8, 14)
_B_KICK_B = (0, 3, 6, 8, 10, 14)
_B_SNARE = (4, 12)
_B_HAT_A = (0, 2, 4, 6, 8, 12, 14)
_B_OPEN_A = (10,)


# exact seamless-loop lengths in seconds (call the theme with fade=0 and a duration that
# is a multiple of these if you want to tile it yourself)
LOOP_SECONDS = {"battle_theme": 8 * 4 * 60.0 / 152.0, "debug_theme": 8 * 4 * 60.0 / 88.0}


@lru_cache(maxsize=1)
def _battle_loop() -> np.ndarray:
    spb = 60.0 / _BATTLE_BPM
    sst = spb / 4.0
    ss = SR * sst
    N = int(round(128 * ss))
    mix = _Buf(N, wrap=True)

    def P(bar, st):
        return int(round((bar * 16 + st) * ss))

    for bar in range(8):
        root, pool = _B_CHORDS[bar]
        second = bar >= 4
        # ---- bass: triangle gallop, octave pops; 12.5% pulse doubling in bars 5-8
        pat = [0, 0, 12, 0] * 4
        if bar == 3:
            pat[12:] = [0, 3, 7, 3]
        if bar == 7:
            pat[12:] = [0, 4, 7, 10]
        for st, off in enumerate(pat):
            m = root + off
            mix.add(_tnote(m, sst * 0.88, a=0.001, d=0.05, s=0.85, r=0.008), P(bar, st), 0.55)
            if second:
                mix.add(_pnote(m + 12, sst * 0.8, duty=0.125, a=0.001, d=0.04, s=0.5, r=0.008),
                        P(bar, st), 0.085)
        # ---- 25% pulse arpeggio comping
        pattern = _B_ARP_B if second else _B_ARP_A
        for st, idx in enumerate(pattern):
            acc = 1.0 if st % 4 == 0 else 0.8
            mix.add(_pnote(pool[idx], sst * 0.8, duty=0.25, a=0.001, d=0.04, s=0.5, r=0.008),
                    P(bar, st), 0.16 * acc)
        # ---- drums
        for st in (_B_KICK_B if second else _B_KICK_A):
            mix.add(_drum("kick"), P(bar, st), 0.85)
        snares = list(_B_SNARE)
        if bar == 3:
            snares += [14, 15]
        if bar == 7:
            snares += [10, 11, 13, 14, 15]
        for st in snares:
            mix.add(_drum("snare"), P(bar, st), 0.5 if st in _B_SNARE else 0.38)
        if second:
            hats = [s for s in range(16) if s != 10 and s not in snares]
        else:
            hats = [s for s in _B_HAT_A if s not in snares]
        for st in hats:
            mix.add(_drum("hat"), P(bar, st), 0.20 if st % 4 == 0 else 0.13)
        mix.add(_drum("open_hat"), P(bar, 10), 0.16)
        if bar in (0, 4):
            mix.add(_drum("crash"), P(bar, 0), 0.16)

    # ---- lead: MOTIF (50% duty) bars 1-4, MOTIF_B (25% duty + echo) bars 5-8
    for bar in range(8):
        for beat, deg, ln in _FORM_MEL[bar]:
            if deg is None:
                continue
            m = degree_to_midi(deg, 76, MINOR)
            gate = ln * spb * (0.93 if ln >= 0.5 else 0.88)
            vib = (5.6, 16.0, 0.10) if ln >= 1.0 else None
            duty = 0.25 if bar >= 4 else 0.5
            y = _pnote(m, gate, duty=duty, a=0.002, d=0.12, s=0.72, r=0.02, vib=vib)
            t0 = P(bar, beat * 4)
            mix.add(y, t0, 0.42 if duty == 0.5 else 0.46)
            if bar >= 4:      # SNES-style dotted-8th echo
                mix.add(y, t0 + int(round(3 * ss)), 0.11)
    return mix.a


def battle_theme(duration: float = 25.0, fade: float = 0.35) -> np.ndarray:
    """Boss-fight theme (E minor, 152 BPM, 8-bar loop, seamless loop point).

    ``fade`` seconds of cosine fade-out are applied at the very end; use fade=0 when
    you will apply ``tape_stop`` afterwards so the tail keeps full level."""
    n = _ns(duration)
    x = _softclip(_resize(_battle_loop(), n), 0.85)
    if fade > 0:
        x = _fade(x, 0.0, min(fade, duration))
    return finalize(x, fade_in=0.002, fade_out=0.004 if fade <= 0 else 0.001)


# ======================================================================
# TITLE / READY FANFARE
# ======================================================================
def title_ready(duration: float = 2.4) -> np.ndarray:
    """STAGE START / READY!  rising arp -> gated blink -> snare roll -> E power accent."""
    D = float(duration)
    n = _ns(D)
    mix = _Buf(n)
    u = D / 24.0                                # one slot (0.1 s at 2.4 s)

    # 1) rising E-minor pulse arpeggio, 12 notes over 6 slots
    arp = [52, 55, 59, 64, 67, 71, 76, 79, 83, 88, 91, 95]
    ts = 6 * u / len(arp)
    for i, m in enumerate(arp):
        y = _pnote(m, ts * 0.85, duty=0.5 if i % 2 == 0 else 0.25, a=0.001, d=0.03, s=0.6, r=0.008)
        mix.add(y, _ns(i * ts), 0.35 + 0.02 * i)
    mix.add(_tnote(40, 6 * u, a=0.02, d=1.0, s=1.0, r=0.02), 0, 0.45)
    mix.add(_drum("crash_soft"), 0, 0.20)

    # 2) gated 'READY' blink: 5 blinks (on 1 slot / off 1 slot) on E5/B5 fifths
    blink_notes = [76, 76, 83, 76, 83]
    for k, m in enumerate(blink_notes):
        t0 = (6 + 2 * k) * u
        g = u * 0.86
        mix.add(_pnote(m, g, duty=0.5, a=0.001, d=0.2, s=0.9, r=0.01), _ns(t0), 0.34)
        mix.add(_pnote(m - 12 + 7 if k % 2 else m - 12, g, duty=0.25, a=0.001, d=0.2, s=0.9, r=0.01),
                _ns(t0), 0.22)
        mix.add(_tnote(40 if k % 2 == 0 else 47, g, a=0.001, d=0.5, s=0.9, r=0.01), _ns(t0), 0.5)
        mix.add(_drum("kick"), _ns(t0), 0.75)
        mix.add(_drum("hat"), _ns(t0 + u), 0.28)

    # 3) snare roll + rising run into the accent (slots 16..20)
    run = [71, 74, 76, 79, 83, 86, 88, 91]
    for i, m in enumerate(run):
        t0 = 16 * u + i * u / 2
        mix.add(_pnote(m, u * 0.42, duty=0.25, a=0.001, d=0.05, s=0.7, r=0.006), _ns(t0), 0.30 + 0.02 * i)
        mix.add(_drum("snare_soft" if i < 3 else "snare"), _ns(t0), 0.22 + 0.05 * i)
    mix.add(_tnote(40, 4 * u, a=0.001, d=1.0, s=1.0, r=0.01), _ns(16 * u), 0.45)

    # 4) punchy accent: E power chord (E-B-E) + kick + snare + crash, decaying to the end
    t0 = _ns(20 * u)
    rest = n - t0
    g = rest / SR - 0.06
    for m, duty, vol in ((40, None, 0.75), (52, None, 0.5)):
        mix.add(_tnote(m, g, a=0.001, d=0.4, s=0.5, r=0.06), t0, vol)
    for m, duty, vol in ((64, 0.5, 0.36), (71, 0.25, 0.30), (76, 0.5, 0.32), (83, 0.25, 0.20), (88, 0.125, 0.12)):
        mix.add(_pnote(m, g, duty=duty, a=0.001, d=0.16, s=0.30, r=0.06), t0, vol)
    mix.add(_drum("kick"), t0, 0.95)
    mix.add(_drum("snare"), t0, 0.6)
    mix.add(_drum("crash"), t0, 0.42)
    return finalize(_softclip(mix.a, 1.25), fade_in=0.001, fade_out=0.006)


# ======================================================================
# TAPE STOP
# ======================================================================
def tape_stop(audio, tail: float = 1.2, curve: float = 1.7) -> np.ndarray:
    """Record/tape-stop on the last ``tail`` seconds: playback rate falls 1 -> 0
    (pitch dives, audio slows), amplitude fades with the rate and a low-pass sweeps
    down (darkening).  Returns a new float32 array of the same length; the level of
    the untouched part is preserved."""
    x = np.nan_to_num(np.asarray(audio, dtype=np.float64))
    n = len(x)
    L = min(n, _ns(tail))
    out = x.copy()
    if L >= 32:
        seg = x[n - L:]
        u = np.arange(L) / L
        rate = (1.0 - u) ** curve
        pos = np.cumsum(rate) - rate
        y = np.interp(pos, np.arange(L), seg)
        y = lowpass_sweep(y, 14000.0 * rate ** 1.3 + 120.0, nbank=12)
        y *= rate ** 0.6
        y[-32:] *= np.linspace(1.0, 0.0, 32)
        out[n - L:] = y
    out = np.clip(out, -0.999, 0.999)
    return out.astype(np.float32)


# ======================================================================
# CRASH STING
# ======================================================================
def _stutter(x, rng, seg=(0.03, 0.09), frag=(0.006, 0.03), p=0.75):
    """Repeat tiny fragments of ``x`` to fill random-length segments (buffer glitch)."""
    out = x.copy()
    n = len(x)
    i = 0
    while i < n:
        sl = max(16, int(rng.uniform(*seg) * SR))
        if rng.random() < p:
            fl = max(8, int(rng.uniform(*frag) * SR))
            f = x[i:i + fl]
            if len(f):
                tiled = np.tile(f, int(np.ceil(sl / len(f))))[:sl][:n - i]
                out[i:i + len(tiled)] = tiled
        i += sl
    return out


def crash_sting(duration: float = 4.0) -> np.ndarray:
    """Game-crash sting: boom + noise burst, glitch/stutter with worsening bit-crush,
    a corrupted descending arpeggio that slows and drops in pitch, then a low quiet
    digital static / mains-hum 'void' (ends with RMS ~0.01)."""
    D = float(duration)
    n = _ns(D)
    sc = min(1.0, D / 4.0)
    rng = np.random.default_rng(1337)
    t = _t(n)

    # ---- boom / impact -------------------------------------------------
    boom = np.zeros(n)
    nb = min(n, _ns(1.4 * sc))
    tb = _t(nb)
    f = 30.0 + 95.0 * np.exp(-tb / 0.09)
    boom[:nb] += 1.0 * np.sin(TWO_PI * np.cumsum(f) / SR) * np.exp(-tb / (0.42 * sc + 0.02))
    nn = min(n, _ns(0.6 * sc))
    tn = _t(nn)
    nz = rng.uniform(-1, 1, nn)
    nz = lowpass_sweep(nz, 12000.0 * np.exp(-tn / 0.10) + 300.0)
    boom[:nn] += 1.1 * nz * np.exp(-tn / (0.16 * sc + 0.01))
    nc = min(n, _ns(0.28 * sc))
    crunch = bitcrush(rng.uniform(-1, 1, nc), bits=3, downsample=3)
    boom[:nc] += 0.55 * crunch * np.exp(-_t(nc) / 0.07)
    boom[:_ns(0.004)] += 1.2 * np.sign(rng.uniform(-1, 1, _ns(0.004)))     # crack
    boom = np.tanh(1.8 * boom)

    # ---- corrupted descending arpeggio (slowing, dropping) --------------
    seq = np.zeros(n)
    notes = [88, 83, 79, 76, 71, 67, 64, 59, 55, 52, 47, 43, 40]
    tc = 0.10 * sc
    for i, m in enumerate(notes):
        d = 0.075 * (1.14 ** i) * sc
        nn_ = _ns(d)
        f0 = float(midi_to_freq(m - 0.25 * i))
        fs = pitch_slide(f0, f0 * (0.92 - 0.025 * i), nn_)
        duty = float(rng.choice([0.125, 0.25, 0.5]))
        y = pulse(fs, nn_, duty) * adsr(nn_, 0.001, 0.06, 0.7, 0.01)
        s0 = _ns(tc)
        if s0 < n:
            seg = y[:n - s0]
            seq[s0:s0 + len(seg)] += seg * (0.55 + 0.25 * rng.random())
        tc += d
    seq += 0.35 * rng.uniform(-1, 1, n) * (rng.random(n) < 0.02)
    seq = _stutter(seq, rng)
    prog = np.clip(t / max(2.4 * sc, 1e-3), 0.0, 1.0)
    seq = bitcrush(seq, bits=8.0 - 5.5 * prog, downsample=np.exp(prog * np.log(22.0)))
    gate = np.ones(n)                                    # random dropouts, increasing
    i = 0
    while i < n:
        sl = int(rng.uniform(0.012, 0.05) * SR)
        if rng.random() < 0.10 + 0.5 * min(1.0, i / SR / max(2.2 * sc, 1e-3)):
            gate[i:i + sl] = 0.0
        i += sl
    aenv = np.clip((t - 0.06 * sc) / (0.03 * sc + 1e-3), 0, 1) * np.clip((3.0 * sc - t) / (1.6 * sc + 1e-3), 0, 1)
    seq = seq * gate * aenv * 0.8

    # ---- dissolving static crackle ------------------------------------
    crack = np.zeros(n)
    i = _ns(1.2 * sc)
    while i < min(n, _ns(3.2 * sc)):
        sl = int(rng.uniform(0.004, 0.03) * SR)
        amp = 0.45 * (1.0 - (i / SR - 1.2 * sc) / (2.0 * sc + 1e-3))
        crack[i:i + sl] = highpass(rng.uniform(-1, 1, min(sl, n - i)), 1500) * max(amp, 0.0)
        i += sl + int(rng.uniform(0.01, 0.09) * SR)

    main = boom * 1.0 + seq * 0.62 + crack * 0.5
    main = _fade(main, 0.0, 0.0)
    main *= 0.8 / max(float(np.max(np.abs(main))), 1e-9)

    # ---- the void: hum + faint static ------------------------------------
    hum_env = np.clip((t - 0.7 * sc) / (1.9 * sc + 1e-3), 0.0, 1.0)
    hum_env = hum_env * (0.75 + 0.25 * np.exp(-np.maximum(t - 2.8 * sc, 0.0) / 1.5))
    hum_f = 50.0 * (1.0 - 0.06 * np.exp(-t / 1.0))
    hph = np.cumsum(hum_f) / SR
    hum = (np.sin(TWO_PI * hph) + 0.45 * np.sin(TWO_PI * 2 * hph + 0.6)
           + 0.16 * np.sin(TWO_PI * 3 * hph + 1.1)) / 1.61
    hum *= 1.0 + 0.10 * np.sin(TWO_PI * 0.35 * t)
    st = lowpass(highpass(rng.uniform(-1, 1, n), 700), 3800)
    st = bitcrush(st, bits=5) * (0.6 + 0.4 * np.sin(TWO_PI * 0.7 * t + 1.0) ** 2)
    void = (0.036 * hum + 0.0075 * st) * hum_env
    x = main + void
    return finalize(x, fade_in=0.001, fade_out=0.02)


# ======================================================================
# DEBUG THEME (A minor -> C major, 88 BPM, 8-bar loop, lo-fi, dialogue-friendly)
# ======================================================================
_DEBUG_BPM = 88.0
# bars: Am7 | Fmaj7 | Dm7 | G || Fmaj7 | G | Cmaj7 | Cmaj7
_D_BASS = [45, 41, 38, 43, 41, 43, 48, 48]
_D_PAD = [(52, 55, 60), (52, 57, 60), (53, 57, 60), (50, 55, 59),
          (52, 57, 60), (50, 55, 59), (52, 55, 59), (52, 55, 59)]
_D_ARP = [(57, 60, 64, 67), (53, 57, 60, 64), (57, 60, 62, 65), (55, 59, 62, 67),
          (53, 57, 60, 64), (55, 59, 62, 67), (55, 59, 60, 64), (55, 59, 60, 64)]
# bell melody bars 1-4 (A minor, sparse, sighing): (step, midi, length_in_steps)
_D_BELL_A = [
    [(0, 76, 8), (8, 72, 4), (12, 74, 4)],     # E5 C5 D5          over Am7
    [(0, 72, 8), (8, 69, 4), (12, 72, 4)],     # C5 A4 C5          over Fmaj7
    [(0, 74, 6), (8, 77, 4), (12, 76, 4)],     # D5 F5 E5          over Dm7
    [(0, 74, 8), (8, 71, 8)],                  # D5 B4             over G
]


# mix gains for the debug loop (bass, pad, arps, bells, hat tick) + speech-pocket depth
_DEBUG_MIX = dict(bass=0.36, pad=0.14, arp=0.15, bell=0.30, tick=1.0, pocket=0.7)


@lru_cache(maxsize=4)
def _debug_loop(mixkey=None) -> np.ndarray:
    g = dict(_DEBUG_MIX)
    if mixkey:
        g.update(dict(mixkey))
    spb = 60.0 / _DEBUG_BPM
    sst = spb / 4.0
    ss = SR * sst
    N = int(round(128 * ss))
    P = lambda bar, st: int(round((bar * 16 + st) * ss))          # noqa: E731

    bass = _Buf(N, True)
    pad = _Buf(N, True)
    arp = _Buf(N, True)
    bell = _Buf(N, True)
    tick = _Buf(N, True)

    bass_pat = [(0, 0, 6), (6, 7, 2), (8, 12, 6), (14, 7, 2)]
    for bar in range(8):
        root = _D_BASS[bar]
        for st, off, ln in bass_pat:
            bass.add(_tnote(root + off, sst * ln * 0.9, a=0.006, d=0.35, s=0.7, r=0.05, steps=16),
                     P(bar, st), 1.0)
        for m in _D_PAD[bar]:
            pad.add(_tnote(m, sst * 15.5, a=0.25, d=1.0, s=1.0, r=0.5, steps=32), P(bar, 0), 1.0)
        pool = _D_ARP[bar]
        for k, idx in enumerate([0, 1, 2, 3, 2, 1, 2, 1]):
            duty = 0.125 if (k // 2) % 2 == 0 else 0.25
            arp.add(_pnote(pool[idx], sst * 1.6, duty=duty, a=0.004, d=0.15, s=0.25, r=0.04),
                    P(bar, 2 * k), 1.0 if k % 2 == 0 else 0.7)
        for st in range(0, 16, 2):
            tick.add(_drum("hat_soft"), P(bar, st), 0.05 if st % 4 == 0 else 0.028)

    # bells: bars 1-4 sighing A-minor line
    for bar, notes in enumerate(_D_BELL_A):
        for st, m, ln in notes:
            bell.add(_bell(m, sst * ln * 1.15 + 0.6, tau=0.7), P(bar, st), 1.0)
    # bars 5-8: the BATTLE MOTIF in slow motion, in C major (hopeful), resolves on C5
    pos = 0.0
    for deg, ln in MOTIF[:12]:               # bars 1-2 of the motif, stretched x2
        if deg is not None:
            m = degree_to_midi(deg, 72, MAJOR)
            st = 4 * 16 + int(round(pos * 2 * 4))
            bell.add(_bell(m, ln * 2 * spb * 1.05 + 0.7, tau=0.85), P(0, st), 1.0)
        pos += ln

    def mid_eq(x):
        band = lowpass(x, 3000.0) - lowpass(x, 300.0)
        return x - g["pocket"] * band

    arpc = _circ(lambda z: lowpass(z, 1500.0, 2), arp.a)
    arpc = echo(arpc, 0.75 * spb, 0.40, 0.42, 4, circular=True)
    bellc = echo(bell.a, 0.75 * spb, 0.40, 0.30, 4, circular=True)
    mix = g["bass"] * bass.a + g["pad"] * pad.a + g["arp"] * arpc + g["bell"] * bellc + g["tick"] * tick.a
    mix = _circ(mid_eq, mix)                       # leave room for dialogue
    mix = _circ(lambda z: lowpass(z, 6500.0, 2), mix)
    mix = bitcrush(mix / max(np.max(np.abs(mix)), 1e-9) * 0.9, bits=11)
    return mix


def debug_theme(duration: float = 24.0, fade: float = 0.8) -> np.ndarray:
    """Calm lo-fi debugging theme (A minor -> C major, 88 BPM, seamless 8-bar loop).
    Low mid-range energy so it sits under speech.  ``fade`` = end fade-out seconds."""
    n = _ns(duration)
    x = _resize(_debug_loop(), n)
    if fade > 0:
        x = _fade(x, 0.0, min(fade, duration))
    return finalize(x, fade_in=0.004, fade_out=0.004)


# ======================================================================
# FRIENDSHIP THEME (same motif, E major, ~112 BPM, builds to a big E chord)
# ======================================================================
_FRIEND_BPM = 112.0
# per form bar: (bass root, arp pool)      E | C#m | F#m7 | B7  (x2)
_F_CHORDS = [
    (40, (56, 59, 64, 68)),
    (37, (56, 61, 64, 68)),
    (42, (57, 61, 64, 66)),
    (35, (59, 63, 66, 69)),
] * 2


def friendship_theme(duration: float = 20.0) -> np.ndarray:
    """The battle motif, happy: E major, I - vi - ii - V (x2), builds (arps -> drums ->
    16th arps -> harmony thirds -> snare roll) and lands on a big E major chord with a
    bell sparkle.  Tempo is ~112 BPM, adjusted slightly so whole bars fill ``duration``."""
    D = float(duration)
    n = _ns(D)
    bar_nom = 4 * 60.0 / _FRIEND_BPM
    n_tot = max(2, int(round(D / bar_nom)))
    bar = D / n_tot                     # seconds per bar (final bar = big chord)
    spb = bar / 4.0
    sst = spb / 4.0
    b = n_tot - 1                       # body bars
    mix = _Buf(n)
    lead = _Buf(n)

    def T(k, st=0.0):
        return _ns(k * bar + st * sst)

    drum_start = max(1, int(round(0.28 * b)))
    harm_start = int(round(0.55 * b))
    sixt_start = int(round(0.62 * b))
    for k in range(b):
        p = k / max(b - 1, 1)
        fi = 7 if k == b - 1 else k % 8
        root, pool = _F_CHORDS[fi]
        g = 0.72 + 0.28 * p
        # ---- triangle bass: root-5-8-5 eighths
        for j, off in enumerate([0, 7, 12, 7, 0, 7, 12, 7]):
            mix.add(_tnote(root + off, sst * 1.85, a=0.003, d=0.2, s=0.7, r=0.03),
                    T(k, 2 * j), 0.5 * g)
        # ---- gentle arps (25% / 12.5% pulses)
        if k >= sixt_start:
            steps, patt = 16, [0, 1, 2, 3, 2, 1, 2, 3, 0, 1, 2, 3, 2, 3, 1, 2]
        else:
            steps, patt = 8, [0, 1, 2, 3, 2, 1, 2, 3]
        for j, idx in enumerate(patt):
            st = j * (16 // steps)
            duty = 0.25 if j % 2 == 0 else 0.125
            mix.add(_pnote(pool[idx], sst * (1.7 if steps == 8 else 0.85), duty=duty,
                           a=0.002, d=0.08, s=0.35, r=0.02), T(k, st), 0.15 * g * (1 if j % 2 == 0 else 0.75))
        # ---- lead: the battle motif (triangle + soft pulse), MOTIF bars 0-3, MOTIF_B 4-7
        for beat, deg, ln in _FORM_MEL[fi]:
            if deg is None:
                continue
            m = degree_to_midi(deg, 76, MAJOR)
            gate = ln * spb * (0.95 if ln >= 0.5 else 0.9)
            vib = (5.2, 13.0, 0.14) if ln >= 1.0 else None
            y = _tnote(m, gate, a=0.004, d=0.15, s=0.85, r=0.03, steps=32, vib=vib) * 1.0
            y = y + _pnote(m, gate, duty=0.25, vol=0.30, a=0.004, d=0.12, s=0.7, r=0.03, vib=vib)
            lead.add(y, T(k, beat * 4), 0.50 * (0.85 + 0.15 * p))
            if k >= harm_start:                          # a diatonic third below
                m2 = degree_to_midi(deg - 2, 76, MAJOR)
                y2 = _tnote(m2, gate, a=0.004, d=0.15, s=0.85, r=0.03, steps=32, vib=vib)
                lead.add(y2, T(k, beat * 4), 0.24)
        # ---- soft drums (enter after the first phrase(s))
        if k >= drum_start:
            for st in (0, 8) if k < harm_start else (0, 6, 8, 14):
                mix.add(_drum("kick_soft"), T(k, st), 0.55 * g)
            last = (k == b - 1)
            for st in (4, 12):
                if last and st == 12:
                    continue
                mix.add(_drum("snare_soft"), T(k, st), 0.36 * g)
            if last:                                     # crescendo snare roll into the chord
                for j in range(8):
                    mix.add(_drum("snare_soft"), T(k, 8 + j), 0.20 + 0.09 * j)
            else:
                hop = 2 if k < sixt_start else 1
                for st in range(0, 16, hop):
                    mix.add(_drum("hat_soft"), T(k, st), (0.16 if st % 4 == 0 else 0.10) * g)
        if k == 0 or (k % 8 == 0 and k > 0):
            mix.add(_drum("crash_soft"), T(k, 0), 0.10)

    # ---- final bar: big resolved E major chord + bell sparkle ----------
    t0 = T(b)
    rest = n - t0
    g_len = rest / SR - 0.02
    hold = dict(a=0.004, d=g_len * 0.8, s=0.45, r=min(0.6, g_len * 0.35))
    mix.add(_tnote(40, g_len, vol=1.0, steps=16, **hold), t0, 0.75)      # E2
    mix.add(_tnote(52, g_len, vol=1.0, steps=16, **hold), t0, 0.40)      # E3
    for m, duty, vol in ((59, 0.25, 0.20), (64, 0.5, 0.26), (68, 0.25, 0.24),
                         (71, 0.5, 0.24), (76, 0.25, 0.18)):             # B3 E4 G#4 B4 E5
        mix.add(_pnote(m, g_len, duty=duty, vol=1.0, **hold), t0, vol)
    ly = _tnote(88, g_len, a=0.004, d=g_len * 0.9, s=0.5, r=min(0.6, g_len * 0.35), steps=32,
                vib=(5.2, 12.0, 0.25))
    ly = ly + _pnote(88, g_len, duty=0.25, vol=0.28, a=0.004, d=g_len * 0.9, s=0.5,
                     r=min(0.6, g_len * 0.35), vib=(5.2, 12.0, 0.25))
    lead.add(ly, t0, 0.42)
    mix.add(_drum("kick_soft"), t0, 0.7)
    mix.add(_drum("crash_soft"), t0, 0.32)
    sp = _sparkle([80, 83, 88, 92, 95, 100], [0.0, 0.07, 0.14, 0.24, 0.36, 0.5], 0.9, 0.55)
    if len(sp) > rest:
        sp = sp[:rest]
    mix.add(sp * np.clip(np.linspace(1.6, 0.5, len(sp)), 0, 1.6) * 0.5, t0 + _ns(0.05), 0.42)

    body = mix.a + lead.a
    body = chorus(lowpass(body, 7500.0), depth_ms=2.0, rate=0.5, mix=0.28)
    x = _softclip(body, 1.15)
    fade_len = min(0.35, rest / SR * 0.5)
    x = _fade(x, 0.0, fade_len)
    return finalize(x, fade_in=0.004, fade_out=0.004)


# ======================================================================
# SFX
# ======================================================================
def sfx_shoot_byte() -> np.ndarray:
    """~0.18 s bright rising-then-falling pulse 'pew' (hero blaster)."""
    D = 0.18
    n = _ns(D)
    u = _t(n) / D
    f = np.where(u < 0.35, 500.0 * (1900.0 / 500.0) ** (u / 0.35),
                 1900.0 * (700.0 / 1900.0) ** ((u - 0.35) / 0.65))
    y = pulse(f, n, 0.5 - 0.3 * u) + 0.3 * pulse(f * 2.0, n, 0.25)
    y *= adsr(n, 0.001, 0.12, 0.45, 0.04)
    return finalize(y, fade_in=0.001, fade_out=0.008)


def sfx_shoot_null() -> np.ndarray:
    """~0.22 s lower, edgier, slightly noisy falling 'zap' (villain blaster)."""
    D = 0.22
    n = _ns(D)
    t = _t(n)
    u = t / D
    f = 520.0 * (90.0 / 520.0) ** (u ** 0.8)
    f = f * (1.0 + 0.05 * np.sign(np.sin(TWO_PI * 38.0 * t)))
    y = 0.7 * saw(f, n) + 0.5 * pulse(f * 0.5, n, 0.125)
    y += 0.30 * noise(n, "short", rate=2500.0 + 6000.0 * (1.0 - u))
    y = bitcrush(y, bits=6, downsample=2)
    y *= adsr(n, 0.001, 0.15, 0.4, 0.05)
    return finalize(y, fade_in=0.001, fade_out=0.008)


def sfx_hit() -> np.ndarray:
    """~0.2 s noisy impact with a pitch drop."""
    D = 0.20
    n = _ns(D)
    t = _t(n)
    nz = noise(n, "long", rate=np.linspace(30000.0, 3000.0, n)) * np.exp(-t / 0.05)
    thump = np.sin(TWO_PI * np.cumsum(55.0 + 220.0 * np.exp(-t / 0.04)) / SR) * np.exp(-t / 0.07)
    zap = pulse(pitch_slide(700.0, 90.0, n), n, 0.25) * np.exp(-t / 0.035)
    y = 0.9 * nz + 0.8 * thump + 0.4 * zap
    y = bitcrush(y, bits=7, downsample=2)
    y *= adsr(n, 0.0005, 0.2, 0.3, 0.03)
    return finalize(y, fade_in=0.0005, fade_out=0.008)


def sfx_block_clang() -> np.ndarray:
    """~0.2 s metallic ping (two shots colliding)."""
    D = 0.20
    n = _ns(D)
    t = _t(n)
    f0 = 1250.0
    y = np.zeros(n)
    for r, a, tau in ((1.0, 1.0, 0.09), (2.76, 0.6, 0.05), (5.40, 0.4, 0.03), (8.93, 0.25, 0.02)):
        y += a * np.sin(TWO_PI * f0 * r * t) * np.exp(-t / tau)
    y += 0.8 * noise(n, "white", seed=11) * np.exp(-t / 0.002)
    y *= 1.0 + 0.25 * np.sin(TWO_PI * 90.0 * t) * np.exp(-t / 0.05)
    y *= np.clip((D - t) / 0.02, 0.0, 1.0)
    return finalize(y, fade_in=0.0004, fade_out=0.006)


def sfx_jump() -> np.ndarray:
    """~0.2 s rising blip."""
    D = 0.20
    n = _ns(D)
    f = pitch_slide(260.0, 900.0, n, power=0.8)
    y = pulse(f, n, 0.25) + 0.25 * pulse(f * 2, n, 0.5)
    y *= adsr(n, 0.002, 0.12, 0.6, 0.04)
    return finalize(y, fade_in=0.001, fade_out=0.008)


def sfx_land() -> np.ndarray:
    """~0.1 s soft thud."""
    D = 0.10
    n = _ns(D)
    t = _t(n)
    body = triangle(55.0 + 90.0 * np.exp(-t / 0.025), n) * np.exp(-t / 0.03)
    dust = lowpass(noise(n, "white", seed=3), 900.0) * np.exp(-t / 0.02)
    y = body + 0.6 * dust
    y *= np.clip((D - t) / 0.015, 0.0, 1.0)
    return finalize(y, fade_in=0.0008, fade_out=0.006)


def sfx_dash() -> np.ndarray:
    """~0.25 s filtered-noise whoosh."""
    D = 0.25
    n = _ns(D)
    t = _t(n)
    u = t / D
    nz = noise(n, "white", seed=5)
    fc = 800.0 * (6000.0 / 800.0) ** np.sin(np.pi * np.clip(u, 0, 1) * 0.5) * (1.0 - 0.55 * u ** 2)
    y = highpass(lowpass_sweep(nz, fc, order=2), 450.0)
    y *= np.sin(np.pi * u ** 0.6) ** 1.2
    return finalize(y, fade_in=0.004, fade_out=0.01)


def sfx_charge(duration: float = 1.6) -> np.ndarray:
    """Rising power-up: pulse sweep + accelerating shimmering arpeggio + growing noise;
    ends bright with an E-major ping."""
    D = float(duration)
    n = _ns(D)
    t = _t(n)
    u = t / D
    sweep = pulse(110.0 * (1760.0 / 110.0) ** (u ** 1.5), n, 0.25) * (0.25 + 0.4 * u)
    rate = 10.0 + 32.0 * u ** 1.3
    stepi = np.floor(np.cumsum(rate) / SR).astype(int)
    base = np.array([64, 67, 71, 76])
    m = base[stepi % 4] + 12 * np.minimum((u * 2.99).astype(int), 2)
    arp = pulse(midi_to_freq(m), n, 0.125) * (0.45 + 0.55 * u) * (0.75 + 0.25 * np.sin(TWO_PI * rate * t / 4))
    nz = highpass(noise(n, "white", seed=7), 1500.0) * (u ** 3) * 0.55
    y = sweep + 0.55 * arp + nz
    ne = min(n, _ns(0.14))
    te = _t(ne)
    ping = sum(np.sin(TWO_PI * midi_to_freq(mm) * te) * a for mm, a in ((88, 1.0), (92, 0.7), (95, 0.6), (100, 0.4)))
    y[-ne:] += 0.9 * ping * np.exp(-te / 0.07) * np.clip((0.14 - te) / 0.03, 0, 1)
    y *= np.minimum(t / 0.01, 1.0)
    return finalize(y, fade_in=0.004, fade_out=0.006)


def sfx_clash() -> np.ndarray:
    """~0.5 s huge white flash: layered noise boom + high crash + low thud."""
    D = 0.50
    n = _ns(D)
    t = _t(n)
    rng = np.random.default_rng(21)
    boom = lowpass_sweep(rng.uniform(-1, 1, n), 9000.0 * np.exp(-t / 0.12) + 250.0, order=2) * np.exp(-t / 0.20)
    crash = highpass(rng.uniform(-1, 1, n), 4000.0) * np.exp(-t / 0.16)
    metal = sum(np.sin(TWO_PI * 1800.0 * r * t) * a for r, a in ((1, 1.0), (2.76, 0.6), (5.4, 0.35))) * np.exp(-t / 0.10) * 0.3
    thud = np.sin(TWO_PI * np.cumsum(38.0 + 90.0 * np.exp(-t / 0.05)) / SR) * np.exp(-t / 0.22)
    flash = rng.uniform(-1, 1, n) * np.exp(-t / 0.004)
    y = 1.0 * boom + 0.7 * crash + metal + 1.0 * thud + 1.2 * flash
    y = np.tanh(1.6 * y)
    y *= np.clip((D - t) / 0.06, 0.0, 1.0)
    return finalize(y, fade_in=0.0004, fade_out=0.01)


def _click(rng, pitch: float = 1.0) -> np.ndarray:
    n = _ns(0.030)
    t = _t(n)
    body = np.sin(TWO_PI * 1900.0 * pitch * t) * np.exp(-t / 0.004) \
        + 0.6 * np.sin(TWO_PI * 3100.0 * pitch * t) * np.exp(-t / 0.003)
    tick = lowpass(highpass(rng.uniform(-1, 1, n), 1200.0), 6500.0) * np.exp(-t / 0.0018)
    return (0.7 * body + 0.8 * tick) * np.clip((0.030 - t) / 0.006, 0.0, 1.0)


def sfx_type_click() -> np.ndarray:
    """~0.03 s tiny keyboard click."""
    return finalize(_click(np.random.default_rng(31)), fade_in=0.0002, fade_out=0.004)


def sfx_type_burst(n: int = 8, cps: float = 14.0) -> np.ndarray:
    """``n`` rapid keyboard clicks at ``cps`` clicks/s with timing + pitch jitter."""
    rng = np.random.default_rng(32)
    n = max(1, int(n))
    total = _ns((n - 1) / cps + 0.05)
    mix = _Buf(total)
    for i in range(n):
        jit = rng.uniform(-0.18, 0.18) / cps if i else 0.0
        c = _click(rng, pitch=rng.uniform(0.85, 1.2)) * rng.uniform(0.75, 1.0)
        mix.add(c, _ns(max(0.0, i / cps + jit)))
    return finalize(mix.a, fade_in=0.0002, fade_out=0.004)


def sfx_error_beep() -> np.ndarray:
    """~0.4 s classic two-tone error beep (E5 -> B-flat4, a tritone down)."""
    mix = _Buf(_ns(0.40))
    mix.add(_pnote(76, 0.155, duty=0.5, a=0.001, d=0.3, s=0.9, r=0.012), 0, 1.0)
    mix.add(_pnote(70, 0.180, duty=0.5, a=0.001, d=0.3, s=0.9, r=0.012), _ns(0.20), 1.0)
    return finalize(mix.a, fade_in=0.001, fade_out=0.006)


def sfx_confirm() -> np.ndarray:
    """~0.35 s happy two-note 'success / compile OK' ding (C6 -> G6)."""
    mix = _Buf(_ns(0.35))
    mix.add(_pnote(84, 0.085, duty=0.5, a=0.001, d=0.1, s=0.7, r=0.012), 0, 0.8)
    mix.add(_pnote(96, 0.085, duty=0.125, a=0.001, d=0.1, s=0.7, r=0.012), 0, 0.25)
    mix.add(_pnote(91, 0.22, duty=0.5, a=0.001, d=0.16, s=0.35, r=0.03), _ns(0.10), 1.0)
    mix.add(_pnote(103, 0.20, duty=0.125, a=0.001, d=0.12, s=0.3, r=0.03), _ns(0.10), 0.25)
    return finalize(mix.a, fade_in=0.001, fade_out=0.008)


def sfx_powerup_join() -> np.ndarray:
    """~1.0 s cheerful ascending E-major arpeggio + sparkle ('PLAYER 2 HAS JOINED')."""
    n = _ns(1.0)
    mix = _Buf(n)
    arp = [64, 68, 71, 76, 80, 83, 88, 92]
    for i, m in enumerate(arp):
        mix.add(_pnote(m, 0.05, duty=0.25 if i % 2 else 0.5, a=0.001, d=0.05, s=0.7, r=0.008),
                _ns(i * 0.055), 0.6 + 0.04 * i)
    t0 = _ns(0.50)
    for m, duty, vol in ((76, 0.5, 0.4), (80, 0.25, 0.32), (83, 0.5, 0.32), (88, 0.25, 0.30)):
        mix.add(_pnote(m, 0.40, duty=duty, a=0.001, d=0.25, s=0.35, r=0.09), t0, vol)
    sp = _sparkle([100, 104, 107, 112, 100], [0.0, 0.07, 0.14, 0.21, 0.30], 0.35, 0.35)
    mix.add(sp, _ns(0.52))
    return finalize(mix.a, fade_in=0.001, fade_out=0.02)


def sfx_text_blip(pitch_hz: float = 440.0, duration: float = 0.045) -> np.ndarray:
    """Short square-wave dialogue blip at ``pitch_hz``."""
    n = _ns(duration)
    y = pulse(float(pitch_hz), n, 0.5) * adsr(n, 0.001, 0.02, 0.75, min(0.008, duration * 0.3))
    return finalize(y, fade_in=0.0005, fade_out=0.004)


def sfx_glitch_burst(duration: float = 0.6) -> np.ndarray:
    """Nasty digital stutter / bit-crushed screech for screen glitches."""
    D = float(duration)
    n = _ns(D)
    rng = np.random.default_rng(66)
    f = np.zeros(n)
    duty = np.zeros(n)
    i = 0
    while i < n:
        sl = int(rng.uniform(0.012, 0.045) * SR)
        f[i:i + sl] = np.exp(rng.uniform(np.log(200.0), np.log(7000.0)))
        duty[i:i + sl] = rng.choice([0.125, 0.25, 0.5])
        i += sl
    y = pulse(f, n, duty)
    nz = np.zeros(n)
    i = 0
    while i < n:
        sl = int(rng.uniform(0.006, 0.03) * SR)
        if rng.random() < 0.35:
            nz[i:i + sl] = rng.uniform(-1, 1, min(sl, n - i))
        i += sl
    y = _stutter(y + 0.8 * nz, rng, seg=(0.025, 0.08), frag=(0.004, 0.02), p=0.85)
    prog = np.clip(_t(n) / D, 0, 1)
    y = bitcrush(y, bits=7.0 - 4.0 * prog, downsample=np.exp(prog * np.log(14.0)))
    gate = np.ones(n)
    i = 0
    while i < n:
        sl = int(rng.uniform(0.005, 0.02) * SR)
        if rng.random() < 0.18:
            gate[i:i + sl] = 0.0
        i += sl
    y = np.tanh(1.5 * y) * gate
    return finalize(y, fade_in=0.001, fade_out=0.006)


def sfx_static(duration: float = 1.0, level: float = 0.12, loop: bool = True) -> np.ndarray:
    """Filtered static bed (quiet: PEAK == ``level``).  loop=True -> seamless tiling
    (circular filtering, integer-cycle flutter, no edge fades); loop=False -> faded."""
    n = _ns(duration)
    rng = np.random.default_rng(88)
    w = rng.uniform(-1, 1, n)
    x = _circ(lambda z: highpass(lowpass(z, 6500.0, 2), 90.0), w)
    crushed = bitcrush(x, bits=5)
    t = _t(n)
    D = max(duration, 1e-3)
    k1 = max(1, int(round(D * 1.7)))
    k2 = max(1, int(round(D * 4.3)))
    flutter = 1.0 + 0.15 * np.sin(TWO_PI * k1 * t / D) + 0.10 * np.sin(TWO_PI * k2 * t / D + 1.0)
    y = (0.85 * x + 0.30 * crushed) * flutter
    y = np.nan_to_num(y)
    m = float(np.max(np.abs(y))) if n else 0.0
    if not loop:
        y = _fade(y, 0.004, 0.005)
    if m > 1e-12:
        y = y * (level / max(float(np.max(np.abs(y))), 1e-12))
    return y.astype(np.float32)


def sfx_ui_bleep() -> np.ndarray:
    """~0.1 s tiny menu bleep (two quick pips)."""
    mix = _Buf(_ns(0.10))
    mix.add(_pnote(84, 0.040, duty=0.25, a=0.001, d=0.05, s=0.7, r=0.008), 0, 1.0)
    mix.add(_pnote(91, 0.038, duty=0.25, a=0.001, d=0.05, s=0.6, r=0.010), _ns(0.052), 1.0)
    return finalize(mix.a, fade_in=0.0008, fade_out=0.005)


# ======================================================================
# DEMO RENDER
# ======================================================================
_DEMO_DIR = "/tmp/claude-0/-home-user-pixel-art-friendship/b697c2ac-0c37-551d-aa66-9d4b10b64655/scratchpad/chiptune_demo"


def _write_wav(path: str, x: np.ndarray) -> None:
    pcm = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    data = (pcm * 32767.0).astype("<i2").tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data)


def _report(name: str, x: np.ndarray, expect_n: int | None = None) -> bool:
    ok = (x.dtype == np.float32 and x.ndim == 1 and np.all(np.isfinite(x)))
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0
    clip = peak >= 0.999
    len_ok = expect_n is None or len(x) == expect_n
    good = ok and not clip and len_ok
    print("%-22s dur=%7.3fs  peak=%.3f  rms=%.3f  finite=%s  clip=%s  len_ok=%s  -> %s" % (
        name, len(x) / SR, peak, rms, bool(np.all(np.isfinite(x))), clip, len_ok,
        "OK" if good else "PROBLEM"))
    return good


def main(out_dir: str = _DEMO_DIR) -> int:
    os.makedirs(out_dir, exist_ok=True)
    assets = [
        ("title_ready", title_ready(2.4), _ns(2.4)),
        ("battle_theme", battle_theme(25.0), _ns(25.0)),
        ("battle_theme_nofade_8s", battle_theme(8.0, fade=0.0), _ns(8.0)),
        ("battle_tapestop_8s", tape_stop(battle_theme(8.0, fade=0.0), 1.2), _ns(8.0)),
        ("crash_sting", crash_sting(4.0), _ns(4.0)),
        ("debug_theme", debug_theme(24.0), _ns(24.0)),
        ("friendship_theme", friendship_theme(20.0), _ns(20.0)),
        ("friendship_theme_10s", friendship_theme(10.0), _ns(10.0)),
        ("sfx_shoot_byte", sfx_shoot_byte(), None),
        ("sfx_shoot_null", sfx_shoot_null(), None),
        ("sfx_hit", sfx_hit(), None),
        ("sfx_block_clang", sfx_block_clang(), None),
        ("sfx_jump", sfx_jump(), None),
        ("sfx_land", sfx_land(), None),
        ("sfx_dash", sfx_dash(), None),
        ("sfx_charge", sfx_charge(1.6), _ns(1.6)),
        ("sfx_clash", sfx_clash(), None),
        ("sfx_type_click", sfx_type_click(), None),
        ("sfx_type_burst", sfx_type_burst(8, 14), None),
        ("sfx_error_beep", sfx_error_beep(), None),
        ("sfx_confirm", sfx_confirm(), None),
        ("sfx_powerup_join", sfx_powerup_join(), None),
        ("sfx_text_blip", sfx_text_blip(), None),
        ("sfx_glitch_burst", sfx_glitch_burst(0.6), _ns(0.6)),
        ("sfx_static", sfx_static(3.0), _ns(3.0)),
        ("sfx_ui_bleep", sfx_ui_bleep(), None),
    ]
    all_ok = True
    for name, x, exp_n in assets:
        all_ok &= _report(name, x, exp_n)
        _write_wav(os.path.join(out_dir, name + ".wav"), x)
    # a rough 'whole film' audio strip for auditioning the transitions
    seq = np.concatenate([
        title_ready(2.4),
        tape_stop(battle_theme(9.0, fade=0.0), 1.2),
        crash_sting(4.0),
        debug_theme(12.0),
        friendship_theme(14.0),
    ])
    all_ok &= _report("demo_sequence", seq.astype(np.float32))
    _write_wav(os.path.join(out_dir, "demo_sequence.wav"), seq)
    print("ALL OK (no clipping / NaN / inf, lengths exact)" if all_ok else "SOME PROBLEMS - see above")
    print("WAVs written to", out_dir)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else _DEMO_DIR))
