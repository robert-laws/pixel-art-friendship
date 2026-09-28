"""Load, clean and level the ElevenLabs (Eleven v4) dialogue takes.

Each take is decoded with ffmpeg, trimmed of leading/trailing silence, long
internal pauses are shortened (keeps the deadpan beats but saves runtime) and
the loudness of every speaker is matched.
"""
import json
import os
import subprocess

import numpy as np

import imageio_ffmpeg

SR = 44100
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(ROOT, "assets", "voice")

MAX_PAUSE = 0.30      # any silent gap longer than PAUSE_TRIGGER is cut down to this (s)
PAUSE_TRIGGER = 0.38
TARGET_RMS = 0.115    # speech loudness target (linear)


def decode(path):
    r = subprocess.run(
        [FFMPEG, "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def _envelope(x, win_s=0.01):
    win = int(win_s * SR)
    return np.sqrt(np.convolve(x * x, np.ones(win) / win, mode="same"))


def clean(x):
    env = _envelope(x)
    thr = max(0.012, env.max() * 0.04)
    act = env > thr
    idx = np.where(act)[0]
    a = max(0, idx[0] - int(0.04 * SR))
    b = min(len(x), idx[-1] + int(0.09 * SR))
    x, act = x[a:b], act[a:b]

    # shorten long internal pauses
    out, i, n = [], 0, len(x)
    fade = int(0.012 * SR)
    while i < n:
        j = i
        if not act[i]:
            while j < n and not act[j]:
                j += 1
            run = j - i
            if run > PAUSE_TRIGGER * SR and i > 0 and j < n:
                keep = int(MAX_PAUSE * SR)
                seg = x[i:i + keep].copy()
                seg[:fade] *= np.linspace(1, 0, fade)
                seg[-fade:] *= np.linspace(0, 1, fade)
                out.append(seg)
            else:
                out.append(x[i:j])
        else:
            while j < n and act[j]:
                j += 1
            out.append(x[i:j])
        i = j
    y = np.concatenate(out)
    # tiny edge fades
    f = int(0.006 * SR)
    y[:f] *= np.linspace(0, 1, f)
    y[-f:] *= np.linspace(1, 0, f)
    return y


def level(y):
    env = _envelope(y)
    active = y[env > max(0.012, env.max() * 0.08)]
    rms = np.sqrt(np.mean(active ** 2)) if len(active) else 1.0
    g = TARGET_RMS / max(rms, 1e-6)
    y = y * g
    peak = np.abs(y).max()
    if peak > 0.92:   # gentle tanh limiter
        y = np.tanh(y / peak * 1.2) / np.tanh(1.2) * 0.92
    return y.astype(np.float32)


def load_lines():
    """Return {line_id: {'speaker','audio','dur','text'}} from assets/voice/manifest.json."""
    with open(os.path.join(VOICE_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    lines = {}
    for e in manifest["lines"]:
        raw = decode(os.path.join(VOICE_DIR, e["file"]))
        y = level(clean(raw))
        lines[e["id"]] = dict(speaker=e["speaker"], audio=y, dur=len(y) / SR,
                              text=e["caption"], tagged=e["prompt"])
    return lines


if __name__ == "__main__":
    L = load_lines()
    tot = 0
    for k, v in L.items():
        tot += v["dur"]
        print(f'{k} {v["speaker"]:5s} {v["dur"]:5.2f}s  {v["text"]}')
    print(f"total speech {tot:.1f}s")
