#!/usr/bin/env python3
"""Render BYTE vs NULL end-to-end: python build.py [--credit-model "NAME"] [--out out/x.mp4]

    voice takes (assets/voice)  ->  timeline  ->  music+SFX+dialogue mix  ->  frames  ->  mp4
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

import imageio_ffmpeg

from friends import fx, mix
from friends.gfx import FPS, SCALE, upscale
from friends.scenes import Film
from friends.timeline import build as build_timeline
from friends.voice import load_lines

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "byte-vs-null.mp4"))
    ap.add_argument("--credit-model", default="CLAUDE",
                    help="name shown in the 'MADE WITH ...' credit on the title + end cards (default: CLAUDE)")
    ap.add_argument("--crf", type=int, default=19, help="x264 quality (lower = bigger/better); 19 keeps the 67 s video under 30 MB")
    ap.add_argument("--audio-only", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None, help="render only the first N seconds (quick test)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    lines = load_lines()
    D = {k: v["dur"] for k, v in lines.items()}
    C = build_timeline(D)
    film = Film(lines, C, D, credit=args.credit_model)
    total = C["end"] if args.max_seconds is None else min(C["end"], args.max_seconds)
    print(f"timeline: {C['end']:.2f}s ({int(np.ceil(C['end'] * FPS))} frames @ {FPS} fps)")

    wav = os.path.splitext(args.out)[0] + ".wav"
    stats = mix.render(lines, C, D, film.events(), C["end"], wav)
    print("audio:", {k: round(v, 2) for k, v in stats.items()})
    if args.audio_only:
        return

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    n = int(np.ceil(total * FPS))
    cmd = [ff, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{320 * SCALE}x{180 * SCALE}", "-r", str(FPS), "-i", "-",
           "-i", wav,
           "-map", "0:v", "-map", "1:a", "-t", f"{total:.3f}",
           "-c:v", "libx264", "-preset", "slow", "-tune", "animation", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
           "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
           "-movflags", "+faststart", args.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(n):
        f = film.frame(i / FPS)
        p.stdin.write(fx.crt_look(upscale(f)).tobytes())
        if i % 150 == 0:
            print(f"  frame {i}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed")
    os.remove(wav)
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
