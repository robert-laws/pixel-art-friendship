#!/usr/bin/env python3
"""Render BYTE vs NULL end-to-end: python build.py [--format landscape|story] [--credit-model "NAME"] [--out x.mp4]

    landscape  1280x720   full cut (~67 s)
    story      1080x1920  Instagram Story cut, 9:16, under 60 s (re-staged for portrait, safe zones respected)

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
from friends.gfx import FPS, H as LH, SCALE, W as LW, upscale
from friends.scenes import Film
from friends.story import SH, SSCALE, SW, StoryFilm
from friends.timeline import STORY_KEEP, build as build_timeline, build_story
from friends.voice import load_lines

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["landscape", "story"], default="landscape")
    ap.add_argument("--out", default=None, help="default: out/byte-vs-null.mp4 (landscape) or out/byte-vs-null-story.mp4")
    ap.add_argument("--credit-model", default="CLAUDE",
                    help="name shown in the 'MADE WITH ...' credit on the title + end cards (default: CLAUDE)")
    ap.add_argument("--crf", type=int, default=None,
                    help="x264 quality (lower = bigger/better). Default 19 for landscape, 22 for story: keeps each under ~30 MB")
    ap.add_argument("--audio-only", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=None, help="render only the first N seconds (quick test)")
    args = ap.parse_args()
    story = args.format == "story"
    if args.crf is None:
        args.crf = 22 if story else 19
    args.out = args.out or os.path.join(ROOT, "out", "byte-vs-null-story.mp4" if story else "byte-vs-null.mp4")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    lines = load_lines()
    if story:
        lines = {k: lines[k] for k in STORY_KEEP}
    D = {k: v["dur"] for k, v in lines.items()}
    C = build_story(D) if story else build_timeline(D)
    film = (StoryFilm if story else Film)(lines, C, D, credit=args.credit_model)
    fw, fh, scale = (SW, SH, SSCALE) if story else (LW, LH, SCALE)
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
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{fw * scale}x{fh * scale}", "-r", str(FPS), "-i", "-",
           "-i", wav,
           "-map", "0:v", "-map", "1:a", "-t", f"{total:.3f}",
           "-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-tune", "animation", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
           "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
           "-movflags", "+faststart", args.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(n):
        f = film.frame(i / FPS)
        p.stdin.write(fx.crt_look(upscale(f, scale), scale).tobytes())
        if i % 150 == 0:
            print(f"  frame {i}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed")
    os.remove(wav)
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
