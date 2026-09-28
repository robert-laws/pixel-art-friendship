"""Master timeline. Every cue is derived from the real voice-line durations, so regenerating
a take (longer or shorter) automatically re-times the whole film."""

# --- fixed beats (seconds) -------------------------------------------------------------
TITLE_END = 3.40          # intro title card ends, arena fades up
BEAM_IN = 3.55            # Mega-Man style beam-in of the two fighters
FIGHT_START = 4.45        # first line of dialogue


def build(D):
    """D = {line_id: duration_seconds} -> dict of named cue times."""
    C = {}
    C["L01"] = FIGHT_START
    C["L02"] = C["L01"] + D["01"] + 0.18
    C["L03"] = C["L02"] + D["02"] + 0.26
    C["fire"] = C["L03"] + D["03"] - 0.30           # shot leaves the barrel on "...BUSTER!"
    C["L04"] = C["L03"] + D["03"] + 0.10            # shot arrives while Null starts talking
    C["L05"] = C["L04"] + D["04"] + 0.30
    C["L06"] = C["L05"] + 0.72                      # Null's flat "Hyah." overlaps Byte's roar
    C["leap"] = C["L05"] + 1.15
    C["clash"] = C["L05"] + 1.62                    # both hit on the exact same frame
    C["freeze"] = C["clash"] + 0.30                 # physics hang
    C["L07"] = C["clash"] + 1.55
    C["L08"] = C["L07"] + D["07"] + 0.22
    C["L09"] = C["L08"] + D["08"] + 0.12
    C["L10"] = C["L09"] + D["09"] + 0.10
    C["void_off"] = C["L10"] + D["10"] + 0.45       # CRT collapse starts
    C["void_on"] = C["void_off"] + 0.55 + 0.18      # void fades in, fighters fall
    C["land_b"] = C["void_on"] + 0.62
    C["land_n"] = C["void_on"] + 0.82
    C["L11"] = C["void_on"] + 1.45
    C["L12"] = C["L11"] + D["11"] + 0.16
    C["L13"] = C["L12"] + D["12"] + 0.28
    C["L14"] = C["L13"] + D["13"] + 0.12
    C["L15"] = C["L14"] + D["14"] + 0.08
    C["L16"] = C["L15"] + D["15"] + 0.28
    C["L17"] = C["L16"] + D["16"] + 0.16
    C["type0"] = C["L17"] + D["17"] + 0.10           # Null starts typing the fix
    C["type_chars"] = "WINNER = NOBODY"
    C["erase_dur"] = 0.30
    C["cps"] = 16.0
    C["type_end"] = C["type0"] + C["erase_dur"] + len(C["type_chars"]) / C["cps"]
    C["err"] = C["type_end"] + 0.30                  # compile error appears
    C["L18"] = C["err"] + 0.22
    C["L19"] = C["L18"] + D["18"] + 0.08
    C["semi"] = C["L19"] + D["19"] + 0.06            # Null types the ';'
    C["ok"] = C["semi"] + 0.30                       # BUILD OK
    C["reboot"] = C["ok"] + 1.05                     # flash to the repaired world
    C["L20"] = C["reboot"] + 0.85
    C["L21"] = C["L20"] + D["20"] + 0.22
    C["bump"] = C["L21"] + D["21"] + 0.30
    C["join"] = C["bump"] + 0.20
    C["credits"] = C["join"] + 1.35
    C["end"] = C["credits"] + 3.45
    return C
