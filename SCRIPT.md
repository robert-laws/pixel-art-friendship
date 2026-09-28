# BYTE vs NULL — script

*A one-minute pixel-art short. Two rivals, one game-breaking bug, one fix.*

**Cast**

| | Look | Voice (ElevenLabs **Eleven v4**) |
|---|---|---|
| **BYTE** | Blue Mega-Man-style hero. Earnest, hot-headed, shouts attack names. | `VALF - Anime Protagonist Charismatic` (`unWb9iRIK7Xxf4A5Hfis`) |
| **NULL** | Crimson boss: horn, dark visor, flowing scarf. Deadpan, tired of this. | `Cavendish` (`Cx1u6YPIa1SPiAbYj3gJ`; the generation metadata reports the same voice as "Rai") |

Square-bracket tags are Eleven v4 audio tags. One take per line, generated in a single
ElevenCreative flow. Text in `*asterisks*` is highlighted in the on-screen captions.

---

## Title card  (0:00 – 0:03)
CRT powers on. **BYTE vs NULL** slams in. *"Two rivals. One game-breaking bug. Can they fix it… together?"*
Progress row lights up: **FIGHT › CRASH › DEBUG › BEFRIEND**. Credits line at the bottom.

## Act I — the boss fight  (Stage 8: Final Boss)
Beam-in, `READY`, boss bars fill. A neon-dusk city scrolls past on a bridge (six parallax layers).

| # | Who | Line |
|---|---|---|
| 01 | BYTE | `[shouting]` This is it, Null! Your reign of terror ends TODAY! |
| 02 | NULL | `[sighs]` You say that every day. |
| 03 | BYTE | `[shouting]` MEGA-BYTE… BUSTER! *(fires a shot)* |
| 04 | NULL | `[flatly]` That's just your regular gun. With a new name. *(hops over the shot, arms still crossed)* |
| 05 | BYTE | `[roaring]` HYAAAAAAH! *(huge charge, aura, speed lines)* |
| 06 | NULL | `[monotone]` Hyah. *(barely charges)* |

Both leap. **They hit each other on the exact same frame.** White-out, hit-stop, both bars hit zero.

## Act II — the crash
The game has no case for a tie. Everyone is frozen mid-air; the parallax layers desync and drift apart,
one layer turns into a missing-texture checkerboard, the sky becomes static, and the win banner can't
decide between *BYTE WINS!* and *NULL WINS!*.

| # | Who | Line |
|---|---|---|
| 07 | BYTE | `[nervous]` Uh… why is the sky made of static? |
| 08 | NULL | `[calm]` We hit on the same frame. It doesn't know who won. |
| 09 | BYTE | `[gasps]` Can it *DO* that?! |
| 10 | NULL | `[flatly]` Evidently. |

The picture collapses like a CRT switching off.

## Act III — the debug void
Missing-texture floor, code rain, a floating terminal (`DEBUG.EXE`). They fall in, land, and sit down.

| # | Who | Line |
|---|---|---|
| 11 | BYTE | `[reluctantly]` …Truce. Just until we fix it. |
| 12 | NULL | `[deadpan]` Wow. Character growth. |
| 13 | NULL | `[reading slowly]` Line four-twenty: TODO, figure out who wins. |
| 14 | BYTE | `[incredulous]` Four *YEARS* of fighting… over a TODO?! |
| 15 | NULL | `[flatly]` Correct. |
| 16 | BYTE | `[thoughtful]` …What if nobody has to win? *(💡)* |
| 17 | NULL | `[surprised]` …That's actually valid syntax. |

Null types the fix — `WINNER = NOBODY` — and hits build.

| # | Who | Line |
|---|---|---|
| 18 | NULL | `[muttering]` Missing semicolon. |
| 19 | BYTE | `[laughing]` It's *ALWAYS* the semicolon! |

One `;` later: **BUILD OK — 0 ERRORS, 0 BUGS.** The crawling bugs turn into butterflies.

## Act IV — a repaired world
Flash to dawn. The battle theme comes back in a major key.

| # | Who | Line |
|---|---|---|
| 20 | BYTE | `[warmly]` Hey, Null? I'm glad we crashed the game. |
| 21 | NULL | `[softly]` …Same. `[dryly]` Don't tell the players. |

Fist bump. **PLAYER 2 HAS JOINED!** → credits card → *(It was a semicolon.)*

---

### Why it works (the joke architecture)
* **The bug is the plot.** The crash is not random: an unhandled tie (`// TODO: figure out who wins`) is the
  reason the rivalry existed at all. Nobody ever wrote the peaceful ending.
* **Deadpan vs. earnest.** Byte escalates, Null de-escalates; the humor is the gap.
* **Callbacks.** "Regular gun / new name" → "valid syntax"; "Hyah." → "Correct."; the minor-key battle
  theme → the same melody in a major key when they become friends.
