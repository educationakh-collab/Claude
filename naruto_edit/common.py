"""Shared timeline: tempo, song structure and event times used by both audio and video."""

BPM = 140
BEAT = 60.0 / BPM
BAR = 4 * BEAT
STEP = BEAT / 4
NBARS = 18
TAIL = 2.6
DUR = NBARS * BAR + TAIL
FPS = 30
W, H = 720, 1280          # render size (upscaled to 1080x1920 on encode)

# song sections (bar ranges, end exclusive)
INTRO = (0, 2)
BUILD = (2, 4)
DROP1 = (4, 12)
BREAK = (12, 14)
DROP2 = (14, 18)

# chord roots for the 4-bar loop: Dm - Bb - F - C
CHORDS = [
    (50, [62, 65, 69]),   # D
    (46, [58, 62, 65]),   # Bb
    (53, [57, 60, 65]),   # F
    (48, [55, 60, 64]),   # C
]
# cowbell arpeggio: (step, chord-tone index, octave shift)
COW_PATTERN = [(0, 2, 0), (3, 1, 0), (6, 0, 0), (8, 1, 0), (10, 2, 0), (11, 1, 0), (13, 0, 12), (14, 2, 0)]

# flute melody: (midi, beats)
FLUTE = [(74, 1.5), (77, .5), (79, 1), (81, 1), (84, 1.5), (81, .5), (79, 2),
         (77, 1), (79, .5), (81, .5), (74, 2), (76, 1), (77, 1), (81, 2)]


def T(bar, step=0.0):
    return bar * BAR + step * STEP


def in_range(bar, rng):
    return rng[0] <= bar < rng[1]


def is_drop(bar):
    return in_range(bar, DROP1) or in_range(bar, DROP2)


def events():
    ev = {k: [] for k in ("kick", "clap", "hat", "ohat", "cow", "bass", "taiko",
                          "impact", "roll", "flute", "riser", "revcym")}
    for bar in range(NBARS):
        ch = CHORDS[bar % 4]
        if is_drop(bar):
            kicks = [0, 6, 10] + ([14] if bar % 2 else [])
            for s in kicks:
                ev["kick"].append(T(bar, s))
                note = ch[0] - 12 + (12 if s == 14 else 0)
                ln = {0: 6, 6: 4, 10: 4, 14: 2}[s]
                ev["bass"].append((T(bar, s), note, ln * STEP))
            for s in (4, 12):
                ev["clap"].append(T(bar, s))
            for s in range(0, 16, 2):
                ev["hat"].append(T(bar, s))
            if bar % 2:
                for s in (13, 13.5, 14, 14.5, 15, 15.5):
                    ev["hat"].append(T(bar, s))
            if in_range(bar, DROP2):
                for s in (2, 10):
                    ev["ohat"].append(T(bar, s))
            for s, idx, octv in COW_PATTERN:
                ev["cow"].append((T(bar, s), ch[1][idx] + 12 + octv))
        elif in_range(bar, BUILD):
            if bar == 2:
                for s in (0, 4, 8, 12):
                    ev["taiko"].append(T(bar, s))
            else:
                for s in (0, 4):
                    ev["taiko"].append(T(bar, s))
                for s in range(8, 12):
                    ev["roll"].append(T(bar, s))
                for k in range(16):
                    ev["roll"].append(T(bar, 12 + k * 0.25))
        elif in_range(bar, BREAK):
            ev["kick"].append(T(bar, 0))
            ev["bass"].append((T(bar, 0), ch[0] - 12, 14 * STEP))
            ev["clap"].append(T(bar, 8))
            if bar == BREAK[1] - 1:
                for k in range(16):
                    ev["roll"].append(T(bar, 12 + k * 0.25))
        elif bar == 1:
            ev["taiko"] += [T(1, 0), T(1, 8), T(1, 14)]
    ev["impact"] = [T(DROP1[0]), T(BREAK[0]), T(DROP2[0]), T(NBARS)]
    ev["riser"] = [(T(BUILD[0]), T(BUILD[1]))]
    ev["revcym"] = [T(DROP1[0]), T(DROP2[0]), T(NBARS)]
    # flute: intro+build, break (first half), drop2 (higher, quieter)
    for start, n, transpose, gain in ((0, 14, 0, 1.0), (BREAK[0], 7, 0, 1.0), (DROP2[0], 14, 12, .45)):
        t = T(start)
        for note, beats in FLUTE[:n]:
            ev["flute"].append((t, note + transpose, beats * BEAT, gain))
            t += beats * BEAT
    for k in ev:
        if k not in ("riser",):
            ev[k].sort(key=lambda e: e if isinstance(e, float) else e[0])
    return ev


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)
