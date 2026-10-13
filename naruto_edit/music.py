"""Synthesizes the soundtrack (phonk beat + shakuhachi-style flute) from scratch with numpy."""
import sys
import wave

import numpy as np

from common import BAR, DUR, NBARS, BREAK, CHORDS, T, events, mtof, is_drop, in_range, DROP2

SR = 44100
N = int(DUR * SR)
rng = np.random.default_rng(7)


def tvec(dur):
    return np.arange(int(dur * SR)) / SR


def place(buf, sig, t0, gain=1.0):
    i = int(t0 * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i] * gain


def spectral_filter(x, lo=None, hi=None, order=2):
    """Gentle butterworth-like magnitude filter applied in the frequency domain."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR) + 1e-6
    g = np.ones_like(f)
    if hi:
        g /= np.sqrt(1 + (f / hi) ** (2 * order))
    if lo:
        g /= np.sqrt(1 + (lo / f) ** (2 * order))
    return np.fft.irfft(X * g, len(x))


def reverb(x, decay=1.6, length=3.0, damp=5000):
    ir = rng.standard_normal(int(length * SR)) * np.exp(-tvec(length) * (6.9 / decay))
    ir = spectral_filter(ir, hi=damp)
    ir[: int(0.012 * SR)] = 0
    ir /= np.sqrt(np.sum(ir ** 2))
    n = 1 << int(np.ceil(np.log2(len(x) + len(ir))))
    y = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n), n)[: len(x)]
    return y


# ---------------------------------------------------------------- instruments
def kick(big=False):
    t = tvec(1.6 if big else 0.55)
    f = 42 + 140 * np.exp(-t * 28)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * (2.2 if big else 6.5))
    s += rng.standard_normal(len(t)) * np.exp(-t * 350) * 0.35
    return np.tanh(s * 2.2) * 0.8


def clap():
    t = tvec(0.45)
    n = spectral_filter(rng.standard_normal(len(t)), lo=900, hi=5000)
    env = np.zeros_like(t)
    for d in (0.0, 0.011, 0.022):
        env += (t >= d) * np.exp(-np.maximum(t - d, 0) * 120)
    env += (t >= 0.03) * np.exp(-np.maximum(t - 0.03, 0) * 14) * 0.7
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30) * 0.4
    return (n / np.abs(n).max() * env + tone) * 0.9


def snare_hit():
    t = tvec(0.25)
    n = spectral_filter(rng.standard_normal(len(t)), lo=1500, hi=8000)
    return (n / np.abs(n).max() * np.exp(-t * 28) + np.sin(2 * np.pi * 210 * t) * np.exp(-t * 40) * .5) * .7


def hat(open_=False):
    t = tvec(0.35 if open_ else 0.06)
    n = spectral_filter(rng.standard_normal(len(t)), lo=7000)
    return n / np.abs(n).max() * np.exp(-t * (9 if open_ else 70)) * 0.5


def cowbell(midi):
    f = mtof(midi)
    t = tvec(0.42)
    sq = lambda fr: np.sign(np.sin(2 * np.pi * fr * t))
    s = sq(f) * 0.6 + sq(f * 1.4836) * 0.4
    s = spectral_filter(s, lo=f * 0.8, hi=f * 4)
    env = np.exp(-t * 11) * (1 - np.exp(-t * 900))
    return np.tanh(s * env * 2.5) * 0.45


def bass808(midi, dur):
    f0 = mtof(midi)
    t = tvec(dur + 0.25)
    f = f0 * (1 + 0.9 * np.exp(-t * 40))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR)
    env = (1 - np.exp(-t * 400)) * np.where(t < dur, np.exp(-t * 1.2), np.exp(-dur * 1.2) * np.exp(-(t - dur) * 25))
    return np.tanh(s * env * 3.2) * 0.75


def taiko():
    t = tvec(1.0)
    f = 62 + 70 * np.exp(-t * 18)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4.5)
    n = spectral_filter(rng.standard_normal(len(t)), hi=900) * np.exp(-t * 25)
    return np.tanh((s + n * 2.5) * 1.6) * 0.9


def flute(midi, dur):
    f0 = mtof(midi)
    t = tvec(dur + 0.4)
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.3 * t) * np.clip((t - 0.25) * 3, 0, 1)
    bend = 1 - 0.03 * np.exp(-t * 18)            # shakuhachi-style scoop into the note
    ph = 2 * np.pi * np.cumsum(f0 * vib * bend) / SR
    s = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    breath = spectral_filter(rng.standard_normal(len(t)), lo=f0 * 0.9, hi=f0 * 2.5)
    breath /= np.abs(breath).max() + 1e-9
    env = np.clip(t / 0.07, 0, 1) * np.where(t < dur, 1.0, np.exp(-(t - dur) * 9))
    env *= 1 - 0.15 * np.clip(t / max(dur, .01), 0, 1)
    return (s * 0.55 + breath * (0.25 + 0.5 * np.exp(-t * 8))) * env * 0.5


def pad_chord(notes, dur):
    t = tvec(dur + 0.5)
    s = np.zeros_like(t)
    for m in notes:
        for det in (-0.12, 0.0, 0.13):
            f = mtof(m + det)
            ph = (f * t + rng.random()) % 1.0
            s += 2 * ph - 1
    env = np.clip(t / 0.6, 0, 1) * np.where(t < dur, 1.0, np.exp(-(t - dur) * 6))
    return s * env * 0.05


def impact():
    t = tvec(3.0)
    f = 28 + 60 * np.exp(-t * 6)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.3)
    n = spectral_filter(rng.standard_normal(len(t)), hi=2500) * np.exp(-t * 3.2)
    n /= np.abs(n).max()
    return np.tanh((sub * 1.3 + n * 0.7) * 1.8) * 0.9


def riser(dur):
    t = tvec(dur)
    x = t / dur
    n = spectral_filter(rng.standard_normal(len(t)), lo=600)
    n /= np.abs(n).max()
    f = 180 * (2 ** (x * 4))
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.25
    return (n * 0.6 + tone) * x ** 2.2 * 0.6


def rev_cymbal(dur=1.6):
    t = tvec(dur)
    n = spectral_filter(rng.standard_normal(len(t)), lo=4000)
    n /= np.abs(n).max()
    return n * np.exp(-t * 3.2)[::-1] * 0.45


# ---------------------------------------------------------------- arrangement
def render(path):
    ev = events()
    drums, bass, mel, pad, fx = (np.zeros(N) for _ in range(5))
    tvals = np.arange(N) / SR

    k_s, kb = kick(), kick(True)
    for t in ev["kick"]:
        place(drums, k_s, t)
    for t in ev["impact"]:
        place(fx, impact(), t, 1.0)
        place(drums, kb, t, 0.9)
    c = clap()
    for t in ev["clap"]:
        place(drums, c, t, 0.85)
    h, ho = hat(), hat(True)
    for i, t in enumerate(ev["hat"]):
        place(drums, h, t, 0.55 + 0.25 * (i % 2))
    for t in ev["ohat"]:
        place(drums, ho, t, 0.5)
    sn = snare_hit()
    for i, t in enumerate(ev["roll"]):
        place(drums, sn, t, 0.35 + 0.5 * (i % 16) / 16)
    tk = taiko()
    for t in ev["taiko"]:
        place(drums, tk, t, 0.9)
    for t, m, d in ev["bass"]:
        place(bass, bass808(m, d), t)
    for t, m in ev["cow"]:
        place(mel, cowbell(m), t)
    for t, m, d, g in ev["flute"]:
        place(mel, flute(m, d), t, 0.9 * g)
    for t0, t1 in ev["riser"]:
        place(fx, riser(t1 - t0), t0, 0.8)
    for t in ev["revcym"]:
        place(fx, rev_cymbal(), t - 1.6)

    # pads: intro/build/break + quietly under drops
    for bar in range(NBARS):
        root, tones = CHORDS[bar % 4] if bar >= 2 else CHORDS[0]
        if bar == 3:
            root, tones = CHORDS[3]
        g = 1.0 if (bar < 4 or in_range(bar, BREAK)) else 0.45
        place(pad, pad_chord(tones + [root], BAR), T(bar), g)
    place(pad, pad_chord(CHORDS[0][1] + [CHORDS[0][0] - 12], 2.4), T(NBARS), 1.2)
    pad = spectral_filter(pad, hi=1800)

    # sidechain duck on kicks
    duck = np.ones(N)
    for t in ev["kick"] + ev["impact"]:
        i = int(t * SR)
        seg = tvec(0.3)
        j = min(N, i + len(seg))
        duck[i:j] = np.minimum(duck[i:j], 1 - 0.65 * np.exp(-seg[: j - i] * 12))
    pad *= duck
    mel_dry = mel * (0.55 + 0.45 * duck)

    # intro + break are filtered ("underwater") before the drop
    filt_mask = np.zeros(N)
    filt_mask[: int(T(4) * SR)] = 1
    filt_mask[int(T(BREAK[0]) * SR): int(T(BREAK[1]) * SR)] = 1
    sm = np.convolve(filt_mask, np.ones(2000) / 2000, mode="same")
    drums_lp = spectral_filter(drums, hi=900)
    drums = drums * (1 - sm) + drums_lp * sm * 1.3

    wet = reverb(mel_dry * 0.6 + pad * 0.5 + fx * 0.25 + drums * 0.08, decay=2.2)
    mix = drums * 0.42 + bass * 0.36 + mel_dry * 0.75 + pad * 1.3 + fx * 0.45 + wet * 0.35
    # tape-ish saturation + limiter
    mix = spectral_filter(mix, lo=25)
    mix /= np.abs(mix).max()
    mix = np.tanh(mix * 1.6) / np.tanh(1.6)
    # fade out tail
    fade = np.clip((DUR - tvals) / 1.2, 0, 1)
    mix *= fade
    mix /= np.abs(mix).max() / 0.93
    # light stereo width from the reverb
    left = mix + wet * 0.08
    right = mix - wet * 0.08
    st = np.stack([left, right], 1)
    st /= np.abs(st).max() / 0.95
    pcm = (st * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "out/music.wav")
