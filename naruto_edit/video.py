"""Procedurally renders every frame of the edit (no external footage) and pipes it to ffmpeg."""
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from common import (BAR, BEAT, BREAK, BUILD, DROP1, DROP2, DUR, FPS, H, INTRO, NBARS, STEP, T, W,
                    events)

HERE = os.path.dirname(os.path.abspath(__file__))
F_LAT = os.path.join(HERE, "fonts", "Anton.ttf")
F_JP = os.path.join(HERE, "fonts", "JP.ttf")
NFRAMES = int(DUR * FPS)

EV = events()
KICKS = np.array(EV["kick"])
CLAPS = np.array(EV["clap"])
IMPACTS = np.array(EV["impact"])
TAIKO = np.array(EV["taiko"])
ROLL = np.array(EV["roll"])
COWS = np.array([c[0] for c in EV["cow"]])

ORANGE = np.array([1.0, 0.45, 0.05], np.float32)
GOLD = np.array([1.0, 0.75, 0.2], np.float32)
BLUE = np.array([0.25, 0.65, 1.0], np.float32)


# ------------------------------------------------------------------ helpers
def last_hit(t, arr):
    arr = arr[arr <= t + 1e-6]
    if len(arr) == 0:
        return None, -1
    return t - arr[-1], len(arr) - 1


def env(t, arr, decay):
    dt, _ = last_hit(t, arr)
    return 0.0 if dt is None else math.exp(-dt / decay)


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lerp(a, b, x):
    return a + (b - a) * x


A = {}  # per-process assets


def font(path, size, weight=None):
    f = ImageFont.truetype(path, size)
    if weight:
        f.set_variation_by_name(weight)
    return f


def text_img(txt, size, fill=(255, 255, 255), jp=False, stroke=0, stroke_fill=(0, 0, 0), vertical=False):
    f = font(F_JP, size, "Black") if jp else font(F_LAT, size)
    if vertical:
        chars = [text_img(c, size, fill, jp, stroke, stroke_fill) for c in txt]
        w = max(c.width for c in chars)
        h = sum(c.height for c in chars) - (len(chars) - 1) * int(size * 0.12)
        im = Image.new("RGBA", (w, h))
        y = 0
        for c in chars:
            im.alpha_composite(c, ((w - c.width) // 2, y))
            y += c.height - int(size * 0.12)
        return im
    l, t, r, b = f.getbbox(txt, stroke_width=stroke)
    pad = 8
    im = Image.new("RGBA", (r - l + pad * 2, b - t + pad * 2))
    ImageDraw.Draw(im).text((pad - l, pad - t), txt, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return im


def glow_text(txt, size, color, jp=False, glow=14, vertical=False):
    core = text_img(txt, size, (255, 255, 255), jp, vertical=vertical)
    m = 2 * glow
    big = Image.new("RGBA", (core.width + 2 * m, core.height + 2 * m))
    tint = Image.new("RGBA", core.size, tuple(int(c * 255) for c in color) + (255,))
    tint.putalpha(core.getchannel("A"))
    big.alpha_composite(tint, (m, m))
    g = big.filter(ImageFilter.GaussianBlur(glow))
    out = Image.new("RGBA", big.size)
    out.alpha_composite(g)
    out.alpha_composite(g)
    out.alpha_composite(core, (m, m))
    return out


def fbm_texture(n=512, seed=1, power=1.6):
    r = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    rad = np.sqrt(fx ** 2 + fy ** 2)
    rad[0, 0] = 1
    spec = np.fft.fft2(r.standard_normal((n, n))) / rad ** power
    spec[0, 0] = 0
    tex = np.real(np.fft.ifft2(spec))
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    return tex.astype(np.float32)


def sample(tex, u, v):
    """Bilinear, wrapping texture lookup (u, v in texels)."""
    n = tex.shape[0]
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    fu = (u - u0).astype(np.float32)
    fv = (v - v0).astype(np.float32)
    u0 %= n
    v0 %= n
    u1 = (u0 + 1) % n
    v1 = (v0 + 1) % n
    a = tex[v0, u0] * (1 - fu) + tex[v0, u1] * fu
    b = tex[v1, u0] * (1 - fu) + tex[v1, u1] * fu
    return a * (1 - fv) + b * fv


def colormap(x, stops):
    xs = [s[0] for s in stops]
    out = np.empty(x.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(x, xs, [s[1][c] for s in stops])
    return out


FIRE = [(0.0, (0, 0, 0)), (0.25, (0.35, 0.02, 0.0)), (0.45, (0.9, 0.18, 0.0)),
        (0.7, (1.0, 0.5, 0.05)), (0.9, (1.0, 0.8, 0.3)), (1.0, (1.0, 0.95, 0.7))]
GOLDFIRE = [(0.0, (0, 0, 0)), (0.3, (0.4, 0.2, 0.0)), (0.55, (1.0, 0.6, 0.05)),
            (0.8, (1.0, 0.9, 0.4)), (1.0, (1.0, 1.0, 1.0))]


def add(frame, layer, color=None, k=1.0):
    if color is None:
        frame += layer * k
    else:
        frame += layer[..., None] * (np.asarray(color, np.float32) * k)
    return frame


def over(frame, alpha, color):
    a = alpha[..., None]
    return frame * (1 - a) + a * np.asarray(color, np.float32)


def paste(img, sprite, cx, cy, scale=1.0, alpha=1.0, rot=0.0):
    """Alpha-composite a PIL RGBA sprite onto PIL RGB(A) image centred at (cx, cy)."""
    if alpha <= 0.01 or scale <= 0.01:
        return
    s = sprite
    if abs(scale - 1) > 1e-3:
        s = s.resize((max(1, int(s.width * scale)), max(1, int(s.height * scale))), Image.BILINEAR)
    if rot:
        s = s.rotate(rot, Image.BILINEAR, expand=True)
    if alpha < 0.999:
        a = s.getchannel("A").point(lambda v: int(v * alpha))
        s = s.copy()
        s.putalpha(a)
    img.alpha_composite(s, (int(cx - s.width / 2), int(cy - s.height / 2)))


def mask_from_draw(fn, ss=2, blur=0, round_r=0):
    im = Image.new("L", (W * ss, H * ss))
    fn(ImageDraw.Draw(im), ss)
    if round_r:   # blur + re-threshold turns hard polygons into organic, rounded shapes
        im = im.filter(ImageFilter.GaussianBlur(round_r * ss))
        im = im.point([0 if v < 110 else 255 if v > 145 else int((v - 110) / 35 * 255) for v in range(256)])
    im = im.resize((W, H), Image.LANCZOS)
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(im, np.float32) / 255.0


# ------------------------------------------------------------------ static assets
def stamp(d, pts, lw):
    """Draws a thick, perfectly round-jointed stroke by stamping discs along the path."""
    r = lw / 2
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / (r * 0.25)))
        for k in range(n + 1):
            x, y = x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)


def leaf_symbol(size=600, stroke=0.07):
    """Hidden Leaf village swirl drawn as polylines."""
    ss = 3
    S = size * ss
    im = Image.new("L", (S, S))
    d = ImageDraw.Draw(im)
    c = S / 2
    R = S * 0.36
    lw = int(S * stroke)
    pts = []
    for i in range(400):
        th = i / 399 * 2.15 * math.pi
        r = R * (0.08 + 0.92 * th / (2.15 * math.pi)) * 0.62
        pts.append((c + r * math.cos(th + 0.6), c - r * math.sin(th + 0.6)))
    stamp(d, pts, lw)
    # outer leaf contour: big arc that ends in a sharp tip toward bottom-left
    outer = []
    for i in range(300):
        th = math.radians(200 - i / 299 * 300)   # 200° -> -100°
        outer.append((c + R * math.cos(th), c - R * math.sin(th)))
    tip = (c - R * 1.55, c + R * 1.05)
    outer.append((c + R * 0.05, c + R * 1.02))
    outer.append(tip)
    outer.append((c - R * 0.95, c + R * 0.1))
    outer.append(outer[0])
    stamp(d, outer, lw)
    # inner notch line ("the cut in the leaf")
    stamp(d, [(c - R * 0.6, c + R * 0.55), (c - R * 1.15, c + R * 0.88)], lw)
    im = im.resize((size, size), Image.LANCZOS)
    return im


def silhouette_mask(t, roof=True):
    """Spiky-haired ninja seen from behind, standing on a rooftop, headband tails waving."""
    def draw(d, ss):
        cx, base = W * 0.5 * ss, H * 0.87 * ss
        P = lambda x, y: (cx + x * ss, base + y * ss)
        E = lambda x, y, rx, ry: d.ellipse([P(x - rx, y - ry), P(x + rx, y + ry)], fill=255)
        if roof:
            d.polygon([(0, base + 30 * ss), (W * ss, base - 15 * ss), (W * ss, H * ss), (0, H * ss)], fill=255)
        # baggy pants, slightly apart stance
        d.polygon([P(-95, 12), P(-78, -150), P(-62, -275), P(-4, -270), P(-12, -140), P(-40, 14)], fill=255)
        d.polygon([P(98, 8), P(80, -150), P(64, -275), P(4, -270), P(14, -140), P(44, 12)], fill=255)
        E(-68, 10, 34, 14)
        E(70, 6, 34, 14)
        # jacket: broad shoulders tapering to the waist
        d.polygon([P(-70, -250), P(-82, -330), P(-100, -440), P(-112, -505), P(-60, -548), P(60, -548),
                   P(112, -505), P(100, -440), P(82, -330), P(70, -250)], fill=255)
        E(-108, -500, 42, 40)
        E(108, -500, 42, 40)
        # arms hanging, fists clenched
        for sx in (-1, 1):
            d.polygon([P(sx * 120, -530), P(sx * 152, -430), P(sx * 150, -330), P(sx * 132, -300),
                       P(sx * 110, -330), P(sx * 108, -440), P(sx * 84, -500)], fill=255)
            E(sx * 140, -298, 24, 27)
        # neck + head
        d.polygon([P(-34, -540), P(-28, -600), P(28, -600), P(34, -540)], fill=255)
        E(0, -655, 66, 74)
        # big swept hair spikes: (angle deg, length, width)
        spikes = [(-200, 70, 34), (-176, 105, 38), (-150, 135, 40), (-125, 150, 40), (-100, 140, 38),
                  (-78, 150, 40), (-55, 140, 40), (-30, 120, 38), (-6, 95, 34), (18, 60, 30)]
        hx, hy = 0, -665
        for ang, ln, wd in spikes:
            a = math.radians(ang)
            bx, by = hx + math.cos(a) * 52, hy + math.sin(a) * 58
            tip_a = a + 0.18                      # slight sweep
            tx, ty = bx + math.cos(tip_a) * ln, by + math.sin(tip_a) * ln
            nx, ny = -math.sin(a), math.cos(a)
            d.polygon([P(bx + nx * wd, by + ny * wd), P(tx, ty), P(bx - nx * wd, by - ny * wd),
                       P(hx, hy)], fill=255)
        # headband knot at the back of the head + two tails streaming in the wind
        E(0, -640, 16, 12)
        for k, off in enumerate((0.0, 1.4)):
            top, bot = [], []
            for i in range(28):
                u = i / 27
                x = 6 + u * (190 + 30 * k)
                y = -640 + u * (55 + 40 * k) + 26 * math.sin(u * 5.5 - t * 8 + off) * u
                wdt = 11 * (1 - 0.45 * u)
                top.append(P(x, y - wdt))
                bot.append(P(x, y + wdt))
            d.polygon(top + bot[::-1], fill=255)
    return mask_from_draw(draw, round_r=3)


def build_assets():
    if A:
        return
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    A["x"] = xx - W / 2
    A["y"] = yy - H / 2
    A["r"] = np.sqrt(A["x"] ** 2 + A["y"] ** 2)
    A["th"] = np.arctan2(A["y"], A["x"])
    hy, hx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
    A["hx"], A["hy"] = hx, hy
    A["noise"] = fbm_texture(512, 3, 1.5)
    A["noise2"] = fbm_texture(512, 11, 1.9)
    A["vignette"] = np.clip(1.25 - (A["r"] / (H * 0.62)) ** 2, 0.15, 1)[..., None].astype(np.float32)
    rg = np.random.default_rng(5)
    A["grain"] = [rg.standard_normal((H, W)).astype(np.float32) for _ in range(6)]
    # stars
    st = np.zeros((H, W), np.float32)
    for _ in range(260):
        x, y = rg.integers(0, W), rg.integers(0, int(H * 0.7))
        st[y, x] = rg.random() ** 2
    A["stars"] = np.asarray(Image.fromarray((st * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7)),
                            np.float32) / 255 * 4
    # rain streaks (tileable vertically)
    rain = Image.new("L", (W, H))
    d = ImageDraw.Draw(rain)
    for _ in range(420):
        x, y = rg.integers(-100, W + 100), rg.integers(0, H)
        L = rg.integers(25, 70)
        for oy in (0, -H, H):
            d.line([(x, y + oy), (x - L * 0.25, y + L + oy)], fill=int(90 + rg.random() * 120), width=1)
    A["rain"] = np.asarray(rain, np.float32) / 255
    A["leaf"] = leaf_symbol(560)
    # speed-line sets
    A["speed"] = []
    for k in range(4):
        nb = 180
        lens = rg.random(nb) ** 0.7
        on = rg.random(nb) < 0.55
        idx = ((A["th"] + math.pi) / (2 * math.pi) * nb).astype(np.int32) % nb
        frac = ((A["th"] + math.pi) / (2 * math.pi) * nb) % 1
        width = np.abs(frac - 0.5) < 0.18 * (A["r"] / (H * 0.5))
        rmin = H * (0.18 + 0.32 * lens[idx])
        A["speed"].append((on[idx] & width & (A["r"] > rmin)).astype(np.float32))
    # text sprites
    A["t_naruto"] = glow_text("NARUTO", 230, ORANGE)
    A["t_uzumaki"] = glow_text("UZUMAKI", 190, ORANGE)
    A["t_rasengan_jp"] = glow_text("螺旋丸", 150, BLUE, jp=True, vertical=True)
    A["t_kyubi"] = glow_text("九尾", 300, (1, 0.3, 0.05), jp=True, glow=20)
    A["t_kurama"] = glow_text("KURAMA", 200, (1, 0.3, 0.05))
    A["t_uzu_jp"] = glow_text("うずまき", 150, (1, 0.5, 0.1), jp=True)
    A["t_naru_jp"] = glow_text("ナルト", 210, (1, 0.5, 0.1), jp=True)
    A["t_believe"] = glow_text("BELIEVE IT!", 160, GOLD)
    A["t_dattebayo"] = glow_text("だってばよ!", 130, GOLD, jp=True)
    A["t_shuriken_jp"] = glow_text("螺旋手裏剣", 120, (0.6, 0.85, 1), jp=True, vertical=True)
    A["t_vs"] = glow_text("VS", 140, (1, 1, 1))
    A["t_naruto_s"] = glow_text("NARUTO", 90, BLUE)
    A["t_sasuke_s"] = glow_text("SASUKE", 90, (0.7, 0.4, 1))
    A["t_end_jp"] = glow_text("七代目火影", 90, ORANGE, jp=True)
    A["t_end"] = glow_text("NARUTO", 200, ORANGE, glow=22)


# ------------------------------------------------------------------ procedural layers
def fire_field(t, speed=1.6, scale=1.0, half=True, rise=0.55, tex="noise"):
    """Rising flame heat in [0,1], computed at half resolution and upscaled."""
    x, y = (A["hx"], A["hy"]) if half else (A["x"] + W / 2, A["y"] + H / 2)
    k = 2 if half else 1
    s = 0.55 * scale * k
    tx = A[tex]
    warp = sample(A["noise2"], x * s * 0.6, y * s * 0.6 + t * 60 * speed)
    n = sample(tx, x * s + warp * 90, y * s + t * 160 * speed + warp * 60)
    n2 = sample(tx, x * s * 2.1 + 37, y * s * 2.1 + t * 300 * speed)
    heat = n * 0.7 + n2 * 0.4
    yy = y / (H / k)
    heat = heat * (0.4 + rise * yy * 1.4) - 0.15
    heat = np.clip(heat * 1.6, 0, 1)
    if half:
        heat = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR),
                          np.float32) / 255
    return heat


def rasengan(frame, t, cx, cy, R, intensity=1.0, color=BLUE, core=(0.55, 0.85, 1.0)):
    x = A["x"] - cx
    y = A["y"] - cy
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)
    rn = r / R
    inside = np.clip(1 - rn, 0, 1)
    swirl = (np.sin(5 * th + 14 * rn - t * 26) * 0.5 + 0.5) * (np.sin(-3 * th + 9 * rn + t * 17) * 0.5 + 0.5)
    body = inside ** 0.6 * (0.35 + 0.9 * swirl)
    glow = np.exp(-np.maximum(rn - 0.6, 0) * 3.2) * 0.55
    coreg = np.exp(-(rn / 0.22) ** 2) * 0.9
    # orbiting chakra rings
    rings = np.zeros_like(r)
    for k in range(5):
        a = t * (3 + k) + k * 1.3
        tilt = 0.25 + 0.15 * k
        ca, sa = math.cos(a * 0.3 + k), math.sin(a * 0.3 + k)
        xr = (x * ca + y * sa) / (R * 1.05)
        yr = (-x * sa + y * ca) / (R * 1.05 * tilt)
        e = np.abs(np.sqrt(xr * xr + yr * yr) - 1)
        phase = (np.arctan2(yr, xr) - a * 2) % (2 * math.pi)
        rings += np.exp(-e * 40) * (phase / (2 * math.pi)) ** 2
    add(frame, (body + glow) * intensity, color * np.array([0.6, 0.85, 1.0], np.float32))
    add(frame, (coreg + rings * 0.7) * intensity, core)
    return frame


def rasenshuriken(frame, t, cx, cy, R, intensity=1.0):
    x = A["x"] - cx
    y = A["y"] - cy
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)
    R1 = R * 3.3
    spin = t * 9
    blades = np.zeros_like(r)
    for trail in range(4):           # motion blur trails
        ph = spin - trail * 0.09
        d = (th - ph - 0.5 * (r / R1)) % (math.pi / 2)
        wdt = 0.75 * np.clip(1 - (r - R * 0.6) / (R1 - R * 0.6), 0, 1)
        edge = np.clip((wdt - d) * 6, 0, 1) * (r > R * 0.6) * (r < R1)
        blades += edge * (1 - trail * 0.22)
    blades = np.clip(blades, 0, 1.6)
    disc = np.exp(-((r - R * 1.4) / (R * 1.4)) ** 2) * 0.25
    add(frame, (blades * 0.9 + disc) * intensity, (0.75, 0.9, 1.0))
    rasengan(frame, t, cx, cy, R, intensity)
    return frame


def sharingan(frame, t, open_amt=1.0, spin_speed=4.0, scale=1.0, mangekyo=False):
    x = A["x"] / scale
    y = A["y"] / scale
    ew, eh = 330, 175 * open_amt
    eye = np.clip((eh * (1 - (x / ew) ** 2) - np.abs(y)) * 0.6, 0, 1) if open_amt > 0.01 else np.zeros_like(x)
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)
    ir = 160
    iris_m = np.clip((ir - r) * 0.8, 0, 1)
    shade = 0.55 + 0.45 * np.clip(1 - r / ir, 0, 1) ** 0.5
    texture = 0.92 + 0.08 * np.sin(th * 60 + r * 0.3)
    iris = colormap(np.clip(r / ir, 0, 1), [(0, (0.9, 0.05, 0.05)), (0.7, (0.75, 0.02, 0.03)), (1, (0.25, 0, 0))])
    iris *= (shade * texture)[..., None]
    ring = np.exp(-np.abs(r - ir * 0.58) * 0.45) * 0.85
    pupil = np.clip((26 - r) * 0.8, 0, 1)
    tom = np.zeros_like(r)
    rot = t * spin_speed
    for k in range(3):
        a = rot + k * 2 * math.pi / 3
        px, py = ir * 0.58 * math.cos(a), ir * 0.58 * math.sin(a)
        dot = np.clip((25 - np.sqrt((x - px) ** 2 + (y - py) ** 2)) * 0.8, 0, 1)
        dth = (a - th) % (2 * math.pi)                # trailing tail behind the dot
        tail_len = 0.8
        thick = 22 * np.clip(1 - dth / tail_len, 0, 1) - np.abs(r - (ir * 0.58 + 14 * dth / tail_len))
        tail = np.clip(thick * 0.8, 0, 1) * (dth < tail_len)
        tom = np.maximum(tom, np.maximum(dot, tail))
    if mangekyo:
        tom = np.maximum(tom, np.clip((14 - np.abs(np.sin((th - rot) * 1.5) * r)) * 0.5, 0, 1) * (r < ir * 0.9))
    sclera = np.array([0.92, 0.9, 0.88], np.float32) * (0.75 + 0.25 * np.clip(1 - np.abs(y) / max(eh, 1), 0, 1))[..., None]
    inner = sclera * (1 - iris_m[..., None]) + iris * iris_m[..., None]
    inner = inner * (1 - np.clip(ring + tom + pupil, 0, 1)[..., None])
    # eyelid shadow + liner
    liner = np.clip(1 - np.abs(eh * (1 - (x / ew) ** 2) - np.abs(y)) / 7, 0, 1) * (np.abs(x) < ew)
    frame[:] = frame * (1 - eye[..., None]) + inner * eye[..., None]
    frame *= (1 - liner * 0.95)[..., None]
    add(frame, np.exp(-np.abs(r - ir) / 20) * eye * 0.25, (1, 0.1, 0.05))
    return frame


def fox_eye(frame, t, open_amt=1.0, scale=1.0):
    x = A["x"] / scale
    y = A["y"] / scale
    ew, eh = 320, 150 * open_amt
    # tilted, sharp eye
    yt = y + x * 0.22
    eye = np.clip((eh * (1 - (x / ew) ** 2) ** 1.4 - np.abs(yt + 20 * (x / ew))) * 0.6, 0, 1) * (np.abs(x) < ew)
    r = np.sqrt(x * x + y * y)
    th = np.arctan2(y, x)
    ir = 175
    iris = colormap(np.clip(r / ir, 0, 1), [(0, (1, 0.95, 0.4)), (0.45, (1, 0.6, 0.05)), (0.85, (0.85, 0.15, 0)),
                                             (1, (0.3, 0, 0))])
    iris *= (0.85 + 0.15 * np.sin(th * 48 + r * 0.2))[..., None]
    slit_w = 16 + 6 * math.sin(t * 3)
    slit = np.clip((slit_w * np.sqrt(np.clip(1 - (y / (ir * 0.95)) ** 2, 0, 1)) - np.abs(x)) * 0.7, 0, 1)
    iris *= (1 - slit)[..., None]
    iris_m = np.clip((ir - r) * 0.8, 0, 1)
    bg = np.array([0.5, 0.05, 0.0], np.float32)
    inner = bg * (1 - iris_m[..., None]) + iris * iris_m[..., None]
    liner = np.clip(1 - np.abs(eh * (1 - (x / ew) ** 2) ** 1.4 - np.abs(yt + 20 * (x / ew))) / 10, 0, 1) * (np.abs(x) < ew)
    frame[:] = frame * (1 - eye[..., None]) + inner * eye[..., None]
    frame *= (1 - liner)[..., None]
    add(frame, np.exp(-r / 200) * eye * 0.3, (1, 0.7, 0.1))
    return frame


def lightning(t, x0, y0, spread, n=7, seed=0, width=3):
    rg = np.random.default_rng(seed)
    im = Image.new("L", (W, H))
    d = ImageDraw.Draw(im)
    for b in range(n):
        ang = rg.uniform(0, 2 * math.pi)
        x, y = x0, y0
        pts = [(x, y)]
        for _ in range(9):
            ang += rg.uniform(-0.8, 0.8)
            step = rg.uniform(0.3, 1.0) * spread / 6
            x += math.cos(ang) * step
            y += math.sin(ang) * step
            pts.append((x, y))
        d.line(pts, fill=255, width=width)
    core = np.asarray(im, np.float32) / 255
    glow = np.asarray(im.filter(ImageFilter.GaussianBlur(9)), np.float32) / 255
    return core, glow


def nine_tails(t, intensity=1.0):
    def draw(d, ss):
        for k in range(9):
            base_ang = math.radians(-160 + k * 17.5)
            pts_l, pts_r = [], []
            for i in range(30):
                u = i / 29
                ang = base_ang + 0.45 * math.sin(t * 2.2 + k * 0.9 + u * 3) * u
                L = H * 0.75 * u
                x = W / 2 + math.cos(ang) * L * 0.9 + 60 * math.sin(u * 4 + t * 3 + k) * u
                y = H * 1.02 + math.sin(ang) * L
                wd = 75 * math.sin(math.pi * (0.15 + 0.85 * u)) * (1 - u * 0.4) + 10
                nx, ny = -math.sin(ang), math.cos(ang)
                pts_l.append((x * ss + nx * wd * ss, y * ss + ny * wd * ss))
                pts_r.append((x * ss - nx * wd * ss, y * ss - ny * wd * ss))
            d.polygon(pts_l + pts_r[::-1], fill=255)
    return mask_from_draw(draw, ss=1, blur=6) * intensity


# ------------------------------------------------------------------ scenes
def scene_intro(t, img_layers):
    f = np.zeros((H, W, 3), np.float32)
    yn = (A["y"] + H / 2) / H
    sky = colormap(yn, [(0, (0.02, 0.02, 0.07)), (0.55, (0.08, 0.05, 0.16)), (0.8, (0.45, 0.15, 0.12)),
                        (1, (0.9, 0.35, 0.1))])
    f += sky
    add(f, A["stars"] * (0.6 + 0.4 * math.sin(t * 3)), (1, 1, 1), 0.7)
    # moon
    mx, my, mr = 0, -260 - t * 6, 210
    rm = np.sqrt((A["x"] - mx) ** 2 + (A["y"] - my) ** 2)
    moon = np.clip((mr - rm) * 0.7, 0, 1)
    craters = 0.88 + 0.12 * sample(A["noise"], A["x"] * 0.9, A["y"] * 0.9)
    f = over(f, moon * craters * 0.97, (1.0, 0.85, 0.65))
    add(f, np.exp(-np.maximum(rm - mr, 0) / 120) * 0.5, (1, 0.6, 0.3))
    # drifting clouds
    cl = fire_field(t * 0.05, speed=0.0, scale=0.5, rise=0.0, tex="noise2")
    cl = np.clip((cl - 0.25) * 1.4, 0, 1) * np.clip(1 - np.abs(yn - 0.33) * 4, 0, 1)
    f = over(f, cl * 0.6, (0.12, 0.08, 0.18))
    # ninja silhouette
    sil = silhouette_mask(t)
    rim = np.clip(sil - np.roll(np.roll(sil, 3, 0), 2, 1), 0, 1)
    f *= (1 - sil * 0.97)[..., None]
    add(f, rim, (1, 0.6, 0.3), 0.9)
    # rain
    rain = np.roll(A["rain"], int(t * 1400) % H, axis=0)
    rain = np.roll(rain, int(-t * 350) % W, axis=1)
    add(f, rain * 0.35, (0.75, 0.8, 1.0))
    return f


def scene_build(t, tb, layers):
    """Bars 2-3: Sharingan opening, Japanese name flashes on taiko hits."""
    beat = int(tb / BEAT)
    f = np.zeros((H, W, 3), np.float32)
    f += np.array([0.05, 0.0, 0.0], np.float32)
    add(f, np.exp(-A["r"] / 380), (0.5, 0.0, 0.02))
    if beat in (1, 3, 5):         # cut-ins
        add(f, A["speed"][beat % 4], (1, 0.5, 0.15), 0.6)
        f = np.clip(f, 0, 1)
        layers.append(("t_uzu_jp" if beat != 5 else "t_naru_jp", W / 2, H / 2, 1.0 + 0.3 * math.exp(-(tb % BEAT) * 6)))
        return f
    open_amt = smooth(tb / (BEAT * 0.8))
    spin = 2 + tb * 3
    f = sharingan(f, t, open_amt, spin, 1.0 + tb * 0.08)
    if beat >= 6:
        add(f, A["speed"][int(t * 15) % 4], (1, 0.2, 0.1), 0.5 + 0.5 * (tb - 6 * BEAT) / (2 * BEAT))
    return f


def scene_rasengan(t, tb, layers, bar_i):
    f = np.zeros((H, W, 3), np.float32)
    beat = int(tb / BEAT)
    # background: orange/dark manga burst
    bgc = np.array([0.08, 0.03, 0.01], np.float32)
    f += bgc
    sp = A["speed"][int(t * 20) % 4]
    add(f, sp, (1, 0.55, 0.15), 0.55)
    add(f, np.exp(-A["r"] / 500), (0.9, 0.35, 0.05), 0.5)
    k = env(t, KICKS, 0.12)
    R = 165 * (1 + 0.12 * k)
    cy = 60 + 10 * math.sin(t * 2)
    rasengan(f, t, 0, cy, R, 1.0 + 0.4 * k)
    # leaf debris spiralling in
    for i in range(14):
        a = t * 2.5 + i * 0.45
        rr = (i * 97 + t * -260) % 600 + 140
        lx, ly = math.cos(a) * rr, math.sin(a) * rr + cy
        d2 = (A["x"] - lx) ** 2 / 90 + (A["y"] - ly) ** 2 / 30
        add(f, np.exp(-d2 / 3), (0.6, 0.9, 1.0), 0.6)
    names = ["t_naruto", "t_rasengan_jp", "t_uzumaki", "t_rasengan_jp"]
    name = names[bar_i % 4]
    slam = 1 + 0.6 * math.exp(-tb * 12)
    if name == "t_rasengan_jp":
        layers.append((name, W - 110, H * 0.5, slam * 0.9))
    else:
        layers.append((name, W / 2, H * 0.17, slam))
    return f


def scene_leafcut(t, tb, layers):
    f = np.ones((H, W, 3), np.float32) * np.array([1, 0.55, 0.1], np.float32)
    add(f, A["speed"][int(t * 30) % 4], (-1, -0.55, -0.1), 1.0)
    layers.append(("leaf_dark", W / 2, H / 2, 1.1 + 0.2 * math.exp(-tb * 10)))
    return f


def scene_kurama(t, tb, layers, bar_i):
    f = np.zeros((H, W, 3), np.float32)
    heat = fire_field(t, speed=1.6)
    k = env(t, KICKS, 0.15)
    f += colormap(np.clip(heat * (0.9 + 0.4 * k), 0, 1), FIRE)
    if bar_i < 2:
        open_amt = smooth(tb / (BAR * 0.6)) if bar_i == 0 else 1.0
        f = fox_eye(f, t, open_amt, 1.0 + 0.04 * k + tb * 0.03)
        if bar_i == 0 and tb > BAR * 0.5:
            layers.append(("t_kyubi", W / 2, H * 0.2, 1 + 0.5 * math.exp(-(tb - BAR * 0.5) * 10)))
        if bar_i == 1:
            layers.append(("t_kyubi", W / 2, H * 0.2, 1.0))
    else:
        f *= 0.35
        tails = nine_tails(t)
        theat = fire_field(t + 3, speed=2.2, scale=1.4)
        col = colormap(np.clip(0.45 + theat * 0.6, 0, 1), FIRE)
        a = tails[..., None]
        f = f * (1 - a * 0.9) + col * a
        # glowing eyes in the dark
        for sx in (-1, 1):
            ex, ey = sx * 95, -150
            d2 = ((A["x"] - ex) / 60) ** 2 + ((A["y"] - ey - sx * (A["x"] - ex) * 0.25) / 14) ** 2
            add(f, np.exp(-d2 * 2), (1, 0.8, 0.2), 2.0)
            add(f, np.exp(-d2 * 0.15), (1, 0.3, 0.0), 0.5)
        layers.append(("t_kurama", W / 2, H * 0.16, 1 + 0.6 * math.exp(-tb * 12)))
    return f


def scene_clash(t, tb, layers):
    """Break: manga-halftone face-off, Rasengan vs Chidori closing in."""
    f = np.zeros((H, W, 3), np.float32)
    prog = tb / (2 * BAR)
    gx = 0.25 + 0.15 * np.sin(A["x"] * 0.01 + t)
    tone = np.clip(0.18 + 0.35 * np.exp(-np.abs(A["x"]) / 300) + 0.2 * gx, 0, 1)
    dots = (np.sin(A["x"] * 0.35) * np.sin(A["y"] * 0.35) * 0.5 + 0.5) < tone
    f += dots[..., None].astype(np.float32) * 0.22
    add(f, A["speed"][int(t * 12) % 4], (1, 1, 1), 0.25)
    sep = lerp(230, 70, smooth(prog))
    rasengan(f, t, -sep, 0, 115)
    c, g = lightning(t, W / 2 + sep, H / 2, 260 + 80 * prog, n=9, seed=int(t * 24))
    add(f, g, (0.6, 0.5, 1.0), 2.0)
    add(f, c, (1, 1, 1), 1.2)
    rch = np.sqrt((A["x"] - sep) ** 2 + A["y"] ** 2)
    add(f, np.exp(-rch / 70), (0.75, 0.7, 1.0), 1.4)
    if prog > 0.85:
        add(f, np.exp(-np.sqrt(A["x"] ** 2 + A["y"] ** 2) / 120), (1, 1, 1), (prog - 0.85) * 25)
    layers.append(("t_naruto_s", W * 0.3, H * 0.84, 1.0))
    layers.append(("t_sasuke_s", W * 0.7, H * 0.16, 1.0))
    layers.append(("t_vs", W / 2, H * 0.68, 1 + 0.3 * math.exp(-(tb % BAR) * 8)))
    return f


def gold_bg(t):
    th, r = A["th"], A["r"]
    bands = np.sin(5 * th + 7 * np.log1p(r / 40) - t * 7) * 0.5 + 0.5
    f = colormap(bands * np.clip(r / 500, 0.2, 1), [(0, (0.1, 0.04, 0)), (0.6, (0.6, 0.3, 0.02)), (1, (1, 0.7, 0.2))])
    return f


def scene_drop2(t, tb, layers):
    beat = int(tb / BEAT)
    bt = tb - beat * BEAT
    k = env(t, KICKS, 0.12)
    shot = {0: "shuriken", 1: "shuriken", 2: "shuriken", 3: "shuriken", 4: "sharingan", 5: "fox", 6: "sage",
            7: "leaf", 8: "sage", 9: "sage", 10: "sage", 11: "sage", 12: "rasengan", 13: "fox", 14: "sharingan",
            15: "shuriken"}[beat]
    if shot == "shuriken":
        f = gold_bg(t) * 0.5
        add(f, A["speed"][int(t * 30) % 4], (1, 0.9, 0.6), 0.4)
        rasenshuriken(f, t, 0, 0, 120 * (1 + 0.15 * k))
        layers.append(("t_shuriken_jp", 90, H * 0.5, 1.0 + 0.3 * math.exp(-bt * 10)))
    elif shot == "sharingan":
        f = np.zeros((H, W, 3), np.float32) + np.array([0.15, 0, 0], np.float32)
        f = sharingan(f, t, 1.0, 14, 1.25 + bt * 0.6, mangekyo=True)
    elif shot == "fox":
        f = colormap(fire_field(t, speed=2.5), FIRE)
        f = fox_eye(f, t, 1.0, 1.3 + bt * 0.6)
    elif shot == "leaf":
        f = gold_bg(t)
        layers.append(("leaf_dark", W / 2, H / 2, 1.2 + 0.3 * math.exp(-bt * 10)))
    elif shot == "rasengan":
        f = gold_bg(t) * 0.4
        rasengan(f, t, 0, 0, 220)
    else:  # sage / kurama-mode silhouette with golden chakra aura
        f = gold_bg(t) * 0.35
        heat = fire_field(t, speed=2.6, scale=1.2)
        sil = silhouette_mask(t)
        aura = np.asarray(Image.fromarray((sil * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(28)),
                          np.float32) / 255
        aura = np.clip(aura * 2.2, 0, 1) * (0.5 + heat)
        f += colormap(np.clip(aura * 0.9, 0, 1), GOLDFIRE)
        f *= (1 - sil * 0.9)[..., None]
        rim = np.clip(sil - np.roll(sil, 4, 1), 0, 1) + np.clip(sil - np.roll(sil, -4, 1), 0, 1)
        add(f, rim, (1, 0.9, 0.5), 1.5)
        if beat >= 8:
            layers.append(("t_believe", W / 2, H * 0.14, 1 + 0.5 * math.exp(-(tb - 8 * BEAT) * 10)))
        if beat >= 10:
            layers.append(("t_dattebayo", W / 2, H * 0.25, 1 + 0.5 * math.exp(-(tb - 10 * BEAT) * 10)))
    return f


def scene_end(t, te, layers):
    f = np.zeros((H, W, 3), np.float32)
    add(f, np.exp(-A["r"] / 420), (0.6, 0.2, 0.02), 0.8 * smooth(te * 2))
    heat = fire_field(t, speed=0.8)
    f += colormap(heat * 0.6, FIRE) * 0.6
    layers.append(("leaf_glow", W / 2, H * 0.42, 1.0 + 0.05 * te))
    if te > 0.35:
        layers.append(("t_end", W / 2, H * 0.72, 1 + 0.4 * math.exp(-(te - 0.35) * 10)))
    if te > 0.8:
        layers.append(("t_end_jp", W / 2, H * 0.81, 1.0, smooth((te - 0.8) * 3)))
    return f


# ------------------------------------------------------------------ frame
def sprite(name):
    if name in A:
        return A[name]
    if name == "leaf_dark":
        L = A["leaf"]
        im = Image.new("RGBA", L.size, (10, 5, 0, 255))
        im.putalpha(L)
        A[name] = im
    elif name == "leaf_glow":
        L = A["leaf"]
        base = Image.new("RGBA", (L.width + 120, L.height + 120))
        tint = Image.new("RGBA", L.size, (255, 120, 20, 255))
        tint.putalpha(L)
        base.alpha_composite(tint, (60, 60))
        g = base.filter(ImageFilter.GaussianBlur(24))
        out = Image.new("RGBA", base.size)
        out.alpha_composite(g)
        out.alpha_composite(g)
        white = Image.new("RGBA", L.size, (255, 230, 190, 255))
        white.putalpha(L)
        out.alpha_composite(white, (60, 60))
        A[name] = out
    return A[name]


def typewriter(img, t):
    lines = [("I NEVER GO BACK ON MY WORD", 0.25, 1.6), ("THAT'S MY NINJA WAY.", 1.9, 3.2)]
    f = font(F_LAT, 46)
    d = ImageDraw.Draw(img)
    for i, (txt, t0, t1) in enumerate(lines):
        if t < t0:
            continue
        n = int(min(1, (t - t0) / (t1 - t0) * 1.4) * len(txt))
        s = txt[:n]
        w = d.textlength(txt, font=f)
        x, y = (W - w) / 2, H * 0.12 + i * 64
        d.text((x, y), s, font=f, fill=(255, 240, 225), stroke_width=3, stroke_fill=(0, 0, 0))


def render_frame(i):
    build_assets()
    t = i / FPS
    bar = int(t / BAR)
    tb = t - bar * BAR
    layers = []
    flash = 0.0
    invert = False

    if bar < BUILD[0]:
        f = scene_intro(t, layers)
        f *= smooth(t / 0.6)
    elif bar < DROP1[0]:
        f = scene_build(t, t - T(BUILD[0]), layers)
    elif bar < DROP1[0] + 4:
        bi = bar - DROP1[0]
        beat = int(tb / BEAT)
        if bi % 2 == 1 and beat == 3:
            f = scene_leafcut(t, tb - 3 * BEAT, layers)
        else:
            f = scene_rasengan(t, tb, layers, bi)
        if bi % 2 == 0 and beat == 3 and (tb % BEAT) < BEAT / 2:
            invert = True
    elif bar < DROP1[1]:
        f = scene_kurama(t, t - T(bar - (bar - 8) % 2), layers, (bar - 8))
        if bar == DROP1[1] - 1 and tb > BAR - BEAT:
            invert = int(tb / STEP) % 2 == 0
    elif bar < BREAK[1]:
        f = scene_clash(t, t - T(BREAK[0]), layers)
    elif bar < DROP2[1]:
        f = scene_drop2(t, t - T(DROP2[0]), layers)
    else:
        f = scene_end(t, t - T(NBARS), layers)

    # ------------- composite text / sprites
    f = np.clip(f, 0, None)
    f = np.where(f < 0.75, f, 0.75 + 0.25 * (1 - np.exp(-(f - 0.75) / 0.25)))
    img = Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    for L in layers:
        name, cx, cy, sc = L[:4]
        al = L[4] if len(L) > 4 else 1.0
        paste(img, sprite(name), cx, cy, sc, al)
    if bar < BUILD[0]:
        typewriter(img, t)
    img = img.convert("RGB")

    # ------------- camera: zoom pulse, shake, roll
    k = env(t, KICKS, 0.09)
    imp = env(t, IMPACTS, 0.35)
    tk = env(t, TAIKO, 0.15)
    rl = env(t, ROLL, 0.05)
    drop = bar >= DROP1[0]
    _, ki = last_hit(t, KICKS)
    rg = np.random.default_rng(ki + 1000 + int(t * 30) * (1 if imp > 0.2 else 0))
    sh = (14 * k + 40 * imp + 10 * tk + 5 * rl) * (1 if drop or bar >= BUILD[0] else 0.3)
    dx, dy = rg.uniform(-1, 1) * sh, rg.uniform(-1, 1) * sh
    zoom = 1.0 + 0.05 * k + 0.12 * imp + 0.04 * tk + 0.02 * math.sin(t * 0.7)
    if bar < BUILD[0]:
        zoom = 1.0 + t * 0.025
    rot = (rg.uniform(-1, 1) * 1.5 * (k + imp)) if drop else 0.0
    ca, sa = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    cx, cy = W / 2, H / 2
    a, b = ca / zoom, sa / zoom
    c = cx - a * cx - b * cy - dx
    d_, e = -sa / zoom, ca / zoom
    f_ = cy - d_ * cx - e * cy - dy
    img = img.transform((W, H), Image.AFFINE, (a, b, c, d_, e, f_), resample=Image.BILINEAR)
    if imp > 0.08:     # radial zoom blur on impacts
        acc = np.asarray(img, np.float32)
        for j in range(1, 5):
            z = 1 + j * 0.035 * imp
            sub = img.transform((W, H), Image.AFFINE, (1 / z, 0, cx - cx / z, 0, 1 / z, cy - cy / z),
                                resample=Image.BILINEAR)
            acc += np.asarray(sub, np.float32)
        out = acc / 5 / 255
    else:
        out = np.asarray(img, np.float32) / 255

    # ------------- glitch slices on cowbell hits (drops only)
    cw = env(t, COWS, 0.05)
    if drop and cw > 0.4 and int(t * FPS) % 3 == 0:
        rg2 = np.random.default_rng(int(t * 1000))
        for _ in range(5):
            y0 = rg2.integers(0, H - 60)
            hh = rg2.integers(8, 60)
            out[y0:y0 + hh] = np.roll(out[y0:y0 + hh], rg2.integers(-50, 50), axis=1)

    # ------------- chromatic aberration
    cl = env(t, CLAPS, 0.1)
    shift = int(2 + 10 * cl + 22 * imp + 6 * k * drop)
    if shift > 0:
        out = out.copy()
        out[..., 0] = np.roll(out[..., 0], shift, axis=1)
        out[..., 2] = np.roll(out[..., 2], -shift, axis=1)

    # ------------- flashes / invert / grade
    flash = 0.85 * imp ** 1.5 + 0.25 * cl * drop + 0.3 * tk * (bar < DROP1[0])
    if invert:
        out = 1 - out
    out = out + flash
    out = out * A["vignette"]
    # filmic contrast and warm grade
    out = np.clip(out, 0, 1)
    out = out * out * (3 - 2 * out) * 0.35 + out * 0.65
    out += A["grain"][i % 6][..., None] * 0.035
    fade = min(1.0, (DUR - t) / 0.9)
    out *= max(fade, 0)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8).tobytes()


def main():
    out_mp4 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "naruto_edit.mp4")
    audio = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "out", "music.wav")
    frames = range(NFRAMES)
    if len(sys.argv) > 3:
        a, b = map(int, sys.argv[3].split(":"))
        frames = range(a, b)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-ss", str(frames[0] / FPS), "-i", audio,
           "-vf", "scale=1080:1920:flags=lanczos", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", out_mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(os.cpu_count()) as pool:
        for n, buf in enumerate(pool.imap(render_frame, frames, chunksize=2)):
            p.stdin.write(buf)
            if n % 60 == 0:
                print(f"frame {n}/{len(frames)}", flush=True)
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    main()
