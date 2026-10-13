"""Renders a contact sheet of stills at given times: python3 preview.py out.png t1 t2 ..."""
import sys, time
import numpy as np
from PIL import Image
import video
from common import W, H, FPS

ts = [float(x) for x in sys.argv[2:]]
ims = []
for t in ts:
    t0 = time.time()
    buf = video.render_frame(int(round(t * FPS)))
    print(f"t={t}: {time.time()-t0:.2f}s", flush=True)
    ims.append(Image.frombytes("RGB", (W, H), buf).resize((W // 2, H // 2)))
cols = min(4, len(ims))
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * W // 2, rows * H // 2))
for k, im in enumerate(ims):
    sheet.paste(im, ((k % cols) * W // 2, (k // cols) * H // 2))
sheet.save(sys.argv[1])
