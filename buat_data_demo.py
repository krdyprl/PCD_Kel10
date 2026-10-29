# -*- coding: utf-8 -*-
"""
buat_data_demo.py - Dataset SINTETIS (BUKAN batik asli) untuk menguji alur kode.

  KAWUNG-sintetis    : elips besar berpotongan, garis tepi tegas, kontras tinggi.
  SIDOMUKTI-sintetis : titik/isen-isen halus rapat, kontras rendah.

Dataset asli ada di Dataset/<MOTIF>/ dengan enam motif; skrip ini menulis <n>.jpg
ke folder sintetis terpisah (default Dataset_sintetis/) agar tidak menimpa dataset asli.

Jalankan:  python buat_data_demo.py [--dataset Dataset_sintetis] [--n 20]
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    import cv2
    p = argparse.ArgumentParser(description="Buat dataset batik sintetis untuk uji kode")
    p.add_argument("--dataset", default="Dataset_sintetis")
    p.add_argument("--n", type=int, default=20, help="citra per kelas")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    def kawung():
        img = np.full((400, 400), rng.integers(170, 230), np.uint8)
        s = int(rng.integers(55, 80))
        ox, oy = rng.integers(0, s, 2)
        for cy in range(-s + oy, 460, s):
            for cx in range(-s + ox, 460, s):
                cv2.ellipse(img, (cx, cy), (s // 2 - 3, s // 3), 45, 0, 360, int(rng.integers(20, 70)), 3)
                cv2.ellipse(img, (cx, cy), (s // 2 - 3, s // 3), -45, 0, 360, int(rng.integers(20, 70)), 3)
                cv2.circle(img, (cx, cy), 4, 30, -1)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)

    def sidomukti():
        base = int(rng.integers(120, 170))
        img = np.full((400, 400), base, np.uint8)
        s = int(rng.integers(12, 18))
        for cy in range(0, 420, s):
            for cx in range(0, 420, s):
                if rng.random() < 0.9:
                    cv2.circle(img, (cx + int(rng.integers(-2, 3)), cy + int(rng.integers(-2, 3))),
                               int(rng.integers(1, 3)), base + int(rng.integers(-35, 35)), -1)
        return np.clip(img + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)

    for nama, fn in (("KAWUNG", kawung), ("SIDOMUKTI", sidomukti)):
        folder = Path(a.dataset) / nama
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(1, a.n + 1):
            cv2.imwrite(str(folder / f"{i}.jpg"), fn())
    print(f"Dataset sintetis dibuat di '{a.dataset}/' ({a.n} citra x 2 kelas). INGAT: bukan batik asli.")


if __name__ == "__main__":
    main()
