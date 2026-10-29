# -*- coding: utf-8 -*-
"""
lbp_melingkar.py - LBP circular neighborhood buatan sendiri (Modul A.2 / Makalah 2.3).

P titik sampling diletakkan pada lingkaran jari-jari R; titik yang tidak jatuh
tepat di grid piksel dihitung dengan interpolasi bilinear.
Konvensi (sama dengan skimage): g_0 di sisi kanan pusat, berlawanan jarum jam,
piksel di luar citra bernilai 0.

File ini:
  1. menggambar titik sampling (Gambar X.1 / 2.1)
  2. membuktikan lbp_sendiri() identik dengan skimage.local_binary_pattern

Jalankan:  python lbp_melingkar.py
"""
from itertools import product

import numpy as np

from common import buat_parser, simpan, plt


def titik_sampling(P, R):
    """Koordinat (dx, dy) tiap titik sampling relatif terhadap pusat (y ke atas)."""
    p = np.arange(P)
    return R * np.cos(2 * np.pi * p / P), R * np.sin(2 * np.pi * p / P)


def lbp_sendiri(img, P, R, method="default"):
    """LBP melingkar tanpa skimage. method: 'default' atau 'uniform'."""
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape
    pad = int(np.ceil(R)) + 1
    ip = np.pad(img, pad, mode="constant", constant_values=0)
    bits = np.zeros((P, H, W), dtype=np.uint8)
    for p in range(P):
        rp = np.round(-R * np.sin(2 * np.pi * p / P), 5)
        cp = np.round(R * np.cos(2 * np.pi * p / P), 5)
        r0, c0 = int(np.floor(rp)), int(np.floor(cp))
        fr, fc = rp - r0, cp - c0

        def s(dr, dc):
            return ip[pad + dr: pad + dr + H, pad + dc: pad + dc + W]

        v = ((1 - fr) * (1 - fc) * s(r0, c0) + (1 - fr) * fc * s(r0, c0 + 1) +
             fr * (1 - fc) * s(r0 + 1, c0) + fr * fc * s(r0 + 1, c0 + 1))
        bits[p] = (v - img >= 0)                       # s(g_p - g_c)
    if method == "default":
        return sum(bits[p].astype(np.int64) << p for p in range(P)).astype(np.float64)
    if method == "uniform":
        U = np.abs(bits.astype(np.int16) - np.roll(bits, 1, axis=0).astype(np.int16)).sum(axis=0)
        return np.where(U <= 2, bits.sum(axis=0), P + 1).astype(np.float64)
    raise ValueError("lbp_sendiri hanya mendukung 'default' dan 'uniform'")


def main():
    from lbp_skimage import lbp_citra
    a = buat_parser("LBP melingkar: titik sampling + verifikasi vs skimage").parse_args()

    # 1. Gambar titik sampling P=8, R=1,2,3
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, R in zip(axes, (1, 2, 3)):
        ax.set_xticks(np.arange(-3, 4))
        ax.set_yticks(np.arange(-3, 4))
        ax.grid(True, color="0.8")
        ax.add_patch(plt.Circle((0, 0), R, fill=False, ls="--", color="0.5"))
        ax.plot(0, 0, "ks", ms=9)
        x, y = titik_sampling(8, R)
        ax.plot(x, y, "ro")
        for p, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(f"g{p}", (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.set_aspect("equal")
        ax.set_title(f"P=8, R={R}")
    plt.tight_layout()
    simpan(fig, a.out, "lbp_melingkar_titik_sampling.png", a.show)

    # 2. Verifikasi
    print("== Verifikasi: lbp_sendiri() vs skimage.local_binary_pattern ==")
    img = np.random.default_rng(1).integers(0, 256, (80, 80)).astype(np.uint8)
    ok_semua = True
    for R, method in product((1, 2, 3), ("default", "uniform")):
        P = 8 * R
        if method == "default" and P > 16:
            continue                                   # 2^24 bin tidak praktis
        a1, a2 = lbp_citra(img, P, R, method), lbp_sendiri(img, P, R, method)
        ok = np.array_equal(a1, a2)
        ok_semua &= ok
        print(f"  R={R} P={P:2d} {method:8s}: {'IDENTIK' if ok else 'BEDA'} "
              f"(piksel beda: {int((a1 != a2).sum())})")
    print("  Hasil:", "semua identik dengan skimage" if ok_semua else "ada perbedaan, periksa")


if __name__ == "__main__":
    main()
