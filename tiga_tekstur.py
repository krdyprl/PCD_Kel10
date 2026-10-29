# -*- coding: utf-8 -*-
"""
tiga_tekstur.py - Membandingkan histogram LBP pada tiga motif batik: KAWUNG,
PARANG, TAMBAL (bagian C.4). LBP uniform, P=8, R=1, histogram
ternormalisasi (10 bin).

Citra dibaca dari Dataset/KAWUNG, Dataset/PARANG, Dataset/TAMBAL (berkas lokal).
Angka pengamatan dicetak dari hasil hitung, bukan klaim tetap.

Jalankan:  python tiga_tekstur.py
"""
import numpy as np

from common import MOTIF_CONTOH, buat_parser, muat_batik, plt, simpan
from lbp_histogram import histogram_lbp
from lbp_skimage import lbp_citra


def main():
    a = buat_parser("Histogram LBP pada tiga motif batik").parse_args()
    R = 1
    P = 8 * R
    names = list(MOTIF_CONTOH)
    imgs = {n: muat_batik(n) for n in names}

    fig, axes = plt.subplots(3, 3, figsize=(9, 8))
    print(f"== LBP uniform P={P}, R={R}: histogram ternormalisasi (%) ==")
    print(f"  {'tekstur':<8}" + "".join(f"{'bin'+str(b):>7}" for b in range(P + 2)) + "   bin tertinggi")
    for j, n in enumerate(names):
        im = imgs[n]
        lbp = lbp_citra(im, P, R, "uniform")
        hist = histogram_lbp(lbp, P, "uniform")
        n_bins = P + 2

        axes[0, j].imshow(im, cmap="gray")
        axes[0, j].set_title(n)
        axes[1, j].imshow(lbp, cmap="gray")
        axes[1, j].set_title("LBP uniform")
        axes[2, j].bar(range(n_bins), hist, color="0.5")
        axes[2, j].set_xticks(range(n_bins))
        axes[0, j].axis("off")
        axes[1, j].axis("off")

        urut = np.argsort(hist)[::-1]
        print(f"  {n:<8}" + "".join(f"{v*100:>7.1f}" for v in hist) +
              f"   bin {urut[0]} ({hist[urut[0]]*100:.1f}%), lalu bin {urut[1]} ({hist[urut[1]]*100:.1f}%)")
    plt.tight_layout()
    simpan(fig, a.out, "tiga_tekstur_histogram.png", a.show)

    # Persentase pola uniform tiap motif pada (8,1) dan (16,2)
    for n in names:
        im = imgs[n]
        for r in (1, 2):
            p = 8 * r
            l = lbp_citra(im, p, r, "uniform")
            print(f"  {n} (P={p}, R={r}): pola uniform = {100 * np.mean(l < p + 1):.1f}%")


if __name__ == "__main__":
    main()
