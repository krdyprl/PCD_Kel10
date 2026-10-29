# -*- coding: utf-8 -*-
"""
lbp_histogram.py - Histogram LBP sebagai vektor fitur (Modul A.2 / Makalah 2.4).

    H(k) = sum_i sum_j f(LBP(i,j), k),   f(x,y)=1 jika x=y
    Normalisasi: H(k)/sum(H)  (setara density=True) -> citra beda ukuran bisa dibandingkan.
Histogram hanya menyimpan frekuensi pola, bukan lokasi spasialnya.

Jalankan:  python lbp_histogram.py
"""
import numpy as np

from common import buat_parser, muat_batik, plt, simpan
from lbp_skimage import jumlah_bin, lbp_citra


def histogram_lbp(lbp, P, method="uniform"):
    """Histogram LBP ternormalisasi (jumlah = 1) -> vektor fitur."""
    nb = jumlah_bin(P, method)
    if nb > 2 ** 20:
        raise MemoryError(f"method='{method}' dengan P={P} -> {nb:,} bin: tidak praktis. "
                          "Gunakan method 'uniform' (P+2 bin).")
    h, _ = np.histogram(lbp.ravel(), bins=nb, range=(0, nb))
    return h.astype(np.float64) / h.sum()


def main():
    import cv2
    a = buat_parser("Histogram LBP").parse_args()

    img = muat_batik("KAWUNG")              # Dataset/KAWUNG/KAWUNG_01.jpg (lokal)
    P, R = 8, 1
    lbp = lbp_citra(img, P, R, "uniform")

    print("== Histogram LBP (batik Kawung, P=8, R=1, uniform) ==")
    h = histogram_lbp(lbp, P, "uniform")
    print(f"  Dimensi vektor fitur : {len(h)}  (kunci: P + 2 = {P + 2})")
    print(f"  Jumlah setelah normalisasi: {h.sum():.6f}")
    for k, v in enumerate(h):
        print(f"    bin {k}: {v:.4f}")

    # H(k) definisi makalah (hitung manual dengan bincount) vs np.histogram
    H_manual = np.bincount(lbp.ravel().astype(int), minlength=len(h))
    print(f"  H(k) manual (bincount) == np.histogram? "
          f"{np.array_equal(H_manual, np.histogram(lbp.ravel(), len(h), (0, len(h)))[0])}")

    # Kenapa perlu normalisasi: citra yang sama dengan ukuran berbeda
    kecil = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    h_kecil_mentah = np.histogram(lbp_citra(kecil, P, R, "uniform").ravel(), 10, (0, 10))[0]
    h_kecil = histogram_lbp(lbp_citra(kecil, P, R, "uniform"), P, "uniform")
    print(f"\n  Citra {img.shape[0]}x{img.shape[1]} vs 256x256: total piksel {lbp.size} vs "
          f"{h_kecil_mentah.sum()} (frekuensi mentah beda),\n"
          f"  tetapi setelah normalisasi selisih maks antar bin = {np.abs(h - h_kecil).max():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].bar(range(len(h)), H_manual, color="gray")
    axes[0].set_title("Histogram mentah (frekuensi)")
    axes[1].bar(range(len(h)), h, color="gray")
    axes[1].set_title("Histogram ternormalisasi (vektor fitur)")
    for ax in axes:
        ax.set_xlabel("nilai LBP (bin)")
    plt.tight_layout()
    simpan(fig, a.out, "lbp_histogram_batik_kawung.png", a.show)


if __name__ == "__main__":
    main()
