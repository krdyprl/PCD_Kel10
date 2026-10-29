# -*- coding: utf-8 -*-
"""
lbp_skimage.py - LBP dengan scikit-image (Modul A.4 / Makalah 2.6).

    skimage.feature.local_binary_pattern(image, P, R, method='default')

method : default | ror | uniform | nri_uniform | var
Pedoman: P = 8R. R kecil -> tekstur halus, R besar -> makrostruktur.

File ini juga mengecek dua klaim di modul/makalah secara empiris:
  - jumlah bin tiap method (tabel method)
  - proporsi pola uniform: "~90% pada (8,1) dan ~70% pada (16,2)" (Ojala dkk.)

Jalankan:  python lbp_skimage.py
"""
import numpy as np

from common import buat_parser, muat_batik, plt, simpan


def lbp_citra(img, P, R, method="uniform"):
    """Pembungkus skimage.feature.local_binary_pattern."""
    from skimage.feature import local_binary_pattern
    return local_binary_pattern(img, P, R, method=method)


def jumlah_bin(P, method):
    """Jumlah bin histogram menurut method (secara teori)."""
    if method == "uniform":
        return P + 2
    if method == "nri_uniform":
        return P * (P - 1) + 3
    if method in ("default", "ror"):
        return 2 ** P
    raise ValueError("method 'var' bernilai kontinu (tanpa bin tetap)")


def main():
    a = buat_parser("LBP dengan scikit-image: method, bin, pengaruh R/P").parse_args()
    img = muat_batik("KAWUNG")              # Dataset/KAWUNG/KAWUNG_01.jpg (lokal)

    print("== Tabel method (P = 8), citra batik Kawung ==")
    print(f"  {'method':<12}{'bin (teori)':>12}{'nilai muncul':>14}   keterangan")
    ket = {"default": "LBP standar, tidak invarian rotasi",
           "ror": "invarian rotasi (36 nilai berbeda secara teori)",
           "uniform": "invarian rotasi + uniform; nilai = jumlah bit 1, non-uniform = P+1",
           "nri_uniform": "uniform tanpa invarian rotasi"}
    for m in ("default", "ror", "uniform", "nri_uniform"):
        l = lbp_citra(img, 8, 1, m)
        teori = 36 if m == "ror" else jumlah_bin(8, m)
        print(f"  {m:<12}{teori:>12}{len(np.unique(l)):>14}   {ket[m]}")
    v = lbp_citra(img, 8, 1, "var")
    print(f"  {'var':<12}{'kontinu':>12}{'-':>14}   varians lokal: min={np.nanmin(v):.2f} maks={np.nanmax(v):.2f}")

    print("\n== Pengaruh R dan P (method=uniform, P = 8R) ==")
    print(f"  {'R':>2} {'P':>3} {'bin':>4} {'% pola uniform':>16}")
    for R in (1, 2, 3):
        P = 8 * R
        l = lbp_citra(img, P, R, "uniform")
        persen_uniform = 100 * np.mean(l < P + 1)     # label P+1 = non-uniform
        print(f"  {R:>2} {P:>3} {jumlah_bin(P, 'uniform'):>4} {persen_uniform:>15.1f}%")
    print("  (Makalah mengutip Ojala dkk.: ~90% uniform pada (8,1), ~70% pada (16,2);\n"
          "   angka di atas khusus untuk citra batik Kawung ini, bisa berbeda per citra.)")

    # Visual: LBP default vs uniform untuk beberapa R
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Citra batik Kawung")
    for ax, R in zip(axes[1:], (1, 2, 3)):
        ax.imshow(lbp_citra(img, 8 * R, R, "uniform"), cmap="gray")
        ax.set_title(f"uniform  R={R}, P={8*R}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    simpan(fig, a.out, "lbp_skimage_pengaruh_R_batik_kawung.png", a.show)


if __name__ == "__main__":
    main()
