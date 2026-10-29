# -*- coding: utf-8 -*-
"""
batik_demo.py - Praktikum batik bagian 1: LBP uniform vs default (Modul D / Makalah 2.8).
Citra dibaca grayscale, resize 300x300, P=8R, R=1. Menampilkan citra asli, citra LBP
dan histogram (uniform & default), lalu mencetak Tabel 2.4 (histogram uniform per motif).
"""
from pathlib import Path

from common import DATASET_DEFAULT, baca_citra, buat_parser, daftar_dataset, plt, simpan
from lbp_histogram import histogram_lbp
from lbp_skimage import jumlah_bin, lbp_citra


def plot_lbp_comparison(img, judul, R=1, out="hasil", tampil=False):
    n_points = 8 * R
    lbp_u = lbp_citra(img, n_points, R, "uniform")
    lbp_d = lbp_citra(img, n_points, R, "default")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title(f"Citra Asli: {judul}")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    axes[0, 1].imshow(lbp_u, cmap="gray")
    axes[0, 1].set_title("LBP (Uniform)")
    axes[0, 1].axis("off")
    nb_u = int(lbp_u.max() + 1)
    axes[0, 2].hist(lbp_u.ravel(), density=True, bins=nb_u, range=(0, nb_u), color="gray")
    axes[0, 2].set_title("Histogram (Uniform)")

    axes[1, 1].imshow(lbp_d, cmap="gray")
    axes[1, 1].set_title("LBP (Default)")
    axes[1, 1].axis("off")
    nb_d = int(lbp_d.max() + 1)
    axes[1, 2].hist(lbp_d.ravel(), density=True, bins=nb_d, range=(0, nb_d), color="gray")
    axes[1, 2].set_title("Histogram (Default)")
    plt.tight_layout()
    simpan(fig, out, f"batik_{judul.replace(' ', '_')}_R{R}.png", tampil)
    return histogram_lbp(lbp_u, n_points, "uniform"), histogram_lbp(lbp_d, n_points, "default")


def main():
    p = buat_parser("Batik: perbandingan LBP uniform vs default")
    p.add_argument("--dataset", default=DATASET_DEFAULT)
    p.add_argument("--gambar", nargs="+", help="path gambar")
    p.add_argument("--nama", nargs="+", help="nama tiap gambar (urut sama)")
    p.add_argument("--radius", type=int, default=1)
    a = p.parse_args()

    print("== Batik demo: uniform vs default ==")
    if a.gambar:
        nama = a.nama or [Path(g).stem for g in a.gambar]
        gambar = list(zip(a.gambar, nama))
    else:
        paths, labels, kelas = daftar_dataset(a.dataset)
        gambar = [(paths[list(labels).index(i)], f"Batik {k.title()}") for i, k in enumerate(kelas)]

    hasil = {}
    for path, nama in gambar:
        try:
            img = baca_citra(path)
        except FileNotFoundError:
            print(f"  Gambar tidak ditemukan: {path}")
            return
        hasil[nama] = plot_lbp_comparison(img, nama, a.radius, a.out, a.show)

    P = 8 * a.radius
    nb = jumlah_bin(P, "uniform")
    print(f"\n  Tabel 2.4 - Ringkasan histogram uniform (R={a.radius}, P={P})")
    print("  " + f"{'Motif':<22}" + "".join(f"Bin{b:<6}" for b in range(nb)))
    for nama, (hu, _) in hasil.items():
        print("  " + f"{nama:<22}" + "".join(f"{v:<9.4f}" for v in hu))
    print("  Spike histogram default pada nilai 255 (semua tetangga >= pusat):")
    for nama, (_, hd) in hasil.items():
        print(f"     {nama:<22} {hd[-1]:.4f}")


if __name__ == "__main__":
    main()
