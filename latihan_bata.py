# -*- coding: utf-8 -*-
"""
latihan_bata.py - Latihan 2: LBP pada tekstur batik (Modul B / Makalah 2.7).

Citra batik Kawung (Dataset/KAWUNG/KAWUNG_01.jpg) dengan R=3, P=24,
method='uniform' -> 26 bin (0..25). Sorot piksel/bin untuk edge (label ~P/2),
flat (~0 dan ~P, plus bin non-uniform P+1), corner (~P/4 dan 3P/4).

Pertanyaan diskusi: apa yang berubah bila radius = 1?  ->  python latihan_bata.py --radius 1

Jalankan:  python latihan_bata.py [--radius 3]
"""
import numpy as np

from common import buat_parser, muat_batik, plt, simpan
from lbp_histogram import histogram_lbp
from lbp_skimage import lbp_citra


def main():
    from skimage.color import label2rgb

    p = buat_parser("Latihan 2: tekstur batik")
    p.add_argument("--radius", type=int, default=3, help="R (default 3; coba 1)")
    a = p.parse_args()

    radius = a.radius
    n_points = 8 * radius
    METHOD = "uniform"
    print(f"== Latihan 2: Tekstur Batik (R={radius}, P={n_points}, method={METHOD}) ==")

    def overlay_labels(image, lbp, labels):
        mask = np.logical_or.reduce([lbp == each for each in labels])
        return label2rgb(mask, image=image, bg_label=0, alpha=0.5)

    def highlight_bars(bars, indexes):
        for i in indexes:
            bars[i].set_facecolor("r")

    def hist(ax, lbp):
        n_bins = int(lbp.max() + 1)
        return ax.hist(lbp.ravel(), density=True, bins=n_bins,
                       range=(0, n_bins), facecolor="0.5")

    image = muat_batik("KAWUNG")           # Dataset/KAWUNG/KAWUNG_01.jpg (lokal)
    lbp = lbp_citra(image, n_points, radius, METHOD)

    w = radius - 1
    edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    i_14 = n_points // 4
    i_34 = 3 * (n_points // 4)
    corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                     list(range(i_34 - w, i_34 + w + 1)))

    label_sets = (edge_labels, flat_labels, corner_labels)
    titles = ("edge", "flat", "corner")

    fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    plt.gray()
    for ax, labels in zip(ax_img, label_sets):
        ax.imshow(overlay_labels(image, lbp, labels))
    for ax, labels, name in zip(ax_hist, label_sets, titles):
        counts, _, bars = hist(ax, lbp)
        highlight_bars(bars, labels)
        ax.set_ylim(top=np.max(counts[:-1]))
        ax.set_xlim(right=n_points + 2)
        ax.set_title(name)
    ax_hist[0].set_ylabel("Percentage")
    for ax in ax_img:
        ax.axis("off")
    plt.tight_layout()
    simpan(fig, a.out, f"latihan2_batik_kawung_R{radius}.png", a.show)

    h = histogram_lbp(lbp, n_points, METHOD)
    print(f"  Jumlah bin = {len(h)}  (uniform: P + 2 = {n_points + 2} bin)")
    for nama, lab in zip(titles, label_sets):
        print(f"  proporsi piksel '{nama}' = {h[list(lab)].sum():.3f}")


if __name__ == "__main__":
    main()
