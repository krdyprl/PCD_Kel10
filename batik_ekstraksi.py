# -*- coding: utf-8 -*-
"""
batik_ekstraksi.py - Latihan Soal modul baru: ekstraksi LBP pada citra batik (semua jenis di Dataset/).

Ketentuan soal yang dikerjakan di sini:
  - baca grayscale, resize 300x300
  - LBP method 'default' dan 'uniform' untuk R = 1, 2, 3 (P = 8R).
    'default' untuk R=2 dan R=3 boleh dihilangkan (2^16 dan 2^24 nilai): secara bawaan hanya R=1
    (tambah --default-r2 untuk R=2; R=3 memang tidak praktis: 16,7 juta bin)
  - opsional (--tambahan): method 'ror' dan 'nri_uniform'
  - histogram ternormalisasi
  - subplot: citra asli, citra LBP, histogram -> satu gambar per jenis batik
  - tabel pengamatan (CSV) + angka-angka bantu untuk analisis (a)-(d)

Angka bantu analisis:
  (a) beda histogram antar motif : histogram uniform rata-rata per motif + jarak chi-square antar motif
  (b) pengaruh R dan P           : dimensi vektor, % pola uniform, entropi histogram
  (c) default vs uniform         : dimensi fitur dan akurasi 1-NN leave-one-out tiap konfigurasi
  (d) layak sebagai ciri?        : akurasi 1-NN leave-one-out (chance = 1/jumlah jenis) dan
                                   rasio jarak antar-kelas / dalam-kelas (> 1 = terpisah)

Jalankan:  python batik_ekstraksi.py [--dataset Dataset] [--indeks 0] [--tambahan] [--default-r2]
"""
import csv
from pathlib import Path

import numpy as np

from common import DATASET_DEFAULT, baca_citra, buat_parser, daftar_dataset, plt, simpan
from knn_manual import jarak_chi2
from lbp_histogram import histogram_lbp
from lbp_skimage import jumlah_bin, lbp_citra


def entropi(h):
    p = h[h > 0]
    return float(-(p * np.log2(p)).sum())


def persen_uniform(h, P, method):
    """Proporsi pola uniform (hanya method uniform/nri_uniform; bin terakhir = non-uniform)."""
    return float(1 - h[-1]) if method in ("uniform", "nri_uniform") else None


def main():
    p = buat_parser("Ekstraksi LBP pada citra batik: semua motif, R=1,2,3, method default/uniform")
    p.add_argument("--dataset", default=DATASET_DEFAULT)
    p.add_argument("--indeks", type=int, default=0, help="citra ke-berapa tiap motif untuk gambar (default 0)")
    p.add_argument("--tambahan", action="store_true", help="tambah method ror dan nri_uniform")
    p.add_argument("--default-r2", action="store_true", help="sertakan method default untuk R=2 (65536 bin)")
    a = p.parse_args()

    paths, y, kelas = daftar_dataset(a.dataset)
    print(f"== Ekstraksi LBP batik: {len(paths)} citra, {len(kelas)} jenis: {kelas} ==")
    print(f"   per jenis: {np.bincount(y).tolist()}  (soal meminta minimal 3 citra per motif)")
    imgs = [baca_citra(pt) for pt in paths]                       # grayscale 300x300

    konfig = [(1, "default"), (1, "uniform"), (2, "uniform"), (3, "uniform")]
    if a.default_r2:
        konfig.insert(1, (2, "default"))
    if a.tambahan:
        konfig += [(1, "ror"), (1, "nri_uniform"), (2, "nri_uniform"), (3, "nri_uniform")]

    # ---- 1. subplot: citra asli | citra LBP | histogram, satu gambar per motif ----------------
    print("\n[1] Gambar per motif")
    for ki, nama in enumerate(kelas):
        idx = np.where(y == ki)[0][min(a.indeks, int((y == ki).sum()) - 1)]
        img = imgs[idx]
        fig, axes = plt.subplots(len(konfig), 3, figsize=(11, 2.6 * len(konfig)))
        for r_, (R, method) in enumerate(konfig):
            P = 8 * R
            lbp = lbp_citra(img, P, R, method)
            nb = jumlah_bin(P, method)
            axes[r_, 0].imshow(img, cmap="gray")
            axes[r_, 0].set_title(f"{nama} (asli)", fontsize=9)
            axes[r_, 1].imshow(lbp, cmap="gray")
            axes[r_, 1].set_title(f"LBP {method}  R={R}, P={P}", fontsize=9)
            axes[r_, 2].hist(lbp.ravel(), density=True, bins=min(nb, 256), range=(0, nb), color="0.5")
            axes[r_, 2].set_title(f"Histogram ({nb} bin)", fontsize=9)
            axes[r_, 0].axis("off")
            axes[r_, 1].axis("off")
        plt.tight_layout()
        simpan(fig, a.out, f"ekstraksi_{nama}.png", a.show)

    # ---- 2. fitur semua citra untuk tiap konfigurasi -------------------------------------------
    fitur = {}
    for R, method in konfig:
        P = 8 * R
        fitur[(R, method)] = np.array([histogram_lbp(lbp_citra(im, P, R, method), P, method) for im in imgs])

    # ---- 3. tabel pengamatan: rata-rata histogram per motif ------------------------------------
    print("\n[2] Tabel pengamatan (rata-rata histogram semua citra tiap motif)")
    kolom = ["Motif", "R", "P", "Method", "Dimensi", "BinTertinggi", "NilaiBinTertinggi",
             "PersenUniform", "Entropi_bit"]
    baris = []
    print("  " + " ".join(f"{c:>14}" for c in kolom))
    for (R, method), X in fitur.items():
        P = 8 * R
        for ki, nama in enumerate(kelas):
            h = X[y == ki].mean(axis=0)
            pu = persen_uniform(h, P, method)
            row = [nama, R, P, method, len(h), int(h.argmax()), round(float(h.max()), 4),
                   "-" if pu is None else round(pu, 4), round(entropi(h), 3)]
            baris.append(row)
            print("  " + " ".join(f"{str(v):>14}" for v in row))
    if a.tambahan:
        print("  Catatan: method 'ror' memakai 256 bin, tetapi hanya 36 nilai yang mungkin terisi (P=8).")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f_csv = out / "tabel_pengamatan.csv"
    with open(f_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(kolom)
        w.writerows(baris)
    print(f"  [csv] {f_csv}")

    # ---- 4. (a) histogram uniform rata-rata per motif + jarak antar motif ----------------------
    print("\n[3] (a) Histogram uniform rata-rata per motif, R=1 (P=8) dalam persen")
    Xu = fitur[(1, "uniform")]
    print("  " + f"{'Motif':<14}" + "".join(f"{'bin'+str(b):>7}" for b in range(Xu.shape[1])))
    mean_u = {}
    for ki, nama in enumerate(kelas):
        mean_u[nama] = Xu[y == ki].mean(axis=0)
        print("  " + f"{nama:<14}" + "".join(f"{v*100:>7.1f}" for v in mean_u[nama]))

    M = np.array([mean_u[n] for n in kelas])
    Dm = jarak_chi2(M, M)
    print("\n  Jarak chi-square antar histogram rata-rata (uniform, R=1); makin besar makin beda")
    print("  " + f"{'':<14}" + "".join(f"{n[:9]:>10}" for n in kelas))
    for i, n in enumerate(kelas):
        print("  " + f"{n:<14}" + "".join(f"{Dm[i, j]:>10.4f}" for j in range(len(kelas))))

    # gambar: histogram uniform rata-rata tiap motif untuk R=1,2,3
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, R in zip(axes, (1, 2, 3)):
        X = fitur[(R, "uniform")]
        for ki, nama in enumerate(kelas):
            ax.plot(X[y == ki].mean(axis=0), marker="o", ms=3, label=nama)
        ax.set_title(f"Histogram uniform rata-rata, R={R}, P={8*R}")
        ax.set_xlabel("bin")
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    simpan(fig, a.out, "histogram_rata_rata_per_motif.png", a.show)

    # ---- 5. (b)(c)(d) keterpisahan antar motif tiap konfigurasi --------------------------------
    print("\n[4] (b)(c)(d) Keterpisahan motif per konfigurasi (jarak chi-square)")
    print(f"  {'R':>2} {'P':>3} {'method':<12}{'dim':>8}{'1-NN LOO':>10}{'antar/dalam':>13}")
    sama = y[:, None] == y[None, :]
    hasil = []
    for (R, method), X in fitur.items():
        D = jarak_chi2(X, X)
        Dn = D.copy()
        np.fill_diagonal(Dn, np.inf)
        acc = float((y[Dn.argmin(axis=1)] == y).mean())
        dalam = D[sama & ~np.eye(len(y), dtype=bool)].mean()
        antar = D[~sama].mean()
        hasil.append((R, 8 * R, method, X.shape[1], acc, antar / dalam))
        print(f"  {R:>2} {8*R:>3} {method:<12}{X.shape[1]:>8}{acc:>10.3f}{antar/dalam:>13.3f}")
    print(f"  (acak = {1/len(kelas):.3f}; rasio antar/dalam > 1 berarti antar-motif lebih jauh daripada sesama motif)")
    with open(out / "keterpisahan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["R", "P", "method", "dimensi", "akurasi_1NN_LOO", "rasio_antar_dalam"])
        w.writerows([(r, p_, m, d, round(ac, 4), round(rs, 4)) for r, p_, m, d, ac, rs in hasil])
    print(f"  [csv] {out / 'keterpisahan.csv'}")


if __name__ == "__main__":
    main()
