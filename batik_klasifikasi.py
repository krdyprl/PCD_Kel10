# -*- coding: utf-8 -*-
"""
batik_klasifikasi.py - Praktikum batik bagian 2: LBP + kNN (Modul D / Makalah 2.9).

Alur: baca citra -> grayscale 300x300 -> LBP (P=8R) -> histogram ternormalisasi
      -> split latih/uji (default 70:30, stratified) -> kNN (k=3) -> akurasi + confusion matrix.

Konfigurasi default = Tabel Hasil Percobaan modul:
  (R=1, default) (R=1, uniform) (R=2, uniform) (R=3, uniform)   dengan P = 8R

Opsi tambahan:
  --k 1 3 5          variasi nilai k
  --cv 5             akurasi k-fold CV (lebih stabil bila data uji hanya beberapa)
  --metrik chi2      jarak Khi-Kuadrat (cocok untuk histogram)
  --lengkap          grid lebih luas (default R=1,2 + uniform & nri_uniform R=1,2,3)
  --laporan          presisi / recall / F1

Dataset:  Dataset/<KELAS>/*.jpg   (minimal 2 kelas, minimal 10 citra per kelas)

Jalankan:  python batik_klasifikasi.py --dataset Dataset --cv 5
"""
import csv
from pathlib import Path

import numpy as np

from common import (DATASET_DEFAULT, UKURAN_CITRA, baca_citra, buat_parser,
                    daftar_dataset, plt, simpan)
from knn_manual import prediksi_knn
from lbp_histogram import histogram_lbp
from lbp_skimage import jumlah_bin, lbp_citra

# Tabel hasil percobaan Modul D: (R, method)  dengan P = 8R
KONFIGURASI_MODUL = [(1, "default"), (1, "uniform"), (2, "uniform"), (3, "uniform")]


def fitur_lbp(path, R=1, method="uniform", P=None, ukuran=UKURAN_CITRA):
    """Path citra -> vektor fitur histogram LBP ternormalisasi. Default P = 8R."""
    P = 8 * R if P is None else P
    img = baca_citra(path, ukuran)
    return histogram_lbp(lbp_citra(img, P, R, method), P, method)


def main():
    from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                                 classification_report, confusion_matrix)
    from sklearn.model_selection import StratifiedKFold, train_test_split

    p = buat_parser("Klasifikasi batik: LBP + kNN")
    p.add_argument("--dataset", default=DATASET_DEFAULT)
    p.add_argument("--k", type=int, nargs="+", default=[3])
    p.add_argument("--test-size", type=float, default=0.3, help="default 0.3 = 70:30")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metrik", choices=["euclidean", "chi2"], default="euclidean")
    p.add_argument("--cv", type=int, default=0)
    p.add_argument("--lengkap", action="store_true")
    p.add_argument("--laporan", action="store_true")
    a = p.parse_args()

    print("== Batik klasifikasi: LBP + kNN ==")
    paths, y, kelas = daftar_dataset(a.dataset)
    print(f"  Dataset: {len(paths)} citra, kelas = {kelas}, jumlah per kelas = {np.bincount(y).tolist()}")
    if np.bincount(y).min() < 10:
        print("  PERINGATAN: modul meminta minimal 10 citra per kelas.")

    konfigurasi = list(KONFIGURASI_MODUL)
    if a.lengkap:
        konfigurasi = [(1, "default"), (2, "default")] + \
                      [(R, m) for R in (1, 2, 3) for m in ("uniform", "nri_uniform")]

    idx = np.arange(len(paths))
    tr, te = train_test_split(idx, test_size=a.test_size, stratify=y, random_state=a.seed)
    hasil, cm_plot = [], []

    for R, method in konfigurasi:
        P = 8 * R
        dim = jumlah_bin(P, method)
        try:
            X = np.array([fitur_lbp(pt, R=R, method=method, P=P) for pt in paths])
        except MemoryError as e:
            print(f"  [lewati] R={R} {method}: {e}")
            continue

        for k in a.k:
            if k > len(tr):
                continue
            pred = prediksi_knn(X[tr], y[tr], X[te], k, a.metrik)
            acc = accuracy_score(y[te], pred)
            cm = confusion_matrix(y[te], pred, labels=range(len(kelas)))
            row = dict(No=len(hasil) + 1, R=R, P=P, Method=method, Dimensi=dim, k=k,
                       Akurasi_split=round(acc, 4))
            if a.cv:
                skf = StratifiedKFold(n_splits=a.cv, shuffle=True, random_state=a.seed)
                accs = [accuracy_score(y[v], prediksi_knn(X[t], y[t], X[v], k, a.metrik))
                        for t, v in skf.split(X, y)]
                row["Akurasi_CV_mean"] = round(float(np.mean(accs)), 4)
                row["Akurasi_CV_std"] = round(float(np.std(accs)), 4)
            hasil.append(row)
            if k == a.k[0]:
                cm_plot.append((f"R={R} P={P} {method}\nk={k}  acc={acc:.2f}", cm))

            print(f"\n  [{row['No']}] R={R} P={P} method={method} dim={dim} k={k} metrik={a.metrik}")
            print(f"      Akurasi (split {round((1-a.test_size)*100)}:{round(a.test_size*100)}, "
                  f"n_uji={len(te)}): {acc:.4f}")
            if a.cv:
                print(f"      Akurasi {a.cv}-fold CV : {row['Akurasi_CV_mean']:.4f} "
                      f"+/- {row['Akurasi_CV_std']:.4f}")
            print("      Confusion Matrix (baris=asli, kolom=prediksi):")
            for line in str(cm).splitlines():
                print("        " + line)
            if a.laporan:
                print(classification_report(y[te], pred, target_names=kelas, zero_division=0, digits=3))

    if not hasil:
        raise SystemExit("Tidak ada hasil.")

    kolom = list(hasil[0].keys())
    print("\n  Tabel Hasil Percobaan")
    print("  " + " | ".join(f"{c:>15}" for c in kolom))
    for r in hasil:
        print("  " + " | ".join(f"{str(r[c]):>15}" for c in kolom))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f_csv = out / "hasil_eksperimen.csv"
    with open(f_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=kolom)
        w.writeheader()
        w.writerows(hasil)
    print(f"\n  [csv] {f_csv}")

    n = len(cm_plot)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), squeeze=False)
    for ax, (judul, cm) in zip(axes[0], cm_plot):
        ConfusionMatrixDisplay(cm, display_labels=kelas).plot(ax=ax, colorbar=False, cmap="Greys")
        ax.set_title(judul, fontsize=9)
        ax.tick_params(axis="x", labelrotation=30)
    plt.tight_layout()
    simpan(fig, a.out, "confusion_matrix.png", a.show)


if __name__ == "__main__":
    main()
