# -*- coding: utf-8 -*-
"""
common.py - fungsi bantu yang dipakai bersama oleh semua file.

Isi: argumen CLI standar (--out, --show), baca citra grayscale 300x300,
pembacaan struktur folder dataset, dan penyimpan gambar.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")          # tanpa jendela; gambar disimpan ke file
import matplotlib.pyplot as plt    # noqa: E402
import numpy as np                 # noqa: E402

UKURAN_CITRA = (300, 300)          # ukuran citra batik di modul
EKSTENSI = {".jpg", ".jpeg", ".png", ".bmp"}

# Semua citra contoh dibaca dari berkas LOKAL di sebelah kode (tanpa internet,
# tanpa citra bawaan scikit-image):  Dataset/<MOTIF>/*.jpg   (satu folder per motif batik)
# scikit-image hanya dipakai untuk algoritma LBP, bukan sebagai sumber citra.
FOLDER_KODE = Path(__file__).resolve().parent
DATASET_DEFAULT = str(FOLDER_KODE / "Dataset")
MOTIF_CONTOH = ("KAWUNG", "PARANG", "TAMBAL")     # dipakai contoh praktikum


def buat_parser(deskripsi):
    """ArgumentParser standar: --out (folder keluaran) dan --show (tampilkan jendela)."""
    p = argparse.ArgumentParser(description=deskripsi)
    p.add_argument("--out", default="hasil", help="folder keluaran gambar/CSV (default: hasil)")
    p.add_argument("--show", action="store_true", help="tampilkan jendela matplotlib")
    return p


def simpan(fig, out, nama, tampil=False):
    """Simpan figure ke <out>/<nama>; opsional tampilkan."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / nama
    fig.savefig(f, dpi=150, bbox_inches="tight")
    print(f"  [gambar] {f}")
    if tampil:
        plt.show()
    plt.close(fig)


def baca_citra(path, ukuran=UKURAN_CITRA):
    """Baca citra grayscale lalu resize (default 300x300)."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, ukuran)


def _folder_dataset():
    """Folder dataset di sebelah kode (nama di disk 'Dataset', toleran huruf besar/kecil)."""
    for nama in ("Dataset", "DATASET", "dataset"):
        p = FOLDER_KODE / nama
        if p.is_dir():
            return p
    raise SystemExit(f"Folder dataset tidak ada di '{FOLDER_KODE}'. "
                     "Letakkan citra batik di Dataset/<MOTIF>/*.jpg")


def muat_batik(motif, indeks=0, ukuran=UKURAN_CITRA):
    """
    Baca citra batik lokal grayscale dari Dataset/<MOTIF>/, lalu resize (default 300x300).

    Berkas yang dicari: <MOTIF>_<NN>.jpg (NN = indeks+1, dua digit).
    Bila tidak ada, dipakai berkas ke-`indeks` menurut urutan sortir folder motif.
    """
    root = _folder_dataset()
    folder = root / motif
    if not folder.is_dir():
        motif_ada = sorted(d.name for d in root.iterdir() if d.is_dir())
        raise SystemExit(f"Folder motif '{folder}' tidak ada. "
                         f"Motif yang tersedia: {motif_ada}; contoh praktikum: {MOTIF_CONTOH}")
    berkas = folder / f"{motif}_{indeks + 1:02d}.jpg"
    if not berkas.exists():
        daftar = sorted(f for f in folder.iterdir() if f.suffix.lower() in EKSTENSI)
        if not daftar:
            raise SystemExit(f"Folder motif kosong: {folder}")
        if not 0 <= indeks < len(daftar):
            raise SystemExit(f"'{folder}' hanya berisi {len(daftar)} citra; "
                             f"indeks {indeks} di luar batas.")
        berkas = daftar[indeks]
    return baca_citra(berkas, ukuran)


def daftar_dataset(root):
    """
    Baca Dataset/<kelas>/*.jpg
    Return: (daftar_path, label_int (np.array), nama_kelas)
    """
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"Folder dataset '{root}' tidak ada. Siapkan Dataset/<KELAS>/*.jpg "
                         "atau jalankan buat_data_demo.py untuk data sintetis.")
    kelas = sorted(d.name for d in root.iterdir() if d.is_dir())
    if len(kelas) < 2:
        raise SystemExit(f"Butuh minimal 2 folder kelas di '{root}', ditemukan: {kelas}")
    paths, labels = [], []
    for i, k in enumerate(kelas):
        files = sorted(f for f in (root / k).iterdir() if f.suffix.lower() in EKSTENSI)
        if not files:
            raise SystemExit(f"Folder kelas kosong: {root / k}")
        paths += files
        labels += [i] * len(files)
    return paths, np.array(labels), kelas
