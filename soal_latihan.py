# -*- coding: utf-8 -*-
"""
soal_latihan.py - Cek kunci jawaban Soal Latihan Bab E lewat kode (Soal 1-7).
Soal 8 bersifat konseptual (membaca histogram), jadi tidak ada kodenya.

Jalankan:  python soal_latihan.py
"""
import numpy as np

from knn_manual import main as knn_demo
from lbp_skimage import jumlah_bin, lbp_citra
from uniform_pattern import hitung_pola_uniform, jumlah_transisi


def main():
    print("== Modul E: cek kunci jawaban lewat kode ==")

    # Soal 1: searah jarum jam dari kiri-atas, ambang >= 5, bobot 1,2,4,...
    m = np.array([[6, 2, 9], [4, 5, 5], [1, 7, 3]])
    tetangga = [int(v) for v in (m[0, 0], m[0, 1], m[0, 2], m[1, 2], m[2, 2], m[2, 1], m[2, 0], m[1, 0])]
    bits = [int(t >= m[1, 1]) for t in tetangga]
    nilai = sum(b << p for p, b in enumerate(bits))
    print(f"\n  Soal 1: tetangga={tetangga}\n          bit={bits} -> LBP = {nilai}   (kunci 45)")
    assert nilai == 45

    # Soal 2
    print("\n  Soal 2:")
    for s in ["00011000", "01010101", "11110000", "10011001", "11111111"]:
        U = jumlah_transisi([int(c) for c in s])
        print(f"     {s}: U={U} -> {'uniform' if U <= 2 else 'bukan uniform'}")

    # Soal 3: P=16
    P = 16
    semua = np.arange(2 ** P, dtype=np.uint32)

    def rot_min(v):
        r, out = v.copy(), v.copy()
        for _ in range(P - 1):
            r = ((r << 1) | (r >> (P - 1))) & (2 ** P - 1)
            out = np.minimum(out, r)
        return out

    n_ror = len(np.unique(rot_min(semua)))
    n_nri = hitung_pola_uniform(P) + 1
    print(f"\n  Soal 3 (P=16): default={jumlah_bin(P,'default'):,}  ror={n_ror:,}  "
          f"uniform={jumlah_bin(P,'uniform')}  nri_uniform={n_nri}")
    print("          (kunci 65.536 / 4.116 / 18 / 243)")
    assert (n_ror, jumlah_bin(P, "uniform"), n_nri) == (4116, 18, 243)

    # Soal 4
    rng = np.random.default_rng(0)
    img = rng.integers(0, 200, (64, 64)).astype(np.uint8)
    l1 = lbp_citra(img, 8, 1, "default")
    l2 = lbp_citra((img + 50).astype(np.uint8), 8, 1, "default")
    sama = np.array_equal(l1[2:-2, 2:-2], l2[2:-2, 2:-2])     # tepi memakai padding 0
    print(f"\n  Soal 4: LBP(img) == LBP(img+50) di bagian dalam citra? {sama}   (kunci: ya)")
    assert sama

    # Soal 5
    kecil = rng.integers(0, 256, (100, 100)).astype(np.uint8)
    besar = np.tile(kecil, (2, 2))
    hk = np.histogram(lbp_citra(kecil, 8, 1, "uniform"), bins=10, range=(0, 10))[0]
    hb = np.histogram(lbp_citra(besar, 8, 1, "uniform"), bins=10, range=(0, 10))[0]
    print(f"\n  Soal 5: total piksel {hk.sum()} vs {hb.sum()} (mentah tidak sebanding), "
          f"normalisasi -> selisih maks = {np.abs(hk/hk.sum() - hb/hb.sum()).max():.4f}")

    # Soal 6
    print("\n  Soal 6:")
    knn_demo()

    # Soal 7
    print(f"\n  Soal 7: dimensi histogram uniform P=8 = {jumlah_bin(8, 'uniform')} "
          f"(9 nilai jumlah-bit-1 + 1 bin non-uniform)   (kunci 10)")


if __name__ == "__main__":
    main()
