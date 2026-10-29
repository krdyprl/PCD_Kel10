# -*- coding: utf-8 -*-
"""
lbp_manual.py - LBP satu piksel pada ketetanggaan 3x3 (Modul A.2 + B Latihan 3).

Empat langkah: (1) pusat sebagai ambang, (2) binerisasi tetangga >= pusat -> 1,
(3) pembobotan 2^p, (4) penjumlahan -> nilai desimal.

    LBP(xc,yc) = sum_{p=0}^{P-1} s(g_p - g_c) * 2^p ,  s(x)=1 jika x>=0 else 0

Urutan tetangga (konvensi modul): kiri-atas, atas, kanan-atas, kanan,
kanan-bawah, bawah, kiri-bawah, kiri.  scikit-image memakai urutan lain
(g_0 di sisi kanan, berlawanan jarum jam, dengan interpolasi), jadi nilai
desimalnya bisa berbeda walau jenis polanya sama.

Jalankan:  python lbp_manual.py
"""
import numpy as np


def lbp_manual_3x3(patch):
    """Return (bits, nilai_desimal) untuk patch 3x3."""
    patch = np.asarray(patch)
    c = patch[1, 1]
    tetangga = [patch[0, 0], patch[0, 1], patch[0, 2], patch[1, 2],
                patch[2, 2], patch[2, 1], patch[2, 0], patch[1, 0]]
    bits = [1 if n >= c else 0 for n in tetangga]
    nilai = sum(b * 2 ** p for p, b in enumerate(bits))
    return bits, nilai


def main():
    print("== Latihan 3: Perhitungan LBP Manual ==")
    patch = np.array([[3, 7, 2],
                      [8, 4, 1],
                      [2, 3, 5]])
    bits, nilai = lbp_manual_3x3(patch)
    print("  Bit biner :", bits, "  (kunci: [0, 1, 0, 0, 1, 0, 0, 1])")
    print("  Rincian   :", " + ".join(f"{b}x{2**p}" for p, b in enumerate(bits)), "=", nilai)
    print("  Nilai LBP :", nilai, "  (kunci: 146)")
    assert bits == [0, 1, 0, 0, 1, 0, 0, 1] and nilai == 146

    # Bandingkan dengan fungsi scikit-image (urutan bobot & interpolasi berbeda -> nilai berbeda)
    from skimage.feature import local_binary_pattern
    lbp_sk = local_binary_pattern(patch.astype(np.uint8), 8, 1, "default")
    print(f"  scikit-image (R=1, P=8): {lbp_sk[1, 1]:.0f}   (kunci modul: 28; beda dari 146 itu wajar:\n"
          "     skimage mulai dari sisi kanan, berlawanan jarum jam, dan menginterpolasi titik diagonal)")

    # Sifat penting LBP: tahan perubahan kecerahan merata (selama tidak terpotong 0/255)
    _, nilai_terang = lbp_manual_3x3(patch + 50)
    print(f"  Patch + 50 (lebih terang): LBP = {nilai_terang}  -> sama? {nilai_terang == nilai}")


if __name__ == "__main__":
    main()
