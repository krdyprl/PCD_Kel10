# -*- coding: utf-8 -*-
"""
uniform_pattern.py - Pola uniform (Modul A.3 + B Latihan 1 / Makalah 2.5).

    U = |s(g_{P-1}-g_c) - s(g_0-g_c)| + sum_{p=1}^{P-1} |s(g_p-g_c) - s(g_{p-1}-g_c)|
    Pola uniform bila U <= 2.   Jumlah pola uniform = P(P-1) + 2  (58 untuk P=8).

File ini: fungsi U + jenis pola, Tabel 2.2, hitung jumlah pola uniform,
dan gambar skema Latihan 1 (+ Tugas Mandiri [1,1,0,0,1,1,1,1]).

Jalankan:  python uniform_pattern.py
"""
import numpy as np

from common import buat_parser, simpan, plt


def jumlah_transisi(bits):
    """Ukuran keseragaman U: jumlah transisi 0/1 secara melingkar."""
    bits = list(bits)
    return sum(abs(bits[p] - bits[p - 1]) for p in range(len(bits)))   # p=0 dibandingkan bit terakhir


def jenis_pola(bits):
    """Klasifikasi heuristik berdasarkan 'Makna Fisis Pola Uniform' di modul."""
    P = len(bits)
    n1 = int(sum(bits))
    if jumlah_transisi(bits) > 2:
        return "non-uniform"
    if n1 == 0:
        return "spot terang"
    if n1 == P:
        return "flat / spot gelap"
    if n1 == P // 2:
        return "edge"
    if n1 in (1, P - 1):
        return "line end"
    return "corner"


def hitung_pola_uniform(P):
    """Hitung (dengan enumerasi) jumlah pola biner P-bit yang uniform."""
    semua = np.arange(2 ** P, dtype=np.uint32)
    bitmat = ((semua[:, None] >> np.arange(P)) & 1).astype(np.int8)
    U = np.abs(bitmat - np.roll(bitmat, 1, axis=1)).sum(axis=1)
    return int((U <= 2).sum())


# --- gambar skema Latihan 1 (sama dengan kode modul) ---

def _plot_circle(ax, center, radius, color):
    ax.add_patch(plt.Circle(center, radius, facecolor=color, edgecolor="0.5"))


def plot_lbp_model(ax, binary_values):
    """Gambar skema ketetanggaan LBP."""
    theta = np.deg2rad(45)
    R, r, w, gray = 1, 0.15, 1.5, "0.5"
    _plot_circle(ax, (0, 0), r, gray)                       # piksel pusat
    for i, facecolor in enumerate(binary_values):           # piksel tetangga
        _plot_circle(ax, (R * np.cos(i * theta), R * np.sin(i * theta)), r, str(facecolor))
    for x in np.linspace(-w, w, 4):                         # grid piksel
        ax.axvline(x, color=gray)
        ax.axhline(x, color=gray)
    ax.axis("image")
    ax.axis("off")
    size = w + 0.2
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)


def main():
    a = buat_parser("Pola uniform LBP").parse_args()

    print("== Tabel 2.2: pola uniform dan non-uniform ==")
    print(f"  {'Pola':<10}{'U':>3}  Jenis")
    for s in ["00000000", "11001111", "00111100", "11001001", "10101001"]:
        b = [int(c) for c in s]
        U = jumlah_transisi(b)
        print(f"  {s:<10}{U:>3}  {'Uniform' if U <= 2 else 'Non-uniform'}  ({jenis_pola(b)})")

    for P in (8, 16):
        print(f"\n  P={P}: pola uniform = {hitung_pola_uniform(P)}  (rumus P(P-1)+2 = {P*(P-1)+2})")

    print("\n== Latihan 1: skema pola ==")
    pola = [np.zeros(8), np.ones(8),
            np.hstack([np.ones(4), np.zeros(4)]),
            np.hstack([np.zeros(3), np.ones(5)]),
            [1, 0, 0, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 1, 1]]                       # Tugas Mandiri
    judul = ["flat", "flat", "edge", "corner", "non-uniform", "Tugas Mandiri"]
    plt.rcParams["font.size"] = 9
    fig, axes = plt.subplots(ncols=len(pola), figsize=(10, 2.3))
    for ax, v, nama in zip(axes, pola, judul):
        plot_lbp_model(ax, v)
        ax.set_title(nama)
    plt.tight_layout()
    simpan(fig, a.out, "latihan1_pola_uniform.png", a.show)

    t = pola[-1]
    U = jumlah_transisi(t)
    print(f"  Tugas mandiri {list(map(int, t))}: U = {U} -> "
          f"{'UNIFORM' if U <= 2 else 'NON-uniform'}   [kunci: uniform, 2 transisi]")


if __name__ == "__main__":
    main()
