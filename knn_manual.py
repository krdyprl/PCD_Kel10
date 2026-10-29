# -*- coding: utf-8 -*-
"""
knn_manual.py - k-Nearest Neighbor (Modul C / Makalah 2.9).

1. Contoh hitung manual: 5 data latih, 2 fitur, 2 kelas, data uji (4,6).
   k=3 -> kelas 2 ; k=5 -> kelas 1.
2. Fungsi prediksi_knn() dengan jarak Euclidean atau Khi-Kuadrat (chi2),
   dipakai oleh batik_klasifikasi.py.

Jalankan:  python knn_manual.py
"""
import numpy as np


def jarak_chi2(A, B):
    """Matriks jarak Khi-Kuadrat antar histogram: 0.5 * sum (a-b)^2/(a+b)."""
    D = np.zeros((len(A), len(B)))
    for i, a_ in enumerate(A):
        D[i] = 0.5 * (((a_ - B) ** 2) / (a_ + B + 1e-10)).sum(axis=1)
    return D


def prediksi_knn(X_tr, y_tr, X_te, k, metrik="euclidean"):
    """Latih kNN pada (X_tr,y_tr) dan prediksi X_te. metrik: 'euclidean' | 'chi2'."""
    from sklearn.neighbors import KNeighborsClassifier
    if metrik == "chi2":
        knn = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        knn.fit(jarak_chi2(X_tr, X_tr), y_tr)
        return knn.predict(jarak_chi2(X_te, X_tr))
    return KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr).predict(X_te)


def main():
    from sklearn.neighbors import KNeighborsClassifier
    print("== Modul C: kNN manual ==")
    X = np.array([[8, 5], [7, 3], [9, 8], [5, 5], [6, 7]], dtype=np.float32)
    y = np.array([1, 1, 1, 2, 2])
    uji = np.array([[4, 6]], dtype=np.float32)

    d2 = ((X - uji) ** 2).sum(axis=1)
    rank = d2.argsort().argsort() + 1
    print("  Data     X1  X2  Kelas  Jarak^2  Peringkat")
    for i in range(len(X)):
        print(f"  Latih {i+1}   {X[i,0]:.0f}   {X[i,1]:.0f}    {y[i]}     {d2[i]:5.0f}      {rank[i]}")

    for k in (3, 5):
        urut = d2.argsort()[:k]
        pred = KNeighborsClassifier(n_neighbors=k).fit(X, y).predict(uji)[0]
        print(f"  k={k}: tetangga terdekat = Latih {[int(i)+1 for i in urut]}, "
              f"kelas {[int(y[i]) for i in urut]} -> prediksi {pred}   (kunci: k=3 -> 2, k=5 -> 1)")


if __name__ == "__main__":
    main()
