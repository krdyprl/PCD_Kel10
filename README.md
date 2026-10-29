# PCD_Kel10

## Praktikum Ekstraksi Ciri Tekstur (LBP) pada Citra Batik

Paket ini berisi kode Python untuk praktikum **Pengolahan Citra Digital — Ekstraksi Ciri
Tekstur dengan Local Binary Pattern (LBP)**. Semua citra dibaca dari folder `Dataset/`
di dalam paket ini, jadi **tidak perlu internet** dan tidak ada citra bawaan scikit-image
yang diunduh otomatis.

Alur praktikum: LBP satu piksel (manual) → LBP melingkar (P, R) → LBP dengan
scikit-image → histogram LBP sebagai vektor fitur → pola uniform → ekstraksi fitur pada
enam motif batik → klasifikasi kNN.

---

## 1. Yang perlu diinstal

Diuji pada **Python 3.13.5** (Windows). Versi paket yang dipakai saat modul ini dibuat:

| Paket | Versi diuji | Dipakai untuk |
|---|---|---|
| numpy | 2.2.6 | operasi matriks dan citra |
| matplotlib | 3.10.3 | semua gambar keluaran |
| opencv-python | 4.13.0 | baca citra (`cv2.imread`), resize, dataset sintetis |
| scikit-image | 0.25.2 | `local_binary_pattern`, `label2rgb` |
| scikit-learn | 1.8.0 | kNN, split data, cross-validation, confusion matrix |

Pasang sekaligus:

```bash
pip install numpy matplotlib opencv-python scikit-image scikit-learn
```

Cek instalasi:

```bash
python -c "import cv2, numpy, matplotlib, skimage, sklearn; print('siap')"
```

Kalau muncul `ModuleNotFoundError`, paket yang disebut di pesan error itu belum terpasang.

---

## 2. Struktur folder

```
lbp_tekstur/
├── README.md                     ← file ini
├── common.py                     ← fungsi bantu (dipakai skrip lain, tidak dijalankan sendiri)
├── lbp_manual.py                 ← LBP 3x3 dihitung manual
├── lbp_melingkar.py              ← LBP melingkar buatan sendiri + titik sampling
├── lbp_skimage.py                ← LBP dengan scikit-image, tabel method
├── lbp_histogram.py              ← histogram LBP sebagai vektor fitur
├── uniform_pattern.py            ← ukuran U dan pola uniform
├── latihan_bata.py               ← Latihan 2: LBP pada tekstur batik
├── knn_manual.py                 ← kNN dihitung manual + fungsi prediksi_knn()
├── soal_latihan.py               ← pengecekan kunci Soal Latihan 1–7
├── tiga_tekstur.py               ← bandingkan 3 motif batik
├── batik_demo.py                 ← praktikum batik bagian 1: uniform vs default
├── batik_ekstraksi.py            ← ekstraksi fitur semua motif + tabel CSV
├── batik_klasifikasi.py          ← LBP + kNN, akurasi, confusion matrix
├── buat_data_demo.py             ← pembuat dataset SINTETIS (bukan batik asli)
├── Dataset/                      ← 6 motif batik, total 132 citra
│   ├── KAWUNG/ (26)  MEGAMENDUNG/ (20)  PARANG/ (20)
│   ├── SIDOLUHUR/ (20)  SIDOMUKTI/ (26)  TAMBAL/ (20)
│   ├── daftar_asal.csv           ← asal tiap file + ukuran aslinya
│   ├── SUMBER.txt                ← sumber citra dataset
│   └── _contoh_semua.jpg         ← satu gambar berisi contoh semua motif
├── TEKSTUR/                      ← 3 tekstur umum untuk latihan: brick, grass, gravel
└── hasil/                        ← semua gambar dan CSV yang dihasilkan skrip
```

**Jalankan semua perintah dari folder utama paket ini** (folder yang berisi `common.py`),
karena setiap skrip mengimpor `common.py`, `lbp_skimage.py`, dan seterusnya dari folder
yang sama.

```bash
cd lbp_tekstur
python lbp_manual.py
```

Folder `hasil/` dibuat otomatis kalau belum ada. Isinya akan tertimpa kalau skrip
dijalankan ulang.

---

## 3. Mulai cepat (5 menit)

```bash
python lbp_manual.py        # LBP satu piksel, hasil 146
python uniform_pattern.py   # pola uniform, jumlah pola untuk P=8 dan P=16
python lbp_skimage.py       # tabel method + pengaruh R
python tiga_tekstur.py      # 3 motif batik berdampingan
```

Setiap skrip mencetak angka pengamatan ke layar dan menyimpan gambar ke `hasil/`.

---

## 4. Penjelasan tiap skrip

### 4.1 Dasar teori LBP

**`lbp_manual.py`** — LBP satu piksel pada ketetanggaan 3×3, empat langkah: pusat sebagai
ambang, binerisasi, pembobotan 2^p, penjumlahan.

```bash
python lbp_manual.py
```

Keluaran: bit `[0,1,0,0,1,0,0,1]` → nilai **146**. Skrip juga menunjukkan bahwa
`skimage` memberi **28** untuk patch yang sama. Itu wajar dan bukan kesalahan: `skimage`
mulai dari titik di sisi kanan lalu berlawanan arah jarum jam dan menginterpolasi titik
diagonal, sedangkan modul memakai urutan mulai kiri-atas searah jarum jam. Terakhir skrip
membuktikan sifat LBP yang penting: patch yang ditambah 50 (lebih terang merata) memberi
nilai LBP yang sama.

**`lbp_melingkar.py`** — LBP pada ketetanggaan melingkar: P titik pada lingkaran
berjari-jari R, titik yang tidak jatuh tepat di grid piksel dihitung dengan interpolasi
bilinear.

```bash
python lbp_melingkar.py
```

Keluaran: `hasil/lbp_melingkar_titik_sampling.png` (skema titik untuk R = 1, 2, 3) dan
pembuktian bahwa fungsi LBP buatan sendiri memberi hasil identik dengan
`skimage.feature.local_binary_pattern` pada citra acak.

**`lbp_skimage.py`** — LBP dengan fungsi pustaka, plus pengecekan klaim modul.

```bash
python lbp_skimage.py
```

Keluaran: tabel lima method (`default`, `ror`, `uniform`, `nri_uniform`, `var`) beserta
jumlah bin teori dan jumlah nilai yang benar-benar muncul pada citra batik Kawung, tabel
pengaruh R dan P (P = 8R) dengan persentase pola uniform, dan
`hasil/lbp_skimage_pengaruh_R_batik_kawung.png`.

**`lbp_histogram.py`** — histogram LBP sebagai vektor fitur.

```bash
python lbp_histogram.py
```

Keluaran: histogram 10 bin (P + 2 = 10 untuk P = 8, method `uniform`) yang jumlahnya
dinormalisasi menjadi 1, pembuktian bahwa perhitungan manual dengan `bincount` sama dengan
`np.histogram`, dan demonstrasi kenapa normalisasi perlu: citra 300×300 dan 256×256
mempunyai frekuensi mentah berbeda tetapi histogram ternormalisasinya hampir sama. Gambar:
`hasil/lbp_histogram_batik_kawung.png`.

**`uniform_pattern.py`** — ukuran keseragaman U dan pola uniform.

```bash
python uniform_pattern.py
```

Keluaran: Tabel pola (uniform/non-uniform beserta jenisnya: flat, edge, corner, line end,
spot), jumlah pola uniform hasil enumerasi untuk P = 8 (58) dan P = 16 (242), cocok dengan
rumus P(P−1) + 2. Gambar: `hasil/latihan1_pola_uniform.png` (termasuk skema Tugas Mandiri
`[1,1,0,0,1,1,1,1]`).

**`knn_manual.py`** — k-Nearest Neighbor dihitung manual dan fungsi `prediksi_knn()`.

```bash
python knn_manual.py
```

Keluaran: contoh 5 data latih, 2 fitur, 2 kelas, data uji (4, 6) → **k = 3 memberi kelas 2,
k = 5 memberi kelas 1**, plus fungsi jarak Khi-Kuadrat yang dipakai
`batik_klasifikasi.py`.

### 4.2 Latihan pada tekstur dan batik

**`latihan_bata.py`** — Latihan 2: LBP pada tekstur batik, default R = 3, P = 24.

```bash
python latihan_bata.py            # R=3, P=24 (default)
python latihan_bata.py --radius 1 # pertanyaan diskusi: apa yang berubah bila R=1?
```

Keluaran: citra LBP dengan piksel yang termasuk pola edge, flat, dan corner disorot
berwarna, plus histogram dengan bin terkait ditandai. Gambar:
`hasil/latihan2_batik_kawung_R3.png` (atau `_R1`).

**`tiga_tekstur.py`** — membandingkan histogram LBP tiga motif (KAWUNG, PARANG, TAMBAL)
dengan P = 8, R = 1, uniform.

```bash
python tiga_tekstur.py
```

Keluaran: tabel histogram ternormalisasi (%) per bin, bin tertinggi tiap motif, dan
`hasil/tiga_tekstur_histogram.png` (baris: citra asli, citra LBP, histogram).

**`batik_demo.py`** — praktikum batik bagian 1: LBP `uniform` dibandingkan `default`.

```bash
python batik_demo.py                    # semua motif di Dataset/
python batik_demo.py --radius 2         # R lain
python batik_demo.py --gambar path/gambar.jpg --nama "nama saya"
```

Keluaran: untuk tiap motif, satu gambar berisi citra asli, citra LBP uniform + histogramnya,
dan citra LBP default + histogramnya → `hasil/batik_<MOTIF>_R1.png`. Skrip juga mencetak
tabel histogram uniform per motif.

**`batik_ekstraksi.py`** — ekstraksi fitur LBP untuk semua motif di `Dataset/`.

```bash
python batik_ekstraksi.py                  # konfigurasi dasar
python batik_ekstraksi.py --tambahan       # + method ror dan nri_uniform
python batik_ekstraksi.py --default-r2     # + method default untuk R=2 (65.536 bin)
```

Keluaran:
- `hasil/ekstraksi_<MOTIF>.png` — satu gambar per motif (citra asli, citra LBP, histogram)
- `hasil/tabel_pengamatan.csv` — tabel pengamatan seluruh citra dan konfigurasi
- `hasil/histogram_rata_rata_per_motif.png` — histogram rata-rata tiap motif
- `hasil/keterpisahan.csv` — jarak antar-kelas dan dalam-kelas
- angka bantu analisis di layar: jarak Khi-Kuadrat antar motif, pengaruh R dan P
  (dimensi vektor, persentase pola uniform, entropi), perbandingan `default` vs `uniform`,
  akurasi 1-NN leave-one-out, dan rasio jarak antar-kelas/dalam-kelas

Method `default` untuk R = 3 sengaja tidak disediakan: 2^24 = 16,7 juta bin, tidak praktis.

**`batik_klasifikasi.py`** — praktikum batik bagian 2: LBP + kNN, akurasi dan confusion
matrix.

```bash
python batik_klasifikasi.py                  # 4 konfigurasi modul, k=3, split 70:30
python batik_klasifikasi.py --cv 5           # akurasi 5-fold cross-validation
python batik_klasifikasi.py --k 1 3 5        # variasi nilai k
python batik_klasifikasi.py --metrik chi2    # jarak Khi-Kuadrat
python batik_klasifikasi.py --laporan        # presisi, recall, F1
python batik_klasifikasi.py --lengkap        # grid konfigurasi lebih luas
```

Konfigurasi default mengikuti tabel hasil percobaan modul: (R=1, `default`),
(R=1, `uniform`), (R=2, `uniform`), (R=3, `uniform`) dengan P = 8R.

Keluaran: `hasil/hasil_eksperimen.csv` (akurasi tiap konfigurasi) dan
`hasil/confusion_matrix.png`. Skrip ini yang paling lama berjalan karena menghitung LBP
dengan P = 24 untuk seluruh citra. Dataset butuh minimal 2 kelas dan minimal 10 citra per
kelas, dan `Dataset/` sudah memenuhi (20–26 citra per motif).

**`soal_latihan.py`** — pengecekan kunci Soal Latihan 1–7 lewat kode.

```bash
python soal_latihan.py
```

Keluaran: nilai LBP Soal 1 (45), klasifikasi uniform Soal 2, jumlah pola ror/uniform untuk
P = 16, dan pemeriksaan soal lainnya. Soal 8 (membaca histogram) bersifat konseptual jadi
tidak ada kodenya.

### 4.3 Pendukung

**`buat_data_demo.py`** — membuat dataset **sintetis** untuk menguji alur kode tanpa
memakai batik asli (KAWUNG-sintetis dan SIDOMUKTI-sintetis).

```bash
python buat_data_demo.py                  # 20 citra per kelas ke Dataset_sintetis/
python buat_data_demo.py --n 30 --seed 7
```

Skrip ini menulis ke folder terpisah (`Dataset_sintetis/`) sehingga **tidak menimpa**
`Dataset/`. Berguna kalau ingin mencoba `batik_klasifikasi.py` dengan data buatan sendiri:

```bash
python buat_data_demo.py
python batik_klasifikasi.py --dataset Dataset_sintetis
```

**`common.py`** — fungsi bantu bersama: pembaca argumen `--out` dan `--show`, pembacaan
citra grayscale 300×300 (`baca_citra`, `muat_batik`), pendaftaran isi folder dataset
(`daftar_dataset`), dan penyimpanan gambar (`simpan`). Tidak dijalankan langsung.

---

## 5. Opsi baris perintah yang sering dipakai

| Opsi | Berlaku di | Arti |
|---|---|---|
| `--out FOLDER` | hampir semua skrip | folder keluaran (default `hasil`) |
| `--show` | hampir semua skrip | tampilkan jendela matplotlib, bukan hanya simpan file |
| `--dataset FOLDER` | `batik_demo`, `batik_ekstraksi`, `batik_klasifikasi` | folder dataset lain |
| `--radius R` | `latihan_bata`, `batik_demo` | radius ketetanggaan |
| `--k 1 3 5` | `batik_klasifikasi` | nilai k untuk kNN |
| `--cv N` | `batik_klasifikasi` | akurasi N-fold cross-validation |
| `--metrik euclidean\|chi2` | `batik_klasifikasi` | metrik jarak |
| `--laporan`, `--lengkap` | `batik_klasifikasi` | presisi/recall/F1, grid konfigurasi lebih luas |
| `--tambahan`, `--default-r2` | `batik_ekstraksi` | method tambahan |

Bantuan tiap skrip: `python nama_skrip.py --help`

Secara bawaan matplotlib memakai backend tanpa jendela (`Agg`) supaya aman dijalankan di
lab, dan semua gambar disimpan ke `hasil/`. Tambahkan `--show` kalau ingin jendela muncul.

---

## 6. Dataset

`Dataset/` berisi enam motif batik dari Yogyakarta, total 132 citra:

| Motif | Jumlah citra |
|---|---|
| KAWUNG | 26 |
| MEGAMENDUNG | 20 |
| PARANG | 20 |
| SIDOLUHUR | 20 |
| SIDOMUKTI | 26 |
| TAMBAL | 20 |

- Nama file berpola `<MOTIF>_<NN>.jpg`, contoh `KAWUNG_01.jpg`. Skrip `muat_batik()`
  mengambil berkas ke-`indeks` sesuai pola ini, jadi **jangan mengubah nama file** kalau
  ingin contoh di modul tetap cocok.
- `daftar_asal.csv` mencatat kelas, nama file, file asal di dataset sumber, dan ukuran
  aslinya.
- `SUMBER.txt` mencantumkan sumber citra.
- Hanya berkas `.jpg`, `.jpeg`, `.png`, dan `.bmp` yang dibaca skrip. Berkas dengan
  ekstensi lain (misalnya `.webp`) diabaikan tanpa pesan error.
- `TEKSTUR/` berisi tiga tekstur umum (brick, grass, gravel) untuk latihan LBP di luar batik.

Semua citra dibaca grayscale lalu di-resize ke 300×300 piksel agar ukuran vektor fitur dan
waktu hitung seragam.

---

## 7. Kalau ada masalah

| Gejala | Penyebab dan solusi |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` (atau `skimage`, `sklearn`) | paket belum terpasang di Python yang dipakai. Jalankan `pip install ...` seperti bagian 1, dan pastikan `python` yang dipanggil adalah Python yang sama |
| `Folder dataset tidak ada di '...'` | skrip dijalankan dari luar folder paket, atau folder `Dataset/` ikut berpindah. Jalankan dari folder utama paket |
| `Folder motif '...' tidak ada. Motif yang tersedia: [...]` | nama motif salah ketik. Yang tersedia: KAWUNG, MEGAMENDUNG, PARANG, SIDOLUHUR, SIDOMUKTI, TAMBAL |
| `ImportError` saat `from common import ...` | file `.py` dipisah dari folder utama. Semua skrip harus berada satu folder dengan `common.py` |
| Gambar tidak muncul di layar | memang disengaja (backend `Agg`). Tambahkan `--show`, atau buka file PNG di `hasil/` |
| Angka LBP manual (146) beda dengan scikit-image (28) | bukan error, urutan tetangga dan interpolasi berbeda. Penjelasan ada di `lbp_manual.py` dan di modul |
| `MemoryError` pada `lbp_histogram.py` | method `default` dengan P besar menghasilkan jutaan bin. Gunakan method `uniform` (P + 2 bin) |

---

## 8. Hubungan dengan modul teori

Modul lengkap (tujuan praktikum, dasar teori LBP, pola uniform, langkah kerja, tabel
pengamatan, soal latihan beserta kunci) ada di dokumen terpisah
**`Modul_Praktikum_Ekstraksi_Ciri_Tekstur.docx`** yang dibagikan bersama paket ini.
README ini hanya menjelaskan sisi teknis: cara memasang, cara menjalankan, dan apa yang
dihasilkan setiap skrip.
