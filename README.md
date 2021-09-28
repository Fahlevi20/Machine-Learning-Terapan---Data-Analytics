# Machine-Learning-Terapan---Data-Analytics
# Laporan Proyek Machine Learning - Mukhammad Fahlevi Ali Rafsanjani

## Domain Proyek
### Latar Belakang

10.000 daftar mobil bekas yang dipisahkan menjadi beberapa file sesuai dengan masing-masing produsen mobil.
Kumpulan data yang sudah dibersihkan berisi informasi :
- model = Volswagen Model(T-Rock, Golf, Polo, T-Cross, Tiguan, Caddy, Etc..)
- year = Registration Year
- price = Price in Pound Britania (£)
- transmission = type of gearbox (Automatic, Manual and Semi-Auto)
- mileage = distance used
- fuelType = Engine Fuel (Diesel, Petrol, Hybird and Other)
- tax = Road Tax
- mpg = Miles per Galon
- engineSize= size in litres

### Permasalahan
- tahun 2021 maraknya jumlah pembelian kendaraan mobil karena kasus covid di dunia mulai menurun sehingga pemerintah memperbolehkan para pekerja untuk membolehkan aktivitas keluar rumah dan wisata, oleh karena itu perlunya prediksi harga jual mobil Volswagen untuk meningkatkan pembelian kendaraan dengan harga terbaik dengan menggunakan model yang kuat.

Pada bagian ini, Anda menguraikan secara singkat informasi mengenai pilihan domain yang akan diselesaikan permasalahannya. 
Sebagai contoh, Anda memilih domain telekomunikasi. Anda dapat menguraikan bagian ini dengan pendekatan berikut:
- Sertakan informasi atau latar belakang yang relevan mengenai pemilihan domain ini.
- Jelaskan mengapa dan bagaimana masalah dalam domain yang Anda pilih tersebut harus diselesaikan.
- Sertakan pula hasil riset terkait atau referensi yang relevan. Anda dapat menggunakan [tautan](https://scholar.google.com/) untuk menuliskan referensi atau rujukan.

Referensi data : https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes

## Business Understanding
Bagian ini menjelaskan proses klarifikasi masalah dan mengajukan minimal satu solusi untuk menyelesaikan permasalahan. Bagian laporan ini mencakup:

### Problem Statements
Tuliskan problem statement Anda di sini. Anda dapat menggunakan kalimat tanya untuk mendefinisikan bagian ini.
- model regresi mana yang dapat menghasilkan prediksi terbaik?

### Goals
Tuliskan dan jelaskan goal proyek yang ingin Anda capai di bagiani ini. Anda dapat menggunakan bullet point jika memiliki lebih dari satu goals proyek.
- ingin mencari model regresi yang kuat dan dapat memprediksi harga jual mobil VolkSwagen.
### Solution statements
Solusi model yang kami berikan menggunakan Linear dan Polynomial, karena dengan metode tersebut cocok untuk melakukan prediksi terkait harga. 
Untuk model yang digunakan :
- **Linear Model**. Model linier adalah cara untuk menggambarkan variabel respon dalam hal kombinasi linier variabel prediktor. Respon harus berupa variabel kontinu dan paling tidak terdistribusi secara normal. 
- **Polynomial Model**. Model polinomial adalah alat yang hebat untuk menentukan faktor input mana yang mendorong respons dan ke arah mana. Ini juga merupakan model yang paling umum digunakan untuk analisis eksperimen yang dirancang. Model polinomial kuadratik (orde kedua) untuk dua variabel penjelas memiliki bentuk persamaan di bawah ini. 
- **LinearRegression**. Linear Regression cocok dengan model linier dengan koefisien w = (w1, …, wp) untuk meminimalkan jumlah sisa kuadrat antara target yang diamati dalam                             kumpulan data, dan target yang diprediksi oleh pendekatan linier. 
- **DecisionTreeRegressor**. Decision Tree Regressor. Decision Tree membangun model regresi atau klasifikasi dalam bentuk struktur pohon. Ini memecah dataset menjadi subset yang lebih kecil dan lebih kecil sementara pada saat yang sama pohon keputusan terkait dikembangkan secara bertahap. Hasil akhirnya adalah pohon dengan simpul keputusan dan simpul daun. 
- **MLPRegressor**. Multi Layer Perceiptron Regressor. Model ini mengoptimalkan kesalahan kuadrat menggunakan LBFGS atau penurunan gradien stokastik.
## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst
Semua variabel yang terdapat di Dataset vw (Volkswagen Dataset) :
- model = Volswagen Model(T-Rock, Golf, Polo, T-Cross, Tiguan, Caddy, Etc..)
- year = Registration Year
- price = Price in Pound Britania (£)
- transmission = type of gearbox (Automatic, Manual and Semi-Auto)
- mileage = distance used
- fuelType = Engine Fuel (Diesel, Petrol, Hybird and Other)
- tax = Road Tax
- mpg = Miles per Galon
- engineSize= size in litres
## Data Preparation
Pada bagian ini Anda menjelaskan teknik yang digunakan pada tahapan Data Preparation. 
- **Exploratory Data Analysis (EDA)**. Exploratory Data Analysis adalah pendekatan untuk menganalisis kumpulan data untuk merangkum karakteristik utamanya, seringkali dengan metode visual. EDA digunakan untuk melihat apa yang data dapat memberitahu kami sebelum tugas pemodelan. dengan EDA dapat mudah untuk melakukan analisis data dengan cepat.
  - Disini saya menganalisis jumlah dari baris dan kolom yaitu sebanyak 15157 dan 9 kolom.
-  **Data Cleaning/Cleansing**. Pembersihan data (Data Cleaning/Cleansing) adalah proses memperbaiki atau menghapus data yang salah, rusak, salah format, duplikat, atau tidak lengkap dalam kumpulan data. Saat menggabungkan beberapa sumber data, ada banyak peluang untuk data diduplikasi atau diberi label yang salah.
    - disini kita melakukan data Cleaning untuk memeriksa apakah ada data yang kosong, lalu mendrop kolom yang tidak penting seperti mendrop Kolom Year, ini dilakukan karena kolom tahun akan berpengaruh dalam prediksi harga sehingga saya mendrop kolom tersebut dan menggantinya dengan feature engineering. 
                             
-  Terapkan minimal satu teknik data preparation dan jelaskan proses yang dilakukan.
- Jelaskan alasan mengapa Anda perlu menerapkan teknik tersebut pada tahap Data Preparation. 
- list item

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. 

Jelaskan bagaimana Anda melakukan proses modeling dalam proyek. Misalnya, Anda menggunakan satu algoritma kemudian melakukan improvement dari baseline model atau Anda menggunakan dua atau lebih algoritma kemudian membandingkan performanya.

Sajikan model terbaik Anda sebagai solusi.
Jelaskan pula hasil dari model Anda (misal, hasil prediksi).

## Evaluation
Bagian ini menjelaskan mengenai metrik evaluasi yang digunakan untuk mengukur kinerja model. Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan dan bagaimana formulanya
- Kelebihan dan kekurangan metrik
- Bagaimana cara menerapkannya ke dalam kode.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
_Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
