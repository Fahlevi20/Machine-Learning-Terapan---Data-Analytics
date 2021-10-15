# Machine-Learning-Terapan
# Laporan Proyek Machine Learning - Mukhammad Fahlevi Ali Rafsanjani

## Domain Proyek
Project Machine Learning Terapan : membuat model Predictive Analysis, menggunakan dataset yang berdomain ekonomi mengenai prediksi harga mobil VolkSwagen.
### Latar Belakang
Latar Belakang pemilihan topik ini adalah dikarenakan ingin melihat tingkat penjualan mobil bekas, dimana dalam kasus ini VolkSwagen, dengan fitur - fitur tertentu yang dapat berpenaguh pada nilai di pasar.

## Business Understanding
Pentingnya bagi para pemilik mobil jika ingin menjual mobilnya perlu untuk melihat harga yang terdapat di pasaran, namun bagi para penjual cukup sulit untuk menentukan harga mobilnya agar mendapatkan harga yang sesuai keinginannya dan juga dapat terjual dengan mudah, oleh karena itu pembuatan prediksi harga yang cocok penting.
### Problem Statements
-Berapa harga mobil bekas dengan jenis transmisi, jarak tempuh dan ukuran mesin yang ditentukan?
- berdasarkan karakteristik yang tersedia manakan yang paling berpengaruh?
### Goals
ingin membuat model Machine Learning yang dapat memberikan prediksi harga mobil bekas VolkSwagen dengan harga terbaik dan menggunakan karaktersitik yang tersedia.
### Solution statements
Solusi model yang kami berikan menggunakan Linear dan Polynomial, karena dengan metode tersebut cocok untuk melakukan prediksi terkait harga. 
Untuk model yang digunakan :
- **Linear Model**. Model linier adalah cara untuk menggambarkan variabel respon dalam hal kombinasi linier variabel prediktor. Respon harus berupa variabel kontinu dan paling tidak terdistribusi secara normal. 
- **Polynomial Model**. Model polinomial adalah alat yang hebat untuk menentukan faktor input mana yang mendorong respons dan ke arah mana. Ini juga merupakan model yang paling umum digunakan untuk analisis eksperimen yang dirancang. Model polinomial kuadratik (orde kedua) untuk dua variabel penjelas memiliki bentuk persamaan di bawah ini. 
- **LinearRegression**. Linear Regression cocok dengan model linier dengan koefisien w = (w1, …, wp) untuk meminimalkan jumlah sisa kuadrat antara target yang diamati dalam                             kumpulan data, dan target yang diprediksi oleh pendekatan linier. 
- **DecisionTreeRegressor**. Decision Tree Regressor. Decision Tree membangun model regresi atau klasifikasi dalam bentuk struktur pohon. Ini memecah dataset menjadi subset yang lebih kecil dan lebih kecil sementara pada saat yang sama pohon keputusan terkait dikembangkan secara bertahap. Hasil akhirnya adalah pohon dengan simpul keputusan dan simpul daun. 
- **MLPRegressor**. Multi Layer Perceiptron Regressor. Model ini mengoptimalkan kesalahan kuadrat menggunakan LBFGS atau penurunan gradien stokastik.
## Data Understanding
 
Untuk mengunduh Dataset dapat mengunjungi link berikut [Kaggle Dataset](https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes).
Disini menggunakan [*100,000 UK Used Car Data set*](https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes?select=cclass.csv) dari situs Kaggle yang berisi data tentang mobil bekas yang terjual di UK dengan variable, mileage, model, engineSize, year, transmission, fuelType, dan price yang menjadi label pada data ini. Dataset ini berisi  3899 dengan 7 kolom dengan 2 kategorikal dan 5 numerikal.
 
Variabel - variabel yang terdapat di Dataset vw (Volkswagen Dataset) :
- model = Volswagen Model(T-Rock, Golf, Polo, T-Cross, Tiguan, Caddy, Etc..)
- year = Registration Year
- price = Price in Pound Britania (£)
- transmission = type of gearbox (Automatic, Manual and Semi-Auto)
- mileage = distance used
- fuelType = Engine Fuel (Diesel, Petrol, Hybird and Other)
- tax = Road Tax
- mpg = Miles per Galon
- engineSize= size in litres

Data Loading sebagai berikut

|model           |year|price|transmission|mileage|fuelType|tax|mpg  |engineSize|
|----------------|----|-----|------------|-------|--------|---|-----|----------|
| T-Roc          |2019|25000|Automatic   |13904  |Diesel  |145|49.6 |2.0       |
| T-Roc          |2019|26883|Automatic   |4562   |Diesel  |145|49.6 |2.0       |
| T-Roc          |2019|20000|Manual      |7414   |Diesel  |145|50.4 |2.0       |
| T-Roc          |2019|33492|Automatic   |4825   |Petrol  |145|32.5 |2.0       |
| T-Roc          |2019|22900|Semi-Auto   |6500   |Petrol  |150|39.8 |1.5       |
| T-Roc          |2020|31895|Manual      |10     |Petrol  |145|42.2 |1.5       |
| T-Roc          |2020|27895|Manual      |10     |Petrol  |145|42.2 |1.5       |
| T-Roc          |2020|39495|Semi-Auto   |10     |Petrol  |145|32.5 |2.0       |
| T-Roc          |2019|21995|Manual      |10     |Petrol  |145|44.1 |1.0       |
| T-Roc          |2019|23285|Manual      |10     |Petrol  |145|42.2 |1.5       |
|---|---|---|---|---|---|---|---|---|
| Eos            |2015|12495|Manual      |41850  |Diesel  |125|58.9 |2.0       |
| Eos            |2014|8950 |Manual      |58000  |Diesel  |125|58.9 |2.0       |
| Eos            |2006|2995 |Manual      |92640  |Diesel  |200|48.0 |2.0       |
| Eos            |2012|5990 |Manual      |74000  |Diesel  |125|58.9 |2.0       |
| Fox            |2008|1799 |Manual      |88102  |Petrol  |145|46.3 |1.2       |
| Fox            |2009|1590 |Manual      |70000  |Petrol  |200|42.0 |1.4       |
| Fox            |2006|1250 |Manual      |82704  |Petrol  |150|46.3 |1.2       |
| Fox            |2007|2295 |Manual      |74000  |Petrol  |145|46.3 |1.2       |

Dataset tersebut juga dapat dilihat deskripsi statistiknya seperti berikut:

|               year|         price|        mileage|           tax  |        mpg| 
|------|-------------|------------------------------|---------------|------------|
|count | 15157.000000|  15157.000000|   15157.000000|  15157.000000 | 15157.000000|   
|mean  |  2017.255789|  16838.952365|   22092.785644|    112.744277 |    53.753355|   
|std   |     2.053059|   7755.015206|   21148.941635|     63.482617 |    13.642182|   
|min   |  2000.000000|    899.000000|       1.000000|      0.000000 |     0.300000|   
|25%   |  2016.000000|  10990.000000|    5962.000000|     30.000000 |    46.300000|   
|50%   |  2017.000000|  15497.000000|   16393.000000|    145.000000 |    53.300000|   
|75%   |  2019.000000|  20998.000000|   31824.000000|    145.000000 |    60.100000|   
|max   |  2020.000000|  69994.000000|  212000.000000|    580.000000 |   188.300000|   

#### Visualization Data

Apabila jenis data dikategorikan seperti diatas dapat dilihat bentuk tabel dan grafik masing masing data sebagai berikut:

Informasi General Dataset ...
|count|unique|top|freq|
|:---:|:---:|:---:|:---:|
|15157|3|Manual|9417|

![Transmission](https://raw.githubusercontent.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/main/Transmission.jpg?token=APMXFUPIEKV4O5IVT4GL5DTBNFHG2)

|count|unique|top|freq|
|:---:|:---:|:---:|:---:|
|15157|4|Petrol|8553|

![fuelType](https://raw.githubusercontent.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/main/Barplot%20fuelType.jpg?token=APMXFUIR5A3SVXM7JNHSXRLBNFHXY)

 **Total Pembelian Mobil VolkSwagen Terbanyak**
  Top 3 Mobil Golf, Tiguan dan juga Polo merupakan mobil yang sering digunakan pada kumpulan data dari semua mobil di VW
    ![3 mobil terbanyak](https://raw.githubusercontent.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/main/Data%20Visualization/terbanyak.png?token=APMXFUJSMID2KDVWT6LA7ODBNFICA)

  **Jumlah Pembelian mobil tiap Tahun**
      - jika dilihat pada tahun 2019 dan 2020 merupakan tahun yang dimana jumlah pembeli mobil VW terbanyak
      ![peningkatan pembelian pertahun](https://user-images.githubusercontent.com/64582353/135224122-d465c32a-55d3-49ca-b48c-2f2bd6db63b8.png)

  **Pair Plot**
    disini saya menggunakan pairplot untuk melihat grafik mana yang memiliki kesamaan sehingga akan mempermudah untuk melakukan prediksi
      
   ![Pair Plot](https://raw.githubusercontent.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/main/pairplot.jpg?token=APMXFUJCYNUFJ34Y5ZO3EQDBNFLG2)
      
## Data Preparation
- **Sebelum datasetnya di latih atau training, dari model sebelumnya perlu melakukan pemisahan data antara data latih dan test lalu melakukan scaling untuk data categorical agar data dapat dilatih.
#### Train-Test Split
Proses splitting data atau pembagian dataset menjadi data latih *(train)* dan data uji *(test)* merupakan hal yang harus dilakukan sebelum melakukan pemodelan supervised. Hal ini karena data uji berperan sebagai data baru yang benar-benar belum pernah dilihat oleh model sebelumnya sehingga informasi yang terdapat pada data uji tidak mengotori informasi yang terdapat pada data latih, alasan lain mengapa menggunakan *train test split* karena untuk efisiensi dan tidak melakukan *data leakage* ketika melakukan scaling. 

#### Standardisasi
Data numerik yang terdapat di dataset perlu dilakukannya proses **Standardisasi** sehingga menghasilkan distribusi dengan nilai standar deviasi 1 dan mean 0. Hal tersebut dilakukan dengan tujuan untuk meningkatkan peforma algoritma machine learning dan membuatnya konvergen lebih cepat selain itu menghindari overfitting dan juga data imbalance.

Insight yang saya dapatkan saat melakukan EDA:

  - Mengetahui jumlah dari baris dan kolom yaitu sebanyak 15157 dan 9 kolom
  - Mengetahui semua kolom
  - dapat mengetahui Mean, Modus, dan nilai minimum
  - Mengetahui tipe data tiap masing-masing variabel
  - Memeriksa Data yang Unik
-  **Data Cleaning/Cleansing**. Pembersihan data (Data Cleaning/Cleansing) adalah proses memperbaiki atau menghapus data yang salah, rusak, salah format, duplikat, atau tidak lengkap dalam kumpulan data. Saat menggabungkan beberapa sumber data, ada banyak peluang untuk data diduplikasi atau diberi label yang salah.
    - disini kita melakukan data Cleaning untuk memeriksa apakah ada data yang kosong, lalu mendrop kolom yang tidak penting seperti mendrop Kolom Year, ini dilakukan karena kolom tahun akan berpengaruh dalam prediksi harga sehingga saya mendrop kolom tersebut dan menggantinya dengan feature engineering. 
-  **Data Visualization**. Visualisasi data adalah proses menerjemahkan kumpulan data besar dan metrik ke dalam bagan, grafik, dan visual lainnya. Dengan Visualisasi Data mempermudah membaca informasi yang banyak dan angka - angka melalui angka.                                                                                                        Insight yang saya dapatkan saat melakukan Visualisasi Data:
    

    - **Total Pembelian Mobil VolkSwagen Terbanyak**
      - Top 3 Mobil Golf, Tiguan dan juga Polo merupakan mobil yang sering digunakan pada kumpulan data dari semua mobil di VW
      ![3 mobil terbanyak](https://raw.githubusercontent.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/main/Data%20Visualization/terbanyak.png?token=APMXFUJSMID2KDVWT6LA7ODBNFICA)

    - **Jumlah Pembelian mobil tiap Tahun**
      - jika dilihat pada tahun 2019 dan 2020 merupakan tahun yang dimana jumlah pembeli mobil VW terbanyak
      ![peningkatan pembelian pertahun](https://user-images.githubusercontent.com/64582353/135224122-d465c32a-55d3-49ca-b48c-2f2bd6db63b8.png)

    - **Pair Plot**
      - disini saya menggunakan pairplot untuk melihat grafik mana yang memiliki kesamaan sehingga akan mempermudah untuk melakukan prediksi
      ![Pair Plot](https://user-images.githubusercontent.com/64582353/135224195-84e55ca4-276c-4d8e-917f-541849dbafb5.png)

- **Data Preprocessing**. Pra-pemrosesan data adalah proses mengubah data mentah menjadi format yang dapat dipahami. Ini juga merupakan langkah penting dalam penambangan data karena tidak dapat bekerja dengan data mentah. Disini saya melakukan preprocessing data agar data tersebut dapat diolah dan diproses dengan baik sehingga menghindari underfit atau overfit.
  - **Normalisasi**. Disini saya melakukan Normalisasi data agar data yang dilatih dan data yang di test akan mudah untuk mencocokan data karena memiliki nilai dan tipe data yang sama. untuk Normalisasi data saya menggunakan Stndar Scaler.
    - **StandardScaler**. StandardScaler menstandardisasi fitur dengan mengurangi mean dan kemudian menskalakan ke varians unit. Varians unit berarti membagi semua nilai dengan standar deviasi. alasan saya menggunakan StandardScaler karena tujuannya untuk memprediksi harga sehingga StandardScaler lebih baik dibandingkan MinMaxScaler yang nilainya 0 dan 1.
   - **Splitting Data Train dan Test**
      - disini saya membagi data menjadi data train dan test secara default yaitu 75% data train dan 25% data latih
   
## Modeling
- Bagian Modelling saya menggunakan Linear Regression, Decision Tree Regressor dan MLP Regressor. untuk modellingnya sendiri saya membandingkan ketiga model tersebut untuk mencari model mana yang lebih baik.
  - **Menggunakan Linear Model**. Pertama saya menggunakan Model Linear untuk mendapatkan akurasi terbaik.
  - ![Menggunakan Linear Model](https://user-images.githubusercontent.com/64582353/135480973-cd596327-7e06-408c-80a3-714020d0414e.png)
    - **Linear Regression(Linear)** . Saat menggunakan model Linear Regression mendapatkan akurasi yang baik yaitu sebesar 	0.929033
    - **Decision Tree Regressor(Linear)**. Saat menggunakan model Decision Tree Regressor mendapatkan akurasi yang sangat baik yaitu sebesar 0.952657
    - **MLP Regressor(Linear)**. Saat menggunakan model MLP Regressor mendapatkan akurasi yang baik yaitu sebesar 0.937266 
  - **Menggunakan Polynomial Model**. Kedua saya mencoba menggunakan Model Polynomial untuk mengetahui apakah dapat menambah akurasi atau tidak. dan hasilnya dibawah:
  - ![Menggunakan Polynomial Model](https://user-images.githubusercontent.com/64582353/135481143-de649c2f-79d3-4abb-8cf7-f784087a3d03.png)
    - **Linear Regression(Polynomial)** . Saat menggunakan model Linear Regression mendapatkan akurasi yang baik yaitu sebesar 0.929033
    - **Decision Tree Regressor(Polynomial)**. Saat menggunakan model Decision Tree Regressor mendapatkan akurasi yang sangat baik yaitu sebesar 0.953241
    - **MLP Regressor(Polynomial)**. Saat menggunakan model MLP Regressor mendapatkan akurasi yang baik yaitu sebesar 0.947744 
  - ![polynomial](https://user-images.githubusercontent.com/64582353/135227538-153e91ff-b31f-45f9-b10d-470a739f089c.png)

- **DecisionTreeRegressor Model Terbaik**. Dapat dilihat Decision Tree Regressor mendapatkan hasil prediksi terbaik dan juga terbaik di keduanya (Linear dan Polynomial) karena menghasilkan prediksi sebesar 0.953241 atau 95%. dengan begitu DecisionTreeRegressor menggunakan Metode Polynomial adalah solusi terbaik.
  
## Evaluation
- R-Squared (coefficient of determination).
 -Disini saya menggunakan Metric Evaluation yaitu R^2_score atau R-squared. R-Squared itu sendiri adalah skor terbaik yang mungkin adalah 1,0 dan bisa negatif (karena modelnya bisa sewenang-wenang lebih buruk). Sebuah model konstan yang selalu memprediksi nilai yang diharapkan dari y, mengabaikan fitur input, akan mendapatkan skor 0,0.
  - untuk persamaannya seperti ini
 
    - ![R2-SQUARED MACHINE LEARNING](https://user-images.githubusercontent.com/64582353/135482517-1f589eb6-d59f-4872-8d9d-eddd673c1124.png)
- **Kelebihannya**
  - dapat memprediksi hasil di masa depan atau pengujian hipotesis , berdasarkan informasi terkait lainnya.
  - memberikan ukuran seberapa baik hasil yang diamati direplikasi oleh model, berdasarkan proporsi variasi total hasil yang dijelaskan oleh model.
  - sangat cocok untuk metrics akurasi pada model Regresi.
- **Kekurangan**
  - tidak menunjukan apakah regresi yang benar digunakan
  - tidak dapat memberitahu apakah model tersebut overfit/underfit dan lainnya.
- **Code**
- untuk codenya yang diterapkan:
```javascript
def regression_model(model):
    regressor = model
    regressor.fit(X_train_transformed, Y_train)
    score = regressor.score(X_test_transformed, Y_test)
    return regressor, score
 ```![Uploading fuelType.png…]()

 - ada juga menggunakan library dari sklearn.metrics:
```javascript
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
```
- Dengan menggunakan R2_score dapat memberikan hasil yang baik sebsar 0.953241
