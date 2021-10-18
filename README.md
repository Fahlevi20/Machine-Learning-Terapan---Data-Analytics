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

![Transmission](https://raw.githubusercontent.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/main/Data%20Visualization/Transmission.jpg?token=APMXFUOYLUOCS6TFANBWXUTBNT6LG)

|count|unique|top|freq|
|:---:|:---:|:---:|:---:|
|15157|4|Petrol|8553|

![fuelType](https://github.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/blob/main/Data%20Visualization/Barplot%20fuelType.jpg)

 **Total Pembelian Mobil VolkSwagen Terbanyak**
  Top 3 Mobil Golf, Tiguan dan juga Polo merupakan mobil yang sering digunakan pada kumpulan data dari semua mobil di VW
    ![3 mobil terbanyak](https://github.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/blob/main/Data%20Visualization/terbanyak.png)

  **Jumlah Pembelian mobil tiap Tahun**
      - jika dilihat pada tahun 2019 dan 2020 merupakan tahun yang dimana jumlah pembeli mobil VW terbanyak
      ![peningkatan pembelian pertahun](https://github.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/blob/main/Data%20Visualization/pertahun.png)

  **Pair Plot**
    disini saya menggunakan pairplot untuk melihat grafik mana yang memiliki kesamaan sehingga akan mempermudah untuk melakukan prediksi
      
   ![Pair Plot](https://github.com/Fahlevi20/Machine-Learning-Terapan---Data-Analytics/blob/main/Data%20Visualization/pairplot.jpg)
      
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
 
 ```python
scaler=StandardScaler()
df_scaler=scaler.fit_transform(df_new)
df_scaler=pd.DataFrame(df_scaler,columns=df_new.columns)
print(df_scaler.shape)

X=df_scaler.drop(columns=['price'])
y=df_scaler['price']
``` 
  - **Normalisasi**. Disini saya melakukan Normalisasi data agar data yang dilatih dan data yang di test akan mudah untuk mencocokan data karena memiliki nilai dan tipe data yang sama. untuk Normalisasi data saya menggunakan Stndar Scaler.
    - **StandardScaler**. StandardScaler menstandardisasi fitur dengan mengurangi mean dan kemudian menskalakan ke varians unit. Varians unit berarti membagi semua nilai dengan standar deviasi. alasan saya menggunakan StandardScaler karena tujuannya untuk memprediksi harga sehingga StandardScaler lebih baik dibandingkan MinMaxScaler yang nilainya 0 dan 1.
   - **Splitting Data Train dan Test**
      - disini saya membagi data menjadi data train dan test secara default yaitu 75% data train dan 25% data latih
   
## Modeling
- Pada Proyek yang dibuat, digunakan model algoritma *Machine Learning* yaitu **Linear Regression**,**Decision Tree Regressor**, dan **Multi Layer Perceptron Regressor**. Model tersebut dipilih dikarenakan permasalahan dari model *Machine Learning* yang dibuat adalah permasalahan regresi. hasil dari model yang dipilih akan dibandingkan berdasarkan label yang telah terpilih sebelmunya yaitu *price*. Berikut adalah potongan kode dari model tersebut.
 ```python
 # Dalam Bentuk Regresi
X_train,X_test,Y_train,Y_test=train_test_split(X,y)
def regression_model(model):
    regressor = model
    regressor.fit(X_train_transformed, Y_train)
    score = regressor.score(X_test_transformed, Y_test)
    return regressor, score 
```
- Lalu melihat hasil model regresi
```python
model_performance = pd.DataFrame(columns = ["Features", "Model", "Score"])

models_to_evaluate = [LinearRegression(), RandomForestRegressor(), MLPRegressor()]

for model in models_to_evaluate:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({"Features": "Linear","Model": model, "Score": score}, ignore_index=True)

model_performance
```
lalu sekarang melihat model dalam bentuk Polynomial untuk mencari nilai K
```python
poly = PolynomialFeatures()
X_train_transformed_poly = poly.fit_transform(X_train)
X_test_transformed_poly = poly.transform(X_test)

print(X_train_transformed_poly.shape)

no_of_features = []
r_squared = []

for k in range(10, 277, 5):
    selector = SelectKBest(f_regression, k = k)
    X_train_transformed = selector.fit_transform(X_train_transformed_poly, Y_train)
    regressor = LinearRegression()
    regressor.fit(X_train_transformed, Y_train)
    no_of_features.append(k)
    r_squared.append(regressor.score(X_train_transformed, Y_train))
    
sns.lineplot(x = no_of_features, y = r_squared)
```
|No|Features|	Model|	Score|
|:---:|:---:|:---:|:---:|
|0|	Linear	|LinearRegression(copy_X=True, fit_intercept=Tr... |	0.929033|
|1|	Linear	|(DecisionTreeRegressor(ccp_alpha=0.0, criterio... |	0.952657|
|2|	Linear	|MLPRegressor(activation='relu', alpha=0.0001, ... |	0.937266|
```python
selector = SelectKBest(f_regression, k = 110)
X_train_transformed = selector.fit_transform(X_train_transformed_poly, Y_train)
X_test_transformed = selector.transform(X_test_transformed_poly)
```
lalu hasil model menggunakan polynomial
```python
models_to_evaluate = [LinearRegression(), RandomForestRegressor(), MLPRegressor()]

for model in models_to_evaluate:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({"Features": "Polynomial","Model": model, "Score": score}, ignore_index=True)

model_performance
```
|No|Features|	Model|	Score|
|:---:|:---:|:---:|:---:|
|0|	Linear	|LinearRegression(copy_X=True, fit_intercept=Tr... |	0.929033|
|1|	Linear	|(DecisionTreeRegressor(ccp_alpha=0.0, criterio... |	0.952657|
|2|	Linear	|MLPRegressor(activation='relu', alpha=0.0001, ... |	0.937266|
|3|	Polynomial|	LinearRegression(copy_X=True, fit_intercept=Tr... |	0.929033|
|4|	Polynomial|	(DecisionTreeRegressor(ccp_alpha=0.0, criterio...	| 0.953241|
|5|	Polynomial|	MLPRegressor(activation='relu', alpha=0.0001, ...	| 0.947744|

Dari Tabel dapat dilihat bahwa nilai *RF* lebih mendekati dengan nilai aslinya, sehingga model yang paling cocok adalah *Decision Tree Regressior* menggunakan Polynomial.
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
```python
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
