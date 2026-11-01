# Catatan Belajar — Simple Linear Regression

> Versi ringkas untuk dibaca ulang. Untuk contoh kode, grafik, dan latihan yang lebih lengkap, buka notebook `.ipynb`.

## 1. Pahami idenya dulu

*Simple linear regression* adalah cara membuat **garis lurus** untuk memperkirakan sebuah nilai angka dari **satu variabel input**. Contoh gampangnya: menggunakan lama belajar (X) untuk memprediksi nilai ujian (y). Semua data contoh dalam modul ini fiktif.

$$\hat y=b_0+b_1x$$

- **Intercept ($b_0$)**: prediksi saat X = 0. Tidak selalu punya arti praktis.
- **Slope ($b_1$)**: perubahan prediksi y setiap X naik 1 unit.
- **$\hat y$**: nilai yang diprediksi oleh model; tidak harus sama dengan nilai asli.

**Intinya:** model melihat pola umum dalam data. Model tidak bisa menjamin hasil yang persis sama untuk setiap orang atau setiap kasus.

## 2. Menghitung garis secara manual

Data mini: (1,52), (2,55), (3,61), (4,63), (5,69).

$$b_1=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2}$$

$$b_0=\bar y-b_1\bar x$$

Rata-rata X = 3 dan rata-rata y = 60. Hasil pembilang = 42 dan penyebut = 10, sehingga $b_1=4{,}2$ dan $b_0=47{,}4$.

$$\boxed{\hat y=47{,}4+4{,}2x}$$

Jika X = 6, prediksinya $47{,}4+4{,}2(6)=72{,}6$. Karena 6 berada sedikit di luar rentang data contoh (1–5), hasilnya tetap perlu dibaca sebagai perkiraan.

**Jangan tertukar:** slope positif menunjukkan hubungan positif dalam data, **bukan** bukti bahwa X pasti menyebabkan perubahan y.

## 3. Residual itu apa?

$$e_i=y_i-\hat y_i$$

Residual = **nilai asli − prediksi**. Misalnya nilai asli 80 dan prediksi 75, residualnya +5. Berarti model memprediksi 5 poin terlalu rendah.

OLS (*ordinary least squares*) memilih garis dengan jumlah kuadrat residual terkecil:

$$\mathrm{SSE}=\sum(y_i-\hat y_i)^2$$

Kenapa pakai kuadrat? Supaya error positif dan negatif tidak saling menghapus, dan error yang besar mendapat penalti lebih kuat.

## 4. Pola kode Python

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Hal yang sering bikin bingung: `X` harus **2D**, dengan bentuk `(jumlah_sampel, jumlah_fitur)`. Walaupun cuma satu fitur, bentuknya tetap dua dimensi. Misalnya `np.array([1, 2, 3]).reshape(-1, 1)`.

Untuk satu input baru, bisa gunakan `model.predict([[6.0]])` jika model dilatih dengan array NumPy tanpa nama kolom. Kalau pelatihannya memakai DataFrame, gunakan DataFrame dengan nama kolom yang sama.

## 5. Pisahkan training dan test

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- **Train (80%)** dipakai untuk mencari garis regresi.
- **Test (20%)** dipakai untuk melihat performa pada data yang belum dipakai melatih model.
- Jangan memakai target dari data test saat membangun model atau membuat preprocessing yang harus dipelajari dari training.

## 6. Membaca metrik evaluasi

| Metrik | Rumus | Maknanya |
|---|---|---|
| MAE | $\frac{1}{n}\sum |y-\hat y|$ | Rata-rata besar error. Semakin kecil, semakin baik. |
| MSE | $\frac{1}{n}\sum(y-\hat y)^2$ | Lebih sensitif terhadap error besar. |
| RMSE | $\sqrt{\mathrm{MSE}}$ | Error dalam satuan yang sama dengan target. |
| $R^2$ | $1-\mathrm{SSE}/\mathrm{SST}$ | Performa relatif terhadap baseline rata-rata target evaluasi; bisa negatif pada test. |

$R^2=0{,}8$ **tidak berarti** 80% prediksinya benar. Sebaiknya cek juga nilai MAE/RMSE, grafik residual, dan bandingkan dengan baseline yang selalu menebak rata-rata target training.

## 7. Hal-hal yang perlu diwaspadai

- Data yang polanya melengkung mungkin tidak cocok dipaksa menjadi satu garis lurus.
- Outlier bisa menarik garis OLS cukup jauh.
- Ekstrapolasi, yaitu prediksi di luar rentang training, sering berisiko.
- Korelasi atau slope positif **bukan** bukti sebab-akibat.
- Untuk inferensi statistik, cek bentuk hubungan, independensi, rata-rata error bersyarat, dan variasi error. Normalitas residual lebih penting untuk inferensi pada sampel kecil daripada untuk sekadar menghasilkan prediksi.
- Model yang terlihat bagus pada training belum tentu bekerja sama bagusnya pada data baru.

## 8. Cek pemahaman

1. Kalau $\hat y=30+7x$, berapa prediksi saat X = 4? **58.**
2. Data (1,3), (2,5), (3,7) menghasilkan garis **$\hat y=1+2x$**.
3. Nilai asli 80, prediksi 75. Residualnya? **+5.**
4. MAE test A = 3 dan B = 7 pada data dan satuan yang sama. **A lebih baik menurut MAE**, tetapi belum tentu unggul dalam semua aspek.

**Ringkasnya:** simple linear regression mencari garis prediksi dari satu fitur. Pahami koefisiennya, periksa residual, dan jangan lupa mengevaluasi menggunakan data yang belum dipakai melatih model.

---

**Materi berikutnya:** multiple linear regression → polynomial regression → regularization → cross-validation.

**Bacaan tambahan:** [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) dan [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).
