import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


def preprocess_data(data):

    df = data.copy()

    cols_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'MasVnrType']
    df = df.drop(columns=cols_to_drop)


    """    # --- Fill Missing Data ---  """ 

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
    
    """     ----------------------     """ 

    return df


df_raw = pd.read_csv('train.csv')
df_clean = preprocess_data(df_raw)


df_model_ready = pd.get_dummies(df_clean)
X = df_model_ready.drop('SalePrice', axis=1) 
y = df_model_ready['SalePrice']

X_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
print("\nModel eğitiliyor...")
model.fit(X_train, y_train)
print("✓ Model eğitildi.")

y_pred = model.predict(x_test) 
print("✓ Tahminler yapıldı.")


print("\n--- Görsel Değerlendirme ---")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Gerçek Fiyatlar (y_test)')
plt.ylabel('Tahmini Fiyatlar (y_pred)')
plt.title('Gerçek Fiyatlar vs. Tahmini Fiyatlar')


min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.show()


# R-squared (R2 Skoru): 1'e ne kadar yakınsa o kadar iyidir.
# %100 = Mükemmel, %0 = Rastgele tahminden farksız.
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2) Skoru: {r2:.4f} (Yani model, fiyatlardaki değişkenliğin yaklaşık %{r2*100:.0f}'ını açıklayabiliyor)")

# Mean Absolute Error (MAE): Ortalama kaç dolar saptığımızı gösterir.
mae = mean_absolute_error(y_test, y_pred)
print(f"Ortalama Mutlak Hata (MAE): {mae:,.2f} $ (Modelimiz, bir evin fiyatını ortalama bu kadar dolar farkla tahmin ediyor)")





"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# DİKKAT: Modelimizi Sınıflandırıcı'dan Regresör'e çeviriyoruz
from sklearn.ensemble import RandomForestRegressor 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


# --- ADIM 1: Veri Temizleme Fonksiyonu ---
def preprocess_data(data):
    #Veriyi yükler, temizler ve eksik değerleri doldurur.
    print("Veri ön işleme başlıyor...")
    df = data.copy()

    # Gereksiz ve çok boş sütunları at
    cols_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'MasVnrType']
    df = df.drop(columns=cols_to_drop)

    # --- Eksik Veri Doldurma ---
    
    # 1. Sayısal Sütunlar (Medyan ile)
    # SalePrice'ı da içerir, ancak train setinde SalePrice'ta boşluk olmadığı için sorun yaratmaz
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print("Sayısal sütunlardaki boşluklar dolduruluyor...")
    for col in numerical_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    print("✓ Sayısal boşluklar tamamlandı.")

    # 2. Kategorik Sütunlar (Mod ile)
    categorical_cols = df.select_dtypes(include=['object']).columns
    print("Kategorik sütunlardaki boşluklar dolduruluyor...")
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
    print("✓ Kategorik boşluklar tamamlandı.")
    
    print("✓ Veri ön işleme bitti.")
    return df

# --- ANA KOD AKIŞI ---

# 1. Veriyi Yükle
df_raw = pd.read_csv('train.csv')

# 2. Veriyi Temizle
df_clean = preprocess_data(df_raw)

# 3. Modeli Hazırla (One-Hot Encoding)
# Tüm metin ('object') sütunlarını bulur ve 0/1'lere çevirir
df_model_ready = pd.get_dummies(df_clean)

# 4. İpuçları (X) ve Hedefi (y) Ayır
X = df_model_ready.drop('SalePrice', axis=1) # Hedef hariç her şey
y = df_model_ready['SalePrice']             # Sadece hedef

# 5. Veriyi Sınav (Test) ve Eğitim Olarak Böl
# (Senin kodundaki 'x_test' değişken adını koruyorum)
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Modeli Seç ve Oluştur
# Bu bir REGRESYON problemidir (sayı tahmini), bu yüzden Regressor kullanırız
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 7. Modeli Eğit
print("\nModel eğitiliyor...")
model.fit(X_train, y_train)
print("✓ Model eğitildi.")

# 8. Tahminleri Yap
# Model, test setindeki evlerin fiyatlarını tahmin eder
y_pred = model.predict(x_test) 

print("✓ Tahminler yapıldı.")
# 'y_pred' değişkeni artık modelin tahminlerini içeriyor.
# 'y_test' değişkeni ise o evlerin gerçek fiyatlarını içeriyor.

# --- ADIM 9: SONUÇLARI DEĞERLENDİRME ---

# 1. GÖRSEL YÖNTEM: Dağılım Grafiği (Scatter Plot)
# Bu, "düzgünce görmek" için en iyi yoldur.
print("\n--- Görsel Değerlendirme ---")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Gerçek Fiyatlar (y_test)')
plt.ylabel('Tahmini Fiyatlar (y_pred)')
plt.title('Gerçek Fiyatlar vs. Tahmini Fiyatlar')

# Mükemmel bir tahminin olacağı 45 derecelik çizgiyi ekleyelim
# Eksenlerin minimum ve maksimum değerlerini bularak çizgiyi tam oturtalım
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.show()


# 2. SAYISAL YÖNTEM: Hata Metrikleri
# Modelimizin ne kadar iyi olduğunu sayılarla ifade edelim.
print("\n--- Sayısal Değerlendirme ---")

# R-squared (R2 Skoru): 1'e ne kadar yakınsa o kadar iyidir.
# %100 = Mükemmel, %0 = Rastgele tahminden farksız.
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2) Skoru: {r2:.4f} (Yani model, fiyatlardaki değişkenliğin yaklaşık %{r2*100:.0f}'ını açıklayabiliyor)")

# Mean Absolute Error (MAE): Ortalama kaç dolar saptığımızı gösterir.
mae = mean_absolute_error(y_test, y_pred)
print(f"Ortalama Mutlak Hata (MAE): {mae:,.2f} $ (Modelimiz, bir evin fiyatını ortalama bu kadar dolar farkla tahmin ediyor)")



"""













