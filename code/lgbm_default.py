import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor # Sadece LGBM'i import ediyoruz
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os


# --- ADIM 1: Veri Temizleme Fonksiyonu ---
def preprocess_data(data):
    """Veriyi yükler, temizler ve eksik değerleri doldurur."""
    print("Veri ön işleme başlıyor...")
    df = data.copy()
    cols_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'MasVnrType']
    df = df.drop(columns=cols_to_drop)
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        
    print("Metin sütunları sayısala çevriliyor...")
    df = pd.get_dummies(df)
    print("✓ Veri ön işleme bitti.")
    return df

# --- ANA KOD AKIŞI ---

# 1. Veriyi Yükle
df_raw = pd.read_csv('data/train.csv') # Ana dizinden bir üste çıkıp data'ya gir

# 2. Veriyi Hazırla
df_model_ready = preprocess_data(df_raw)

# 3. İpuçları (X) ve Hedefi (y) Ayır
X = df_model_ready.drop('SalePrice', axis=1) 
y = df_model_ready['SalePrice']

# 4. Veriyi Böl
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#--------------------------------------------------------------------------------
# --- ADIM 5: MODEL EĞİTİMİ (LGBM - FABRİKA AYARLARI) ---
#--------------------------------------------------------------------------------
print("\nModel (Default - LGBM) eğitiliyor...")

# Sadece 'random_state' veriyoruz, diğer her şey varsayılan (default)
model = LGBMRegressor(random_state=42)

model.fit(X_train, y_train)
print("✓ Model eğitildi.")

y_pred = model.predict(x_test) 
print("✓ Tahminler yapıldı.")

#--------------------------------------------------------------------------------
# --- ADIM 6: DEĞERLENDİRME ---
#--------------------------------------------------------------------------------

print("\n--- Sayısal Değerlendirme (LGBM - Default) ---")
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2) Skoru: {r2:.4f}")

mae = mean_absolute_error(y_test, y_pred)
print(f"Ortalama Mutlak Hata (MAE): {mae:,.2f} $")

print("\n--- Görsel Değerlendirme ---")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Gerçek Fiyatlar (y_test)')
plt.ylabel('Tahmini Fiyatlar (y_pred)')
plt.title('Gerçek Fiyatlar vs. Tahmini Fiyatlar (LGBM - Default)')

min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# Grafiği kaydet
graph_filename = os.path.join('results/images/', 'py_lgbm_default_scatter.png')
plt.savefig(graph_filename, dpi=300)
print(f"Grafik şuraya kaydedildi: {graph_filename}")
# plt.show() 

# --- Detaylı Tahmin İncelemesi ---
print("\n--- Tahmin Sonuçlarının Detaylı İncelenmesi (LGBM - Default) ---")
y_test_reset = y_test.reset_index(drop=True)
results_df = pd.DataFrame({'Gerçek Fiyat': y_test_reset, 'Tahmini Fiyat': y_pred})
results_df['Fark (Gerçek - Tahmin)'] = results_df['Gerçek Fiyat'] - results_df['Tahmini Fiyat']
results_df['Mutlak Hata'] = results_df['Fark (Gerçek - Tahmin)'].abs()
pd.options.display.float_format = '{:,.0f}'.format

print("\n--- EN İYİ 5 TAHMİN (LGBM - Default) ---")
print(results_df.sort_values(by='Mutlak Hata', ascending=True).head(5))

print("\n--- EN KÖTÜ 5 TAHMİN (LGBM - Default) ---")
print(results_df.sort_values(by='Mutlak Hata', ascending=False).head(5))

print("\nDeney 1 (LGBM - Default) tamamlandı.")