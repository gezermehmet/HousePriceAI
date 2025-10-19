import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor


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

    df = pd.get_dummies(df)

    
    



    return df






df_raw = pd.read_csv('train.csv')
df_clean = preprocess_data(df_raw)

df_clean.info()



































# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# # DİKKAT: Modelimizi Sınıflandırıcı'dan Regresör'e çeviriyoruz
# from sklearn.ensemble import RandomForestRegressor 
# import seaborn as sns

# # --- ADIM 1: Veri Temizleme Fonksiyonu ---
# def preprocess_data(data):
#     """Veriyi yükler, temizler ve eksik değerleri doldurur."""
#     print("Veri ön işleme başlıyor...")
#     df = data.copy()

#     # Gereksiz ve çok boş sütunları at
#     cols_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'MasVnrType']
#     df = df.drop(columns=cols_to_drop)

#     # --- Eksik Veri Doldurma ---
    
#     # 1. Sayısal Sütunlar (Medyan ile)
#     numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
#     print("Sayısal sütunlardaki boşluklar dolduruluyor...")
#     for col in numerical_cols:
#         median_val = df[col].median()
#         df[col] = df[col].fillna(median_val)
#     print("✓ Sayısal boşluklar tamamlandı.")

#     # 2. Kategorik Sütunlar (Mod ile)
#     categorical_cols = df.select_dtypes(include=['object']).columns
#     print("Kategorik sütunlardaki boşluklar dolduruluyor...")
#     for col in categorical_cols:
#         mode_val = df[col].mode()[0]
#         df[col] = df[col].fillna(mode_val)
#     print("✓ Kategorik boşluklar tamamlandı.")
    
#     print("✓ Veri ön işleme bitti.")
#     return df

# # --- ANA KOD AKIŞI ---

# # 1. Veriyi Yükle
# df_raw = pd.read_csv('train.csv')

# # 2. Veriyi Temizle
# df_clean = preprocess_data(df_raw)

# # 3. Modeli Hazırla (One-Hot Encoding)
# # Tüm metin ('object') sütunlarını bulur ve 0/1'lere çevirir
# df_model_ready = pd.get_dummies(df_clean)

# # 4. İpuçları (X) ve Hedefi (y) Ayır
# X = df_model_ready.drop('SalePrice', axis=1) # Hedef hariç her şey
# y = df_model_ready['SalePrice']             # Sadece hedef

# # 5. Veriyi Sınav (Test) ve Eğitim Olarak Böl
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 6. Modeli Seç ve Oluştur
# # Bu bir REGRESYON problemidir (sayı tahmini), bu yüzden Regressor kullanırız
# model = RandomForestRegressor(n_estimators=10, random_state=42)

# # 7. Modeli Eğit
# print("\nModel eğitiliyor...")
# model.fit(X_train, y_train)
# print("✓ Model eğitildi.")

# # 8. Tahminleri Yap
# # Model, test setindeki evlerin fiyatlarını tahmin eder
# y_pred = model.predict(X_test) 

# print("\n✓ Tahminler yapıldı.")
# # 'y_pred' değişkeni artık modelin tahminlerini içeriyor.
# # 'y_test' değişkeni ise o evlerin gerçek fiyatlarını içeriyor.
# print("\n--- Tahmin ve Gerçek Değer Karşılaştırması (İlk 10 örnek) ---")
# for i in range(10):
#     print(f"Tahmin: {y_pred[i]:.0f}  |  Gerçek: {y_test.iloc[i]:.0f}")

# # Ortalama Hata (MAE) ve R-Kare gibi metrikler
# from sklearn.metrics import mean_absolute_error, r2_score

# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"\nOrtalama Mutlak Hata (MAE): {mae:.2f}")
# print(f"R^2 Skoru (1'e ne kadar yakınsa o kadar iyi): {r2:.4f}")