import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 

def preprocess_data(data_path):
    """
    Veriyi yükler, temizler, eksik değerleri doldurur, 
    SADECE İPUÇLARINI ÖLÇEKLER ve modeli hazırlar.
    """
    print("Veri ön işleme başlıyor...")
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"HATA: '{data_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol et.")
        return None, None, None
        
    df = data.copy()
    
    # 1. Gereksiz sütunları at
    cols_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'MasVnrType']
    df = df.drop(columns=cols_to_drop, errors='ignore') 
    
    # --- DÜZELTME BURADA BAŞLIYOR ---
    # Hedef (y) ve Özellikleri (X) DAHA ERKEN ayır
    try:
        y = df['SalePrice']
        X = df.drop('SalePrice', axis=1) 
    except KeyError:
        print(f"HATA: 'SalePrice' sütunu veri setinde bulunamadı.")
        return None, None, None
        
    # 2. Sayısal boşlukları doldur (Medyan ile)
    # Artık X üzerinde çalışıyoruz, y'ye dokunmuyoruz
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    print("Sayısal sütunlardaki boşluklar dolduruluyor...")
    for col in numerical_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        
    # 3. Kategorik boşlukları doldur (Mod ile)
    categorical_cols = X.select_dtypes(include=['object']).columns
    print("Kategorik sütunlardaki boşluklar dolduruluyor...")
    for col in categorical_cols:
        mode_val = X[col].mode()[0]
        X[col] = X[col].fillna(mode_val)
        
    # 4. YENİ ADIM: Sayısal Verileri Ölçeklendirme
    print("Sayısal veriler ölçekleniyor (StandardScaler)...")
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # 5. Metin sütunlarını sayısala çevir (One-Hot Encoding)
    print("Metin sütunları sayısala çevriliyor (One-Hot Encoding)...")
    X = pd.get_dummies(X)
    
    feature_names = X.columns.to_list()
    print("✓ Veri ön işleme bitti.")
    return X, y, feature_names # Artık y (SalePrice) ASLA ölçeklenmedi