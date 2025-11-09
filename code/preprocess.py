import pandas as pd
import numpy as np

def preprocess_data(data_path):
    """
    Veriyi yükler, temizler, eksik değerleri doldurur ve 
    model için X (ipuçları) ve y (hedef) olarak ayırır.
    """
    print("Veri ön işleme başlıyor...")
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"HATA: '{data_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol et.")
        return None, None, None # X, y ve sütun listesi için None döndür
        
    df = data.copy()
    
    # 1. Gereksiz sütunları at
    cols_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'MasVnrType']
    df = df.drop(columns=cols_to_drop, errors='ignore') # 'errors=ignore' hata vermesini engeller
    
    # 2. Sayısal boşlukları doldur (Medyan ile)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        if col != 'SalePrice':
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        
    # 3. Kategorik boşlukları doldur (Mod ile)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        
    # 4. Metin sütunlarını sayısala çevir (One-Hot Encoding)
    print("Metin sütunları sayısala çevriliyor...")
    df = pd.get_dummies(df)
    
    # Hedef (y) ve Özellikleri (X) ayır
    try:
        X = df.drop('SalePrice', axis=1) 
        y = df['SalePrice']
        feature_names = X.columns.to_list() # Sütun isimlerini kaydet
        print("✓ Veri ön işleme bitti.")
        return X, y, feature_names
    except KeyError:
        print(f"HATA: 'SalePrice' sütunu veri setinde bulunamadı.")
        return None, None, None