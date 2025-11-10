import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os
import time

# --- KÜTÜPHANE İMPORTLARI ---
from sklearn.linear_model import LinearRegression # YENİ MODELİ İMPORT ET
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from preprocess import preprocess_data # DÜZELTİLMİŞ PREPROCESS'İ IMPORT ET

#--------------------------------------------------------------------------------
# --- KONTROL PANELİ 🎛️ ---
print(" Modeller ")
print("1: Gradient Boosting Regressor")
print("2: LGBM Regressor")
print("3: Random Forest Regressor")
print("4: Linear Regression")
secim = input("Lütfen Eğitmek İstediğiniz Modeli seçiniz (1-4): ")

if secim == '1':
    MODEL_TO_TEST = 'GradientBoosting'
elif secim == '2':
    MODEL_TO_TEST = 'LGBM'
elif secim == '3':
    MODEL_TO_TEST = 'RandomForest'
elif secim == '4':
    MODEL_TO_TEST = 'LinearRegression'
else:
    raise ValueError("Geçersiz seçim! Lütfen 1, 2, 3 veya 4 giriniz.")
#--------------------------------------------------------------------------------

# --- YOL TANIMLARI ---
DATA_PATH = 'data/train.csv'
RESULTS_PATH = 'results/'
IMAGES_PATH = 'results/images/'
LOG_DOSYASI = os.path.join(RESULTS_PATH, 'experiment_log.csv')

# --- Klasörlerin var olduğundan emin ol ---
if not os.path.exists(IMAGES_PATH): os.makedirs(IMAGES_PATH)
if not os.path.exists(RESULTS_PATH): os.makedirs(RESULTS_PATH)

# --- ANA KOD AKIŞI ---
print(f"Deney Başlatıldı: {MODEL_TO_TEST} (Fabrika Ayarları)")
start_time = time.time()

# 1. Veriyi Hazırla (preprocess.py'den import edildi - YENİ SCALING VERSİYONU)
X, y, feature_names = preprocess_data(DATA_PATH) # Artık X ve y'yi doğrudan döndürüyor

if X is not None:
    # 2. Veriyi Böl
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #--------------------------------------------------------------------------------
    # --- ADIM 3: MODEL EĞİTİMİ (FABRİKA AYARLARI) ---
    #--------------------------------------------------------------------------------
    print(f"\nModel ({MODEL_TO_TEST}) eğitiliyor...")

    if MODEL_TO_TEST == 'LinearRegression':
        model = LinearRegression(n_jobs=-1)
        MODEL_ADI = "LR_Default"
    elif MODEL_TO_TEST == 'GradientBoosting':
        model = GradientBoostingRegressor(random_state=42)
        MODEL_ADI = "GBR_Default"
    elif MODEL_TO_TEST == 'LGBM':
        model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1) 
        MODEL_ADI = "LGBM_Default"
    elif MODEL_TO_TEST == 'RandomForest':
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        MODEL_ADI = "RF_Default"
    else:
        # Bu satır aslında gereksiz çünkü yukarıda hata verirdik, ama güvenlik için kalsın
        raise ValueError("MODEL_TO_TEST değişkeni tanınmıyor!")

    model.fit(X_train, y_train)
    print("✓ Model eğitildi.")

    y_pred = model.predict(x_test) 
    print("✓ Tahminler yapıldı.")

    #--------------------------------------------------------------------------------
    # --- ADIM 4: DEĞERLENDİRME VE RAPORLAMA ---
    #--------------------------------------------------------------------------------

    print(f"\n--- Sayısal Değerlendirme ({MODEL_ADI}) ---")
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"R-squared (R2) Skoru: {r2:.4f}")
    print(f"Ortalama Mutlak Hata (MAE): {mae:,.2f} $")

    # --- Sonuçları CSV Log Dosyasına Kaydetme ---
    # ... (Loglama kodu aynı, buraya kopyalamaya gerek yok) ...
    print(f"\nSonuçlar '{LOG_DOSYASI}' dosyasına kaydediliyor...")
    log_entry = {
        'model_adi': MODEL_ADI, 'mae_test_seti': mae, 'r2_test_seti': r2,
        'en_iyi_ayarlar': 'N/A (Default)', 'cv_mae_skoru': 'N/A (Default)',
        'tarih': pd.to_datetime('today').strftime('%Y-%m-%d %H:%M')
    }
    log_df = pd.DataFrame([log_entry])
    try:
        if not os.path.exists(LOG_DOSYASI): log_df.to_csv(LOG_DOSYASI, index=False)
        else: log_df.to_csv(LOG_DOSYASI, mode='a', header=False, index=False)
        print("✓ Sonuçlar kaydedildi.")
    except Exception as e: print(f"HATA: Log dosyası kaydedilemedi! {e}")

    # --- Grafikleri Kaydetme ---
    print("\n--- Görsel Değerlendirmeler Hazırlanıyor ---")
    
    # Grafik 1: Gerçek vs. Tahmin
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Gerçek Fiyatlar'); plt.ylabel('Tahmini Fiyatlar')
    plt.title(f'Gerçek vs. Tahmin ({MODEL_ADI})')
    min_val = min(min(y_test), min(y_pred)); max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    graph_filename = os.path.join(IMAGES_PATH, f'py_scatter_{MODEL_ADI}.png')
    plt.savefig(graph_filename, dpi=300)
    print(f"Grafik kaydedildi: {graph_filename}")

    # Grafik 2: Özellik Önem Düzeyi
    plt.figure(figsize=(10, 8))
    MODEL_ADI_BASLIK = f'En Önemli 20 Özellik ({MODEL_ADI})'
    graph_filename = os.path.join(IMAGES_PATH, f'py_feature_imp_{MODEL_ADI}.png')
    
    try:
        if hasattr(model, 'feature_importances_'): # Ağaç modelleri (RF, GBR, LGBM) için
            print("Özellik önemi 'feature_importances_' ile hesaplanıyor...")
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
        elif hasattr(model, 'coef_'): # LinearRegression için
            print("Özellik önemi 'coef_' (katsayılar) ile hesaplanıyor...")
            # Katsayıların mutlak değerini (büyüklüğünü) alırız
            importances = np.abs(model.coef_)
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
        else:
            raise AttributeError("Model 'feature_importances_' veya 'coef_' desteklemiyor.")

        sns.barplot(x=feat_imp.head(20), y=feat_imp.head(20).index)
        plt.title(MODEL_ADI_BASLIK)
        plt.xlabel('Önem Düzeyi / Katsayı Büyüklüğü'); plt.ylabel('Özellikler')
        plt.tight_layout()
        plt.savefig(graph_filename, dpi=300)
        print(f"Özellik önemi grafiği kaydedildi: {graph_filename}")
    
    except Exception as e: 
        print(f"HATA: Özellik önemi grafiği oluşturulamadı! {e}")
        # Hata olsa bile grafiği kapat ki sonraki kod çalışsın
        plt.close()
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nDeney ({MODEL_ADI}) {total_time:.2f} saniyede tamamlandı.")
else:
    print("Veri yüklenemediği için analiz durduruldu.")