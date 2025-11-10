import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
# from lightgbm import LGBMRegressor # LGBM kullanacaksanız bunu açın
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os
import time
import json # Ayarları okumak için JSON kütüphanesini import et
from code.preprocess import preprocess_data

# --- KONTROL PANELİ ---
# Hangi ayar dosyasını kullanarak HIZLI bir eğitim yapmak istiyorsun?
PARAMS_DOSYASI = 'results/best_params_gbr.json' 
MODEL_ADI = "GradientBoosting_FINAL"
DATA_PATH = 'data/train.csv'
RESULTS_PATH = 'results/'
IMAGES_PATH = 'results/images/'
LOG_DOSYASI = 'results/experiment_log.csv'
# --------------------

# --- Klasörlerin var olduğundan emin ol ---
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)


# --- ANA KOD AKIŞI ---
print(f"Final Script Başlatıldı: {MODEL_ADI}")
start_time = time.time()
X, y = preprocess_data(DATA_PATH)

if X is not None:
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #--------------------------------------------------------------------------------
    # --- ADIM 3: FİNAL MODELİNİ EĞİTME ---
    #--------------------------------------------------------------------------------
    print("\nFinal modeli, bulunan en iyi ayarlarla eğitiliyor...")

    # --- Ayarları ELLE GİRMEK YERİNE DOSYADAN OKUMA ---
    try:
        with open(PARAMS_DOSYASI, 'r') as f:
            best_params = json.load(f)
        print(f"En iyi ayarlar '{PARAMS_DOSYASI}' dosyasından okundu: {best_params}")
    except FileNotFoundError:
        print(f"HATA: '{PARAMS_DOSYASI}' bulunamadı. Lütfen önce 'train_optimized.py' scriptini çalıştırın!")
        exit()
    # --- OKUMA BİTTİ ---
    
    # Hangi modeli okuduysak onu kullanalım (GBR veya LGBM)
    # Not: Hangi modeli optimize ettiysen, o modelin import'unun açık olduğundan emin ol
    if 'num_leaves' in best_params: # Bu LGBM'e ait bir ayar
        from lightgbm import LGBMRegressor
        final_model = LGBMRegressor(**best_params, random_state=42)
        MODEL_ADI = f"LGBM_FINAL_{best_params.get('n_estimators')}"
    else: # Bu GBR'ye ait
        final_model = GradientBoostingRegressor(**best_params, random_state=42)
        MODEL_ADI = f"GBR_FINAL_{best_params.get('n_estimators')}"

    # Modeli sadece BİR KEZ eğitiyoruz
    start_fit_time = time.time()
    final_model.fit(X_train, y_train)
    fit_time = time.time() - start_fit_time

    print(f"✓ Model {fit_time:.2f} saniyede eğitildi.") # Bu, 25 saniyeden ÇOK daha kısa sürecek

    # Tahminleri yap
    y_pred = final_model.predict(x_test) 
    print("✓ Tahminler yapıldı.")

    #--------------------------------------------------------------------------------
    # --- ADIM 4: DEĞERLENDİRME VE RAPORLAMA ---
    #--------------------------------------------------------------------------------
    
    print(f"\n--- Sayısal Değerlendirme (Final Model - {MODEL_ADI}) ---")
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared (R2) Skoru: {r2:.4f}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Ortalama Mutlak Hata (MAE): {mae:,.2f} $")

    # --- Sonuçları CSV Log Dosyasına Kaydetme ---
    # (Bu scripti her çalıştırdığımızda loga bir satır ekler)
    # ... (kodun geri kalanı train_optimized.py'deki ADIM 4 ve 5 ile aynı) ...
    # (Sadece `best_params_` yerine `best_params` değişkenini kullanır)
    print(f"\nSonuçlar '{LOG_DOSYASI}' dosyasına kaydediliyor...")
    log_entry = {
        'model_adi': MODEL_ADI,
        'mae_test_seti': mae,
        'r2_test_seti': r2,
        'en_iyi_ayarlar': str(best_params),
        'cv_mae_skoru': 'N/A (Final Script)',
        'tarih': pd.to_datetime('today').strftime('%Y-%m-%d %H:%M')
    }
    log_df = pd.DataFrame([log_entry])
    try:
        if not os.path.exists(LOG_DOSYASI):
            log_df.to_csv(LOG_DOSYASI, index=False)
        else:
            log_df.to_csv(LOG_DOSYASI, mode='a', header=False, index=False)
        print("✓ Sonuçlar kaydedildi.")
    except Exception as e:
        print(f"HATA: Log dosyası kaydedilemedi! {e}")

    # --- Grafikleri Kaydetme ---
    print("\n--- Görsel Değerlendirmeler Hazırlanıyor ---")
    
    # Grafik 1: Gerçek vs. Tahmin (Scatter Plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Gerçek Fiyatlar (y_test)')
    plt.ylabel('Tahmini Fiyatlar (y_pred)')
    plt.title(f'Gerçek vs. Tahmini Fiyatlar ({MODEL_ADI})')
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    graph_filename = os.path.join(IMAGES_PATH, f'py_scatter_{MODEL_ADI}.png')
    plt.savefig(graph_filename, dpi=300)
    print(f"Grafik kaydedildi: {graph_filename}")
    # plt.show() # Terminalde çalışırken bu satır kapatılabilir

    # Grafik 2: Hata Dağılımı (Residual Plot)
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Gerçek Fiyatlar (y_test)')
    plt.ylabel('Hata Payı (Gerçek - Tahmin)')
    plt.title(f'Hata Dağılım Grafiği ({MODEL_ADI})')
    graph_filename = os.path.join(IMAGES_PATH, f'py_residuals_{MODEL_ADI}.png')
    plt.savefig(graph_filename, dpi=300)
    print(f"Hata grafiği kaydedildi: {graph_filename}")
    # plt.show()
    
    # Grafik 3: Özellik Önem Düzeyi (Feature Importance Plot)
    try:
        importances = final_model.feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=feat_imp.head(20), y=feat_imp.head(20).index)
        plt.title(f'En Önemli 20 Özellik ({MODEL_ADI})')
        plt.xlabel('Önem Düzeyi')
        plt.ylabel('Özellikler')
        plt.tight_layout()
        graph_filename = os.path.join(IMAGES_PATH, f'py_feature_imp_{MODEL_ADI}.png')
        plt.savefig(graph_filename, dpi=300)
        print(f"Özellik önemi grafiği kaydedildi: {graph_filename}")
    except Exception as e:
        print(f"HATA: Özellik önemi grafiği oluşturulamadı! {e}")
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nFinal Deneyi ({MODEL_ADI}) {total_time:.2f} saniyede tamamlandı.")
else:
    print("Veri yüklenemediği için analiz durduruldu.")