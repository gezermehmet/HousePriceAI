import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os
import time
import json 

# --- KÜTÜPHANE İMPORTLARI ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor # YENİ IMPORT
from preprocess import preprocess_data 

#--------------------------------------------------------------------------------
# --- KONTROL PANELİ 🎛️ ---
print(" Kayıtlı Ayarlarla Çalıştırılacak Modeller ")
print("1: Gradient Boosting Regressor")
print("2: LGBM Regressor")
print("3: Random Forest Regressor")
print("4: XGBoost Regressor") # YENİ SEÇENEK
secim = input("Lütfen Modeli seçiniz (1-4): ")

if secim == '1':
    MODEL_TYPE_TO_LOAD = 'GradientBoosting'
elif secim == '2':
    MODEL_TYPE_TO_LOAD = 'LGBM'
elif secim == '3':
    MODEL_TYPE_TO_LOAD = 'RandomForest'
elif secim == '4':
    MODEL_TYPE_TO_LOAD = 'XGBoost'
else:
    raise ValueError("Geçersiz seçim!")
#--------------------------------------------------------------------------------

# --- YOL TANIMLARI ---
DATA_PATH = 'data/train.csv'
RESULTS_PATH = 'results/'
IMAGES_PATH = 'results/images/'
LOG_DOSYASI = os.path.join(RESULTS_PATH, 'experiment_log.csv')

model_param_files = {
    'GradientBoosting': os.path.join(RESULTS_PATH, 'best_params_gbr.json'),
    'LGBM': os.path.join(RESULTS_PATH, 'best_params_lgbm.json'),
    'RandomForest': os.path.join(RESULTS_PATH, 'best_params_rf.json'),
    'XGBoost': os.path.join(RESULTS_PATH, 'best_params_xgb.json') # YENİ DOSYA YOLU
}

PARAMS_DOSYASI = model_param_files[MODEL_TYPE_TO_LOAD]
MODEL_ADI = f"{MODEL_TYPE_TO_LOAD}_Final"

if not os.path.exists(IMAGES_PATH): os.makedirs(IMAGES_PATH)
if not os.path.exists(RESULTS_PATH): os.makedirs(RESULTS_PATH)
    
# --- ANA KOD AKIŞI ---
print(f"Final Script Başlatıldı: {MODEL_ADI}")
start_time = time.time()

X, y, feature_names = preprocess_data(DATA_PATH)

if X is not None:
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- ADIM 3: FİNAL MODELİNİ EĞİTME ---
    print("\nFinal modeli, bulunan en iyi ayarlarla eğitiliyor...")

    try:
        with open(PARAMS_DOSYASI, 'r') as f:
            best_params = json.load(f)
        print(f"En iyi ayarlar okundu: {best_params}")
    except FileNotFoundError:
        print(f"HATA: '{PARAMS_DOSYASI}' bulunamadı. Lütfen önce o model için 'train_optimized.py' çalıştırın!")
        exit()

    # Model Seçimi
    if MODEL_TYPE_TO_LOAD == 'GradientBoosting':
        final_model = GradientBoostingRegressor(**best_params, random_state=42)
    elif MODEL_TYPE_TO_LOAD == 'LGBM':
        final_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
    elif MODEL_TYPE_TO_LOAD == 'RandomForest':
        final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    elif MODEL_TYPE_TO_LOAD == 'XGBoost': # YENİ MODEL
        final_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)

    final_model.fit(X_train, y_train)
    
    y_pred = final_model.predict(x_test) 
    print("✓ Tahminler yapıldı.")

    # --- ADIM 4: DEĞERLENDİRME ---
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n--- Sayısal Değerlendirme ({MODEL_ADI}) ---")
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"R-squared (R2) Skoru: {r2:.4f}")
    print(f"Ortalama Mutlak Hata (MAE): {mae:,.2f} $")

    # Log Kaydetme
    print(f"\nSonuçlar '{LOG_DOSYASI}' dosyasına kaydediliyor...")
    log_entry = {
        'model_adi': MODEL_ADI,
        'mae_test_seti': mae,
        'r2_test_seti': r2,
        'en_iyi_ayarlar': str(best_params),
        'cv_mae_skoru': 'N/A (Final Script)',
        'toplam_sure_saniye': round(total_time, 2),
        'tarih': pd.to_datetime('today').strftime('%Y-%m-%d %H:%M')
    }
    log_df = pd.DataFrame([log_entry])
    try:
        if not os.path.exists(LOG_DOSYASI): log_df.to_csv(LOG_DOSYASI, index=False)
        else: log_df.to_csv(LOG_DOSYASI, mode='a', header=False, index=False)
        print("✓ Sonuçlar kaydedildi.")
    except Exception as e: print(f"HATA: Log dosyası kaydedilemedi! {e}")

    # Grafik Kaydetme
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Gerçek Fiyatlar'); plt.ylabel('Tahmini Fiyatlar')
    plt.title(f'Gerçek vs. Tahmin ({MODEL_ADI})')
    min_val = min(min(y_test), min(y_pred)); max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    graph_filename = os.path.join(IMAGES_PATH, f'py_scatter_{MODEL_ADI}.png')
    plt.savefig(graph_filename, dpi=300)
    print(f"Grafik kaydedildi: {graph_filename}")

    # ... (Grafik 2: Residual Plot) ...
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Gerçek Fiyatlar'); plt.ylabel('Hata Payı (Gerçek - Tahmin)')
    plt.title(f'Hata Dağılım Grafiği ({MODEL_ADI})')
    graph_filename = os.path.join(IMAGES_PATH, f'py_residuals_{MODEL_ADI}.png')
    plt.savefig(graph_filename, dpi=300)
    print(f"Hata grafiği kaydedildi: {graph_filename}")

    # Grafik 3: Özellik Önem Düzeyi (Final model için)
    try:
        plt.figure(figsize=(10, 8))
        if hasattr(final_model, 'feature_importances_'): 
            importances = final_model.feature_importances_
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            sns.barplot(x=feat_imp.head(20), y=feat_imp.head(20).index)
            plt.title(f'En Önemli 20 Özellik ({MODEL_ADI})')
            graph_filename = os.path.join(IMAGES_PATH, f'py_feature_imp_{MODEL_ADI}.png')
            plt.savefig(graph_filename, dpi=300)
            print(f"Özellik önemi grafiği kaydedildi: {graph_filename}")
    except Exception as e: 
        pass # Özellik önemi her modelde olmayabilir veya hata verebilir, sorun değil

    print(f"\nFinal Deneyi ({MODEL_ADI}) {total_time:.2f} saniyede tamamlandı.")
else:
    print("Veri yüklenemediği için analiz durduruldu.")
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os
import time
import json # Ayarları okumak için

# --- KÜTÜPHANE İMPORTLARI ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from preprocess import preprocess_data 

#--------------------------------------------------------------------------------
# --- KONTROL PANELİ 🎛️ ---
print(" Modeller ")
print("1: Gradient Boosting Regressor")
print("2: LGBM Regressor")
print("3: Random Forest Regressor")
secim = input("Lütfen Eğitmek İstediğiniz Modeli seçiniz (1-3): ")

if secim == '1':
    MODEL_TYPE_TO_LOAD = 'GradientBoosting'
elif secim == '2':
    MODEL_TYPE_TO_LOAD = 'LGBM'
elif secim == '3':
    MODEL_TYPE_TO_LOAD = 'RandomForest'
else:
    raise ValueError("Geçersiz seçim! Lütfen 1, 2, veya 3 giriniz.")
#--------------------------------------------------------------------------------

# --- YOL TANIMLARI ---
DATA_PATH = 'data/train.csv'
RESULTS_PATH = 'results/'
IMAGES_PATH = 'results/images/'
LOG_DOSYASI = os.path.join(RESULTS_PATH, 'experiment_log.csv')

# --- Model Ayar Dosyalarının Yolları ---
model_param_files = {
    'GradientBoosting': os.path.join(RESULTS_PATH, 'best_params_gbr.json'),
    'LGBM': os.path.join(RESULTS_PATH, 'best_params_lgbm.json'),
    'RandomForest': os.path.join(RESULTS_PATH, 'best_params_rf.json')
}

# --- Seçilen modelin ayar dosyasını al ---
if MODEL_TYPE_TO_LOAD not in model_param_files:
    raise ValueError(f"MODEL_TYPE_TO_LOAD '{MODEL_TYPE_TO_LOAD}' olarak ayarlandı, ancak 'model_param_files' içinde tanınmıyor.")

PARAMS_DOSYASI = model_param_files[MODEL_TYPE_TO_LOAD]
MODEL_ADI = f"{MODEL_TYPE_TO_LOAD}_Final"

# --- Klasörlerin var olduğundan emin ol ---
if not os.path.exists(IMAGES_PATH): os.makedirs(IMAGES_PATH)
if not os.path.exists(RESULTS_PATH): os.makedirs(RESULTS_PATH)
    
# --- ANA KOD AKIŞI ---
print(f"Final Script Başlatıldı: {MODEL_ADI}")
start_time = time.time()

# 1. Veriyi Hazırla
X, y, feature_names = preprocess_data(DATA_PATH)

if X is not None:
    # 2. Veriyi Böl
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
        print(f"HATA: '{PARAMS_DOSYASI}' bulunamadı. Lütfen önce o model için 'train_optimized.py' scriptini çalıştırın!")
        exit()
    # --- OKUMA BİTTİ ---

    # Hangi modeli yükleyeceğimizi seçelim
    if MODEL_TYPE_TO_LOAD == 'GradientBoosting':
        final_model = GradientBoostingRegressor(**best_params, random_state=42)
    elif MODEL_TYPE_TO_LOAD == 'LGBM':
        final_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
    elif MODEL_TYPE_TO_LOAD == 'RandomForest':
        final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)

    # Modeli sadece BİR KEZ eğitiyoruz (HIZLI KISIM)
    start_fit_time = time.time()
    final_model.fit(X_train, y_train)
    fit_time = time.time() - start_fit_time
    print(f"✓ Model {fit_time:.2f} saniyede eğitildi.")

    y_pred = final_model.predict(x_test) 
    print("✓ Tahminler yapıldı.")

    #--------------------------------------------------------------------------------
    # --- ADIM 4: DEĞERLENDİRME VE RAPORLAMA ---
    #--------------------------------------------------------------------------------

    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n--- Sayısal Değerlendirme (Final Model - {MODEL_ADI}) ---")
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"R-squared (R2) Skoru: {r2:.4f}")
    print(f"Ortalama Mutlak Hata (MAE): {mae:,.2f} $")

    # --- Sonuçları CSV Log Dosyasına Kaydetme ---
    print(f"\nSonuçlar '{LOG_DOSYASI}' dosyasına kaydediliyor...")
    log_entry = {
        'model_adi': MODEL_ADI,
        'mae_test_seti': mae,
        'r2_test_seti': r2,
        'en_iyi_ayarlar': str(best_params),
        'cv_mae_skoru': 'N/A (Final Script)',
        'toplam_sure_saniye': round(total_time, 2),
        'tarih': pd.to_datetime('today').strftime('%Y-%m-%d %H:%M')
    }
    log_df = pd.DataFrame([log_entry])
    try:
        if not os.path.exists(LOG_DOSYASI): log_df.to_csv(LOG_DOSYASI, index=False)
        else: log_df.to_csv(LOG_DOSYASI, mode='a', header=False, index=False)
        print("✓ Sonuçlar kaydedildi.")
    except Exception as e: print(f"HATA: Log dosyası kaydedilemedi! {e}")

    # --- Grafikleri Kaydetme (Scatter, Residuals, Feature Importance) ---
    print("\n--- Görsel Değerlendirmeler Hazırlanıyor ---")
    
    # ... (Grafik 1: Scatter Plot) ...
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Gerçek Fiyatlar'); plt.ylabel('Tahmini Fiyatlar')
    plt.title(f'Gerçek vs. Tahmin ({MODEL_ADI})')
    min_val = min(min(y_test), min(y_pred)); max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    graph_filename = os.path.join(IMAGES_PATH, f'py_scatter_{MODEL_ADI}.png')
    plt.savefig(graph_filename, dpi=300)
    print(f"Grafik kaydedildi: {graph_filename}")
    
    # ... (Grafik 2: Residual Plot) ...
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Gerçek Fiyatlar'); plt.ylabel('Hata Payı (Gerçek - Tahmin)')
    plt.title(f'Hata Dağılım Grafiği ({MODEL_ADI})')
    graph_filename = os.path.join(IMAGES_PATH, f'py_residuals_{MODEL_ADI}.png')
    plt.savefig(graph_filename, dpi=300)
    print(f"Hata grafiği kaydedildi: {graph_filename}")

    # ... (Grafik 3: Feature Importance Plot) ...
    try:
        importances = final_model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(10, 8))
        sns.barplot(x=feat_imp.head(20), y=feat_imp.head(20).index)
        plt.title(f'En Önemli 20 Özellik ({MODEL_ADI})')
        plt.xlabel('Önem Düzeyi'); plt.ylabel('Özellikler')
        plt.tight_layout()
        graph_filename = os.path.join(IMAGES_PATH, f'py_feature_imp_{MODEL_ADI}.png')
        plt.savefig(graph_filename, dpi=300)
        print(f"Özellik önemi grafiği kaydedildi: {graph_filename}")
    except Exception as e: print(f"HATA: Özellik önemi grafiği oluşturulamadı! {e}")
        
    
    print(f"\nFinal Deneyi ({MODEL_ADI}) {total_time:.2f} saniyede tamamlandı.")
else:
    print("Veri yüklenemediği için analiz durduruldu.")"""