import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
# from lightgbm import LGBMRegressor # LGBM optimize edeceksen bunu aç, GradientBoosting'i kapat
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import os
import json # Ayarları kaydetmek için JSON kütüphanesini import et
import time
from code.preprocess import preprocess_data

# --- KONTROL PANELİ ---
# Hangi modeli optimize edeceksin?
BASE_MODEL = GradientBoostingRegressor(random_state=42)
MODEL_ADI = "GradientBoosting_Optimized_v1"
PARAMS_DOSYASI = 'results/best_params_gbr.json'
PARAM_GRID = {
    'n_estimators': [50,100, 300, 500],
    'max_depth': [1, 3, 5, 10],
    'learning_rate': [0.1, 0.05, 0.15, 0.2]
}
# --------------------

LOG_DOSYASI = 'results/experiment_log.csv'
DATA_PATH = 'data/train.csv'
RESULTS_PATH = 'results/'
IMAGES_PATH = 'results/images/'

# --- Klasörlerin var olduğundan emin ol ---
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
    
# --- ANA KOD AKIŞI ---
print(f"Deney Başlatıldı: {MODEL_ADI}")
start_time = time.time()
X, y = preprocess_data(DATA_PATH)

if X is not None:
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #--------------------------------------------------------------------------------
    # --- ADIM 3: HİPERPARAMETRE OPTİ... (GridSearchCV) ---
    #--------------------------------------------------------------------------------
    print(f"\nModel ({MODEL_ADI}) için en iyi ayarlar aranıyor...")
    print(f"Denenecek Parametre Izgarası: {PARAM_GRID}")
    print(f"Aday kombinasyon sayısı: {np.prod([len(v) for v in PARAM_GRID.values()])}")
          
    grid_search = GridSearchCV(
        estimator=BASE_MODEL,
        param_grid=PARAM_GRID,
        cv=5, 
        scoring='neg_mean_absolute_error', # MAE'ye göre optimize et
        n_jobs=-1, # Tüm işlemcileri kullan
        verbose=1  # İlerlemeyi göster
    )
    grid_search.fit(X_train, y_train)

    print("\n--- Optimizasyon Sonuçları ---")
    print(f"En iyi ayarlar bulundu: {grid_search.best_params_}")
    
    # --- EN İYİ AYARLARI DOSYAYA KAYDET ---
    try:
        with open(PARAMS_DOSYASI, 'w') as f:
            json.dump(grid_search.best_params_, f, indent=4) 
        print(f"En iyi ayarlar şuraya kaydedildi: {PARAMS_DOSYASI}")
    except Exception as e:
        print(f"HATA: Parametre dosyası kaydedilemedi! {e}")
    # --- KAYIT BİTTİ ---

    best_mae_cv = -grid_search.best_score_
    print(f"En iyi 'Cross-Validation' MAE (Deneme Sınavı Ortalaması): {best_mae_cv:,.2f} $")

    # En iyi modeli al
    best_model = grid_search.best_estimator_

    # Tahminleri en iyi modelle yap
    print("✓ Tahminler en iyi modelle yapıldı (Final Sınavı).")
    y_pred = best_model.predict(x_test)
    
    #--------------------------------------------------------------------------------
    # --- ADIM 4: DEĞERLENDİRME VE RAPORLAMA ---
    #--------------------------------------------------------------------------------

    print(f"\n--- Sayısal Değerlendirme (Final Test Seti - {MODEL_ADI}) ---")
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared (R2) Skoru: {r2:.4f}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Ortalama Mutlak Hata (MAE): {mae:,.2f} $")

    # --- Sonuçları CSV Log Dosyasına Kaydetme ---
    print(f"\nSonuçlar '{LOG_DOSYASI}' dosyasına kaydediliyor...")
    log_entry = {
        'model_adi': MODEL_ADI,
        'mae_test_seti': mae,
        'r2_test_seti': r2,
        'en_iyi_ayarlar': str(grid_search.best_params_),
        'cv_mae_skoru': best_mae_cv,
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
        importances = best_model.feature_importances_
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
    print(f"\nDeney ({MODEL_ADI}) {total_time:.2f} saniyede tamamlandı.")
else:
    print("Veri yüklenemediği için analiz durduruldu.")