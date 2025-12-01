import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
import os
import json 
import time

# --- KÜTÜPHANE İMPORTLARI ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor # YENİ IMPORT
from preprocess import preprocess_data

#--------------------------------------------------------------------------------
# --- KONTROL PANELİ 🎛️ ---
print(" Optimize Edilecek Modeller ")
print("1: Gradient Boosting Regressor")
print("2: LGBM Regressor")
print("3: Random Forest Regressor")
print("4: XGBoost Regressor") # YENİ SEÇENEK
secim = input("Lütfen Optimize Etmek İstediğiniz Modeli seçiniz (1-4): ")

if secim == '1':
    MODEL_TO_OPTIMIZE = 'GradientBoosting'
elif secim == '2':
    MODEL_TO_OPTIMIZE = 'LGBM'
elif secim == '3':
    MODEL_TO_OPTIMIZE = 'RandomForest'
elif secim == '4':
    MODEL_TO_OPTIMIZE = 'XGBoost'
else:
    raise ValueError("Geçersiz seçim! Lütfen 1-4 arasında bir değer giriniz.")
#--------------------------------------------------------------------------------

# --- YOL TANIMLARI ---
DATA_PATH = 'data/train.csv'
RESULTS_PATH = 'results/'
IMAGES_PATH = 'results/images/'
LOG_DOSYASI = os.path.join(RESULTS_PATH, 'experiment_log.csv')

if not os.path.exists(IMAGES_PATH): os.makedirs(IMAGES_PATH)
if not os.path.exists(RESULTS_PATH): os.makedirs(RESULTS_PATH)

# --- Model Ayar Menüleri (Parametre Izgaraları) ---
model_configs = {
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.05]
        },
        'json_path': os.path.join(RESULTS_PATH, 'best_params_gbr.json')
    },
    'LGBM': {
        'model': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        'params': {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.1, 0.05],
            'num_leaves': [10, 20, 31]
        },
        'json_path': os.path.join(RESULTS_PATH, 'best_params_lgbm.json')
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200], 
            'max_depth': [10, 20, None],
            'min_samples_leaf': [1, 2, 4]
        },
        'json_path': os.path.join(RESULTS_PATH, 'best_params_rf.json')
    },
    'XGBoost': { # YENİ KONFİGÜRASYON
        'model': XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        'params': {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.05],
            'subsample': [0.8, 1.0] # XGBoost'a özel ekstra parametre
        },
        'json_path': os.path.join(RESULTS_PATH, 'best_params_xgb.json')
    }
}

# --- Seçilen modelin ayarlarını al ---
config = model_configs[MODEL_TO_OPTIMIZE]
BASE_MODEL = config['model']
PARAM_GRID = config['params']
PARAMS_DOSYASI = config['json_path']
MODEL_ADI = f"{MODEL_TO_OPTIMIZE}_Optimized"

# --- ANA KOD AKIŞI ---
print(f"Deney Başlatıldı: {MODEL_ADI} Optimizasyonu")
start_time = time.time()

X, y, feature_names = preprocess_data(DATA_PATH)

if X is not None:
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- ADIM 3: OPTİMİZASYON ---
    print(f"\nModel ({MODEL_ADI}) için en iyi ayarlar aranıyor...")
    
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
          
    grid_search = GridSearchCV(
        estimator=BASE_MODEL,
        param_grid=PARAM_GRID,
        cv=5, 
        scoring=mae_scorer, 
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("\n--- Optimizasyon Sonuçları ---")
    print(f"En iyi ayarlar bulundu: {grid_search.best_params_}")
    
    try:
        with open(PARAMS_DOSYASI, 'w') as f:
            json.dump(grid_search.best_params_, f, indent=4) 
        print(f"En iyi ayarlar şuraya kaydedildi: {PARAMS_DOSYASI}")
    except Exception as e:
        print(f"HATA: Parametre dosyası kaydedilemedi! {e}")

    best_mae_cv = -grid_search.best_score_
    print(f"En iyi 'Cross-Validation' MAE: {best_mae_cv:,.2f} $")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    
    # --- ADIM 4: DEĞERLENDİRME ---
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n--- Sayısal Değerlendirme (Final Test Seti - {MODEL_ADI}) ---")
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
        'en_iyi_ayarlar': str(grid_search.best_params_),
        'cv_mae_skoru': best_mae_cv,
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
        
    print(f"\nDeney ({MODEL_ADI}) {total_time:.2f} saniyede tamamlandı.")
else:
    print("Veri yüklenemediği için analiz durduruldu.")
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
import os
import json # Ayarları kaydetmek için
import time

# --- KÜTÜPHANE İMPORTLARI ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from preprocess import preprocess_data

#--------------------------------------------------------------------------------
# --- KONTROL PANELİ 🎛️ ---
# Hangi modeli optimize etmek istiyorsun?
# Seçenekler: 'GradientBoosting', 'LGBM', 'RandomForest'
MODEL_TO_OPTIMIZE = 'RandomForest'  # BURADA MODELİ SEÇİN
#--------------------------------------------------------------------------------

# --- YOL TANIMLARI ---
DATA_PATH = 'data/train.csv'
RESULTS_PATH = 'results/best_params/'
IMAGES_PATH = 'results/images/'
LOG_DOSYASI = os.path.join(RESULTS_PATH, 'experiment_log.csv')

# --- Model Ayar Menüleri (Parametre Izgaraları) ---
model_configs = {
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.05]
        },
        'json_path': os.path.join(RESULTS_PATH, 'best_params_gbr.json')
    },
    'LGBM': {
        'model': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        'params': {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.1, 0.05],
            'num_leaves': [10, 20, 31]
        },
        'json_path': os.path.join(RESULTS_PATH, 'best_params_lgbm.json')
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200], # RF yavaş olduğu için daha az ayar
            'max_depth': [10, 20, None],
            'min_samples_leaf': [1, 2, 4]
        },
        'json_path': os.path.join(RESULTS_PATH, 'best_params_rf.json')
    }
}

# --- Seçilen modelin ayarlarını al ---
if MODEL_TO_OPTIMIZE not in model_configs:
    raise ValueError(f"MODEL_TO_OPTIMIZE '{MODEL_TO_OPTIMIZE}' olarak ayarlandı, ancak 'model_configs' içinde tanınmıyor.")

config = model_configs[MODEL_TO_OPTIMIZE]
BASE_MODEL = config['model']
PARAM_GRID = config['params']
PARAMS_DOSYASI = config['json_path']
MODEL_ADI = f"{MODEL_TO_OPTIMIZE}_Optimized"

# --- Klasörlerin var olduğundan emin ol ---
if not os.path.exists(IMAGES_PATH): os.makedirs(IMAGES_PATH)
if not os.path.exists(RESULTS_PATH): os.makedirs(RESULTS_PATH)
    
# --- ANA KOD AKIŞI ---
print(f"Deney Başlatıldı: {MODEL_ADI} Optimizasyonu")
start_time = time.time()

# 1. Veriyi Hazırla
X, y, feature_names = preprocess_data(DATA_PATH)

if X is not None:
    # 2. Veriyi Böl
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #--------------------------------------------------------------------------------
    # --- ADIM 3: HİPERPARAMETRE OPTİ... (GridSearchCV) ---
    #--------------------------------------------------------------------------------
    print(f"\nModel ({MODEL_ADI}) için en iyi ayarlar aranıyor...")
    print(f"Denenecek Parametre Izgarası: {PARAM_GRID}")
    
    # MAE'yi 'scoring' olarak kullanmak için (scikit-learn negatifi maksimize eder)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
          
    grid_search = GridSearchCV(
        estimator=BASE_MODEL,
        param_grid=PARAM_GRID,
        cv=5, 
        scoring=mae_scorer, 
        n_jobs=-1,
        verbose=1
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
    y_pred = best_model.predict(x_test)
    
    #--------------------------------------------------------------------------------
    # --- ADIM 4: DEĞERLENDİRME VE RAPORLAMA ---
    #--------------------------------------------------------------------------------

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n--- Sayısal Değerlendirme (Final Test Seti - {MODEL_ADI}) ---")
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
        'en_iyi_ayarlar': str(grid_search.best_params_),
        'cv_mae_skoru': best_mae_cv,
        'test_suresi': f"{time.time() - start_time:.2f} saniye",
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
    
    # Grafik 2: Hata Grafiği
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Gerçek Fiyatlar'); plt.ylabel('Hata Payı (Gerçek - Tahmin)')
    plt.title(f'Hata Dağılım Grafiği ({MODEL_ADI})')
    graph_filename = os.path.join(IMAGES_PATH, f'py_residuals_{MODEL_ADI}.png')
    plt.savefig(graph_filename, dpi=300)
    print(f"Hata grafiği kaydedildi: {graph_filename}")

    # Grafik 3: Özellik Önem Düzeyi
    try:
        importances = best_model.feature_importances_
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
        

    print(f"\nDeney ({MODEL_ADI}) {total_time:.2f} saniyede tamamlandı.")
else:
    print("Veri yüklenemediği için analiz durduruldu.")"""