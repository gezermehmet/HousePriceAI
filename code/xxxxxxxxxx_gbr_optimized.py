import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import os 
import time

# --- KONTROL PANELİ ---
MODEL_ADI = "GradientBoosting_Optimized_v1"
LOG_DOSYASI = 'results/experiment_log.csv'
RESULTS_PATH = 'results/'
IMAGES_PATH = 'results/images/'
DATA_PATH = 'data/train.csv'

# --- Klasörlerin var olduğundan emin ol ---
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)
    print(f"Klasör oluşturuldu: {IMAGES_PATH}")
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
    print(f"Klasör oluşturuldu: {RESULTS_PATH}")

# --- ADIM 1: Veri Temizleme Fonksiyonu ---
def preprocess_data(data_path):
    """Veriyi yükler, temizler ve eksik değerleri doldurur."""
    print("Veri ön işleme başlıyor...")
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"HATA: '{data_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol et.")
        return None, None

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
        if col != 'SalePrice': # Hedef sütunu doldurma
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        
    # 2. Kategorik Sütunlar (Mod ile)
    categorical_cols = df.select_dtypes(include=['object']).columns
    print("Kategorik sütunlardaki boşluklar dolduruluyor...")
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        
    print("Metin sütunları sayısala çevriliyor (One-Hot Encoding)...")
    df = pd.get_dummies(df)
    print("✓ Veri ön işleme bitti.")
    
    # Hedef (y) ve Özellikleri (X) ayır
    X = df.drop('SalePrice', axis=1) 
    y = df['SalePrice']
    
    return X, y

# --- ANA KOD AKIŞI ---

start_time = time.time() # Toplam süreyi ölçmek için zamanlayıcıyı başlat

# 1. Veriyi Hazırla (Temizle, Doldur, Dönüştür, Ayır)
X, y = preprocess_data(DATA_PATH)

if X is not None:
    # 2. Veriyi Sınav (Test) ve Eğitim Olarak Böl
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #--------------------------------------------------------------------------------
    # --- ADIM 3: HİPERPARAMETRE OPTİ... (GridSearchCV) ---
    #--------------------------------------------------------------------------------
    print(f"\nModel ({MODEL_ADI}) için en iyi ayarlar aranıyor...")

    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.05]
    }
    base_model = GradientBoostingRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5, # 5-katlı çapraz doğrulama
        scoring='neg_mean_absolute_error', # MAE'ye göre optimize et
        n_jobs=-1, # Tüm işlemcileri kullan
        verbose=1  # İlerlemeyi göster
    )
    grid_search.fit(X_train, y_train)

    print("\n--- Optimizasyon Sonuçları ---")
    print(f"En iyi ayarlar bulundu: {grid_search.best_params_}")
    best_mae_cv = -grid_search.best_score_
    print(f"En iyi 'Cross-Validation' MAE (Deneme Sınavı Ortalaması): {best_mae_cv:,.2f} $")

    # En iyi modeli al (Artık bu bizim final modelimiz)
    best_model = grid_search.best_estimator_

    # Tahminleri en iyi modelle yap
    y_pred = best_model.predict(x_test)
    print("✓ Tahminler en iyi modelle yapıldı (Final Sınavı).")

    #--------------------------------------------------------------------------------
    # --- ADIM 4: DEĞERLENDİRME (Optimize Edilmiş) ---
    #--------------------------------------------------------------------------------

    print("\n--- Sayısal Değerlendirme (Final Test Seti) ---")
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared (R2) Skoru: {r2:.4f}")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Ortalama Mutlak Hata (MAE): {mae:,.2f} $")

    # --- ADIM 5: SONUÇLARI OTOMATİK KAYDETME ---
    print(f"\nSonuçlar '{LOG_DOSYASI}' dosyasına kaydediliyor...")
    log_entry = {
        'model_adi': MODEL_ADI,
        'mae_test_seti': mae,
        'r2_test_seti': r2,
        'en_iyi_ayarlar': str(grid_search.best_params_),
        'tarih': pd.to_datetime('today').strftime('%Y-%m-%d %H:%M')
    }
    log_df = pd.DataFrame([log_entry])

    # Dosya yoksa başlık (header) ile oluştur, varsa üzerine ekle
    try:
        if not os.path.exists(LOG_DOSYASI):
            log_df.to_csv(LOG_DOSYASI, index=False)
        else:
            log_df.to_csv(LOG_DOSYASI, mode='a', header=False, index=False)
        print("✓ Sonuçlar kaydedildi.")
    except Exception as e:
        print(f"HATA: Log dosyası kaydedilemedi! {e}")

    #--------------------------------------------------------------------------------
    # --- ADIM 6: GELİŞMİŞ GRAFİKLERİ KAYDETME ---
    #--------------------------------------------------------------------------------
    print("\n--- Görsel Değerlendirmeler Hazırlanıyor ---")
    pd.options.display.float_format = '{:,.0f}'.format # Raporlama için formatlama

    # --- GRAFİK 1: Gerçek vs. Tahmin (Scatter Plot) ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Gerçek Fiyatlar (y_test)')
    plt.ylabel('Tahmini Fiyatlar (y_pred)')
    plt.title(f'Gerçek vs. Tahmin ({MODEL_ADI})')
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    graph_filename = os.path.join(IMAGES_PATH, f'py_scatter_{MODEL_ADI}.png')
    plt.savefig(graph_filename, dpi=300)
    print(f"Grafik kaydedildi: {graph_filename}")

    # --- GRAFİK 2: Hata Grafiği (Residual Plot) ---
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Gerçek Fiyatlar (y_test)')
    plt.ylabel('Hata Payı (Residuals)')
    plt.title(f'Hata Grafiği ({MODEL_ADI})')
    graph_filename = os.path.join(IMAGES_PATH, f'py_residuals_{MODEL_ADI}.png')
    plt.savefig(graph_filename, dpi=300)
    print(f"Hata grafiği kaydedildi: {graph_filename}")

    # --- GRAFİK 3: Özellik Önem Düzeyi (Feature Importance) ---
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

    # plt.show() # Script modunda show() engelleme yapabilir, o yüzden kapalı

    # --- Detaylı Tahmin İncelemesi ---
    print("\n--- Tahmin Sonuçlarının Detaylı İncelenmesi ---")
    y_test_reset = y_test.reset_index(drop=True)
    results_df = pd.DataFrame({'Gerçek Fiyat': y_test_reset, 'Tahmini Fiyat': y_pred})
    results_df['Fark'] = results_df['Gerçek Fiyat'] - results_df['Tahmini Fiyat']
    results_df['Mutlak Hata'] = results_df['Fark'].abs()

    print("\n--- EN İYİ 5 TAHMİN (Optimize Edilmiş) ---")
    print(results_df.sort_values(by='Mutlak Hata', ascending=True).head(5))

    print("\n--- EN KÖTÜ 5 TAHMİN (Optimize Edilmiş) ---")
    print(results_df.sort_values(by='Mutlak Hata', ascending=False).head(5))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nDeney ({MODEL_ADI}) {total_time:.2f} saniyede tamamlandı.")

else:
    print("Veri yüklenemediği için analiz durduruldu.")