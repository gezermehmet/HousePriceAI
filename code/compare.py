import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- VERİLERİ ELLE GİRME ---
# Raporundan aldığın en iyi XGBoost optimize sonuçlarını buraya gir.
# (Senin verdiğin hayali sonuçları kullanıyorum)
XGBOOST_PYTHON_MAE = 14929.70  # Python MAE (optimize edilmiş)
XGBOOST_R_MAE = 14252.00      # R MAE (optimize edilmiş)

XGBOOST_PYTHON_R2 = 0.9262     # Python R²
XGBOOST_R_R2 = 0.8970          # R R²

# --- GRAFİK VERİSİNİ OLUŞTURMA ---

# Metrikleri uzun (long) formatta tutmak, Seaborn için en iyi yoldur.
data = {
    'Platform': ['Python (scikit-learn)', 'R (native)', 'Python (scikit-learn)', 'R (native)'],
    'Metrik': ['MAE (Ortalama Hata)', 'MAE (Ortalama Hata)', 'R-Kare (Açıklama Gücü)', 'R-Kare (Açıklama Gücü)'],
    'Değer': [XGBOOST_PYTHON_MAE, XGBOOST_R_MAE, XGBOOST_PYTHON_R2, XGBOOST_R_R2]
}

df_comparison = pd.DataFrame(data)

# --- GÖRSELLEŞTİRME ---

plt.style.use('ggplot') # Grafiğe şık bir stil verelim
plt.figure(figsize=(14, 6))

# FacetGrid kullanarak MAE ve R-Kare'yi yan yana iki ayrı grafikte göster
g = sns.catplot(
    data=df_comparison, 
    x='Platform', 
    y='Değer', 
    col='Metrik', # Metriklere göre sütunları ayır (MAE ve R2 yan yana)
    kind='bar',
    sharey=False, # Y ekseni ölçeklerini farklı tut (Çünkü MAE ve R2 farklı aralıklarda)
    palette={'Python (scikit-learn)': '#1f77b4', 'R (native)': '#d62728'} # Mavi ve Kırmızı
)

# Başlıkları ve etiketleri temizle
g.set_titles("{col_name}")
g.set_axis_labels("", "Değer / Hata Payı")

# MAE grafiğine değeri yaz
for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=5)

# R2 grafiği için eksen limitlerini 0.8'den başlatalım (daha net fark için)
g.axes.flat[1].set_ylim(0.85, 1.0) 

# Ana başlık ekle
plt.suptitle('XGBoost Modelinde Platformların Kıyaslaması', y=1.05, fontsize=16)

# --- GRAFİĞİ KAYDETME ---
IMAGES_PATH = 'results/images/'
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)

graph_filename = os.path.join(IMAGES_PATH, 'platform_comparison_xgb.png')
plt.savefig(graph_filename, dpi=300, bbox_inches='tight')

print(f"\n✓ Kıyaslama grafiği başarıyla kaydedildi: {graph_filename}")
print("Bu grafiği raporunuza ekleyebilirsiniz.")