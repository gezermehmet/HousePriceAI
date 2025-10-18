import pandas as pd
import numpy as np


def preprocess_data(data):
    df = data.copy()

    cols_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'MasVnrType']
    df = df.drop(columns=cols_to_drop)

    median_lot_frontage = df['LotFrontage'].median()
    df['LotFrontage'] = df['LotFrontage'].fillna(median_lot_frontage)


    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns


    print("Sayısal sütunlardaki boşluklar dolduruluyor...")
    for col in numerical_cols:
        
        median = df[col].median()
        df[col] = df[col].fillna(median)
    
    print("✓ Sayısal boşluklar tamamlandı.")

    categorical_cols = df.select_dtypes(include=['object']).columns

    print("Kategorik sütunlardaki boşluklar dolduruluyor...")

    for col in categorical_cols:
        
        mode = df[col].mode()[0]
        df[col] = df[col].fillna(mode)
    
    print("✓ Kategorik boşluklar tamamlandı.")

    return df


df_raw = pd.read_csv('train.csv')

df_clean = preprocess_data(df_raw)
df_clean.info()


# for column in df.columns:
#     missing_count = df[column].isnull().sum()
#     if missing_count > 0:
#         print(f"Column '{column}' has {missing_count} missing values.")