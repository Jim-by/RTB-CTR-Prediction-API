import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing data for the model"""

    # Time
    df['hour'] = df['hour'] % 100
    df['day'] = (df['hour'] // 100) % 7
    df['is_weekend'] = (df['day'] >= 5).astype('int8')
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Categorical features
    cat_cols = ['site_id', 'site_domain', 'site_category', 'app_id',
                'app_domain', 'app_category', 'device_id']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    # Interactions
    df['hour_site'] = df['hour'].astype(str) + "_" + df['site_id'].astype(str)
    df['hour_app'] = df['hour'].astype(str) + "_" + df['app_id'].astype(str)
    df['site_app'] = df['site_id'].astype(str) + "_" + df['app_id'].astype(str)

    # Removing unnecessary columns
    df = df.drop(['day'], axis=1)

    return df
