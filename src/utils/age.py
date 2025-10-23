import numpy as np
import pandas as pd

def add_age_optimality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Age.30 and Age.int features for both players, per match.
      - A_Age30, B_Age30: |age - 30|
      - A_AgeInt, B_AgeInt: distance to the optimal interval [28, 32]
                            (0 if within, else distance to the nearest bound)
    Leaves existing columns unchanged and preserves NaNs.
    """
    a_age = pd.to_numeric(df['player_A_age'], errors='coerce')
    b_age = pd.to_numeric(df['player_B_age'], errors='coerce')

    df['A_Age30'] = (a_age - 30.0).abs()
    df['B_Age30'] = (b_age - 30.0).abs()

    lower, upper = 28.0, 32.0
    df['A_AgeInt'] = np.where(a_age < lower, lower - a_age,
                        np.where(a_age > upper, a_age - upper, 0.0))
    df['B_AgeInt'] = np.where(b_age < lower, lower - b_age,
                        np.where(b_age > upper, b_age - upper, 0.0))
    df['Age30_Diff'] = df['A_Age30'] - df['B_Age30']
    df['AgeInt_Diff'] = df['A_AgeInt'] - df['B_AgeInt']

    return df
