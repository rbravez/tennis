## We assume here the dataset has been preprocessed and ratings are in a standard format.

from collections import defaultdict
import pandas as pd
import math

def calculate_general_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Global historical evolution Elo ratings for players."""
    df = df.sort_values(by='tourney_date').reset_index(drop=True)

    elo_overall: defaultdict = defaultdict(lambda: 1200)
    match_counts: defaultdict = defaultdict(int)

    def expected_score(elo_a: float, elo_b: float) -> float:
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    
    def update_rating(R: float, E: float, S: float, K: float) -> float:
        return R + K * (S - E)

    def get_dynamic_k(matches_played: int) -> float:
        if matches_played < 30:
            return 40 - (matches_played * 0.8)
        else:
            return 16

    df['A_Elo_Overall'] = 1200.0
    df['B_Elo_Overall'] = 1200.0

    for index, row in df.iterrows():
        player_a = row['player_A_id']
        player_b = row['player_B_id']
        target = row['target']
        
        R_A = elo_overall[player_a]
        R_B = elo_overall[player_b]
        
        df.at[index, 'A_Elo_Overall'] = R_A
        df.at[index, 'B_Elo_Overall'] = R_B
        
        E_A = expected_score(R_A, R_B)
        E_B = expected_score(R_B, R_A)
        
        K_A = get_dynamic_k(match_counts[player_a])
        K_B = get_dynamic_k(match_counts[player_b])

        R_A_new = update_rating(R_A, E_A, target, K_A)
        R_B_new = update_rating(R_B, E_B, 1 - target, K_B)

        elo_overall[player_a] = R_A_new
        elo_overall[player_b] = R_B_new

        match_counts[player_a] += 1
        match_counts[player_b] += 1
        
    return df

def calculate_surface_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Surface-specific historical evolution Elo ratings for players."""
    df = df.sort_values(by='tourney_date').reset_index(drop=True)
    
    elo_surface = defaultdict(lambda: defaultdict(lambda: 1200))
    match_counts_surface = defaultdict(lambda: defaultdict(int))

    def expected_score(elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    
    def update_rating(R, E, S, K):
        return R + K * (S - E)
    
    def get_dynamic_k(matches_played):
        if matches_played < 30:
            return 40 - (matches_played * 0.8)
        else:
            return 16

    df['A_Elo_Surface'] = 1200.0
    df['B_Elo_Surface'] = 1200.0

    for index, row in df.iterrows():
        player_a = row['player_A_id']
        player_b = row['player_B_id']
        surface = row['surface']
        target = row['target']
        
        R_A = elo_surface[player_a][surface]
        R_B = elo_surface[player_b][surface]
        
        df.at[index, 'A_Elo_Surface'] = R_A
        df.at[index, 'B_Elo_Surface'] = R_B
        
        E_A = expected_score(R_A, R_B)
        E_B = expected_score(R_B, R_A)
        
        K_A = get_dynamic_k(match_counts_surface[player_a][surface])
        K_B = get_dynamic_k(match_counts_surface[player_b][surface])

        R_A_new = update_rating(R_A, E_A, target, K_A)
        R_B_new = update_rating(R_B, E_B, 1 - target, K_B)

        elo_surface[player_a][surface] = R_A_new
        elo_surface[player_b][surface] = R_B_new

        match_counts_surface[player_a][surface] += 1
        match_counts_surface[player_b][surface] += 1
        
    return df