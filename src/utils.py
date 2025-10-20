import pandas as pd
import numpy as np
import math
from collections import defaultdict

def get_target(df):
    """
    Creates a balanced dataset for binary classification in tennis match outcomes.
    1 refers to player A winning, 0 to player B winning.
    
    Preserves chronological order by using stratified assignment
    rather than random shuffling, preventing temporal data leakage.
    """
    df = df.reset_index(drop=True)

    df['_temp_flip'] = df.index % 2

    df_part1 = df[df['_temp_flip'] == 0].copy()
    df_part2 = df[df['_temp_flip'] == 1].copy()

    def rename_part1(col):
        if col.startswith('winner_'):
            return col.replace('winner_', 'A_')
        elif col.startswith('loser_'):
            return col.replace('loser_', 'B_')
        elif col.startswith('w_'):
            return col.replace('w_', 'A_')
        elif col.startswith('l_'):
            return col.replace('l_', 'B_')
        else:
            return col
    def rename_part2(col):
        if col.startswith('winner_'):
            return col.replace('winner_', 'B_')
        elif col.startswith('loser_'):
            return col.replace('loser_', 'A_')
        elif col.startswith('w_'):
            return col.replace('w_', 'B_')
        elif col.startswith('l_'):
            return col.replace('l_', 'A_')
        else:
            return col
        
    df_part1 = df_part1.rename(columns=rename_part1)
    df_part2 = df_part2.rename(columns=rename_part2)

    df_part1['target'] = 1
    df_part2['target'] = 0

    final_df = pd.concat([df_part1, df_part2], ignore_index=False)
    final_df = final_df.sort_index().reset_index(drop=True)

    final_df = final_df.drop(columns=['_temp_flip'], errors='ignore')
    
    cols_to_drop = [
        col for col in final_df.columns 
        if any(prefix in col for prefix in ['winner_', 'loser_', 'w_', 'l_'])
    ]
    
    final_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    return final_df

def c_general_elo(df):
    """Calculates Global historical evolution Elo ratings for players."""
    # Ensure sorted by date
    df = df.sort_values(by='tourney_date').reset_index(drop=True)
    
    elo_overall = defaultdict(lambda: 1200)
    match_counts = defaultdict(int)

    def expected_score(elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    
    def update_rating(R, E, S, K):
        return R + K * (S - E)
    
    def get_dynamic_k(matches_played):
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

def c_surface_elo(df):
    """Calculates Surface specific historical evolution Elo ratings for players."""
    df = df.sort_values(by='tourney_date').reset_index(drop=True)
    
    elo_surface = defaultdict(lambda: 1200)
    match_counts_surface = defaultdict(int)

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
        
        key_a = (player_a, surface)
        key_b = (player_b, surface)
        
        R_A = elo_surface[key_a]
        R_B = elo_surface[key_b]
        
        df.at[index, 'A_Elo_Surface'] = R_A
        df.at[index, 'B_Elo_Surface'] = R_B
        
        E_A = expected_score(R_A, R_B)
        E_B = expected_score(R_B, R_A)
        
        K_A = get_dynamic_k(match_counts_surface[key_a])
        K_B = get_dynamic_k(match_counts_surface[key_b])

        R_A_new = update_rating(R_A, E_A, target, K_A)
        R_B_new = update_rating(R_B, E_B, 1 - target, K_B)

        elo_surface[key_a] = R_A_new
        elo_surface[key_b] = R_B_new

        match_counts_surface[key_a] += 1
        match_counts_surface[key_b] += 1
        
    return df

def c_general_glicko(df):
    """Calculates Global historical evolution Glicko ratings for players."""
    df = df.sort_values(by='tourney_date').reset_index(drop=True)
    glicko_rating = defaultdict(lambda: 1500)
    glicko_rd = defaultdict(lambda: 350)
    last_played = defaultdict(lambda: None)

    Q = math.log(10) / 400
    C_SQUARED = 50**2

    def g(rd):
        return 1 / math.sqrt(1 + 3 * Q**2 * rd**2 / math.pi**2)

    def expected_score(r, r_j, rd_j):
        return 1 / (1 + 10 ** (-g(rd_j) * (r - r_j) / 400))
    
    def update_rd_for_inactivity(rd, periods_inactive):
        if periods_inactive <= 0:
            return rd
        new_rd_squared = rd**2 + C_SQUARED * periods_inactive
        return min(math.sqrt(new_rd_squared), 350)
    
    def update_rating(r, rd, outcomes):
        if not outcomes:
            return r, rd
        
        d_squared_inv = sum(
            g(opp_rd)**2 * expected_score(r, opp_r, opp_rd) * 
            (1 - expected_score(r, opp_r, opp_rd))
            for opp_r, opp_rd, _ in outcomes
        )
        d_squared = 1 / (Q**2 * d_squared_inv) if d_squared_inv > 0 else float('inf')
        
        rating_change = sum(
            g(opp_rd) * (score - expected_score(r, opp_r, opp_rd))
            for opp_r, opp_rd, score in outcomes
        )
        new_r = r + (Q / (1/rd**2 + 1/d_squared)) * rating_change
        new_rd = math.sqrt(1 / (1/rd**2 + 1/d_squared))
        
        return new_r, new_rd
    for index, row in df.iterrows():
        player_a = row['player_A_id']
        player_b = row['player_B_id']
        target = row['target']
        current_date = row['tourney_date']
        
        for player in [player_a, player_b]:
            if last_played[player] is not None:
                days_inactive = (current_date - last_played[player]).days
                periods = days_inactive // 90
                if periods > 0:
                    glicko_rd[player] = update_rd_for_inactivity(glicko_rd[player], periods)
        
        R_A = glicko_rating[player_a]
        RD_A = glicko_rd[player_a]
        R_B = glicko_rating[player_b]
        RD_B = glicko_rd[player_b]
        
        df.at[index, 'A_Glicko_Rating'] = R_A
        df.at[index, 'B_Glicko_Rating'] = R_B
        df.at[index, 'A_Glicko_RD'] = RD_A
        df.at[index, 'B_Glicko_RD'] = RD_B
        
        R_A_new, RD_A_new = update_rating(R_A, RD_A, [(R_B, RD_B, target)])
        R_B_new, RD_B_new = update_rating(R_B, RD_B, [(R_A, RD_A, 1 - target)])
        
        glicko_rating[player_a] = R_A_new
        glicko_rd[player_a] = RD_A_new
        glicko_rating[player_b] = R_B_new
        glicko_rd[player_b] = RD_B_new
        
        last_played[player_a] = current_date
        last_played[player_b] = current_date
        
    return df

def c_surface_glicko(df):
    """Calculates Surface specific historical evolution Glicko ratings for players."""
    df = df.sort_values(by='tourney_date').reset_index(drop=True)
    
    glicko_rating = defaultdict(lambda: 1500)
    glicko_rd = defaultdict(lambda: 350)
    last_played = defaultdict(lambda: None)
    
    Q = math.log(10) / 400
    C_SQUARED = 50**2
    
    def g(rd):
        return 1 / math.sqrt(1 + 3 * Q**2 * rd**2 / math.pi**2)
    
    def expected_score(r, r_j, rd_j):
        return 1 / (1 + 10 ** (-g(rd_j) * (r - r_j) / 400))
    
    def update_rd_for_inactivity(rd, periods_inactive):
        if periods_inactive <= 0:
            return rd
        new_rd_squared = rd**2 + C_SQUARED * periods_inactive
        return min(math.sqrt(new_rd_squared), 350)
    
    def update_rating(r, rd, outcomes):
        if not outcomes:
            return r, rd
        
        d_squared_inv = sum(
            g(opp_rd)**2 * expected_score(r, opp_r, opp_rd) * 
            (1 - expected_score(r, opp_r, opp_rd))
            for opp_r, opp_rd, _ in outcomes
        )
        d_squared = 1 / (Q**2 * d_squared_inv) if d_squared_inv > 0 else float('inf')
        
        rating_change = sum(
            g(opp_rd) * (score - expected_score(r, opp_r, opp_rd))
            for opp_r, opp_rd, score in outcomes
        )
        new_r = r + (Q / (1/rd**2 + 1/d_squared)) * rating_change
        new_rd = math.sqrt(1 / (1/rd**2 + 1/d_squared))
        
        return new_r, new_rd

    df['A_Glicko_Surface_Rating'] = 1500.0
    df['B_Glicko_Surface_Rating'] = 1500.0
    df['A_Glicko_Surface_RD'] = 350.0
    df['B_Glicko_Surface_RD'] = 350.0

    for index, row in df.iterrows():
        player_a = row['player_A_id']
        player_b = row['player_B_id']
        surface = row['surface']
        target = row['target']
        current_date = row['tourney_date']
        
        key_a = (player_a, surface)
        key_b = (player_b, surface)
        
        for key in [key_a, key_b]:
            if last_played[key] is not None:
                days_inactive = (current_date - last_played[key]).days
                periods = days_inactive // 90
                if periods > 0:
                    glicko_rd[key] = update_rd_for_inactivity(glicko_rd[key], periods)
        
        R_A = glicko_rating[key_a]
        RD_A = glicko_rd[key_a]
        R_B = glicko_rating[key_b]
        RD_B = glicko_rd[key_b]
        
        df.at[index, 'A_Glicko_Surface_Rating'] = R_A
        df.at[index, 'B_Glicko_Surface_Rating'] = R_B
        df.at[index, 'A_Glicko_Surface_RD'] = RD_A
        df.at[index, 'B_Glicko_Surface_RD'] = RD_B
        
        R_A_new, RD_A_new = update_rating(R_A, RD_A, [(R_B, RD_B, target)])
        R_B_new, RD_B_new = update_rating(R_B, RD_B, [(R_A, RD_A, 1 - target)])
        
        glicko_rating[key_a] = R_A_new
        glicko_rd[key_a] = RD_A_new
        glicko_rating[key_b] = R_B_new
        glicko_rd[key_b] = RD_B_new
        
        last_played[key_a] = current_date
        last_played[key_b] = current_date
        
    return df