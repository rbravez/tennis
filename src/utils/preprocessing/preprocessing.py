import pandas as pd
from collections import defaultdict

def get_target(df):
    """
    Creates a balanced dataset for binary classification in tennis match outcomes.
    1 refers to player A winning, 0 to player B winning.
    
    CRITICAL: Preserves chronological order by using stratified assignment
    rather than random shuffling, preventing temporal data leakage.
    """
    # DO NOT SHUFFLE - preserve chronological order!
    df = df.reset_index(drop=True)
    
    # Create alternating pattern while preserving time order
    # Use modulo to alternate which player is A vs B
    df['_temp_flip'] = df.index % 2
    
    df_part1 = df[df['_temp_flip'] == 0].copy()
    df_part2 = df[df['_temp_flip'] == 1].copy()

    def rename_part1(col):
        if col.startswith('winner_'):
            return col.replace('winner_', 'player_A_', 1)
        elif col.startswith('loser_'):
            return col.replace('loser_', 'player_B_', 1)
        elif col.startswith('w_'):
            return col.replace('w_', 'player_A_', 1)
        elif col.startswith('l_'):
            return col.replace('l_', 'player_B_', 1)
        return col
    
    def rename_part2(col):
        if col.startswith('winner_'):
            return col.replace('winner_', 'player_B_', 1)
        elif col.startswith('loser_'):
            return col.replace('loser_', 'player_A_', 1)
        elif col.startswith('w_'):
            return col.replace('w_', 'player_B_', 1)
        elif col.startswith('l_'):
            return col.replace('l_', 'player_A_', 1)
        return col
    
    df_part1 = df_part1.rename(columns=rename_part1)
    df_part1['target'] = 1

    df_part2 = df_part2.rename(columns=rename_part2)
    df_part2['target'] = 0

    # Concatenate and re-sort by original index to preserve time order
    final_df = pd.concat([df_part1, df_part2], ignore_index=False)
    final_df = final_df.sort_index().reset_index(drop=True)

    # Drop temporary column and unwanted columns
    final_df = final_df.drop(columns=['_temp_flip'], errors='ignore')
    
    cols_to_drop = [
        col for col in final_df.columns 
        if any(prefix in col for prefix in ['winner_', 'loser_', 'w_', 'l_'])
    ]
    
    final_df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    return final_df