import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta

def h2h_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cumulative total H2H matches and H2H wins for each player (directional),
    before each match in chronological order.

    Produces:
        A_H2H_Total_Matches
        B_H2H_Total_Matches
        A_H2H_Wins
        B_H2H_Wins
    """
    df = df.sort_values('tourney_date').reset_index(drop=True)

    h2h_total_matches = defaultdict(int)
    h2h_total_wins = defaultdict(int)

    df['A_H2H_Total_Matches'] = 0
    df['B_H2H_Total_Matches'] = 0
    df['A_H2H_Wins'] = 0
    df['B_H2H_Wins'] = 0

    for i, r in df.iterrows():
        a, b, res = r['player_A_id'], r['player_B_id'], r['target']
        key_a_b = (a, b)
        key_b_a = (b, a)

        # Pre-match values
        df.loc[i, 'A_H2H_Total_Matches'] = h2h_total_matches[key_a_b]
        df.loc[i, 'B_H2H_Total_Matches'] = h2h_total_matches[key_b_a]
        df.loc[i, 'A_H2H_Wins'] = h2h_total_wins[key_a_b]
        df.loc[i, 'B_H2H_Wins'] = h2h_total_wins[key_b_a]

        # Update after match
        h2h_total_matches[key_a_b] += 1
        h2h_total_matches[key_b_a] += 1

        if res == 1:
            h2h_total_wins[key_a_b] += 1
        else:
            h2h_total_wins[key_b_a] += 1

    return df


def h2h_surface_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cumulative total H2H matches and H2H wins on the same surface (directional),
    before each match in chronological order.

    Produces:
        A_H2H_Surface_Matches
        B_H2H_Surface_Matches
        A_H2H_Surface_Wins
        B_H2H_Surface_Wins
    """
    df = df.sort_values('tourney_date').reset_index(drop=True)

    h2h_surface_matches = defaultdict(int)
    h2h_surface_wins = defaultdict(int)

    df['A_H2H_Surface_Matches'] = 0
    df['B_H2H_Surface_Matches'] = 0
    df['A_H2H_Surface_Wins'] = 0
    df['B_H2H_Surface_Wins'] = 0

    for i, r in df.iterrows():
        a, b, s, res = r['player_A_id'], r['player_B_id'], r['surface'], r['target']
        key_a_b = (a, b, s)
        key_b_a = (b, a, s)

        # Pre-match values
        df.loc[i, 'A_H2H_Surface_Matches'] = h2h_surface_matches[key_a_b]
        df.loc[i, 'B_H2H_Surface_Matches'] = h2h_surface_matches[key_b_a]
        df.loc[i, 'A_H2H_Surface_Wins'] = h2h_surface_wins[key_a_b]
        df.loc[i, 'B_H2H_Surface_Wins'] = h2h_surface_wins[key_b_a]

        # Update after match
        h2h_surface_matches[key_a_b] += 1
        h2h_surface_matches[key_b_a] += 1

        if res == 1:
            h2h_surface_wins[key_a_b] += 1
        else:
            h2h_surface_wins[key_b_a] += 1

    return df


def h2h_streak(df: pd.DataFrame) -> pd.DataFrame:
    """Current consecutive-win streak for each player vs the other before the match."""
    df = df.sort_values('tourney_date').reset_index(drop=True)

    h2h_streaks = defaultdict(int)

    df['A_H2H_Streak'] = 0
    df['B_H2H_Streak'] = 0

    for i, r in df.iterrows():
        a, b, res = r['player_A_id'], r['player_B_id'], r['target']
        key_a_b = (a, b)
        key_b_a = (b, a)

        df.loc[i, 'A_H2H_Streak'] = h2h_streaks[key_a_b]
        df.loc[i, 'B_H2H_Streak'] = h2h_streaks[key_b_a]

        # Update streaks after match
        if res == 1:
            h2h_streaks[key_a_b] = h2h_streaks[key_a_b] + 1
            h2h_streaks[key_b_a] = 0
        else:
            h2h_streaks[key_b_a] = h2h_streaks[key_b_a] + 1
            h2h_streaks[key_a_b] = 0

    return df


def h2h_days_since_last(df: pd.DataFrame) -> pd.DataFrame:
    """Days since last meeting overall and on the same surface."""
    df = df.sort_values('tourney_date').reset_index(drop=True)

    last_meeting = {}
    last_meeting_surface = {}

    df['H2H_Days_Since_Last'] = pd.NA
    df['H2H_Days_Since_Last_SameSurface'] = pd.NA

    for i, r in df.iterrows():
        a, b, s = r['player_A_id'], r['player_B_id'], r['surface']
        date_int = int(r['tourney_date'])
        cur_date = datetime.strptime(str(date_int), "%Y%m%d")

        key = (a, b)
        key_surface = (a, b, s)

        if key in last_meeting:
            prev = datetime.strptime(str(last_meeting[key]), "%Y%m%d")
            df.loc[i, 'H2H_Days_Since_Last'] = (cur_date - prev).days
        if key_surface in last_meeting_surface:
            prev = datetime.strptime(str(last_meeting_surface[key_surface]), "%Y%m%d")
            df.loc[i, 'H2H_Days_Since_Last_SameSurface'] = (cur_date - prev).days

        last_meeting[key] = date_int
        last_meeting_surface[key_surface] = date_int

    return df


def h2h_last_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Outcome of the most recent prior meeting (overall and same surface), from A's perspective."""
    df = df.sort_values('tourney_date').reset_index(drop=True)

    last_outcome = {}
    last_outcome_surface = {}

    df['A_H2H_LastOutcome'] = pd.NA
    df['A_H2H_LastOutcome_SameSurface'] = pd.NA

    for i, r in df.iterrows():
        a, b, s, res = r['player_A_id'], r['player_B_id'], r['surface'], r['target']
        key = (a, b)
        key_surface = (a, b, s)

        if key in last_outcome:
            df.loc[i, 'A_H2H_LastOutcome'] = last_outcome[key]
        if key_surface in last_outcome_surface:
            df.loc[i, 'A_H2H_LastOutcome_SameSurface'] = last_outcome_surface[key_surface]

        # Update both directions
        if res == 1:
            last_outcome[(a, b)] = 1
            last_outcome[(b, a)] = 0
            last_outcome_surface[(a, b, s)] = 1
            last_outcome_surface[(b, a, s)] = 0
        else:
            last_outcome[(b, a)] = 1
            last_outcome[(a, b)] = 0
            last_outcome_surface[(b, a, s)] = 1
            last_outcome_surface[(a, b, s)] = 0

    return df


def h2h_level_weighted(df: pd.DataFrame, weights=None) -> pd.DataFrame:
    """Weighted cumulative H2H wins before the match, using tournament-level weights."""
    if weights is None:
        weights = {'G': 4.0, 'M': 2.5, 'A': 1.5, 'B': 1.0, 'D': 0.5}

    df = df.sort_values('tourney_date').reset_index(drop=True)
    h2h_weighted = defaultdict(float)

    df['A_H2H_LevelWeighted_Wins'] = 0.0
    df['B_H2H_LevelWeighted_Wins'] = 0.0

    for i, r in df.iterrows():
        a, b, res = r['player_A_id'], r['player_B_id'], r['target']
        lvl = r.get('tourney_level', None)
        w = float(weights.get(lvl, 1.0))

        df.loc[i, 'A_H2H_LevelWeighted_Wins'] = h2h_weighted[(a, b)]
        df.loc[i, 'B_H2H_LevelWeighted_Wins'] = h2h_weighted[(b, a)]

        # Update
        if res == 1:
            h2h_weighted[(a, b)] += w
        else:
            h2h_weighted[(b, a)] += w

    return df
