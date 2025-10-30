import pandas as pd
from collections import defaultdict
from typing import Optional, Dict, Tuple


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

def _dyn_k(matches_played: int, k_base: float, k_min: float, drop_matches: int) -> float:
    if matches_played < drop_matches:
        k = k_base - (matches_played * (k_base - k_min) / max(1, drop_matches))
    else:
        k = k_min
    return max(k_min, float(k))

def _parse_and_sort(df: pd.DataFrame, date_col: str, extra_sort_cols: Optional[list]) -> pd.DataFrame:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        s = df[date_col]
        if pd.api.types.is_integer_dtype(s) or (s.astype(str).str.len().eq(8).all()):
            df[date_col] = pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")
        else:
            df[date_col] = pd.to_datetime(s, errors="coerce")
    sort_cols = [date_col] + ([c for c in (extra_sort_cols or []) if c in df.columns])
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

def _apply_event_weights(
    df: pd.DataFrame, event_weight_col: Optional[str], event_weight_map: Optional[Dict[str, float]]
) -> pd.DataFrame:
    if event_weight_col and event_weight_map:
        df["_w"] = df[event_weight_col].map(event_weight_map).fillna(1.0)
    else:
        df["_w"] = 1.0
    return df

def _init_outputs(df: pd.DataFrame, start_elo: float) -> None:
    df["A_Elo_Overall"] = start_elo
    df["B_Elo_Overall"] = start_elo
    df["A_Elo_Surface"] = start_elo
    df["B_Elo_Surface"] = start_elo
    df["A_Elo_Overall_Post"] = pd.NA
    df["B_Elo_Overall_Post"] = pd.NA
    df["A_Elo_Surface_Post"] = pd.NA
    df["B_Elo_Surface_Post"] = pd.NA

def _surface_keys(pa: int, pb: int, surface: Optional[str]) -> Tuple[Tuple[int, Optional[str]], Tuple[int, Optional[str]]]:
    return (pa, surface), (pb, surface)

# ---------- main ----------
def calculate_general_elo(
    df: pd.DataFrame,
    *,
    start_elo: float = 1250.0,
    k_overall_base: float = 32.0,
    k_overall_min: float = 16.0,
    k_linear_drop_matches: int = 30,
    # surface K (driven by per-surface matches; smaller by default)
    k_surface_base: float = 20.0,
    k_surface_min: float = 10.0,
    k_surface_drop_matches: int = 10,
    # event weighting
    event_weight_col: Optional[str] = None,
    event_weight_map: Optional[Dict[str, float]] = None,
    # columns
    date_col: str = "tourney_date",
    target_col: str = "target",
    player_a_col: str = "player_A_id",
    player_b_col: str = "player_B_id",
    surface_col: str = "surface",
    extra_sort_cols: Optional[list] = None,
    # surface init: "overall" (default) or "baseline" (start_elo)
    surface_init_mode: str = "overall",
) -> pd.DataFrame:
    """
    Elo for overall + surface (no decay). Surface Elo is initialized from overall by default,
    and its K depends on per-surface match counts so it can move quickly with few surface matches.
    """

    df = _parse_and_sort(df, date_col, extra_sort_cols)
    df = _apply_event_weights(df, event_weight_col, event_weight_map)
    _init_outputs(df, start_elo)

    elo_overall = defaultdict(lambda: start_elo)            # player_id -> rating
    elo_surface = defaultdict(lambda: start_elo)            # (player_id, surface) -> rating
    match_counts_overall = defaultdict(int)                 # per-player (all surfaces)
    match_counts_surface  = defaultdict(int)                # per-player-per-surface

    for i, row in df.iterrows():
        pa, pb = row[player_a_col], row[player_b_col]
        s_a = float(row[target_col]); s_b = 1.0 - s_a
        surf = row[surface_col] if surface_col in df.columns else None
        w = float(row["_w"])

        # Overall pre
        ra_o, rb_o = elo_overall[pa], elo_overall[pb]

        # Surface pre (initialize)
        key_a, key_b = _surface_keys(pa, pb, surf)
        if surf is not None and pd.notna(surf):
            if key_a not in elo_surface:
                elo_surface[key_a] = ra_o if surface_init_mode == "overall" else start_elo
            if key_b not in elo_surface:
                elo_surface[key_b] = rb_o if surface_init_mode == "overall" else start_elo
            ra_s, rb_s = elo_surface[key_a], elo_surface[key_b]
        else:
            ra_s, rb_s = ra_o, rb_o  # if surface missing, mirror overall for reporting

        # Record pre
        df.at[i, "A_Elo_Overall"] = ra_o
        df.at[i, "B_Elo_Overall"] = rb_o
        df.at[i, "A_Elo_Surface"] = ra_s
        df.at[i, "B_Elo_Surface"] = rb_s

        # Expectations (compute once)
        ea_o = _expected(ra_o, rb_o); eb_o = 1.0 - ea_o
        ea_s = _expected(ra_s, rb_s); eb_s = 1.0 - ea_s

        # Dynamic K: overall uses overall matches; surface uses per-surface matches
        k_oa = _dyn_k(match_counts_overall[pa], k_overall_base, k_overall_min, k_linear_drop_matches) * w
        k_ob = _dyn_k(match_counts_overall[pb], k_overall_base, k_overall_min, k_linear_drop_matches) * w

        if surf is not None and pd.notna(surf):
            k_sa = _dyn_k(match_counts_surface[key_a], k_surface_base, k_surface_min, k_surface_drop_matches) * w
            k_sb = _dyn_k(match_counts_surface[key_b], k_surface_base, k_surface_min, k_surface_drop_matches) * w
        else:
            k_sa = k_sb = 0.0  # no surface update if surface is unknown

        # Update overall
        ra_o_new = ra_o + k_oa * (s_a - ea_o)
        rb_o_new = rb_o + k_ob * (s_b - eb_o)
        elo_overall[pa], elo_overall[pb] = ra_o_new, rb_o_new
        df.at[i, "A_Elo_Overall_Post"] = ra_o_new
        df.at[i, "B_Elo_Overall_Post"] = rb_o_new

        # Update surface
        if surf is not None and pd.notna(surf):
            ra_s_new = ra_s + k_sa * (s_a - ea_s)
            rb_s_new = rb_s + k_sb * (s_b - eb_s)
            elo_surface[key_a], elo_surface[key_b] = ra_s_new, rb_s_new
            df.at[i, "A_Elo_Surface_Post"] = ra_s_new
            df.at[i, "B_Elo_Surface_Post"] = rb_s_new

            # increase per-surface counts
            match_counts_surface[key_a] += 1
            match_counts_surface[key_b] += 1

        # increase overall counts
        match_counts_overall[pa] += 1
        match_counts_overall[pb] += 1

    df.drop(columns=["_w"], inplace=True)
    return df
