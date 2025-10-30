import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

_BASE_COLORS = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors) + list(plt.cm.Dark2.colors)
_PLAYER_COLOR_MAP = {}

def _get_player_color(name: str):
    if name not in _PLAYER_COLOR_MAP:
        _PLAYER_COLOR_MAP[name] = _BASE_COLORS[len(_PLAYER_COLOR_MAP) % len(_BASE_COLORS)]
    return _PLAYER_COLOR_MAP[name]

def plot_players_elo(
    df: pd.DataFrame, K: int = 6, figsize=(10, 6), surface: str | None = None, min_matches: int = 30, rating: str = "Elo"
):
    title = f"Top Players by {rating} Rating"
    plt.style.use("dark_background")
    mpl.rcParams.update({
        "axes.facecolor":"#111111","figure.facecolor":"#111111",
        "axes.edgecolor":"#CCCCCC","axes.labelcolor":"#FFFFFF",
        "xtick.color":"#BBBBBB","ytick.color":"#BBBBBB",
        "grid.color":"#FFFFFF","text.color":"#FFFFFF","legend.edgecolor":"#333333",
    })

    df = df.copy()

    use_surface = surface is not None and {f"A_{rating}_Surface",f"B_{rating}_Surface"}.issubset(df.columns)
    a_rating_col = f"A_{rating}_Surface" if use_surface else f"A_{rating}_Overall"
    b_rating_col = f"B_{rating}_Surface" if use_surface else f"B_{rating}_Overall"

    if surface is not None and "surface" in df.columns:
        df = df[df["surface"] == surface].copy()

    if not pd.api.types.is_datetime64_any_dtype(df["tourney_date"]):
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    df = df.dropna(subset=["tourney_date"])

    a = df[["tourney_date","player_A_id","player_A_name",a_rating_col]].rename(
        columns={"player_A_id":"player_id","player_A_name":"player_name",a_rating_col:"Elo"}
    )
    b = df[["tourney_date","player_B_id","player_B_name",b_rating_col]].rename(
        columns={"player_B_id":"player_id","player_B_name":"player_name",b_rating_col:"Elo"}
    )
    long = pd.concat([a,b], ignore_index=True)

    match_counts = (long.groupby(["player_id","player_name"])
                         .size()
                         .reset_index(name="matches"))

    per_day = long.groupby(["player_id","player_name","tourney_date"], as_index=False)["Elo"].last()

    # >>> PEAK Elo ranking with min_matches <<<
    career_peak = (
        per_day.groupby(["player_id", "player_name"])[rating]
               .max()
               .reset_index(name=f"{rating}_peak")
               .merge(match_counts, on=["player_id", "player_name"], how="left")
    )
    career_peak = career_peak[career_peak["matches"] >= min_matches] \
                           .sort_values(f"{rating}_peak", ascending=False)

    if career_peak.empty:
        raise ValueError("No players meet the minimum match threshold after filtering.")

    top = career_peak.head(K)
    top_names = set(top["player_name"])
    best_name = top.iloc[0]["player_name"]

    top_data = per_day[per_day["player_name"].isin(top_names)].sort_values(["player_name","tourney_date"])

    fig, ax = plt.subplots(figsize=figsize)
    for name, g in top_data.groupby("player_name"):
        color = _get_player_color(name)
        alpha = 1.0 if name == best_name else 0.3
        lw = 2 if name == best_name else 1.5
        ax.plot(g["tourney_date"], g[rating], label=name, color=color, linewidth=lw, alpha=alpha)

    suffix = f" — {surface}" if surface else ""
    ylabel = f"{rating} Rating (Surface)" if use_surface else f"{rating} Rating (Overall)"
    ax.set_title(f"{title}{suffix}", fontsize=14, weight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(title="Player", loc="upper left", facecolor="#1C1C1C", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show()
    return fig, ax
