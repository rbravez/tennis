""""This module implements the Glicko-1 and Glicko-2 rating systems for player ratings.
It includes functions to calculate both global and surface-specific ratings,
as well as helper functions for scaling and updating ratings."""
import math
import pandas as pd
import numpy as np
from collections import defaultdict

# Glicko 1 Implementation
# Glicko-1 Global Constants
R_INITIAL_G1: float = 1200.0
RD_INITIAL_G1: float = 350.0
Q_FACTOR_G1: float = math.log(10) / 400
C_FACTOR_G1: float = 50.0

# Glicko-1 Helper Functions 

def scale_down_g1(r: float, rd: float) -> tuple:
    """Converts R and RD to the Glicko-1 scale (mu and phi)."""
    # Glicko-1 and Glicko-2 use the same scaling formula
    return Q_FACTOR_G1 * (r - R_INITIAL_G1), Q_FACTOR_G1 * rd

def scale_up_g1(mu: float, phi: float) -> tuple:
    """Converts mu and phi back to the Glicko scale (R and RD)."""
    phi = max(phi, 0.000001)
    r = (mu / Q_FACTOR_G1) + R_INITIAL_G1
    rd = phi / Q_FACTOR_G1
    return r, rd

def g_func_g1(phi: float) -> float:
    """Glicko-1 g function (same as Glicko-2)."""
    return 1 / math.sqrt(1 + (3 * phi**2) / math.pi**2)

def E_func_g1(mu: float, mu_j: float, phi_j: float) -> float:
    """Glicko-1 Expected Score function (same as Glicko-2)."""
    return 1 / (1 + math.exp(-g_func_g1(phi_j) * (mu - mu_j)))

def calculate_d_sq(mu: float, opponents: list) -> float:
    """
    Calculates the variance of the expected outcomes squared ($d^2$) 
    for Glicko-1.
    """
    sum_inv = sum(
        g_func_g1(opp_phi)**2 * E_func_g1(mu, opp_mu, opp_phi) * (1 - E_func_g1(mu, opp_mu, opp_phi))
        for opp_mu, opp_phi, _ in opponents
    )
    return 1 / sum_inv if sum_inv > 0 else float('inf')

# --- Glicko-1 Rating System ---

def update_glicko_1_values(r: float, rd: float, opponents: list, c_factor: float = C_FACTOR_G1) -> tuple:
    """
    Core Glicko-1 update.
    """
    # 1. Increase RD based on uncertainty (inactivity is handled in the main function)
    # This RD update occurs after a rating period (which is one match in this case)
    # The Glicko-1 paper simplifies this to just the C factor for time.
    
    mu, phi = scale_down_g1(r, rd)
    
    opponents_scaled = [(scale_down_g1(opp_r, opp_rd)[0], scale_down_g1(opp_r, opp_rd)[1], score) 
                        for opp_r, opp_rd, score in opponents]
    
    # 2. Compute $d^2$ (variance of the expected outcomes)
    d_sq = calculate_d_sq(mu, opponents_scaled)
    
    # 3. Update mu
    sum_term = sum(
        g_func_g1(opp_phi) * (score - E_func_g1(mu, opp_mu, opp_phi))
        for opp_mu, opp_phi, score in opponents_scaled
    )
    
    # mu_prime = mu + (phi^2 / (1/d^2 + phi^2)) * sum_term
    mu_prime = mu + (phi**2 / (1 / d_sq + phi**2)) * sum_term
    
    # 4. Update phi
    # phi_prime^2 = 1 / (1 / phi^2 + 1 / d^2)
    phi_prime = 1 / math.sqrt(1 / phi**2 + 1 / d_sq)

    # 5. Convert back to Glicko scale
    new_r, new_rd = scale_up_g1(mu_prime, phi_prime)
    
    # Constrain RD to reasonable bounds
    new_rd = np.clip(new_rd, 30, 500)
    
    return new_r, new_rd


def calculate_global_glicko_1(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Global historical evolution Glicko-1 ratings for players."""
    
    df = df.sort_values(by=['tourney_date', 'match_num'] if 'match_num' in df.columns else ['tourney_date']).reset_index(drop=True)
    date_series = pd.to_datetime(df['tourney_date'].astype(str), format='%Y%m%d')

    glicko_rating = defaultdict(lambda: R_INITIAL_G1)
    glicko_rd = defaultdict(lambda: RD_INITIAL_G1)
    last_played = defaultdict(lambda: None)

    df['A_Glicko1_Rating'] = float('nan')
    df['B_Glicko1_Rating'] = float('nan')
    df['A_Glicko1_RD'] = float('nan')
    df['B_Glicko1_RD'] = float('nan')

    error_count = 0
    for index, row in df.iterrows():
        if pd.isna(row.get('player_A_id')) or pd.isna(row.get('player_B_id')) or pd.isna(row.get('target')) or pd.isna(row.get('tourney_date')):
            error_count += 1
            continue
            
        player_a = row['player_A_id']
        player_b = row['player_B_id']
        target = row['target']
        current_date = date_series.iat[index] 
        
        # RD Inflation for Inactivity
        for player in [player_a, player_b]:
            r, rd = glicko_rating[player], glicko_rd[player]
            
            # Glicko-1 RD inflation: RD_new = min(RD_INITIAL, sqrt(RD_old^2 + c^2 * t))
            # Inactivity is modeled as RD_new = sqrt(RD_old^2 + C_FACTOR_G1^2 * periods)
            if last_played[player] is not None:
                days_inactive = (current_date - last_played[player]).days
                # A single match period is an arbitrary choice, 90 days for Glicko-2
                # Let's use days_inactive/90, and C_FACTOR_G1 is used as the volatility * 90 days
                periods = days_inactive / INACTIVITY_PERIOD_DAYS
                
                if periods > 0:
                    rd_new = math.sqrt(rd**2 + (C_FACTOR_G1**2 * periods))
                    # Clamp to max RD
                    glicko_rd[player] = min(rd_new, RD_INITIAL_G1)
        
        # Get current values
        R_A = glicko_rating[player_a]
        RD_A = glicko_rd[player_a]
        
        R_B = glicko_rating[player_b]
        RD_B = glicko_rd[player_b]
        
        # Log current values
        df.at[index, 'A_Glicko1_Rating'] = R_A
        df.at[index, 'B_Glicko1_Rating'] = R_B
        df.at[index, 'A_Glicko1_RD'] = RD_A
        df.at[index, 'B_Glicko1_RD'] = RD_B
        
        # Update ratings
        R_A_new, RD_A_new = update_glicko_1_values(
            R_A, RD_A, [(R_B, RD_B, target)] # Opponent R, RD, and outcome
        )
        R_B_new, RD_B_new = update_glicko_1_values(
            R_B, RD_B, [(R_A, RD_A, 1 - target)] # Opponent R, RD, and outcome
        )
        
        # Store new values
        glicko_rating[player_a] = R_A_new
        glicko_rd[player_a] = RD_A_new
        
        glicko_rating[player_b] = R_B_new
        glicko_rd[player_b] = RD_B_new
        
        last_played[player_a] = current_date
        last_played[player_b] = current_date
    
    if error_count > 0:
        print(f"Warning: Skipped {error_count} rows in c_general_glicko_1 due to missing values or date.")
        
    return df


def c_surface_glicko_1(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Surface specific historical evolution Glicko-1 ratings."""
    
    df = df.sort_values(by=['tourney_date', 'match_num'] if 'match_num' in df.columns else ['tourney_date']).reset_index(drop=True)
    date_series = pd.to_datetime(df['tourney_date'].astype(str), format='%Y%m%d')

    glicko_rating = defaultdict(lambda: R_INITIAL_G1)
    glicko_rd = defaultdict(lambda: RD_INITIAL_G1)
    last_played = defaultdict(lambda: None)
    
    df['A_Glicko1_Surface_Rating'] = float('nan')
    df['B_Glicko1_Surface_Rating'] = float('nan')
    df['A_Glicko1_Surface_RD'] = float('nan')
    df['B_Glicko1_Surface_RD'] = float('nan')

    error_count = 0
    for index, row in df.iterrows():
        if pd.isna(row.get('player_A_id')) or pd.isna(row.get('player_B_id')) or pd.isna(row.get('target')) or pd.isna(row.get('surface')) or pd.isna(row.get('tourney_date')):
            error_count += 1
            continue
        
        player_a = row['player_A_id']
        player_b = row['player_B_id']
        surface = row['surface']
        target = row['target']
        current_date = date_series.iat[index]

        key_a = (player_a, surface)
        key_b = (player_b, surface)
        
        # RD Inflation for Inactivity
        for key in [key_a, key_b]:
            r, rd = glicko_rating[key], glicko_rd[key]
            
            if last_played[key] is not None:
                days_inactive = (current_date - last_played[key]).days
                periods = days_inactive / INACTIVITY_PERIOD_DAYS
                
                if periods > 0:
                    rd_new = math.sqrt(rd**2 + (C_FACTOR_G1**2 * periods))
                    glicko_rd[key] = min(rd_new, RD_INITIAL_G1)
        
        # Get current values
        R_A = glicko_rating[key_a]
        RD_A = glicko_rd[key_a]
        
        R_B = glicko_rating[key_b]
        RD_B = glicko_rd[key_b]
        
        # Log current values
        df.at[index, 'A_Glicko1_Surface_Rating'] = R_A
        df.at[index, 'B_Glicko1_Surface_Rating'] = R_B
        df.at[index, 'A_Glicko1_Surface_RD'] = RD_A
        df.at[index, 'B_Glicko1_Surface_RD'] = RD_B
        
        # Update ratings
        R_A_new, RD_A_new = update_glicko_1_values(
            R_A, RD_A, [(R_B, RD_B, target)]
        )
        R_B_new, RD_B_new = update_glicko_1_values(
            R_B, RD_B, [(R_A, RD_A, 1 - target)]
        )
        
        # Store new values
        glicko_rating[key_a] = R_A_new
        glicko_rd[key_a] = RD_A_new
        
        glicko_rating[key_b] = R_B_new
        glicko_rd[key_b] = RD_B_new

        last_played[key_a] = current_date
        last_played[key_b] = current_date
    
    if error_count > 0:
        print(f"Warning: Skipped {error_count} rows in c_surface_glicko_1 due to missing values or date.")
        
    return df

# Glicko 2 Implementation
# Glicko-2 Global Constants
R_INITIAL: float = 1200.0
RD_INITIAL: float = 350.0
SIGMA_INITIAL: float = 0.5
TAU: float = 0.3
Q_FACTOR: float = math.log(10) / 400
INACTIVITY_PERIOD_DAYS: int = 90

# Glicko-2 Helper Functions

def scale_down(r: float, rd: float) -> tuple:
    """Converts R and RD to the Glicko-2 scale (mu and phi)."""
    mu = Q_FACTOR * (r - R_INITIAL)
    phi = Q_FACTOR * rd
    return mu, phi

def scale_up(mu: float, phi: float) -> tuple:
    """Converts mu and phi back to the Glicko scale (R and RD)."""
    phi = max(phi, 0.000001)
    r = (mu / Q_FACTOR) + R_INITIAL
    rd = phi / Q_FACTOR
    return r, rd

def g_func(phi: float) -> float:
    """Glicko-2 g function."""
    return 1 / math.sqrt(1 + (3 * phi**2) / math.pi**2)

def E_func(mu: float, mu_j: float, phi_j: float) -> float:
    """Glicko-2 Expected Score function."""
    return 1 / (1 + math.exp(-g_func(phi_j) * (mu - mu_j)))

def calculate_v(mu: float, phi: float, opponents: list) -> float:
    """Calculates the variance of the expected outcomes (v)."""
    sum_inv = sum(
        g_func(opp_phi)**2 * E_func(mu, opp_mu, opp_phi) * (1 - E_func(mu, opp_mu, opp_phi))
        for opp_mu, opp_phi, _ in opponents
    )
    return 1 / sum_inv if sum_inv > 0 else float('inf')

def calculate_delta(mu: float, v: float, opponents: list) -> float:
    """Calculates Delta (the estimated improvement in skill)."""
    return v * sum(
        g_func(opp_phi) * (score - E_func(mu, opp_mu, opp_phi))
        for opp_mu, opp_phi, score in opponents
    )

def update_glicko2_values(r: float, rd: float, sigma: float, outcomes: list, tau: float = TAU) -> tuple:
    """
    Core Glicko-2 update with numerical stability fixes.
    """
    mu, phi = scale_down(r, rd)
    
    # 1. Update phi for the rating period
    phi_star = math.sqrt(phi**2 + sigma**2)
    
    # 2. Compute v and delta
    opponents_scaled = [(scale_down(opp_r, opp_rd)[0], scale_down(opp_r, opp_rd)[1], score) 
                        for opp_r, opp_rd, score in outcomes]

    v = calculate_v(mu, phi_star, opponents_scaled)
    delta = calculate_delta(mu, v, opponents_scaled)
    
    # 3. Determine New Volatility with numerical stability (Newton-Raphson-like/Illinois)
    
    def f(x, mu_val, phi_star_val, delta_val, v_val, A_val, tau_val):
        # Clamp x to prevent overflow
        x_clamped = np.clip(x, -700, 700)
        
        try:
            exp_x = math.exp(x_clamped)
        except OverflowError:
            exp_x = math.exp(700) if x_clamped > 0 else math.exp(-700)
        
        # Add small epsilon to denominators to prevent division by zero
        epsilon = 1e-10
        denominator = 2 * (phi_star_val**2 + v_val + exp_x)**2 + epsilon
        
        term1 = exp_x * (delta_val**2 - phi_star_val**2 - v_val - exp_x) / denominator
        term2 = (x_clamped - A_val) / (tau_val**2)
        return term1 - term2

    A = math.log(sigma**2)
    
    # Better initial B value
    if delta**2 > (phi_star**2 + v):
        B = math.log(delta**2 - phi_star**2 - v)
    else:
        # Search for B that makes f(B) negative
        k = 1
        B = A - k * tau
        # Pass constants to f to avoid relying on scope variables
        while f(B, mu, phi_star, delta, v, A, tau) >= 0 and k < 100:
            k += 1
            B = A - k * tau
    
    # Illinois algorithm (more stable than pure secant method)
    fA = f(A, mu, phi_star, delta, v, A, tau)
    fB = f(B, mu, phi_star, delta, v, A, tau)
    
    epsilon = 0.000001
    max_iterations = 100
    
    for iteration in range(max_iterations):
        if abs(B - A) < epsilon:
            break
        
        # Check for degenerate case
        if abs(fB - fA) < 1e-12:
            break
            
        # Illinois algorithm step
        C = A - fA * (B - A) / (fB - fA)
        
        # Ensure C is bounded
        C = np.clip(C, min(A, B) - 10, max(A, B) + 10)
        
        fC = f(C, mu, phi_star, delta, v, A, tau)
        
        if fC * fB < 0:
            A = B
            fA = fB
        else:
            # Illinois modification
            # Avoid division by zero, though fC * fB < 0 check should cover most cases
            if abs(fB + fC) < 1e-12:
                 fA = fA / 2
            else:
                 fA = fA * fB / (fB + fC)

        
        B = C
        fB = fC
    
    new_log_sigma_sq = B
    # Clamp to prevent math domain error or extremely large/small values
    new_sigma = math.sqrt(math.exp(np.clip(new_log_sigma_sq, -20, 5)))
    
    # Constrain sigma to reasonable bounds
    new_sigma = np.clip(new_sigma, 0.001, 0.5)
    
    # 4. Update Phi (RD) and Mu (Rating)
    # phi_prime is the new phi before period update
    phi_prime = 1 / math.sqrt(1 / phi_star**2 + 1 / v)
    mu_prime = mu + phi_prime**2 * delta / v
    
    # 5. Convert back to Glicko scale
    new_r, new_rd = scale_up(mu_prime, phi_prime)
    
    # Constrain RD to reasonable bounds
    new_rd = np.clip(new_rd, 30, 500)
    
    return new_r, new_rd, new_sigma

# Glicko 2 Rating System

def c_general_glicko_2(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Global historical evolution Glicko-2 ratings for players."""
    # Already sorted by tourney_date in prepare_data_for_modeling
    df = df.sort_values(by=['tourney_date', 'match_num'] if 'match_num' in df.columns else ['tourney_date']).reset_index(drop=True)
    date_series = pd.to_datetime(df['tourney_date'].astype(str), format='%Y%m%d')

    glicko_rating = defaultdict(lambda: R_INITIAL)
    glicko_rd = defaultdict(lambda: RD_INITIAL)
    glicko_volatility = defaultdict(lambda: SIGMA_INITIAL)
    last_played = defaultdict(lambda: None)

    df['A_Glicko_Rating'] = float('nan')
    df['B_Glicko_Rating'] = float('nan')
    df['A_Glicko_RD'] = float('nan')
    df['B_Glicko_RD'] = float('nan')
    df['A_Glicko_Sigma'] = float('nan')
    df['B_Glicko_Sigma'] = float('nan')

    error_count = 0
    for index, row in df.iterrows():
        # Check for missing values
        if pd.isna(row.get('player_A_id')) or pd.isna(row.get('player_B_id')) or pd.isna(row.get('target')) or pd.isna(row.get('tourney_date')):
            error_count += 1
            continue
            
        player_a = row['player_A_id']
        player_b = row['player_B_id']
        target = row['target']
        current_date = date_series.iat[index]
        
        # RD Inflation for Inactivity
        for player in [player_a, player_b]:
            if last_played[player] is not None:
                days_inactive = (current_date - last_played[player]).days
                periods = days_inactive // INACTIVITY_PERIOD_DAYS
                
                if periods > 0:
                    r, rd = glicko_rating[player], glicko_rd[player]
                    sigma = glicko_volatility[player]
                    
                    mu, phi = scale_down(r, rd)
                    
                    for _ in range(periods):
                        # Glicko-2 Step 1: RD update for an interval of no play
                        phi = math.sqrt(phi**2 + sigma**2)
                    
                    glicko_rating[player], glicko_rd[player] = scale_up(mu, phi)
        
        # Get current values
        R_A = glicko_rating[player_a]
        RD_A = glicko_rd[player_a]
        sigma_A = glicko_volatility[player_a]
        
        R_B = glicko_rating[player_b]
        RD_B = glicko_rd[player_b]
        sigma_B = glicko_volatility[player_b]
        
        # Log current values
        df.at[index, 'A_Glicko_Rating'] = R_A
        df.at[index, 'B_Glicko_Rating'] = R_B
        df.at[index, 'A_Glicko_RD'] = RD_A
        df.at[index, 'B_Glicko_RD'] = RD_B
        df.at[index, 'A_Glicko_Sigma'] = sigma_A
        df.at[index, 'B_Glicko_Sigma'] = sigma_B
        
        # Update ratings
        R_A_new, RD_A_new, sigma_A_new = update_glicko2_values(
            R_A, RD_A, sigma_A, [(R_B, RD_B, target)] # Opponent R, RD, and outcome
        )
        R_B_new, RD_B_new, sigma_B_new = update_glicko2_values(
            R_B, RD_B, sigma_B, [(R_A, RD_A, 1 - target)] # Opponent R, RD, and outcome
        )
        
        # Store new values
        glicko_rating[player_a] = R_A_new
        glicko_rd[player_a] = RD_A_new
        glicko_volatility[player_a] = sigma_A_new
        
        glicko_rating[player_b] = R_B_new
        glicko_rd[player_b] = RD_B_new
        glicko_volatility[player_b] = sigma_B_new
        
        last_played[player_a] = current_date
        last_played[player_b] = current_date
    
    if error_count > 0:
        print(f"Warning: Skipped {error_count} rows in c_general_glicko_2 due to missing values or date.")
        
    return df


def c_surface_glicko_2(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Surface specific historical evolution Glicko-2 ratings."""
    # Already sorted by tourney_date in prepare_data_for_modeling
    
    df = df.sort_values(by=['tourney_date', 'match_num'] if 'match_num' in df.columns else ['tourney_date']).reset_index(drop=True)
    date_series = pd.to_datetime(df['tourney_date'].astype(str), format='%Y%m%d')

    glicko_rating = defaultdict(lambda: R_INITIAL)
    glicko_rd = defaultdict(lambda: RD_INITIAL)
    glicko_volatility = defaultdict(lambda: SIGMA_INITIAL)
    last_played = defaultdict(lambda: None)
    
    df['A_Glicko_Surface_Rating'] = float('nan')
    df['B_Glicko_Surface_Rating'] = float('nan')
    df['A_Glicko_Surface_RD'] = float('nan')
    df['B_Glicko_Surface_RD'] = float('nan')
    df['A_Glicko_Surface_Sigma'] = float('nan')
    df['B_Glicko_Surface_Sigma'] = float('nan')

    error_count = 0
    for index, row in df.iterrows():
        # Check for missing values
        if pd.isna(row.get('player_A_id')) or pd.isna(row.get('player_B_id')) or pd.isna(row.get('target')) or pd.isna(row.get('surface')) or pd.isna(row.get('tourney_date')):
            error_count += 1
            continue
        
        player_a = row['player_A_id']
        player_b = row['player_B_id']
        surface = row['surface']
        target = row['target']
        current_date = date_series.iat[index]
        
        # Fix: Move debug print to after variable assignment
        if index < 5:
            print(f"\nRow {index}: A={player_a}, B={player_b}, Surface={surface}, Target={target}")
            print(f"Initial A: R={glicko_rating[(player_a, surface)]:.2f}, RD={glicko_rd[(player_a, surface)]:.2f}")
            print(f"Initial B: R={glicko_rating[(player_b, surface)]:.2f}, RD={glicko_rd[(player_b, surface)]:.2f}")

        key_a = (player_a, surface)
        key_b = (player_b, surface)
        
        # RD Inflation for Inactivity
        for key in [key_a, key_b]:
            if last_played[key] is not None:
                days_inactive = (current_date - last_played[key]).days
                periods = days_inactive // INACTIVITY_PERIOD_DAYS
                
                if periods > 0:
                    r, rd = glicko_rating[key], glicko_rd[key]
                    sigma = glicko_volatility[key]
                    
                    mu, phi = scale_down(r, rd)
                    for _ in range(periods):
                        phi = math.sqrt(phi**2 + sigma**2)
                    
                    glicko_rating[key], glicko_rd[key] = scale_up(mu, phi)
        
        # Get current values
        R_A = glicko_rating[key_a]
        RD_A = glicko_rd[key_a]
        sigma_A = glicko_volatility[key_a]
        
        R_B = glicko_rating[key_b]
        RD_B = glicko_rd[key_b]
        sigma_B = glicko_volatility[key_b]
        
        # Log current values
        df.at[index, 'A_Glicko_Surface_Rating'] = R_A
        df.at[index, 'B_Glicko_Surface_Rating'] = R_B
        df.at[index, 'A_Glicko_Surface_RD'] = RD_A
        df.at[index, 'B_Glicko_Surface_RD'] = RD_B
        df.at[index, 'A_Glicko_Surface_Sigma'] = sigma_A
        df.at[index, 'B_Glicko_Surface_Sigma'] = sigma_B
        
        # Update ratings
        R_A_new, RD_A_new, sigma_A_new = update_glicko2_values(
            R_A, RD_A, sigma_A, [(R_B, RD_B, target)]
        )
        R_B_new, RD_B_new, sigma_B_new = update_glicko2_values(
            R_B, RD_B, sigma_B, [(R_A, RD_A, 1 - target)]
        )
        
        # Store new values
        glicko_rating[key_a] = R_A_new
        glicko_rd[key_a] = RD_A_new
        glicko_volatility[key_a] = sigma_A_new
        
        glicko_rating[key_b] = R_B_new
        glicko_rd[key_b] = RD_B_new
        glicko_volatility[key_b] = sigma_B_new
        
        if index < 5:
            # Note: The values at `df.at[index, ...]` were the *prior* ratings
            print(f"Assigned A_Rating at index {index}: {df.at[index, 'A_Glicko_Surface_Rating']:.2f}")
            print(f"New A_Rating for next match: {R_A_new:.2f}")

        last_played[key_a] = current_date
        last_played[key_b] = current_date
    
    if error_count > 0:
        print(f"Warning: Skipped {error_count} rows in c_surface_glicko_2 due to missing values or date.")
        
    return df