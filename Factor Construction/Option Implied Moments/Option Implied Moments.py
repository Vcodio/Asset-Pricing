# ================================================================
# Triple Sort Analysis: FSKEW, FVOL, and 27 Portfolios
# WARNING: THIS ANALYSIS SUFFERS FROM SURVIVORSHIP BIAS
# ================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import time
from pandas_datareader import data as pdr
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning)

# Get the script's directory and construct paths relative to it
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
data_dir = os.path.join(base_dir, "Data")
tickers_csv_path = os.path.join(data_dir, "Tickers US.csv")

# Create cache directories
vix_skew_index_cache_dir = os.path.join(data_dir, "VIX_SKEW_Index_Cache")
us_constituents_cache_dir = os.path.join(data_dir, "US_Const_Cache")
analysis_results_dir = os.path.join(base_dir, "analysis_results")
failed_tickers_cache_path = os.path.join(data_dir, "failed_tickers_cache.txt")

os.makedirs(vix_skew_index_cache_dir, exist_ok=True)
os.makedirs(us_constituents_cache_dir, exist_ok=True)
os.makedirs(analysis_results_dir, exist_ok=True)

# --------------------------
# PARAMETERS
# --------------------------
start_date = "1995-01-01"
end_date = "2025-12-31"
window_sizes = [21]
data_cache_dir = vix_skew_index_cache_dir

# In-sample / Out-of-sample split
USE_FULL_SAMPLE = True  # If True, use full sample (no split). If False, use in-sample/out-of-sample split
split_date = None  # None to use percentage split, or specify date like "2010-01-01"
in_sample_pct = 0.7  # Fraction for in-sample (only used if split_date is None and USE_FULL_SAMPLE is False)

# Parallel processing settings
N_JOBS = -1  # Use all available CPUs

# CRSP market returns cache file
crsp_cache_file = os.path.join(us_constituents_cache_dir, "CRSP_Market_Daily.csv")

# --------------------------
# FUNCTIONS
# --------------------------

def fetch_data(ticker, start, end, refresh=False, cache_dir=None, max_retries=3, retry_delay=5):
    """Fetch Close prices with robust caching and rate limit handling."""
    if cache_dir is None:
        cache_dir = data_cache_dir
    
    cache_path = os.path.join(cache_dir, f"{ticker}.csv")
    price_column = 'Close'

    if os.path.exists(cache_path) and not refresh:
        try:
            file_size = os.path.getsize(cache_path)
            if file_size == 0:
                os.remove(cache_path)
                print(f"WARNING: Empty cache file for {ticker}. Will redownload.")
            elif file_size > 50 * 1024 * 1024:  # 50MB
                print(f"WARNING: Cache file for {ticker} is suspiciously large ({file_size / 1024 / 1024:.1f}MB). Skipping.")
                os.remove(cache_path)
            else:
                series = pd.read_csv(
                    cache_path,
                    index_col=0,
                    parse_dates=[0],
                    header=None,
                    names=['Date', price_column]
                ).iloc[:, 0]
                if not series.empty:
                    series.name = ticker
                    series.index = pd.to_datetime(series.index)
                    series = series.loc[start:end]
                    if not series.empty:
                        return series, True
                    else:
                        print(f"WARNING: Cached data for {ticker} has no data in range {start} to {end}. Will redownload.")
                        os.remove(cache_path)
        except Exception as e:
            print(f"WARNING: Error reading cache file for {ticker}: {e}")
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            except:
                pass

    # Download with retry logic
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            
            if df.empty or price_column not in df.columns:
                print(f"WARNING: No valid '{price_column}' data for {ticker}")
                return pd.Series(dtype=float), False

            df[price_column].to_csv(cache_path, header=False)
            series = df[price_column]
            series.index = pd.to_datetime(series.index)
            series.name = ticker
            return series, False
            
        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = any(keyword in error_msg for keyword in [
                'rate limit', '429', 'too many requests', 'forbidden', 
                'unauthorized', 'timeout', 'connection'
            ])
            
            if is_rate_limit and attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"\nWARNING: Rate limit or connection error for {ticker} (attempt {attempt + 1}/{max_retries})")
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            else:
                if is_rate_limit:
                    print(f"ERROR: Rate limited for {ticker} after {max_retries} attempts. Skipping.")
                else:
                    print(f"ERROR: Failed to download {ticker}: {e}")
                return pd.Series(dtype=float), False
    
    return pd.Series(dtype=float), False

def calculate_index_changes(index_series):
    """Daily percentage changes for a given index."""
    if isinstance(index_series, pd.DataFrame):
        if index_series.shape[1] > 1:
            raise ValueError("Input must be a Series or a single-column DataFrame.")
        index_series = index_series.squeeze()

    index_series = index_series.dropna()
    df = index_series.to_frame(name='Index')
    df['Index_Change'] = df['Index'].pct_change()
    return df.dropna()

def forward_returns(prices, window):
    """Forward returns over the next 'window' days."""
    returns = prices.pct_change(periods=window).shift(-window)
    return returns

def eiv_beta_calculator(stock_returns, index_changes, window):
    """Calculates rolling EIV betas for all stocks."""
    aligned_data = stock_returns.join(index_changes, how='inner')

    if aligned_data.empty:
        return pd.DataFrame()

    beta_series_list = []
    index_changes_column_name = index_changes.name
    rolling_var_x = aligned_data[index_changes_column_name].rolling(window=window).var()
    denominator = rolling_var_x / 2
    denominator[denominator.abs() < 1e-9] = np.nan

    tickers_to_process = aligned_data.columns.drop(index_changes_column_name)
    for ticker in tqdm(tickers_to_process, desc=f"Calculating betas (window={window})", unit="ticker", leave=False):
        stock_series = aligned_data[ticker]
        index_series = aligned_data[index_changes_column_name]
        rolling_cov_yx = stock_series.rolling(window=window).cov(index_series)
        betas = rolling_cov_yx / denominator
        betas.name = ticker
        beta_series_list.append(betas)

    betas_df = pd.concat(beta_series_list, axis=1)
    betas_df = betas_df.dropna(how='all')
    return betas_df

def calculate_cagr(cumulative_returns, years):
    """
    Calculates the Compound Annual Growth Rate from cumulative returns.
    
    cumulative_returns should be a Series of cumulative values (e.g., starts at 1.0, ends at 1.5 for 50% return).
    CAGR = (final_value / initial_value)^(1/years) - 1
    """
    if cumulative_returns.empty:
        return np.nan
    
    initial_value = cumulative_returns.iloc[0]
    final_value = cumulative_returns.iloc[-1]
    
    if initial_value <= 0 or final_value <= 0:
        return np.nan
    
    if years <= 0:
        return np.nan
    
    return (final_value / initial_value)**(1/years) - 1

def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown from cumulative returns."""
    if cumulative_returns.empty:
        return np.nan
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """Calculate annualized Sharpe ratio."""
    if returns.empty or returns.std() == 0:
        return np.nan
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
    return sharpe

def calculate_calmar_ratio(cumulative_returns, years):
    """Calculate Calmar ratio (CAGR / Max Drawdown)."""
    cagr = calculate_cagr(cumulative_returns, years)
    max_dd = abs(calculate_max_drawdown(cumulative_returns))
    if max_dd == 0 or np.isnan(max_dd) or np.isnan(cagr):
        return np.nan
    return cagr / max_dd

def calculate_portfolio_summary_stats(returns_df, cumulative_returns_df, period_name, window):
    """
    Calculate summary statistics for portfolios.
    
    Returns a dictionary with statistics for each portfolio.
    """
    stats = {}
    years = (returns_df.index[-1] - returns_df.index[0]).days / 365.25 if not returns_df.empty else 1
    
    for col in returns_df.columns:
        port_returns = returns_df[col].dropna()
        port_cumulative = cumulative_returns_df[col].dropna()
        
        if port_returns.empty or port_cumulative.empty:
            continue
        
        port_stats = {
            'Mean (Annualized)': port_returns.mean() * 252 * 100,  # Convert to annualized percentage
            'Volatility (Annualized)': port_returns.std() * np.sqrt(252) * 100,
            'Variance (Annualized)': (port_returns.std() * np.sqrt(252))**2,
            'Max Drawdown': calculate_max_drawdown(port_cumulative) * 100,
            'CAGR': calculate_cagr(port_cumulative, years) * 100,
            'Sharpe Ratio': calculate_sharpe_ratio(port_returns),
            'Calmar Ratio': calculate_calmar_ratio(port_cumulative, years),
            'Min Return': port_returns.min() * 100,
            'Max Return': port_returns.max() * 100,
            'Skewness': port_returns.skew(),
            'Kurtosis': port_returns.kurtosis(),
            'Total Return': (port_cumulative.iloc[-1] - 1) * 100
        }
        
        stats[col] = port_stats
    
    return stats

def load_crsp_market_returns(start_date, end_date, cache_file):
    """Load CRSP market returns from Fama-French data."""
    if os.path.exists(cache_file):
        try:
            print("Loading CRSP market returns from cache...")
            cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            cached_data = cached_data.squeeze()
            cached_data.name = 'CRSP_Market'
            cached_data = cached_data.loc[start_date:end_date]
            
            if not cached_data.empty:
                print(f"Loaded {len(cached_data)} days of CRSP data from cache")
                return cached_data.dropna()
        except Exception as e:
            print(f"WARNING: Error reading CRSP cache: {e}. Will re-download.")
    
    # Download Fama-French DAILY data
    print("Downloading Fama-French daily data using pandas_datareader...")
    try:
        ff_data = pdr.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=start_date, end=end_date)
        
        if isinstance(ff_data, dict):
            if 0 in ff_data:
                ff_df = ff_data[0]
            elif len(ff_data) > 0:
                ff_df = list(ff_data.values())[0]
            else:
                raise ValueError("Empty dictionary returned from pandas_datareader")
        elif isinstance(ff_data, list):
            ff_df = ff_data[0]
        elif isinstance(ff_data, pd.DataFrame):
            ff_df = ff_data
        else:
            raise ValueError(f"Unexpected data type from pandas_datareader: {type(ff_data)}")
        
        if hasattr(ff_df.index, 'to_timestamp'):
            ff_df.index = ff_df.index.to_timestamp()
        
        if not isinstance(ff_df.index, pd.DatetimeIndex):
            ff_df.index = pd.to_datetime(ff_df.index)
        
        ff_df = ff_df.loc[start_date:end_date]
        
        if ff_df.empty:
            print(f"WARNING: No Fama-French daily data in range {start_date} to {end_date}")
            return None
        
        crsp_data = (ff_df['Mkt-RF'] + ff_df['RF']) / 100.0
        crsp_data.name = 'CRSP_Market'
        crsp_data = crsp_data.dropna()
        
        try:
            crsp_data.to_csv(cache_file, header=True)
            print(f"Cached CRSP daily data to {cache_file} ({len(crsp_data)} days)")
        except Exception as e:
            print(f"WARNING: Could not cache CRSP data: {e}")
        
        return crsp_data
        
    except Exception as e:
        print(f"ERROR: Could not load Fama-French daily data: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_failed_tickers_cache():
    """Load the list of failed tickers from cache file."""
    if os.path.exists(failed_tickers_cache_path):
        try:
            with open(failed_tickers_cache_path, 'r') as f:
                failed_tickers = [line.strip() for line in f.readlines() if line.strip()]
            return set(failed_tickers)
        except Exception as e:
            print(f"WARNING: Could not load failed tickers cache: {e}")
            return set()
    return set()

def save_failed_tickers_cache(failed_tickers):
    """Save the list of failed tickers to cache file."""
    try:
        existing_failed = load_failed_tickers_cache()
        all_failed = existing_failed | set(failed_tickers)
        with open(failed_tickers_cache_path, 'w') as f:
            for ticker in sorted(all_failed):
                f.write(f"{ticker}\n")
    except Exception as e:
        print(f"WARNING: Could not save failed tickers cache: {e}")

def _calculate_single_stock_beta(stock, aligned_stocks, factor_col, control_cols, window):
    """Helper function to calculate regression beta for a single stock."""
    stock_idx = aligned_stocks.columns.get_loc(stock)
    factor_idx = aligned_stocks.columns.get_loc(factor_col)
    control_idxs = [aligned_stocks.columns.get_loc(col) for col in control_cols]
    
    data_matrix = aligned_stocks.values
    n_dates = len(data_matrix)
    betas = np.full(n_dates, np.nan, dtype=np.float64)
    
    y_all = data_matrix[:, stock_idx]
    factor_all = data_matrix[:, factor_idx]
    control_all = data_matrix[:, control_idxs]
    
    for i in range(window, n_dates):
        start_idx = i - window
        end_idx = i + 1
        
        y_window = y_all[start_idx:end_idx]
        factor_window = factor_all[start_idx:end_idx]
        control_window = control_all[start_idx:end_idx, :]
        
        valid_mask = (
            ~np.isnan(y_window) & 
            ~np.isnan(factor_window) & 
            ~np.isnan(control_window).any(axis=1)
        )
        
        n_valid = valid_mask.sum()
        if n_valid < max(5, window * 0.5):
            continue
        
        y_valid = y_window[valid_mask]
        factor_valid = factor_window[valid_mask]
        control_valid = control_window[valid_mask, :]
        
        n_valid = len(y_valid)
        X_design = np.empty((n_valid, 1 + len(control_cols) + 1), dtype=np.float64)
        X_design[:, 0] = 1.0  # Intercept
        X_design[:, 1:-1] = control_valid
        X_design[:, -1] = factor_valid
        
        try:
            if n_valid < 50:
                XTX = X_design.T @ X_design
                XTy = X_design.T @ y_valid
                try:
                    coeffs = np.linalg.solve(XTX, XTy)
                    betas[i] = coeffs[-1]
                except np.linalg.LinAlgError:
                    coeffs, _, rank, _ = np.linalg.lstsq(X_design, y_valid, rcond=None)
                    if rank >= len(coeffs):
                        betas[i] = coeffs[-1]
            else:
                coeffs, _, rank, _ = np.linalg.lstsq(X_design, y_valid, rcond=None)
                if rank >= len(coeffs):
                    betas[i] = coeffs[-1]
        except (np.linalg.LinAlgError, ValueError):
            pass
    
    betas_series = pd.Series(betas, index=aligned_stocks.index, name=stock)
    return betas_series

def calculate_regression_betas(stock_returns, factor_returns, control_returns, window, n_jobs=-1):
    """Calculate betas using time-series regression with controls."""
    if isinstance(control_returns, pd.Series):
        control_returns = control_returns.to_frame()
    
    factor_data = pd.concat([factor_returns, control_returns], axis=1, join='inner').dropna()
    if factor_data.empty:
        return pd.DataFrame()
    
    factor_col = factor_data.columns[0]
    control_cols = factor_data.columns[1:].tolist()
    
    aligned_stocks = stock_returns.join(factor_data, how='inner')
    if aligned_stocks.empty:
        return pd.DataFrame()
    
    stock_cols = stock_returns.columns.tolist()
    
    print(f"Calculating regression betas for {len(stock_cols)} stocks (parallel processing, n_jobs={n_jobs})...")
    
    batch_size = max(1, len(stock_cols) // 20)
    pbar = tqdm(total=len(stock_cols), desc="Calculating regression betas", unit="stock", leave=False)
    
    beta_series_list = []
    for i in range(0, len(stock_cols), batch_size):
        batch = stock_cols[i:i+batch_size]
        batch_results = Parallel(n_jobs=n_jobs, backend='multiprocessing', verbose=0)(
            delayed(_calculate_single_stock_beta)(
                stock, aligned_stocks, factor_col, control_cols, window
            )
            for stock in batch
        )
        beta_series_list.extend(batch_results)
        pbar.update(len(batch))
    
    pbar.close()
    
    betas_df = pd.concat(beta_series_list, axis=1)
    return betas_df.dropna(how='all')

def triple_sort_portfolios(betas_dict, fwd_returns_df, n_groups=3):
    """
    Create factor-mimicking portfolios using triple independent sort (Market, VOL, SKEW).
    
    Creates 3×3×3 = 27 portfolios and extracts FSKEW and FVOL factor-mimicking portfolios.
    """
    # Get common dates across all beta DataFrames
    common_dates = None
    for beta_df in betas_dict.values():
        if common_dates is None:
            common_dates = beta_df.index
        else:
            common_dates = common_dates.intersection(beta_df.index)
    
    common_dates = common_dates.intersection(fwd_returns_df.index)
    
    if len(common_dates) == 0:
        return {}
    
    # Initialize portfolio returns DataFrame
    portfolio_returns = pd.DataFrame(index=common_dates)
    portfolio_assignments = {}
    portfolio_id = 0
    
    # Create all 27 portfolios
    for mkt_group in range(1, n_groups + 1):
        for vol_group in range(1, n_groups + 1):
            for skew_group in range(1, n_groups + 1):
                portfolio_id += 1
                portfolio_assignments[(mkt_group, vol_group, skew_group)] = portfolio_id
                portfolio_returns[portfolio_id] = np.nan
    
    # For each date, assign stocks to portfolios and calculate returns
    for date in tqdm(common_dates, desc="Triple sort portfolios", leave=False):
        mkt_betas = betas_dict['Market'].loc[date].dropna()
        vol_betas = betas_dict['VOL'].loc[date].dropna()
        skew_betas = betas_dict['SKEW'].loc[date].dropna()
        fwd_returns = fwd_returns_df.loc[date].dropna()
        
        common_stocks = (set(mkt_betas.index) & set(vol_betas.index) & 
                        set(skew_betas.index) & set(fwd_returns.index))
        
        if len(common_stocks) < n_groups * n_groups:
            continue
        
        common_stocks_list = list(common_stocks)
        
        mkt_betas = mkt_betas.loc[common_stocks_list]
        vol_betas = vol_betas.loc[common_stocks_list]
        skew_betas = skew_betas.loc[common_stocks_list]
        fwd_returns = fwd_returns.loc[common_stocks_list]
        
        try:
            mkt_groups = pd.qcut(mkt_betas, n_groups, labels=False, duplicates='drop') + 1
            vol_groups = pd.qcut(vol_betas, n_groups, labels=False, duplicates='drop') + 1
            skew_groups = pd.qcut(skew_betas, n_groups, labels=False, duplicates='drop') + 1
        except ValueError:
            continue
        
        portfolio_returns_date = {pid: [] for pid in range(1, portfolio_id + 1)}
        
        for stock in common_stocks:
            mkt_g = mkt_groups.loc[stock] if stock in mkt_groups.index else np.nan
            vol_g = vol_groups.loc[stock] if stock in vol_groups.index else np.nan
            skew_g = skew_groups.loc[stock] if stock in skew_groups.index else np.nan
            
            if any(pd.isna([mkt_g, vol_g, skew_g])):
                continue
            
            pid = portfolio_assignments.get((int(mkt_g), int(vol_g), int(skew_g)))
            if pid is not None:
                portfolio_returns_date[pid].append(fwd_returns.loc[stock])
        
        for pid in range(1, portfolio_id + 1):
            if portfolio_returns_date[pid]:
                portfolio_returns.loc[date, pid] = np.mean(portfolio_returns_date[pid])
    
    # Create factor-mimicking portfolios
    factor_mimicking = {}
    
    # FSKEW: portfolios with SKEW_group=3 (high) vs SKEW_group=1 (low)
    fskew_high = [pid for (m, v, s), pid in portfolio_assignments.items() if s == 3]
    fskew_low = [pid for (m, v, s), pid in portfolio_assignments.items() if s == 1]
    
    if fskew_high and fskew_low:
        fskew_high_returns = portfolio_returns[fskew_high].mean(axis=1)
        fskew_low_returns = portfolio_returns[fskew_low].mean(axis=1)
        # FSKEW = Low SKEW - High SKEW (represents return to skewness risk premium)
        factor_mimicking['FSKEW'] = (fskew_low_returns - fskew_high_returns) / len(fskew_high)
    
    # FVOL: portfolios with VOL_group=3 (high) vs VOL_group=1 (low)
    fvol_high = [pid for (m, v, s), pid in portfolio_assignments.items() if v == 3]
    fvol_low = [pid for (m, v, s), pid in portfolio_assignments.items() if v == 1]
    
    if fvol_high and fvol_low:
        fvol_high_returns = portfolio_returns[fvol_high].mean(axis=1)
        fvol_low_returns = portfolio_returns[fvol_low].mean(axis=1)
        # FVOL = Low VOL - High VOL (lowest quantile has highest return per paper)
        factor_mimicking['FVOL'] = (fvol_low_returns - fvol_high_returns) / len(fvol_high)
    
    return {
        'portfolios': portfolio_returns,
        'portfolio_assignments': portfolio_assignments,
        'factor_mimicking': factor_mimicking
    }

def run_triple_sort_analysis(start_date, end_date, windows, use_full_sample=False, split_date=None, in_sample_pct=0.7):
    """
    Run triple sort analysis to generate FSKEW, FVOL, and 27 portfolios.
    
    Based on methodology from:
    "Market Skewness Risk and the Cross Section of Stock Returns"
    by Bo Young Chang, Peter Christoffersen, and Kris Jacobs
    
    Parameters:
    -----------
    use_full_sample : bool
        If True, use full sample (no split). If False, use in-sample/out-of-sample split.
    """
    print(f"\n{'#'*80}")
    print(f"# FULL SAMPLE MODE: {'ENABLED' if use_full_sample else 'DISABLED'}")
    print(f"# use_full_sample parameter = {use_full_sample}")
    print(f"{'#'*80}\n")
    print("="*80)
    print("TRIPLE SORT ANALYSIS: FSKEW, FVOL, and 27 Portfolios")
    print("Methodology based on: Market Skewness Risk and the Cross Section of Stock Returns")
    print("By Bo Young Chang, Peter Christoffersen, and Kris Jacobs")
    print("="*80)
    
    # Determine in-sample and out-of-sample periods
    if use_full_sample:
        # Use full sample - no split
        in_sample_start = start_date
        in_sample_end = end_date
        out_sample_start = None
        out_sample_end = None
        print(f"\nUsing FULL SAMPLE: {start_date} to {end_date}")
    elif split_date is not None:
        split_date_dt = pd.to_datetime(split_date)
        in_sample_start = start_date
        in_sample_end = split_date_dt.strftime("%Y-%m-%d")
        out_sample_start = (split_date_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        out_sample_end = end_date
    else:
        in_sample_start = start_date
        in_sample_end = None
        out_sample_start = None
        out_sample_end = end_date
    
    # Load all required data
    print("\nLoading market, VIX, and SKEW data...")
    
    crsp_returns = load_crsp_market_returns(start_date, end_date, crsp_cache_file)
    if crsp_returns is None:
        print("ERROR: Could not load CRSP market returns. Exiting.")
        return
    
    crsp_changes = crsp_returns
    
    vix_prices, _ = fetch_data("^VIX", start_date, end_date)
    if vix_prices.empty:
        print("ERROR: Could not load VIX data. Exiting.")
        return
    vix_changes = calculate_index_changes(vix_prices)['Index_Change']
    
    skew_prices, _ = fetch_data("^Skew", start_date, end_date)
    if skew_prices.empty:
        print("ERROR: Could not load SKEW data. Exiting.")
        return
    skew_changes = calculate_index_changes(skew_prices)['Index_Change']
    
    # Determine split date if using percentage (skip if using full sample)
    if not use_full_sample:
        if split_date is None:
            common_dates = crsp_changes.index.intersection(vix_changes.index).intersection(skew_changes.index)
            if len(common_dates) > 0:
                split_idx = int(len(common_dates) * in_sample_pct)
                split_date_dt = common_dates[split_idx]
                in_sample_end = split_date_dt.strftime("%Y-%m-%d")
                out_sample_start = common_dates[min(split_idx + 1, len(common_dates) - 1)].strftime("%Y-%m-%d")
                print(f"\nUsing percentage split ({in_sample_pct*100:.0f}% in-sample):")
                print(f"  In-sample: {start_date} to {in_sample_end}")
                print(f"  Out-of-sample: {out_sample_start} to {end_date}")
            else:
                print("WARNING: Could not determine common dates. Using full sample.")
                in_sample_end = end_date
                out_sample_start = end_date
        else:
            print(f"\nUsing date-based split:")
            print(f"  In-sample: {in_sample_start} to {in_sample_end}")
            print(f"  Out-of-sample: {out_sample_start} to {out_sample_end}")
    
    # Load stock data
    min_required_days = max(windows)
    price_list = []
    failed_tickers = []
    
    known_failed_tickers = load_failed_tickers_cache()
    if known_failed_tickers:
        print(f"Loaded {len(known_failed_tickers)} known failed tickers from cache (will skip)")
    
    sp500_tickers_filtered = [t for t in sp500_tickers if t not in known_failed_tickers]
    
    print(f"Loading stock data for {len(sp500_tickers_filtered)} tickers...")
    cached_tickers = []
    uncached_tickers = []
    
    for ticker in sp500_tickers_filtered:
        cache_path = os.path.join(us_constituents_cache_dir, f"{ticker}.csv")
        if os.path.exists(cache_path):
            cached_tickers.append(ticker)
        else:
            uncached_tickers.append(ticker)
    
    print(f"Found {len(cached_tickers)} cached tickers, {len(uncached_tickers)} need downloading")
    
    if cached_tickers:
        for ticker in tqdm(cached_tickers, desc="Loading from cache", unit="ticker", leave=False):
            try:
                s, _ = fetch_data(ticker, start_date, end_date, refresh=False, cache_dir=us_constituents_cache_dir)
                if s.empty or len(s.dropna()) < min_required_days:
                    continue
                price_list.append(s)
            except:
                continue
    
    if uncached_tickers:
        for ticker in tqdm(uncached_tickers, desc="Downloading", unit="ticker"):
            s, _ = fetch_data(ticker, start_date, end_date, refresh=False, cache_dir=us_constituents_cache_dir)
            if s.empty or len(s.dropna()) < min_required_days:
                continue
            price_list.append(s)
            time.sleep(0.1)
    
    if not price_list:
        print("ERROR: No stock data available. Exiting.")
        return
    
    prices = pd.concat(price_list, axis=1, join='outer').ffill().loc[start_date:end_date]
    stock_returns = prices.pct_change().dropna()
    
    print(f"Loaded data for {len(stock_returns.columns)} stocks")
    
    # Run analysis for both in-sample and out-of-sample periods (or full sample)
    print(f"\n{'='*80}")
    print(f"FULL SAMPLE MODE: {use_full_sample}")
    print(f"{'='*80}")
    
    if use_full_sample:
        periods = [
            ("Full_Sample", in_sample_start, in_sample_end)
        ]
        print(f"✓ Using FULL SAMPLE mode - single period: Full_Sample from {in_sample_start} to {in_sample_end}")
    else:
        periods = [
            ("In_Sample", in_sample_start, in_sample_end),
            ("Out_of_Sample", out_sample_start, out_sample_end)
        ]
        print(f"✓ Using IN-SAMPLE/OUT-OF-SAMPLE split mode - two periods")
    
    for period_name, period_start, period_end in periods:
        if period_start is None or period_end is None or period_start >= period_end:
            print(f"\nSkipping {period_name} period (invalid dates)")
            continue
            
        print(f"\n{'='*80}")
        print(f"{period_name.upper()} ANALYSIS: {period_start} to {period_end}")
        print(f"{'='*80}")
        
        for window in tqdm(windows, desc=f"Processing windows ({period_name})", unit="window"):
            print(f"\n{'='*60}")
            print(f"Processing window {window} days...")
            print(f"{'='*60}")
        
            # Filter stocks with sufficient data for this period
            analysis_end_date = pd.to_datetime(period_end)
            analysis_start_date = pd.to_datetime(period_start)
            last_fwd_return_date = analysis_end_date - pd.Timedelta(days=window)
            required_end_date = last_fwd_return_date + pd.Timedelta(days=window)
            
            window_prices = prices.loc[analysis_start_date:required_end_date]
            required_period_prices = window_prices.loc[analysis_start_date:last_fwd_return_date]
            total_required_days = len(required_period_prices.index)
            min_required_days_for_window = int(total_required_days * 0.8)
            
            stocks_data_coverage = required_period_prices.notna().sum()
            stocks_with_full_data = stocks_data_coverage >= min_required_days_for_window
            valid_stocks = stocks_with_full_data[stocks_with_full_data].index.tolist()
            
            if len(valid_stocks) < 9:  # Need at least 9 stocks for triple sort
                print(f"ERROR: Only {len(valid_stocks)} stocks have sufficient data. Skipping.")
                continue
            
            window_prices_filtered = window_prices[valid_stocks]
            window_stock_returns = window_prices_filtered.pct_change().dropna()
            fwd_returns_df = forward_returns(window_prices_filtered, window)
            
            # Filter factor changes to this period
            vix_changes_period = vix_changes.loc[period_start:period_end]
            crsp_changes_period = crsp_changes.loc[period_start:period_end]
            skew_changes_period = skew_changes.loc[period_start:period_end]
            
            # Calculate regression betas for each factor (with market as control)
            print("\nCalculating regression-based betas with controls...")
            
            market_betas_reg = eiv_beta_calculator(window_stock_returns, crsp_changes_period, window)
            
            vol_betas_reg = calculate_regression_betas(
                window_stock_returns, 
                vix_changes_period, 
                crsp_changes_period, 
                window,
                n_jobs=N_JOBS
            )
            
            skew_betas_reg = calculate_regression_betas(
                window_stock_returns,
                skew_changes_period,
                crsp_changes_period,
                window,
                n_jobs=N_JOBS
            )
            
            # Perform triple sort
            print("\nCreating 27 portfolios via triple independent sort...")
            
            if not market_betas_reg.empty and not vol_betas_reg.empty and not skew_betas_reg.empty:
                common_dates_triple = (market_betas_reg.index
                                     .intersection(vol_betas_reg.index)
                                     .intersection(skew_betas_reg.index)
                                     .intersection(fwd_returns_df.index))
                
                if len(common_dates_triple) > 0:
                    common_dates_triple = [d for d in common_dates_triple 
                                         if fwd_returns_df.loc[d].notna().any()]
                    
                    if len(common_dates_triple) > 0:
                        market_betas_triple = market_betas_reg.loc[common_dates_triple]
                        vol_betas_triple = vol_betas_reg.loc[common_dates_triple]
                        skew_betas_triple = skew_betas_reg.loc[common_dates_triple]
                        fwd_returns_triple = fwd_returns_df.loc[common_dates_triple]
                        
                        triple_sort_result = triple_sort_portfolios(
                            {
                                'Market': market_betas_triple,
                                'VOL': vol_betas_triple,
                                'SKEW': skew_betas_triple
                            },
                            fwd_returns_triple,
                            n_groups=3
                        )
                        
                        if triple_sort_result:
                            # Generate plots and exports
                            print(f"\nGenerating plots and exports for {period_name}...")
                            plt.style.use("dark_background")

                            period_folder = period_name.replace('-', '_')
                            period_results_dir = os.path.join(analysis_results_dir, period_folder)
                            os.makedirs(period_results_dir, exist_ok=True)

                            portfolios_df = triple_sort_result['portfolios']
                            portfolio_assignments = triple_sort_result.get('portfolio_assignments', {})
                            factor_mimicking = triple_sort_result.get('factor_mimicking', {})
                            
                            # Create meaningful labels for each portfolio
                            portfolio_labels = {}
                            for (mkt_g, vol_g, skew_g), pid in portfolio_assignments.items():
                                portfolio_labels[pid] = f"MKT{mkt_g}_VOL{vol_g}_SKEW{skew_g}"
                            
                            # Plot 27 portfolios
                            if not portfolios_df.empty:
                                # FIX: portfolios_df contains multi-period forward returns (21-day returns)
                                # Convert to daily-equivalent returns before compounding: daily = (1 + multi_period)^(1/window) - 1
                                # This prevents absurdly high cumulative returns from compounding multi-period returns
                                daily_equiv_returns = (1 + portfolios_df).pow(1.0 / window) - 1
                                cumulative_returns = (1 + daily_equiv_returns).cumprod()
                                backtest_years = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days / 365.25 if not cumulative_returns.empty else 0
                                
                                if backtest_years > 0:
                                    # Calculate CAGR for each portfolio
                                    cagrs = {}
                                    for pid in portfolios_df.columns:
                                        port_cumulative = cumulative_returns[pid].dropna()
                                        if not port_cumulative.empty:
                                            cagr = calculate_cagr(port_cumulative, backtest_years)
                                            label = portfolio_labels.get(pid, f"Portfolio_{pid}")
                                            cagrs[label] = cagr
                                    
                                    # Also calculate CRSP Market CAGR for comparison
                                    crsp_period = crsp_returns.loc[period_start:period_end]
                                    common_dates = cumulative_returns.index.intersection(crsp_period.index)
                                    crsp_cumulative = pd.Series(dtype=float, index=cumulative_returns.index)
                                    if len(common_dates) > 0:
                                        crsp_cumulative = (crsp_period.loc[common_dates] + 1).cumprod()
                                        crsp_cumulative = crsp_cumulative.reindex(cumulative_returns.index, method='ffill')
                                        if not crsp_cumulative.empty:
                                            cagrs['CRSP Market'] = calculate_cagr(crsp_cumulative, backtest_years)
                                    
                                    # CAGR bar chart
                                    if cagrs:
                                        cagr_df = pd.Series(cagrs)
                                        cagr_df = cagr_df.sort_values(ascending=False)
                                
                                        fig, ax = plt.subplots(figsize=(16, 8))
                                        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(cagr_df)))
                                        if 'CRSP Market' in cagr_df.index:
                                            colors_list = list(colors)
                                            crsp_idx = list(cagr_df.index).index('CRSP Market')
                                            colors_list[crsp_idx] = 'orange'
                                            colors = colors_list
                                
                                        cagr_df.plot(kind='bar', color=colors, ax=ax)
                                        ax.set_title(f"CAGR for All 27 Triple-Sort Portfolios ({window}-day window) - {period_name}", fontsize=14)
                                        ax.set_xlabel("Portfolio", fontsize=12)
                                        ax.set_ylabel("CAGR (%)", fontsize=12)
                                        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
                                        ax.grid(axis='y', alpha=0.3)
                                        plt.xticks(rotation=90, ha='right', fontsize=8)
                                        plt.tight_layout()
                                
                                        plot_path = os.path.join(period_results_dir, f"TripleSort_27Portfolios_cagr_{window}d.png")
                                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                                        plt.close()
                                        print(f"Saved: {plot_path}")
                                
                                        # Cumulative returns plot
                                        fig, ax = plt.subplots(figsize=(14, 8))
                                
                                        if not crsp_cumulative.empty:
                                            ax.plot(crsp_cumulative, label='CRSP Market', color='orange', linestyle='--', linewidth=2, alpha=0.8)
                                        
                                        colors_port = plt.cm.viridis(np.linspace(0.2, 0.9, len(portfolios_df.columns)))
                                        for i, pid in enumerate(portfolios_df.columns):
                                            port_cumulative = cumulative_returns[pid].dropna()
                                            if not port_cumulative.empty:
                                                label = portfolio_labels.get(pid, f"Port_{pid}")
                                                ax.plot(port_cumulative, color=colors_port[i], alpha=0.4, linewidth=0.8, label=label if i < 5 else '')
                                
                                        ax.set_title(f"Cumulative Returns: All 27 Triple-Sort Portfolios ({window}-day window) - {period_name}", fontsize=14)
                                        ax.set_xlabel("Date", fontsize=12)
                                        ax.set_ylabel("Cumulative Return (Log Scale)", fontsize=12)
                                        ax.set_yscale('log')
                                        ax.legend(loc='best', fontsize=8, ncol=2)
                                        ax.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                
                                        plot_path = os.path.join(period_results_dir, f"TripleSort_27Portfolios_cumulative_{window}d.png")
                                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                                        plt.close()
                                        print(f"Saved: {plot_path}")
                                
                                        # Export portfolio returns
                                        portfolios_export = portfolios_df.copy()
                                        portfolios_export.columns = [portfolio_labels.get(pid, f"Portfolio_{pid}") for pid in portfolios_df.columns]
                                
                                        if not crsp_cumulative.empty:
                                            crsp_returns_aligned = crsp_period.reindex(portfolios_df.index, method='ffill')
                                            portfolios_export['CRSP Market'] = crsp_returns_aligned
                                        
                                        csv_path = os.path.join(period_results_dir, f"TripleSort_27Portfolios_returns_{window}d.csv")
                                        portfolios_export.to_csv(csv_path, index=True)
                                        print(f"Exported portfolio returns to: {csv_path}")
                                        
                                        # Export cumulative returns
                                        cumulative_export = cumulative_returns.copy()
                                        cumulative_export.columns = [portfolio_labels.get(pid, f"Portfolio_{pid}") for pid in cumulative_returns.columns]
                                        if not crsp_cumulative.empty:
                                            cumulative_export['CRSP Market'] = crsp_cumulative
                                
                                        csv_path_cum = os.path.join(period_results_dir, f"TripleSort_27Portfolios_cumulative_{window}d.csv")
                                        cumulative_export.to_csv(csv_path_cum, index=True)
                                        print(f"Exported cumulative returns to: {csv_path_cum}")
                                        
                                        # Calculate and export summary statistics
                                        print(f"Calculating summary statistics for 27 portfolios...")
                                        # Use daily-equivalent returns for summary stats (with proper column names)
                                        daily_equiv_returns_labeled = daily_equiv_returns.copy()
                                        daily_equiv_returns_labeled.columns = [portfolio_labels.get(pid, f"Portfolio_{pid}") for pid in daily_equiv_returns.columns]
                                        cumulative_returns_labeled = cumulative_returns.copy()
                                        cumulative_returns_labeled.columns = [portfolio_labels.get(pid, f"Portfolio_{pid}") for pid in cumulative_returns.columns]
                                        
                                        portfolio_stats = calculate_portfolio_summary_stats(
                                            daily_equiv_returns_labeled, 
                                            cumulative_returns_labeled,
                                            period_name,
                                            window
                                        )
                                        
                                        if portfolio_stats:
                                            # Create formatted summary text
                                            stats_text = f"""
{'='*100}
SUMMARY STATISTICS: 27 TRIPLE-SORT PORTFOLIOS
Period: {period_name}
Window: {window} days
{'='*100}

"""
                                            
                                            # Create a DataFrame for easier formatting
                                            stats_rows = []
                                            for port_name, stats_dict in portfolio_stats.items():
                                                row = {
                                                    'Portfolio': port_name,
                                                    'Mean (Annualized %)': f"{stats_dict['Mean (Annualized)']:.4f}",
                                                    'Volatility (Annualized %)': f"{stats_dict['Volatility (Annualized)']:.4f}",
                                                    'Variance (Annualized)': f"{stats_dict['Variance (Annualized)']:.6f}",
                                                    'Max Drawdown (%)': f"{stats_dict['Max Drawdown']:.4f}",
                                                    'CAGR (%)': f"{stats_dict['CAGR']:.4f}",
                                                    'Sharpe Ratio': f"{stats_dict['Sharpe Ratio']:.4f}",
                                                    'Calmar Ratio': f"{stats_dict['Calmar Ratio']:.4f}",
                                                    'Min Return (%)': f"{stats_dict['Min Return']:.4f}",
                                                    'Max Return (%)': f"{stats_dict['Max Return']:.4f}",
                                                    'Skewness': f"{stats_dict['Skewness']:.4f}",
                                                    'Kurtosis': f"{stats_dict['Kurtosis']:.4f}",
                                                    'Total Return (%)': f"{stats_dict['Total Return']:.4f}"
                                                }
                                                stats_rows.append(row)
                                            
                                            stats_df = pd.DataFrame(stats_rows)
                                            
                                            # Add to text
                                            stats_text += stats_df.to_string(index=False)
                                            stats_text += f"\n\n{'='*100}\n"
                                            stats_text += f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                            stats_text += f"{'='*100}\n"
                                            
                                            # Save to file
                                            stats_path = os.path.join(period_results_dir, f"TripleSort_27Portfolios_summary_stats_{window}d.txt")
                                            with open(stats_path, 'w') as f:
                                                f.write(stats_text)
                                            print(f"Exported summary statistics to: {stats_path}")
                            
                            # Plot FSKEW and FVOL
                            crsp_period = crsp_returns.loc[period_start:period_end]
                            
                            for factor_name in ['FSKEW', 'FVOL']:
                                if factor_name in factor_mimicking:
                                    factor_returns = factor_mimicking[factor_name]
                                    if not factor_returns.empty:
                                        # Calculate cumulative returns starting from 1.0
                                        # Prepend 1.0 before first return date so cumulative starts at 1.0
                                        one_plus_returns = 1 + factor_returns.fillna(0)
                                        initial_1 = pd.Series([1.0], index=[factor_returns.index[0] - pd.Timedelta(days=1)])
                                        series_to_cumprod = pd.concat([initial_1, one_plus_returns]).sort_index()
                                        cumulative_factor = series_to_cumprod.cumprod()
                                        
                                        common_dates = cumulative_factor.index.intersection(crsp_period.index)
                                        crsp_cumulative = pd.Series(dtype=float, index=cumulative_factor.index)
                                        if len(common_dates) > 0:
                                            crsp_returns_common = crsp_period.loc[common_dates]
                                            # Start CRSP cumulative at 1.0 as well
                                            crsp_one_plus = 1 + crsp_returns_common.fillna(0)
                                            crsp_initial_1 = pd.Series([1.0], index=[common_dates[0] - pd.Timedelta(days=1)])
                                            crsp_series_to_cumprod = pd.concat([crsp_initial_1, crsp_one_plus]).sort_index()
                                            crsp_cumulative_raw = crsp_series_to_cumprod.cumprod()
                                            crsp_cumulative = crsp_cumulative_raw.reindex(cumulative_factor.index, method='ffill')
                                        
                                        # Time series plot
                                        fig, ax = plt.subplots(figsize=(12, 6))
                                        ax.plot(factor_returns, label=factor_name, linewidth=2)
                                        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
                                        ax.set_title(f"Factor-Mimicking Portfolio: {factor_name} ({window}-day) - {period_name}")
                                        ax.set_xlabel("Date")
                                        ax.set_ylabel("Factor Return (%)")
                                        ax.legend()
                                        ax.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        plot_path = os.path.join(period_results_dir, f"{factor_name}_factor_mimicking_{window}d.png")
                                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                                        plt.close()
                                        print(f"Saved: {plot_path}")
                                        
                                        # Cumulative returns plot
                                        fig, ax = plt.subplots(figsize=(12, 8))
                                        if not crsp_cumulative.empty:
                                            ax.plot(crsp_cumulative, label='CRSP Market', color='white', linestyle='--', alpha=0.5)
                                        ax.plot(cumulative_factor, label=factor_name, linewidth=2)
                                        ax.axhline(y=1, color='white', linestyle='--', alpha=0.3)
                                        ax.set_title(f"Factor-Mimicking Portfolio Cumulative Returns: {factor_name} ({window}-day) - {period_name}")
                                        ax.set_xlabel("Date")
                                        ax.set_ylabel("Cumulative Return")
                                        ax.legend()
                                        ax.grid(True, alpha=0.3)
                                        plt.tight_layout()
                                        plot_path = os.path.join(period_results_dir, f"{factor_name}_cumulative_{window}d.png")
                                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                                        plt.close()
                                        print(f"Saved: {plot_path}")
                                        
                                        # Calculate and plot CAGR
                                        # Simply use start and end values of cumulative returns
                                        backtest_years = (cumulative_factor.index[-1] - cumulative_factor.index[0]).days / 365.25 if not cumulative_factor.empty else 0
                                        if backtest_years > 0:
                                            cagr_factor = calculate_cagr(cumulative_factor, backtest_years)
                                            cagr_crsp = calculate_cagr(crsp_cumulative, backtest_years) if not crsp_cumulative.empty else np.nan
                                            
                                            cagrs = {factor_name: cagr_factor}
                                            if not np.isnan(cagr_crsp):
                                                cagrs['CRSP Market'] = cagr_crsp
                                            
                                            cagr_df = pd.Series(cagrs)
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            cagr_df.plot(kind='bar', color=plt.cm.magma(np.linspace(0.2, 0.8, len(cagr_df))), ax=ax)
                                            ax.set_title(f"CAGR for {factor_name} Factor Portfolio ({window}-day window) - {period_name}")
                                            ax.set_xlabel("Portfolio")
                                            ax.set_ylabel("CAGR (%)")
                                            plt.xticks(rotation=45)
                                            plt.tight_layout()
                                            plot_path = os.path.join(period_results_dir, f"{factor_name}_cagr_{window}d.png")
                                            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                                            plt.close()
                                            print(f"Saved: {plot_path}")
                                        
                                        # Export factor returns
                                        factor_returns_export = pd.DataFrame({
                                            factor_name: factor_returns,
                                            'CRSP Market': crsp_period.reindex(factor_returns.index, method='ffill')
                                        })
                                        csv_path = os.path.join(period_results_dir, f"{factor_name}_returns_{window}d.csv")
                                        factor_returns_export.to_csv(csv_path, index=True)
                                        print(f"Exported {factor_name} returns to: {csv_path}")
                                        
                                        # Export cumulative returns
                                        factor_cumulative_export = pd.DataFrame({
                                            factor_name: cumulative_factor,
                                            'CRSP Market': crsp_cumulative.reindex(cumulative_factor.index, method='ffill') if not crsp_cumulative.empty else pd.Series(dtype=float, index=cumulative_factor.index)
                                        })
                                        csv_path_cum = os.path.join(period_results_dir, f"{factor_name}_cumulative_{window}d.csv")
                                        factor_cumulative_export.to_csv(csv_path_cum, index=True)
                                        print(f"Exported {factor_name} cumulative returns to: {csv_path_cum}")
    
        print(f"\n--- {period_name} analysis complete ---")
    
    print("\n--- Triple sort analysis complete (all periods) ---")

# --------------------------
# RUN
# --------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("WARNING: THIS ANALYSIS SUFFERS FROM SURVIVORSHIP BIAS")
    print("=" * 80)
    print("This analysis only includes stocks that currently exist in the ticker list.")
    print("Stocks that were delisted, merged, or went bankrupt are excluded.")
    print("This may lead to inflated performance metrics.")
    print("=" * 80)
    print()
    
    # Load US constituent tickers
    try:
        tickers_df = pd.read_csv(tickers_csv_path)
        sp500_tickers = tickers_df['Ticker'].dropna().tolist()
        sp500_tickers = [t.replace(".", "-") for t in sp500_tickers]
        print(f"Loaded {len(sp500_tickers)} US constituent tickers from {tickers_csv_path}")
    except Exception as e:
        print(f"Failed to load ticker list from {tickers_csv_path}: {e}")
        sp500_tickers = []

    # Run triple sort analysis
    run_triple_sort_analysis(start_date, end_date, window_sizes,
                              use_full_sample=USE_FULL_SAMPLE,
                              split_date=split_date, in_sample_pct=in_sample_pct)
