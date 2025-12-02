"""
PERFORMANCE OPTIMIZATIONS APPLIED:
1. Non-interactive matplotlib backend (Agg) for faster plotting
2. Pre-parse dates during CSV loading (avoid repeated parsing)
3. Use categorical dtypes for repeated string columns (signalname, port)
4. Pre-load summary file once instead of loading per signal
5. Vectorized rolling CAGR calculations (log-space instead of apply)
6. Vectorized rolling moments (use built-in rolling functions)
7. Cache pivoted dataframes (avoid repeated pivots)
8. Use portfolio_returns instead of groupby operations
9. Vectorized months calculation for aligned series
10. Optimize matplotlib settings (path simplification)
11. Sort data once at load time
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster plotting
import matplotlib.pyplot as plt
from pandas_datareader import data as web
import datetime
import os
import warnings

# ==============================================================================================
# USER INPUT
# ==============================================================================================
# Change this to a list of any signals in your CSV you want to test
# This list is used if 'test_all_strategies' is set to False
signals_to_test = [
    "IO_ShortInterest"
]
#==============================================================================================
# NEW TOGGLE TO TEST ALL STRATEGIES
# Set to True to automatically test all unique signals found in PredictorPortsFull.csv
# This will override the signals_to_test list above.
# Set to False to use the specific signals in the list above.
test_all_strategies = False
#==============================================================================================
# NEW SETTING FOR BAR GRAPH PLOTS
# If testing a large number of strategies, this will split the final bar graph
# into multiple plots, each containing this many strategies.
strategies_per_plot = 25
#==============================================================================================
# TOGGLE FOR DATE LOADING
# Set to True to automatically load in-sample dates from PredictorSummary.csv
# Set to False to manually specify your own dates below
auto_load_dates = True

# MANUALLY SPECIFY DATES IF auto_load_dates IS False
manual_start_year = 2003
manual_end_year = 2023
#==============================================================================================
# TOGGLE FOR PRE-PUBLICATION OOS
# Set to True to include a pre-sample period for backtesting.
run_pre_sample = False

# NEW SETTING FOR DYNAMIC PRE-SAMPLE PERIOD
# Set to True to use ALL available historical data before the in-sample period.
# This overrides any manual pre-sample date settings from a previous version.
run_pre_sample_all_data = False

# MINIMUM PRE-PUBLICATION PERIOD
# Strategies with pre-sample data less than this value will be skipped from the analysis.
min_pre_sample_years = 5
#==============================================================================================
# TOGGLE FOR FULL SAMPLE BACKTEST
# Set to True to run a single, full-sample backtest with no out-of-sample split.
# This will override run_pre_sample and auto_load_dates.
run_full_sample = False
#==============================================================================================
# TOGGLE FOR MOMENT DISTRIBUTION PLOTS
# Set to True to generate distribution plots of statistical moments across all strategies
# Set to False to skip moment distribution plots
generate_moment_distributions = True

# TOGGLE FOR RETURN DISTRIBUTION PLOTS
# Set to True to plot return distributions for all portfolios (LS, 1, 2, 3, etc.)
# Set to False to plot only the Long-Short (LS) portfolio
plot_all_portfolios = True
# MOMENT DISTRIBUTION SETTING
# Set to 1-4 to control which moments are included in the distribution plots:
# 1 = Mean (CAGR) only
# 2 = Mean (CAGR) and Variance
# 3 = Mean (CAGR), Variance, and Skewness
# 4 = Mean (CAGR), Variance, Skewness, and Kurtosis
moment_distribution_level = 4

# ROLLING WINDOW SETTING FOR MOMENT DISTRIBUTIONS
# Number of months to use for rolling window calculations of moments
# This creates distributions from rolling statistics rather than single point estimates
rolling_window_months = 24

# TITLE SUFFIX FOR PLOTS
# This will be prompted at runtime - see below

# Define the file paths for your data
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PORTS_FILE = os.path.join(SCRIPT_DIR, "PredictorPortsFull.csv")
PREDICTOR_SUMMARY_FILE = os.path.join(SCRIPT_DIR, "PredictorSummary.csv")

# ================================================================
# PROMPT FOR PLOT TITLE SUFFIX (needed before creating output folders)
# ================================================================
print("\n" + "="*80)
print("PLOT TITLE SUFFIX")
print("="*80)
print("Enter a suffix to add to all plot titles (e.g., 'Momentum', 'Value Strategies', etc.)")
print("This will also be used as the output folder name to organize all outputs.")
print("Type 'NONE' or press Enter to skip adding a suffix (will use 'Output' as folder name).")
suffix_input = input("Plot title suffix: ").strip()

# Process the input: if NONE, empty, or just whitespace, set to empty string
if suffix_input.upper() == "NONE" or suffix_input == "":
    plot_title_suffix = ""
    output_folder_name = "Output"
    print("No suffix will be added to plot titles.")
    print(f"Outputs will be organized in folder: '{output_folder_name}'")
else:
    plot_title_suffix = suffix_input
    # Sanitize folder name: replace invalid filesystem characters
    output_folder_name = plot_title_suffix.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace('"', "_").replace("<", "_").replace(">", "_").replace("|", "_")
    print(f"Plot titles will include: '{plot_title_suffix}'")
    print(f"Outputs will be organized in folder: '{output_folder_name}'")
print("="*80 + "\n")

# Create base output directory with suffix name
OUTPUT_BASE = os.path.join(SCRIPT_DIR, output_folder_name)

# Create output directories under the base folder
OUTPUT_BACKTEST = os.path.join(OUTPUT_BASE, "backtest_plots")
OUTPUT_TRAILING_CAGR = os.path.join(OUTPUT_BASE, "trailing_cagr_plots")
OUTPUT_RAW_RETURNS = os.path.join(OUTPUT_BASE, "raw_returns")
OUTPUT_SUMMARY = os.path.join(OUTPUT_BASE, "summary_plots")
OUTPUT_ROLLING = os.path.join(OUTPUT_BASE, "rolling_plots")
OUTPUT_DISTRIBUTIONS = os.path.join(OUTPUT_BASE, "Distributions")
OUTPUT_DISTRIBUTIONS_AGGREGATE = os.path.join(OUTPUT_DISTRIBUTIONS, "aggregate")
OUTPUT_DISTRIBUTIONS_INDIVIDUAL = os.path.join(OUTPUT_DISTRIBUTIONS, "individual")

# Create moment-specific subfolders
MOMENT_NAMES = ['Mean', 'Variance', 'Skewness', 'Kurtosis']
OUTPUT_DISTRIBUTIONS_AGGREGATE_MOMENTS = {
    moment: os.path.join(OUTPUT_DISTRIBUTIONS_AGGREGATE, moment) for moment in MOMENT_NAMES
}
OUTPUT_DISTRIBUTIONS_INDIVIDUAL_MOMENTS = {
    moment: os.path.join(OUTPUT_DISTRIBUTIONS_INDIVIDUAL, moment) for moment in MOMENT_NAMES
}

# Create directories if they don't exist
base_dirs = [OUTPUT_BASE, OUTPUT_BACKTEST, OUTPUT_TRAILING_CAGR, OUTPUT_RAW_RETURNS, OUTPUT_SUMMARY, 
             OUTPUT_ROLLING, OUTPUT_DISTRIBUTIONS, OUTPUT_DISTRIBUTIONS_AGGREGATE, OUTPUT_DISTRIBUTIONS_INDIVIDUAL]
moment_dirs = list(OUTPUT_DISTRIBUTIONS_AGGREGATE_MOMENTS.values()) + list(OUTPUT_DISTRIBUTIONS_INDIVIDUAL_MOMENTS.values())
for output_dir in base_dirs + moment_dirs:
    os.makedirs(output_dir, exist_ok=True)

# ================================================================
# CAGR FUNCTION
# ================================================================
def calculate_cagr(returns):
    """Calculates the Compound Annual Growth Rate (CAGR) from a series of monthly returns."""
    if returns.empty:
        return np.nan
    returns = returns / 100.0
    backtest_series = (1 + returns).cumprod()
    num_years = len(returns) / 12
    if num_years <= 0:
        return np.nan
    final_value = backtest_series.iloc[-1]
    if final_value <= 0:
        return np.nan
    return ((final_value ** (1 / num_years)) - 1) * 100

# ================================================================
# STATISTICS FUNCTIONS
# ================================================================
def calculate_max_drawdown(returns):
    """Calculates the maximum drawdown from a series of monthly returns."""
    if returns.empty:
        return np.nan
    returns = returns / 100.0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min() * 100  # Return as percentage

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculates the annualized Sharpe ratio from monthly returns."""
    if returns.empty or len(returns) < 2:
        return np.nan
    returns = returns / 100.0
    excess_returns = returns - (risk_free_rate / 12.0)  # Convert annual RF to monthly
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    if std_excess == 0:
        return np.nan
    # Annualize: multiply by sqrt(12) for monthly data
    return (mean_excess / std_excess) * np.sqrt(12)

def calculate_calmar_ratio(returns):
    """Calculates the Calmar ratio (CAGR / abs(max drawdown))."""
    cagr = calculate_cagr(returns)
    max_dd = calculate_max_drawdown(returns)
    if np.isnan(cagr) or np.isnan(max_dd) or max_dd == 0:
        return np.nan
    return cagr / abs(max_dd)

def calculate_rolling_moments(returns, window_months):
    """Calculates rolling moments (CAGR, variance, skewness, kurtosis) over a rolling window (optimized)."""
    if returns.empty or len(returns) < window_months:
        return {
            'CAGR (%)': pd.Series(dtype=float),
            'Variance': pd.Series(dtype=float),
            'Skewness': pd.Series(dtype=float),
            'Kurtosis': pd.Series(dtype=float)
        }
    
    # Optimized rolling CAGR using vectorized operations
    decimal_returns = returns / 100.0
    log_returns = np.log1p(decimal_returns)
    rolling_log_sum = log_returns.rolling(window=window_months, min_periods=window_months).sum()
    num_years = window_months / 12.0
    rolling_cagr = (np.exp(rolling_log_sum / num_years) - 1) * 100
    
    # Use built-in rolling functions (much faster than apply)
    rolling_variance = returns.rolling(window=window_months, min_periods=window_months).var()
    rolling_skewness = returns.rolling(window=window_months, min_periods=window_months).skew()
    rolling_kurtosis = returns.rolling(window=window_months, min_periods=window_months).kurt()
    
    return {
        'CAGR (%)': rolling_cagr,
        'Variance': rolling_variance,
        'Skewness': rolling_skewness,
        'Kurtosis': rolling_kurtosis
    }

# ================================================================
# STEP 1: DATA LOADING (OPTIMIZED)
# ================================================================
print(f"Loading data from '{PREDICTOR_PORTS_FILE}'...")
try:
    # Optimize CSV reading with dtype hints and parse dates immediately
    df = pd.read_csv(PREDICTOR_PORTS_FILE, parse_dates=['date'], 
                     dtype={'signalname': 'category', 'port': 'category', 'ret': 'float64'})
    required_cols = ['date', 'signalname', 'port', 'ret']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV file is missing one or more required columns: {required_cols}")
except FileNotFoundError:
    print(f"\nERROR: '{PREDICTOR_PORTS_FILE}' not found.")
    exit()
except Exception as e:
    print(f"\nAn error occurred while loading the CSV file: {e}")
    exit()

# Sort once for better performance
df = df.sort_values(['signalname', 'date'])

# Filter for the signals we're interested in
if test_all_strategies:
    print("Testing all strategies found in the data.")
    signals_to_test = df['signalname'].cat.categories.tolist() if df['signalname'].dtype.name == 'category' else df['signalname'].unique().tolist()
    df_subset = df.copy()
else:
    print(f"Testing a subset of strategies: {signals_to_test}")
    df_subset = df[df['signalname'].isin(signals_to_test)].copy()

if df_subset.empty:
    print(f"\nERROR: No data found for any of the signals. Check spelling or file contents.")
    exit()
print(f"Data for {len(signals_to_test)} signals loaded successfully.")

# Storage for all LS results for the final summary graph
ls_cagr_results = []

# Storage for all strategy CAGR values for distribution plot
all_strategy_cagrs = []

# Storage for all strategy rolling moments for distribution plots (by period AND portfolio)
# Structure: {port: {period: {moment: []}}}
# This ensures each portfolio's plot shows statistics only for that portfolio type
all_strategy_rolling_moments_by_port_period = {}

# Storage for final (non-rolling) statistics by period for aggregate plots
all_strategy_final_moments_by_period = {
    'In-Sample': {
        'CAGR (%)': [],
        'Variance': [],
        'Skewness': [],
        'Kurtosis': []
    },
    'Out-of-Sample': {
        'CAGR (%)': [],
        'Variance': [],
        'Skewness': [],
        'Kurtosis': []
    },
    'Post-Publication': {
        'CAGR (%)': [],
        'Variance': [],
        'Skewness': [],
        'Kurtosis': []
    },
    'Full Sample': {
        'CAGR (%)': [],
        'Variance': [],
        'Skewness': [],
        'Kurtosis': []
    }
}

# Storage for rolling 5-year growth aligned by months since publication
aligned_series = {}

# ================================================================
# PRE-LOAD SUMMARY FILE (Load once before loop)
# ================================================================
summary_df = None
if auto_load_dates:
    print(f"\nPre-loading summary data from '{PREDICTOR_SUMMARY_FILE}'...")
    try:
        summary_df = pd.read_csv(PREDICTOR_SUMMARY_FILE, dtype={'signalname': 'category'})
        summary_cols = ['signalname', 'SampleStartYear', 'SampleEndYear']
        if not all(col in summary_df.columns for col in summary_cols):
            raise ValueError(f"CSV file is missing one or more required columns: {summary_cols}")
        # Create lookup dictionary for faster access
        summary_lookup = summary_df.set_index('signalname').to_dict('index')
        print("Summary data loaded successfully.")
    except FileNotFoundError:
        print(f"\nERROR: '{PREDICTOR_SUMMARY_FILE}' not found.")
        exit()
    except Exception as e:
        print(f"\nAn error occurred while loading the summary file: {e}")
        exit()
else:
    summary_lookup = None

# ================================================================
# CACHE FAMA-FRENCH DATA (Download once before loop)
# ================================================================
# Determine the date range needed for all signals
df_subset_dates = df_subset.copy()
df_subset_dates["date"] = pd.to_datetime(df_subset_dates["date"])
global_ff_start = df_subset_dates['date'].min()
global_ff_end = df_subset_dates['date'].max()

print(f"\nCaching Fama-French data for date range: {global_ff_start.strftime('%Y-%m')} to {global_ff_end.strftime('%Y-%m')}...")

# Create cache directory for Fama-French data
ff_cache_dir = os.path.join(SCRIPT_DIR, "ff_cache")
os.makedirs(ff_cache_dir, exist_ok=True)
ff_cache_file = os.path.join(ff_cache_dir, f"F-F_Research_Data_Factors_{global_ff_start.strftime('%Y%m')}_{global_ff_end.strftime('%Y%m')}.csv")

# Try to load from cache
if os.path.exists(ff_cache_file):
    try:
        print("Loading Fama-French data from cache...")
        ff_cache_df = pd.read_csv(ff_cache_file, index_col=0, parse_dates=[0])
        print("Fama-French data loaded from cache successfully.")
    except Exception as e:
        print(f"Cache file corrupted. Re-downloading... Error: {e}")
        ff_cache_df = None
else:
    ff_cache_df = None

# Download if cache doesn't exist or is corrupted
if ff_cache_df is None:
    print("Downloading Fama-French data...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        ff_downloaded = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=global_ff_start, end=global_ff_end)[0]
    ff_downloaded.index = ff_downloaded.index.to_timestamp()
    ff_downloaded.index.name = 'Date'
    ff_cache_df = ff_downloaded
    # Save to cache
    ff_cache_df.to_csv(ff_cache_file)
    print("Fama-French data downloaded and cached successfully.")

# Create a function to get CRSP market returns from cached data
def get_crsp_market_returns(start_date, end_date):
    """Get CRSP value-weighted market returns from cached Fama-French data."""
    crsp_returns = (ff_cache_df['Mkt-RF'] + ff_cache_df['RF']).loc[start_date:end_date]
    crsp_df = crsp_returns.to_frame('monthly_ret_pct')
    crsp_df.index = pd.to_datetime(crsp_df.index)
    return crsp_df

# ================================================================
# MAIN LOOP FOR EACH SIGNAL
# ================================================================
for signal_to_test in signals_to_test:
    print("\n" + "="*80)
    print(f"Processing signal: {signal_to_test}")
    print("="*80)

    # Filter signal data (dates already parsed during CSV load)
    signal_df = df_subset[df_subset['signalname'] == signal_to_test].copy()

    # Check if this signal has the LS portfolio
    if 'LS' not in signal_df['port'].unique():
        print(f"WARNING: No 'LS' portfolio found for signal '{signal_to_test}'. Skipping this signal in the final summary graph.")
        continue

    # ================================================================
    # STEP 1.5: DEFINE IN-SAMPLE & OOS DATES
    # ================================================================
    if run_full_sample:
        print("\nRunning full-sample backtest. All other splits are disabled.")
        # Dates already parsed, no need to parse again
        start_date_full = signal_df['date'].min()
        end_date_full = signal_df['date'].max()
        pre_sample_start_date = None
        pre_sample_end_date = None
        sample_start_date = start_date_full
        sample_end_date = end_date_full
        out_of_sample_start_date = end_date_full + pd.Timedelta(days=1)

    else:
        # Define in-sample dates
        if auto_load_dates:
            if summary_lookup is None or signal_to_test not in summary_lookup:
                print(f"\nERROR: Signal '{signal_to_test}' not found in summary data.")
                continue
            signal_info = summary_lookup[signal_to_test]
            start_year = int(signal_info['SampleStartYear'])
            end_year = int(signal_info['SampleEndYear'])
        else:
            start_year = manual_start_year
            end_year = manual_end_year
            print(f"\nManually specified in-sample period: {start_year} to {end_year}")

        sample_start_date = pd.to_datetime(f"{start_year}-01-01")
        sample_end_date = pd.to_datetime(f"{end_year}-12-31")
        out_of_sample_start_date = sample_end_date + pd.Timedelta(days=1)

        # New dynamic pre-sample date logic
        if run_pre_sample and run_pre_sample_all_data:
            # Dates already parsed, no need to parse again
            pre_sample_start_date = signal_df['date'].min()
            pre_sample_end_date = sample_start_date - pd.Timedelta(days=1)

            # CHECK FOR MINIMUM PRE-SAMPLE PERIOD
            pre_sample_duration = (pre_sample_end_date - pre_sample_start_date).days / 365.25
            if pre_sample_duration < min_pre_sample_years:
                print(f"WARNING: Pre-sample period for '{signal_to_test}' is only {pre_sample_duration:.2f} years. Skipping because it is less than the minimum required {min_pre_sample_years} years.")
                continue

            print(f"Using all available data for pre-sample period: {pre_sample_start_date.strftime('%Y-%m')} to {pre_sample_end_date.strftime('%Y-%m')}")

            # CHECK FOR NaN VALUES AND SKIP STRATEGY
            pre_sample_data = signal_df[(signal_df['date'] >= pre_sample_start_date) & (signal_df['date'] <= pre_sample_end_date)]
            if pre_sample_data.empty:
                print(f"WARNING: No data available for the pre-sample period for '{signal_to_test}'. Skipping.")
                continue

            ls_pre_sample_data = pre_sample_data[pre_sample_data['port'] == 'LS']
            if ls_pre_sample_data['ret'].isnull().any():
                print(f"WARNING: Found NaN values in the Long-Short portfolio for signal '{signal_to_test}' during the pre-sample period. Skipping this strategy.")
                continue

        else:
            pre_sample_start_date = None
            pre_sample_end_date = None

    # ================================================================
    # STEP 2: PREPARE SIGNAL DATA
    # ================================================================
    # Dates already parsed and data already sorted, just create ret_pct column
    signal_df = signal_df.sort_values("date")  # Ensure sorted (should already be sorted)
    signal_df["ret_pct"] = signal_df["ret"]

    start_date = signal_df['date'].min()
    end_date = signal_df['date'].max()
    print(f"Data starts on: {start_date.strftime('%Y-%m')} and ends on: {end_date.strftime('%Y-%m')}")

    # Check if the data range covers the specified periods
    if pre_sample_start_date and start_date > pre_sample_start_date:
        print(f"WARNING: Pre-sample period start date ({pre_sample_start_date.strftime('%Y-%m')}) is before data start date ({start_date.strftime('%Y-%m')}). Using data start date instead.")
        pre_sample_start_date = start_date
    if start_date > sample_start_date:
         print(f"WARNING: In-sample period start date ({sample_start_date.strftime('%Y-%m')}) is before data start date ({start_date.strftime('%Y-%m')}). Using data start date instead.")
         sample_start_date = start_date
    if end_date < out_of_sample_start_date:
        print(f"WARNING: No data available for the out-of-sample period. The data ends before the out-of-sample period begins.")

    # ================================================================
    # STEP 3: GET CRSP MARKET RETURNS (from cached Fama-French data)
    # ================================================================
    ff_start_date = pre_sample_start_date if pre_sample_start_date else start_date
    ff_end_date = end_date
    crsp_market_df = get_crsp_market_returns(ff_start_date, ff_end_date)

    # ================================================================
    # STEP 4: CALCULATE CUMULATIVE RETURNS
    # ================================================================
    # Pivot the signal data to have separate columns for each portfolio's returns
    # Cache this pivot as it's used multiple times
    portfolio_returns = signal_df.pivot_table(index="date", columns="port", values="ret_pct", aggfunc='first')

    # Calculate the cumulative returns for each portfolio (vectorized)
    cumulative_returns = (1 + portfolio_returns / 100).cumprod()

    # Calculate cumulative returns for the CRSP Market
    crsp_market_cumulative_returns = (1 + crsp_market_df['monthly_ret_pct'] / 100).cumprod()

    # ================================================================
    # STEP 5: PLOTTING CUMULATIVE RETURNS (MODIFIED FOR 3 PANELS)
    # ================================================================
    plt.style.use("dark_background")
    # Optimize matplotlib for faster plotting
    plt.ioff()  # Turn off interactive mode
    matplotlib.rcParams['path.simplify'] = True
    matplotlib.rcParams['path.simplify_threshold'] = 1.0
    special_colors = {'LS': 'cyan', '1': 'yellow', '2': 'magenta', '3': 'lime'}
    unique_ports = signal_df['port'].unique()
    extra_ports = [p for p in unique_ports if p not in special_colors]
    cmap = plt.get_cmap('tab10')
    auto_colors = {p: cmap(i % 10) for i, p in enumerate(sorted(extra_ports))}
    color_map = {**special_colors, **auto_colors}

    if run_full_sample:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        fig.suptitle(f"Full-Sample Cumulative Returns: {signal_to_test} vs. CRSP Market (Logarithmic Scale)", fontsize=16, color="white")

        cumulative_returns_norm = cumulative_returns / cumulative_returns.iloc[0]
        crsp_market_cumulative_returns_norm = crsp_market_cumulative_returns / crsp_market_cumulative_returns.iloc[0]

        for port in cumulative_returns_norm.columns:
            ax.plot(cumulative_returns_norm.index, cumulative_returns_norm[port], label=f"{signal_to_test} {port}", color=color_map.get(port, 'white'))
        ax.plot(crsp_market_cumulative_returns_norm.index, crsp_market_cumulative_returns_norm, label="CRSP Market", color="orange", linestyle='--')
        ax.set_title("Full Sample Period", color="white")
        ax.set_xlabel("Date", color="white")
        ax.set_ylabel("Cumulative Return (Normalized)", color="white")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_yscale('log')
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        output_path = os.path.join(OUTPUT_BACKTEST, f"backtest_full_sample_plot_{signal_to_test}.png")
        plt.savefig(output_path)
        print(f"Backtest plot saved to: {output_path}")
        plt.close()

    elif run_pre_sample and run_pre_sample_all_data:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        fig.suptitle(f"Cumulative Returns Backtest: {signal_to_test} vs. CRSP Market (Logarithmic Scale)", fontsize=16, color="white")

        # Pre-Sample Plot
        pre_sample_returns = cumulative_returns[cumulative_returns.index >= pre_sample_start_date]
        pre_sample_returns = pre_sample_returns[pre_sample_returns.index <= pre_sample_end_date]
        pre_sample_crsp_market = crsp_market_cumulative_returns[crsp_market_cumulative_returns.index >= pre_sample_start_date]
        pre_sample_crsp_market = pre_sample_crsp_market[pre_sample_crsp_market.index <= pre_sample_end_date]
        if not pre_sample_returns.empty:
            pre_sample_returns_norm = pre_sample_returns / pre_sample_returns.iloc[0]
            pre_sample_crsp_market_norm = pre_sample_crsp_market / pre_sample_crsp_market.iloc[0]
            for port in pre_sample_returns_norm.columns:
                axes[0].plot(pre_sample_returns_norm.index, pre_sample_returns_norm[port], label=f"{signal_to_test} {port}", color=color_map.get(port, 'white'))
            axes[0].plot(pre_sample_crsp_market_norm.index, pre_sample_crsp_market_norm, label="CRSP Market", color="orange", linestyle='--')
        axes[0].set_title("Pre-Publication Period", color="white")
        axes[0].set_xlabel("Date", color="white")
        axes[0].set_ylabel("Cumulative Return (Normalized)", color="white")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.4)
        axes[0].set_yscale('log')
        axes[0].axvline(pre_sample_end_date, color='lime', linestyle='--', linewidth=1, label="Pre-Publication End")
        axes[0].legend()

        # In-Sample Plot
        in_sample_returns = cumulative_returns[cumulative_returns.index > pre_sample_end_date]
        in_sample_returns = in_sample_returns[in_sample_returns.index <= sample_end_date]
        in_sample_crsp_market = crsp_market_cumulative_returns[crsp_market_cumulative_returns.index > pre_sample_end_date]
        in_sample_crsp_market = in_sample_crsp_market[in_sample_crsp_market.index <= sample_end_date]
        if not in_sample_returns.empty:
            in_sample_returns_norm = in_sample_returns / in_sample_returns.iloc[0]
            in_sample_crsp_market_norm = in_sample_crsp_market / in_sample_crsp_market.iloc[0]
            for port in in_sample_returns_norm.columns:
                axes[1].plot(in_sample_returns_norm.index, in_sample_returns_norm[port], label=f"{signal_to_test} {port}", color=color_map.get(port, 'white'))
            axes[1].plot(in_sample_crsp_market_norm.index, in_sample_crsp_market_norm, label="CRSP Market", color="orange", linestyle='--')
        axes[1].set_title("In-Sample Period", color="white")
        axes[1].set_xlabel("Date", color="white")
        axes[1].set_ylabel("Cumulative Return (Normalized)", color="white")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].set_yscale('log')
        axes[1].axvline(sample_end_date, color='lime', linestyle='--', linewidth=1, label="In-Sample End")
        axes[1].legend()

        # Out-of-Sample Plot
        out_of_sample_returns = cumulative_returns[cumulative_returns.index >= out_of_sample_start_date]
        out_of_sample_crsp_market = crsp_market_cumulative_returns[crsp_market_cumulative_returns.index >= out_of_sample_start_date]
        if not out_of_sample_returns.empty:
            out_of_sample_returns_norm = out_of_sample_returns / out_of_sample_returns.iloc[0]
            out_of_sample_crsp_market_norm = out_of_sample_crsp_market / out_of_sample_crsp_market.iloc[0]
            for port in out_of_sample_returns_norm.columns:
                axes[2].plot(out_of_sample_returns_norm.index, out_of_sample_returns_norm[port], label=f"{signal_to_test} {port}", color=color_map.get(port, 'white'))
            axes[2].plot(out_of_sample_crsp_market_norm.index, out_of_sample_crsp_market_norm, label="CRSP Market", color="orange", linestyle='--')
        axes[2].set_title("Post-Publication Period", color="white")
        axes[2].set_xlabel("Date", color="white")
        axes[2].set_ylabel("Cumulative Return (Normalized)", color="white")
        axes[2].legend()
        axes[2].grid(True, linestyle="--", alpha=0.4)
        axes[2].set_yscale('log')

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        output_path = os.path.join(OUTPUT_BACKTEST, f"backtest_plot_{signal_to_test}.png")
        plt.savefig(output_path)
        print(f"Backtest plot saved to: {output_path}")
        plt.close()

    else: # Existing in-sample / out-of-sample plot logic
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f"Cumulative Returns Backtest: {signal_to_test} vs. CRSP Market (Logarithmic Scale)", fontsize=16, color="white")

        in_sample_returns = cumulative_returns[cumulative_returns.index <= sample_end_date]
        in_sample_crsp_market = crsp_market_cumulative_returns[crsp_market_cumulative_returns.index <= sample_end_date]
        if not in_sample_returns.empty:
            in_sample_returns_norm = in_sample_returns / in_sample_returns.iloc[0]
            in_sample_crsp_market_norm = in_sample_crsp_market / in_sample_crsp_market.iloc[0]
            for port in in_sample_returns_norm.columns:
                axes[0].plot(in_sample_returns_norm.index, in_sample_returns_norm[port], label=f"{signal_to_test} {port}", color=color_map.get(port, 'white'))
            axes[0].plot(in_sample_crsp_market_norm.index, in_sample_crsp_market_norm, label="CRSP Market", color="orange", linestyle='--')
        axes[0].set_title("In-Sample Period", color="white")
        axes[0].set_xlabel("Date", color="white")
        axes[0].set_ylabel("Cumulative Return (Normalized)", color="white")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.4)
        axes[0].set_yscale('log')

        out_of_sample_returns = cumulative_returns[cumulative_returns.index >= out_of_sample_start_date]
        out_of_sample_crsp_market = crsp_market_cumulative_returns[crsp_market_cumulative_returns.index >= out_of_sample_start_date]
        if not out_of_sample_returns.empty:
            out_of_sample_returns_norm = out_of_sample_returns / out_of_sample_returns.iloc[0]
            out_of_sample_crsp_market_norm = out_of_sample_crsp_market / out_of_sample_crsp_market.iloc[0]
            for port in out_of_sample_returns_norm.columns:
                axes[1].plot(out_of_sample_returns_norm.index, out_of_sample_returns_norm[port], label=f"{signal_to_test} {port}", color=color_map.get(port, 'white'))
            axes[1].plot(out_of_sample_crsp_market_norm.index, out_of_sample_crsp_market_norm, label="CRSP Market", color="orange", linestyle='--')
        axes[1].set_title("Out-of-Sample Period", color="white")
        axes[1].set_xlabel("Date", color="white")
        axes[1].set_ylabel("Cumulative Return (Normalized)", color="white")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].set_yscale('log')

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        output_path = os.path.join(OUTPUT_BACKTEST, f"backtest_plot_{signal_to_test}.png")
        plt.savefig(output_path)
        print(f"Backtest plot saved to: {output_path}")
        plt.close()

    # ================================================================
    # STEP 5.5: PLOTTING ROLLING 5-YEAR RETURNS (CONTINUOUS) (MODIFIED)
    # ================================================================
    print("Generating rolling 5-year return plot...")

    def calculate_rolling_cagr(returns_series, window_size=60):
        """Calculates the rolling CAGR over a specified window (optimized vectorized version)."""
        decimal_returns = returns_series / 100.0
        # Use log-space for numerical stability and speed
        log_returns = np.log1p(decimal_returns)
        # Rolling sum of log returns = log of product
        rolling_log_sum = log_returns.rolling(window=window_size, min_periods=window_size).sum()
        # Convert back: exp(sum) = product, then annualize
        cagr = (np.exp(rolling_log_sum * (12 / window_size)) - 1) * 100
        return cagr

    portfolio_rolling_cagr = portfolio_returns.apply(calculate_rolling_cagr)
    crsp_market_rolling_cagr = calculate_rolling_cagr(crsp_market_df['monthly_ret_pct'])

    fig, ax = plt.subplots(figsize=(12, 7))
    if run_full_sample:
        fig.suptitle(f"Trailing 5-Year CAGR: {signal_to_test} vs. CRSP Market (Full Sample)", fontsize=16, color="white")
        ax.set_title("Entire Backtesting Period", color="white")
    else:
        fig.suptitle(f"Trailing 5-Year CAGR: {signal_to_test} vs. CRSP Market (Continuous Plot)", fontsize=16, color="white")
        if run_pre_sample and run_pre_sample_all_data:
            ax.axvline(x=pre_sample_end_date, color='grey', linestyle=':', linewidth=2)
            ax.axvline(x=out_of_sample_start_date, color='grey', linestyle=':', linewidth=2)
            # Note: ylim values will be set after plotting, so we'll position text after plotting
        else:
            ax.axvline(x=out_of_sample_start_date, color='grey', linestyle=':', linewidth=2, label='In-Sample End')


    for port in portfolio_rolling_cagr.columns:
        ax.plot(portfolio_rolling_cagr.index, portfolio_rolling_cagr[port], label=f"{signal_to_test} {port}", color=color_map.get(port, 'white'))
    ax.plot(crsp_market_rolling_cagr.index, crsp_market_rolling_cagr, label="CRSP Market", color="orange", linestyle='--')

    # Add text labels after plotting so we can get ylim values
    if not run_full_sample and run_pre_sample and run_pre_sample_all_data:
        y_top = ax.get_ylim()[1]
        ax.text(pre_sample_end_date, y_top * 0.9, 'Pre-Publication End', color='white', ha='right', fontsize=9)
        ax.text(out_of_sample_start_date, y_top * 0.9, 'In-Sample End', color='white', ha='left', fontsize=9)

    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Trailing 5-Year CAGR (%)", color="white")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if run_full_sample:
        output_path = os.path.join(OUTPUT_TRAILING_CAGR, f"trailing_cagr_{signal_to_test}.png")
        plt.savefig(output_path)
        print(f"Full sample rolling 5-year return plot saved to: {output_path}")
    else:
        output_path = os.path.join(OUTPUT_TRAILING_CAGR, f"trailing_cagr_{signal_to_test}.png")
        plt.savefig(output_path)
        print(f"Continuous rolling 5-year return plot saved to: {output_path}")

    plt.close()

    # ================================================================
    # STEP 6: PERFORMANCE TABLE (MODIFIED)
    # ================================================================
    if run_full_sample:
        print(f"\n--- Full Sample Performance (Annual CAGR) for {signal_to_test} ---")
        print(f"Period: {sample_start_date.strftime('%Y-%m')} → {sample_end_date.strftime('%Y-%m')}")
        # Use portfolio_returns instead of groupby for faster calculation
        full_sample_cagr = portfolio_returns.apply(calculate_cagr)
        full_sample_crsp_market = crsp_market_df[(crsp_market_df.index >= sample_start_date) & (crsp_market_df.index <= sample_end_date)]
        full_sample_crsp_market_cagr = calculate_cagr(full_sample_crsp_market['monthly_ret_pct'])
        performance_df = pd.DataFrame({'Full Sample CAGR (%)': full_sample_cagr})
        crsp_market_row = pd.DataFrame({'Full Sample CAGR (%)': [full_sample_crsp_market_cagr]}, index=['CRSP Market'])
        performance_df = pd.concat([performance_df, crsp_market_row])
        print(performance_df.round(2).to_string())

    elif run_pre_sample and run_pre_sample_all_data:
        print(f"\n--- Multi-Period Performance (Annual CAGR) for {signal_to_test} ---")
        print(f"Pre-Publication: {pre_sample_start_date.strftime('%Y-%m')} → {pre_sample_end_date.strftime('%Y-%m')}")
        print(f"In-Sample: {sample_start_date.strftime('%Y-%m')} → {sample_end_date.strftime('%Y-%m')}")
        print(f"Post-Publication: {out_of_sample_start_date.strftime('%Y-%m')} → {end_date.strftime('%Y-%m')}")

        # Use portfolio_returns (already pivoted) instead of filtering signal_df and groupby
        pre_sample_returns = portfolio_returns[(portfolio_returns.index >= pre_sample_start_date) & (portfolio_returns.index <= pre_sample_end_date)]
        in_sample_returns_perf = portfolio_returns[(portfolio_returns.index > pre_sample_end_date) & (portfolio_returns.index <= sample_end_date)]
        out_of_sample_returns_perf = portfolio_returns[portfolio_returns.index >= out_of_sample_start_date]

        pre_sample_crsp_market = crsp_market_df[(crsp_market_df.index >= pre_sample_start_date) & (crsp_market_df.index <= pre_sample_end_date)]
        in_sample_crsp_market = crsp_market_df[(crsp_market_df.index > pre_sample_end_date) & (crsp_market_df.index <= sample_end_date)]
        out_of_sample_crsp_market = crsp_market_df[crsp_market_df.index >= out_of_sample_start_date]

        pre_sample_cagr = pre_sample_returns.apply(calculate_cagr)
        in_sample_cagr = in_sample_returns_perf.apply(calculate_cagr)
        out_of_sample_cagr = out_of_sample_returns_perf.apply(calculate_cagr)

        pre_sample_crsp_market_cagr = calculate_cagr(pre_sample_crsp_market['monthly_ret_pct'])
        in_sample_crsp_market_cagr = calculate_cagr(in_sample_crsp_market['monthly_ret_pct'])
        out_of_sample_crsp_market_cagr = calculate_cagr(out_of_sample_crsp_market['monthly_ret_pct'])

        performance_df = pd.DataFrame({
            'Pre-Publication CAGR (%)': pre_sample_cagr,
            'In-Sample CAGR (%)': in_sample_cagr,
            'Post-Publication CAGR (%)': out_of_sample_cagr
        })
        crsp_market_row = pd.DataFrame({
            'Pre-Publication CAGR (%)': [pre_sample_crsp_market_cagr],
            'In-Sample CAGR (%)': [in_sample_crsp_market_cagr],
            'Post-Publication CAGR (%)': [out_of_sample_crsp_market_cagr]
        }, index=['CRSP Market'])
        performance_df = pd.concat([performance_df, crsp_market_row])
        print(performance_df.round(2).to_string())

        # Store results for the final summary graph
        ls_cagr_results.append({
            'signal': signal_to_test,
            'Pre-Publication CAGR': pre_sample_cagr.get('LS', np.nan),
            'In-Sample CAGR': in_sample_cagr.get('LS', np.nan),
            'Post-Publication CAGR': out_of_sample_cagr.get('LS', np.nan)
        })

    else: # The existing in-sample/out-of-sample logic
        print(f"\n--- In-Sample and Out-of-Sample Performance (Annual CAGR) for {signal_to_test} ---")
        print(f"In-Sample: {sample_start_date.strftime('%Y-%m')} → {sample_end_date.strftime('%Y-%m')}")
        print(f"Out-of-Sample: {out_of_sample_start_date.strftime('%Y-%m')} → {end_date.strftime('%Y-%m')}")

        # Use portfolio_returns (already pivoted) instead of filtering signal_df and groupby
        in_sample_returns_perf = portfolio_returns[(portfolio_returns.index >= sample_start_date) & (portfolio_returns.index <= sample_end_date)]
        out_of_sample_returns_perf = portfolio_returns[portfolio_returns.index >= out_of_sample_start_date]
        in_sample_crsp_market = crsp_market_df[(crsp_market_df.index >= sample_start_date) & (crsp_market_df.index <= sample_end_date)]
        out_of_sample_crsp_market = crsp_market_df[crsp_market_df.index >= out_of_sample_start_date]

        in_sample_cagr = in_sample_returns_perf.apply(calculate_cagr)
        out_of_sample_cagr = out_of_sample_returns_perf.apply(calculate_cagr)

        ls_cagr_results.append({
            'signal': signal_to_test,
            'in_sample_cagr': in_sample_cagr.get('LS', np.nan),
            'out_of_sample_cagr': out_of_sample_cagr.get('LS', np.nan)
        })

        in_sample_crsp_market_cagr = calculate_cagr(in_sample_crsp_market['monthly_ret_pct'])
        out_of_sample_crsp_market_cagr = calculate_cagr(out_of_sample_crsp_market['monthly_ret_pct'])

        performance_df = pd.DataFrame({
            'In-Sample CAGR (%)': in_sample_cagr,
            'Out-of-Sample CAGR (%)': out_of_sample_cagr
        })
        crsp_market_row = pd.DataFrame({
            'In-Sample CAGR (%)': [in_sample_crsp_market_cagr],
            'Out-of-Sample CAGR (%)': [out_of_sample_crsp_market_cagr]
        }, index=['CRSP Market'])
        performance_df = pd.concat([performance_df, crsp_market_row])
        print(performance_df.round(2).to_string())

    # ================================================================
    # STEP 6.5: STATISTICS CALCULATION (Return distribution plots removed - redundant with moment distributions)
    # ================================================================
    print("\nCalculating statistics...")
    
    # Filter portfolios based on toggle
    if not plot_all_portfolios:
        # Only analyze LS portfolio
        portfolios_to_analyze = ['LS'] if 'LS' in portfolio_returns.columns else []
    else:
        # Analyze all portfolios
        portfolios_to_analyze = portfolio_returns.columns.tolist()
    
    # Prepare data for different periods
    if run_full_sample:
        # For full sample, only show full sample period
        in_sample_returns = portfolio_returns
        out_of_sample_returns = pd.DataFrame()  # Empty
        full_sample_returns = portfolio_returns
        periods_to_show = ['Full Sample']
    elif run_pre_sample and run_pre_sample_all_data:
        # For pre-sample mode, show pre-sample, in-sample, and post-sample
        in_sample_returns = portfolio_returns[(portfolio_returns.index > pre_sample_end_date) & 
                                               (portfolio_returns.index <= sample_end_date)]
        out_of_sample_returns = portfolio_returns[portfolio_returns.index >= out_of_sample_start_date]
        full_sample_returns = portfolio_returns
        periods_to_show = ['In-Sample', 'Post-Publication', 'Full Sample']
    else:
        # Standard mode: show in-sample, out-of-sample, and full sample
        in_sample_returns = portfolio_returns[(portfolio_returns.index >= sample_start_date) & 
                                               (portfolio_returns.index <= sample_end_date)]
        out_of_sample_returns = portfolio_returns[portfolio_returns.index >= out_of_sample_start_date]
        full_sample_returns = portfolio_returns
        periods_to_show = ['In-Sample', 'Out-of-Sample', 'Full Sample']
    
    # Calculate statistics for each portfolio and period
    all_stats = {}  # {port: {period: stats_dict}}
    
    for port in portfolios_to_analyze:
        if port not in portfolio_returns.columns:
            continue
        
        all_stats[port] = {}
        
        # Calculate statistics for each period
        for period_name, period_returns in [
            ('In-Sample', in_sample_returns),
            ('Out-of-Sample', out_of_sample_returns),
            ('Post-Publication', out_of_sample_returns if run_pre_sample and run_pre_sample_all_data else pd.DataFrame()),
            ('Full Sample', full_sample_returns)
        ]:
            if period_name not in periods_to_show:
                continue
            
            if port not in period_returns.columns or period_returns.empty:
                continue
            
            port_returns = period_returns[port].dropna()
            if len(port_returns) == 0:
                continue
            
            # Calculate all statistics
            mean_ret = port_returns.mean()
            variance = port_returns.var()
            std_dev = port_returns.std()
            skewness = port_returns.skew()
            kurtosis = port_returns.kurtosis()
            max_dd = calculate_max_drawdown(port_returns)
            sharpe = calculate_sharpe_ratio(port_returns)
            calmar = calculate_calmar_ratio(port_returns)
            cagr = calculate_cagr(port_returns)
            
            all_stats[port][period_name] = {
                'returns': port_returns,
                'Mean (%)': mean_ret,
                'Variance': variance,
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'Max Drawdown (%)': max_dd,
                'Sharpe Ratio': sharpe,
                'Calmar Ratio': calmar,
                'CAGR (%)': cagr
            }
            
            # Store CAGR for final distribution plot (only for LS portfolio, use out-of-sample or post-publication)
            if port == 'LS' and not np.isnan(cagr) and period_name in ['Out-of-Sample', 'Post-Publication']:
                all_strategy_cagrs.append(cagr)
            
            # Calculate and store moments for moment distribution plots
            # Include all portfolios if plot_all_portfolios is True, otherwise just LS
            if generate_moment_distributions and (plot_all_portfolios or port == 'LS'):
                # Calculate rolling moments for this period (for individual strategy plots)
                rolling_moments = calculate_rolling_moments(port_returns, rolling_window_months)
                
                # Store rolling values by portfolio AND period for individual strategy plots
                # This ensures each portfolio's plot shows statistics only for that portfolio type
                if port not in all_strategy_rolling_moments_by_port_period:
                    all_strategy_rolling_moments_by_port_period[port] = {
                        'In-Sample': {'CAGR (%)': [], 'Variance': [], 'Skewness': [], 'Kurtosis': []},
                        'Out-of-Sample': {'CAGR (%)': [], 'Variance': [], 'Skewness': [], 'Kurtosis': []},
                        'Post-Publication': {'CAGR (%)': [], 'Variance': [], 'Skewness': [], 'Kurtosis': []},
                        'Full Sample': {'CAGR (%)': [], 'Variance': [], 'Skewness': [], 'Kurtosis': []}
                    }
                
                period_key = period_name
                if period_key in all_strategy_rolling_moments_by_port_period[port]:
                    for moment_name in ['CAGR (%)', 'Variance', 'Skewness', 'Kurtosis']:
                        rolling_values = rolling_moments[moment_name].dropna().tolist()
                        all_strategy_rolling_moments_by_port_period[port][period_key][moment_name].extend(rolling_values)
                
                # Store FINAL (non-rolling) statistics for aggregate plots (one value per strategy)
                # Store by period for all periods
                # ONLY use LS portfolios for aggregate distributions
                if port == 'LS':
                    period_key = period_name
                    if period_key in all_strategy_final_moments_by_period:
                        # Use final statistics, not rolling - one value per strategy per period
                        if not np.isnan(cagr):
                            all_strategy_final_moments_by_period[period_key]['CAGR (%)'].append(cagr)
                        if not np.isnan(variance):
                            all_strategy_final_moments_by_period[period_key]['Variance'].append(variance)
                        if not np.isnan(skewness):
                            all_strategy_final_moments_by_period[period_key]['Skewness'].append(skewness)
                        if not np.isnan(kurtosis):
                            all_strategy_final_moments_by_period[period_key]['Kurtosis'].append(kurtosis)
    
    # Skip return distribution plots - they're redundant with moment distributions
    # Statistics are still calculated and printed below
    # Return distribution plotting code removed - moment distributions are used instead
    
    # Print statistics tables
    if len(all_stats) > 0:
        print(f"\n--- Statistics for {signal_to_test} ---")
        for port in all_stats.keys():
            print(f"\n{signal_to_test} {port}:")
            for period_name in periods_to_show:
                if period_name in all_stats[port]:
                    stats = all_stats[port][period_name]
                    print(f"  {period_name}:")
                    print(f"    Mean: {stats['Mean (%)']:.4f}%, Variance: {stats['Variance']:.4f}, "
                          f"Skewness: {stats['Skewness']:.4f}, Kurtosis: {stats['Kurtosis']:.4f}")
                    print(f"    Max DD: {stats['Max Drawdown (%)']:.4f}%, Sharpe: {stats['Sharpe Ratio']:.4f}, "
                          f"Calmar: {stats['Calmar Ratio']:.4f}, CAGR: {stats['CAGR (%)']:.4f}%")
    
    # ================================================================
    # INDIVIDUAL STRATEGY MOMENT DISTRIBUTION PLOTS
    # ================================================================
    if generate_moment_distributions and len(all_stats) > 0:
        print(f"\nGenerating individual moment distribution plots for {signal_to_test}...")
        
        # Create for all portfolios if plot_all_portfolios is True, otherwise just LS
        portfolios_for_moments = list(all_stats.keys()) if plot_all_portfolios else (['LS'] if 'LS' in all_stats else [])
        
        # Create distribution plots for each moment (showing distribution across all strategies by period)
        moment_labels = {
            'CAGR (%)': 'Rolling CAGR (%)',
            'Variance': 'Rolling Variance',
            'Skewness': 'Rolling Skewness',
            'Kurtosis': 'Rolling Kurtosis'
        }
        
        moment_folder_map = {
            'CAGR (%)': 'Mean',
            'Variance': 'Variance',
            'Skewness': 'Skewness',
            'Kurtosis': 'Kurtosis'
        }
        
        # Determine which moments to plot
        moments_to_plot_individual = []
        if moment_distribution_level >= 1:
            moments_to_plot_individual.append('CAGR (%)')
        if moment_distribution_level >= 2:
            moments_to_plot_individual.append('Variance')
        if moment_distribution_level >= 3:
            moments_to_plot_individual.append('Skewness')
        if moment_distribution_level >= 4:
            moments_to_plot_individual.append('Kurtosis')
        
        # Get rolling moments for all portfolios
        all_portfolio_rolling_moments = {}
        for port in portfolios_for_moments:
            port_stats = all_stats[port]
            portfolio_rolling_moments_by_period = {}
            for period_name in periods_to_show:
                if period_name in port_stats:
                    period_returns = port_stats[period_name]['returns']
                    rolling_moments = calculate_rolling_moments(period_returns, rolling_window_months)
                    portfolio_rolling_moments_by_period[period_name] = rolling_moments
            all_portfolio_rolling_moments[port] = portfolio_rolling_moments_by_period
        
        # Create one PNG per moment, with subplots for each portfolio
        for moment in moments_to_plot_individual:
            num_ports = len(portfolios_for_moments)
            num_periods = len(periods_to_show)
            
            # Create subplots: rows = num_ports, cols = num_periods
            plt.style.use("dark_background")
            fig, axes = plt.subplots(num_ports, num_periods, figsize=(7*num_periods, 5.5*num_ports))
            
            # Handle axes indexing
            if num_ports == 1 and num_periods == 1:
                axes_array = np.array([[axes]], dtype=object)
            elif num_ports == 1:
                if not isinstance(axes, np.ndarray):
                    axes_array = np.array([axes])
                else:
                    axes_array = axes.reshape(1, -1)
            elif num_periods == 1:
                if not isinstance(axes, np.ndarray):
                    axes_array = np.array([[axes]], dtype=object)
                else:
                    axes_array = axes.reshape(-1, 1)
            else:
                axes_array = axes
            
            title_suffix_text = f" - {plot_title_suffix}" if plot_title_suffix else ""
            fig.suptitle(f"{signal_to_test} - Distribution of {moment_labels[moment]} Across All Strategies{title_suffix_text}", 
                         fontsize=16, color="white", y=0.995, ha='center')
            
            for port_idx, port in enumerate(portfolios_for_moments):
                portfolio_rolling_moments_by_period = all_portfolio_rolling_moments[port]
                
                for period_idx, period_name in enumerate(periods_to_show):
                    # Get the correct axis
                    if num_ports == 1 and num_periods == 1:
                        ax = axes_array[0, 0]
                    elif num_ports == 1:
                        ax = axes_array[0, period_idx]
                    elif num_periods == 1:
                        ax = axes_array[port_idx, 0]
                    else:
                        ax = axes_array[port_idx, period_idx]
                    
                    # Get distribution data for this period across all strategies
                    # IMPORTANT: Use portfolio-specific data so each plot shows stats for that portfolio type only
                    period_key = period_name
                    if port in all_strategy_rolling_moments_by_port_period:
                        if period_key in all_strategy_rolling_moments_by_port_period[port]:
                            moment_data = all_strategy_rolling_moments_by_port_period[port][period_key][moment]
                        else:
                            moment_data = []
                    else:
                        moment_data = []
                    
                    # Get this portfolio's rolling values for this period (for highlighting distribution)
                    if period_name in portfolio_rolling_moments_by_period:
                        portfolio_rolling_values = portfolio_rolling_moments_by_period[period_name][moment].dropna().tolist()
                    else:
                        portfolio_rolling_values = []
                    
                    if len(moment_data) == 0:
                        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                               ha='center', va='center', color='white', fontsize=12)
                        ax.set_title(f"{port} - {period_name}", color="white", fontsize=11, pad=10)
                        continue
                    
                    # Create histogram with portfolio-specific color
                    port_color = color_map.get(port, 'white')
                    n, bins, patches = ax.hist(moment_data, bins=30, color=port_color, 
                                              alpha=0.7, edgecolor='white', linewidth=1)
                    
                    # Highlight this portfolio's rolling values distribution
                    if len(portfolio_rolling_values) > 0:
                        # Show mean of this portfolio's rolling values
                        portfolio_mean = np.mean(portfolio_rolling_values)
                        ax.axvline(portfolio_mean, color=port_color, linestyle='-', linewidth=3, 
                                  label=f"{port} Mean: {portfolio_mean:.4f}", zorder=10)
                        # Optionally show min/max range
                        portfolio_min = np.min(portfolio_rolling_values)
                        portfolio_max = np.max(portfolio_rolling_values)
                        ax.axvspan(portfolio_min, portfolio_max, alpha=0.2, color=port_color, 
                                  label=f"{port} Range: [{portfolio_min:.4f}, {portfolio_max:.4f}]")
                    
                    # Add vertical lines at mean and median of all strategies
                    mean_val = np.mean(moment_data)
                    median_val = np.median(moment_data)
                    ax.axvline(mean_val, color='orange', linestyle='--', linewidth=2, 
                              label=f"Mean: {mean_val:.4f}")
                    ax.axvline(median_val, color='lime', linestyle='--', linewidth=2, 
                              label=f"Median: {median_val:.4f}")
                    
                    # Calculate and display statistics
                    std_val = np.std(moment_data)
                    min_val = np.min(moment_data)
                    max_val = np.max(moment_data)
                    
                    # Calculate additional statistics
                    skewness_val = pd.Series(moment_data).skew() if len(moment_data) > 2 else 0
                    kurtosis_val = pd.Series(moment_data).kurtosis() if len(moment_data) > 2 else 0
                    
                    stats_text = (
                        f"All {port} Portfolios:\n"
                        f"Number of Observations: {len(moment_data)}\n"
                        f"Mean: {mean_val:.4f}\n"
                        f"Median: {median_val:.4f}\n"
                        f"Std Dev: {std_val:.4f}\n"
                        f"Skewness: {skewness_val:.4f}\n"
                        f"Kurtosis: {kurtosis_val:.4f}\n"
                        f"Min: {min_val:.4f}\n"
                        f"Max: {max_val:.4f}"
                    )
                    
                    ax.set_title(f"{port} - {period_name}", color="white", fontsize=11, pad=10)
                    ax.set_xlabel(moment_labels[moment], color="white")
                    ax.set_ylabel("Frequency", color="white")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    ax.legend(fontsize=8, loc='upper right')
                    
                    # Add statistics text box
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                           fontsize=8, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white'),
                           color='white', family='monospace')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            if num_ports == 1:
                plt.subplots_adjust(top=0.90, wspace=0.3)
            else:
                plt.subplots_adjust(top=0.92, hspace=0.35, wspace=0.3)
            
            suffix_filename = f"_{plot_title_suffix.replace(' ', '_')}" if plot_title_suffix else ""
            folder_name = moment_folder_map[moment]
            output_path = os.path.join(OUTPUT_DISTRIBUTIONS_INDIVIDUAL_MOMENTS[folder_name], 
                                      f"{signal_to_test}_{folder_name.lower()}_distribution{suffix_filename}.png")
            plt.savefig(output_path)
            print(f"  {moment_labels[moment]} distribution plot for {signal_to_test} saved to: {output_path}")
            plt.close()

    # ================================================================
    # STEP 7: BUILD ALIGNED SERIES FOR ROLLING PLOTS
    # ================================================================
    # Calculate rolling 5-year multiplicative growth and align by months since publication
    if not run_full_sample and 'LS' in portfolio_returns.columns:
        # Get LS portfolio returns
        ls_returns = portfolio_returns['LS'].dropna()
        
        if len(ls_returns) >= 60:  # Need at least 60 months for 5-year rolling window
            # Calculate rolling 5-year multiplicative growth (optimized vectorized version)
            decimal_returns = ls_returns / 100.0
            # Use log-space for numerical stability and speed
            log_returns = np.log1p(decimal_returns)
            rolling_log_sum = log_returns.rolling(window=60, min_periods=60).sum()
            rolling_growth = np.exp(rolling_log_sum)
            rolling_growth = rolling_growth.dropna()
            
            # Align by months since publication (sample_end_date is publication date)
            publication_date = sample_end_date
            # Vectorized calculation of months since publication
            pub_year_month = publication_date.year * 12 + publication_date.month
            months_since_pub = (rolling_growth.index.year * 12 + rolling_growth.index.month) - pub_year_month
            
            if len(months_since_pub) > 0:
                aligned_series[signal_to_test] = pd.Series(rolling_growth.values, index=months_since_pub)

    # ================================================================
    # STEP 8: SAVE RAW RETURNS TO CSV
    # ================================================================
    print("\nSaving raw portfolio returns to CSV...")
    # Reuse portfolio_returns (already pivoted) instead of pivoting again
    raw_returns = portfolio_returns.sort_index()
    crsp_market_df_subset = crsp_market_df[['monthly_ret_pct']].rename(columns={'monthly_ret_pct': 'CRSP Market'})
    crsp_market_df_subset = crsp_market_df_subset.reindex(raw_returns.index)
    returns_export = pd.concat([raw_returns, crsp_market_df_subset['CRSP Market']], axis=1)
    output_file = os.path.join(OUTPUT_RAW_RETURNS, f"{signal_to_test}_raw_returns.csv")
    returns_export.to_csv(output_file, index_label="Date")
    print(f"Raw returns exported to: {output_file}")

# ================================================================
# FINAL SUMMARY BAR GRAPH (MODIFIED)
# ================================================================
print("\n" + "="*80)
print("Generating final summary bar graph for all LS portfolios...")
print("="*80)

summary_df = pd.DataFrame(ls_cagr_results)

if not summary_df.empty:
    summary_df = summary_df.set_index('signal')
    if 'Pre-Publication CAGR' in summary_df.columns:
        # Reorder columns as requested
        summary_df = summary_df[['Pre-Publication CAGR', 'In-Sample CAGR', 'Post-Publication CAGR']]
    else:
        summary_df = summary_df.rename(columns={'in_sample_cagr': 'In-Sample CAGR', 'out_of_sample_cagr': 'Out-of-Sample CAGR'})

    num_strategies = len(summary_df)
    num_plots = (num_strategies + strategies_per_plot - 1) // strategies_per_plot

    for i in range(num_plots):
        start_index = i * strategies_per_plot
        end_index = start_index + strategies_per_plot
        chunk = summary_df.iloc[start_index:end_index]

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(15, 8))

        chunk.plot(kind='bar', ax=ax, rot=45, colormap='Paired')

        title_suffix_text = f" - {plot_title_suffix}" if plot_title_suffix else ""
        
        # Build title: show range as (1-X) format per user request
        if end_index > num_strategies:
            end_index = num_strategies
        
        chunk_count = end_index - start_index  # Number of strategies in this chunk
        
        if num_strategies <= strategies_per_plot:
            # All strategies fit in one plot - show (1-X) where X is total number
            title_range = f"(1-{num_strategies})"
        else:
            # Multiple plots - show range for this specific chunk
            title_range = f"({start_index+1}-{end_index})"
        
        part_text = f" (Part {i+1})" if num_plots > 1 else ""
        ax.set_title(f"Long-Short (LS) Portfolio Annual CAGR {title_range}{part_text}{title_suffix_text}", fontsize=16, color="white")
        ax.set_xlabel("Strategy Signal", color="white")
        ax.set_ylabel("Annual CAGR (%)", color="white")
        ax.grid(axis='y', linestyle="--", alpha=0.4)
        ax.legend(title="Performance Period")
        plt.tight_layout()

        # Build filename: only include part number if there are multiple parts
        if num_plots > 1:
            output_file = os.path.join(OUTPUT_SUMMARY, f"final_ls_cagr_summary_part_{i+1}.png")
        else:
            output_file = os.path.join(OUTPUT_SUMMARY, "final_ls_cagr_summary.png")
        plt.savefig(output_file)
        print(f"Summary plot saved to: {output_file}")
        plt.close()

else:
    print("No LS portfolio data found to create a summary graph. This may be due to 'run_full_sample' being True, or all strategies having NaN values in the pre-sample period.")

# ======================================================================
# ROLLING PLOTS ALIGNED BY MONTHS SINCE PUBLICATION
# ======================================================================
min_fraction = 0.3  # Only plot months where >=30% of strategies have data
selected_signals = signals_to_test  # Use the same signals as in your script

# ======================================================================
# BUILD ALIGNED DATAFRAME
# ======================================================================
if len(aligned_series) == 0:
    print("\nNo usable aligned series found for rolling plots.")
else:
    # Filter to only selected signals
    aligned_series_filtered = {s: ser for s, ser in aligned_series.items() if s in selected_signals}

    if not aligned_series_filtered:
        print("No aligned series found for the selected signals.")
    else:
        # Combine all selected strategies into a single DataFrame aligned by Months Since Publication
        # Use pd.concat instead of loop assignment to avoid PerformanceWarning
        all_months = sorted(set().union(*[s.index.values for s in aligned_series_filtered.values()]))
        aligned_series_list = []
        for s, ser in aligned_series_filtered.items():
            reindexed_ser = ser.reindex(all_months)
            reindexed_ser.name = s
            aligned_series_list.append(reindexed_ser)
        aligned_df = pd.concat(aligned_series_list, axis=1)

        # ==================================================================
        # APPLY COVERAGE MASK
        # ==================================================================
        coverage_frac = aligned_df.notnull().sum(axis=1) / aligned_df.shape[1]
        mask = coverage_frac >= min_fraction
        aligned_df_masked = aligned_df[mask]
        composite_masked = aligned_df_masked.mean(axis=1, skipna=True)

        # ==================================================================
        # 1️⃣ Multiplicative-Growth Rolling 5-Year Plot
        # ==================================================================
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(16, 9))
        title_suffix_text = f" - {plot_title_suffix}" if plot_title_suffix else ""
        fig.suptitle(f"Rolling 5-Year Multiplicative Growth (×) Aligned by Months Since Publication{title_suffix_text}",
                     fontsize=16, color="white")

        # Plot individual strategies with transparency
        for s in aligned_df_masked.columns:
            ser = aligned_df_masked[s]
            if ser.dropna().empty:
                continue

            # Plot contiguous non-NaN blocks separately
            notnull = ser.notnull().astype(int)
            diffs = notnull.diff().fillna(notnull.iloc[0]).values
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            if notnull.iloc[0] == 1:
                starts = np.insert(starts, 0, 0)
            if notnull.iloc[-1] == 1:
                ends = np.append(ends, len(notnull))
            for st_idx, end_idx in zip(starts, ends):
                x_block = aligned_df_masked.index[st_idx:end_idx]
                y_block = ser.iloc[st_idx:end_idx].values
                ax.plot(x_block, y_block, alpha=0.25, linewidth=1)

        # EW Composite line
        ax.plot(composite_masked.index, composite_masked.values, color="white", linewidth=2.6, label="Equal-Weight Composite")

        # Publication marker
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.5)
        if not composite_masked.dropna().empty:
            y_top = composite_masked.dropna().max()
            ax.text(0, y_top * 0.95, "Publication", color="white", ha="left", va="top")

        ax.set_yscale("log")
        ax.set_ylabel("Rolling 5-Year Growth (×)", color="white")
        ax.set_xlabel("Months Since Publication", color="white")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        suffix_filename = f"_{plot_title_suffix.replace(' ', '_')}" if plot_title_suffix else ""
        output_path = os.path.join(OUTPUT_ROLLING, f"rolling_5yr{suffix_filename}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

        # ==================================================================
        # 2️⃣ Thinner Percentile-Band Plot (IQR Only + Median)
        # ==================================================================
        p25 = aligned_df_masked.quantile(0.25, axis=1)
        p50 = aligned_df_masked.quantile(0.50, axis=1)
        p75 = aligned_df_masked.quantile(0.75, axis=1)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.suptitle(f"Rolling 5-Year Multiplicative Growth (×) Percentile Bands (IQR Only)\nAligned by Months Since Publication{title_suffix_text}",
                     fontsize=16, color="white")

        # Shaded IQR band
        ax.fill_between(aligned_df_masked.index, p25, p75, color="cyan", alpha=0.35, label="25th–75th percentile (IQR)")

        # Median line
        ax.plot(p50.index, p50.values, color="lime", linewidth=2, label="Median (50th percentile)")

        # EW Composite line
        ax.plot(composite_masked.index, composite_masked.values, color="white", linewidth=2.6, label="Equal-Weight Composite")

        # Publication marker
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.5)
        if not composite_masked.dropna().empty:
            y_top = composite_masked.dropna().max()
            ax.text(0, y_top * 0.95, "Publication", color="white", ha="left", va="top")

        ax.set_yscale("log")
        ax.set_ylabel("Rolling 5-Year Growth (×)", color="white")
        ax.set_xlabel("Months Since Publication", color="white")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        suffix_filename = f"_{plot_title_suffix.replace(' ', '_')}" if plot_title_suffix else ""
        output_path = os.path.join(OUTPUT_ROLLING, f"rolling_5yr_percentile_bands{suffix_filename}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved: {output_path}")

# ================================================================
# CAGR DISTRIBUTION PLOT FOR ALL STRATEGIES
# ================================================================
print("\n" + "="*80)
print("Generating CAGR distribution plot for all strategies...")
print("="*80)

if len(all_strategy_cagrs) > 0:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    title_suffix_text = f" - {plot_title_suffix}" if plot_title_suffix else ""
    fig.suptitle(f"Distribution of All Strategy CAGR Values{title_suffix_text}", 
                 fontsize=16, color="white")
    
    # Create histogram
    n, bins, patches = ax.hist(all_strategy_cagrs, bins=30, color='cyan', 
                              alpha=0.7, edgecolor='white', linewidth=1)
    
    # Add vertical line at mean
    mean_cagr = np.mean(all_strategy_cagrs)
    median_cagr = np.median(all_strategy_cagrs)
    ax.axvline(mean_cagr, color='yellow', linestyle='--', linewidth=2, 
              label=f"Mean: {mean_cagr:.2f}%")
    ax.axvline(median_cagr, color='lime', linestyle='--', linewidth=2, 
              label=f"Median: {median_cagr:.2f}%")
    
    # Calculate and display statistics
    std_cagr = np.std(all_strategy_cagrs)
    min_cagr = np.min(all_strategy_cagrs)
    max_cagr = np.max(all_strategy_cagrs)
    
    stats_text = (
        f"Number of Strategies: {len(all_strategy_cagrs)}\n"
        f"Mean: {mean_cagr:.2f}%\n"
        f"Median: {median_cagr:.2f}%\n"
        f"Std Dev: {std_cagr:.2f}%\n"
        f"Min: {min_cagr:.2f}%\n"
        f"Max: {max_cagr:.2f}%"
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white'),
           color='white', family='monospace')
    
    ax.set_xlabel("CAGR (%)", color="white")
    ax.set_ylabel("Frequency", color="white")
    ax.set_title("Long-Short (LS) Portfolio CAGR Distribution", color="white")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    suffix_filename = f"_{plot_title_suffix.replace(' ', '_')}" if plot_title_suffix else ""
    output_path = os.path.join(OUTPUT_DISTRIBUTIONS_AGGREGATE, f"cagr_distribution{suffix_filename}.png")
    plt.savefig(output_path)
    print(f"CAGR distribution plot saved to: {output_path}")
    plt.close()
    
    print(f"\n--- CAGR Distribution Statistics ---")
    print(f"Number of Strategies: {len(all_strategy_cagrs)}")
    print(f"Mean CAGR: {mean_cagr:.2f}%")
    print(f"Median CAGR: {median_cagr:.2f}%")
    print(f"Std Dev: {std_cagr:.2f}%")
    print(f"Min CAGR: {min_cagr:.2f}%")
    print(f"Max CAGR: {max_cagr:.2f}%")
else:
    print("No CAGR values available to create distribution plot.")

# ================================================================
# MOMENT DISTRIBUTION PLOTS FOR ALL STRATEGIES (AGGREGATE)
# ================================================================
if generate_moment_distributions:
    print("\n" + "="*80)
    print("Generating aggregate moment distribution plots for all strategies...")
    print("="*80)
    
    # Define which moments to plot based on moment_distribution_level
    moments_to_plot = []
    moment_folder_map = {}  # Maps moment key to folder name
    if moment_distribution_level >= 1:
        moments_to_plot.append('CAGR (%)')
        moment_folder_map['CAGR (%)'] = 'Mean'
    if moment_distribution_level >= 2:
        moments_to_plot.append('Variance')
        moment_folder_map['Variance'] = 'Variance'
    if moment_distribution_level >= 3:
        moments_to_plot.append('Skewness')
        moment_folder_map['Skewness'] = 'Skewness'
    if moment_distribution_level >= 4:
        moments_to_plot.append('Kurtosis')
        moment_folder_map['Kurtosis'] = 'Kurtosis'
    
    # Moment labels and units
    moment_labels = {
        'CAGR (%)': 'CAGR (%)',
        'Variance': 'Variance',
        'Skewness': 'Skewness',
        'Kurtosis': 'Kurtosis'
    }
    
    # Determine which periods to show (same logic as individual plots)
    # We need to check the backtesting mode - use the same logic as in the main loop
    # For simplicity, we'll check if we have data for different periods
    periods_to_show_aggregate = []
    if 'Full Sample' in all_strategy_final_moments_by_period:
        periods_to_show_aggregate.append('Full Sample')
    if 'In-Sample' in all_strategy_final_moments_by_period:
        if 'Out-of-Sample' in all_strategy_final_moments_by_period:
            periods_to_show_aggregate = ['In-Sample', 'Out-of-Sample', 'Full Sample']
        elif 'Post-Publication' in all_strategy_final_moments_by_period:
            periods_to_show_aggregate = ['In-Sample', 'Post-Publication', 'Full Sample']
        else:
            periods_to_show_aggregate = ['In-Sample', 'Full Sample']
    else:
        periods_to_show_aggregate = ['Full Sample']
    
    # Create separate PNG for each moment
    for moment in moments_to_plot:
        # Check if we have data for any period
        has_data = False
        for period_name in periods_to_show_aggregate:
            if period_name in all_strategy_final_moments_by_period:
                if len(all_strategy_final_moments_by_period[period_name][moment]) > 0:
                    has_data = True
                    break
        
        if not has_data:
            print(f"No data available for {moment_labels[moment]}")
            continue
        
        plt.style.use("dark_background")
        num_periods = len(periods_to_show_aggregate)
        fig, axes = plt.subplots(1, num_periods, figsize=(7*num_periods, 6))
        
        # Handle axes indexing for single vs. multiple subplots
        if num_periods == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        title_suffix_text = f" - {plot_title_suffix}" if plot_title_suffix else ""
        fig.suptitle(f"Distribution of {moment_labels[moment]} Across All Strategies (LS Only){title_suffix_text}", 
                     fontsize=16, color="white", y=0.995, ha='center')
        
        for period_idx, period_name in enumerate(periods_to_show_aggregate):
            ax = axes[period_idx]
            
            # Get moment data for this period
            if period_name in all_strategy_final_moments_by_period:
                moment_data = all_strategy_final_moments_by_period[period_name][moment]
            else:
                moment_data = []
            
            if len(moment_data) == 0:
                ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                       ha='center', va='center', color='white', fontsize=12)
                ax.set_title(period_name, color="white", fontsize=12, pad=10)
                continue
            
            # Create histogram
            n, bins, patches = ax.hist(moment_data, bins=30, color='cyan', 
                                      alpha=0.7, edgecolor='white', linewidth=1)
            
            # Add vertical lines at mean and median
            mean_val = np.mean(moment_data)
            median_val = np.median(moment_data)
            ax.axvline(mean_val, color='yellow', linestyle='--', linewidth=2, 
                      label=f"Mean: {mean_val:.4f}")
            ax.axvline(median_val, color='lime', linestyle='--', linewidth=2, 
                      label=f"Median: {median_val:.4f}")
            
            # Calculate and display statistics
            std_val = np.std(moment_data)
            min_val = np.min(moment_data)
            max_val = np.max(moment_data)
            
            # Calculate additional statistics
            skewness_val = pd.Series(moment_data).skew() if len(moment_data) > 2 else 0
            kurtosis_val = pd.Series(moment_data).kurtosis() if len(moment_data) > 2 else 0
            
            stats_text = (
                f"Number of Strategies: {len(moment_data)}\n"
                f"Mean: {mean_val:.4f}\n"
                f"Median: {median_val:.4f}\n"
                f"Std Dev: {std_val:.4f}\n"
                f"Skewness: {skewness_val:.4f}\n"
                f"Kurtosis: {kurtosis_val:.4f}\n"
                f"Min: {min_val:.4f}\n"
                f"Max: {max_val:.4f}"
            )
            
            ax.set_title(period_name, color="white", fontsize=12, pad=10)
            ax.set_xlabel(moment_labels[moment], color="white")
            ax.set_ylabel("Frequency", color="white")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(fontsize=9)
            
            # Add statistics text box
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white'),
                   color='white', family='monospace')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if num_periods == 1:
            plt.subplots_adjust(top=0.90, wspace=0.3)
        else:
            plt.subplots_adjust(top=0.92, wspace=0.3)
        
        suffix_filename = f"_{plot_title_suffix.replace(' ', '_')}" if plot_title_suffix else ""
        folder_name = moment_folder_map[moment]
        output_path = os.path.join(OUTPUT_DISTRIBUTIONS_AGGREGATE_MOMENTS[folder_name], 
                                  f"{folder_name.lower()}_distribution{suffix_filename}.png")
        plt.savefig(output_path)
        print(f"  {moment_labels[moment]} distribution plot saved to: {output_path}")
        plt.close()
        
        # Print statistics
        print(f"\n--- {moment_labels[moment]} Distribution Statistics ---")
        print(f"  Number of Strategies: {len(moment_data)}")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Median: {median_val:.4f}")
        print(f"  Std Dev: {std_val:.4f}")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
else:
    print("\nMoment distribution plots are disabled (generate_moment_distributions = False).")