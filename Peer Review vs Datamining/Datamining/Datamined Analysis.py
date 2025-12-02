"""
Data-Mined Factors Analysis
Standalone script for analyzing data-mined factors with comprehensive backtests and visualizations.

This script:
1. Loads data-mined factor returns and signal names
2. Performs comprehensive backtests for selected or all strategies
3. Creates cumulative return plots with datamining cutoff and structural break markers
4. Creates rolling 5-year return plots
5. Calculates performance statistics
6. Creates distribution plots
7. Creates rolling plots aligned by months since datamining cutoff/structural break
8. Creates summary visualizations

PERFORMANCE OPTIMIZATIONS APPLIED:
1. Non-interactive matplotlib backend (Agg) for faster plotting
2. Pre-parse dates during CSV loading
3. Vectorized rolling CAGR calculations
4. Optimize matplotlib settings
5. Multiprocessing for parallel file existence checks
6. Skip already-processed signals (incremental processing)
7. Reduced print statements for faster execution
8. Configurable DPI and plot options
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for faster plotting
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
import time
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Create a dummy tqdm that just passes through the iterable
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Try to import numba for JIT compilation (optional)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

warnings.filterwarnings('ignore')

# ==============================================================================================
# USER INPUT
# ==============================================================================================
# Change this to a list of signal IDs you want to test
# This list is used if 'test_all_strategies' is set to False
# You can specify signal IDs (integers) or leave empty to test all
signals_to_test = []  # Example: [1, 2, 3, 100, 200] or leave empty for all

#==============================================================================================
# NEW TOGGLE TO TEST ALL STRATEGIES
# Set to True to automatically test all unique signals found in the data
# This will override the signals_to_test list above.
# Set to False to use the specific signals in the list above.
test_all_strategies = True
#==============================================================================================
# NEW SETTING FOR BAR GRAPH PLOTS
# If testing a large number of strategies, this will split the final bar graph
# into multiple plots, each containing this many strategies.
strategies_per_plot = 25
#==============================================================================================
# TOGGLE FOR FULL SAMPLE BACKTEST
# Set to True to run a single, full-sample backtest with no period splits.
# Set to False to split into datamining period, post-datamining, and post-structural break periods.
run_full_sample = False
#==============================================================================================
# TOGGLE FOR SIGNIFICANT FACTORS ONLY
# Set to True to only analyze factors that were significant (p ≤ 0.05) in the FDR analysis
# Set to False to analyze all factors
use_significant_only = True
#==============================================================================================
# TOGGLE FOR FLIPPING NEGATIVE STRATEGIES
# Set to True to flip returns for strategies with negative t-statistics (take opposite side)
# Set to False to use returns as-is
# If True, will flip if t_stat < 0 (regardless of significance)
# Set REQUIRE_SIGNIFICANCE_FOR_FLIP = True to only flip if also p ≤ 0.05
flip_negative_strategies = True
REQUIRE_SIGNIFICANCE_FOR_FLIP = False  # If True, only flip if t_stat < 0 AND p ≤ 0.05
#==============================================================================================
# TOGGLE FOR MOMENT DISTRIBUTION PLOTS
# Set to True to generate distribution plots of statistical moments across all strategies
# Set to False to skip moment distribution plots
generate_moment_distributions = True

# MOMENT DISTRIBUTION SETTING
# Set to 1-4 to control which moments are included in the distribution plots:
# 1 = Mean (CAGR) only
# 2 = Mean (CAGR) and Variance
# 3 = Mean (CAGR), Variance, and Skewness
# 4 = Mean (CAGR), Variance, Skewness, and Kurtosis
moment_distribution_level = 1  # OPTIMIZED: Only calculate CAGR/Mean for speed

# ROLLING WINDOW SETTING FOR MOMENT DISTRIBUTIONS
# Number of months to use for rolling window calculations of moments
rolling_window_months = 24
#==============================================================================================
# MULTIPROCESSING SETTINGS
# Number of parallel processes to use (None = use all available CPU cores)
# Set to a number (e.g., 4) to limit CPU usage
n_processes = None  # None = use all cores, or set to a number like 4, 8, etc.
#==============================================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# 
# OPTIMIZATION SUMMARY:
# - moment_distribution_level = 1: Only calculate CAGR/Mean (saves ~75% computation time)
# - skip_individual_moment_plots = True: Skip individual plots (saves ~50% plot generation time)
# - use_low_dpi = True: Lower DPI for faster plot saving (saves ~30% I/O time)
# 
# Combined effect: ~5-10x speedup for processing many signals
# 
# Set to True to skip individual moment distribution plots (saves significant time)
# Aggregate moment distributions will still be generated
# RECOMMENDED: Set to True for faster processing when testing many signals
skip_individual_moment_plots = True  # OPTIMIZED: Skip individual plots for speed
# Set to True to use lower DPI for plots (faster, smaller files)
use_low_dpi = True  # OPTIMIZED: Use lower DPI for faster plot generation
plot_dpi = 100 if use_low_dpi else 150
#==============================================================================================

# Important dates for analysis
DATAMINING_CUTOFF_DATE = pd.Timestamp('1980-01-01')  # When datamining period ended
STRUCTURAL_BREAK_DATE = pd.Timestamp('2004-01-01')  # Structural break date

# Define the file paths for your data
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RETURNS_FILE = os.path.join(SCRIPT_DIR, "DataMinedLongShortReturnsEW.csv")
SIGNAL_LIST_FILE = os.path.join(SCRIPT_DIR, "DataMinedSignalList.csv")
# FDR results are in output/fdr_analysis/results/ (new structure) or output/results/ (old structure)
FDR_RESULTS_FILE_NEW = os.path.join(SCRIPT_DIR, "output", "fdr_analysis", "results", "fdr_analysis_results.csv")
FDR_RESULTS_FILE_OLD = os.path.join(SCRIPT_DIR, "output", "results", "fdr_analysis_results.csv")
FDR_SUMMARY_FILE_NEW = os.path.join(SCRIPT_DIR, "output", "fdr_analysis", "results", "fdr_summary.csv")
FDR_SUMMARY_FILE_OLD = os.path.join(SCRIPT_DIR, "output", "results", "fdr_summary.csv")
P_TARGET = 0.05  # Significance threshold

# ================================================================
# HELPER FUNCTIONS (defined at module level BEFORE main block)
# ================================================================
def generate_signal_name(signal_form, v1, v2):
    """Generate a readable signal name from formula and variables."""
    name = signal_form.replace('v1', v1).replace('v2', v2)
    return name

def sanitize_filename(filename):
    """Sanitize a string to be safe for use as a filename by replacing invalid characters."""
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    sanitized = filename
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    return sanitized

def load_signal_names():
    """Load signal names from DataMinedSignalList.csv and create lookup dictionary."""
    try:
        signal_list = pd.read_csv(SIGNAL_LIST_FILE)
        signal_names = {}
        for _, row in signal_list.iterrows():
            signal_id = int(row['signalid'])
            signal_name = generate_signal_name(
                str(row['signal_form']),
                str(row['v1']),
                str(row['v2'])
            )
            signal_names[signal_id] = signal_name
        return signal_names
    except FileNotFoundError:
        print(f"  WARNING: Signal list file not found: {SIGNAL_LIST_FILE}")
        return {}
    except Exception as e:
        print(f"  WARNING: Error loading signal names: {e}")
        return {}

@jit(nopython=True, cache=True, nogil=True)
def _calculate_cagr_numba(returns_array):
    """Numba-optimized CAGR calculation."""
    n = len(returns_array)
    if n == 0:
        return np.nan
    if n < 12:
        return np.nan
    
    cumprod = 1.0
    for i in range(n):
        cumprod *= (1.0 + returns_array[i] / 100.0)
    
    if cumprod <= 0:
        return np.nan
    
    num_years = n / 12.0
    return ((cumprod ** (1.0 / num_years)) - 1.0) * 100.0

def calculate_cagr(returns):
    """Calculates the Compound Annual Growth Rate (CAGR) from monthly returns."""
    if len(returns) == 0:
        return np.nan
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return np.nan
    returns_array = returns_clean.values
    return _calculate_cagr_numba(returns_array)

def calculate_max_drawdown(returns):
    """Calculates the maximum drawdown from a series of monthly returns."""
    if returns.empty:
        return np.nan
    returns = returns / 100.0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min() * 100

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculates the annualized Sharpe ratio from monthly returns."""
    if returns.empty or len(returns) < 2:
        return np.nan
    returns = returns / 100.0
    excess_returns = returns - (risk_free_rate / 12.0)
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    if std_excess == 0:
        return np.nan
    return (mean_excess / std_excess) * np.sqrt(12)

def calculate_calmar_ratio(returns):
    """Calculates the Calmar ratio (CAGR / abs(max drawdown))."""
    cagr = calculate_cagr(returns)
    max_dd = calculate_max_drawdown(returns)
    if np.isnan(cagr) or np.isnan(max_dd) or max_dd == 0:
        return np.nan
    return cagr / abs(max_dd)

def calculate_rolling_moments(returns, window_months, moment_level=4):
    """Calculates rolling moments (CAGR, variance, skewness, kurtosis) over a rolling window.
    
    OPTIMIZED: Only calculates moments up to moment_level to save computation time.
    moment_level: 1=CAGR only, 2=+Variance, 3=+Skewness, 4=+Kurtosis
    """
    if returns.empty or len(returns) < window_months:
        result = {'CAGR (%)': pd.Series(dtype=float)}
        if moment_level >= 2:
            result['Variance'] = pd.Series(dtype=float)
        if moment_level >= 3:
            result['Skewness'] = pd.Series(dtype=float)
        if moment_level >= 4:
            result['Kurtosis'] = pd.Series(dtype=float)
        return result
    
    result = {}
    
    # Always calculate CAGR (moment_level >= 1)
    decimal_returns = returns / 100.0
    log_returns = np.log1p(decimal_returns)
    rolling_log_sum = log_returns.rolling(window=window_months, min_periods=window_months).sum()
    num_years = window_months / 12.0
    result['CAGR (%)'] = (np.exp(rolling_log_sum / num_years) - 1) * 100
    
    # Only calculate higher moments if needed
    if moment_level >= 2:
        result['Variance'] = returns.rolling(window=window_months, min_periods=window_months).var()
    if moment_level >= 3:
        result['Skewness'] = returns.rolling(window=window_months, min_periods=window_months).skew()
    if moment_level >= 4:
        result['Kurtosis'] = returns.rolling(window=window_months, min_periods=window_months).kurt()
    
    return result

def calculate_rolling_cagr(returns_series, window_size=60):
    """Calculates the rolling CAGR over a specified window (optimized vectorized version)."""
    decimal_returns = returns_series / 100.0
    log_returns = np.log1p(decimal_returns)
    rolling_log_sum = log_returns.rolling(window=window_size, min_periods=window_size).sum()
    cagr = (np.exp(rolling_log_sum * (12 / window_size)) - 1) * 100
    return cagr

def check_signal_files_exist(args):
    """Check if files exist for a signal - for parallel processing"""
    (signal_id, signal_name, output_dirs) = args
    safe_signal_name = sanitize_filename(signal_name).replace(" ", "_")
    
    backtest_exists = False
    trailing_exists = False
    raw_returns_exists = False
    
    # Check backtest plot
    for output_dir in output_dirs['backtest_dirs']:
        backtest_file = os.path.join(output_dir, f"backtest_plot_{safe_signal_name}.png")
        if os.path.exists(backtest_file):
            backtest_exists = True
            break
    
    # Check trailing CAGR plot
    for trailing_dir in output_dirs['trailing_dirs']:
        trailing_file = os.path.join(trailing_dir, f"trailing_cagr_{safe_signal_name}.png")
        if os.path.exists(trailing_file):
            trailing_exists = True
            break
    
    # Check raw returns CSV
    raw_returns_file = os.path.join(output_dirs['raw_returns_dir'], f"{safe_signal_name}_raw_returns.csv")
    raw_returns_exists = os.path.exists(raw_returns_file)
    
    should_skip = backtest_exists and (trailing_exists or raw_returns_exists)
    return (signal_id, should_skip)

# ================================================================
# MAIN EXECUTION - Windows multiprocessing requires this guard
# ================================================================
if __name__ == '__main__':
    # ================================================================
    # PROMPT FOR PLOT TITLE SUFFIX (needed before creating output folders)
    # ================================================================
    print("\n" + "="*80)
    print("PLOT TITLE SUFFIX")
    print("="*80)
    print("Enter a suffix to add to all plot titles (e.g., 'Top Performers', 'Significant Only', etc.)")
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
    
    # Discovery type subdirectories (organized by Expected True/False/Not Significant)
    OUTPUT_BACKTEST_TRUE = os.path.join(OUTPUT_BACKTEST, "Expected_True_Discoveries")
    OUTPUT_BACKTEST_FALSE = os.path.join(OUTPUT_BACKTEST, "Expected_False_Discoveries")
    OUTPUT_BACKTEST_NOT_SIG = os.path.join(OUTPUT_BACKTEST, "Not_Significant")
    OUTPUT_TRAILING_TRUE = os.path.join(OUTPUT_TRAILING_CAGR, "Expected_True_Discoveries")
    OUTPUT_TRAILING_FALSE = os.path.join(OUTPUT_TRAILING_CAGR, "Expected_False_Discoveries")
    OUTPUT_TRAILING_NOT_SIG = os.path.join(OUTPUT_TRAILING_CAGR, "Not_Significant")
    
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
    discovery_dirs = [OUTPUT_BACKTEST_TRUE, OUTPUT_BACKTEST_FALSE, OUTPUT_BACKTEST_NOT_SIG,
                      OUTPUT_TRAILING_TRUE, OUTPUT_TRAILING_FALSE, OUTPUT_TRAILING_NOT_SIG]
    for output_dir in base_dirs + moment_dirs + discovery_dirs:
        os.makedirs(output_dir, exist_ok=True)
    
    # ================================================================
    # STEP 1: DATA LOADING
    # ================================================================
    print("="*80)
    print("Data-Mined Factors Analysis")
    print("="*80)
    print(f"\nLoading data from '{RETURNS_FILE}'...")
    
    try:
        returns_df = pd.read_csv(RETURNS_FILE, parse_dates=False)
        returns_df['date'] = pd.to_datetime(returns_df[['year', 'month']].assign(day=1))
        required_cols = ['date', 'signalid', 'ret']
        if not all(col in returns_df.columns for col in required_cols):
            raise ValueError(f"CSV file is missing one or more required columns: {required_cols}")
    except FileNotFoundError:
        print(f"\nERROR: '{RETURNS_FILE}' not found.")
        exit()
    except Exception as e:
        print(f"\nAn error occurred while loading the CSV file: {e}")
        exit()
    
    # Sort once for better performance
    returns_df = returns_df.sort_values(['signalid', 'date'])
    
    # Load signal names
    print("\nLoading signal names...")
    signal_names_dict = load_signal_names()
    if len(signal_names_dict) > 0:
        print(f"  Loaded names for {len(signal_names_dict):,} signals")
    else:
        print("  Using signal IDs as names")
    
    def get_signal_name(signal_id):
        """Get signal name for a given signal ID, fallback to ID if not found."""
        if signal_id in signal_names_dict:
            return signal_names_dict[signal_id]
        return f"Signal {int(signal_id)}"
    
    # Load FDR results for t-statistics, filtering, and discovery classification
    # Always load FDR results to check for negative t-stats (regardless of use_significant_only)
    fdr_stats_dict = {}  # Maps signal_id to {'t_stat': float, 'p_value': float, 'discovery_type': str}
    significant_signal_ids = None
    fdr_value = None  # Overall FDR from summary
    
    print("\nLoading FDR analysis results...")
    # Try new location first, then fall back to old location
    if os.path.exists(FDR_RESULTS_FILE_NEW):
        FDR_RESULTS_FILE = FDR_RESULTS_FILE_NEW
        FDR_SUMMARY_FILE = FDR_SUMMARY_FILE_NEW
    elif os.path.exists(FDR_RESULTS_FILE_OLD):
        FDR_RESULTS_FILE = FDR_RESULTS_FILE_OLD
        FDR_SUMMARY_FILE = FDR_SUMMARY_FILE_OLD
    else:
        FDR_RESULTS_FILE = None
        FDR_SUMMARY_FILE = None
    
    try:
        if FDR_RESULTS_FILE is None:
            raise FileNotFoundError("FDR results file not found in either location")
        
        fdr_results = pd.read_csv(FDR_RESULTS_FILE)
        print(f"  Loaded FDR results from: {FDR_RESULTS_FILE}")
        
        # Check if discovery_type column exists in CSV
        has_discovery_type_column = 'discovery_type' in fdr_results.columns
        has_expected_false_column = 'expected_false' in fdr_results.columns
        
        if has_discovery_type_column:
            print(f"  Found discovery_type column in CSV - will use it for classification")
        else:
            print(f"  NOTE: discovery_type column not found in CSV - will calculate from p-values")
        
        # Load FDR summary to get overall FDR (needed for fallback calculation if CSV doesn't have discovery_type)
        try:
            if FDR_SUMMARY_FILE and os.path.exists(FDR_SUMMARY_FILE):
                fdr_summary = pd.read_csv(FDR_SUMMARY_FILE)
                fdr_value = float(fdr_summary['fdr'].iloc[0])
                pi0 = float(fdr_summary['pi0'].iloc[0])
                n_significant = int(fdr_summary['n_significant'].iloc[0])
                print(f"  Overall FDR: {fdr_value:.2%}, π₀: {pi0:.2%}, Significant factors: {n_significant:,}")
            else:
                print("  WARNING: FDR summary file not found. Discovery classification may be limited.")
                fdr_value = None
                pi0 = None
                n_significant = 0
        except Exception as e:
            print(f"  WARNING: Could not load FDR summary: {e}. Discovery classification may be limited.")
            fdr_value = None
            pi0 = None
            n_significant = 0
        
        # Read discovery_type from CSV if available (preferred method)
        # Otherwise, calculate it from p-values and FDR summary
        
        if has_discovery_type_column:
            print(f"  Using discovery_type column from FDR results CSV")
            for _, row in fdr_results.iterrows():
                signal_id = int(row['signalid'])
                p_value = float(row['p_value'])
                t_stat = float(row['t_stat'])
                
                # Use discovery_type from CSV
                discovery_type = str(row['discovery_type']) if pd.notna(row['discovery_type']) else "Unknown"
                expected_false = bool(row['expected_false']) if has_expected_false_column and pd.notna(row['expected_false']) else None
                
                fdr_stats_dict[signal_id] = {
                    't_stat': t_stat,
                    'p_value': p_value,
                    'discovery_type': discovery_type,
                    'expected_false': expected_false
                }
        elif fdr_value is not None and n_significant > 0:
            # Fallback: Calculate discovery type if column not in CSV
            print(f"  WARNING: discovery_type column not found in CSV. Calculating from p-values and FDR.")
            prop_significant = n_significant / len(fdr_results)
            for _, row in fdr_results.iterrows():
                signal_id = int(row['signalid'])
                p_value = float(row['p_value'])
                t_stat = float(row['t_stat'])
                
                # Classify discovery type
                if p_value > P_TARGET:
                    discovery_type = "Not Significant"
                    expected_false = False
                else:
                    # Calculate expected false discovery probability for this factor
                    # Using the simple bound: E[False Discoveries] = p_value * pi0 / prop_significant
                    expected_false_prob = (p_value * pi0) / prop_significant if prop_significant > 0 else 0
                    # If expected false discovery prob > 0.5, classify as expected false discovery
                    if expected_false_prob > 0.5:
                        discovery_type = "Expected False Discovery"
                        expected_false = True
                    else:
                        discovery_type = "Expected True Discovery"
                        expected_false = False
                
                fdr_stats_dict[signal_id] = {
                    't_stat': t_stat,
                    'p_value': p_value,
                    'discovery_type': discovery_type,
                    'expected_false': expected_false
                }
        else:
            # Fallback: just use p-value threshold
            print(f"  WARNING: discovery_type column not found and FDR summary unavailable. Using p-value threshold only.")
            for _, row in fdr_results.iterrows():
                signal_id = int(row['signalid'])
                p_value = float(row['p_value'])
                t_stat = float(row['t_stat'])
                
                if p_value > P_TARGET:
                    discovery_type = "Not Significant"
                else:
                    discovery_type = "Significant"  # Can't classify further without FDR summary
                
                fdr_stats_dict[signal_id] = {
                    't_stat': t_stat,
                    'p_value': p_value,
                    'discovery_type': discovery_type,
                    'expected_false': None
                }
        
        print(f"  Loaded FDR statistics for {len(fdr_stats_dict):,} signals")
        
        # Count by discovery type
        if fdr_value is not None:
            true_discoveries = sum(1 for stats in fdr_stats_dict.values() if stats['discovery_type'] == "Expected True Discovery")
            false_discoveries = sum(1 for stats in fdr_stats_dict.values() if stats['discovery_type'] == "Expected False Discovery")
            not_significant = sum(1 for stats in fdr_stats_dict.values() if stats['discovery_type'] == "Not Significant")
            print(f"  Classification: {true_discoveries:,} Expected True, {false_discoveries:,} Expected False, {not_significant:,} Not Significant")
        
        if flip_negative_strategies:
            if REQUIRE_SIGNIFICANCE_FOR_FLIP:
                # Count how many will be flipped (negative t-stat AND significant)
                negative_count = sum(1 for stats in fdr_stats_dict.values() 
                                           if stats['t_stat'] < 0 and stats['p_value'] <= P_TARGET)
                print(f"  Found {negative_count:,} strategies with negative t-stat and p ≤ {P_TARGET} (will be flipped)")
            else:
                # Count how many will be flipped (any negative t-stat)
                negative_count = sum(1 for stats in fdr_stats_dict.values() 
                                           if stats['t_stat'] < 0)
                print(f"  Found {negative_count:,} strategies with negative t-stat (will be flipped)")
        
        # If filtering to significant factors only
        # IMPORTANT: This includes BOTH Expected True AND Expected False Discoveries
        # (both are significant, but Expected False have higher probability of being false positives)
        if use_significant_only:
            significant_mask = fdr_results['p_value'] <= P_TARGET
            significant_signal_ids = set(fdr_results[significant_mask]['signalid'].unique())
            
            # Count by discovery type for significant factors
            # Prefer using CSV column if available, otherwise use fdr_stats_dict
            if has_discovery_type_column:
                significant_fdr = fdr_results[significant_mask].copy()
                true_count = len(significant_fdr[significant_fdr['discovery_type'] == 'Expected True Discovery'])
                false_count = len(significant_fdr[significant_fdr['discovery_type'] == 'Expected False Discovery'])
            else:
                # Count from fdr_stats_dict
                true_count = sum(1 for sid in significant_signal_ids 
                               if sid in fdr_stats_dict and fdr_stats_dict[sid].get('discovery_type') == 'Expected True Discovery')
                false_count = sum(1 for sid in significant_signal_ids 
                                if sid in fdr_stats_dict and fdr_stats_dict[sid].get('discovery_type') == 'Expected False Discovery')
            
            print(f"  Found {len(significant_signal_ids):,} significant factors (p ≤ {P_TARGET})")
            print(f"    - Expected True Discoveries: {true_count:,}")
            print(f"    - Expected False Discoveries: {false_count:,}")
            print(f"    - Note: Both types are included in analysis output")
    except FileNotFoundError:
        print(f"  WARNING: FDR results file not found in either location:")
        print(f"    - {FDR_RESULTS_FILE_NEW}")
        print(f"    - {FDR_RESULTS_FILE_OLD}")
        print("  Cannot flip negative strategies or filter by significance.")
        print("  Set use_significant_only = False or run FDR analysis first.")
        if use_significant_only:
            use_significant_only = False
    except Exception as e:
        print(f"  WARNING: Error loading FDR results: {e}")
        print("  Cannot flip negative strategies or filter by significance.")
        if use_significant_only:
            use_significant_only = False
    
    # Filter for the signals we're interested in
    if test_all_strategies:
        print("Testing all strategies found in the data.")
        signals_to_test = returns_df['signalid'].unique().tolist()
        df_subset = returns_df.copy()
    else:
        if len(signals_to_test) == 0:
            print("No signals specified. Testing all strategies.")
            signals_to_test = returns_df['signalid'].unique().tolist()
            df_subset = returns_df.copy()
        else:
            print(f"Testing a subset of strategies: {len(signals_to_test)} signals")
            df_subset = returns_df[returns_df['signalid'].isin(signals_to_test)].copy()
    
    # Apply significant factors filter if enabled
    if use_significant_only and significant_signal_ids is not None:
        print(f"\nFiltering to significant factors only...")
        original_count = len(signals_to_test)
        signals_to_test = [s for s in signals_to_test if s in significant_signal_ids]
        df_subset = df_subset[df_subset['signalid'].isin(signals_to_test)].copy()
        print(f"  Filtered from {original_count:,} to {len(signals_to_test):,} significant factors")
    
    if df_subset.empty:
        print(f"\nERROR: No data found for any of the signals.")
        exit()
    
    print(f"\nData for {len(signals_to_test):,} signals loaded successfully.")
    print(f"Date range: {df_subset['date'].min().strftime('%Y-%m')} to {df_subset['date'].max().strftime('%Y-%m')}")
    
    print("\nDEBUG: About to initialize storage variables...")
    
    # Storage for all results for the final summary graph
    all_cagr_results = []
    
    # Storage for all strategy CAGR values for distribution plot
    all_strategy_cagrs = []
    
    # Storage for all strategy rolling moments for distribution plots (by period)
    all_strategy_rolling_moments_by_period = {
        'Datamining Period': {
            'CAGR (%)': [],
            'Variance': [],
            'Skewness': [],
            'Kurtosis': []
        },
        'Post-Datamining': {
            'CAGR (%)': [],
            'Variance': [],
            'Skewness': [],
            'Kurtosis': []
        },
        'Post-Structural Break': {
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
    
    # Storage for final (non-rolling) statistics by period for aggregate plots
    all_strategy_final_moments_by_period = {
        'Datamining Period': {
            'CAGR (%)': [],
            'Variance': [],
            'Skewness': [],
            'Kurtosis': []
        },
        'Post-Datamining': {
            'CAGR (%)': [],
            'Variance': [],
            'Skewness': [],
            'Kurtosis': []
        },
        'Post-Structural Break': {
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
    
    # Storage for rolling 5-year growth aligned by months since datamining cutoff
    aligned_series = {}
    
    # Storage for summary of all signals with their classifications
    signals_summary = []

    print("DEBUG: Storage variables initialized. About to check which signals need processing...")

    # ================================================================
    # PRE-FILTER SIGNALS: CHECK WHICH ONES NEED PROCESSING
    # ================================================================
    print("\n" + "="*80)
    print("Checking which signals need processing...")
    print("="*80)
    
    # Prepare output directories for file checking
    output_dirs_config = {
        'backtest_dirs': [OUTPUT_BACKTEST, OUTPUT_BACKTEST_TRUE, OUTPUT_BACKTEST_FALSE, OUTPUT_BACKTEST_NOT_SIG],
        'trailing_dirs': [OUTPUT_TRAILING_CAGR, OUTPUT_TRAILING_TRUE, OUTPUT_TRAILING_FALSE, OUTPUT_TRAILING_NOT_SIG],
        'raw_returns_dir': OUTPUT_RAW_RETURNS
    }
    
    # Use multiprocessing to check file existence in parallel
    if n_processes is None:
        n_processes = cpu_count()
    
    print(f"Using {n_processes} processes for file existence checks...")
    file_check_args = [
        (signal_id, get_signal_name(signal_id), output_dirs_config)
        for signal_id in signals_to_test
    ]
    
    # Parallel file existence checks
    signals_to_skip = set()
    if len(file_check_args) > 0:
        with Pool(processes=n_processes) as pool:
            file_check_results = pool.map(check_signal_files_exist, file_check_args)
        
        for signal_id, should_skip in file_check_results:
            if should_skip:
                signals_to_skip.add(signal_id)
    
    skipped_count = len(signals_to_skip)
    signals_to_process = [s for s in signals_to_test if s not in signals_to_skip]
    
    print(f"\n" + "="*80)
    print(f"Signal Processing Summary:")
    print(f"  Total signals in data: {len(signals_to_test):,}")
    print(f"  Already processed (will skip): {skipped_count:,}")
    print(f"  Will process: {len(signals_to_process):,} signals")
    print("="*80)
    
    if len(signals_to_process) == 0:
        print("\nWARNING: No signals to process! All signals appear to be already processed.")
        print("If you want to reprocess, delete the existing output files or use a different suffix.")
        exit()
    
    if skipped_count > 0 and skipped_count <= 10:
        # Show which ones are being skipped
        for signal_id in list(signals_to_skip)[:10]:
            print(f"  Skipping: {get_signal_name(signal_id)} (ID: {signal_id})")
    
    # ================================================================
    # MAIN LOOP FOR EACH SIGNAL
    # ================================================================
    # Track processed signals
    processed_count = 0
    start_time = time.time()
    
    # Optimize matplotlib settings globally
    plt.style.use("dark_background")
    plt.ioff()
    matplotlib.rcParams['path.simplify'] = True
    matplotlib.rcParams['path.simplify_threshold'] = 1.0
    matplotlib.rcParams['figure.max_open_warning'] = 0  # Suppress warning about too many open figures
    
    # Create progress bar
    if HAS_TQDM:
        pbar = tqdm(signals_to_process, desc="Processing signals", unit="signal", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        pbar = signals_to_process
        print(f"\nProcessing {len(signals_to_process):,} signals...")
    
    for signal_id in pbar:
        signal_name = get_signal_name(signal_id)
        processed_count += 1
        
        # Update progress bar description with current signal
        if HAS_TQDM:
            pbar.set_postfix_str(f"Current: {signal_name[:30]}...")
        
        # Detailed progress update every 100 signals or first 5
        if processed_count <= 5 or processed_count % 100 == 0:
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            remaining = len(signals_to_process) - processed_count
            eta = remaining / rate if rate > 0 else 0
            if not HAS_TQDM or processed_count % 100 == 0:
                print(f"\nProgress: {processed_count:,}/{len(signals_to_process):,} signals processed "
                      f"({rate:.1f} signals/sec, ETA: {eta/60:.1f} min)")
        
        if processed_count <= 5 or processed_count % 100 == 0:
            print("\n" + "="*80)
            print(f"Processing signal: {signal_name} (ID: {signal_id}) [{processed_count:,}/{len(signals_to_process):,}]")
            print("="*80)
        
        # Filter signal data
        signal_df = df_subset[df_subset['signalid'] == signal_id].copy()
        
        if signal_df.empty:
            print(f"WARNING: No data found for signal ID {signal_id}. Skipping.")
            continue

        # ================================================================
        # STEP 1.5: DEFINE PERIODS
        # ================================================================
        start_date = signal_df['date'].min()
        end_date = signal_df['date'].max()
        if processed_count <= 5 or processed_count % 100 == 0:
            print(f"Data starts on: {start_date.strftime('%Y-%m')} and ends on: {end_date.strftime('%Y-%m')}")

        if run_full_sample:
            if processed_count <= 5 or processed_count % 100 == 0:
                print("\nRunning full-sample backtest. All period splits are disabled.")
            datamining_start_date = start_date
            datamining_end_date = end_date
            post_datamining_start_date = end_date + pd.Timedelta(days=1)
            post_structural_break_start_date = end_date + pd.Timedelta(days=1)
            periods_to_show = ['Full Sample']
        else:
            # Define periods based on datamining cutoff and structural break
            # Datamining period: up to 1980 (use actual data range if it starts after 1980)
            datamining_start_date = start_date
            datamining_end_date = min(DATAMINING_CUTOFF_DATE, end_date)
            # Only include datamining period if we have data before the cutoff
            if start_date < DATAMINING_CUTOFF_DATE:
                datamining_period_available = True
            else:
                datamining_period_available = False
                if processed_count <= 5 or processed_count % 100 == 0:
                    print(f"  Note: Data starts after datamining cutoff ({DATAMINING_CUTOFF_DATE.year}). Skipping datamining period.")
            
            # Post-datamining: 1980 to 2004
            post_datamining_start_date = max(DATAMINING_CUTOFF_DATE + pd.Timedelta(days=1), start_date)
            post_datamining_end_date = min(STRUCTURAL_BREAK_DATE, end_date)
            # Post-structural break: after 2004
            post_structural_break_start_date = max(STRUCTURAL_BREAK_DATE + pd.Timedelta(days=1), start_date)
            
            # Build periods list based on available data
            periods_to_show = []
            if datamining_period_available and start_date < DATAMINING_CUTOFF_DATE:
                periods_to_show.append('Datamining Period')
            if post_datamining_start_date <= post_datamining_end_date:
                periods_to_show.append('Post-Datamining')
            if post_structural_break_start_date <= end_date:
                periods_to_show.append('Post-Structural Break')
            periods_to_show.append('Full Sample')
        
        # ================================================================
        # STEP 1.6: CHECK DATA SUFFICIENCY (minimum 5 years = 60 months per period)
        # ================================================================
        MIN_MONTHS_PER_PERIOD = 60  # 5 years minimum
        
        if not run_full_sample:
            # Check each period has sufficient data
            insufficient_periods = []
            
            if 'Datamining Period' in periods_to_show:
                datamining_months = ((datamining_end_date.year - datamining_start_date.year) * 12 + 
                                    (datamining_end_date.month - datamining_start_date.month) + 1)
                if datamining_months < MIN_MONTHS_PER_PERIOD:
                    insufficient_periods.append(f'Datamining Period ({datamining_months} months)')
            
            if 'Post-Datamining' in periods_to_show:
                post_datamining_months = ((post_datamining_end_date.year - post_datamining_start_date.year) * 12 + 
                                         (post_datamining_end_date.month - post_datamining_start_date.month) + 1)
                if post_datamining_months < MIN_MONTHS_PER_PERIOD:
                    insufficient_periods.append(f'Post-Datamining ({post_datamining_months} months)')
            
            if 'Post-Structural Break' in periods_to_show:
                post_break_months = ((end_date.year - post_structural_break_start_date.year) * 12 + 
                                    (end_date.month - post_structural_break_start_date.month) + 1)
                if post_break_months < MIN_MONTHS_PER_PERIOD:
                    insufficient_periods.append(f'Post-Structural Break ({post_break_months} months)')
            
            # Check full sample
            full_sample_months = ((end_date.year - start_date.year) * 12 + 
                                 (end_date.month - start_date.month) + 1)
            if full_sample_months < MIN_MONTHS_PER_PERIOD:
                insufficient_periods.append(f'Full Sample ({full_sample_months} months)')
            
            if insufficient_periods:
                if processed_count <= 5 or processed_count % 100 == 0:
                    print(f"  SKIPPING: Insufficient data - need at least {MIN_MONTHS_PER_PERIOD} months (5 years) per period")
                    print(f"    Insufficient periods: {', '.join(insufficient_periods)}")
                continue  # Skip this signal
        
        # ================================================================
        # STEP 2: PREPARE SIGNAL DATA
        # ================================================================
        signal_df = signal_df.sort_values("date")
        signal_df["ret_pct"] = signal_df["ret"]

        # Get returns series
        returns_series = signal_df.set_index('date')['ret_pct'].sort_index()
        
        # Get discovery classification and FDR stats
        discovery_type = "Unknown"
        expected_false = None
        p_value = np.nan
        t_stat = np.nan
        
        if signal_id in fdr_stats_dict:
            stats = fdr_stats_dict[signal_id]
            discovery_type = stats.get('discovery_type', 'Unknown')
            expected_false = stats.get('expected_false', None)
            p_value = stats['p_value']
            t_stat = stats['t_stat']
            
            # Print discovery classification (only for first few or periodically)
            if processed_count <= 5 or processed_count % 100 == 0:
                status_icon = "✓" if discovery_type == "Expected True Discovery" else "⚠" if discovery_type == "Expected False Discovery" else "○"
                print(f"  [{status_icon}] {discovery_type} (p={p_value:.4f}, t={t_stat:.2f})")
            
            # Store for summary
            signals_summary.append({
                'signal_id': signal_id,
                'signal_name': signal_name,
                'discovery_type': discovery_type,
                'expected_false': expected_false,
                'p_value': p_value,
                't_stat': t_stat,
                'significant': p_value <= P_TARGET
            })
        
        # Check if we need to flip returns (negative t-stat)
        # Flip if the strategy has a negative t-statistic (takes opposite side of trade)
        if flip_negative_strategies and signal_id in fdr_stats_dict:
            
            # Determine if we should flip based on configuration
            if REQUIRE_SIGNIFICANCE_FOR_FLIP:
                # Only flip if negative t-stat AND significant (p ≤ 0.05)
                should_flip = (t_stat < 0 and p_value <= P_TARGET)
                if should_flip:
                    if processed_count <= 5 or processed_count % 100 == 0:
                        print(f"  Flipping returns (t-stat = {t_stat:.2f}, p-value = {p_value:.4f} ≤ {P_TARGET})")
                    returns_series = -returns_series  # Take the opposite side of the trade
            else:
                # Flip if negative t-stat (regardless of significance)
                should_flip = (t_stat < 0)
                if should_flip:
                    if processed_count <= 5 or processed_count % 100 == 0:
                        print(f"  Flipping returns (t-stat = {t_stat:.2f})")
                    returns_series = -returns_series  # Take the opposite side of the trade
        elif flip_negative_strategies and signal_id not in fdr_stats_dict:
            if processed_count <= 5 or processed_count % 100 == 0:
                print(f"  WARNING: No FDR statistics found for signal {signal_id}. Using returns as-is.")
        
        # ================================================================
        # STEP 3: CALCULATE CUMULATIVE RETURNS
        # ================================================================
        cumulative_returns = (1 + returns_series / 100).cumprod()

        # ================================================================
        # STEP 4: PLOTTING CUMULATIVE RETURNS
        # ================================================================
        # Matplotlib settings already configured globally above

        if run_full_sample:
            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            title_suffix_text = f" - {plot_title_suffix}" if plot_title_suffix else ""
            discovery_label = f" [{discovery_type}]" if discovery_type != "Unknown" else ""
            fig.suptitle(f"Full-Sample Cumulative Returns: {signal_name}{discovery_label}{title_suffix_text}", fontsize=16, color="white")

            cumulative_returns_norm = cumulative_returns / cumulative_returns.iloc[0]
            ax.plot(cumulative_returns_norm.index, cumulative_returns_norm, label=signal_name, color='cyan', linewidth=2)
            ax.set_title("Full Sample Period", color="white")
            ax.set_xlabel("Date", color="white")
            ax.set_ylabel("Cumulative Return (Normalized)", color="white")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_yscale('log')
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            # Sanitize signal name for filename
            safe_signal_name = sanitize_filename(signal_name).replace(" ", "_")
            
            # Organize by discovery type if available
            if discovery_type == "Expected True Discovery":
                output_dir = OUTPUT_BACKTEST_TRUE
            elif discovery_type == "Expected False Discovery":
                output_dir = OUTPUT_BACKTEST_FALSE
            elif discovery_type == "Not Significant":
                output_dir = OUTPUT_BACKTEST_NOT_SIG
            else:
                output_dir = OUTPUT_BACKTEST
            
            output_path = os.path.join(output_dir, f"backtest_full_sample_plot_{safe_signal_name}.png")
            try:
                plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
                plt.close()
                if processed_count <= 5 or processed_count % 100 == 0:
                    print(f"Backtest plot saved to: {output_path}")
            except Exception as e:
                print(f"ERROR saving backtest plot for {signal_name}: {e}")
                plt.close()

        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 7))
            title_suffix_text = f" - {plot_title_suffix}" if plot_title_suffix else ""
            discovery_label = f" [{discovery_type}]" if discovery_type != "Unknown" else ""
            fig.suptitle(f"Cumulative Returns Backtest: {signal_name}{discovery_label}{title_suffix_text}", fontsize=16, color="white")

            # Normalize the entire cumulative returns series by its first value for continuity
            cumulative_returns_norm = cumulative_returns / cumulative_returns.iloc[0]
            
            # Plot the entire series as one continuous line
            ax.plot(cumulative_returns_norm.index, cumulative_returns_norm, label=signal_name, color='cyan', linewidth=2)
            
            # Add vertical dotted lines for datamining cutoff and structural break
            if start_date < DATAMINING_CUTOFF_DATE:
                ax.axvline(DATAMINING_CUTOFF_DATE, color='orange', linestyle='--', linewidth=1.5, label=f"Datamining Cutoff ({DATAMINING_CUTOFF_DATE.year})")
            if start_date < STRUCTURAL_BREAK_DATE:
                ax.axvline(STRUCTURAL_BREAK_DATE, color='gray', linestyle='--', linewidth=1.5, label="Structural Break (2004)")
            
            ax.set_xlabel("Date", color="white")
            ax.set_ylabel("Cumulative Return (Normalized)", color="white")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_yscale('log')

            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            # Sanitize signal name for filename
            safe_signal_name = sanitize_filename(signal_name).replace(" ", "_")
            
            # Organize by discovery type if available
            if discovery_type == "Expected True Discovery":
                output_dir = OUTPUT_BACKTEST_TRUE
            elif discovery_type == "Expected False Discovery":
                output_dir = OUTPUT_BACKTEST_FALSE
            elif discovery_type == "Not Significant":
                output_dir = OUTPUT_BACKTEST_NOT_SIG
            else:
                output_dir = OUTPUT_BACKTEST
            
            output_path = os.path.join(output_dir, f"backtest_plot_{safe_signal_name}.png")
            try:
                plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
                plt.close()
                if processed_count <= 5 or processed_count % 100 == 0:
                    print(f"Backtest plot saved to: {output_path}")
            except Exception as e:
                print(f"ERROR saving backtest plot for {signal_name}: {e}")
                plt.close()

        # ================================================================
        # STEP 5: PLOTTING ROLLING 5-YEAR RETURNS
        # ================================================================
        if processed_count <= 5 or processed_count % 100 == 0:
            print("Generating rolling 5-year return plot...")

        portfolio_rolling_cagr = calculate_rolling_cagr(returns_series)

        fig, ax = plt.subplots(figsize=(12, 7))
        title_suffix_text = f" - {plot_title_suffix}" if plot_title_suffix else ""
        discovery_label = f" [{discovery_type}]" if discovery_type != "Unknown" else ""
        if run_full_sample:
            fig.suptitle(f"Trailing 5-Year CAGR: {signal_name}{discovery_label} (Full Sample){title_suffix_text}", fontsize=16, color="white")
            ax.set_title("Entire Backtesting Period", color="white")
        else:
            fig.suptitle(f"Trailing 5-Year CAGR: {signal_name}{discovery_label} (Continuous Plot){title_suffix_text}", fontsize=16, color="white")
            ax.axvline(x=DATAMINING_CUTOFF_DATE, color='orange', linestyle='--', linewidth=2, label=f'Datamining Cutoff ({DATAMINING_CUTOFF_DATE.year})')
            ax.axvline(x=STRUCTURAL_BREAK_DATE, color='gray', linestyle='--', linewidth=2, label='Structural Break (2004)')

        ax.plot(portfolio_rolling_cagr.index, portfolio_rolling_cagr, label=signal_name, color='cyan', linewidth=2)

        # Add text labels
        if not run_full_sample:
            y_top = ax.get_ylim()[1]
            ax.text(DATAMINING_CUTOFF_DATE, y_top * 0.9, f'Datamining Cutoff ({DATAMINING_CUTOFF_DATE.year})', color='orange', ha='right', fontsize=9)
            ax.text(STRUCTURAL_BREAK_DATE, y_top * 0.9, 'Structural Break (2004)', color='white', ha='left', fontsize=9)

        ax.set_xlabel("Date", color="white")
        ax.set_ylabel("Trailing 5-Year CAGR (%)", color="white")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

        # Sanitize signal name for filename
        safe_signal_name = sanitize_filename(signal_name).replace(" ", "_")
        
        # Organize by discovery type if available
        if discovery_type == "Expected True Discovery":
            trailing_dir = OUTPUT_TRAILING_TRUE
        elif discovery_type == "Expected False Discovery":
            trailing_dir = OUTPUT_TRAILING_FALSE
        elif discovery_type == "Not Significant":
            trailing_dir = OUTPUT_TRAILING_NOT_SIG
        else:
            trailing_dir = OUTPUT_TRAILING_CAGR
        
        output_path = os.path.join(trailing_dir, f"trailing_cagr_{safe_signal_name}.png")
        try:
            plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
            plt.close()
            # Verify file was created
            if os.path.exists(output_path):
                if processed_count <= 5 or processed_count % 100 == 0:
                    if run_full_sample:
                        print(f"Full sample rolling 5-year return plot saved to: {output_path}")
                    else:
                        print(f"Continuous rolling 5-year return plot saved to: {output_path}")
            else:
                print(f"ERROR: Trailing CAGR plot file not created: {output_path}")
        except Exception as e:
            print(f"ERROR saving trailing CAGR plot for {signal_name}: {e}")
            import traceback
            traceback.print_exc()
            plt.close()

        # ================================================================
        # STEP 6: PERFORMANCE TABLE
        # ================================================================
        # OPTIMIZED: Cache period returns once to avoid repeated filtering
        period_returns_dict = {}
        if not run_full_sample:
            if datamining_period_available:
                period_returns_dict['Datamining Period'] = returns_series[
                    (returns_series.index >= datamining_start_date) & 
                    (returns_series.index <= datamining_end_date)
                ].dropna()
            period_returns_dict['Post-Datamining'] = returns_series[
                (returns_series.index >= post_datamining_start_date) & 
                (returns_series.index <= post_datamining_end_date)
            ].dropna()
            period_returns_dict['Post-Structural Break'] = returns_series[
                returns_series.index >= post_structural_break_start_date
            ].dropna()
        period_returns_dict['Full Sample'] = returns_series.dropna()
        
        if run_full_sample:
            full_sample_cagr = calculate_cagr(period_returns_dict['Full Sample'])
            if processed_count <= 5 or processed_count % 100 == 0:
                print(f"\n--- Full Sample Performance (Annual CAGR) for {signal_name} ---")
                print(f"Period: {start_date.strftime('%Y-%m')} → {end_date.strftime('%Y-%m')}")
                print(f"Full Sample CAGR: {full_sample_cagr:.2f}%")
            all_cagr_results.append({
                'signal': signal_name,
                'signal_id': signal_id,
                'Full Sample CAGR': full_sample_cagr
            })
        else:
            if processed_count <= 5 or processed_count % 100 == 0:
                print(f"\n--- Multi-Period Performance (Annual CAGR) for {signal_name} ---")
                if datamining_period_available:
                    print(f"Datamining Period: {datamining_start_date.strftime('%Y-%m')} → {datamining_end_date.strftime('%Y-%m')}")
                print(f"Post-Datamining: {post_datamining_start_date.strftime('%Y-%m')} → {post_datamining_end_date.strftime('%Y-%m')}")
                print(f"Post-Structural Break: {post_structural_break_start_date.strftime('%Y-%m')} → {end_date.strftime('%Y-%m')}")

            # OPTIMIZED: Use cached period returns
            datamining_cagr = calculate_cagr(period_returns_dict.get('Datamining Period', pd.Series(dtype=float))) if 'Datamining Period' in period_returns_dict else np.nan
            post_datamining_cagr = calculate_cagr(period_returns_dict.get('Post-Datamining', pd.Series(dtype=float))) if 'Post-Datamining' in period_returns_dict else np.nan
            post_structural_break_cagr = calculate_cagr(period_returns_dict.get('Post-Structural Break', pd.Series(dtype=float))) if 'Post-Structural Break' in period_returns_dict else np.nan
            full_sample_cagr = calculate_cagr(period_returns_dict['Full Sample'])

            result_dict = {
                'signal': signal_name,
                'signal_id': signal_id,
                'Full Sample CAGR': full_sample_cagr
            }
            if datamining_period_available:
                result_dict['Datamining Period CAGR'] = datamining_cagr
                if processed_count <= 5 or processed_count % 100 == 0:
                    print(f"Datamining Period CAGR: {datamining_cagr:.2f}%")
            if processed_count <= 5 or processed_count % 100 == 0:
                print(f"Post-Datamining CAGR: {post_datamining_cagr:.2f}%")
                print(f"Post-Structural Break CAGR: {post_structural_break_cagr:.2f}%")
                print(f"Full Sample CAGR: {full_sample_cagr:.2f}%")
            
            if not datamining_period_available:
                result_dict['Post-Datamining CAGR'] = post_datamining_cagr
                result_dict['Post-Structural Break CAGR'] = post_structural_break_cagr
            else:
                result_dict['Datamining Period CAGR'] = datamining_cagr
                result_dict['Post-Datamining CAGR'] = post_datamining_cagr
                result_dict['Post-Structural Break CAGR'] = post_structural_break_cagr
            
            all_cagr_results.append(result_dict)

        # ================================================================
        # STEP 6.5: STATISTICS CALCULATION
        # ================================================================
        if processed_count <= 5 or processed_count % 100 == 0:
            print("\nCalculating statistics...")
        
        all_stats = {}
        
        # OPTIMIZED: Reuse period_returns_dict already calculated above in STEP 6
        
        # Process each period
        for period_name, period_returns_clean in period_returns_dict.items():
            # Skip if period has no data
            if period_returns_clean.empty or len(period_returns_clean) == 0:
                continue
            
            # Only calculate detailed stats if period is in periods_to_show
            # But always collect aggregate data if period has valid data
            should_calculate_stats = period_name in periods_to_show
            
            # OPTIMIZED: Only calculate statistics we actually need
            mean_ret = period_returns_clean.mean()
            cagr = calculate_cagr(period_returns_clean)
            
            # Only calculate higher moments if needed
            variance = period_returns_clean.var() if moment_distribution_level >= 2 else np.nan
            skewness = period_returns_clean.skew() if moment_distribution_level >= 3 else np.nan
            kurtosis = period_returns_clean.kurtosis() if moment_distribution_level >= 4 else np.nan
            
            # Always calculate risk metrics (needed for stats table)
            max_dd = calculate_max_drawdown(period_returns_clean)
            sharpe = calculate_sharpe_ratio(period_returns_clean)
            calmar = calculate_calmar_ratio(period_returns_clean)
            
            all_stats[period_name] = {
                'returns': period_returns_clean,
                'Mean (%)': mean_ret,
                'Variance': variance,
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'Max Drawdown (%)': max_dd,
                'Sharpe Ratio': sharpe,
                'Calmar Ratio': calmar,
                'CAGR (%)': cagr
            }
            
            # Store CAGR for final distribution plot (use post-structural break or full sample)
            if not np.isnan(cagr) and period_name in ['Post-Structural Break', 'Full Sample']:
                all_strategy_cagrs.append(cagr)
            
            # Calculate and store moments for moment distribution plots
            if generate_moment_distributions:
                # OPTIMIZED: Only calculate rolling moments if we need them for individual plots
                # For aggregate plots, we only need the final (non-rolling) statistics
                if not skip_individual_moment_plots:
                    rolling_moments = calculate_rolling_moments(period_returns_clean, rolling_window_months, moment_distribution_level)
                    
                    # Store rolling values by period for individual strategy plots
                    if period_name in all_strategy_rolling_moments_by_period:
                        # Only store moments we actually calculated
                        for moment_name in rolling_moments.keys():
                            rolling_values = rolling_moments[moment_name].dropna().tolist()
                            all_strategy_rolling_moments_by_period[period_name][moment_name].extend(rolling_values)
                
                # Store FINAL (non-rolling) statistics for aggregate plots (always needed)
                if period_name in all_strategy_final_moments_by_period:
                    if not np.isnan(cagr):
                        all_strategy_final_moments_by_period[period_name]['CAGR (%)'].append(cagr)
                    # Only store other moments if we need them
                    if moment_distribution_level >= 2 and not np.isnan(variance):
                        all_strategy_final_moments_by_period[period_name]['Variance'].append(variance)
                    if moment_distribution_level >= 3 and not np.isnan(skewness):
                        all_strategy_final_moments_by_period[period_name]['Skewness'].append(skewness)
                    if moment_distribution_level >= 4 and not np.isnan(kurtosis):
                        all_strategy_final_moments_by_period[period_name]['Kurtosis'].append(kurtosis)
        
        # Print statistics tables (only for first few or periodically)
        if len(all_stats) > 0 and (processed_count <= 5 or processed_count % 100 == 0):
            print(f"\n--- Statistics for {signal_name} ---")
            for period_name in periods_to_show:
                if period_name in all_stats:
                    stats = all_stats[period_name]
                    print(f"  {period_name}:")
                    print(f"    Mean: {stats['Mean (%)']:.4f}%, Variance: {stats['Variance']:.4f}, "
                          f"Skewness: {stats['Skewness']:.4f}, Kurtosis: {stats['Kurtosis']:.4f}")
                    print(f"    Max DD: {stats['Max Drawdown (%)']:.4f}%, Sharpe: {stats['Sharpe Ratio']:.4f}, "
                          f"Calmar: {stats['Calmar Ratio']:.4f}, CAGR: {stats['CAGR (%)']:.4f}%")
        
        # ================================================================
        # INDIVIDUAL STRATEGY MOMENT DISTRIBUTION PLOTS
        # NOTE: These are expensive to generate. Consider setting skip_individual_moment_plots = True
        # to speed up processing. Aggregate distributions will still be generated.
        # ================================================================
        if generate_moment_distributions and len(all_stats) > 0 and not skip_individual_moment_plots:
            if processed_count <= 5 or processed_count % 100 == 0:
                print(f"\nGenerating individual moment distribution plots for {signal_name}...")
            
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
            
            # Get rolling moments for this strategy
            strategy_rolling_moments_by_period = {}
            for period_name in periods_to_show:
                if period_name in all_stats:
                    period_returns = all_stats[period_name]['returns']
                    rolling_moments = calculate_rolling_moments(period_returns, rolling_window_months, moment_distribution_level)
                    strategy_rolling_moments_by_period[period_name] = rolling_moments
            
            # Create one PNG per moment
            for moment in moments_to_plot_individual:
                num_periods = len(periods_to_show)
                
                plt.style.use("dark_background")
                fig, axes = plt.subplots(1, num_periods, figsize=(7*num_periods, 5.5))
                
                # Handle axes indexing
                if num_periods == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
                
                title_suffix_text = f" - {plot_title_suffix}" if plot_title_suffix else ""
                fig.suptitle(f"{signal_name} - Distribution of {moment_labels[moment]} Across All Strategies{title_suffix_text}", 
                             fontsize=16, color="white", y=0.995, ha='center')
                
                for period_idx, period_name in enumerate(periods_to_show):
                    ax = axes[period_idx]
                    
                    # Get distribution data for this period across all strategies
                    if period_name in all_strategy_rolling_moments_by_period:
                        moment_data = all_strategy_rolling_moments_by_period[period_name][moment]
                    else:
                        moment_data = []
                    
                    # Get this strategy's rolling values for this period
                    if period_name in strategy_rolling_moments_by_period:
                        strategy_rolling_values = strategy_rolling_moments_by_period[period_name][moment].dropna().tolist()
                    else:
                        strategy_rolling_values = []
                    
                    if len(moment_data) == 0:
                        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                               ha='center', va='center', color='white', fontsize=12)
                        ax.set_title(period_name, color="white", fontsize=11, pad=10)
                        continue
                    
                    # Create histogram
                    n, bins, patches = ax.hist(moment_data, bins=30, color='cyan', 
                                              alpha=0.7, edgecolor='white', linewidth=1)
                    
                    # Highlight this strategy's rolling values
                    if len(strategy_rolling_values) > 0:
                        strategy_mean = np.mean(strategy_rolling_values)
                        ax.axvline(strategy_mean, color='yellow', linestyle='-', linewidth=3, 
                                  label=f"{signal_name} Mean: {strategy_mean:.4f}", zorder=10)
                        strategy_min = np.min(strategy_rolling_values)
                        strategy_max = np.max(strategy_rolling_values)
                        ax.axvspan(strategy_min, strategy_max, alpha=0.2, color='yellow', 
                                  label=f"{signal_name} Range: [{strategy_min:.4f}, {strategy_max:.4f}]")
                    
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
                    
                    ax.set_title(period_name, color="white", fontsize=11, pad=10)
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
                if num_periods == 1:
                    plt.subplots_adjust(top=0.90, wspace=0.3)
                else:
                    plt.subplots_adjust(top=0.92, wspace=0.3)
                
                suffix_filename = f"_{sanitize_filename(plot_title_suffix).replace(' ', '_')}" if plot_title_suffix else ""
                folder_name = moment_folder_map[moment]
                # Sanitize signal name for filename
                safe_signal_name = sanitize_filename(signal_name).replace(" ", "_")
                output_path = os.path.join(OUTPUT_DISTRIBUTIONS_INDIVIDUAL_MOMENTS[folder_name], 
                                          f"{safe_signal_name}_{folder_name.lower()}_distribution{suffix_filename}.png")
                plt.savefig(output_path, dpi=plot_dpi, bbox_inches='tight')
                plt.close()
                if processed_count <= 5 or processed_count % 100 == 0:
                    print(f"  {moment_labels[moment]} distribution plot for {signal_name} saved to: {output_path}")

        # ================================================================
        # STEP 7: BUILD ALIGNED SERIES FOR ROLLING PLOTS
        # ================================================================
        # Calculate rolling 5-year multiplicative growth and align by months since datamining cutoff
        if not run_full_sample and len(returns_series) >= 60:
            decimal_returns = returns_series / 100.0
            log_returns = np.log1p(decimal_returns)
            rolling_log_sum = log_returns.rolling(window=60, min_periods=60).sum()
            rolling_growth = np.exp(rolling_log_sum)
            rolling_growth = rolling_growth.dropna()
            
            # Align by months since datamining cutoff
            datamining_year_month = DATAMINING_CUTOFF_DATE.year * 12 + DATAMINING_CUTOFF_DATE.month
            months_since_datamining = (rolling_growth.index.year * 12 + rolling_growth.index.month) - datamining_year_month
            
            if len(months_since_datamining) > 0:
                aligned_series[signal_id] = pd.Series(rolling_growth.values, index=months_since_datamining)

        # ================================================================
        # STEP 8: SAVE RAW RETURNS TO CSV
        # ================================================================
        raw_returns = returns_series.to_frame('ret_pct')
        # Sanitize signal name for filename
        safe_signal_name = sanitize_filename(signal_name).replace(" ", "_")
        output_file = os.path.join(OUTPUT_RAW_RETURNS, f"{safe_signal_name}_raw_returns.csv")
        raw_returns.to_csv(output_file, index_label="Date")
        if processed_count <= 5 or processed_count % 100 == 0:
            print(f"Raw returns exported to: {output_file}")
    
    # ================================================================
    # SAVE SIGNALS SUMMARY (after all signals processed)
    # ================================================================
    if len(signals_summary) > 0:
        summary_df_all = pd.DataFrame(signals_summary)
        summary_file = os.path.join(OUTPUT_SUMMARY, "signals_classification_summary.csv")
        summary_df_all.to_csv(summary_file, index=False)
        print(f"\nSignals classification summary saved to: {summary_file}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("SIGNALS CLASSIFICATION SUMMARY")
        print("="*80)
        if 'discovery_type' in summary_df_all.columns:
            print("\nBy Discovery Type:")
            print(summary_df_all['discovery_type'].value_counts().to_string())
        if 'significant' in summary_df_all.columns:
            print(f"\nSignificant (p ≤ {P_TARGET}): {summary_df_all['significant'].sum():,} / {len(summary_df_all):,}")
    
    # ================================================================
    # FINAL SUMMARY BAR GRAPH
    # ================================================================
    print("\n" + "="*80)
    print("Generating final summary bar graph for all strategies...")
    print("="*80)
    
    summary_df = pd.DataFrame(all_cagr_results)
    
    if not summary_df.empty:
        summary_df = summary_df.set_index('signal')
    if 'Full Sample CAGR' in summary_df.columns:
        summary_df = summary_df[['Full Sample CAGR']]
    else:
        summary_df = summary_df[['Datamining Period CAGR', 'Post-Datamining CAGR', 'Post-Structural Break CAGR', 'Full Sample CAGR']]

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
        
        if end_index > num_strategies:
            end_index = num_strategies
        
        chunk_count = end_index - start_index
        
        if num_strategies <= strategies_per_plot:
            title_range = f"(1-{num_strategies})"
        else:
            title_range = f"({start_index+1}-{end_index})"
        
        part_text = f" (Part {i+1})" if num_plots > 1 else ""
        ax.set_title(f"Annual CAGR {title_range}{part_text}{title_suffix_text}", fontsize=16, color="white")
        ax.set_xlabel("Strategy Signal", color="white")
        ax.set_ylabel("Annual CAGR (%)", color="white")
        ax.grid(axis='y', linestyle="--", alpha=0.4)
        ax.legend(title="Performance Period")
        plt.tight_layout()

        if num_plots > 1:
            output_file = os.path.join(OUTPUT_SUMMARY, f"final_cagr_summary_part_{i+1}.png")
        else:
            output_file = os.path.join(OUTPUT_SUMMARY, "final_cagr_summary.png")
        plt.savefig(output_file)
        print(f"Summary plot saved to: {output_file}")
        plt.close()
    
    else:
        print("No data found to create a summary graph.")
    
    # ======================================================================
    # ROLLING PLOTS ALIGNED BY MONTHS SINCE DATAMINING CUTOFF
    # ======================================================================
    min_fraction = 0.3  # Only plot months where >=30% of strategies have data
    selected_signals = signals_to_test
    
    if len(aligned_series) == 0:
        print("\nNo usable aligned series found for rolling plots.")
    else:
        # Filter to only selected signals
        aligned_series_filtered = {s: ser for s, ser in aligned_series.items() if s in selected_signals}

        if not aligned_series_filtered:
            print("No aligned series found for the selected signals.")
        else:
            # Combine all selected strategies into a single DataFrame aligned by Months Since Datamining Cutoff
            all_months = sorted(set().union(*[s.index.values for s in aligned_series_filtered.values()]))
            aligned_series_list = []
            for s, ser in aligned_series_filtered.items():
                reindexed_ser = ser.reindex(all_months)
                reindexed_ser.name = s
                aligned_series_list.append(reindexed_ser)
            aligned_df = pd.concat(aligned_series_list, axis=1)
            
            # Apply coverage mask
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
            fig.suptitle(f"Rolling 5-Year Multiplicative Growth (×) Aligned by Months Since Datamining Cutoff{title_suffix_text}",
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
            
            # Datamining cutoff marker (at month 0)
            ax.axvline(0, color="orange", linestyle="--", linewidth=1.5, label=f"Datamining Cutoff ({DATAMINING_CUTOFF_DATE.year})")
            if not composite_masked.dropna().empty:
                y_top = composite_masked.dropna().max()
                ax.text(0, y_top * 0.95, f"Datamining Cutoff ({DATAMINING_CUTOFF_DATE.year})", color="orange", ha="left", va="top")
            
            # Structural break marker
            structural_break_year_month = STRUCTURAL_BREAK_DATE.year * 12 + STRUCTURAL_BREAK_DATE.month
            datamining_year_month = DATAMINING_CUTOFF_DATE.year * 12 + DATAMINING_CUTOFF_DATE.month
            months_at_structural_break = structural_break_year_month - datamining_year_month
            ax.axvline(months_at_structural_break, color="gray", linestyle="--", linewidth=1.5, label="Structural Break (2004)")
            if not composite_masked.dropna().empty:
                y_top = composite_masked.dropna().max()
                ax.text(months_at_structural_break, y_top * 0.90, "Structural Break (2004)", color="white", ha="left", va="top")
            
            ax.set_yscale("log")
            ax.set_ylabel("Rolling 5-Year Growth (×)", color="white")
            ax.set_xlabel("Months Since Datamining Cutoff", color="white")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend()
            plt.tight_layout()
            suffix_filename = f"_{sanitize_filename(plot_title_suffix).replace(' ', '_')}" if plot_title_suffix else ""
            output_path = os.path.join(OUTPUT_ROLLING, f"rolling_5yr{suffix_filename}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Saved: {output_path}")
            
            # ==================================================================
            # 2️⃣ Percentile-Band Plot (IQR Only + Median)
            # ==================================================================
            p25 = aligned_df_masked.quantile(0.25, axis=1)
            p50 = aligned_df_masked.quantile(0.50, axis=1)
            p75 = aligned_df_masked.quantile(0.75, axis=1)
            
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(16, 9))
            fig.suptitle(f"Rolling 5-Year Multiplicative Growth (×) Percentile Bands (IQR Only)\nAligned by Months Since Datamining Cutoff{title_suffix_text}",
                         fontsize=16, color="white")
            
            # Shaded IQR band
            ax.fill_between(aligned_df_masked.index, p25, p75, color="cyan", alpha=0.35, label="25th–75th percentile (IQR)")
            
            # Median line
            ax.plot(p50.index, p50.values, color="lime", linewidth=2, label="Median (50th percentile)")
            
            # EW Composite line
            ax.plot(composite_masked.index, composite_masked.values, color="white", linewidth=2.6, label="Equal-Weight Composite")
            
            # Datamining cutoff marker (at month 0)
            ax.axvline(0, color="orange", linestyle="--", linewidth=1.5, label=f"Datamining Cutoff ({DATAMINING_CUTOFF_DATE.year})")
            if not composite_masked.dropna().empty:
                y_top = composite_masked.dropna().max()
                ax.text(0, y_top * 0.95, f"Datamining Cutoff ({DATAMINING_CUTOFF_DATE.year})", color="orange", ha="left", va="top")
            
            # Structural break marker
            ax.axvline(months_at_structural_break, color="gray", linestyle="--", linewidth=1.5, label="Structural Break (2004)")
            if not composite_masked.dropna().empty:
                y_top = composite_masked.dropna().max()
                ax.text(months_at_structural_break, y_top * 0.90, "Structural Break (2004)", color="white", ha="left", va="top")
            
            ax.set_yscale("log")
            ax.set_ylabel("Rolling 5-Year Growth (×)", color="white")
            ax.set_xlabel("Months Since Datamining Cutoff", color="white")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend()
            plt.tight_layout()
            suffix_filename = f"_{sanitize_filename(plot_title_suffix).replace(' ', '_')}" if plot_title_suffix else ""
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
        
        n, bins, patches = ax.hist(all_strategy_cagrs, bins=30, color='cyan', 
                                  alpha=0.7, edgecolor='white', linewidth=1)
        
        mean_cagr = np.mean(all_strategy_cagrs)
        median_cagr = np.median(all_strategy_cagrs)
        ax.axvline(mean_cagr, color='yellow', linestyle='--', linewidth=2, 
                  label=f"Mean: {mean_cagr:.2f}%")
        ax.axvline(median_cagr, color='lime', linestyle='--', linewidth=2, 
                  label=f"Median: {median_cagr:.2f}%")
        
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
        ax.set_title("CAGR Distribution", color="white")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        suffix_filename = f"_{sanitize_filename(plot_title_suffix).replace(' ', '_')}" if plot_title_suffix else ""
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
        
        # Debug: Print how many signals contributed to each period
        print("\nData collection summary for aggregate plots:")
        for period_name in ['Datamining Period', 'Post-Datamining', 'Post-Structural Break', 'Full Sample']:
            if period_name in all_strategy_final_moments_by_period:
                n_cagr = len(all_strategy_final_moments_by_period[period_name]['CAGR (%)'])
                n_var = len(all_strategy_final_moments_by_period[period_name]['Variance'])
                print(f"  {period_name}: {n_cagr} signals (CAGR), {n_var} signals (Variance)")
        print(f"  Total CAGR values collected: {len(all_strategy_cagrs)}")
        print("="*80)
    
    moments_to_plot = []
    moment_folder_map = {}
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
    
    moment_labels = {
        'CAGR (%)': 'CAGR (%)',
        'Variance': 'Variance',
        'Skewness': 'Skewness',
        'Kurtosis': 'Kurtosis'
    }
    
    # Determine which periods to show
    periods_to_show_aggregate = []
    if 'Full Sample' in all_strategy_final_moments_by_period:
        periods_to_show_aggregate.append('Full Sample')
    if 'Datamining Period' in all_strategy_final_moments_by_period:
        periods_to_show_aggregate = ['Datamining Period', 'Post-Datamining', 'Post-Structural Break', 'Full Sample']
    else:
        periods_to_show_aggregate = ['Full Sample']
    
    # Create separate PNG for each moment
    for moment in moments_to_plot:
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
        
        if num_periods == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        title_suffix_text = f" - {plot_title_suffix}" if plot_title_suffix else ""
        fig.suptitle(f"Distribution of {moment_labels[moment]} Across All Strategies{title_suffix_text}", 
                     fontsize=16, color="white", y=0.995, ha='center')
        
        for period_idx, period_name in enumerate(periods_to_show_aggregate):
            ax = axes[period_idx]
            
            if period_name in all_strategy_final_moments_by_period:
                moment_data = all_strategy_final_moments_by_period[period_name][moment]
            else:
                moment_data = []
            
            if len(moment_data) == 0:
                ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                       ha='center', va='center', color='white', fontsize=12)
                ax.set_title(period_name, color="white", fontsize=12, pad=10)
                continue
            
            n, bins, patches = ax.hist(moment_data, bins=30, color='cyan', 
                                      alpha=0.7, edgecolor='white', linewidth=1)
            
            mean_val = np.mean(moment_data)
            median_val = np.median(moment_data)
            ax.axvline(mean_val, color='yellow', linestyle='--', linewidth=2, 
                      label=f"Mean: {mean_val:.4f}")
            ax.axvline(median_val, color='lime', linestyle='--', linewidth=2, 
                      label=f"Median: {median_val:.4f}")
            
            std_val = np.std(moment_data)
            min_val = np.min(moment_data)
            max_val = np.max(moment_data)
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
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white'),
                   color='white', family='monospace')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if num_periods == 1:
            plt.subplots_adjust(top=0.90, wspace=0.3)
        else:
            plt.subplots_adjust(top=0.92, wspace=0.3)
        
        suffix_filename = f"_{sanitize_filename(plot_title_suffix).replace(' ', '_')}" if plot_title_suffix else ""
        folder_name = moment_folder_map[moment]
        output_path = os.path.join(OUTPUT_DISTRIBUTIONS_AGGREGATE_MOMENTS[folder_name], 
                                  f"{folder_name.lower()}_distribution{suffix_filename}.png")
        plt.savefig(output_path)
        print(f"  {moment_labels[moment]} distribution plot saved to: {output_path}")
        plt.close()
    else:
        print("\nMoment distribution plots are disabled (generate_moment_distributions = False).")
    
    total_time = time.time() - start_time
    print("\n" + "="*80)
    # Close progress bar if using tqdm
    if HAS_TQDM:
        pbar.close()
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nProcessed: {processed_count:,} signals")
    if skipped_count > 0:
        print(f"Skipped (already exist): {skipped_count:,} signals")
    if processed_count > 0:
        avg_time_per_signal = total_time / processed_count
        print(f"\nPerformance:")
        print(f"  Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
        print(f"  Average time per signal: {avg_time_per_signal:.2f} seconds")
        print(f"  Processing rate: {processed_count/total_time:.1f} signals/second")
    print(f"\nAll outputs saved to: {OUTPUT_BASE}")

