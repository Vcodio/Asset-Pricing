"""
Data-Mined Factors Backtest Analysis
Backtests all data-mined strategies and highlights expected false vs. true discoveries
based on FDR analysis results.

This script:
1. Loads FDR analysis results to identify significant factors
2. Classifies factors as expected false discoveries vs. true discoveries
3. Performs comprehensive backtests for all strategies
4. Creates visualizations highlighting the distinction between false and true discoveries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import time
from datetime import datetime

# Try to import numba for JIT compilation (optional - falls back to regular functions)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create a dummy decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================
# PROMPT FOR PLOT TITLE SUFFIX (needed before creating output folders)
# ================================================================
print("\n" + "="*80)
print("PLOT TITLE SUFFIX")
print("="*80)
print("Enter a suffix to add to all plot titles (e.g., 'Pre-2004 Data', 'Full Sample', etc.)")
print("This will also be used as the output folder name to organize all outputs.")
print("Type 'NONE' or press Enter to skip adding a suffix (will use 'Output' as folder name).")
suffix_input = input("Plot title suffix: ").strip()

# Process the input: if NONE, empty, or just whitespace, set to empty string
if suffix_input.upper() == "NONE" or suffix_input == "":
    plot_title_suffix = ""
    output_folder_name = "output"  # Use lowercase 'output' as default
    print("No suffix will be added to plot titles.")
    print(f"Outputs will be organized in folder: '{output_folder_name}'")
else:
    plot_title_suffix = suffix_input
    # Sanitize folder name: replace invalid filesystem characters
    output_folder_name = plot_title_suffix.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace('"', "_").replace("<", "_").replace(">", "_").replace("|", "_")
    print(f"Plot titles will include: '{plot_title_suffix}'")
    print(f"Outputs will be organized in folder: '{output_folder_name}'")
print("="*80 + "\n")

# Input files - try to find FDR results in multiple locations
OUTPUT_BASE = os.path.join(SCRIPT_DIR, output_folder_name)

# Possible locations for FDR results (in order of preference)
FDR_LOCATIONS = [
    # 1. New structure: suffix folder
    (os.path.join(OUTPUT_BASE, "fdr_analysis", "results", "fdr_analysis_results.csv"),
     os.path.join(OUTPUT_BASE, "fdr_analysis", "results", "fdr_summary.csv")),
    # 2. Old structure: output/results
    (os.path.join(SCRIPT_DIR, "output", "results", "fdr_analysis_results.csv"),
     os.path.join(SCRIPT_DIR, "output", "results", "fdr_summary.csv")),
    # 3. Default output folder (lowercase)
    (os.path.join(SCRIPT_DIR, "output", "fdr_analysis", "results", "fdr_analysis_results.csv"),
     os.path.join(SCRIPT_DIR, "output", "fdr_analysis", "results", "fdr_summary.csv")),
]

# Find the first location that exists
FDR_RESULTS_FILE = None
FDR_SUMMARY_FILE = None
for results_path, summary_path in FDR_LOCATIONS:
    if os.path.exists(results_path) and os.path.exists(summary_path):
        FDR_RESULTS_FILE = results_path
        FDR_SUMMARY_FILE = summary_path
        print(f"Using FDR results from: {results_path}")
        break

if FDR_RESULTS_FILE is None:
    print("\nWARNING: Could not find FDR analysis results files!")
    print("  Expected locations:")
    for results_path, _ in FDR_LOCATIONS:
        print(f"    - {results_path}")
    print("\n  Please run 'datamined_fdr_analysis.py' first, or use 'organize_existing_output.py'")
    print("  to organize existing results from the old 'output/' folder.")
    exit(1)

RETURNS_FILE = os.path.join(SCRIPT_DIR, "DataMinedLongShortReturnsEW.csv")
SIGNAL_LIST_FILE = os.path.join(SCRIPT_DIR, "DataMinedSignalList.csv")

# Output directories - organized structure with suffix-based folder
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "backtests")
OUTPUT_PLOTS = os.path.join(OUTPUT_DIR, "plots")
OUTPUT_RESULTS = os.path.join(OUTPUT_DIR, "results")
OUTPUT_COMPARISONS = os.path.join(OUTPUT_DIR, "comparisons")

for dir_path in [OUTPUT_DIR, OUTPUT_PLOTS, OUTPUT_RESULTS, OUTPUT_COMPARISONS]:
    os.makedirs(dir_path, exist_ok=True)

# Significance threshold
P_TARGET = 0.05

# Configuration toggles
INCLUDE_ALL_STRATEGIES = True  # If True, analyze all 29k strategies, not just significant ones
GENERATE_CUMULATIVE_BACKTESTS = True  # If True, generate cumulative return backtest plots
GENERATE_ROLLING_PLOTS = True  # If True, generate rolling 5-year percentile bands plots

# Important dates for analysis
DATAMINING_CUTOFF_DATE = pd.Timestamp('1980-01-01')  # When datamining period ended
STRUCTURAL_BREAK_DATE = pd.Timestamp('2004-01-01')  # Structural break date

# ============================================================================
# HELPER FUNCTIONS (OPTIMIZED WITH NUMBA)
# ============================================================================
@jit(nopython=True, cache=True, nogil=True)
def _calculate_cagr_numba(returns_array):
    """Numba-optimized CAGR calculation."""
    n = len(returns_array)
    if n == 0:
        return np.nan
    if n < 12:
        return np.nan
    
    # Calculate cumulative product
    cumprod = 1.0
    for i in range(n):
        cumprod *= (1.0 + returns_array[i] / 100.0)
    
    if cumprod <= 0:
        return np.nan
    
    num_years = n / 12.0
    return ((cumprod ** (1.0 / num_years)) - 1.0) * 100.0

@jit(nopython=True, cache=True, nogil=True)
def _calculate_sharpe_numba(returns_array):
    """Numba-optimized Sharpe ratio calculation."""
    n = len(returns_array)
    if n < 2:
        return np.nan
    
    # Calculate mean
    mean_val = 0.0
    for i in range(n):
        mean_val += returns_array[i]
    mean_val = mean_val / n / 100.0
    
    # Calculate std
    variance = 0.0
    for i in range(n):
        diff = (returns_array[i] / 100.0) - mean_val
        variance += diff * diff
    variance = variance / (n - 1)
    std_val = np.sqrt(variance)
    
    if std_val == 0:
        return np.nan
    
    return (mean_val / std_val) * np.sqrt(12.0)

@jit(nopython=True, cache=True, nogil=True)
def _calculate_max_dd_numba(returns_array):
    """Numba-optimized max drawdown calculation."""
    n = len(returns_array)
    if n == 0:
        return np.nan
    
    # Calculate cumulative returns
    cumprod = np.empty(n)
    cumprod[0] = 1.0 + returns_array[0] / 100.0
    for i in range(1, n):
        cumprod[i] = cumprod[i-1] * (1.0 + returns_array[i] / 100.0)
    
    # Calculate running max and drawdown
    running_max = cumprod[0]
    min_dd = 0.0
    for i in range(n):
        if cumprod[i] > running_max:
            running_max = cumprod[i]
        dd = (cumprod[i] - running_max) / running_max
        if dd < min_dd:
            min_dd = dd
    
    return min_dd * 100.0

def calculate_cagr(returns):
    """Calculates the Compound Annual Growth Rate (CAGR) from monthly returns."""
    if len(returns) == 0:
        return np.nan
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return np.nan
    returns_array = returns_clean.values
    return _calculate_cagr_numba(returns_array)

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculates the annualized Sharpe ratio from monthly returns."""
    if len(returns) == 0:
        return np.nan
    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        return np.nan
    returns_array = returns_clean.values
    return _calculate_sharpe_numba(returns_array)

def calculate_max_drawdown(returns):
    """Calculates the maximum drawdown from monthly returns."""
    if len(returns) == 0:
        return np.nan
    returns_clean = returns.dropna()
    if len(returns_clean) == 0:
        return np.nan
    returns_array = returns_clean.values
    return _calculate_max_dd_numba(returns_array)

# ============================================================================
# LOAD SIGNAL NAMES
# ============================================================================
def generate_signal_name(signal_form, v1, v2):
    """Generate a readable signal name from formula and variables."""
    # Replace v1 and v2 with actual variable names
    name = signal_form.replace('v1', v1).replace('v2', v2)
    return name

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

def get_signal_name(signal_id):
    """Get signal name for a given signal ID, fallback to ID if not found."""
    if signal_id in signal_names_dict:
        return signal_names_dict[signal_id]
    return f"Signal {int(signal_id)}"

# ============================================================================
# LOAD FDR RESULTS
# ============================================================================
print("="*80)
print("Data-Mined Factors Backtest Analysis")
print("="*80)

# Load signal names first
print("\nLoading signal names...")
signal_names_dict = load_signal_names()
if len(signal_names_dict) > 0:
    print(f"  Loaded names for {len(signal_names_dict):,} signals")
else:
    print("  Using signal IDs as names")

print("\nLoading FDR analysis results...")

try:
    fdr_results = pd.read_csv(FDR_RESULTS_FILE)
    fdr_summary = pd.read_csv(FDR_SUMMARY_FILE)
    print(f"  Loaded FDR results for {len(fdr_results):,} factors")
    
    # Extract FDR metrics
    fdr_value = fdr_summary['fdr'].iloc[0]
    n_significant = int(fdr_summary['n_significant'].iloc[0])
    pi0 = fdr_summary['pi0'].iloc[0]
    
    print(f"  Significant factors (p ≤ {P_TARGET}): {n_significant:,}")
    print(f"  Estimated FDR: {fdr_value:.2%}")
    print(f"  Expected false discoveries: {int(n_significant * fdr_value):,}")
    print(f"  Expected true discoveries: {int(n_significant * (1 - fdr_value)):,}")
    
except FileNotFoundError:
    print(f"\nERROR: Could not find FDR results file: {FDR_RESULTS_FILE}")
    print("Please run datamined_fdr_analysis.py first to generate FDR results.")
    exit(1)

# ============================================================================
# CLASSIFY FACTORS
# ============================================================================
print("\nClassifying factors...")

# Identify significant factors
significant_mask = fdr_results['p_value'] <= P_TARGET
significant_factors = fdr_results[significant_mask].copy()

# Calculate expected number of false discoveries
expected_false = int(len(significant_factors) * fdr_value)
expected_true = len(significant_factors) - expected_false

# Classify factors: sort by p-value and assign first N as expected false
# (Lower p-values are more likely to be true, higher p-values among significant are more likely false)
significant_factors_sorted = significant_factors.sort_values('p_value', ascending=False)
significant_factors_sorted['expected_false'] = False
significant_factors_sorted.iloc[:expected_false, significant_factors_sorted.columns.get_loc('expected_false')] = True

# Merge back classification
fdr_results['expected_false'] = False
fdr_results.loc[significant_factors_sorted.index, 'expected_false'] = significant_factors_sorted['expected_false']
fdr_results['discovery_type'] = 'Not Significant'
fdr_results.loc[significant_factors_sorted.index, 'discovery_type'] = 'Expected True Discovery'
fdr_results.loc[significant_factors_sorted[significant_factors_sorted['expected_false']].index, 'discovery_type'] = 'Expected False Discovery'

print(f"  Classified {len(significant_factors):,} significant factors:")
print(f"    - Expected False Discoveries: {expected_false:,}")
print(f"    - Expected True Discoveries: {expected_true:,}")

# Determine which factors to analyze
if INCLUDE_ALL_STRATEGIES:
    print(f"\n  Including ALL {len(fdr_results):,} strategies in backtest analysis (not just significant ones)")
    factors_to_analyze = fdr_results.copy()
else:
    print(f"\n  Analyzing only {len(significant_factors):,} significant factors (p ≤ {P_TARGET})")
    factors_to_analyze = significant_factors_sorted.copy()

# ============================================================================
# LOAD RETURNS DATA
# ============================================================================
print("\nLoading returns data...")
try:
    returns_df = pd.read_csv(RETURNS_FILE)
    returns_df['date'] = pd.to_datetime(returns_df[['year', 'month']].assign(day=1))
    print(f"  Loaded {len(returns_df):,} return observations")
    print(f"  Date range: {returns_df['date'].min().strftime('%Y-%m')} to {returns_df['date'].max().strftime('%Y-%m')}")
except FileNotFoundError:
    print(f"\nERROR: Could not find returns file: {RETURNS_FILE}")
    exit(1)

# ============================================================================
# PERFORM BACKTESTS (OPTIMIZED WITH VECTORIZATION & MULTIPROCESSING)
# ============================================================================
print("\n" + "="*80)
print("Performing Backtests")
print("="*80)

# Get factors to analyze (either all or just significant)
signal_ids_to_analyze = factors_to_analyze['signalid'].values
n_factors = len(signal_ids_to_analyze)

if INCLUDE_ALL_STRATEGIES:
    print(f"\nBacktesting {n_factors:,} factors (all strategies)...")
else:
    print(f"\nBacktesting {n_factors:,} significant factors...")
print("Using optimized vectorized processing with multiprocessing...")

start_time = time.time()

# OPTIMIZATION: Pre-filter returns dataframe to only signals we'll analyze
returns_filtered = returns_df[returns_df['signalid'].isin(signal_ids_to_analyze)].copy()

# OPTIMIZATION: Pre-create factor info lookup dictionary
factor_info_dict = {}
for _, row in fdr_results.iterrows():
    factor_info_dict[row['signalid']] = {
        'expected_false': row['expected_false'],
        'discovery_type': row['discovery_type'],
        'p_value': row['p_value'],
        't_stat': row.get('t_stat', np.nan)  # Get t-statistic if available
    }

# OPTIMIZATION: Use groupby for vectorized processing
def process_signal_group(group):
    """Process a group of returns for a single signal - optimized version."""
    signal_id = group.name
    signal_returns = group['ret']
    
    if len(signal_returns) == 0:
        return None
    
    # Get classification
    if signal_id not in factor_info_dict:
        return None
    factor_info = factor_info_dict[signal_id]
    
    # Calculate statistics using optimized functions
    returns_clean = signal_returns.dropna()
    if len(returns_clean) < 12:  # Need at least 12 months
        return None
    
    # Use numpy arrays for faster computation
    returns_array = returns_clean.values
    
    # Flip negative t-statistics (take the other side of the trade)
    # If t-stat < 0, the strategy predicts negative returns, so we flip to get positive returns
    t_stat = factor_info.get('t_stat', np.nan)
    flipped = False
    if not np.isnan(t_stat) and t_stat < 0:
        returns_array = -returns_array  # Flip the sign (take the other side)
        flipped = True
    
    cagr = _calculate_cagr_numba(returns_array)
    sharpe = _calculate_sharpe_numba(returns_array)
    max_dd = _calculate_max_dd_numba(returns_array)
    mean_ret = np.mean(returns_array)
    std_ret = np.std(returns_array, ddof=1)
    
    # Get date range
    dates = group['date']
    start_date = dates.min()
    end_date = dates.max()
    
    return {
        'signalid': signal_id,
        'signal_name': get_signal_name(signal_id),
        'discovery_type': factor_info['discovery_type'],
        'expected_false': factor_info['expected_false'],
        'p_value': factor_info['p_value'],
        't_stat_original': t_stat,
        'flipped': flipped,
        'mean_return': mean_ret,
        'std_return': std_ret,
        'cagr': cagr,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'n_obs': len(returns_clean),
        'start_date': start_date,
        'end_date': end_date
    }

# OPTIMIZATION: Use groupby.apply for vectorized processing
print("  Processing signals in batches...")
results_series = returns_filtered.groupby('signalid', sort=False).apply(
    process_signal_group, 
    include_groups=False
)

# Convert to list and filter out None results
if isinstance(results_series, pd.Series):
    results_list = results_series.tolist()
else:
    results_list = [results_series] if results_series is not None else []

backtest_results = [r for r in results_list if r is not None]

backtest_df = pd.DataFrame(backtest_results)
elapsed_time = time.time() - start_time
print(f"\nCompleted backtests in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
print(f"  Processing rate: {len(backtest_df)/elapsed_time:.0f} factors/second")

# Report on flipped strategies (negative t-stats)
if 'flipped' in backtest_df.columns:
    n_flipped = backtest_df['flipped'].sum()
    print(f"  Strategies flipped (negative t-stat → positive returns): {n_flipped:,} ({n_flipped/len(backtest_df)*100:.1f}%)")

# Save backtest results
backtest_file = os.path.join(OUTPUT_RESULTS, "backtest_results.csv")
backtest_df.to_csv(backtest_file, index=False)
print(f"\nBacktest results saved to: {backtest_file}")
print(f"  Full path: {os.path.abspath(backtest_file)}")
print(f"  Location: Datamining/output/backtests/results/")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("Summary Statistics")
print("="*80)

# Filter by discovery_type to ensure correct separation
false_discoveries = backtest_df[backtest_df['discovery_type'] == 'Expected False Discovery']
true_discoveries = backtest_df[backtest_df['discovery_type'] == 'Expected True Discovery']
not_significant = backtest_df[backtest_df['discovery_type'] == 'Not Significant']

print(f"\nExpected False Discoveries ({len(false_discoveries):,} factors):")
if len(false_discoveries) > 0:
    print(f"  Mean CAGR: {false_discoveries['cagr'].mean():.2f}%")
    print(f"  Median CAGR: {false_discoveries['cagr'].median():.2f}%")
    print(f"  Mean Sharpe: {false_discoveries['sharpe_ratio'].mean():.3f}")
    print(f"  Mean Max DD: {false_discoveries['max_drawdown'].mean():.2f}%")

print(f"\nExpected True Discoveries ({len(true_discoveries):,} factors):")
if len(true_discoveries) > 0:
    print(f"  Mean CAGR: {true_discoveries['cagr'].mean():.2f}%")
    print(f"  Median CAGR: {true_discoveries['cagr'].median():.2f}%")
    print(f"  Mean Sharpe: {true_discoveries['sharpe_ratio'].mean():.3f}")
    print(f"  Mean Max DD: {true_discoveries['max_drawdown'].mean():.2f}%")

if INCLUDE_ALL_STRATEGIES:
    print(f"\nNot Significant ({len(not_significant):,} factors):")
    if len(not_significant) > 0:
        print(f"  Mean CAGR: {not_significant['cagr'].mean():.2f}%")
        print(f"  Median CAGR: {not_significant['cagr'].median():.2f}%")
        print(f"  Mean Sharpe: {not_significant['sharpe_ratio'].mean():.3f}")
        print(f"  Mean Max DD: {not_significant['max_drawdown'].mean():.2f}%")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("Creating Visualizations")
print("="*80)

plt.style.use("dark_background")

# Helper variable for plot title suffix
title_suffix_text = f" - {plot_title_suffix}" if plot_title_suffix else ""

# 1. CAGR Distribution Comparison (3 groups)
print("  Creating CAGR distribution comparison...")
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
fig.suptitle(f"CAGR Distribution: All Discovery Types{title_suffix_text}", fontsize=16, color="white", y=1.02)

# False discoveries
ax = axes[0]
if len(false_discoveries) > 0:
    cagr_false = false_discoveries['cagr'].dropna()
    if len(cagr_false) > 0:
        n, bins, patches = ax.hist(cagr_false, bins=50, color='red', alpha=0.7, edgecolor='white')
        mean_cagr = cagr_false.mean()
        median_cagr = cagr_false.median()
        ax.axvline(mean_cagr, color='yellow', linestyle='--', linewidth=2, label=f'Mean: {mean_cagr:.2f}%')
        ax.axvline(median_cagr, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_cagr:.2f}%')
        ax.text(0.02, 0.98, f'N: {len(cagr_false):,}\nMean: {mean_cagr:.2f}%\nMedian: {median_cagr:.2f}%',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
ax.set_title('Expected False Discoveries', color='white', fontsize=13)
ax.set_xlabel('CAGR (%)', color='white', fontsize=12)
ax.set_ylabel('Frequency', color='white', fontsize=12)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.3)

# True discoveries
ax = axes[1]
if len(true_discoveries) > 0:
    cagr_true = true_discoveries['cagr'].dropna()
    if len(cagr_true) > 0:
        n, bins, patches = ax.hist(cagr_true, bins=50, color='green', alpha=0.7, edgecolor='white')
        mean_cagr = cagr_true.mean()
        median_cagr = cagr_true.median()
        ax.axvline(mean_cagr, color='yellow', linestyle='--', linewidth=2, label=f'Mean: {mean_cagr:.2f}%')
        ax.axvline(median_cagr, color='lime', linestyle='--', linewidth=2, label=f'Median: {median_cagr:.2f}%')
        ax.text(0.02, 0.98, f'N: {len(cagr_true):,}\nMean: {mean_cagr:.2f}%\nMedian: {median_cagr:.2f}%',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
ax.set_title('Expected True Discoveries', color='white', fontsize=13)
ax.set_xlabel('CAGR (%)', color='white', fontsize=12)
ax.set_ylabel('Frequency', color='white', fontsize=12)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.3)

# Not Significant
ax = axes[2]
if INCLUDE_ALL_STRATEGIES and len(not_significant) > 0:
    cagr_not_sig = not_significant['cagr'].dropna()
    if len(cagr_not_sig) > 0:
        n, bins, patches = ax.hist(cagr_not_sig, bins=50, color='gray', alpha=0.7, edgecolor='white')
        mean_cagr = cagr_not_sig.mean()
        median_cagr = cagr_not_sig.median()
        ax.axvline(mean_cagr, color='yellow', linestyle='--', linewidth=2, label=f'Mean: {mean_cagr:.2f}%')
        ax.axvline(median_cagr, color='lightgray', linestyle='--', linewidth=2, label=f'Median: {median_cagr:.2f}%')
        ax.text(0.02, 0.98, f'N: {len(cagr_not_sig):,}\nMean: {mean_cagr:.2f}%\nMedian: {median_cagr:.2f}%',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
ax.set_title('Not Significant', color='white', fontsize=13)
ax.set_xlabel('CAGR (%)', color='white', fontsize=12)
ax.set_ylabel('Frequency', color='white', fontsize=12)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plot_file = os.path.join(OUTPUT_COMPARISONS, "cagr_distribution_comparison.png")
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {plot_file}")

# 2. Sharpe Ratio Comparison (3 groups)
print("  Creating Sharpe ratio comparison...")
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
fig.suptitle(f"Sharpe Ratio Distribution: All Discovery Types{title_suffix_text}", fontsize=16, color="white", y=1.02)

# False discoveries
ax = axes[0]
if len(false_discoveries) > 0:
    sharpe_false = false_discoveries['sharpe_ratio'].dropna()
    sharpe_false = sharpe_false[(sharpe_false >= -5) & (sharpe_false <= 5)]  # Remove outliers
    if len(sharpe_false) > 0:
        n, bins, patches = ax.hist(sharpe_false, bins=50, color='red', alpha=0.7, edgecolor='white')
        mean_sharpe = sharpe_false.mean()
        median_sharpe = sharpe_false.median()
        ax.axvline(mean_sharpe, color='yellow', linestyle='--', linewidth=2, label=f'Mean: {mean_sharpe:.3f}')
        ax.axvline(median_sharpe, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_sharpe:.3f}')
        ax.text(0.02, 0.98, f'N: {len(sharpe_false):,}\nMean: {mean_sharpe:.3f}\nMedian: {median_sharpe:.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
ax.set_title('Expected False Discoveries', color='white', fontsize=13)
ax.set_xlabel('Sharpe Ratio', color='white', fontsize=12)
ax.set_ylabel('Frequency', color='white', fontsize=12)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.3)

# True discoveries
ax = axes[1]
if len(true_discoveries) > 0:
    sharpe_true = true_discoveries['sharpe_ratio'].dropna()
    sharpe_true = sharpe_true[(sharpe_true >= -5) & (sharpe_true <= 5)]  # Remove outliers
    if len(sharpe_true) > 0:
        n, bins, patches = ax.hist(sharpe_true, bins=50, color='green', alpha=0.7, edgecolor='white')
        mean_sharpe = sharpe_true.mean()
        median_sharpe = sharpe_true.median()
        ax.axvline(mean_sharpe, color='yellow', linestyle='--', linewidth=2, label=f'Mean: {mean_sharpe:.3f}')
        ax.axvline(median_sharpe, color='lime', linestyle='--', linewidth=2, label=f'Median: {median_sharpe:.3f}')
        ax.text(0.02, 0.98, f'N: {len(sharpe_true):,}\nMean: {mean_sharpe:.3f}\nMedian: {median_sharpe:.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
ax.set_title('Expected True Discoveries', color='white', fontsize=13)
ax.set_xlabel('Sharpe Ratio', color='white', fontsize=12)
ax.set_ylabel('Frequency', color='white', fontsize=12)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.3)

# Not Significant
ax = axes[2]
if INCLUDE_ALL_STRATEGIES and len(not_significant) > 0:
    sharpe_not_sig = not_significant['sharpe_ratio'].dropna()
    sharpe_not_sig = sharpe_not_sig[(sharpe_not_sig >= -5) & (sharpe_not_sig <= 5)]  # Remove outliers
    if len(sharpe_not_sig) > 0:
        n, bins, patches = ax.hist(sharpe_not_sig, bins=50, color='gray', alpha=0.7, edgecolor='white')
        mean_sharpe = sharpe_not_sig.mean()
        median_sharpe = sharpe_not_sig.median()
        ax.axvline(mean_sharpe, color='yellow', linestyle='--', linewidth=2, label=f'Mean: {mean_sharpe:.3f}')
        ax.axvline(median_sharpe, color='lightgray', linestyle='--', linewidth=2, label=f'Median: {median_sharpe:.3f}')
        ax.text(0.02, 0.98, f'N: {len(sharpe_not_sig):,}\nMean: {mean_sharpe:.3f}\nMedian: {median_sharpe:.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
ax.set_title('Not Significant', color='white', fontsize=13)
ax.set_xlabel('Sharpe Ratio', color='white', fontsize=12)
ax.set_ylabel('Frequency', color='white', fontsize=12)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plot_file = os.path.join(OUTPUT_COMPARISONS, "sharpe_distribution_comparison.png")
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {plot_file}")

# 3. Scatter Plot: P-value vs CAGR
print("  Creating p-value vs CAGR scatter plot...")
fig, ax = plt.subplots(figsize=(14, 8))
fig.suptitle(f"P-value vs. CAGR: All Discovery Types{title_suffix_text}", fontsize=16, color="white")

# Plot false discoveries
if len(false_discoveries) > 0:
    ax.scatter(false_discoveries['p_value'], false_discoveries['cagr'], 
               alpha=0.5, s=20, color='red', label=f'Expected False ({len(false_discoveries):,})', zorder=2)

# Plot true discoveries
if len(true_discoveries) > 0:
    ax.scatter(true_discoveries['p_value'], true_discoveries['cagr'], 
               alpha=0.5, s=20, color='green', label=f'Expected True ({len(true_discoveries):,})', zorder=2)

# Plot not significant
if INCLUDE_ALL_STRATEGIES and len(not_significant) > 0:
    ax.scatter(not_significant['p_value'], not_significant['cagr'], 
               alpha=0.3, s=15, color='gray', label=f'Not Significant ({len(not_significant):,})', zorder=1)

ax.axvline(P_TARGET, color='yellow', linestyle='--', linewidth=2, label=f'Significance threshold (p={P_TARGET})', zorder=1)
ax.axhline(0, color='white', linestyle=':', linewidth=1, alpha=0.5, zorder=1)

ax.set_xlabel('P-value', color='white', fontsize=12)
ax.set_ylabel('CAGR (%)', color='white', fontsize=12)
ax.set_xscale('log')
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_facecolor('black')

plt.tight_layout()
plot_file = os.path.join(OUTPUT_COMPARISONS, "pvalue_vs_cagr_scatter.png")
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {plot_file}")

# 4. Top Performers Comparison (3 groups)
print("  Creating top performers comparison...")
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle(f"Top 50 Performers by CAGR: All Discovery Types{title_suffix_text}", fontsize=16, color="white", y=1.02)

# Top false discoveries
ax = axes[0]
if len(false_discoveries) > 0:
    top_false = false_discoveries.nlargest(50, 'cagr')
    y_pos = np.arange(len(top_false))
    bars = ax.barh(y_pos, top_false['cagr'], color='red', alpha=0.7, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([get_signal_name(sid) for sid in top_false['signalid']], fontsize=8)
    ax.set_xlabel('CAGR (%)', color='white', fontsize=12)
    ax.set_title(f'Top 50 Expected False Discoveries\n(Mean: {top_false["cagr"].mean():.2f}%)', 
                 color='white', fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.axvline(0, color='white', linewidth=1)

# Top true discoveries
ax = axes[1]
if len(true_discoveries) > 0:
    top_true = true_discoveries.nlargest(50, 'cagr')
    y_pos = np.arange(len(top_true))
    bars = ax.barh(y_pos, top_true['cagr'], color='green', alpha=0.7, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([get_signal_name(sid) for sid in top_true['signalid']], fontsize=8)
    ax.set_xlabel('CAGR (%)', color='white', fontsize=12)
    ax.set_title(f'Top 50 Expected True Discoveries\n(Mean: {top_true["cagr"].mean():.2f}%)', 
                 color='white', fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.axvline(0, color='white', linewidth=1)

# Top not significant
ax = axes[2]
if INCLUDE_ALL_STRATEGIES and len(not_significant) > 0:
    top_not_sig = not_significant.nlargest(50, 'cagr')
    y_pos = np.arange(len(top_not_sig))
    bars = ax.barh(y_pos, top_not_sig['cagr'], color='gray', alpha=0.7, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([get_signal_name(sid) for sid in top_not_sig['signalid']], fontsize=8)
    ax.set_xlabel('CAGR (%)', color='white', fontsize=12)
    ax.set_title(f'Top 50 Not Significant\n(Mean: {top_not_sig["cagr"].mean():.2f}%)', 
                 color='white', fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.axvline(0, color='white', linewidth=1)

plt.tight_layout()
plot_file = os.path.join(OUTPUT_COMPARISONS, "top_performers_comparison.png")
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {plot_file}")

# 5. Performance Metrics Summary (3 groups)
print("  Creating performance metrics summary...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f"Performance Metrics Comparison: All Discovery Types{title_suffix_text}", fontsize=16, color="white", y=0.995)

metrics = [
    ('cagr', 'CAGR (%)', 0, 0),
    ('sharpe_ratio', 'Sharpe Ratio', 0, 1),
    ('max_drawdown', 'Max Drawdown (%)', 1, 0),
    ('mean_return', 'Mean Return (%)', 1, 1)
]

for metric, label, row, col in metrics:
    ax = axes[row, col]
    
    false_vals = false_discoveries[metric].dropna()
    true_vals = true_discoveries[metric].dropna()
    not_sig_vals = not_significant[metric].dropna() if INCLUDE_ALL_STRATEGIES and len(not_significant) > 0 else pd.Series(dtype=float)
    
    if metric == 'max_drawdown':
        false_vals = false_vals[false_vals >= -100]  # Remove extreme outliers
        true_vals = true_vals[true_vals >= -100]
        if len(not_sig_vals) > 0:
            not_sig_vals = not_sig_vals[not_sig_vals >= -100]
    
    # Prepare data for boxplot
    data_to_plot = []
    labels = []
    colors = []
    
    if len(false_vals) > 0:
        data_to_plot.append(false_vals)
        labels.append('False')
        colors.append('red')
    
    if len(true_vals) > 0:
        data_to_plot.append(true_vals)
        labels.append('True')
        colors.append('green')
    
    if len(not_sig_vals) > 0:
        data_to_plot.append(not_sig_vals)
        labels.append('Not Sig')
        colors.append('gray')
    
    if len(data_to_plot) > 0:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color the boxes
        for i, (box, color) in enumerate(zip(bp['boxes'], colors)):
            box.set_facecolor(color)
            box.set_alpha(0.7)
        
        # Color the medians
        for median in bp['medians']:
            median.set_color('white')
            median.set_linewidth(2)
        
        ax.set_ylabel(label, color='white', fontsize=11)
        ax.set_title(label, color='white', fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.set_facecolor('black')
        
        # Add statistics
        stats_parts = []
        if len(false_vals) > 0:
            stats_parts.append(f'False: N={len(false_vals):,}\nMean={false_vals.mean():.2f}')
        if len(true_vals) > 0:
            stats_parts.append(f'True: N={len(true_vals):,}\nMean={true_vals.mean():.2f}')
        if len(not_sig_vals) > 0:
            stats_parts.append(f'Not Sig: N={len(not_sig_vals):,}\nMean={not_sig_vals.mean():.2f}')
        stats_text = '\n\n'.join(stats_parts)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                color='white')

plt.tight_layout()
plot_file = os.path.join(OUTPUT_COMPARISONS, "performance_metrics_comparison.png")
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {plot_file}")

# ============================================================================
# STANDALONE COMPARISON PLOTS: Not Significant vs Other Groups
# ============================================================================
if INCLUDE_ALL_STRATEGIES and len(not_significant) > 0:
    print("\n  Creating standalone comparison plots (Not Significant vs Others)...")
    
    # 6. CAGR Distribution: Not Significant vs Significant
    print("    Creating CAGR comparison: Not Significant vs Significant...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"CAGR Distribution: Not Significant vs Significant Discoveries{title_suffix_text}", fontsize=16, color="white", y=1.02)
    
    # Not Significant
    ax = axes[0]
    cagr_not_sig = not_significant['cagr'].dropna()
    if len(cagr_not_sig) > 0:
        n, bins, patches = ax.hist(cagr_not_sig, bins=50, color='gray', alpha=0.7, edgecolor='white')
        mean_cagr = cagr_not_sig.mean()
        median_cagr = cagr_not_sig.median()
        ax.axvline(mean_cagr, color='yellow', linestyle='--', linewidth=2, label=f'Mean: {mean_cagr:.2f}%')
        ax.axvline(median_cagr, color='lightgray', linestyle='--', linewidth=2, label=f'Median: {median_cagr:.2f}%')
        ax.text(0.02, 0.98, f'N: {len(cagr_not_sig):,}\nMean: {mean_cagr:.2f}%\nMedian: {median_cagr:.2f}%',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
    ax.set_title('Not Significant', color='white', fontsize=13)
    ax.set_xlabel('CAGR (%)', color='white', fontsize=12)
    ax.set_ylabel('Frequency', color='white', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Significant (False + True combined)
    ax = axes[1]
    significant_combined = pd.concat([false_discoveries, true_discoveries])
    cagr_sig = significant_combined['cagr'].dropna()
    if len(cagr_sig) > 0:
        n, bins, patches = ax.hist(cagr_sig, bins=50, color='cyan', alpha=0.7, edgecolor='white')
        mean_cagr = cagr_sig.mean()
        median_cagr = cagr_sig.median()
        ax.axvline(mean_cagr, color='yellow', linestyle='--', linewidth=2, label=f'Mean: {mean_cagr:.2f}%')
        ax.axvline(median_cagr, color='lime', linestyle='--', linewidth=2, label=f'Median: {median_cagr:.2f}%')
        ax.text(0.02, 0.98, f'N: {len(cagr_sig):,}\nMean: {mean_cagr:.2f}%\nMedian: {median_cagr:.2f}%',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
    ax.set_title('Significant (False + True)', color='white', fontsize=13)
    ax.set_xlabel('CAGR (%)', color='white', fontsize=12)
    ax.set_ylabel('Frequency', color='white', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_COMPARISONS, "cagr_not_significant_vs_significant.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {plot_file}")
    
    # 7. Sharpe Ratio: Not Significant vs Significant
    print("    Creating Sharpe comparison: Not Significant vs Significant...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Sharpe Ratio Distribution: Not Significant vs Significant Discoveries{title_suffix_text}", fontsize=16, color="white", y=1.02)
    
    # Not Significant
    ax = axes[0]
    sharpe_not_sig = not_significant['sharpe_ratio'].dropna()
    sharpe_not_sig = sharpe_not_sig[(sharpe_not_sig >= -5) & (sharpe_not_sig <= 5)]
    if len(sharpe_not_sig) > 0:
        n, bins, patches = ax.hist(sharpe_not_sig, bins=50, color='gray', alpha=0.7, edgecolor='white')
        mean_sharpe = sharpe_not_sig.mean()
        median_sharpe = sharpe_not_sig.median()
        ax.axvline(mean_sharpe, color='yellow', linestyle='--', linewidth=2, label=f'Mean: {mean_sharpe:.3f}')
        ax.axvline(median_sharpe, color='lightgray', linestyle='--', linewidth=2, label=f'Median: {median_sharpe:.3f}')
        ax.text(0.02, 0.98, f'N: {len(sharpe_not_sig):,}\nMean: {mean_sharpe:.3f}\nMedian: {median_sharpe:.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
    ax.set_title('Not Significant', color='white', fontsize=13)
    ax.set_xlabel('Sharpe Ratio', color='white', fontsize=12)
    ax.set_ylabel('Frequency', color='white', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Significant (False + True combined)
    ax = axes[1]
    sharpe_sig = significant_combined['sharpe_ratio'].dropna()
    sharpe_sig = sharpe_sig[(sharpe_sig >= -5) & (sharpe_sig <= 5)]
    if len(sharpe_sig) > 0:
        n, bins, patches = ax.hist(sharpe_sig, bins=50, color='cyan', alpha=0.7, edgecolor='white')
        mean_sharpe = sharpe_sig.mean()
        median_sharpe = sharpe_sig.median()
        ax.axvline(mean_sharpe, color='yellow', linestyle='--', linewidth=2, label=f'Mean: {mean_sharpe:.3f}')
        ax.axvline(median_sharpe, color='lime', linestyle='--', linewidth=2, label=f'Median: {median_sharpe:.3f}')
        ax.text(0.02, 0.98, f'N: {len(sharpe_sig):,}\nMean: {mean_sharpe:.3f}\nMedian: {median_sharpe:.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8), color='white')
    ax.set_title('Significant (False + True)', color='white', fontsize=13)
    ax.set_xlabel('Sharpe Ratio', color='white', fontsize=12)
    ax.set_ylabel('Frequency', color='white', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_COMPARISONS, "sharpe_not_significant_vs_significant.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {plot_file}")
    
    # 8. Box Plot Comparison: Not Significant vs False vs True
    print("    Creating box plot comparison: Not Significant vs False vs True...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Performance Metrics: Not Significant vs False vs True Discoveries{title_suffix_text}", fontsize=16, color="white", y=0.995)
    
    for metric, label, row, col in metrics:
        ax = axes[row, col]
        
        false_vals = false_discoveries[metric].dropna()
        true_vals = true_discoveries[metric].dropna()
        not_sig_vals = not_significant[metric].dropna()
        
        if metric == 'max_drawdown':
            false_vals = false_vals[false_vals >= -100]
            true_vals = true_vals[true_vals >= -100]
            not_sig_vals = not_sig_vals[not_sig_vals >= -100]
        
        data_to_plot = []
        labels = []
        colors = []
        
        if len(not_sig_vals) > 0:
            data_to_plot.append(not_sig_vals)
            labels.append('Not Sig')
            colors.append('gray')
        
        if len(false_vals) > 0:
            data_to_plot.append(false_vals)
            labels.append('False')
            colors.append('red')
        
        if len(true_vals) > 0:
            data_to_plot.append(true_vals)
            labels.append('True')
            colors.append('green')
        
        if len(data_to_plot) > 0:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            for i, (box, color) in enumerate(zip(bp['boxes'], colors)):
                box.set_facecolor(color)
                box.set_alpha(0.7)
            
            for median in bp['medians']:
                median.set_color('white')
                median.set_linewidth(2)
            
            ax.set_ylabel(label, color='white', fontsize=11)
            ax.set_title(label, color='white', fontsize=12)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax.set_facecolor('black')
            
            stats_parts = []
            if len(not_sig_vals) > 0:
                stats_parts.append(f'Not Sig: N={len(not_sig_vals):,}\nMean={not_sig_vals.mean():.2f}')
            if len(false_vals) > 0:
                stats_parts.append(f'False: N={len(false_vals):,}\nMean={false_vals.mean():.2f}')
            if len(true_vals) > 0:
                stats_parts.append(f'True: N={len(true_vals):,}\nMean={true_vals.mean():.2f}')
            stats_text = '\n\n'.join(stats_parts)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                    color='white')
    
    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_COMPARISONS, "performance_not_significant_comparison.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Saved: {plot_file}")

print(f"\nAll visualizations saved to: {OUTPUT_COMPARISONS}")

# ============================================================================
# CUMULATIVE RETURN BACKTESTS AND ROLLING PLOTS
# ============================================================================
print("\n" + "="*80)
print("Creating Cumulative Return Backtests and Rolling Plots")
print("="*80)

# Reference date for alignment (2004-01-01) - now using STRUCTURAL_BREAK_DATE
REFERENCE_DATE = STRUCTURAL_BREAK_DATE

# Storage for aligned series (by months since 2004) - separated by discovery type
aligned_series_false = {}
aligned_series_true = {}
aligned_series_not_sig = {}

# Storage for cumulative returns by strategy
cumulative_returns_by_strategy = {}

print("\nCalculating cumulative returns and rolling 5-year growth...")
print("  Using optimized vectorized calculations...")
start_calc_time = time.time()

# Verify backtest_df has discovery_type column
if 'discovery_type' not in backtest_df.columns:
    print("  ERROR: 'discovery_type' column missing from backtest_df!")
    print(f"  Available columns: {backtest_df.columns.tolist()}")
    # Try to merge it from fdr_results
    backtest_df = backtest_df.merge(
        fdr_results[['signalid', 'discovery_type', 'expected_false']], 
        on='signalid', 
        how='left'
    )
    print("  Merged discovery_type from fdr_results")
else:
    # Verify we have the expected values
    unique_types = backtest_df['discovery_type'].unique()
    print(f"  Found discovery_type values: {unique_types}")
    print(f"  Counts: {backtest_df['discovery_type'].value_counts().to_dict()}")

# Process each strategy to get cumulative returns and rolling growth
# OPTIMIZATION: Add progress reporting for long-running loop
total_strategies = len(backtest_df)
print(f"  Processing {total_strategies:,} strategies...")
for idx, row in backtest_df.iterrows():
    # Progress update every 1000 strategies
    if (idx + 1) % 1000 == 0:
        elapsed = time.time() - start_calc_time
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        remaining = total_strategies - (idx + 1)
        eta = remaining / rate if rate > 0 else 0
        print(f"    Progress: {idx+1:,}/{total_strategies:,} strategies ({rate:.0f} strategies/sec, ETA: {eta/60:.1f} min)")
    signal_id = row['signalid']
    
    # Get returns for this signal
    signal_data = returns_df[returns_df['signalid'] == signal_id].copy()
    if len(signal_data) == 0:
        continue
    
    # Get returns and check if flipped
    signal_returns = signal_data['ret'].copy()
    if row.get('flipped', False):
        signal_returns = -signal_returns  # Apply flip if needed
    
    signal_data['ret_processed'] = signal_returns
    signal_data = signal_data.sort_values('date')
    
    # Calculate cumulative returns
    signal_data_clean = signal_data.dropna(subset=['ret_processed']).sort_values('date')
    if len(signal_data_clean) < 60:  # Need at least 60 months for 5-year rolling
        continue
    
    returns_decimal = signal_data_clean['ret_processed'].values / 100.0
    dates_clean = signal_data_clean['date'].values
    
    cumulative_returns = (1 + returns_decimal).cumprod()
    cumulative_returns_by_strategy[signal_id] = {
        'returns': pd.Series(cumulative_returns, index=dates_clean),
        'dates': dates_clean,
        'discovery_type': row['discovery_type'],
        'expected_false': row['expected_false']
    }
    
    # Calculate rolling 5-year multiplicative growth (OPTIMIZED: vectorized log-space calculation)
    # Create a series with dates as index for proper alignment
    returns_series = pd.Series(returns_decimal, index=dates_clean)
    
    # OPTIMIZATION: Use log-space for numerical stability and speed
    # log(product) = sum(log), then exp(sum) = product
    log_returns = np.log1p(returns_series)  # log(1 + r) is more stable than log(1+r)
    rolling_log_sum = log_returns.rolling(window=60, min_periods=60).sum()
    rolling_growth = np.exp(rolling_log_sum)
    rolling_growth = rolling_growth.dropna()
    
    # OPTIMIZATION: Vectorized month calculation instead of loop
    if len(rolling_growth) > 0:
        # Convert dates to numpy datetime64 for vectorized operations
        dates_array = rolling_growth.index.to_numpy()
        # Vectorized calculation: months since 2004
        ref_year_month = REFERENCE_DATE.year * 12 + REFERENCE_DATE.month
        # Extract year and month from datetime64 array
        years = pd.DatetimeIndex(dates_array).year
        months = pd.DatetimeIndex(dates_array).month
        months_since_2004 = (years * 12 + months) - ref_year_month
        growth_values = rolling_growth.values
    else:
        months_since_2004 = []
        growth_values = []
    
    if months_since_2004 and growth_values:
        series_data = pd.Series(growth_values, index=months_since_2004)
        # Separate by discovery type (strip whitespace to handle any edge cases)
        discovery_type = str(row['discovery_type']).strip()
        if discovery_type == 'Expected False Discovery':
            aligned_series_false[signal_id] = series_data
        elif discovery_type == 'Expected True Discovery':
            aligned_series_true[signal_id] = series_data
        elif discovery_type == 'Not Significant':
            aligned_series_not_sig[signal_id] = series_data
        else:
            # Debug: log any unexpected values
            print(f"  WARNING: Unexpected discovery_type '{discovery_type}' for signal {signal_id}")
            aligned_series_not_sig[signal_id] = series_data  # Default to not significant

# Combine all for overall plot
aligned_series = {**aligned_series_false, **aligned_series_true, **aligned_series_not_sig}

calc_elapsed = time.time() - start_calc_time
print(f"  Processed {len(aligned_series):,} strategies with sufficient data")
print(f"    - Expected False Discoveries: {len(aligned_series_false):,}")
print(f"    - Expected True Discoveries: {len(aligned_series_true):,}")
print(f"    - Not Significant: {len(aligned_series_not_sig):,}")
print(f"  Cumulative returns calculated for {len(cumulative_returns_by_strategy):,} strategies")
print(f"  Calculation time: {calc_elapsed:.1f} seconds ({calc_elapsed/60:.1f} minutes)")
if len(backtest_df) > 0:
    print(f"  Processing rate: {len(backtest_df)/calc_elapsed:.0f} strategies/second")

# Verify separation is working correctly
if len(aligned_series) != (len(aligned_series_false) + len(aligned_series_true) + len(aligned_series_not_sig)):
    print(f"  WARNING: Group counts don't match! Total: {len(aligned_series)}, Sum of groups: {len(aligned_series_false) + len(aligned_series_true) + len(aligned_series_not_sig)}")
    
# Debug: Check for any overlap in signal IDs
false_ids = set(aligned_series_false.keys())
true_ids = set(aligned_series_true.keys())
not_sig_ids = set(aligned_series_not_sig.keys())
if false_ids & true_ids:
    print(f"  WARNING: Overlap between False and True discoveries: {len(false_ids & true_ids)} signals")
if false_ids & not_sig_ids:
    print(f"  WARNING: Overlap between False and Not Significant: {len(false_ids & not_sig_ids)} signals")
if true_ids & not_sig_ids:
    print(f"  WARNING: Overlap between True and Not Significant: {len(true_ids & not_sig_ids)} signals")

# ============================================================================
# CREATE ROLLING 5-YEAR PERCENTILE BANDS PLOT (SEPARATED BY GROUP)
# ============================================================================
if GENERATE_ROLLING_PLOTS and len(aligned_series) > 0:
    print("\nCreating rolling 5-year percentile bands plots (separated by discovery type)...")
    
    def create_percentile_plot(aligned_dict, group_name, color, output_suffix):
        """Helper function to create percentile bands plot for a group"""
        if len(aligned_dict) == 0:
            return
        
        # Combine strategies into DataFrame
        all_months = sorted(set().union(*[s.index.values for s in aligned_dict.values()]))
        aligned_series_list = []
        for signal_id, ser in aligned_dict.items():
            reindexed_ser = ser.reindex(all_months)
            reindexed_ser.name = signal_id
            aligned_series_list.append(reindexed_ser)
        
        aligned_df = pd.concat(aligned_series_list, axis=1)
        
        # Apply coverage mask
        min_fraction = 0.3
        coverage_frac = aligned_df.notnull().sum(axis=1) / aligned_df.shape[1]
        mask = coverage_frac >= min_fraction
        aligned_df_masked = aligned_df[mask]
        composite_masked = aligned_df_masked.mean(axis=1, skipna=True)
        
        # Calculate percentiles
        p25 = aligned_df_masked.quantile(0.25, axis=1)
        p50 = aligned_df_masked.quantile(0.50, axis=1)
        p75 = aligned_df_masked.quantile(0.75, axis=1)
        
        # Create the plot
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.suptitle(f"Rolling 5-Year Multiplicative Growth (×) Percentile Bands (IQR Only)\n{group_name} - Aligned by Months Since 2004{title_suffix_text}",
                     fontsize=16, color="white")
        
        # Shaded IQR band
        ax.fill_between(aligned_df_masked.index, p25, p75, color=color, alpha=0.35, label="25th–75th percentile (IQR)")
        
        # Median line
        ax.plot(p50.index, p50.values, color="lime", linewidth=2, label="Median (50th percentile)")
        
        # EW Composite line
        ax.plot(composite_masked.index, composite_masked.values, color="white", linewidth=2.6, label="Equal-Weight Composite")
        
        # Datamining cutoff marker (1980)
        datamining_year_month = DATAMINING_CUTOFF_DATE.year * 12 + DATAMINING_CUTOFF_DATE.month
        ref_year_month = REFERENCE_DATE.year * 12 + REFERENCE_DATE.month
        months_at_datamining = datamining_year_month - ref_year_month
        ax.axvline(months_at_datamining, color="orange", linestyle="--", linewidth=1.5, label=f"Datamining Cutoff ({DATAMINING_CUTOFF_DATE.year})")
        if not composite_masked.dropna().empty:
            y_top = composite_masked.dropna().max()
            ax.text(months_at_datamining, y_top * 0.90, f"{DATAMINING_CUTOFF_DATE.year}", color="orange", ha="left", va="top", fontsize=9)
        
        # Structural break marker (2004)
        months_at_2004 = 0
        ax.axvline(months_at_2004, color="gray", linestyle="--", linewidth=1.5, label="Structural Break (2004)")
        if not composite_masked.dropna().empty:
            y_top = composite_masked.dropna().max()
            ax.text(months_at_2004, y_top * 0.95, "Structural Break (2004)", color="white", ha="left", va="top", fontsize=10)
        
        ax.set_yscale("log")
        ax.set_ylabel("Rolling 5-Year Growth (×)", color="white", fontsize=12)
        ax.set_xlabel("Months Since 2004", color="white", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        
        plot_file = os.path.join(OUTPUT_PLOTS, f"rolling_5yr_percentile_bands_{output_suffix}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {plot_file}")
    
    # Create separate plots for each group
    create_percentile_plot(aligned_series_false, "Expected False Discoveries", "red", "false_discoveries")
    create_percentile_plot(aligned_series_true, "Expected True Discoveries", "green", "true_discoveries")
    create_percentile_plot(aligned_series_not_sig, "Not Significant", "gray", "not_significant")
    
    # Also create combined plot
    print("\nCreating combined rolling 5-year percentile bands plot (all groups)...")
    
    # Combine all strategies into a single DataFrame aligned by Months Since 2004
    all_months = sorted(set().union(*[s.index.values for s in aligned_series.values()]))
    aligned_series_list = []
    for signal_id, ser in aligned_series.items():
        reindexed_ser = ser.reindex(all_months)
        reindexed_ser.name = signal_id
        aligned_series_list.append(reindexed_ser)
    
    aligned_df = pd.concat(aligned_series_list, axis=1)
    
    # Apply coverage mask
    min_fraction = 0.3
    coverage_frac = aligned_df.notnull().sum(axis=1) / aligned_df.shape[1]
    mask = coverage_frac >= min_fraction
    aligned_df_masked = aligned_df[mask]
    composite_masked = aligned_df_masked.mean(axis=1, skipna=True)
    
    # Calculate percentiles
    p25 = aligned_df_masked.quantile(0.25, axis=1)
    p50 = aligned_df_masked.quantile(0.50, axis=1)
    p75 = aligned_df_masked.quantile(0.75, axis=1)
    
    # Create the combined plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.suptitle(f"Rolling 5-Year Multiplicative Growth (×) Percentile Bands (IQR Only)\nAll Strategies - Aligned by Months Since 2004{title_suffix_text}",
                 fontsize=16, color="white")
    
    # Shaded IQR band
    ax.fill_between(aligned_df_masked.index, p25, p75, color="cyan", alpha=0.35, label="25th–75th percentile (IQR)")
    
    # Median line
    ax.plot(p50.index, p50.values, color="lime", linewidth=2, label="Median (50th percentile)")
    
    # EW Composite line
    ax.plot(composite_masked.index, composite_masked.values, color="white", linewidth=2.6, label="Equal-Weight Composite")
    
    # Datamining cutoff marker (1980)
    datamining_year_month = DATAMINING_CUTOFF_DATE.year * 12 + DATAMINING_CUTOFF_DATE.month
    ref_year_month = REFERENCE_DATE.year * 12 + REFERENCE_DATE.month
    months_at_datamining = datamining_year_month - ref_year_month
    ax.axvline(months_at_datamining, color="orange", linestyle="--", linewidth=1.5, label=f"Datamining Cutoff ({DATAMINING_CUTOFF_DATE.year})")
    if not composite_masked.dropna().empty:
        y_top = composite_masked.dropna().max()
        ax.text(months_at_datamining, y_top * 0.90, f"{DATAMINING_CUTOFF_DATE.year}", color="orange", ha="left", va="top", fontsize=9)
    
    # Structural break marker (2004)
    months_at_2004 = 0
    ax.axvline(months_at_2004, color="gray", linestyle="--", linewidth=1.5, label="Structural Break (2004)")
    if not composite_masked.dropna().empty:
        y_top = composite_masked.dropna().max()
        ax.text(months_at_2004, y_top * 0.95, "Structural Break (2004)", color="white", ha="left", va="top", fontsize=10)
    
    ax.set_yscale("log")
    ax.set_ylabel("Rolling 5-Year Growth (×)", color="white", fontsize=12)
    ax.set_xlabel("Months Since 2004", color="white", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    
    plot_file = os.path.join(OUTPUT_PLOTS, "rolling_5yr_percentile_bands.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {plot_file}")
    
    # Create a 3-panel comparison plot showing all groups side-by-side
    print("\nCreating 3-panel comparison plot (False, True, Not Significant)...")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f"Rolling 5-Year Multiplicative Growth (×) Percentile Bands by Discovery Type\nAligned by Months Since 2004{title_suffix_text}",
                 fontsize=16, color="white", y=1.02)
    
    groups_data = [
        (aligned_series_false, "Expected False Discoveries", "red", axes[0]),
        (aligned_series_true, "Expected True Discoveries", "green", axes[1]),
        (aligned_series_not_sig, "Not Significant", "gray", axes[2])
    ]
    
    for aligned_dict, group_name, color, ax in groups_data:
        if len(aligned_dict) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                   ha='center', va='center', color='white', fontsize=12)
            ax.set_title(group_name, color='white', fontsize=13)
            ax.set_facecolor('black')
            continue
        
        # Combine strategies into DataFrame
        all_months = sorted(set().union(*[s.index.values for s in aligned_dict.values()]))
        aligned_series_list = []
        for signal_id, ser in aligned_dict.items():
            reindexed_ser = ser.reindex(all_months)
            aligned_series_list.append(reindexed_ser)
        
        aligned_df = pd.concat(aligned_series_list, axis=1)
        
        # Apply coverage mask
        min_fraction = 0.3
        coverage_frac = aligned_df.notnull().sum(axis=1) / aligned_df.shape[1]
        mask = coverage_frac >= min_fraction
        aligned_df_masked = aligned_df[mask]
        composite_masked = aligned_df_masked.mean(axis=1, skipna=True)
        
        # Calculate percentiles
        p25 = aligned_df_masked.quantile(0.25, axis=1)
        p50 = aligned_df_masked.quantile(0.50, axis=1)
        p75 = aligned_df_masked.quantile(0.75, axis=1)
        
        # Shaded IQR band
        ax.fill_between(aligned_df_masked.index, p25, p75, color=color, alpha=0.35, label="25th–75th percentile (IQR)")
        
        # Median line
        ax.plot(p50.index, p50.values, color="lime", linewidth=2, label="Median")
        
        # EW Composite line
        ax.plot(composite_masked.index, composite_masked.values, color="white", linewidth=2.6, label="Composite")
        
        # Datamining cutoff marker (1980)
        datamining_year_month = DATAMINING_CUTOFF_DATE.year * 12 + DATAMINING_CUTOFF_DATE.month
        ref_year_month = REFERENCE_DATE.year * 12 + REFERENCE_DATE.month
        months_at_datamining = datamining_year_month - ref_year_month
        ax.axvline(months_at_datamining, color="orange", linestyle="--", linewidth=1.5)
        if not composite_masked.dropna().empty:
            y_top = composite_masked.dropna().max()
            ax.text(months_at_datamining, y_top * 0.90, f"{DATAMINING_CUTOFF_DATE.year}", color="orange", ha="left", va="top", fontsize=8)
        
        # Structural break marker (2004)
        months_at_2004 = 0
        ax.axvline(months_at_2004, color="gray", linestyle="--", linewidth=1.5)
        if not composite_masked.dropna().empty:
            y_top = composite_masked.dropna().max()
            ax.text(months_at_2004, y_top * 0.95, "Structural Break (2004)", color="white", ha="left", va="top", fontsize=9)
        
        ax.set_yscale("log")
        ax.set_ylabel("Rolling 5-Year Growth (×)", color="white", fontsize=11)
        ax.set_xlabel("Months Since 2004", color="white", fontsize=11)
        ax.set_title(f"{group_name}\n(N={len(aligned_dict):,})", color="white", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_facecolor('black')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plot_file = os.path.join(OUTPUT_PLOTS, "rolling_5yr_by_discovery_type.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {plot_file}")

# ============================================================================
# CREATE CUMULATIVE RETURN BACKTESTS (SAMPLE OF TOP PERFORMERS)
# ============================================================================
if GENERATE_CUMULATIVE_BACKTESTS and len(cumulative_returns_by_strategy) > 0:
    print("\nCreating cumulative return backtests for top performers...")
    
    # Get top 20 performers by CAGR
    top_performers = backtest_df.nlargest(20, 'cagr')
    top_signal_ids = top_performers['signalid'].values

    # Create backtest plots for top performers
    n_plots = min(20, len(top_signal_ids))
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle(f"Cumulative Return Backtests - Top 20 Performers\n(Data-Mined Factors){title_suffix_text}", 
                 fontsize=16, color="white", y=0.995)

    axes = axes.flatten()

    for idx, signal_id in enumerate(top_signal_ids[:n_plots]):
        signal_name = get_signal_name(signal_id)
        ax = axes[idx]
        
        if signal_id in cumulative_returns_by_strategy:
            strategy_data = cumulative_returns_by_strategy[signal_id]
            cumulative = strategy_data['returns']
            dates = pd.to_datetime(strategy_data['dates'])
            
            # Normalize to start at 1
            if len(cumulative) > 0:
                cumulative_norm = cumulative / cumulative.iloc[0]
                
                # Color by discovery type
                if strategy_data['expected_false']:
                    color = 'red'
                    label_suffix = ' (False)'
                else:
                    color = 'green'
                    label_suffix = ' (True)'
                
                ax.plot(dates, cumulative_norm, color=color, linewidth=1.5, alpha=0.8)
                # Datamining cutoff line (1980)
                ax.axvline(DATAMINING_CUTOFF_DATE, color='orange', linestyle='--', linewidth=1, alpha=0.7, label=f'Datamining Cutoff ({DATAMINING_CUTOFF_DATE.year})')
                # Structural break line (2004)
                ax.axvline(STRUCTURAL_BREAK_DATE, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Structural Break (2004)')
                ax.axhline(1.0, color='white', linestyle=':', linewidth=0.5, alpha=0.3)
                
                # Get CAGR for title
                cagr_val = top_performers[top_performers['signalid'] == signal_id]['cagr'].iloc[0]
                ax.set_title(f"{signal_name}\nCAGR: {cagr_val:.1f}%{label_suffix}", 
                            color='white', fontsize=9)
                ax.set_yscale('log')
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.set_facecolor('black')
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                       ha='center', va='center', color='white')
                ax.set_facecolor('black')
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                   ha='center', va='center', color='white')
            ax.set_facecolor('black')

    # Remove extra subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].remove()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plot_file = os.path.join(OUTPUT_PLOTS, "cumulative_returns_top_20.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {plot_file}")
    
    print(f"\nAll backtest plots saved to: {OUTPUT_PLOTS}")
elif GENERATE_CUMULATIVE_BACKTESTS:
    print("\nWarning: No cumulative returns calculated. Cannot create backtest plots.")
    print(f"  Strategies processed: {len(backtest_df):,}")
    print(f"  Strategies with sufficient data: {len(cumulative_returns_by_strategy):,}")

# ============================================================================
# SAVE SUMMARY REPORT
# ============================================================================
summary_file = os.path.join(OUTPUT_RESULTS, "backtest_summary.txt")
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("Data-Mined Factors Backtest Summary\n")
    f.write("="*80 + "\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"Total Factors Analyzed: {len(backtest_df):,}\n")
    f.write(f"Expected False Discoveries: {len(false_discoveries):,}\n")
    f.write(f"Expected True Discoveries: {len(true_discoveries):,}\n\n")
    
    f.write("Expected False Discoveries Statistics:\n")
    f.write("-" * 40 + "\n")
    if len(false_discoveries) > 0:
        f.write(f"  Mean CAGR: {false_discoveries['cagr'].mean():.2f}%\n")
        f.write(f"  Median CAGR: {false_discoveries['cagr'].median():.2f}%\n")
        f.write(f"  Mean Sharpe: {false_discoveries['sharpe_ratio'].mean():.3f}\n")
        f.write(f"  Mean Max DD: {false_discoveries['max_drawdown'].mean():.2f}%\n\n")
    
    f.write("Expected True Discoveries Statistics:\n")
    f.write("-" * 40 + "\n")
    if len(true_discoveries) > 0:
        f.write(f"  Mean CAGR: {true_discoveries['cagr'].mean():.2f}%\n")
        f.write(f"  Median CAGR: {true_discoveries['cagr'].median():.2f}%\n")
        f.write(f"  Mean Sharpe: {true_discoveries['sharpe_ratio'].mean():.3f}\n")
        f.write(f"  Mean Max DD: {true_discoveries['max_drawdown'].mean():.2f}%\n")

print(f"\nSummary report saved to: {summary_file}")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_BASE}")
print(f"  - Backtests: {OUTPUT_DIR}")
print(f"    - Plots: {OUTPUT_PLOTS}")
print(f"    - Results: {OUTPUT_RESULTS}")
print(f"    - Comparisons: {OUTPUT_COMPARISONS}")

