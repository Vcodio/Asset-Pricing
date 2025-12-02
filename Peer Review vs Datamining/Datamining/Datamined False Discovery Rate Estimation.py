"""
False Discovery Rate (FDR) Analysis for Data-Mined Factors
Replicates the analysis from:
"Most claimed statistical findings in cross-sectional return predictability are likely true"
(Andrew Y. Chen, 2025)

This script:
1. Tests ~29,000 data-mined factors using Fama-MacBeth regressions (t-tests on mean returns)
2. Estimates π₀ (proportion of true nulls) using Storey's method
3. Calculates False Discovery Rate (FDR) to determine what proportion of significant
   findings are false positives vs. true discoveries

The FDR answers: "Of all the factors I found to be significant, how many are real vs. false?"
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
import time
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SIGNAL_LIST_FILE = os.path.join(SCRIPT_DIR, "DataMinedSignalList.csv")
RETURNS_FILE = os.path.join(SCRIPT_DIR, "DataMinedLongShortReturnsEW.csv")
LAMBDA_THRESHOLD = 0.5  # For Storey's π₀ estimator
P_TARGET = 0.05  # Significance threshold for FDR calculation

# METHODOLOGICAL SETTING: Use only pre-2004 data for FDR analysis?
# If True: Only uses data before 2004-01-01 (avoids look-ahead bias if factors were data-mined pre-2004)
# If False: Uses full sample (current default)
USE_PRE_2004_ONLY = True  # Recommended: True to avoid look-ahead bias
CUTOFF_DATE = pd.Timestamp('2004-01-01')

# Datamining period cutoff: When did the datamining happen?
# This is the date up to which data was used for discovering/testing factors
DATAMINING_CUTOFF_DATE = pd.Timestamp('1980-01-01')

# Structural break date: When did the structural break occur?
STRUCTURAL_BREAK_DATE = pd.Timestamp('2004-01-01')

# ================================================================
# PROMPT FOR PLOT TITLE SUFFIX (needed before creating output folders)
# ================================================================
print("\n" + "="*80)
print("PLOT TITLE SUFFIX")
print("="*80)
print("Enter a suffix to add to all plot titles (e.g., 'Pre-2004 Data', 'Full Sample', etc.)")
print("This will also be used as the output folder name to organize all outputs.")
print("Type 'NONE' or press Enter to skip adding a suffix (will use 'output' as folder name).")
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

# Output directories - organized structure with suffix-based folder
OUTPUT_BASE = os.path.join(SCRIPT_DIR, output_folder_name)
OUTPUT_DIR = os.path.join(OUTPUT_BASE, "fdr_analysis")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Create output directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================
print("="*80)
print("FDR Analysis for Data-Mined Factors")
print("="*80)
print(f"\nLoading data...")

try:
    # Load signal list
    signal_list = pd.read_csv(SIGNAL_LIST_FILE)
    print(f"  Loaded {len(signal_list):,} signals from {SIGNAL_LIST_FILE}")
    
    # Load long-short returns
    returns_df = pd.read_csv(RETURNS_FILE)
    print(f"  Loaded {len(returns_df):,} return observations from {RETURNS_FILE}")
    
    # Convert year/month to datetime for easier handling
    returns_df['date'] = pd.to_datetime(returns_df[['year', 'month']].assign(day=1))
    
    # METHODOLOGICAL: Filter to pre-2004 data if specified
    if USE_PRE_2004_ONLY:
        returns_df_original = returns_df.copy()
        returns_df = returns_df[returns_df['date'] < CUTOFF_DATE].copy()
        print(f"  Filtered to pre-2004 data: {len(returns_df):,} observations ({len(returns_df)/len(returns_df_original)*100:.1f}% of total)")
        print(f"  Date range: {returns_df['date'].min().strftime('%Y-%m')} to {returns_df['date'].max().strftime('%Y-%m')}")
        print(f"  Datamining cutoff: {DATAMINING_CUTOFF_DATE.strftime('%Y-%m')}")
        print(f"  Structural break: {STRUCTURAL_BREAK_DATE.strftime('%Y-%m')}")
        print(f"  Note: Post-structural break period not available (data ends before {STRUCTURAL_BREAK_DATE.strftime('%Y-%m')})")
    else:
        print(f"  Using full sample: {returns_df['date'].min().strftime('%Y-%m')} to {returns_df['date'].max().strftime('%Y-%m')}")
        print(f"  Datamining cutoff: {DATAMINING_CUTOFF_DATE.strftime('%Y-%m')}")
        print(f"  Structural break: {STRUCTURAL_BREAK_DATE.strftime('%Y-%m')}")
    
except FileNotFoundError as e:
    print(f"\nERROR: Could not find data files. {e}")
    print("Please ensure DataMinedSignalList.csv and DataMinedLongShortReturnsSample.csv are in the same directory.")
    exit(1)

# ============================================================================
# FAMA-MACBETH REGRESSION FUNCTION
# ============================================================================
def fama_macbeth_regression(returns_series, min_obs=12):
    """
    Performs Fama-MacBeth regression for a single factor.
    
    For long-short returns, this is equivalent to testing if the mean return
    is significantly different from zero using a t-test.
    
    Parameters:
    -----------
    returns_series : pd.Series
        Time series of long-short returns for a factor
    min_obs : int
        Minimum number of observations required (default: 12 months)
        
    Returns:
    --------
    dict with keys:
        'mean_return' : mean return
        'std_error' : standard error
        't_stat' : t-statistic
        'p_value' : two-tailed p-value
        'n_obs' : number of observations
    """
    # OPTIMIZATION: Convert to numpy array for faster operations
    returns_array = returns_series.values
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    # Require minimum observations for reliable statistics
    n_obs = len(returns_clean)
    if n_obs < min_obs:
        return {
            'mean_return': np.nan,
            'std_error': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'n_obs': n_obs
        }
    
    # OPTIMIZATION: Use numpy operations (faster than pandas)
    mean_ret = np.mean(returns_clean)
    std_ret = np.std(returns_clean, ddof=1)  # Sample std (ddof=1)
    
    # Check for zero variance (all returns are the same)
    if std_ret == 0 or n_obs < min_obs:
        return {
            'mean_return': mean_ret,
            'std_error': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'n_obs': n_obs
        }
    
    std_error = std_ret / np.sqrt(n_obs)
    t_stat = mean_ret / std_error if std_error > 0 else np.nan
    
    # Two-tailed p-value
    if not np.isnan(t_stat):
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_obs-1))
    else:
        p_value = np.nan
    
    return {
        'mean_return': mean_ret,
        'std_error': std_error,
        't_stat': t_stat,
        'p_value': p_value,
        'n_obs': n_obs
    }

# ============================================================================
# IDENTIFY SIGNALS WITH DATA
# ============================================================================
# OPTIMIZATION: Use faster unique() method
signals_with_data = returns_df['signalid'].unique()
n_signals_with_data = len(signals_with_data)
n_signals_in_list = len(signal_list)

print(f"\nSignals in list: {n_signals_in_list:,}")
print(f"Signals with return data in file: {n_signals_with_data:,}")

# ============================================================================
# FILTER SIGNALS BY DATA SUFFICIENCY (minimum 5 years = 60 months per period)
# ============================================================================
MIN_MONTHS_PER_PERIOD = 60  # 5 years minimum

print(f"\nFiltering signals by data sufficiency (minimum {MIN_MONTHS_PER_PERIOD} months per period)...")
print("This may take a moment...")

# OPTIMIZED: Use vectorized groupby operations instead of loop
# Convert dates to year-month integers for faster calculations
returns_df['year_month'] = returns_df['date'].dt.year * 12 + returns_df['date'].dt.month

# Group by signal and calculate period statistics
def check_signal_sufficiency(group):
    """Check if a signal has sufficient data in all required periods"""
    # Note: signal_id is passed as the key when iterating, not as group.name
    
    # Get date range
    min_date = group['date'].min()
    max_date = group['date'].max()
    min_ym = group['year_month'].min()
    max_ym = group['year_month'].max()
    
    # Full sample months
    full_sample_months = max_ym - min_ym + 1
    if full_sample_months < MIN_MONTHS_PER_PERIOD:
        return False
    
    # Datamining period (before cutoff)
    datamining_ym = DATAMINING_CUTOFF_DATE.year * 12 + DATAMINING_CUTOFF_DATE.month
    datamining_data = group[group['year_month'] < datamining_ym]
    if len(datamining_data) > 0:
        datamining_min = datamining_data['year_month'].min()
        datamining_max = datamining_data['year_month'].max()
        datamining_months = datamining_max - datamining_min + 1
        if datamining_months < MIN_MONTHS_PER_PERIOD:
            return False
    
    # Post-datamining period (between cutoff and structural break)
    structural_break_ym = STRUCTURAL_BREAK_DATE.year * 12 + STRUCTURAL_BREAK_DATE.month
    post_datamining_data = group[(group['year_month'] >= datamining_ym) & 
                                 (group['year_month'] < structural_break_ym)]
    if len(post_datamining_data) > 0:
        post_datamining_min = post_datamining_data['year_month'].min()
        post_datamining_max = post_datamining_data['year_month'].max()
        post_datamining_months = post_datamining_max - post_datamining_min + 1
        if post_datamining_months < MIN_MONTHS_PER_PERIOD:
            return False
    
    # Post-structural break period (only if using full sample)
    if not USE_PRE_2004_ONLY:
        post_break_data = group[group['year_month'] >= structural_break_ym]
        if len(post_break_data) > 0:
            post_break_min = post_break_data['year_month'].min()
            post_break_max = post_break_data['year_month'].max()
            post_break_months = post_break_max - post_break_min + 1
            if post_break_months < MIN_MONTHS_PER_PERIOD:
                return False
    
    return True

# OPTIMIZED: Use groupby.apply for vectorized processing
grouped = returns_df.groupby('signalid', sort=False)
n_signals_to_check = len(grouped)

if HAS_TQDM:
    print(f"  Checking data sufficiency for {n_signals_to_check:,} signals...")
    # Create progress bar wrapper
    results = {}
    pbar = tqdm(total=n_signals_to_check, desc="  Checking signals", unit="signal", 
               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    for signal_id, group in grouped:
        results[signal_id] = check_signal_sufficiency(group)
        pbar.update(1)
    pbar.close()
    sufficient_mask = pd.Series(results)
else:
    print(f"  Checking data sufficiency for {n_signals_to_check:,} signals...")
    sufficient_mask = grouped.apply(check_signal_sufficiency)

signals_with_sufficient_data = sufficient_mask[sufficient_mask].index.tolist()
signals_insufficient = sufficient_mask[~sufficient_mask].index.tolist()

n_signals_sufficient = len(signals_with_sufficient_data)
n_signals_insufficient = len(signals_insufficient)

print(f"  Signals with sufficient data (≥{MIN_MONTHS_PER_PERIOD} months in each period): {n_signals_sufficient:,}")
print(f"  Signals with insufficient data (skipped): {n_signals_insufficient:,}")

if n_signals_insufficient > 0 and n_signals_insufficient <= 10:
    print(f"\n  Examples of skipped signals (insufficient data):")
    for signal_id in signals_insufficient[:5]:
        signal_data = returns_df[returns_df['signalid'] == signal_id]
        start_date = signal_data['date'].min()
        end_date = signal_data['date'].max()
        months = ((end_date.year - start_date.year) * 12 + 
                 (end_date.month - start_date.month) + 1)
        print(f"    Signal {signal_id}: {months} months total")

# Only analyze signals that have actual return data AND sufficient data in all periods
all_signals_to_analyze = signals_with_sufficient_data
n_signals_to_analyze = len(all_signals_to_analyze)

if n_signals_with_data < n_signals_in_list:
    print(f"\nNote: Only {n_signals_with_data:,} of {n_signals_in_list:,} signals have return data.")
    print(f"Of those, {n_signals_sufficient:,} have sufficient data in all required periods.")
    print(f"Analyzing only signals with sufficient data.")

print(f"\nTotal signals to analyze: {n_signals_to_analyze:,}")

# OPTIMIZATION: Pre-filter dataframe to only signals we'll analyze
# This reduces memory and speeds up groupby operations
returns_df = returns_df[returns_df['signalid'].isin(all_signals_to_analyze)].copy()

# ============================================================================
# RUN FAMA-MACBETH FOR ALL FACTORS (OPTIMIZED)
# ============================================================================
print(f"\nRunning Fama-MacBeth regressions for {n_signals_to_analyze:,} factors...")
print("This may take a few minutes...")

# OPTIMIZATION: Use groupby for vectorized processing instead of loop
# This is much faster than filtering the dataframe for each signal
def process_signal_group(group):
    """Process a group of returns for a single signal"""
    # Note: signal_id is passed as the key when iterating, not as group.name
    signal_returns = group['ret']
    result = fama_macbeth_regression(signal_returns)
    # signal_id will be set when iterating
    return result

# OPTIMIZATION: Group by signalid and apply - much faster than manual loop
# sort=False speeds up grouping when order doesn't matter
start_time = time.time()

if HAS_TQDM:
    print("  Running Fama-MacBeth regressions...")
    # Create progress bar for groupby apply
    grouped = returns_df.groupby('signalid', sort=False)
    total_groups = len(grouped)
    
    results_list = []
    pbar = tqdm(total=total_groups, desc="  Processing signals", unit="signal",
               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for signal_id, group in grouped:
        result = process_signal_group(group)
        result['signalid'] = signal_id  # Set signal_id from the iteration key
        results_list.append(result)
        pbar.update(1)
    
    pbar.close()
else:
    # When using apply, we need to handle signal_id differently
    def process_with_id(group):
        result = process_signal_group(group)
        result['signalid'] = group.name  # group.name works with apply()
        return result
    
    results_list = returns_df.groupby('signalid', sort=False).apply(
        process_with_id, 
        include_groups=False
    ).tolist()

# Convert list of dicts to DataFrame
results_df = pd.DataFrame(results_list)
elapsed_time = time.time() - start_time
print(f"  Completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

print(f"\nCompleted Fama-MacBeth regressions for {len(results_df):,} factors")
print(f"Note: Only analyzing signals that have return data in the sample file.")

# ============================================================================
# CALCULATE π₀ (PROPORTION OF TRUE NULLS) - STOREY'S METHOD
# ============================================================================
def estimate_pi0(p_values, lambda_threshold=0.5):
    """
    Estimates π₀ (proportion of true null hypotheses) using Storey's method.
    
    π₀ = (# of p-values > λ) / ((1 - λ) * N)
    
    Parameters:
    -----------
    p_values : array-like
        Array of p-values
    lambda_threshold : float
        Threshold for Storey's estimator (default 0.5)
        
    Returns:
    --------
    float : estimated π₀
    """
    p_values_clean = np.array(p_values)[~np.isnan(p_values)]
    
    if len(p_values_clean) == 0:
        return 1.0
    
    n_total = len(p_values_clean)
    n_above_lambda = np.sum(p_values_clean > lambda_threshold)
    
    pi0 = n_above_lambda / ((1 - lambda_threshold) * n_total)
    
    # Bound π₀ between 0 and 1
    pi0 = min(max(pi0, 0), 1)
    
    return pi0

# Get p-values
p_values = results_df['p_value'].values
p_values_clean = p_values[~np.isnan(p_values)]

print(f"\nValid p-values: {len(p_values_clean):,} / {len(p_values):,}")

# Estimate π₀
pi0 = estimate_pi0(p_values_clean, lambda_threshold=LAMBDA_THRESHOLD)
print(f"\nEstimated π₀ (proportion of true nulls): {pi0:.4f}")

# ============================================================================
# CALCULATE FALSE DISCOVERY RATE (FDR)
# ============================================================================
def calculate_fdr(p_values, p_target, pi0):
    """
    Calculates the False Discovery Rate using the simple bound method:
    
    FDR ≤ (p_target * π₀) / (proportion of significant tests)
    
    Parameters:
    -----------
    p_values : array-like
        Array of p-values
    p_target : float
        Significance threshold (e.g., 0.05)
    pi0 : float
        Estimated proportion of true nulls
        
    Returns:
    --------
    float : FDR estimate
    """
    p_values_clean = np.array(p_values)[~np.isnan(p_values)]
    
    if len(p_values_clean) == 0:
        return np.nan
    
    n_total = len(p_values_clean)
    n_significant = np.sum(p_values_clean <= p_target)
    prop_significant = n_significant / n_total
    
    if prop_significant == 0:
        return np.nan
    
    fdr = (p_target * pi0) / prop_significant
    
    # FDR cannot exceed 1
    fdr = min(fdr, 1.0)
    
    return fdr

# Calculate FDR
fdr = calculate_fdr(p_values_clean, p_target=P_TARGET, pi0=pi0)

# ============================================================================
# CLASSIFY DISCOVERY TYPES FOR EACH FACTOR
# ============================================================================
print(f"\nClassifying discovery types for each factor...")

# Calculate expected number of false discoveries
n_significant = np.sum(p_values_clean <= P_TARGET)
expected_false_count = int(n_significant * fdr) if not np.isnan(fdr) else 0
expected_true_count = n_significant - expected_false_count

print(f"  Significant factors (p ≤ {P_TARGET}): {n_significant:,}")
print(f"  Expected false discoveries: {expected_false_count:,}")
print(f"  Expected true discoveries: {expected_true_count:,}")

# Initialize all as "Not Significant"
results_df['discovery_type'] = 'Not Significant'
results_df['expected_false'] = False

# Identify significant factors
significant_mask = results_df['p_value'] <= P_TARGET
significant_factors = results_df[significant_mask].copy()

if len(significant_factors) > 0 and expected_false_count > 0:
    # Rank significant factors by p-value (highest p-values = most likely false)
    # Sort by p-value descending (highest p-values first)
    significant_factors_sorted = significant_factors.sort_values('p_value', ascending=False)
    
    # Assign top N (where N = expected_false_count) as Expected False Discoveries
    # These are the factors with the highest p-values among significant ones
    false_discovery_indices = significant_factors_sorted.head(expected_false_count).index
    true_discovery_indices = significant_factors_sorted.tail(expected_true_count).index
    
    # Set discovery types
    results_df.loc[false_discovery_indices, 'discovery_type'] = 'Expected False Discovery'
    results_df.loc[false_discovery_indices, 'expected_false'] = True
    results_df.loc[true_discovery_indices, 'discovery_type'] = 'Expected True Discovery'
    results_df.loc[true_discovery_indices, 'expected_false'] = False
elif len(significant_factors) > 0:
    # If no expected false discoveries, all significant are true
    results_df.loc[significant_factors.index, 'discovery_type'] = 'Expected True Discovery'
    results_df.loc[significant_factors.index, 'expected_false'] = False

# Handle NaN p-values
nan_pvalue_mask = results_df['p_value'].isna()
results_df.loc[nan_pvalue_mask, 'discovery_type'] = 'Unknown'
results_df.loc[nan_pvalue_mask, 'expected_false'] = None

# Count by discovery type
true_discoveries = len(results_df[results_df['discovery_type'] == "Expected True Discovery"])
false_discoveries = len(results_df[results_df['discovery_type'] == "Expected False Discovery"])
not_significant = len(results_df[results_df['discovery_type'] == "Not Significant"])
unknown = len(results_df[results_df['discovery_type'] == "Unknown"])

print(f"  Classification complete:")
print(f"    Expected True Discoveries: {true_discoveries:,}")
print(f"    Expected False Discoveries: {false_discoveries:,}")
print(f"    Not Significant: {not_significant:,}")
if unknown > 0:
    print(f"    Unknown (missing p-values): {unknown:,}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
n_significant = np.sum(p_values_clean <= P_TARGET)
prop_significant = n_significant / len(p_values_clean)

# Additional statistics
mean_t_stat = results_df['t_stat'].dropna().mean()
median_t_stat = results_df['t_stat'].dropna().median()
mean_p_value = np.nanmean(p_values_clean)
median_p_value = np.nanmedian(p_values_clean)

# Check if we have sufficient data for meaningful FDR analysis
n_signals_analyzed = len(p_values_clean)
min_signals_for_fdr = 100  # Minimum recommended for reliable FDR estimation

# ============================================================================
# OUTPUT RESULTS
# ============================================================================
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nTotal number of factors analyzed: {len(results_df):,}")
print(f"Factors with valid p-values: {len(p_values_clean):,}")
print(f"\nSignificance threshold (p_target): {P_TARGET}")
print(f"Number of significant factors (p ≤ {P_TARGET}): {n_significant:,}")
print(f"Proportion of significant factors: {prop_significant:.4f} ({prop_significant*100:.2f}%)")

print(f"\n--- π₀ Estimation (Storey's Method) ---")
print(f"Lambda threshold: {LAMBDA_THRESHOLD}")
print(f"Estimated π₀ (proportion of true nulls): {pi0:.4f}")
print(f"Estimated proportion of true factors: {1-pi0:.4f}")

print(f"\n--- False Discovery Rate (FDR) ---")
if np.isnan(fdr):
    print(f"FDR cannot be calculated: No factors are significant at p ≤ {P_TARGET}")
    print(f"  (Need at least 1 significant factor to calculate FDR)")
else:
    print(f"FDR estimate (p ≤ {P_TARGET}): {fdr:.4f} ({fdr*100:.2f}%)")
    print(f"\nInterpretation:")
    print(f"  Of the {n_significant:,} factors significant at p ≤ {P_TARGET},")
    print(f"  approximately {fdr*100:.2f}% are expected to be false discoveries.")
    if n_significant > 0:
        print(f"  This implies {n_significant * (1-fdr):.0f} true discoveries.")

# Data quality warning
if n_signals_analyzed < min_signals_for_fdr:
    print(f"\n⚠️  WARNING: Limited Sample Size")
    print(f"  Only {n_signals_analyzed} signals have sufficient data in the sample file.")
    print(f"  For reliable FDR analysis, recommend analyzing at least {min_signals_for_fdr:,} signals.")
    print(f"  The current results are based on a very small sample and may not be representative.")

print(f"\n--- Additional Statistics ---")
print(f"Mean t-statistic: {mean_t_stat:.4f}")
print(f"Median t-statistic: {median_t_stat:.4f}")
print(f"Mean p-value: {mean_p_value:.4f}")
print(f"Median p-value: {median_p_value:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
output_file = os.path.join(RESULTS_DIR, "fdr_analysis_results.csv")
results_df.to_csv(output_file, index=False)
print(f"\nDetailed results saved to: {output_file}")

# Save summary
summary = {
    'n_factors': len(results_df),
    'n_valid_pvalues': len(p_values_clean),
    'n_significant': n_significant,
    'prop_significant': prop_significant,
    'pi0': pi0,
    'fdr': fdr,
    'p_target': P_TARGET,
    'lambda_threshold': LAMBDA_THRESHOLD
}

summary_df = pd.DataFrame([summary])
summary_file = os.path.join(RESULTS_DIR, "fdr_summary.csv")
summary_df.to_csv(summary_file, index=False)
print(f"Summary statistics saved to: {summary_file}")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================
print(f"\nCreating visualizations...")

# Set style
plt.style.use("dark_background")

# 1. P-value Distribution Histogram
print("  Creating p-value distribution plot...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
title_suffix_text = f" - {plot_title_suffix}" if plot_title_suffix else ""
fig.suptitle(f"FDR Analysis Visualizations{title_suffix_text}", fontsize=16, color="white", y=0.995)

# Top-left: P-value histogram
ax = axes[0, 0]
p_vals_plot = p_values_clean[p_values_clean <= 1.0]  # Remove any outliers
n, bins, patches = ax.hist(p_vals_plot, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
ax.axvline(P_TARGET, color='red', linestyle='--', linewidth=2, label=f'Significance threshold (p={P_TARGET})')
ax.axvline(LAMBDA_THRESHOLD, color='orange', linestyle='--', linewidth=2, label=f'Lambda threshold (λ={LAMBDA_THRESHOLD})')
ax.axhline(len(p_vals_plot) * (1 - LAMBDA_THRESHOLD) / 50, color='yellow', linestyle=':', 
           linewidth=1, alpha=0.5, label='Expected uniform distribution')
ax.set_xlabel('P-value', color='white', fontsize=11)
ax.set_ylabel('Frequency', color='white', fontsize=11)
ax.set_title('P-value Distribution', color='white', fontsize=12, pad=10)
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.3)
ax.text(0.02, 0.95, f'Total factors: {len(p_vals_plot):,}\nSignificant (p≤{P_TARGET}): {n_significant:,}\nP > λ: {np.sum(p_values_clean > LAMBDA_THRESHOLD):,}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

# Top-right: P-value Q-Q plot (uniform distribution)
ax = axes[0, 1]
sorted_pvals = np.sort(p_values_clean)
n_vals = len(sorted_pvals)
theoretical_quantiles = np.linspace(0, 1, n_vals)
ax.scatter(theoretical_quantiles, sorted_pvals, s=1, alpha=0.5, color='steelblue')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect uniform distribution')
ax.set_xlabel('Theoretical Quantiles (Uniform)', color='white', fontsize=11)
ax.set_ylabel('Observed P-values', color='white', fontsize=11)
ax.set_title('P-value Q-Q Plot (Uniform Distribution)', color='white', fontsize=12, pad=10)
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.3)
ax.text(0.05, 0.95, 'If all factors were null,\np-values would follow\nthe red line',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

# Bottom-left: T-statistic distribution
ax = axes[1, 0]
t_stats_clean = results_df['t_stat'].dropna()
t_stats_plot = t_stats_clean[(t_stats_clean >= -10) & (t_stats_clean <= 10)]  # Remove extreme outliers
n, bins, patches = ax.hist(t_stats_plot, bins=50, color='green', edgecolor='white', alpha=0.7)
ax.axvline(0, color='white', linestyle='-', linewidth=1, alpha=0.5)
# Critical t-values for p=0.05 (two-tailed, approximate)
critical_t = stats.t.ppf(0.975, df=250)  # Approximate df
ax.axvline(critical_t, color='red', linestyle='--', linewidth=2, label=f'Critical t (p={P_TARGET})')
ax.axvline(-critical_t, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('T-statistic', color='white', fontsize=11)
ax.set_ylabel('Frequency', color='white', fontsize=11)
ax.set_title('T-statistic Distribution', color='white', fontsize=12, pad=10)
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.3)
ax.text(0.02, 0.95, f'Mean t-stat: {mean_t_stat:.3f}\nMedian t-stat: {median_t_stat:.3f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

# Bottom-right: Mean return distribution
ax = axes[1, 1]
mean_returns_clean = results_df['mean_return'].dropna()
mean_returns_plot = mean_returns_clean[(mean_returns_clean >= -2) & (mean_returns_clean <= 2)]  # Remove extreme outliers
n, bins, patches = ax.hist(mean_returns_plot, bins=50, color='purple', edgecolor='white', alpha=0.7)
ax.axvline(0, color='white', linestyle='-', linewidth=2, label='Zero return')
mean_ret_mean = mean_returns_clean.mean()
ax.axvline(mean_ret_mean, color='yellow', linestyle='--', linewidth=2, label=f'Mean: {mean_ret_mean:.4f}')
ax.set_xlabel('Mean Return (per month)', color='white', fontsize=11)
ax.set_ylabel('Frequency', color='white', fontsize=11)
ax.set_title('Mean Return Distribution', color='white', fontsize=12, pad=10)
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.3)
median_ret = mean_returns_clean.median()
ax.text(0.02, 0.95, f'Mean: {mean_ret_mean:.4f}\nMedian: {median_ret:.4f}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

plt.tight_layout()
plt.subplots_adjust(top=0.95)
plot_file_1 = os.path.join(PLOTS_DIR, "fdr_analysis_overview.png")
plt.savefig(plot_file_1, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {plot_file_1}")

# 2. FDR Analysis Summary Plot
print("  Creating FDR summary plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
title_suffix_text = f" - {plot_title_suffix}" if plot_title_suffix else ""
fig.suptitle(f"False Discovery Rate (FDR) Analysis Summary{title_suffix_text}", fontsize=16, color="white", y=1.02)

# Add date markers annotation
fig.text(0.5, 0.01, 
         f'Datamining Cutoff: {DATAMINING_CUTOFF_DATE.strftime("%Y")} | Structural Break: {STRUCTURAL_BREAK_DATE.strftime("%Y")}',
         ha='center', fontsize=10, color='yellow', style='italic')

# Left: Significance breakdown
ax = axes[0]
significant_mask = p_values_clean <= P_TARGET
n_sig = np.sum(significant_mask)
n_not_sig = len(p_values_clean) - n_sig
expected_false = n_sig * fdr if not np.isnan(fdr) else 0
expected_true = n_sig - expected_false

categories = ['Not Significant\n(p > 0.05)', 'Significant\n(p ≤ 0.05)']
counts = [n_not_sig, n_sig]
colors = ['gray', 'steelblue']
bars = ax.bar(categories, counts, color=colors, edgecolor='white', linewidth=2, alpha=0.8)

# Add value labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}\n({count/len(p_values_clean)*100:.1f}%)',
            ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')

ax.set_ylabel('Number of Factors', color='white', fontsize=12)
ax.set_title('Factors by Significance', color='white', fontsize=13, pad=10)
ax.grid(True, axis='y', linestyle='--', alpha=0.3)
ax.set_facecolor('black')

# Add breakdown of significant factors
if not np.isnan(fdr) and n_sig > 0:
    y_pos = n_sig * 0.7
    ax.text(1, y_pos, f'Expected False Discoveries:\n{expected_false:.0f} ({fdr*100:.1f}%)',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
    ax.text(1, y_pos * 0.4, f'Expected True Discoveries:\n{expected_true:.0f} ({(1-fdr)*100:.1f}%)',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))

# Right: π₀ estimation visualization
ax = axes[1]
# Create bins for p-values
bins = np.linspace(0, 1, 51)
n, bins_edges, patches = ax.hist(p_values_clean, bins=bins, color='steelblue', 
                                  edgecolor='white', alpha=0.7, label='Observed p-values')

# Color bars above lambda threshold
lambda_idx = np.searchsorted(bins_edges, LAMBDA_THRESHOLD)
for i, patch in enumerate(patches):
    if i >= lambda_idx:
        patch.set_facecolor('orange')
        patch.set_alpha(0.7)

# Expected uniform distribution line
expected_uniform = len(p_values_clean) / len(bins)
ax.axhline(expected_uniform, color='yellow', linestyle=':', linewidth=2, 
           label=f'Expected if all null (N/{len(bins)})')

# Mark lambda threshold
ax.axvline(LAMBDA_THRESHOLD, color='orange', linestyle='--', linewidth=2, 
           label=f'λ = {LAMBDA_THRESHOLD} (Storey threshold)')
ax.axvline(P_TARGET, color='red', linestyle='--', linewidth=2, 
           label=f'p = {P_TARGET} (significance threshold)')

# Add π₀ calculation annotation (positioned to avoid legend overlap)
n_above_lambda = np.sum(p_values_clean > LAMBDA_THRESHOLD)
expected_if_all_null = len(p_values_clean) * (1 - LAMBDA_THRESHOLD)
# Position annotation in upper left to avoid legend in upper right
ax.text(0.02, 0.98, 
        f'π₀ Estimation:\n'
        f'P-values > λ: {n_above_lambda:,}\n'
        f'Expected if all null: {expected_if_all_null:.0f}\n'
        f'π₀ = {n_above_lambda:,} / {expected_if_all_null:.0f} = {pi0:.4f}\n\n'
        f'Datamining: ≤{DATAMINING_CUTOFF_DATE.strftime("%Y")}\n'
        f'Structural Break: {STRUCTURAL_BREAK_DATE.strftime("%Y")}',
        fontsize=10, color='white',
        transform=ax.transAxes,  # Use axes coordinates (0-1)
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

ax.set_xlabel('P-value', color='white', fontsize=12)
ax.set_ylabel('Frequency', color='white', fontsize=12)
ax.set_title('Storey\'s π₀ Estimation', color='white', fontsize=13, pad=10)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_facecolor('black')

plt.tight_layout()
plot_file_2 = os.path.join(PLOTS_DIR, "fdr_summary.png")
plt.savefig(plot_file_2, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {plot_file_2}")

# 3. Detailed P-value Distribution (zoomed in on [0, 0.1])
print("  Creating detailed p-value distribution (0-0.1)...")
fig, ax = plt.subplots(figsize=(12, 7))
title_suffix_text = f" - {plot_title_suffix}" if plot_title_suffix else ""
fig.suptitle(f"P-value Distribution (Detailed View: 0 to 0.1){title_suffix_text}", fontsize=16, color="white")

# Add date markers annotation
fig.text(0.5, 0.01, 
         f'Datamining Cutoff: {DATAMINING_CUTOFF_DATE.strftime("%Y")} | Structural Break: {STRUCTURAL_BREAK_DATE.strftime("%Y")}',
         ha='center', fontsize=10, color='yellow', style='italic')

p_vals_zoom = p_values_clean[(p_values_clean >= 0) & (p_values_clean <= 0.1)]
n, bins, patches = ax.hist(p_vals_zoom, bins=50, color='steelblue', edgecolor='white', alpha=0.7)

# Color significant factors
sig_idx = np.searchsorted(bins, P_TARGET)
for i, patch in enumerate(patches):
    if i < sig_idx:
        patch.set_facecolor('green')
        patch.set_alpha(0.7)

ax.axvline(P_TARGET, color='red', linestyle='--', linewidth=2, 
           label=f'Significance threshold (p={P_TARGET})')
ax.axhline(len(p_vals_zoom) / 50, color='yellow', linestyle=':', linewidth=1, 
           alpha=0.5, label='Expected uniform distribution')

ax.set_xlabel('P-value', color='white', fontsize=12)
ax.set_ylabel('Frequency', color='white', fontsize=12)
ax.set_title('Zoomed View: Most Significant Factors', color='white', fontsize=13, pad=10)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_facecolor('black')

# Add statistics
n_sig_zoom = np.sum(p_vals_zoom <= P_TARGET)
ax.text(0.02, 0.95, 
        f'Factors with p ≤ {P_TARGET}: {n_sig_zoom:,}\n'
        f'Total in range [0, 0.1]: {len(p_vals_zoom):,}\n\n'
        f'Datamining: ≤{DATAMINING_CUTOFF_DATE.strftime("%Y")}\n'
        f'Structural Break: {STRUCTURAL_BREAK_DATE.strftime("%Y")}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

plt.tight_layout()
plot_file_3 = os.path.join(PLOTS_DIR, "pvalue_distribution_detailed.png")
plt.savefig(plot_file_3, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {plot_file_3}")

print(f"\nAll visualizations saved to: {PLOTS_DIR}")
print(f"\nAll output files organized in: {OUTPUT_BASE}")
print(f"  - FDR Analysis: {OUTPUT_DIR}")
print(f"    - Results (CSV): {RESULTS_DIR}")
print(f"    - Plots (PNG): {PLOTS_DIR}")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)

