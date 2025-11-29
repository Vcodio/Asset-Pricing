# Analysis of 27 Triple-Sort Portfolio Results

## Portfolio Naming Convention
- **MKT**: Market Beta (1=Low, 2=Medium, 3=High)
- **VOL**: Volatility Beta (1=Low, 2=Medium, 3=High)  
- **SKEW**: Skewness Beta (1=Low, 2=Medium, 3=High)

## Key Findings

### 1. Top Performers (by CAGR)
1. **MKT1_VOL3_SKEW1**: 23.41% CAGR - Low Market Beta, High Vol Beta, Low Skew Beta
2. **MKT1_VOL1_SKEW1**: 21.74% CAGR - Low Market/Vol/Skew Betas
3. **MKT3_VOL1_SKEW1**: 20.57% CAGR - High Market Beta, Low Vol/Skew Betas
4. **MKT1_VOL3_SKEW3**: 20.93% CAGR - Low Market Beta, High Vol Beta, High Skew Beta

### 2. Bottom Performers (by CAGR)
1. **MKT3_VOL2_SKEW3**: 12.07% CAGR - High Market Beta, Medium Vol Beta, High Skew Beta
2. **MKT2_VOL2_SKEW3**: 11.58% CAGR - Medium Market Beta, Medium Vol Beta, High Skew Beta
3. **MKT2_VOL2_SKEW2**: 12.05% CAGR - Medium Market/Vol/Skew Betas

### 3. Risk-Return Analysis

#### Sharpe Ratios (Risk-Adjusted Returns)
- **Highest**: MKT1_VOL1_SKEW1 (4.51) - Low beta portfolio with excellent risk-adjusted returns
- **Lowest**: MKT3_VOL2_SKEW3 (2.05) - High market, medium vol, high skew
- **Pattern**: Low beta portfolios generally show better Sharpe ratios (3.7-4.5) vs high beta (2.4-3.1)

#### Volatility (Annualized)
- **Lowest**: MKT1_VOL2_SKEW2 (3.11%) - Low market, medium vol, medium skew
- **Highest**: MKT3_VOL3_SKEW3 (7.19%) - High betas across all dimensions
- **Pattern**: Volatility generally increases with higher betas, as expected

#### Max Drawdowns
- **Best**: MKT3_VOL3_SKEW1 (-42.74%)
- **Worst**: MKT3_VOL2_SKEW3 (-63.06%)
- **Pattern**: High skew portfolios tend to have worse drawdowns

### 4. Beta Effect Analysis

#### Market Beta Effect (holding VOL and SKEW constant):
- **Low Market (MKT1)**: Average CAGR ~17%
- **Medium Market (MKT2)**: Average CAGR ~13%
- **High Market (MKT3)**: Average CAGR ~15%
- **Observation**: Low market beta portfolios outperform, consistent with low-beta anomaly

#### Volatility Beta Effect (holding MKT and SKEW constant):
- **Low Vol (VOL1)**: Average CAGR ~17%
- **Medium Vol (VOL2)**: Average CAGR ~13%
- **High Vol (VOL3)**: Average CAGR ~17%
- **Observation**: Low and high vol portfolios outperform medium vol

#### Skewness Beta Effect (holding MKT and VOL constant):
- **Low Skew (SKEW1)**: Average CAGR ~17%
- **Medium Skew (SKEW2)**: Average CAGR ~14%
- **High Skew (SKEW3)**: Average CAGR ~15%
- **Observation**: Low skew portfolios show best returns, suggesting skewness risk premium

### 5. Return Distribution Characteristics

#### Skewness
- **Most Positive**: MKT1_VOL3_SKEW1 (2.73) - right-skewed returns
- **Most Negative**: MKT1_VOL2_SKEW2 (-1.79) - left-skewed returns
- **Pattern**: Low skew beta portfolios tend toward positive skewness, high skew beta toward negative

#### Kurtosis (Tail Risk)
- **Highest**: MKT1_VOL2_SKEW1 (42.13) - extreme tail events
- **Lowest**: MKT3_VOL3_SKEW2 (4.78) - more normal distribution
- **Pattern**: Higher kurtosis in low-to-medium volatility portfolios

### 6. Best Risk-Adjusted Portfolio

**MKT1_VOL2_SKEW2** stands out with:
- **Sharpe Ratio**: 4.47 (highest)
- **CAGR**: 14.64%
- **Volatility**: 3.11% (lowest overall)
- **Calmar Ratio**: 0.35
- **Drawdown**: -41.47% (moderate)

This portfolio offers excellent risk-adjusted returns with low volatility.

### 7. Observations & Potential Issues

⚠️ **Total Return Values**: Some portfolios show extremely high total returns (e.g., MKT1_VOL1_SKEW1: 12,348%), which may indicate:
- Compounding calculation issues
- Very long time period
- Or correctly calculated cumulative returns over a long period

📊 **Key Patterns**:
1. Low beta stocks outperform (low-beta anomaly confirmed)
2. Low skew beta portfolios outperform (skewness risk premium)
3. The combination of low market + high vol + low skew creates the strongest portfolio
4. Medium beta portfolios tend to underperform

### 8. Portfolio Ranking Summary

| Rank | Portfolio | CAGR | Sharpe | Volatility | Max DD |
|------|-----------|------|--------|------------|--------|
| 1 | MKT1_VOL3_SKEW1 | 23.41% | 4.12 | 5.21% | -46.91% |
| 2 | MKT1_VOL1_SKEW1 | 21.74% | 4.51 | 4.45% | -44.77% |
| 3 | MKT3_VOL1_SKEW1 | 20.57% | 2.90 | 6.63% | -51.92% |
| 4 | MKT1_VOL3_SKEW3 | 20.93% | 4.06 | 4.77% | -48.13% |
| 5 | MKT3_VOL3_SKEW1 | 19.41% | 2.74 | 6.64% | -42.74% |

### 9. Implications for Factor Construction

Based on these results:
- **FSKEW** (Low SKEW - High SKEW) should be positive (low skew outperforms)
- **FVOL** (Low VOL - High VOL) shows mixed results depending on market beta
- The **27 portfolio structure** successfully captures the cross-sectional variation in returns

---

*Analysis Date: 2025-11-29*
*Period: Full Sample*
*Window: 21 days*

