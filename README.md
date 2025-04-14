# Chapter 28: Hidden Markov Models — Regime-Switching Trading Strategy

## Overview

Финансовые рынки функционируют в различных режимах (bull, bear, sideways), каждый со своими статистическими характеристиками. Hidden Markov Models (HMM) позволяют автоматически определять текущий режим и вероятности перехода между режимами. В этой главе мы строим стратегию, которая адаптирует свое поведение к текущему рыночному режиму.

## Trading Strategy

**Суть стратегии:** HMM определяет 3 режима рынка. Для каждого режима — своя подстратегия:
- **Bull regime:** Aggressive momentum, high equity allocation
- **Bear regime:** Defensive, low equity / high bonds / hedges
- **Sideways regime:** Mean-reversion, pairs trading

**Сигнал на переключение:** Когда posterior probability нового режима превышает порог (например, 70%)

**Position Sizing:** Зависит от confidence в текущем режиме и волатильности режима

## Technical Specification

### Notebooks to Create

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_data_preparation.ipynb` | Загрузка S&P 500, VIX, yields, macro данных |
| 2 | `02_hmm_theory.ipynb` | Теория HMM: forward-backward, Viterbi, Baum-Welch |
| 3 | `03_regime_features.ipynb` | Feature engineering для определения режимов |
| 4 | `04_gaussian_hmm.ipynb` | Обучение Gaussian HMM с hmmlearn |
| 5 | `05_regime_interpretation.ipynb` | Интерпретация режимов, визуализация |
| 6 | `06_regime_characteristics.ipynb` | Статистика каждого режима (return, vol, duration) |
| 7 | `07_substrategy_bull.ipynb` | Momentum стратегия для bull режима |
| 8 | `08_substrategy_bear.ipynb` | Defensive стратегия для bear режима |
| 9 | `09_substrategy_sideways.ipynb` | Mean-reversion для sideways режима |
| 10 | `10_strategy_switching.ipynb` | Логика переключения между стратегиями |
| 11 | `11_backtesting.ipynb` | Full backtest с transaction costs |
| 12 | `12_robustness_analysis.ipynb` | Sensitivity analysis, out-of-sample |

### Data Requirements

```
Market Data:
├── S&P 500 index daily (20+ лет для разных режимов)
├── VIX index
├── US Treasury yields (2Y, 10Y)
├── Credit spreads (BAA-AAA)
└── Gold, USD index

Macro Indicators:
├── NBER recession dates (ground truth)
├── Leading Economic Index (LEI)
├── Yield curve slope
└── Unemployment rate
```

### HMM Configuration

```python
from hmmlearn import hmm

# 3-state Gaussian HMM
model = hmm.GaussianHMM(
    n_components=3,           # Bull, Bear, Sideways
    covariance_type="full",   # Full covariance matrix
    n_iter=1000,
    random_state=42
)

# Features for regime detection
features = [
    'return_20d',      # 20-day return
    'volatility_20d',  # 20-day realized vol
    'vix_level',       # VIX
    'yield_slope',     # 10Y - 2Y
    'credit_spread'    # BAA - AAA
]
```

### Regime Characteristics (Expected)

| Regime | Avg Return | Volatility | Avg Duration | Typical Indicators |
|--------|------------|------------|--------------|-------------------|
| Bull | +15% ann. | 12% | 24 months | Low VIX, steep yield curve |
| Bear | -20% ann. | 25% | 8 months | High VIX, inverted curve |
| Sideways | +5% ann. | 15% | 12 months | Mid VIX, flat curve |

### Sub-strategies per Regime

```
Bull Regime Strategy:
├── Long equity (100% allocation)
├── Momentum factor tilt
├── Small cap overweight
└── No hedges

Bear Regime Strategy:
├── Reduced equity (30% allocation)
├── Long-term treasuries (40%)
├── Gold allocation (20%)
├── Cash buffer (10%)
└── Optional: VIX calls hedge

Sideways Regime Strategy:
├── Market neutral (50% equity)
├── Pairs trading overlay
├── Sector rotation (momentum within)
└── Enhanced yield strategies
```

### Transition Matrix Example

```
         To Bull  To Bear  To Sideways
From Bull    0.92    0.03       0.05
From Bear    0.08    0.85       0.07
From Sideways 0.10   0.08       0.82
```

### Key Metrics

- **Regime Detection:** Accuracy vs NBER, Average regime duration, False switches
- **Strategy:** Sharpe Ratio, Max Drawdown, Calmar Ratio
- **Comparison:** vs Buy&Hold, vs 60/40, vs static momentum

### Dependencies

```python
hmmlearn>=0.3.0
pomegranate>=1.0.0  # Alternative HMM library
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
yfinance>=0.2.0
```

## Expected Outcomes

1. **3-state HMM** с интерпретируемыми режимами (bull/bear/sideways)
2. **Visualization** режимов на историческом графике S&P 500
3. **Статистика режимов** — duration, returns, volatility per regime
4. **3 подстратегии** оптимизированные для каждого режима
5. **Switching logic** с учетом transaction costs и whipsaws
6. **Backtest results** с улучшением risk-adjusted returns vs buy&hold

## References

- [Regime Switching Models for Financial Time Series](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1339685)
- [Hidden Markov Models for Time Series](https://www.routledge.com/Hidden-Markov-Models-for-Time-Series-An-Introduction-Using-R/Zucchini-MacDonald-Langrock/p/book/9781482253832)
- [hmmlearn Documentation](https://hmmlearn.readthedocs.io/)
- [Market Regime Detection Using Machine Learning](https://jpm.pm-research.com/)

## Difficulty Level

⭐⭐⭐☆☆ (Intermediate)

Требуется понимание: HMM theory, Time series analysis, Portfolio construction
