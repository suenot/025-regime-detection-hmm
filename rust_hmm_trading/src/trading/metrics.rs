//! Trading performance metrics

/// Calculate Sharpe ratio
pub fn calculate_sharpe(returns: &[f64], risk_free_rate: f64, annualization: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std = variance.sqrt();

    if std < 1e-10 {
        return 0.0;
    }

    let excess_return = mean - risk_free_rate / annualization;
    (excess_return / std) * annualization.sqrt()
}

/// Calculate maximum drawdown
pub fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }

    let mut max_equity = equity_curve[0];
    let mut max_drawdown = 0.0;

    for &equity in equity_curve {
        if equity > max_equity {
            max_equity = equity;
        }
        let drawdown = (max_equity - equity) / max_equity;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    max_drawdown
}

/// Calculate Calmar ratio (return / max drawdown)
pub fn calculate_calmar(returns: &[f64], equity_curve: &[f64], annualization: f64) -> f64 {
    let annual_return = returns.iter().sum::<f64>() / returns.len() as f64 * annualization;
    let max_dd = calculate_max_drawdown(equity_curve);

    if max_dd < 1e-10 {
        return 0.0;
    }

    annual_return / max_dd
}

/// Calculate Sortino ratio (downside deviation)
pub fn calculate_sortino(returns: &[f64], risk_free_rate: f64, annualization: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let target = risk_free_rate / annualization;

    // Downside deviation: only negative deviations from target
    let downside_var = returns
        .iter()
        .map(|r| {
            let excess = r - target;
            if excess < 0.0 {
                excess.powi(2)
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / returns.len() as f64;

    let downside_std = downside_var.sqrt();

    if downside_std < 1e-10 {
        return 0.0;
    }

    let excess_return = mean - target;
    (excess_return / downside_std) * annualization.sqrt()
}

/// Collection of trading metrics
#[derive(Debug, Clone)]
pub struct TradingMetrics {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annual_return: f64,
    /// Annualized volatility
    pub annual_volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Win rate
    pub win_rate: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
}

impl TradingMetrics {
    /// Calculate all metrics from returns and equity curve
    pub fn calculate(
        returns: &[f64],
        equity_curve: &[f64],
        trade_returns: &[f64],
        annualization: f64,
        risk_free_rate: f64,
    ) -> Self {
        let total_return = if equity_curve.len() >= 2 {
            (equity_curve.last().unwrap() - equity_curve.first().unwrap())
                / equity_curve.first().unwrap()
        } else {
            0.0
        };

        let mean_return = if !returns.is_empty() {
            returns.iter().sum::<f64>() / returns.len() as f64
        } else {
            0.0
        };

        let variance = if !returns.is_empty() {
            returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / returns.len() as f64
        } else {
            0.0
        };

        let annual_return = mean_return * annualization;
        let annual_volatility = variance.sqrt() * annualization.sqrt();

        let sharpe_ratio = calculate_sharpe(returns, risk_free_rate, annualization);
        let sortino_ratio = calculate_sortino(returns, risk_free_rate, annualization);
        let max_drawdown = calculate_max_drawdown(equity_curve);
        let calmar_ratio = calculate_calmar(returns, equity_curve, annualization);

        let num_trades = trade_returns.len();
        let wins = trade_returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = if num_trades > 0 {
            wins as f64 / num_trades as f64
        } else {
            0.0
        };

        let avg_trade_return = if num_trades > 0 {
            trade_returns.iter().sum::<f64>() / num_trades as f64
        } else {
            0.0
        };

        let gross_profit: f64 = trade_returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = trade_returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let profit_factor = if gross_loss > 1e-10 {
            gross_profit / gross_loss
        } else {
            f64::INFINITY
        };

        Self {
            total_return,
            annual_return,
            annual_volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            num_trades,
            avg_trade_return,
            profit_factor,
        }
    }

    /// Print metrics summary
    pub fn print_summary(&self) {
        println!("\n=== Trading Metrics ===\n");
        println!("Total Return:      {:.2}%", self.total_return * 100.0);
        println!("Annual Return:     {:.2}%", self.annual_return * 100.0);
        println!("Annual Volatility: {:.2}%", self.annual_volatility * 100.0);
        println!("Sharpe Ratio:      {:.2}", self.sharpe_ratio);
        println!("Sortino Ratio:     {:.2}", self.sortino_ratio);
        println!("Max Drawdown:      {:.2}%", self.max_drawdown * 100.0);
        println!("Calmar Ratio:      {:.2}", self.calmar_ratio);
        println!("Win Rate:          {:.1}%", self.win_rate * 100.0);
        println!("Number of Trades:  {}", self.num_trades);
        println!("Avg Trade Return:  {:.2}%", self.avg_trade_return * 100.0);
        println!("Profit Factor:     {:.2}", self.profit_factor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio() {
        // Positive returns with low volatility should give high Sharpe
        let returns = vec![0.01, 0.02, 0.01, 0.015, 0.01];
        let sharpe = calculate_sharpe(&returns, 0.02, 252.0);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 105.0, 90.0, 95.0, 100.0];
        let dd = calculate_max_drawdown(&equity);
        // Max drawdown from 110 to 90 = 18.18%
        assert!((dd - 0.1818).abs() < 0.01);
    }

    #[test]
    fn test_calmar_ratio() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.01];
        let equity = vec![100.0, 101.0, 103.0, 102.0, 103.5, 104.5];
        let calmar = calculate_calmar(&returns, &equity, 252.0);
        assert!(calmar.is_finite());
    }
}
