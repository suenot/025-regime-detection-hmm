//! Portfolio management

use crate::data::Candle;

/// Simple portfolio tracker
#[derive(Debug, Clone)]
pub struct Portfolio {
    /// Current cash balance
    pub cash: f64,
    /// Current position size (units of asset)
    pub position: f64,
    /// Average entry price
    pub entry_price: f64,
    /// Total equity (cash + position value)
    pub equity: f64,
    /// Transaction cost rate
    pub transaction_cost: f64,
    /// Equity history
    pub equity_history: Vec<f64>,
    /// Returns history
    pub returns_history: Vec<f64>,
    /// Trade returns (closed trades)
    pub trade_returns: Vec<f64>,
}

impl Portfolio {
    /// Create new portfolio with initial capital
    pub fn new(initial_capital: f64) -> Self {
        Self {
            cash: initial_capital,
            position: 0.0,
            entry_price: 0.0,
            equity: initial_capital,
            transaction_cost: 0.001, // 0.1% default
            equity_history: vec![initial_capital],
            returns_history: vec![],
            trade_returns: vec![],
        }
    }

    /// Set transaction cost rate
    pub fn with_transaction_cost(mut self, cost: f64) -> Self {
        self.transaction_cost = cost;
        self
    }

    /// Check if has position
    pub fn has_position(&self) -> bool {
        self.position.abs() > 1e-10
    }

    /// Get current position value
    pub fn position_value(&self, price: f64) -> f64 {
        self.position * price
    }

    /// Get unrealized PnL
    pub fn unrealized_pnl(&self, price: f64) -> f64 {
        if self.has_position() {
            self.position * (price - self.entry_price)
        } else {
            0.0
        }
    }

    /// Update equity and record history
    pub fn update(&mut self, price: f64) {
        let prev_equity = self.equity;
        self.equity = self.cash + self.position_value(price);

        if prev_equity > 0.0 {
            let ret = (self.equity - prev_equity) / prev_equity;
            self.returns_history.push(ret);
        }

        self.equity_history.push(self.equity);
    }

    /// Set position to target size
    pub fn set_position(&mut self, target_position: f64, price: f64) {
        let delta = target_position - self.position;

        if delta.abs() < 1e-10 {
            return;
        }

        // Calculate trade value and cost
        let trade_value = delta * price;
        let cost = trade_value.abs() * self.transaction_cost;

        // Record trade return if closing
        if self.has_position() && target_position.abs() < 1e-10 {
            let trade_return = (price - self.entry_price) / self.entry_price * self.position.signum();
            self.trade_returns.push(trade_return);
        }

        // Update cash
        self.cash -= trade_value + cost;

        // Update position
        if target_position.abs() > 1e-10 {
            // If opening or adding to position, update entry price
            if (self.position >= 0.0 && delta > 0.0) || (self.position <= 0.0 && delta < 0.0) {
                // Adding to position - calculate average entry
                let total_value =
                    self.position * self.entry_price + delta * price;
                self.entry_price = total_value / (self.position + delta);
            }
        }

        self.position = target_position;
    }

    /// Close all positions
    pub fn close_position(&mut self, price: f64) {
        self.set_position(0.0, price);
    }

    /// Get position as fraction of equity
    pub fn position_fraction(&self, price: f64) -> f64 {
        if self.equity > 1e-10 {
            self.position_value(price) / self.equity
        } else {
            0.0
        }
    }

    /// Target position by fraction of equity
    pub fn target_fraction(&mut self, fraction: f64, price: f64) {
        let target_value = self.equity * fraction;
        let target_units = target_value / price;
        self.set_position(target_units, price);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_creation() {
        let portfolio = Portfolio::new(10000.0);
        assert_eq!(portfolio.cash, 10000.0);
        assert!(!portfolio.has_position());
    }

    #[test]
    fn test_buy_and_sell() {
        let mut portfolio = Portfolio::new(10000.0).with_transaction_cost(0.0);

        // Buy at 100
        portfolio.set_position(50.0, 100.0);
        assert_eq!(portfolio.position, 50.0);
        assert_eq!(portfolio.cash, 5000.0);

        // Price goes up to 110
        portfolio.update(110.0);
        assert!((portfolio.equity - 10500.0).abs() < 1e-10);

        // Sell
        portfolio.close_position(110.0);
        assert!(!portfolio.has_position());
        assert!((portfolio.cash - 10500.0).abs() < 1e-10);
    }

    #[test]
    fn test_transaction_costs() {
        let mut portfolio = Portfolio::new(10000.0).with_transaction_cost(0.01); // 1%

        // Buy 50 units at 100 = 5000 value, 50 cost
        portfolio.set_position(50.0, 100.0);
        assert!((portfolio.cash - 4950.0).abs() < 1e-10);
    }
}
