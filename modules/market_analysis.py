#!/usr/bin/env python3
"""
PHOENIX Market Analysis Module
Autonomous trading decision system with AI-powered analysis
Integrates with market dashboard for real-time trading signals
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import pickle
from typing import Dict, List, Tuple, Optional, Any
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optional PHOENIX integration - will work standalone if modules not available
try:
    from core.phoenix_core import PhoenixCore
    from modules.memory_manager import MemoryManager
    from modules.advanced_learning import AdvancedLearning
    from modules.safety_guard import SafetyGuard
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

class MarketAnalysisModule:
    """
    PHOENIX Market Analysis Module
    Performs autonomous trading analysis and decision-making
    """

    def __init__(self, phoenix_core=None):
        """Initialize market analysis module"""
        # Initialize PHOENIX components if available
        if PHOENIX_AVAILABLE:
            try:
                self.phoenix = phoenix_core or PhoenixCore()

                # Initialize with config
                config = {
                    'db_path': str(Path.home() / 'PHOENIX' / 'memory' / 'phoenix_memory.db'),
                    'vector_store_path': str(Path.home() / 'PHOENIX' / 'memory' / 'chroma')
                }

                self.memory = MemoryManager(config)

                # Initialize learning with proper config
                learning_config = {
                    'model_path': str(Path.home() / 'PHOENIX' / 'models'),
                    'learning_rate': 0.001
                }
                self.learning = AdvancedLearning(learning_config)

                self.safety = SafetyGuard()
            except Exception as e:
                print(f"Warning: PHOENIX integration limited: {e}")
                self.phoenix = None
                self.memory = None
                self.learning = None
                self.safety = None
        else:
            self.phoenix = None
            self.memory = None
            self.learning = None
            self.safety = None

        # Market data paths
        self.market_data_dir = Path.home() / '.market_widgets' / 'history'
        self.decisions_db = Path.home() / 'PHOENIX' / 'data' / 'trading_decisions.db'
        self.models_dir = Path.home() / 'PHOENIX' / 'models' / 'market'

        # Technical indicators parameters
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.ma_periods = [20, 50, 200]

        # Risk parameters (adjustable)
        self.risk_config = {
            'max_position_size': 0.1,  # Max 10% of portfolio per trade
            'stop_loss': 0.02,          # 2% stop loss
            'take_profit': 0.05,        # 5% take profit
            'confidence_threshold': 0.7, # Minimum 70% confidence for trade
            'max_daily_trades': 5,       # Max trades per day
            'risk_level': 'moderate'     # conservative, moderate, aggressive
        }

        # Initialize components
        self.setup_database()
        self.load_models()

        # Logger
        self.logger = logging.getLogger('PHOENIX.MarketAnalysis')

    def setup_database(self):
        """Setup decision tracking database"""
        self.models_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.decisions_db)
        cursor = conn.cursor()

        # Trading decisions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                signal TEXT,
                confidence REAL,
                price REAL,
                indicators TEXT,
                explanation TEXT,
                risk_assessment TEXT,
                executed BOOLEAN,
                outcome TEXT,
                profit_loss REAL,
                learning_notes TEXT
            )
        ''')

        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_profit_loss REAL,
                win_rate REAL,
                avg_profit REAL,
                avg_loss REAL,
                sharpe_ratio REAL,
                max_drawdown REAL
            )
        ''')

        # Pattern recognition table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                discovered_date TEXT,
                pattern_name TEXT,
                pattern_data TEXT,
                success_rate REAL,
                occurrences INTEGER,
                last_seen TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def load_models(self):
        """Load or initialize ML models for pattern recognition"""
        self.models = {}

        # Load existing models if available
        models_files = {
            'pattern_classifier': 'pattern_classifier.pkl',
            'risk_assessor': 'risk_assessor.pkl',
            'signal_generator': 'signal_generator.pkl'
        }

        for model_name, filename in models_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                except:
                    self.models[model_name] = None
            else:
                self.models[model_name] = None

    def get_market_data(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """Get market data from local JSON files"""
        history_file = self.market_data_dir / f"{symbol}.json"

        if not history_file.exists():
            return pd.DataFrame()

        try:
            with open(history_file, 'r') as f:
                history = json.load(f)

            # Convert to DataFrame
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            # Filter for requested time period
            cutoff = datetime.now() - timedelta(hours=hours)
            df = df[df.index > cutoff]

            # Add OHLCV data if not present (use price as all values)
            if 'open' not in df.columns:
                df['open'] = df['price']
                df['high'] = df['price']
                df['low'] = df['price']
                df['close'] = df['price']

            if 'volume' not in df.columns:
                df['volume'] = 0

            return df

        except Exception as e:
            self.logger.error(f"Error loading market data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate Relative Strength Index"""
        period = period or self.rsi_period

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = prices.ewm(span=self.macd_slow, adjust=False).mean()

        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd - signal

        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }

    def calculate_moving_averages(self, prices: pd.Series) -> Dict[int, pd.Series]:
        """Calculate multiple moving averages"""
        mas = {}
        for period in self.ma_periods:
            mas[period] = prices.rolling(window=period).mean()
        return mas

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        return {
            'upper': ma + (std * std_dev),
            'middle': ma,
            'lower': ma - (std * std_dev)
        }

    def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        # On-Balance Volume (OBV)
        obv = (df['volume'] * (~df['close'].diff().le(0) * 2 - 1)).cumsum()

        # Volume Rate of Change
        vroc = ((df['volume'] - df['volume'].shift(12)) / df['volume'].shift(12)) * 100

        # Money Flow Index (MFI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']

        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_flow_sum = positive_flow.rolling(14).sum()
        negative_flow_sum = negative_flow.rolling(14).sum()

        money_flow_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_flow_ratio))

        return {
            'obv': obv,
            'vroc': vroc,
            'mfi': mfi
        }

    def identify_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Identify chart patterns using AI"""
        patterns = []

        # Get recent price action
        prices = df['close'].values[-100:]  # Last 100 data points

        if len(prices) < 20:
            return patterns

        # Support and resistance levels
        support = prices.min()
        resistance = prices.max()
        current = prices[-1]

        # Trend detection
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20

        trend = 'uptrend' if sma_20 > sma_50 else 'downtrend'

        # Pattern detection using PHOENIX AI
        pattern_prompt = f"""
        Analyze this price data for trading patterns:
        Current Price: {current}
        Support: {support}
        Resistance: {resistance}
        Trend: {trend}
        Recent Price Movement: {prices[-10:].tolist()}

        Identify any of these patterns:
        - Head and Shoulders
        - Double Top/Bottom
        - Triangle (Ascending/Descending/Symmetrical)
        - Flag/Pennant
        - Wedge
        - Cup and Handle

        Return only the pattern name and confidence level.
        """

        try:
            if self.phoenix:
                ai_response = self.phoenix.process_request(pattern_prompt)
            else:
                ai_response = None

            # Parse AI response
            if ai_response and 'pattern' in ai_response.lower():
                patterns.append({
                    'type': 'ai_detected',
                    'name': ai_response,
                    'confidence': 0.75,
                    'support': support,
                    'resistance': resistance,
                    'trend': trend
                })
        except:
            pass

        # Traditional pattern detection
        # Breakout detection
        if current > resistance * 0.98:
            patterns.append({
                'type': 'breakout',
                'name': 'Resistance Breakout',
                'confidence': 0.8,
                'level': resistance
            })

        if current < support * 1.02:
            patterns.append({
                'type': 'breakdown',
                'name': 'Support Breakdown',
                'confidence': 0.8,
                'level': support
            })

        return patterns

    def assess_risk(self, symbol: str, signal: str, confidence: float, price: float) -> Dict[str, Any]:
        """Assess risk for a potential trade"""
        risk_assessment = {
            'approved': False,
            'risk_score': 0.0,
            'position_size': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'reasons': []
        }

        # Check confidence threshold
        if confidence < self.risk_config['confidence_threshold']:
            risk_assessment['reasons'].append(f"Confidence too low: {confidence:.2%}")
            return risk_assessment

        # Check daily trade limit
        today = datetime.now().date()
        conn = sqlite3.connect(self.decisions_db)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) FROM trading_decisions
            WHERE DATE(timestamp) = DATE(?)
            AND executed = 1
        ''', (today,))

        today_trades = cursor.fetchone()[0]
        conn.close()

        if today_trades >= self.risk_config['max_daily_trades']:
            risk_assessment['reasons'].append("Daily trade limit reached")
            return risk_assessment

        # Calculate position size based on risk level
        base_position = self.risk_config['max_position_size']

        if self.risk_config['risk_level'] == 'conservative':
            position_multiplier = 0.5
        elif self.risk_config['risk_level'] == 'moderate':
            position_multiplier = 1.0
        else:  # aggressive
            position_multiplier = 1.5

        risk_assessment['position_size'] = base_position * position_multiplier * confidence

        # Set stop loss and take profit
        if signal == 'BUY':
            risk_assessment['stop_loss'] = price * (1 - self.risk_config['stop_loss'])
            risk_assessment['take_profit'] = price * (1 + self.risk_config['take_profit'])
        else:  # SELL
            risk_assessment['stop_loss'] = price * (1 + self.risk_config['stop_loss'])
            risk_assessment['take_profit'] = price * (1 - self.risk_config['take_profit'])

        # Calculate risk score (0-100)
        risk_score = confidence * 100

        # Adjust for market conditions
        # This could be enhanced with volatility analysis
        risk_assessment['risk_score'] = risk_score
        risk_assessment['approved'] = risk_score >= 70

        if risk_assessment['approved']:
            risk_assessment['reasons'].append(f"Trade approved with {risk_score:.1f}% risk score")
        else:
            risk_assessment['reasons'].append(f"Risk score too low: {risk_score:.1f}%")

        return risk_assessment

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signal using all indicators and AI"""
        # Get market data
        df = self.get_market_data(symbol, hours=48)

        if df.empty or len(df) < 30:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'explanation': 'Insufficient data for analysis'
            }

        # Calculate all indicators
        prices = df['close']

        # Technical indicators
        rsi = self.calculate_rsi(prices).iloc[-1]
        macd_data = self.calculate_macd(prices)
        macd = macd_data['macd'].iloc[-1]
        macd_signal = macd_data['signal'].iloc[-1]
        macd_histogram = macd_data['histogram'].iloc[-1]

        mas = self.calculate_moving_averages(prices)
        ma_20 = mas[20].iloc[-1] if 20 in mas else prices.iloc[-1]
        ma_50 = mas[50].iloc[-1] if 50 in mas else prices.iloc[-1]

        bollinger = self.calculate_bollinger_bands(prices)
        bb_upper = bollinger['upper'].iloc[-1]
        bb_lower = bollinger['lower'].iloc[-1]

        volume_indicators = self.calculate_volume_indicators(df)
        mfi = volume_indicators['mfi'].iloc[-1] if not pd.isna(volume_indicators['mfi'].iloc[-1]) else 50

        current_price = prices.iloc[-1]

        # Pattern recognition
        patterns = self.identify_patterns(df)

        # Scoring system for signals
        buy_score = 0
        sell_score = 0
        factors = []

        # RSI signals
        if rsi < 30:
            buy_score += 2
            factors.append("RSI oversold")
        elif rsi > 70:
            sell_score += 2
            factors.append("RSI overbought")

        # MACD signals
        if macd > macd_signal and macd_histogram > 0:
            buy_score += 2
            factors.append("MACD bullish crossover")
        elif macd < macd_signal and macd_histogram < 0:
            sell_score += 2
            factors.append("MACD bearish crossover")

        # Moving average signals
        if current_price > ma_20 > ma_50:
            buy_score += 1
            factors.append("Price above moving averages")
        elif current_price < ma_20 < ma_50:
            sell_score += 1
            factors.append("Price below moving averages")

        # Bollinger Band signals
        if current_price < bb_lower:
            buy_score += 1
            factors.append("Price at lower Bollinger Band")
        elif current_price > bb_upper:
            sell_score += 1
            factors.append("Price at upper Bollinger Band")

        # MFI signals
        if mfi < 20:
            buy_score += 1
            factors.append("MFI oversold")
        elif mfi > 80:
            sell_score += 1
            factors.append("MFI overbought")

        # Pattern signals
        for pattern in patterns:
            if 'breakout' in pattern.get('type', ''):
                buy_score += 2
                factors.append(f"Bullish pattern: {pattern['name']}")
            elif 'breakdown' in pattern.get('type', ''):
                sell_score += 2
                factors.append(f"Bearish pattern: {pattern['name']}")

        # AI-enhanced decision making
        ai_prompt = f"""
        Analyze these trading indicators for {symbol}:
        - RSI: {rsi:.2f}
        - MACD: {macd:.4f} (Signal: {macd_signal:.4f})
        - Price: ${current_price:.2f}
        - MA20: ${ma_20:.2f}, MA50: ${ma_50:.2f}
        - MFI: {mfi:.2f}
        - Patterns detected: {patterns}
        - Technical factors: {factors}

        Current scores: Buy={buy_score}, Sell={sell_score}

        Provide a trading decision (BUY/SELL/HOLD) with confidence level (0-1).
        Consider market momentum, risk, and probability of success.
        Format: SIGNAL:confidence:reasoning
        """

        try:
            if self.phoenix:
                ai_response = self.phoenix.process_request(ai_prompt)
            else:
                ai_response = None

            # Parse AI response
            if ai_response and ':' in ai_response:
                parts = ai_response.split(':')
                if len(parts) >= 3:
                    ai_signal = parts[0].strip().upper()
                    ai_confidence = float(parts[1].strip())
                    ai_reasoning = parts[2].strip()

                    # Combine AI decision with technical analysis
                    if ai_signal == 'BUY':
                        buy_score += 3 * ai_confidence
                    elif ai_signal == 'SELL':
                        sell_score += 3 * ai_confidence

                    factors.append(f"AI: {ai_reasoning}")
        except Exception as e:
            self.logger.error(f"AI analysis error: {e}")

        # Final decision
        total_score = max(buy_score, sell_score)

        if buy_score > sell_score and buy_score >= 3:
            signal = 'BUY'
            confidence = min(buy_score / 10, 1.0)
        elif sell_score > buy_score and sell_score >= 3:
            signal = 'SELL'
            confidence = min(sell_score / 10, 1.0)
        else:
            signal = 'HOLD'
            confidence = 0.5

        # Create detailed explanation
        explanation = f"Signal: {signal} | Confidence: {confidence:.1%}\n"
        explanation += f"Technical Factors: {', '.join(factors)}\n"
        explanation += f"Indicators - RSI: {rsi:.1f}, MACD: {macd:.4f}, MFI: {mfi:.1f}"

        return {
            'signal': signal,
            'confidence': confidence,
            'price': current_price,
            'explanation': explanation,
            'indicators': {
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'mfi': mfi,
                'patterns': patterns
            },
            'factors': factors
        }

    def record_decision(self, symbol: str, decision: Dict[str, Any], executed: bool = False) -> int:
        """Record trading decision in database"""
        conn = sqlite3.connect(self.decisions_db)
        cursor = conn.cursor()

        risk_assessment = self.assess_risk(
            symbol,
            decision['signal'],
            decision['confidence'],
            decision['price']
        )

        cursor.execute('''
            INSERT INTO trading_decisions (
                timestamp, symbol, signal, confidence, price,
                indicators, explanation, risk_assessment, executed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            symbol,
            decision['signal'],
            decision['confidence'],
            decision['price'],
            json.dumps(decision['indicators']),
            decision['explanation'],
            json.dumps(risk_assessment),
            executed
        ))

        decision_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return decision_id

    def learn_from_outcome(self, decision_id: int, outcome: str, profit_loss: float):
        """Learn from trading outcome to improve future decisions"""
        conn = sqlite3.connect(self.decisions_db)
        cursor = conn.cursor()

        # Get original decision
        cursor.execute('''
            SELECT symbol, signal, confidence, indicators, explanation
            FROM trading_decisions
            WHERE id = ?
        ''', (decision_id,))

        decision_data = cursor.fetchone()

        if decision_data:
            symbol, signal, confidence, indicators_json, explanation = decision_data
            indicators = json.loads(indicators_json)

            # Create learning note
            learning_note = f"Outcome: {outcome}, P/L: {profit_loss:.2%}\n"

            if profit_loss > 0:
                learning_note += f"Successful {signal} signal. Indicators were favorable.\n"
                # Increase confidence in similar patterns
                if self.memory:
                    self.memory.store_fact(
                        f"trading_success_{symbol}",
                        {
                            'indicators': indicators,
                            'profit': profit_loss,
                            'signal': signal
                        }
                    )
            else:
                learning_note += f"Unsuccessful {signal} signal. Review indicators.\n"
                # Store failure pattern to avoid
                if self.memory:
                    self.memory.store_fact(
                        f"trading_failure_{symbol}",
                        {
                            'indicators': indicators,
                            'loss': profit_loss,
                            'signal': signal
                        }
                    )

            # Update decision record
            cursor.execute('''
                UPDATE trading_decisions
                SET outcome = ?, profit_loss = ?, learning_notes = ?
                WHERE id = ?
            ''', (outcome, profit_loss, learning_note, decision_id))

            conn.commit()

            # Update performance metrics
            self.update_performance_metrics()

        conn.close()

    def update_performance_metrics(self):
        """Update overall performance metrics"""
        conn = sqlite3.connect(self.decisions_db)
        cursor = conn.cursor()

        # Get today's performance
        today = datetime.now().date()

        cursor.execute('''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losses,
                SUM(profit_loss) as total_pnl,
                AVG(CASE WHEN profit_loss > 0 THEN profit_loss ELSE NULL END) as avg_profit,
                AVG(CASE WHEN profit_loss < 0 THEN profit_loss ELSE NULL END) as avg_loss
            FROM trading_decisions
            WHERE DATE(timestamp) = DATE(?)
            AND executed = 1
        ''', (today,))

        metrics = cursor.fetchone()

        if metrics[0] > 0:  # If there are trades today
            win_rate = metrics[1] / metrics[0] if metrics[0] > 0 else 0

            # Calculate Sharpe ratio (simplified)
            cursor.execute('''
                SELECT profit_loss FROM trading_decisions
                WHERE executed = 1
                ORDER BY timestamp DESC
                LIMIT 30
            ''')

            recent_returns = [r[0] for r in cursor.fetchall() if r[0] is not None]

            if recent_returns:
                sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6)
                max_drawdown = min(recent_returns) if recent_returns else 0
            else:
                sharpe = 0
                max_drawdown = 0

            # Insert or update metrics
            cursor.execute('''
                INSERT OR REPLACE INTO performance_metrics (
                    date, total_trades, winning_trades, losing_trades,
                    total_profit_loss, win_rate, avg_profit, avg_loss,
                    sharpe_ratio, max_drawdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                today,
                metrics[0],
                metrics[1] or 0,
                metrics[2] or 0,
                metrics[3] or 0,
                win_rate,
                metrics[4] or 0,
                metrics[5] or 0,
                sharpe,
                max_drawdown
            ))

        conn.commit()
        conn.close()

    def get_all_signals(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Get trading signals for multiple symbols"""
        if not symbols:
            # Get all available symbols from market data
            symbols = [f.stem for f in self.market_data_dir.glob('*.json')]

        signals = {}

        for symbol in symbols:
            try:
                signal = self.generate_signal(symbol)

                # Assess risk
                risk = self.assess_risk(
                    symbol,
                    signal['signal'],
                    signal['confidence'],
                    signal['price']
                )

                signals[symbol] = {
                    **signal,
                    'risk_assessment': risk,
                    'timestamp': datetime.now().isoformat()
                }

                # Record decision if approved
                if risk['approved'] and signal['signal'] != 'HOLD':
                    decision_id = self.record_decision(symbol, signal, executed=False)
                    signals[symbol]['decision_id'] = decision_id

            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
                signals[symbol] = {
                    'signal': 'ERROR',
                    'confidence': 0.0,
                    'explanation': str(e)
                }

        return signals

    def get_portfolio_analysis(self) -> Dict[str, Any]:
        """Get comprehensive portfolio analysis"""
        conn = sqlite3.connect(self.decisions_db)
        cursor = conn.cursor()

        # Get recent performance
        cursor.execute('''
            SELECT * FROM performance_metrics
            ORDER BY date DESC
            LIMIT 30
        ''')

        recent_metrics = cursor.fetchall()

        # Get active positions (simulated)
        cursor.execute('''
            SELECT symbol, signal, price, confidence, timestamp
            FROM trading_decisions
            WHERE executed = 1
            AND outcome IS NULL
            ORDER BY timestamp DESC
        ''')

        active_positions = cursor.fetchall()

        # Get best performing patterns
        cursor.execute('''
            SELECT pattern_name, success_rate, occurrences
            FROM market_patterns
            WHERE success_rate > 0.6
            ORDER BY success_rate DESC
            LIMIT 10
        ''')

        best_patterns = cursor.fetchall()

        conn.close()

        return {
            'recent_performance': recent_metrics,
            'active_positions': active_positions,
            'best_patterns': best_patterns,
            'risk_level': self.risk_config['risk_level'],
            'daily_limit': self.risk_config['max_daily_trades']
        }

    def update_risk_parameters(self, new_params: Dict[str, Any]):
        """Update risk management parameters"""
        for key, value in new_params.items():
            if key in self.risk_config:
                self.risk_config[key] = value

        # Save updated config
        config_file = Path.home() / 'PHOENIX' / 'config' / 'market_risk_config.json'
        config_file.parent.mkdir(exist_ok=True)

        with open(config_file, 'w') as f:
            json.dump(self.risk_config, f, indent=2)

    def autonomous_trading_loop(self):
        """Main autonomous trading loop"""
        self.logger.info("Starting PHOENIX Autonomous Trading System")

        while True:
            try:
                # Get signals for all tracked symbols
                signals = self.get_all_signals()

                # Filter for actionable signals
                actionable = {
                    symbol: data for symbol, data in signals.items()
                    if data.get('risk_assessment', {}).get('approved', False)
                    and data['signal'] != 'HOLD'
                }

                if actionable:
                    self.logger.info(f"Found {len(actionable)} actionable signals")

                    # Here you would interface with trading execution
                    # For now, we just log the decisions
                    for symbol, data in actionable.items():
                        self.logger.info(
                            f"{symbol}: {data['signal']} @ ${data['price']:.2f} "
                            f"(Confidence: {data['confidence']:.1%})"
                        )

                # Update learning models based on recent outcomes
                if self.learning:
                    self.learning.update_model('market_patterns', signals)

                # Sleep for market update interval (30 seconds)
                time.sleep(30)

            except KeyboardInterrupt:
                self.logger.info("Stopping autonomous trading")
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait longer on error


# Integration point for PHOENIX core
if __name__ == "__main__":
    # Initialize market analysis module
    market_module = MarketAnalysisModule()

    # Run autonomous trading
    market_module.autonomous_trading_loop()