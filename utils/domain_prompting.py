#!/usr/bin/env python3
"""
Domain-Specific Prompting System for Cryptocurrency Time Series Prediction

This module provides sophisticated, market-aware prompt generation for TimeLLM
that incorporates market regime awareness, external data context, and trading insights.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
import json


@dataclass
class MarketContext:
    """Market context information for prompt generation"""
    regime: str = "unknown"  # bull_low_vol, bear_high_vol, etc.
    volatility_level: str = "medium"  # low, medium, high
    trend_direction: str = "neutral"  # upward, downward, neutral
    price_momentum: float = 0.0
    volume_trend: str = "stable"  # increasing, decreasing, stable
    
    # External data context
    sentiment_score: float = 0.5  # 0-1
    fear_greed_index: int = 50    # 0-100
    macro_environment: str = "neutral"  # bullish, bearish, neutral
    onchain_strength: str = "moderate"  # weak, moderate, strong
    
    # Time context
    time_of_day: str = "market_hours"
    day_of_week: str = "weekday"
    market_session: str = "regular"


class DomainPromptGenerator:
    """Advanced prompt generator for cryptocurrency time series prediction"""
    
    def __init__(self):
        self.base_descriptions = self._load_base_descriptions()
        self.regime_contexts = self._load_regime_contexts()
        self.trading_contexts = self._load_trading_contexts()
        self.external_data_contexts = self._load_external_data_contexts()
    
    def _load_base_descriptions(self) -> Dict[str, str]:
        """Load base dataset descriptions for different contexts"""
        return {
            "basic": """The Binance Bitcoin Hourly Candlesticks (BTC-H) dataset captures granular financial data from the Binance.us cryptocurrency exchange. It spans nearly four years, from September 2019 to July 2023, with hourly-level resolution. Each record contains second-to-second updates for key trading indicators: Open, High, Low, Close (OHLC) prices and traded volume in USDT. Timestamps are stored in Unix time format. Inactive periods (with no trading activity) are represented with NaN values, while missing timestamps may reflect exchange/API downtime or data collection limitations. The dataset has been carefully deduplicated and validated, and is updated nightly to ensure consistency and completeness.""",
            
            "enhanced": """The Enhanced Binance Bitcoin Candlesticks dataset combines comprehensive OHLCV price data with 68+ advanced technical indicators and external market signals. This multi-dimensional dataset spans September 2019 to July 2023 with hourly resolution, featuring sophisticated momentum indicators (RSI, MACD, Stochastic), volatility measures (Bollinger Bands, ATR, realized volatility), market microstructure signals (VWAP, money flow), and temporal patterns. Enhanced with external data including social sentiment, macroeconomic indicators, and Bitcoin network fundamentals for comprehensive market context.""",
            
            "external": """The Comprehensive Bitcoin Market Intelligence dataset integrates price data with multi-source external signals for holistic market analysis. Beyond traditional OHLCV data, it incorporates real-time social sentiment from Twitter and Reddit, macroeconomic indicators (Fed rates, Treasury yields, stock indices), and Bitcoin network fundamentals (hash rate, active addresses, exchange flows). This 100+ feature dataset enables regime-aware predictions by capturing the complex interplay between price action, market psychology, economic conditions, and blockchain network health.""",
            
            "regime_aware": """The Market Regime-Aware Bitcoin dataset provides context-sensitive time series data that adapts to changing market conditions. The dataset automatically classifies market regimes into nine categories based on volatility and trend characteristics (bull/bear/sideways × low/medium/high volatility), enabling regime-specific pattern recognition. Each data point includes regime classification, volatility metrics, trend strength, and regime-appropriate feature weighting for adaptive forecasting across different market environments."""
        }
    
    def _load_regime_contexts(self) -> Dict[str, str]:
        """Load market regime-specific contexts"""
        return {
            "bull_low_vol": "steadily rising market with low volatility, indicating strong institutional accumulation and stable upward momentum",
            "bull_med_vol": "rising market with moderate volatility, suggesting healthy price discovery and active participation",
            "bull_high_vol": "rapidly rising market with high volatility, indicating FOMO-driven buying and potential overextension",
            
            "bear_low_vol": "declining market with low volatility, suggesting controlled selling and potential capitulation phase",
            "bear_med_vol": "falling market with moderate volatility, indicating active selling pressure and distribution",
            "bear_high_vol": "rapidly falling market with high volatility, suggesting panic selling and potential washout",
            
            "sideways_low_vol": "range-bound market with low volatility, indicating accumulation/distribution and low interest",
            "sideways_med_vol": "consolidating market with moderate volatility, suggesting indecision and potential breakout preparation",
            "sideways_high_vol": "choppy market with high volatility, indicating conflicting forces and uncertainty",
            
            "unknown": "market conditions require analysis of current patterns and context to determine appropriate regime classification"
        }
    
    def _load_trading_contexts(self) -> Dict[str, str]:
        """Load trading-focused contexts"""
        return {
            "scalping": "ultra-short-term prediction focusing on minute-level price movements and market microstructure for high-frequency trading opportunities",
            "day_trading": "intraday prediction targeting hourly price swings and momentum patterns for same-day position management",
            "swing_trading": "multi-day prediction capturing intermediate-term trends and regime changes for position trading over days to weeks",
            "position_trading": "long-term prediction identifying major trend shifts and cycle patterns for strategic portfolio allocation",
            
            "volatility_trading": "prediction optimized for volatility breakouts and mean reversion patterns in high-volatility environments",
            "momentum_trading": "prediction focused on trend continuation and momentum signals for directional strategies",
            "mean_reversion": "prediction targeting oversold/overbought conditions and range-bound price action for contrarian strategies",
            "breakout_trading": "prediction specialized in identifying accumulation phases and potential breakout directions"
        }
    
    def _load_external_data_contexts(self) -> Dict[str, str]:
        """Load external data-specific contexts"""
        return {
            "sentiment_driven": "incorporating social media sentiment, fear/greed index, and news sentiment to capture market psychology and crowd behavior",
            "macro_influenced": "integrating macroeconomic indicators including interest rates, inflation data, and stock market performance to understand broader financial market context",
            "onchain_focused": "utilizing Bitcoin network fundamentals such as hash rate, active addresses, and exchange flows to assess underlying network health and adoption trends",
            "multi_source": "combining sentiment analysis, macroeconomic indicators, and on-chain metrics for comprehensive market intelligence and multi-dimensional prediction",
            
            "correlation_aware": "leveraging cross-asset correlations and market relationships to understand Bitcoin's position within the broader financial ecosystem",
            "event_driven": "incorporating scheduled economic events, Fed meetings, earnings releases, and crypto-specific catalysts that may impact price action"
        }
    
    def generate_enhanced_prompt(self, 
                                market_context: MarketContext,
                                dataset_type: str = "enhanced",
                                trading_style: str = "swing_trading",
                                prediction_horizon: int = 24,
                                current_stats: Dict[str, Any] = None) -> str:
        """Generate sophisticated domain-specific prompt"""
        
        # Base dataset description
        base_desc = self.base_descriptions.get(dataset_type, self.base_descriptions["basic"])
        
        # Market regime context
        regime_context = self.regime_contexts.get(market_context.regime, self.regime_contexts["unknown"])
        
        # Trading context
        trading_context = self.trading_contexts.get(trading_style, self.trading_contexts["swing_trading"])
        
        # External data context
        external_context = self._get_external_data_context(market_context)
        
        # Market psychology context
        psychology_context = self._get_market_psychology_context(market_context)
        
        # Temporal context
        temporal_context = self._get_temporal_context(market_context)
        
        # Risk context
        risk_context = self._get_risk_context(market_context, prediction_horizon)
        
        # Construct enhanced prompt
        prompt = f"""Dataset description: {base_desc}

Market Regime Analysis: Current market conditions indicate a {regime_context}. This regime suggests specific patterns and behaviors that should inform prediction methodology.

Trading Context: {trading_context}

External Market Intelligence: {external_context}

Market Psychology: {psychology_context}

Temporal Context: {temporal_context}

Risk Assessment: {risk_context}

Prediction Objective: Generate precise {prediction_horizon}-step forecasts that account for current regime characteristics, external market forces, and risk-adjusted expectations. Focus on regime-appropriate patterns while considering multi-timeframe confluence and external data signals."""
        
        return prompt
    
    def _get_external_data_context(self, market_context: MarketContext) -> str:
        """Generate external data context based on current market conditions"""
        sentiment_desc = self._describe_sentiment_level(market_context.sentiment_score)
        fear_greed_desc = self._describe_fear_greed_level(market_context.fear_greed_index)
        macro_desc = f"Macroeconomic environment is {market_context.macro_environment}"
        onchain_desc = f"On-chain fundamentals show {market_context.onchain_strength} network health"
        
        return f"{sentiment_desc} {fear_greed_desc} {macro_desc}, while {onchain_desc}. These external factors provide crucial context for understanding current price dynamics beyond pure technical analysis."
    
    def _describe_sentiment_level(self, sentiment_score: float) -> str:
        """Describe sentiment level based on score"""
        if sentiment_score < 0.2:
            return "Market sentiment is extremely bearish with widespread pessimism and fear."
        elif sentiment_score < 0.4:
            return "Market sentiment is bearish with negative social media activity and skepticism."
        elif sentiment_score < 0.6:
            return "Market sentiment is neutral with balanced optimism and caution."
        elif sentiment_score < 0.8:
            return "Market sentiment is bullish with positive social engagement and optimism."
        else:
            return "Market sentiment is extremely bullish with euphoric social activity and FOMO."
    
    def _describe_fear_greed_level(self, fear_greed_index: int) -> str:
        """Describe fear/greed level based on index"""
        if fear_greed_index < 25:
            return "The Fear & Greed Index shows extreme fear, often marking potential market bottoms."
        elif fear_greed_index < 45:
            return "The Fear & Greed Index indicates fear, suggesting cautious market sentiment."
        elif fear_greed_index < 55:
            return "The Fear & Greed Index shows neutral sentiment with balanced emotions."
        elif fear_greed_index < 75:
            return "The Fear & Greed Index indicates greed, suggesting optimistic market conditions."
        else:
            return "The Fear & Greed Index shows extreme greed, often signaling potential market tops."
    
    def _get_market_psychology_context(self, market_context: MarketContext) -> str:
        """Generate market psychology context"""
        base_psychology = f"Current market psychology reflects {market_context.regime} conditions"
        
        if "bull" in market_context.regime:
            psychology_details = "with increasing confidence, FOMO potential, and upward momentum bias"
        elif "bear" in market_context.regime:
            psychology_details = "with declining confidence, fear dominance, and downward pressure"
        else:
            psychology_details = "with indecision, conflicting signals, and directional uncertainty"
        
        volatility_psychology = self._get_volatility_psychology(market_context.volatility_level)
        
        return f"{base_psychology} {psychology_details}. {volatility_psychology}"
    
    def _get_volatility_psychology(self, volatility_level: str) -> str:
        """Get psychology context based on volatility"""
        if volatility_level == "low":
            return "Low volatility suggests institutional participation and controlled price action."
        elif volatility_level == "high":
            return "High volatility indicates emotional trading and potential trend acceleration or exhaustion."
        else:
            return "Moderate volatility reflects normal market dynamics with balanced participation."
    
    def _get_temporal_context(self, market_context: MarketContext) -> str:
        """Generate temporal context for predictions"""
        time_factors = []
        
        if market_context.time_of_day == "asian_session":
            time_factors.append("Asian trading session typically shows lower volume and range-bound behavior")
        elif market_context.time_of_day == "european_session":
            time_factors.append("European trading session often brings increased volatility and trend development")
        elif market_context.time_of_day == "us_session":
            time_factors.append("US trading session frequently features high volume and momentum moves")
        else:
            time_factors.append("Current time period shows typical trading characteristics")
        
        if market_context.day_of_week == "weekend":
            time_factors.append("Weekend trading typically exhibits lower volume and increased volatility")
        elif market_context.day_of_week == "monday":
            time_factors.append("Monday opening often features gap movements and trend continuation")
        elif market_context.day_of_week == "friday":
            time_factors.append("Friday closing may show position squaring and reduced momentum")
        
        return ". ".join(time_factors) + "."
    
    def _get_risk_context(self, market_context: MarketContext, prediction_horizon: int) -> str:
        """Generate risk assessment context"""
        risk_factors = []
        
        # Volatility risk
        if market_context.volatility_level == "high":
            risk_factors.append("elevated volatility increases prediction uncertainty and requires wider confidence intervals")
        elif market_context.volatility_level == "low":
            risk_factors.append("low volatility environment supports higher prediction confidence but watch for breakout potential")
        
        # Regime risk
        if "high_vol" in market_context.regime:
            risk_factors.append("current regime shows elevated risk of sudden directional changes")
        
        # Time horizon risk
        if prediction_horizon <= 6:
            risk_factors.append("short-term predictions should focus on momentum and microstructure signals")
        elif prediction_horizon <= 24:
            risk_factors.append("medium-term predictions should balance technical patterns with regime characteristics")
        else:
            risk_factors.append("longer-term predictions must incorporate regime changes and external factor evolution")
        
        return "Risk considerations: " + "; ".join(risk_factors) + "."
    
    def detect_market_regime(self, 
                           price_data: np.ndarray, 
                           volume_data: np.ndarray = None,
                           window: int = 30) -> str:
        """Detect current market regime from price data"""
        if len(price_data) < window:
            return "unknown"
        
        # Calculate returns and volatility
        returns = np.diff(price_data) / price_data[:-1]
        recent_returns = returns[-window:]
        
        avg_return = np.mean(recent_returns)
        volatility = np.std(recent_returns)
        
        # Classify trend
        if avg_return > 0.001:  # 0.1% threshold
            trend = "bull"
        elif avg_return < -0.001:
            trend = "bear"
        else:
            trend = "sideways"
        
        # Classify volatility
        if volatility < 0.015:  # 1.5% threshold
            vol_level = "low_vol"
        elif volatility > 0.04:  # 4% threshold
            vol_level = "high_vol"
        else:
            vol_level = "med_vol"
        
        return f"{trend}_{vol_level}"
    
    def extract_market_context_from_data(self, 
                                        data_row: pd.Series,
                                        external_features: Dict[str, float] = None) -> MarketContext:
        """Extract market context from current data"""
        context = MarketContext()
        
        # Extract basic market info
        if 'close' in data_row:
            # Calculate simple momentum
            if 'price_lag_1' in data_row:
                context.price_momentum = (data_row['close'] - data_row['price_lag_1']) / data_row['price_lag_1']
        
        # Extract external data context if available
        if external_features:
            context.sentiment_score = external_features.get('sentiment_score', 0.5)
            context.fear_greed_index = int(external_features.get('fear_greed_index', 50))
            
            # Classify macro environment
            fed_rate = external_features.get('fed_funds_rate', 5.0)
            sp500_change = external_features.get('risk_on_sentiment', 0.0)
            
            if sp500_change > 2 and fed_rate < 3:
                context.macro_environment = "bullish"
            elif sp500_change < -2 or fed_rate > 6:
                context.macro_environment = "bearish"
            else:
                context.macro_environment = "neutral"
            
            # Classify on-chain strength
            hash_rate_momentum = external_features.get('hash_rate_momentum', 0)
            net_exchange_flow = external_features.get('net_exchange_flow', 0)
            
            if hash_rate_momentum > 0 and net_exchange_flow < 0:  # Hash rate up, outflows
                context.onchain_strength = "strong"
            elif hash_rate_momentum < 0 and net_exchange_flow > 0:  # Hash rate down, inflows
                context.onchain_strength = "weak"
            else:
                context.onchain_strength = "moderate"
        
        # Extract temporal context
        if 'hour' in data_row:
            hour = int(data_row['hour'])
            if 0 <= hour < 8:
                context.time_of_day = "asian_session"
            elif 8 <= hour < 16:
                context.time_of_day = "european_session"
            else:
                context.time_of_day = "us_session"
        
        if 'day_of_week' in data_row:
            dow = int(data_row['day_of_week'])
            if dow >= 5:  # Saturday, Sunday
                context.day_of_week = "weekend"
            elif dow == 0:  # Monday
                context.day_of_week = "monday"
            elif dow == 4:  # Friday
                context.day_of_week = "friday"
            else:
                context.day_of_week = "weekday"
        
        return context


class DynamicPromptManager:
    """Manages dynamic prompt generation based on real-time market conditions"""
    
    def __init__(self, config_file: str = None):
        self.prompt_generator = DomainPromptGenerator()
        self.config = self._load_config(config_file)
        self.prompt_cache = {}
        self.performance_tracker = {}
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load prompting configuration"""
        default_config = {
            "enable_regime_awareness": True,
            "enable_external_data_context": True,
            "enable_temporal_context": True,
            "enable_risk_context": True,
            "cache_prompts": True,
            "update_frequency": "per_batch",  # per_sample, per_batch, per_epoch
            "trading_style": "swing_trading",
            "dataset_type": "enhanced"
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def generate_dynamic_prompt(self,
                               price_data: np.ndarray,
                               current_data: pd.Series = None,
                               external_features: Dict[str, float] = None,
                               prediction_horizon: int = 24) -> str:
        """Generate dynamic prompt based on current market conditions"""
        
        # Detect market regime
        regime = self.prompt_generator.detect_market_regime(price_data)
        
        # Extract market context
        if current_data is not None:
            market_context = self.prompt_generator.extract_market_context_from_data(
                current_data, external_features
            )
        else:
            market_context = MarketContext()
        
        market_context.regime = regime
        
        # Generate cache key
        cache_key = self._generate_cache_key(market_context, prediction_horizon)
        
        # Check cache
        if self.config["cache_prompts"] and cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]
        
        # Generate new prompt
        prompt = self.prompt_generator.generate_enhanced_prompt(
            market_context=market_context,
            dataset_type=self.config["dataset_type"],
            trading_style=self.config["trading_style"],
            prediction_horizon=prediction_horizon
        )
        
        # Cache prompt
        if self.config["cache_prompts"]:
            self.prompt_cache[cache_key] = prompt
        
        return prompt
    
    def _generate_cache_key(self, market_context: MarketContext, prediction_horizon: int) -> str:
        """Generate cache key for prompt caching"""
        key_components = [
            market_context.regime,
            market_context.volatility_level,
            market_context.macro_environment,
            market_context.onchain_strength,
            str(prediction_horizon)
        ]
        return "_".join(key_components)
    
    def update_prompt_performance(self, prompt_key: str, performance_metrics: Dict[str, float]):
        """Track prompt performance for optimization"""
        if prompt_key not in self.performance_tracker:
            self.performance_tracker[prompt_key] = []
        
        self.performance_tracker[prompt_key].append(performance_metrics)
    
    def get_best_performing_prompts(self, metric: str = "mse", top_k: int = 5) -> List[Tuple[str, float]]:
        """Get best performing prompt configurations"""
        prompt_scores = []
        
        for prompt_key, performances in self.performance_tracker.items():
            if performances:
                avg_score = np.mean([p.get(metric, float('inf')) for p in performances])
                prompt_scores.append((prompt_key, avg_score))
        
        # Sort by score (lower is better for error metrics)
        prompt_scores.sort(key=lambda x: x[1])
        return prompt_scores[:top_k]
    
    def save_prompt_templates(self, output_dir: str = "./dataset/prompt_bank/enhanced/"):
        """Save enhanced prompt templates"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save templates for different scenarios
        scenarios = [
            ("bull_market", MarketContext(regime="bull_med_vol", sentiment_score=0.7, fear_greed_index=70)),
            ("bear_market", MarketContext(regime="bear_high_vol", sentiment_score=0.3, fear_greed_index=20)),
            ("sideways_market", MarketContext(regime="sideways_low_vol", sentiment_score=0.5, fear_greed_index=50)),
            ("high_volatility", MarketContext(regime="bull_high_vol", sentiment_score=0.8, fear_greed_index=85)),
            ("low_volatility", MarketContext(regime="sideways_low_vol", sentiment_score=0.5, fear_greed_index=45))
        ]
        
        for scenario_name, context in scenarios:
            prompt = self.prompt_generator.generate_enhanced_prompt(context)
            
            with open(f"{output_dir}/CRYPTEX_{scenario_name}.txt", 'w') as f:
                f.write(prompt)
        
        print(f"Enhanced prompt templates saved to {output_dir}")


# Convenience functions for easy integration
def generate_crypto_prompt(price_data: np.ndarray, 
                          external_data: Dict[str, float] = None,
                          trading_style: str = "swing_trading") -> str:
    """Quick function to generate crypto-specific prompt"""
    manager = DynamicPromptManager()
    return manager.generate_dynamic_prompt(price_data, external_features=external_data)


def create_regime_aware_prompts(output_dir: str = "./dataset/prompt_bank/enhanced/"):
    """Create and save regime-aware prompt templates"""
    manager = DynamicPromptManager()
    manager.save_prompt_templates(output_dir)


if __name__ == "__main__":
    # Example usage
    print("Domain-Specific Prompting System for Cryptocurrency Prediction")
    
    # Create enhanced prompt templates
    create_regime_aware_prompts()
    
    # Example dynamic prompt generation
    sample_price_data = np.random.randn(100) * 0.02 + 1  # Sample price data
    sample_price_data = np.cumprod(sample_price_data) * 50000  # Make it look like Bitcoin prices
    
    prompt = generate_crypto_prompt(sample_price_data)
    print("\nSample Dynamic Prompt:")
    print("=" * 80)
    print(prompt)