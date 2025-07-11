#!/usr/bin/env python3
"""
Enhanced Tools for Domain-Specific Prompting Integration with TimeLLM

This module extends the original tools.py with advanced prompt generation capabilities
that integrate market regime awareness and external data context.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import warnings

# Import original tools
from utils.tools import *  # Keep all original functionality

try:
    from utils.domain_prompting import DynamicPromptManager, MarketContext
    DOMAIN_PROMPTING_AVAILABLE = True
except ImportError:
    DOMAIN_PROMPTING_AVAILABLE = False
    warnings.warn("Domain prompting not available - falling back to basic prompts")


def load_enhanced_content(args):
    """
    Enhanced content loader that supports dynamic prompt generation
    based on market conditions and external data context.
    """
    # Fallback to original content loading if dynamic prompting is not available
    if not DOMAIN_PROMPTING_AVAILABLE or not getattr(args, 'enable_dynamic_prompting', False):
        return load_content(args)
    
    # Determine base dataset type
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    
    # Check if enhanced prompt templates exist
    enhanced_prompt_path = f'./dataset/prompt_bank/{file}.txt'
    if not os.path.exists(enhanced_prompt_path):
        # Fall back to original content loading
        return load_content(args)
    
    # Load base prompt template
    with open(enhanced_prompt_path, 'r') as f:
        base_content = f.read()
    
    # If no dynamic prompting requested, return base content
    if not hasattr(args, 'enable_regime_aware_prompts') or not args.enable_regime_aware_prompts:
        return base_content
    
    # Generate dynamic content based on current market conditions
    try:
        dynamic_manager = DynamicPromptManager()
        
        # Extract market context from current data if available
        market_context = extract_market_context_from_args(args)
        
        # Generate dynamic prompt enhancement
        dynamic_enhancement = generate_dynamic_prompt_enhancement(
            market_context, 
            args.pred_len,
            getattr(args, 'trading_style', 'swing_trading')
        )
        
        # Combine base content with dynamic enhancement
        enhanced_content = f"{base_content}\n\n{dynamic_enhancement}"
        
        return enhanced_content
        
    except Exception as e:
        warnings.warn(f"Dynamic prompt generation failed: {e}. Using base content.")
        return base_content


def extract_market_context_from_args(args) -> MarketContext:
    """Extract market context from training arguments and data"""
    context = MarketContext()
    
    # Set basic context from args
    if hasattr(args, 'trading_style'):
        context.trading_style = args.trading_style
    
    if hasattr(args, 'prediction_horizon'):
        context.prediction_horizon = args.prediction_horizon
    else:
        context.prediction_horizon = getattr(args, 'pred_len', 24)
    
    # Extract temporal context
    current_time = pd.Timestamp.now()
    context.time_of_day = get_trading_session(current_time.hour)
    context.day_of_week = get_day_classification(current_time.dayofweek)
    
    # Try to extract market regime from recent data if available
    if hasattr(args, 'data_path') and hasattr(args, 'root_path'):
        try:
            data_file = os.path.join(args.root_path, args.data_path)
            if os.path.exists(data_file):
                # Load recent data for regime detection
                df = pd.read_csv(data_file)
                if 'close' in df.columns and len(df) >= 30:
                    recent_prices = df['close'].tail(30).values
                    context.regime = detect_regime_from_prices(recent_prices)
                    context.volatility_level = detect_volatility_level(recent_prices)
                    context.trend_direction = detect_trend_direction(recent_prices)
        except Exception:
            pass  # Use default context if data loading fails
    
    return context


def get_trading_session(hour: int) -> str:
    """Classify trading session based on hour"""
    if 0 <= hour < 8:
        return "asian_session"
    elif 8 <= hour < 16:
        return "european_session"
    else:
        return "us_session"


def get_day_classification(dayofweek: int) -> str:
    """Classify day of week for trading context"""
    if dayofweek >= 5:  # Saturday, Sunday
        return "weekend"
    elif dayofweek == 0:  # Monday
        return "monday"
    elif dayofweek == 4:  # Friday
        return "friday"
    else:
        return "weekday"


def detect_regime_from_prices(prices: np.ndarray, window: int = 20) -> str:
    """Simple regime detection from price array"""
    if len(prices) < window:
        return "unknown"
    
    returns = np.diff(prices) / prices[:-1]
    recent_returns = returns[-window:]
    
    avg_return = np.mean(recent_returns)
    volatility = np.std(recent_returns)
    
    # Classify trend
    if avg_return > 0.001:
        trend = "bull"
    elif avg_return < -0.001:
        trend = "bear"
    else:
        trend = "sideways"
    
    # Classify volatility
    if volatility < 0.015:
        vol_level = "low_vol"
    elif volatility > 0.04:
        vol_level = "high_vol"
    else:
        vol_level = "med_vol"
    
    return f"{trend}_{vol_level}"


def detect_volatility_level(prices: np.ndarray) -> str:
    """Detect volatility level from prices"""
    if len(prices) < 10:
        return "medium"
    
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns)
    
    if volatility < 0.015:
        return "low"
    elif volatility > 0.04:
        return "high"
    else:
        return "medium"


def detect_trend_direction(prices: np.ndarray, window: int = 10) -> str:
    """Detect trend direction from prices"""
    if len(prices) < window:
        return "neutral"
    
    recent_change = (prices[-1] - prices[-window]) / prices[-window]
    
    if recent_change > 0.02:  # 2% increase
        return "upward"
    elif recent_change < -0.02:  # 2% decrease
        return "downward"
    else:
        return "neutral"


def generate_dynamic_prompt_enhancement(context: MarketContext, 
                                      prediction_horizon: int,
                                      trading_style: str = "swing_trading") -> str:
    """Generate dynamic prompt enhancement based on market context"""
    
    enhancements = []
    
    # Regime-specific enhancement
    if context.regime != "unknown":
        regime_desc = get_regime_description(context.regime)
        enhancements.append(f"Current Market Regime: {regime_desc}")
    
    # Volatility-specific enhancement
    vol_desc = get_volatility_description(context.volatility_level)
    enhancements.append(f"Volatility Environment: {vol_desc}")
    
    # Temporal enhancement
    temporal_desc = get_temporal_description(context.time_of_day, context.day_of_week)
    enhancements.append(f"Temporal Context: {temporal_desc}")
    
    # Trading style enhancement
    trading_desc = get_trading_style_description(trading_style, prediction_horizon)
    enhancements.append(f"Trading Focus: {trading_desc}")
    
    # Prediction guidance
    prediction_guidance = get_prediction_guidance(context, prediction_horizon)
    enhancements.append(f"Prediction Approach: {prediction_guidance}")
    
    return "\n\n".join(enhancements)


def get_regime_description(regime: str) -> str:
    """Get description for market regime"""
    regime_descriptions = {
        "bull_low_vol": "Steadily rising market with controlled volatility, suggesting institutional accumulation and sustainable upward momentum.",
        "bull_med_vol": "Rising market with healthy volatility, indicating active price discovery and broad participation.",
        "bull_high_vol": "Rapidly rising market with elevated volatility, suggesting strong momentum but potential overextension risks.",
        "bear_low_vol": "Declining market with contained volatility, indicating controlled selling and potential accumulation areas.",
        "bear_med_vol": "Falling market with moderate volatility, showing active distribution and selling pressure.",
        "bear_high_vol": "Rapidly declining market with high volatility, suggesting panic selling and potential capitulation.",
        "sideways_low_vol": "Range-bound market with low volatility, indicating accumulation/distribution and reduced interest.",
        "sideways_med_vol": "Consolidating market with moderate volatility, suggesting indecision and potential breakout preparation.",
        "sideways_high_vol": "Choppy market with high volatility, indicating conflicting forces and directional uncertainty."
    }
    return regime_descriptions.get(regime, "Market regime requires analysis to determine appropriate trading approach.")


def get_volatility_description(volatility_level: str) -> str:
    """Get description for volatility level"""
    volatility_descriptions = {
        "low": "Low volatility environment supports precise predictions but requires monitoring for potential breakout scenarios.",
        "medium": "Moderate volatility provides balanced prediction conditions with normal market dynamics.",
        "high": "High volatility environment requires wider confidence intervals and increased focus on risk management."
    }
    return volatility_descriptions.get(volatility_level, "Standard volatility conditions apply.")


def get_temporal_description(time_of_day: str, day_of_week: str) -> str:
    """Get temporal context description"""
    time_descriptions = {
        "asian_session": "Asian trading session typically features lower volume and range-bound behavior",
        "european_session": "European trading session often brings increased volatility and trend development",
        "us_session": "US trading session frequently features high volume and momentum-driven moves"
    }
    
    day_descriptions = {
        "weekend": "Weekend trading shows reduced volume and increased volatility potential",
        "monday": "Monday trading often features gap movements and trend continuation patterns",
        "friday": "Friday trading may show position squaring and momentum reduction",
        "weekday": "Regular weekday trading with typical institutional participation"
    }
    
    time_desc = time_descriptions.get(time_of_day, "Current trading session")
    day_desc = day_descriptions.get(day_of_week, "regular market conditions")
    
    return f"{time_desc} during {day_desc}."


def get_trading_style_description(trading_style: str, prediction_horizon: int) -> str:
    """Get trading style specific description"""
    style_descriptions = {
        "scalping": f"Ultra-short-term prediction over {prediction_horizon} steps focusing on market microstructure and momentum for rapid entry/exit opportunities.",
        "day_trading": f"Intraday prediction over {prediction_horizon} steps targeting momentum patterns and intraday trends for same-session position management.",
        "swing_trading": f"Multi-period prediction over {prediction_horizon} steps capturing intermediate trends and regime patterns for position holds over multiple sessions.",
        "position_trading": f"Long-term prediction over {prediction_horizon} steps identifying major trend shifts and cycle patterns for strategic allocation decisions."
    }
    return style_descriptions.get(trading_style, f"Medium-term prediction over {prediction_horizon} steps for balanced risk-reward opportunities.")


def get_prediction_guidance(context: MarketContext, prediction_horizon: int) -> str:
    """Generate specific prediction guidance based on context"""
    guidance_parts = []
    
    # Horizon-specific guidance
    if prediction_horizon <= 6:
        guidance_parts.append("Focus on momentum and microstructure signals for short-term directional bias")
    elif prediction_horizon <= 24:
        guidance_parts.append("Balance technical patterns with regime characteristics for medium-term trends")
    else:
        guidance_parts.append("Incorporate regime changes and external factors for longer-term directional analysis")
    
    # Regime-specific guidance
    if "bull" in context.regime:
        guidance_parts.append("emphasize upside momentum and pullback entry opportunities")
    elif "bear" in context.regime:
        guidance_parts.append("focus on downside continuation and bounce-selling opportunities")
    else:
        guidance_parts.append("prepare for breakout/breakdown scenarios with directional clarity")
    
    # Volatility-specific guidance
    if context.volatility_level == "high":
        guidance_parts.append("while maintaining wide risk parameters for elevated volatility")
    elif context.volatility_level == "low":
        guidance_parts.append("with precision targeting due to reduced volatility environment")
    
    return "; ".join(guidance_parts) + "."


# Enhanced argument parser for domain-specific prompting
def add_domain_prompting_args(parser):
    """Add domain-specific prompting arguments to argument parser"""
    parser.add_argument('--enable_dynamic_prompting', action='store_true', 
                       help='Enable dynamic prompt generation based on market conditions')
    parser.add_argument('--enable_regime_aware_prompts', action='store_true',
                       help='Enable market regime-aware prompt enhancement')
    parser.add_argument('--trading_style', type=str, default='swing_trading',
                       choices=['scalping', 'day_trading', 'swing_trading', 'position_trading'],
                       help='Trading style for prompt optimization')
    parser.add_argument('--prompt_update_frequency', type=str, default='per_batch',
                       choices=['per_sample', 'per_batch', 'per_epoch'],
                       help='Frequency of dynamic prompt updates')
    parser.add_argument('--external_prompt_context', action='store_true',
                       help='Include external data context in prompts')
    
    return parser


# Backward compatibility - ensure original load_content still works
def load_content_with_fallback(args):
    """Load content with fallback to original implementation"""
    try:
        return load_enhanced_content(args)
    except Exception as e:
        warnings.warn(f"Enhanced content loading failed: {e}. Using original content loader.")
        return load_content(args)


# Convenience function for testing
def test_dynamic_prompting():
    """Test dynamic prompting functionality"""
    print("Testing Dynamic Prompting System")
    print("=" * 50)
    
    # Create sample context
    context = MarketContext()
    context.regime = "bull_med_vol"
    context.volatility_level = "medium"
    context.time_of_day = "us_session"
    context.day_of_week = "weekday"
    
    # Generate enhancement
    enhancement = generate_dynamic_prompt_enhancement(context, 24, "swing_trading")
    
    print("Sample Dynamic Prompt Enhancement:")
    print("-" * 50)
    print(enhancement)
    print("-" * 50)
    
    return enhancement


if __name__ == "__main__":
    test_dynamic_prompting()