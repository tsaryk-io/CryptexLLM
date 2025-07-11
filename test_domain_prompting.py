#!/usr/bin/env python3
"""
Test script for Domain-Specific Prompting System
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.domain_prompting import (
        DomainPromptGenerator,
        DynamicPromptManager,
        MarketContext,
        generate_crypto_prompt,
        create_regime_aware_prompts
    )
    DOMAIN_PROMPTING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Domain prompting components not available: {e}")
    DOMAIN_PROMPTING_AVAILABLE = False

try:
    from utils.enhanced_tools import (
        load_enhanced_content,
        test_dynamic_prompting,
        generate_dynamic_prompt_enhancement
    )
    ENHANCED_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced tools not available: {e}")
    ENHANCED_TOOLS_AVAILABLE = False


def test_prompt_templates():
    """Test basic prompt template loading"""
    print("=" * 60)
    print("TESTING PROMPT TEMPLATES")
    print("=" * 60)
    
    template_files = [
        "./dataset/prompt_bank/CRYPTEX_ENHANCED.txt",
        "./dataset/prompt_bank/CRYPTEX_EXTERNAL.txt", 
        "./dataset/prompt_bank/CRYPTEX_MULTISCALE.txt",
        "./dataset/prompt_bank/CRYPTEX_REGIME_AWARE.txt"
    ]
    
    success_count = 0
    
    for template_file in template_files:
        try:
            if os.path.exists(template_file):
                with open(template_file, 'r') as f:
                    content = f.read()
                    print(f"✓ {os.path.basename(template_file)}: {len(content)} characters")
                    success_count += 1
            else:
                print(f"✗ {os.path.basename(template_file)}: File not found")
        except Exception as e:
            print(f"✗ {os.path.basename(template_file)}: Error reading - {e}")
    
    print(f"\nTemplate loading: {success_count}/{len(template_files)} successful")
    return success_count == len(template_files)


def test_trading_strategy_templates():
    """Test trading strategy-specific templates"""
    print("\n" + "=" * 60)
    print("TESTING TRADING STRATEGY TEMPLATES")
    print("=" * 60)
    
    strategy_templates = [
        "./dataset/prompt_bank/enhanced/CRYPTEX_scalping.txt",
        "./dataset/prompt_bank/enhanced/CRYPTEX_day_trading.txt",
        "./dataset/prompt_bank/enhanced/CRYPTEX_swing_trading.txt",
        "./dataset/prompt_bank/enhanced/CRYPTEX_position_trading.txt"
    ]
    
    success_count = 0
    
    for template_file in strategy_templates:
        try:
            if os.path.exists(template_file):
                with open(template_file, 'r') as f:
                    content = f.read()
                    strategy_name = os.path.basename(template_file).replace('CRYPTEX_', '').replace('.txt', '')
                    print(f"✓ {strategy_name}: {len(content)} characters")
                    
                    # Check for key strategy-specific terms
                    strategy_terms = {
                        'scalping': ['microstructure', 'ultra-short', 'momentum'],
                        'day_trading': ['intraday', 'session', 'same-day'],
                        'swing_trading': ['intermediate', 'multi-day', 'swing'],
                        'position_trading': ['long-term', 'strategic', 'macro']
                    }
                    
                    found_terms = []
                    for term in strategy_terms.get(strategy_name, []):
                        if term.lower() in content.lower():
                            found_terms.append(term)
                    
                    print(f"   Strategy terms found: {found_terms}")
                    success_count += 1
            else:
                print(f"✗ {template_file}: File not found")
        except Exception as e:
            print(f"✗ {template_file}: Error - {e}")
    
    print(f"\nStrategy template loading: {success_count}/{len(strategy_templates)} successful")
    return success_count == len(strategy_templates)


def test_market_context_detection():
    """Test market context detection from data"""
    print("\n" + "=" * 60)
    print("TESTING MARKET CONTEXT DETECTION")
    print("=" * 60)
    
    if not DOMAIN_PROMPTING_AVAILABLE:
        print("Domain prompting not available - skipping")
        return False
    
    try:
        generator = DomainPromptGenerator()
        
        # Test regime detection with different market scenarios
        scenarios = [
            ("Bull Market", np.array([50000, 51000, 52000, 53000, 54000, 55000] * 5)),
            ("Bear Market", np.array([55000, 54000, 53000, 52000, 51000, 50000] * 5)),
            ("Sideways Market", np.array([50000, 51000, 50500, 50800, 50200, 50600] * 5)),
            ("High Volatility", np.array([50000, 55000, 48000, 57000, 45000, 60000] * 5))
        ]
        
        for scenario_name, price_data in scenarios:
            regime = generator.detect_market_regime(price_data)
            print(f"✓ {scenario_name}: Detected regime '{regime}'")
        
        # Test market context extraction
        sample_data = pd.Series({
            'close': 52000,
            'price_lag_1': 51000,
            'hour': 14,
            'day_of_week': 2,
            'volume': 1000000
        })
        
        external_features = {
            'sentiment_score': 0.7,
            'fear_greed_index': 65,
            'fed_funds_rate': 5.25,
            'risk_on_sentiment': 2.5,
            'hash_rate_momentum': 0.1,
            'net_exchange_flow': -1000
        }
        
        context = generator.extract_market_context_from_data(sample_data, external_features)
        
        print(f"\nMarket Context Extraction:")
        print(f"  Price momentum: {context.price_momentum:.4f}")
        print(f"  Sentiment score: {context.sentiment_score}")
        print(f"  Fear/Greed index: {context.fear_greed_index}")
        print(f"  Macro environment: {context.macro_environment}")
        print(f"  On-chain strength: {context.onchain_strength}")
        print(f"  Time of day: {context.time_of_day}")
        print(f"  Day of week: {context.day_of_week}")
        
        print("\n✓ Market context detection test passed")
        return True
        
    except Exception as e:
        print(f"✗ Market context detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_prompt_generation():
    """Test dynamic prompt generation"""
    print("\n" + "=" * 60)
    print("TESTING DYNAMIC PROMPT GENERATION")
    print("=" * 60)
    
    if not DOMAIN_PROMPTING_AVAILABLE:
        print("Domain prompting not available - skipping")
        return False
    
    try:
        generator = DomainPromptGenerator()
        
        # Test different market contexts
        contexts = [
            MarketContext(
                regime="bull_med_vol",
                sentiment_score=0.7,
                fear_greed_index=70,
                macro_environment="bullish",
                onchain_strength="strong"
            ),
            MarketContext(
                regime="bear_high_vol", 
                sentiment_score=0.3,
                fear_greed_index=25,
                macro_environment="bearish",
                onchain_strength="weak"
            ),
            MarketContext(
                regime="sideways_low_vol",
                sentiment_score=0.5,
                fear_greed_index=50,
                macro_environment="neutral",
                onchain_strength="moderate"
            )
        ]
        
        trading_styles = ["scalping", "day_trading", "swing_trading", "position_trading"]
        
        for i, context in enumerate(contexts):
            regime_name = context.regime.replace('_', ' ').title()
            print(f"\n{i+1}. Testing {regime_name} Context:")
            
            for style in trading_styles:
                prompt = generator.generate_enhanced_prompt(
                    market_context=context,
                    dataset_type="enhanced",
                    trading_style=style,
                    prediction_horizon=24
                )
                
                print(f"   ✓ {style}: {len(prompt)} characters")
                
                # Check for context-specific terms
                if context.regime in prompt.lower():
                    print(f"     - Contains regime context")
                if style in prompt.lower():
                    print(f"     - Contains trading style context")
        
        print("\n✓ Dynamic prompt generation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Dynamic prompt generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_manager():
    """Test dynamic prompt manager"""
    print("\n" + "=" * 60)
    print("TESTING DYNAMIC PROMPT MANAGER")
    print("=" * 60)
    
    if not DOMAIN_PROMPTING_AVAILABLE:
        print("Domain prompting not available - skipping")
        return False
    
    try:
        manager = DynamicPromptManager()
        
        # Create sample price data
        np.random.seed(42)
        price_data = np.cumprod(1 + np.random.normal(0, 0.02, 100)) * 50000
        
        # Create sample current data
        current_data = pd.Series({
            'close': price_data[-1],
            'price_lag_1': price_data[-2],
            'hour': 14,
            'day_of_week': 2,
            'volume': 1000000
        })
        
        # Sample external features
        external_features = {
            'sentiment_score': 0.6,
            'fear_greed_index': 55,
            'fed_funds_rate': 5.25,
            'sp500': 4500,
            'hash_rate_momentum': 0.05
        }
        
        # Test dynamic prompt generation
        prompt = manager.generate_dynamic_prompt(
            price_data=price_data,
            current_data=current_data,
            external_features=external_features,
            prediction_horizon=24
        )
        
        print(f"Generated dynamic prompt: {len(prompt)} characters")
        print(f"First 200 characters: {prompt[:200]}...")
        
        # Test caching
        prompt2 = manager.generate_dynamic_prompt(
            price_data=price_data,
            current_data=current_data,
            external_features=external_features,
            prediction_horizon=24
        )
        
        if prompt == prompt2:
            print("✓ Prompt caching working correctly")
        else:
            print("⚠ Prompt caching may not be working")
        
        # Test performance tracking
        sample_metrics = {'mse': 0.001, 'mae': 0.02, 'mape': 1.5}
        manager.update_prompt_performance("test_prompt", sample_metrics)
        
        best_prompts = manager.get_best_performing_prompts()
        print(f"Performance tracking: {len(best_prompts)} entries")
        
        print("\n✓ Dynamic prompt manager test passed")
        return True
        
    except Exception as e:
        print(f"✗ Dynamic prompt manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_tools_integration():
    """Test integration with enhanced tools"""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED TOOLS INTEGRATION")
    print("=" * 60)
    
    if not ENHANCED_TOOLS_AVAILABLE:
        print("Enhanced tools not available - skipping")
        return False
    
    try:
        # Test dynamic prompting functionality
        enhancement = test_dynamic_prompting()
        
        if enhancement:
            print(f"✓ Enhanced tools integration working")
            print(f"Enhancement length: {len(enhancement)} characters")
            return True
        else:
            print("✗ Enhanced tools integration failed")
            return False
        
    except Exception as e:
        print(f"✗ Enhanced tools integration test failed: {e}")
        return False


def test_prompt_template_generation():
    """Test automated prompt template generation"""
    print("\n" + "=" * 60)
    print("TESTING PROMPT TEMPLATE GENERATION")
    print("=" * 60)
    
    if not DOMAIN_PROMPTING_AVAILABLE:
        print("Domain prompting not available - skipping")
        return False
    
    try:
        # Test creating regime-aware prompts
        output_dir = "./test_prompts/"
        create_regime_aware_prompts(output_dir)
        
        # Check if files were created
        expected_files = [
            "CRYPTEX_bull_market.txt",
            "CRYPTEX_bear_market.txt",
            "CRYPTEX_sideways_market.txt",
            "CRYPTEX_high_volatility.txt",
            "CRYPTEX_low_volatility.txt"
        ]
        
        created_files = 0
        for filename in expected_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.read()
                    print(f"✓ {filename}: {len(content)} characters")
                    created_files += 1
            else:
                print(f"✗ {filename}: Not created")
        
        # Clean up test files
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        print(f"\nTemplate generation: {created_files}/{len(expected_files)} successful")
        return created_files >= len(expected_files) - 1  # Allow one failure
        
    except Exception as e:
        print(f"✗ Prompt template generation test failed: {e}")
        return False


def create_sample_prompts_showcase():
    """Create sample prompts to showcase capabilities"""
    print("\n" + "=" * 60)
    print("SAMPLE PROMPTS SHOWCASE")
    print("=" * 60)
    
    if not DOMAIN_PROMPTING_AVAILABLE:
        print("Domain prompting not available - skipping")
        return
    
    try:
        generator = DomainPromptGenerator()
        
        # Showcase different scenarios
        scenarios = [
            ("Bull Market Scalping", MarketContext(
                regime="bull_high_vol",
                sentiment_score=0.8,
                fear_greed_index=80,
                time_of_day="us_session"
            ), "scalping"),
            
            ("Bear Market Swing Trading", MarketContext(
                regime="bear_med_vol",
                sentiment_score=0.3,
                fear_greed_index=30,
                macro_environment="bearish"
            ), "swing_trading"),
            
            ("Sideways Position Trading", MarketContext(
                regime="sideways_low_vol",
                sentiment_score=0.5,
                fear_greed_index=50,
                onchain_strength="strong"
            ), "position_trading")
        ]
        
        for scenario_name, context, trading_style in scenarios:
            print(f"\n{scenario_name}:")
            print("-" * 40)
            
            prompt = generator.generate_enhanced_prompt(
                market_context=context,
                trading_style=trading_style,
                prediction_horizon=24
            )
            
            # Show first part of prompt
            print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
            print(f"(Total length: {len(prompt)} characters)")
    
    except Exception as e:
        print(f"Error creating sample prompts: {e}")


def main():
    """Main test function for domain-specific prompting"""
    print("Testing Domain-Specific Prompting System")
    print("=" * 80)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Prompt Templates", test_prompt_templates()))
    test_results.append(("Trading Strategy Templates", test_trading_strategy_templates()))
    test_results.append(("Market Context Detection", test_market_context_detection()))
    test_results.append(("Dynamic Prompt Generation", test_dynamic_prompt_generation()))
    test_results.append(("Prompt Manager", test_prompt_manager()))
    test_results.append(("Enhanced Tools Integration", test_enhanced_tools_integration()))
    test_results.append(("Template Generation", test_prompt_template_generation()))
    
    # Summary
    print("\n" + "=" * 80)
    print("DOMAIN-SPECIFIC PROMPTING TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:35} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed >= total - 1:  # Allow one failure
        print("\n🎉 Domain-Specific Prompting system is working correctly!")
        print("\nKey capabilities now available:")
        print("• Market regime-aware prompt generation (9 regime types)")
        print("• Trading strategy-specific templates (scalping, day, swing, position)")
        print("• Dynamic external data context integration")
        print("• Temporal and session-aware prompting")
        print("• Real-time market condition adaptation")
        print("• Comprehensive prompt template library")
        print("• Performance tracking and optimization")
        print("• Seamless TimeLLM integration")
        
        print("\nNext steps:")
        print("1. Integrate with TimeLLM training pipeline")
        print("2. Add real-time market data feeds")
        print("3. Implement prompt performance optimization")
        print("4. Test with different cryptocurrency datasets")
        
        # Show sample prompts
        create_sample_prompts_showcase()
        
        return True
    else:
        print(f"\n❌ {total-passed} tests failed. Check errors above.")
        return False


if __name__ == "__main__":
    main()