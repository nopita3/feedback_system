"""
Cache Helper Module
Helping to track and log cache metrics for Gemini API calls
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


def extract_cache_metrics(usage_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract cache-related metrics from the usage metadata returned by Gemini API.
    
    Args:
        usage_metadata: Dict containing token usage information from callback
        
    Returns:
        Dict with cache metrics including:
        - cache_creation_input_tokens: Tokens used for cache creation
        - cache_read_input_tokens: Tokens saved by reading from cache
        - input_tokens: Original input tokens
        - output_tokens: Output tokens generated
        - total_tokens: Total tokens used
    """
    if not usage_metadata:
        return {}
    
    metrics = {
        'cache_creation_input_tokens': usage_metadata.get('cache_creation_input_tokens', 0),
        'cache_read_input_tokens': usage_metadata.get('cache_read_input_tokens', 0),
        'input_tokens': usage_metadata.get('input_tokens', 0),
        'output_tokens': usage_metadata.get('output_tokens', 0),
        'total_tokens': usage_metadata.get('total_tokens', 0),
    }
    
    return metrics


def calculate_cache_savings(cache_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate token and cost savings from cache usage.
    
    Args:
        cache_metrics: Dict with cache-related metrics
        
    Returns:
        Dict with savings analysis:
        - tokens_saved: Number of tokens saved from cache reads
        - tokens_with_cache: Total tokens if all from cache
        - savings_percentage: Percentage of tokens saved
    """
    cache_read = cache_metrics.get('cache_read_input_tokens', 0)
    cache_create = cache_metrics.get('cache_creation_input_tokens', 0)
    input_tokens = cache_metrics.get('input_tokens', 0)
    
    total_without_cache = cache_create + input_tokens
    tokens_saved = cache_read * 0.9  # Cache reads cost 90% less
    
    return {
        'tokens_saved': tokens_saved,
        'total_tokens_without_cache': total_without_cache,
        'savings_percentage': (tokens_saved / total_without_cache * 100) if total_without_cache > 0 else 0,
        'cache_read_tokens': cache_read,
        'cache_create_tokens': cache_create,
    }


def log_cache_metrics(callback_metadata: Dict[str, Any], process_name: str = "process", log_file: str = "cache_metrics.log"):
    """
    Log cache metrics to a file for later analysis.
    
    Args:
        callback_metadata: Usage metadata from the callback
        process_name: Name of the process being measured
        log_file: Path to the log file
    """
    cache_metrics = extract_cache_metrics(callback_metadata)
    savings = calculate_cache_savings(cache_metrics)
    
    log_entry = {
        'timestamp': str(datetime.now()),
        'process': process_name,
        'cache_metrics': cache_metrics,
        'savings_analysis': savings,
    }
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False, default=str) + '\n')
    except Exception as e:
        print(f"Error writing cache metrics log: {e}")


def print_cache_summary(callback_metadata: Dict[str, Any], process_name: str = ""):
    """
    Print a human-readable summary of cache usage.
    
    Args:
        callback_metadata: Usage metadata from the callback
        process_name: Optional name of the process
    """
    metrics = extract_cache_metrics(callback_metadata)
    savings = calculate_cache_savings(metrics)
    
    prefix = f"[{process_name}] " if process_name else ""
    
    print(f"\n{prefix}Cache Metrics Summary:")
    print(f"  Cache Creation Input Tokens: {metrics['cache_creation_input_tokens']:,}")
    print(f"  Cache Read Input Tokens: {metrics['cache_read_input_tokens']:,}")
    print(f"  Regular Input Tokens: {metrics['input_tokens']:,}")
    print(f"  Output Tokens: {metrics['output_tokens']:,}")
    print(f"  Total Tokens: {metrics['total_tokens']:,}")
    
    if savings['tokens_saved'] > 0:
        print(f"\n  💾 Tokens Saved from Cache: {savings['tokens_saved']:,.0f}")
        print(f"  📊 Savings: {savings['savings_percentage']:.1f}%")
