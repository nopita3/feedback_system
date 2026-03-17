"""
Example: Context Caching Usage with Gemini API

This script demonstrates how to use context caching to reduce token usage
when processing multiple similar requests (e.g., OCR on multiple PDF pages).
"""

import json
from pathlib import Path
from datetime import datetime

# Import helpers
from config import get_gemini_model, create_cached_message
from cache_helper import extract_cache_metrics, calculate_cache_savings, print_cache_summary


def example_simple_text_cache():
    """
    Example 1: Simple text caching
    Using cache_control for repeated prompt
    """
    print("=" * 60)
    print("Example 1: Simple Text Caching")
    print("=" * 60)
    
    llm, callback = get_gemini_model(enable_cache=True)
    
    system_prompt = """
    You are an expert in analyzing physics exam questions.
    Extract question information and return ONLY valid JSON.
    """
    
    # First request - creates cache
    message1 = create_cached_message(
        content=system_prompt + "\nAnalyze: E=mc²",
        cache_control=True
    )
    response1 = llm.invoke([message1])
    print(f"\nFirst Request (Cache Creation):")
    print_cache_summary(callback.usage_metadata, "Request 1")
    
    # Second request - reads from cache
    message2 = create_cached_message(
        content=system_prompt + "\nAnalyze: F=ma",
        cache_control=True
    )
    response2 = llm.invoke([message2])
    print(f"\nSecond Request (Cache Read):")
    print_cache_summary(callback.usage_metadata, "Request 2")


def example_batch_processing():
    """
    Example 2: Batch processing with cache
    Process multiple items with the same prompt template
    """
    print("\n\n" + "=" * 60)
    print("Example 2: Batch Processing with Cache")
    print("=" * 60)
    
    llm, callback = get_gemini_model(enable_cache=True)
    
    # System instructions that will be cached
    system_instruction = """
    You are an OCR specialist for physics exams.
    Extract questions in JSON format.
    Don't include answer choices.
    """
    
    # Multiple questions to process
    questions = [
        "Question 1: A ball is thrown...",
        "Question 2: An electron moves...",
        "Question 3: What is momentum...",
    ]
    
    total_metrics = {
        'cache_creation_input_tokens': 0,
        'cache_read_input_tokens': 0,
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
    }
    
    for i, question in enumerate(questions, 1):
        prompt = f"{system_instruction}\n\n{question}"
        message = create_cached_message(prompt, cache_control=True)
        
        response = llm.invoke([message])
        metrics = extract_cache_metrics(callback.usage_metadata)
        
        print(f"\nRequest {i}:")
        print(f"  Total tokens used: {metrics['total_tokens']}")
        if metrics['cache_read_input_tokens'] > 0:
            print(f"  💾 Tokens from cache: {metrics['cache_read_input_tokens']}")
        
        # Accumulate metrics
        for key in total_metrics:
            total_metrics[key] += metrics[key]
    
    # Calculate total savings
    print("\n" + "-" * 60)
    print("Batch Processing Summary:")
    savings = calculate_cache_savings(total_metrics)
    print(f"Total tokens processed: {total_metrics['total_tokens']:,}")
    print(f"Tokens saved by caching: {savings['tokens_saved']:,.0f}")
    if total_metrics['total_tokens'] > 0:
        print(f"Savings percentage: {savings['savings_percentage']:.1f}%")


def example_complex_content():
    """
    Example 3: Complex content (text + image) with cache
    Demonstrates caching with multimodal inputs
    """
    print("\n\n" + "=" * 60)
    print("Example 3: Multimodal Content with Cache")
    print("=" * 60)
    
    llm, callback = get_gemini_model(enable_cache=True)
    
    # Example content structure for image + text
    content = [
        {
            "type": "text",
            "text": "Analyze this physics exam question and extract: question_id, question_content, skill_tags, error_type",
            "cache_control": {"type": "ephemeral"}  # Cache the instruction
        },
        {
            "type": "image_url",
            "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        }
    ]
    
    from langchain_core.messages import HumanMessage
    message = HumanMessage(content=content)
    
    print("Multimodal message structure:")
    print(f"  - Text portion cached: Yes")
    print(f"  - Image included: Yes")
    print(f"  - Cache type: Ephemeral (5-minute duration)")
    
    print_cache_summary(callback.usage_metadata, "Multimodal Request")


def log_cache_statistics(log_file: str = "cache_statistics.json"):
    """
    Example: Log and analyze cache statistics
    """
    print("\n\n" + "=" * 60)
    print("Example 4: Logging Cache Statistics")
    print("=" * 60)
    
    llm, callback = get_gemini_model(enable_cache=True)
    
    # Perform some requests
    llm.invoke(["Test prompt 1"])
    metrics1 = extract_cache_metrics(callback.usage_metadata)
    
    llm.invoke(["Test prompt 2"])
    metrics2 = extract_cache_metrics(callback.usage_metadata)
    
    # Log statistics
    stats = {
        'timestamp': datetime.now().isoformat(),
        'requests': [
            {'request': 1, 'metrics': metrics1},
            {'request': 2, 'metrics': metrics2},
        ],
        'total_savings': calculate_cache_savings({
            'cache_creation_input_tokens': metrics1['cache_creation_input_tokens'],
            'cache_read_input_tokens': metrics2['cache_read_input_tokens'],
            'input_tokens': metrics1['input_tokens'] + metrics2['input_tokens'],
        })
    }
    
    with open(log_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Cache statistics logged to {log_file}")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Context Caching Examples")
    print("Gemini API with LangChain Integration")
    print("=" * 60)
    
    # Uncomment the example you want to run:
    
    # example_simple_text_cache()
    # example_batch_processing()
    # example_complex_content()
    # log_cache_statistics()
    
    print("\n✅ Context caching is now implemented!")
    print("\nTo use in your code:")
    print("1. Import: from config import get_gemini_model, create_cached_message")
    print("2. Get model: llm, callback = get_gemini_model(enable_cache=True)")
    print("3. Create message: message = create_cached_message(prompt, cache_control=True)")
    print("4. Call model: response = llm.invoke([message])")
    print("\nSee CONTEXT_CACHING_GUIDE.md for detailed information.")
