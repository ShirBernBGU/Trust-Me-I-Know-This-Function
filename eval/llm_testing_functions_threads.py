import pandas as pd
import time
import requests
import json
from typing import Optional
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random

# API Configuration - Add your actual API keys here
OPENAI_API_KEY = "" # Add your OpenAI API key here
ANTHROPIC_API_KEY = "" # Add your Anthropic API key here 
GOOGLE_API_KEY = "" # Add your Google API key here
OPENROUTER_API_KEY = "" # Add your OpenRouter API key here


# Retry Configuration
MAX_RETRIES = 3
BASE_DELAY = 1.0  # Base delay in seconds
MAX_DELAY = 600.0  # Maximum delay in seconds
MAX_TOKENS_REGULAR = 4000
MAX_TOKENS_REASONING = 8000

# Path to files:
INPUT_PATH = "" # Add your input CSV path here (e.g. "input/llm_testing_data.csv")
OUTPUT_PATH = "" # Add your output CSV path here (e.g. "output/")

# check if the input and output paths exist:
def check_paths():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
    if not os.path.exists(OUTPUT_PATH):
        raise FileNotFoundError(f"Output directory not found: {OUTPUT_PATH}")
    
check_paths()

def wait_with_jitter(delay: float):
    """Wait with random jitter to avoid thundering herd"""
    jitter = random.uniform(0.1, 0.3) * delay
    time.sleep(delay + jitter)

def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is worth retrying"""
    error_str = str(error).lower()
    
    # Retryable errors
    retryable_patterns = [
        'rate limit', 'timeout', 'connection', 'network', 
        'temporary', '429', '500', '502', '503', '504',
        'internal server error', 'bad gateway', 'service unavailable'
    ]
    
    # Non-retryable errors (don't waste time retrying these)
    non_retryable_patterns = [
        'unauthorized', 'invalid api key', 'authentication', 
        'forbidden', '401', '403', 'quota exceeded', 'billing'
    ]
    
    # Check for non-retryable first
    for pattern in non_retryable_patterns:
        if pattern in error_str:
            return False
    
    # Check for retryable
    for pattern in retryable_patterns:
        if pattern in error_str:
            return True
    
    # Default: retry unknown errors (might be temporary)
    return True

def test_with_retry(test_function, query: str, llm_name: str, max_retries: int = MAX_RETRIES) -> str:
    """
    Test LLM with intelligent retry logic
    
    Args:
        test_function: The LLM test function to call
        query: Query to send to LLM
        llm_name: Name of LLM for logging
        max_retries: Maximum number of retry attempts
    
    Returns:
        LLM response or error message
    """
    last_error = None
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            response = test_function(query)
            
            # Check if response looks like an error
            if response.startswith("Error:"):
                raise Exception(response[7:])  # Remove "Error: " prefix
                
            return response
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            # Log the attempt
            if attempt == 0:
                print(f"    ⚠️  {llm_name} error: {error_str}")
            else:
                print(f"    🔄 {llm_name} retry {attempt}/{max_retries}: {error_str}")
            
            # Don't retry if it's the last attempt
            if attempt >= max_retries:
                break
                
            # Don't retry non-retryable errors
            if not is_retryable_error(e):
                print(f"    ❌ {llm_name} non-retryable error, giving up")
                break
            
            # Calculate delay with exponential backoff
            delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
            print(f"    ⏳ {llm_name} waiting {delay:.1f}s before retry...")
            wait_with_jitter(delay)
    
    # All retries failed
    final_error = f"Failed after {max_retries + 1} attempts: {str(last_error)}"
    print(f"    💀 {llm_name} giving up: {final_error}")
    return f"Error: {final_error}"

# def test_claude(query: str) -> str:
#     """Test Claude with the current Anthropic API format"""
#     try:
#         import anthropic
        
#         client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
#         message = client.messages.create(
#             model="claude-3-5-sonnet-20240620",  # or "claude-sonnet-4-20250514"
#             max_tokens=MAX_TOKENS_REGULAR,
#             temperature=0.5,
#             messages=[
#                 {"role": "user", "content": query}
#             ]
#         )
#         return message.content[0].text.strip()
        
#     except Exception as e:
#         # Let the retry logic handle this
#         raise e

def test_claude(query: str) -> str:
    """Test Claude Sonnet 3.5 via OpenRouter"""
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        
        completion = client.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "user", "content": query}
            ],
            temperature=0.5,
            max_tokens=MAX_TOKENS_REGULAR
        )
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        raise e
    

def test_gpt(query: str) -> str:
    """Test GPT with the current OpenAI API format"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4o-mini" for cheaper option
            messages=[
                {"role": "user", "content": query}
            ],
            temperature=0.5,
            max_tokens=MAX_TOKENS_REGULAR
        )
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        # Let the retry logic handle this
        raise e

# def test_gemini(query: str) -> str:
#     """Test Gemini with the current Google GenAI SDK (recommended)"""
#     try:
#         from google import genai
#         from google.genai import types
        
#         client = genai.Client(api_key=GOOGLE_API_KEY)
#         response = client.models.generate_content(
#             model='gemini-2.0-flash',  # or 'gemini-2.5-flash-preview-04-17'
#             contents=query,
#             config=types.GenerateContentConfig(
#                 temperature=0.5,
#                 max_output_tokens=MAX_TOKENS_REGULAR
#             )
#         )
#         return response.text.strip()
        
#     except Exception as e:
#         # Let the retry logic handle this
#         raise e

def test_gemini(query: str) -> str:
    """Test Gemini 2.0 flash via OpenRouter"""
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        
        completion = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {"role": "user", "content": query}
            ],
            temperature=0.5,
            max_tokens=MAX_TOKENS_REGULAR
        )
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        raise e

    
############# NEW MODELS ##############

# ADD these new functions:
def test_claude_opus41(query: str) -> str:
    """Test Claude Opus 4.1 with extended thinking"""
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=300.0)
        message = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=MAX_TOKENS_REASONING,
            thinking={
                "type": "enabled",
                "budget_tokens": 5000
            },
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        response_parts = []
        for block in message.content:
            if block.type == "text":
                response_parts.append(block.text)
        
        return "\n".join(response_parts).strip()
    except Exception as e:
        raise e

def test_gpt_o3(query: str) -> str:
    """Test GPT o3"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model="o3",
            messages=[{"role": "user", "content": query}],
            max_completion_tokens=MAX_TOKENS_REASONING,  # Max tokens for o3 models
            reasoning_effort="medium"
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise e

def test_gpt_5(query: str) -> str:
    """Test GPT-5"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": query}],
            temperature=0.5,
            max_tokens=MAX_TOKENS_REGULAR
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise e

def test_gpt_5_thinking(query: str) -> str:
    """Test GPT-5 with thinking"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": query}],
            # temperature=0.5,
            # max_tokens=MAX_TOKENS_REASONING,
            max_completion_tokens=MAX_TOKENS_REASONING,
            reasoning_effort="medium"
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise e

# def test_gemini_25_pro(query: str) -> str:
#     """Test Gemini 2.5 Pro with thinking"""
#     try:
#         from google import genai
#         from google.genai import types
        
#         client = genai.Client(api_key=GOOGLE_API_KEY)
#         response = client.models.generate_content(
#             model='gemini-2.5-pro',  # Stable version
#             contents=query,
#             config=types.GenerateContentConfig(
#                 temperature=0.5,
#                 max_output_tokens=MAX_TOKENS_REASONING
#             )
#         )
#         return response.text.strip()
#     except Exception as e:
#         raise e

def test_gemini_25_pro(query: str) -> str:
    """Test Gemini 2.5 Pro via OpenRouter"""
    try:
        from openai import OpenAI
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        
        completion = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=[
                {"role": "user", "content": query}
            ],
            temperature=0.5,
            max_tokens=MAX_TOKENS_REASONING
        )
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        raise e
    
def test_claude_sonnet_4_thinking(query: str) -> str:
    """Test Claude Sonnet 4 with extended thinking enabled"""
    try:
        import anthropic
        
        # NOTE: Placeholder Model ID - Update this to the official Claude 4 Sonnet ID when available
        MODEL_ID = "claude-4-sonnet-20250514"
        
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=300.0)
        message = client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS_REASONING,
            # Enable the 'thinking' feature
            thinking={
                "type": "enabled",
                "budget_tokens": 5000  # Adjust thinking budget as needed
            },
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        response_parts = []
        for block in message.content:
            if block.type == "text":
                response_parts.append(block.text)
        
        return "\n".join(response_parts).strip()
    except Exception as e:
        # Assuming ANTHROPIC_API_KEY and MAX_TOKENS_REASONING are defined globally
        raise e
    
def test_claude_sonnet_4(query: str) -> str:
    """Test Claude Sonnet 4 standard call"""
    try:
        import anthropic
        
        # NOTE: Placeholder Model ID - Update this to the official Claude 4 Sonnet ID when available
        MODEL_ID = "claude-4-sonnet-20250514"
        
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=300.0)
        message = client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS_REGULAR,
            # The 'thinking' block is intentionally omitted for a standard call
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        response_parts = []
        for block in message.content:
            if block.type == "text":
                response_parts.append(block.text)
        
        return "\n".join(response_parts).strip()
    except Exception as e:
        # Assuming ANTHROPIC_API_KEY and MAX_TOKENS_REASONING are defined globally
        raise e
    
###########################################

def get_llm_function(llm_name: str):
    """Return the appropriate LLM function"""
    llm_functions = {
        # Original models
        "Claude": test_claude,
        "GPT": test_gpt,
        "Gemini": test_gemini,
        # New models
        "Claude_opus41": test_claude_opus41,
        "GPT_o3": test_gpt_o3,
        "GPT_5": test_gpt_5,
        "GPT_5_thinking": test_gpt_5_thinking,
        "Gemini_2.5_pro": test_gemini_25_pro,
        "Claude_sonnet_4_thinking": test_claude_sonnet_4_thinking,
        "Claude_sonnet_4": test_claude_sonnet_4
    }
    return llm_functions.get(llm_name)

def test_single_row(row: pd.Series, add_delay: bool = True) -> tuple:
    """
    Test a single row with both original and modified queries (WITH RETRY LOGIC)
    
    Args:
        row: Single row from DataFrame
        add_delay: Whether to add delay between calls
    
    Returns:
        tuple: (row_index, original_response, modified_response)
    """
    llm_name = row['LLM']
    query_original = row['query_original']
    query_modified = row['query_modified']
    row_index = row.name  # Get the index of this row
    
    # Get the appropriate function
    llm_function = get_llm_function(llm_name)
    if not llm_function:
        error_msg = f"Error: Unknown LLM '{llm_name}'"
        return row_index, error_msg, error_msg
    
    # Test original query with retry logic
    original_response = None
    if pd.isna(row.get('LLM_output_original')):
        print(f"    🔵 Testing {llm_name} original query...")
        original_response = test_with_retry(llm_function, query_original, llm_name)
        if add_delay and not original_response.startswith("Error:"):
            time.sleep(0.5)  # Only delay on success
    
    # Test modified query with retry logic
    modified_response = None
    if pd.isna(row.get('LLM_output_modified')):
        print(f"    🟡 Testing {llm_name} modified query...")
        modified_response = test_with_retry(llm_function, query_modified, llm_name)
        if add_delay and not modified_response.startswith("Error:"):
            time.sleep(0.5)  # Only delay on success
    
    return row_index, original_response, modified_response

def process_dataframe_threaded(df: pd.DataFrame, max_workers: int = 20, save_interval: int = 10, start_from: int = 0) -> pd.DataFrame:
    """
    Process the DataFrame using threading for faster execution (WITH RETRY LOGIC)
    
    Args:
        df: DataFrame with LLM testing data
        max_workers: Maximum number of concurrent threads (be careful with API rate limits)
        save_interval: Save progress every N completed rows
        start_from: Row index to start from (for resuming)
    
    Returns:
        Updated DataFrame with LLM responses
    """
    df_copy = df.copy()
    total_rows = len(df_copy)
    completed = 0
    errors = 0
    lock = threading.Lock()  # For thread-safe DataFrame updates
    
    print(f"🚀 Starting threaded testing with {max_workers} workers...")
    print(f"📊 Retry config: max_retries={MAX_RETRIES}, base_delay={BASE_DELAY}s")
    print(f"Processing {total_rows} rows (starting from row {start_from})...")
    
    # Filter rows that need processing (starting from start_from)
    rows_to_process = []
    for idx in range(start_from, total_rows):
        row = df_copy.iloc[idx]
        # Only process if either output is missing
        if pd.isna(row.get('LLM_output_original')) or pd.isna(row.get('LLM_output_modified')):
            rows_to_process.append((idx, row))
    
    print(f"Found {len(rows_to_process)} rows that need processing...")
    
    if not rows_to_process:
        print("✅ All rows already completed!")
        return df_copy
    
    def process_and_update(idx_row_tuple):
        """Process a single row and return results for updating"""
        nonlocal completed, errors  # Declare nonlocal at the very beginning
        
        idx, row = idx_row_tuple
        try:
            llm_name = row['LLM']
            primitive_name = row.get('primitive_name', 'Unknown')
            trial_id = row.get('trial_id', 'Unknown')
            
            print(f"  🎯 Starting: {llm_name} - {primitive_name} - {trial_id}")
            
            # Process the row (this now includes retry logic)
            row_index, original_response, modified_response = test_single_row(row)
            
            # Count errors
            error_count = 0
            if original_response and original_response.startswith("Error:"):
                error_count += 1
            if modified_response and modified_response.startswith("Error:"):
                error_count += 1
            
            # Thread-safe logging
            with lock:
                completed += 1
                errors += error_count
                
                if error_count > 0:
                    print(f"  ⚠️  [{completed}/{len(rows_to_process)}] Completed with {error_count} error(s): {llm_name} - {primitive_name} - {trial_id}")
                else:
                    print(f"  ✅ [{completed}/{len(rows_to_process)}] Success: {llm_name} - {primitive_name} - {trial_id}")
            
            return idx, original_response, modified_response, error_count
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            with lock:
                completed += 1
                errors += 2  # Both queries failed
                print(f"  💀 [{completed}/{len(rows_to_process)}] Complete failure in row {idx}: {e}")
            
            return idx, error_msg, error_msg, 2
    
    # Process rows in parallel
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(process_and_update, idx_row): idx_row[0] 
                         for idx_row in rows_to_process}
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx, original_response, modified_response, error_count = future.result()
            
            # Update DataFrame (thread-safe)
            with lock:
                if original_response is not None:
                    df_copy.at[idx, 'LLM_output_original'] = original_response
                if modified_response is not None:
                    df_copy.at[idx, 'LLM_output_modified'] = modified_response
                
                # Save progress periodically
                if completed % save_interval == 0:
                    filename = f"{OUTPUT_PATH}llm_testing_threaded_progress_{completed}.csv"
                    df_copy.to_csv(filename, index=False)
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    success_rate = ((completed * 2 - errors) / (completed * 2)) * 100 if completed > 0 else 0
                    print(f"  💾 Saved progress: {completed} completed ({rate:.1f} rows/sec, {success_rate:.1f}% success)")
    
    # Final save and statistics
    elapsed = time.time() - start_time
    final_rate = len(rows_to_process) / elapsed if elapsed > 0 else 0
    total_queries = len(rows_to_process) * 2
    success_rate = ((total_queries - errors) / total_queries) * 100 if total_queries > 0 else 0
    
    df_copy.to_csv(f"{OUTPUT_PATH}llm_testing_threaded_final_results.csv", index=False)
    
    print(f"\n🎉 Completed threaded testing!")
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print(f"📊 Average rate: {final_rate:.1f} rows/sec")
    print(f"✅ Success rate: {success_rate:.1f}% ({total_queries - errors}/{total_queries} queries)")
    print(f"❌ Failed queries: {errors}")
    print(f"💾 Final results saved to 'llm_testing_threaded_final_results.csv'")
    
    return df_copy

def process_dataframe(df: pd.DataFrame, save_interval: int = 10, start_from: int = 0) -> pd.DataFrame:
    """
    Process the entire DataFrame row by row (SEQUENTIAL with RETRY LOGIC)
    
    Args:
        df: DataFrame with LLM testing data
        save_interval: Save progress every N rows
        start_from: Row index to start from (for resuming)
    
    Returns:
        Updated DataFrame with LLM responses
    """
    df_copy = df.copy()
    total_rows = len(df_copy)
    processed = 0
    errors = 0
    
    print(f"Starting sequential testing of {total_rows} rows (starting from row {start_from})...")
    print(f"📊 Retry config: max_retries={MAX_RETRIES}, base_delay={BASE_DELAY}s")
    
    for idx in range(start_from, total_rows):
        row = df_copy.iloc[idx]
        
        try:
            llm_name = row['LLM']
            primitive_name = row.get('primitive_name', 'Unknown')
            trial_id = row.get('trial_id', 'Unknown')
            
            print(f"Testing row {idx + 1}/{total_rows}: {llm_name} - {primitive_name} - {trial_id}")
            
            # Test this row (now with retry logic)
            row_index, original_response, modified_response = test_single_row(row)
            
            # Count errors
            error_count = 0
            if original_response and original_response.startswith("Error:"):
                error_count += 1
            if modified_response and modified_response.startswith("Error:"):
                error_count += 1
            
            errors += error_count
            
            # Update the DataFrame
            if original_response is not None:
                df_copy.at[idx, 'LLM_output_original'] = original_response
            if modified_response is not None:
                df_copy.at[idx, 'LLM_output_modified'] = modified_response
            
            processed += 1
            
            # Save progress periodically
            if processed % save_interval == 0:
                filename = f"llm_testing_sequential_progress_{idx + 1}.csv"
                df_copy.to_csv(filename, index=False)
                success_rate = ((processed * 2 - errors) / (processed * 2)) * 100 if processed > 0 else 0
                print(f"  💾 Saved progress at row {idx + 1} ({success_rate:.1f}% success)")
                
        except Exception as e:
            print(f"💀 Complete failure processing row {idx}: {e}")
            df_copy.at[idx, 'LLM_output_original'] = f"Error: {str(e)}"
            df_copy.at[idx, 'LLM_output_modified'] = f"Error: {str(e)}"
            errors += 2
            processed += 1
    
    # Final statistics
    total_queries = processed * 2
    success_rate = ((total_queries - errors) / total_queries) * 100 if total_queries > 0 else 0
    
    df_copy.to_csv("llm_testing_sequential_final_results.csv", index=False)
    print(f"✅ Completed sequential testing! Success rate: {success_rate:.1f}% ({total_queries - errors}/{total_queries} queries)")
    
    return df_copy

def filter_by_llm(df: pd.DataFrame, llm_name: str) -> pd.DataFrame:
    """Filter DataFrame to only include specific LLM"""
    return df[df['LLM'] == llm_name].copy()

def test_specific_llm(df: pd.DataFrame, llm_name: str, save_interval: int = 10, use_threading: bool = True, max_workers: int = 5) -> pd.DataFrame:
    """Test only a specific LLM from the DataFrame"""
    filtered_df = filter_by_llm(df, llm_name)
    print(f"Testing only {llm_name}: {len(filtered_df)} rows")
    
    if use_threading:
        return process_dataframe_threaded(filtered_df, max_workers=max_workers, save_interval=save_interval)
    else:
        return process_dataframe(filtered_df, save_interval=save_interval)

def test_small_sample(df: pd.DataFrame, n_rows: int = 20, use_threading: bool = True, max_workers: int = 3):
    """Test a small sample first"""
    sample_df = df.head(n_rows)
    print(f"Testing small sample of {n_rows} rows")
    
    if use_threading:
        return process_dataframe_threaded(sample_df, max_workers=max_workers, save_interval=5)
    else:
        return process_dataframe(sample_df, save_interval=5)

def resume_testing(filename: str, start_from: int):
    """Resume testing from a specific row"""
    df = pd.read_csv(filename)
    return process_dataframe(df, start_from=start_from)

def setup_check():
    """Check if required packages are installed"""
    missing_packages = []
    
    try:
        import openai
    except ImportError:
        missing_packages.append("openai")
    
    try:
        import anthropic
    except ImportError:
        missing_packages.append("anthropic")
    
    try:
        from google import genai
    except ImportError:
        missing_packages.append("google-genai")
    
    if missing_packages:
        print("Missing packages. Install with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are available!")
    print("Don't forget to add your API keys to the configuration!")
    return True

# Main execution
if __name__ == "__main__":
    # Check setup
    if not setup_check():
        print("Please install required packages first.")
        exit(1)
    
    print("\n" + "="*60)
    print("🔄 ENHANCED RETRY SYSTEM ACTIVE!")
    print("="*60)
    print("RETRY CONFIGURATION:")
    print(f"• Max retries per query: {MAX_RETRIES}")
    print(f"• Base delay: {BASE_DELAY}s (with exponential backoff)")
    print(f"• Max delay: {MAX_DELAY}s")
    print(f"• Jitter: ±10-30% random variation")
    print()
    print("SMART ERROR HANDLING:")
    print("• ✅ Retries: Rate limits, timeouts, 5xx errors")
    print("• ❌ No retry: Auth errors, invalid API keys, billing issues")
    print("• 📊 Tracks success rates and error counts")
    print("="*60)
    
    # Load your filled DataFrame
    df = pd.read_csv(INPUT_PATH)
    
    # # Option 1: Test a small sample first (RECOMMENDED)
    # results = test_small_sample(df, 22, use_threading=True, max_workers=20)
    
    # Option 2: Test only one LLM first
    # results = test_specific_llm(df, "Claude", use_threading=True, max_workers=3)
    
    # Option 3: Test everything
    results = process_dataframe_threaded(df, max_workers=50, save_interval=1000)
    
    # Display some results
    print("\nSample results:")
    sample_cols = ['primitive_name', 'LLM', 'trial_id', 'LLM_output_original', 'LLM_output_modified']
    print(results[sample_cols].head(10))
