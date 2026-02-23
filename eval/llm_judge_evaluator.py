import pandas as pd
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# API Configuration
OPENAI_API_KEY = ""  # Add your OpenAI API key here

# Paths
INPUT_PATH = "" # Add your input CSV path here
OUTPUT_PATH = "" # Add your output CSV path prefix here (e.g. "output/evaluation_results_")

def judge_llm_output(expected_output: str, llm_output: str, query_type: str) -> tuple:
    """
    Judge if LLM output matches expected output
    
    Returns:
        tuple: (success: bool, notes: str)
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
#         prompt = f"""You are an automated judge that checks whether an LLM's predicted code output matches the expected output.

# You will be given two pieces of information:
# - Expected output: {expected_output}
# - LLM prediction: {llm_output}

# Rules for C boolean equivalence:
# - False = false = 0 (all equivalent)
# - True = true = 1 (all equivalent)
# - Ignore case differences and extra whitespace.

# Follow these exact steps:
# 1. If the final prediction equals the expected output, reply **CORRECT**.
# 2. If the final prediction is wrong, reply **INCORRECT**.
# 3. If the LLM response is cut off or did not finish, reply **UNFINISHED**.

# After the single‑word verdict, on the next line provide a 1–2‑sentence explanation of your reasoning.
# Do not add any extra text, headings, or formatting."""

        prompt = f"""You are an automated judge that checks whether an LLM's predicted code output matches the expected output. the predicted code output may contain reasoning steps before the final answer.

You will be given two pieces of information:
- Expected output: {expected_output}
- LLM prediction: {llm_output}

Rules for boolean equivalence:
- False = false = 0 = "0" (all equivalent)
- True = true = 1 = "1" (all equivalent)
- Ignore case differences and extra whitespace.

Follow these exact steps:
1. If the final prediction equals the expected output, reply **CORRECT**.
2. If the final prediction is wrong, reply **INCORRECT**.
3. If the LLM response is cut off or did not finish, reply **UNFINISHED**.

After the single‑word verdict, on the next line provide a 1–2‑sentence explanation of your reasoning.
Do not add any extra text, headings, or formatting."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        judge_response = response.choices[0].message.content.strip()
        
    #     # Determine success based on response
    #     success = judge_response.upper().startswith("CORRECT")
        
    #     return success, judge_response
        
    # except Exception as e:
    #     return False, f"Judge error: {str(e)}"
            # Determine result based on response

        if judge_response.upper().startswith("CORRECT"):
            return True, judge_response
        elif judge_response.upper().startswith("UNFINISHED"):
            return "unfinished", judge_response
        else:
            return False, judge_response
        
    except Exception as e:
        return False, f"Judge error: {str(e)}"

def evaluate_single_row(row: pd.Series) -> tuple:
    """
    Evaluate a single row
    
    Returns:
        tuple: (row_index, success_orig, notes_orig, success_mod, notes_mod)
    """
    row_index = row.name
    
    # Get values
    expected_orig = str(row.get('original_output', ''))
    expected_mod = str(row.get('modified_output', ''))
    llm_orig = str(row.get('LLM_output_original', ''))
    llm_mod = str(row.get('LLM_output_modified', ''))
    
    # Skip if missing data
    if any(pd.isna([expected_orig, expected_mod, llm_orig, llm_mod])):
        return row_index, False, "Missing data", False, "Missing data"
    
    # Skip if LLM errors
    if "Error: Unknown LLM" in llm_orig or "Error: Unknown LLM" in llm_mod:
        return row_index, False, "Skipped - Unknown LLM error", False, "Skipped - Unknown LLM error"
    
    # Judge original
    success_orig, notes_orig = judge_llm_output(expected_orig, llm_orig, "original")
    
    # Judge modified
    success_mod, notes_mod = judge_llm_output(expected_mod, llm_mod, "modified")
    
    return row_index, success_orig, notes_orig, success_mod, notes_mod

def evaluate_dataframe_threaded(df: pd.DataFrame, max_workers: int = 10, save_interval: int = 100) -> pd.DataFrame:
    """
    Evaluate DataFrame with threading
    """
    df_copy = df.copy()
    total_rows = len(df_copy)
    completed = 0
    lock = threading.Lock()
    
    print(f"🎯 Starting evaluation with {max_workers} workers...")
    print(f"Processing {total_rows} rows...")
    
    # # Add judgment columns
    # df_copy['judged_success_original'] = None
    # df_copy['judge_notes_original'] = None
    # df_copy['judged_success_modified'] = None
    # df_copy['judge_notes_modified'] = None


    # Find rows to process
    rows_to_process = []
    skipped_count = 0
    for idx, row in df_copy.iterrows():
        if pd.isna(row.get('judged_success_original')) or pd.isna(row.get('judged_success_modified')):
            rows_to_process.append((idx, row))
        else:
            skipped_count += 1
    
    print(f"Found {len(rows_to_process)} rows to evaluate...")
    print(f"Skipped {skipped_count} rows.")
    print(f"Total rows: {len(df_copy)}")
    
    if not rows_to_process:
        print("✅ All rows already evaluated!")
        return df_copy
    
    def process_row(idx_row_tuple):
        """Process single row"""
        nonlocal completed
        
        idx, row = idx_row_tuple
        
        # Evaluate row
        row_index, success_orig, notes_orig, success_mod, notes_mod = evaluate_single_row(row)
        
        # Thread-safe logging
        with lock:
            completed += 1
            llm_name = row.get('LLM', 'Unknown')
            print(f"[{completed}/{len(rows_to_process)}] {llm_name} - Row {idx}")
        
        return row_index, success_orig, notes_orig, success_mod, notes_mod
    
    # Process in parallel
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_idx = {executor.submit(process_row, idx_row): idx_row[0] 
                         for idx_row in rows_to_process}
        
        # Collect results
        for future in as_completed(future_to_idx):
            row_index, success_orig, notes_orig, success_mod, notes_mod = future.result()
            
            # Update DataFrame
            with lock:
                df_copy.at[row_index, 'judged_success_original'] = success_orig
                df_copy.at[row_index, 'judge_notes_original'] = notes_orig
                df_copy.at[row_index, 'judged_success_modified'] = success_mod
                df_copy.at[row_index, 'judge_notes_modified'] = notes_mod
                
                # Save progress
                if completed % save_interval == 0:
                    filename = f"{OUTPUT_PATH}evaluation_progress_{completed}.csv"
                    df_copy.to_csv(filename, index=False)
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"💾 Saved progress: {completed} completed ({rate:.1f} rows/sec)")
    
    # Final save
    elapsed = time.time() - start_time
    rate = len(rows_to_process) / elapsed if elapsed > 0 else 0
    
    df_copy.to_csv(f"{OUTPUT_PATH}evaluation_final_results.csv", index=False)
    
    print(f"🎉 Evaluation completed!")
    print(f"⏱️  Time: {elapsed:.1f}s ({rate:.1f} rows/sec)")
    print(f"💾 Results saved to 'evaluation_final_results.csv'")
    
    return df_copy

# Main execution
if __name__ == "__main__":
    print("🔍 SIMPLE LLM EVALUATOR")
    print("=" * 30)
    
    # Check API key
    if OPENAI_API_KEY == "your-openai-api-key":
        print("❌ Set your OpenAI API key!")
        exit(1)
    
    # Load results
    df = pd.read_csv(INPUT_PATH)
    # print(len(df))
    # df = df.head(40)  # For testing, remove this line in production
        
    print(f"📊 Loaded {len(df)} rows")
    
    # Evaluate
    results = evaluate_dataframe_threaded(df, max_workers=50, save_interval=1000)
    
    # Quick stats
    orig_correct = (results['judged_success_original'] == True).sum()
    mod_correct = (results['judged_success_modified'] == True).sum()
    orig_unfinished = (results['judged_success_original'] == "unfinished").sum()
    mod_unfinished = (results['judged_success_modified'] == "unfinished").sum()
    total = len(results)

    print(f"\n📈 RESULTS:")
    print(f"Original - Correct: {orig_correct}/{total} ({orig_correct/total*100:.1f}%), Unfinished: {orig_unfinished}")
    print(f"Modified - Correct: {mod_correct}/{total} ({mod_correct/total*100:.1f}%), Unfinished: {mod_unfinished}")
    print("✅ Done!")








#             prompt = f"""Judge if the LLM correctly predicted the code output.

# Expected output: {expected_output}
# LLM prediction: {llm_output}

# Does the LLM's prediction match the expected output? Look for the exact number in their response.

# Answer with:
# - "CORRECT" if the LLM determined the right number ({expected_output})
# - "INCORRECT" if the LLM gave a wrong number or couldn't determine it
# - "UNFINISHED" if the LLM didn't finish its response

# Then explain your reasoning briefly."""