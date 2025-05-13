import asyncio
import random
import json
import re
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import litellm
import unifai
import argparse
from dotenv import load_dotenv

load_dotenv()

MODEL = "anthropic/claude-3-7-sonnet-20250219"
INPUT_TOKEN_PRICE = 3 / 1000000

# Configure litellm
litellm.set_verbose=False

async def get_all_toolkit_ids():
    """Get all toolkit IDs from UnifAI."""
    tools = unifai.Tools(api_key="")
    available_actions = await tools._api.search_tools(arguments={"limit": 100})
    
    toolkit_ids = set()
    for action in available_actions:
        action_id = action.get("action", "")
        if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', action_id):
            print(f"Skipping action ID: {action_id}")
            continue
        parts = action_id.split("--")
        if len(parts) > 1:
            toolkit_ids.add(parts[1])
    
    return list(toolkit_ids)

async def run_benchmark(n_toolkits, toolkit_ids, results_file="benchmark_results.jsonl"):
    """Run a single benchmark with n toolkits."""
    # Randomly select n toolkits
    selected_toolkits = random.sample(toolkit_ids, n_toolkits)
    
    # Get tools for these toolkits
    tools = unifai.Tools(api_key="")
    tool_list = await tools.get_tools(
        static_toolkits=selected_toolkits
    )
    
    n_tools = len(tool_list)

    # Make a call to Claude 3.7 with these tools
    start_time = time.time()
    response = await litellm.acompletion(
        model=MODEL,
        messages=[
            {"role": "system", "content": "echo the user message"},
            {"role": "user", "content": "hi"}
        ],
        tools=tool_list
    )
    end_time = time.time()
    
    # Extract token usage and response time
    response_time = end_time - start_time
    input_tokens = response.usage.prompt_tokens
    
    # Store result
    result = {
        "n_toolkits": n_toolkits,
        "n_tools": n_tools,
        "input_tokens": input_tokens,
        "response_time": response_time
    }
    
    # Append to results file
    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")
    
    return result

def plot_results(results_file="benchmark_results.jsonl", show=False):
    """Plot the results with linear regression."""
    # Read all results
    results = []
    with open(results_file, "r") as f:
        for line in f:
            results.append(json.loads(line))
    
    # Extract data for plotting
    n_tools = [r["n_tools"] for r in results]
    input_tokens = [r["input_tokens"] for r in results]
    costs = [r["input_tokens"] * INPUT_TOKEN_PRICE for r in results]
    response_times = [r["response_time"] for r in results]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Set global font sizes - approximately 2x larger
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 14,
        'figure.titlesize': 22
    })
    
    # Plot 1: Input tokens vs number of tools
    ax1.scatter(n_tools, costs)
    ax1.set_xlabel("Number of Tools", fontsize=18)
    ax1.set_ylabel("Cost Overhead Per Query (USD)", fontsize=18)
    ax1.set_title("Cost Overhead vs Number of Tools", fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    
    # Linear regression for costs
    z = np.polyfit(n_tools, costs, 1)
    p = np.poly1d(z)
    ax1.plot(n_tools, p(n_tools), "g--", alpha=0.8)
    # Add arrow at x=2 pointing to regression line
    ax1.annotate("Dynamic\nTools", xy=(2, p(2)), xytext=(5, p(2)+0.035), 
                 arrowprops=dict(facecolor='red', edgecolor='red', shrink=0.05), 
                 color='red', ha='center', fontsize=16)
    # ax1.text(0.05, 0.95, f"y = {z[0]:.2f}x + {z[1]:.2f}", transform=ax1.transAxes)
    
    # Plot 2: Response time vs number of tools
    ax2.scatter(n_tools, response_times)
    ax2.set_xlabel("Number of Tools", fontsize=18)
    ax2.set_ylabel("Response Time (s)", fontsize=18)
    ax2.set_title("Response Time vs Number of Tools", fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    
    # Linear regression for response time
    z = np.polyfit(n_tools, response_times, 1)
    p = np.poly1d(z)
    ax2.plot(n_tools, p(n_tools), "g--", alpha=0.8)
    # Add arrow at x=2 pointing to regression line
    ax2.annotate("Dynamic\nTools", xy=(2, p(2)), xytext=(5, p(2)+2.5), 
                 arrowprops=dict(facecolor='red', edgecolor='red', shrink=0.05), 
                 color='red', ha='center', fontsize=16)
    # ax2.text(0.05, 0.95, f"y = {z[0]:.2f}x + {z[1]:.2f}", transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    if show:
        plt.show()

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run benchmarks for tool call costs')
    parser.add_argument('-n', type=int, default=10, help='Number of benchmark runs to perform')
    args = parser.parse_args()
    
    # Get all toolkit IDs
    toolkit_ids = await get_all_toolkit_ids()
    N = len(toolkit_ids)
    print(f"Found {N} total toolkits")
    
    # Initialize or clear results file
    results_file = "benchmark_results.jsonl"
    if os.path.exists(results_file):
        # Read existing results first
        print("Found existing results...")
        # plot_results(results_file)
    else:
        with open(results_file, "w") as f:
            pass
    
    # Run benchmarks
    num_runs = args.n
    progress_bar = tqdm(range(num_runs))
    for _ in progress_bar:
        # Choose a random n between 1 and N
        n_toolkits = random.randint(0, N)
        result = await run_benchmark(n_toolkits, toolkit_ids, results_file)
        plot_results(results_file, show=False)
        progress_bar.write(f"Run with {n_toolkits} toolkits, {result['n_tools']} tools: {result['input_tokens']} tokens, {result['response_time']:.2f}s")
    
    plot_results(results_file, show=False)

if __name__ == "__main__":
    asyncio.run(main()) 