import os
import re
import json
import argparse
import statistics
import subprocess
from pathlib import Path
import concurrent.futures
from tabulate import tabulate
from collections import defaultdict

def run_cmake(script_dir: Path, build_dir: Path) -> None:
    """Configure and build all variants."""
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(['cmake', '-S', script_dir, '-B', build_dir], check=True)
    subprocess.run(['cmake', '--build', build_dir, f'-j{os.cpu_count()}'], check=True)

def extract_variants(cmake_file: Path) -> list[str]:
    """Extract variant names from CMakeLists.txt."""
    with open(cmake_file) as f:
        content = f.read()
    return re.findall(r'"(__[^"]*)"', content)

def run_variant(build_dir: Path, variant: str) -> dict:
    """Run a single benchmark variant and return its output."""
    executable = build_dir / f'benchmark_cpp_fast{variant}'
    result = subprocess.run([executable], capture_output=True, text=True, check=True)
    return {variant: json.loads(result.stdout.splitlines()[-1].strip())}

def run_benchmark_round(build_dir: Path, variants: list[str]) -> dict:
    """Run all variants once in parallel and return results."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_variant, build_dir, v) for v in variants]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.update(result)
            except Exception as e:
                print(f"Error running variant: {e}")
    return results

def calculate_statistics(all_results: list[dict]) -> dict:
    """Calculate statistics across multiple runs."""
    stats = defaultdict(lambda: {'cycles': [], 'mtps': [], 'tokens': set()})
    
    for results in all_results:
        for variant, result in results.items():
            stats[variant]['cycles'].append(result['cycles_per_token'])
            stats[variant]['mtps'].append(result['tokens_per_s'] / 1_000_000)
            stats[variant]['tokens'].add(result['total_tokens'])
    
    final_stats = {}
    for variant, data in stats.items():
        final_stats[variant] = {
            'cycles_min': min(data['cycles']),
            'cycles_max': max(data['cycles']),
            'cycles_avg': statistics.mean(data['cycles']),
            'cycles_stdev': statistics.stdev(data['cycles']) if len(data['cycles']) > 1 else 0,
            'mtps_min': min(data['mtps']),
            'mtps_max': max(data['mtps']),
            'mtps_avg': statistics.mean(data['mtps']),
            'total_tokens': next(iter(data['tokens']))  # All runs should have same token count
        }
    return final_stats

def main():
    parser = argparse.ArgumentParser(description='Run tokenizer benchmarks')
    parser.add_argument('-n', type=int, default=3, help='Number of times to run each benchmark')
    args = parser.parse_args()

    script_dir = Path(__file__).parent.absolute()
    build_dir = script_dir / 'build'
    
    # Configure and build
    print("Configuring and building...")
    run_cmake(script_dir, build_dir)

    # Extract variants
    variants = extract_variants(script_dir / 'CMakeLists.txt')
    print(f"Detected variants: {' '.join(variants)}")

    # Run benchmarks n times
    print(f"\nStarting {args.n} rounds of benchmarks...")
    all_results = []
    for i in range(args.n):
        print(f"\nRound {i+1}/{args.n}")
        results = run_benchmark_round(build_dir, variants)
        all_results.append(results)
        
    # Calculate statistics
    stats = calculate_statistics(all_results)
    
    # Calculate relative performance based on average cycles
    best_avg_cycles = min(s['cycles_avg'] for s in stats.values())
    
    # Prepare table data
    table_data = []
    for variant, stat in sorted(stats.items(), key=lambda x: x[1]['cycles_avg']):
        pct_slower = (stat['cycles_avg'] / best_avg_cycles - 1.0) * 100
        table_data.append([
            variant,
            f"{stat['mtps_avg']:.3f}",
            f"{stat['mtps_min']:.3f}",
            f"{stat['mtps_max']:.3f}",
            f"{stat['cycles_avg']:.1f}",
            f"{stat['cycles_min']:.1f}",
            f"{stat['cycles_max']:.1f}",
            f"{stat['cycles_stdev']:.2f}",
            f"{pct_slower:.1f}%",
            f"{stat['total_tokens']:,d}"
        ])

    # Print table
    print(f"\nPerformance Summary ({args.n} runs)")
    headers = ['Variant', 'MTok/s avg', 'MTok/s min', 'MTok/s max', 
              'Cycles avg', 'Cycles min', 'Cycles max', 'Cycles σ', 
              '% slower', 'Total Tok']
    print(tabulate(table_data, headers=headers, tablefmt='psql'))

    # Validate token counts
    token_counts = {v: s['total_tokens'] for v, s in stats.items()}
    if len(set(token_counts.values())) > 1:
        print("\nWARNING: Inconsistent token counts detected!")
        for variant, count in token_counts.items():
            print(f"{variant}: {count:,d} tokens")

if __name__ == '__main__':
    main()
