import os
import re
import json
import subprocess
from pathlib import Path
import concurrent.futures
from tabulate import tabulate

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

def main():
    script_dir = Path(__file__).parent.absolute()
    build_dir = script_dir / 'build'
    
    # Configure and build
    print("Configuring and building...")
    run_cmake(script_dir, build_dir)

    # Extract variants
    variants = extract_variants(script_dir / 'CMakeLists.txt')
    print(f"Detected variants: {' '.join(variants)}")

    # Run benchmarks in parallel
    print("\nStarting benchmarks...")
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_variant, build_dir, v) for v in variants]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.update(result)
                variant = next(iter(result))
                print("-" * 40)
                print(f"Results for {variant}:")
                print(json.dumps(result[variant], indent=2))
                print("-" * 40)
            except Exception as e:
                print(f"Error running variant: {e}")

    # Calculate best performance for relative comparisons
    best_cycles = min(r['cycles_per_token'] for r in results.values())
    
    # Prepare table data
    table_data = []
    for variant, result in sorted(results.items(), key=lambda x: x[1]['cycles_per_token']):
        mtps = result['tokens_per_s'] / 1_000_000
        pct_slower = (result['cycles_per_token'] / best_cycles - 1.0) * 100
        table_data.append([
            variant,
            f"{result['time']:.3f}",
            f"{mtps:.3f}",
            f"{result['cycles_per_token']:.1f}",
            f"{pct_slower:.1f}%",
            f"{result['total_tokens']:,d}"
        ])

    # Print table
    print("\nPerformance Summary")
    headers = ['Variant', 'Time (s)', 'MTok/s', 'Cycles/tok', '% slower', 'Total Tok']
    print(tabulate(table_data, headers=headers, tablefmt='psql'))

    # Validate token counts
    token_counts = {v: r['total_tokens'] for v, r in results.items()}
    if len(set(token_counts.values())) > 1:
        print("\nWARNING: Inconsistent token counts detected!")
        for variant, count in token_counts.items():
            print(f"{variant}: {count:,d} tokens")

if __name__ == '__main__':
    main()
