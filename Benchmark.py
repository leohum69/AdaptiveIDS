import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import time

# --- CONFIGURATION ---
SCRIPT_TO_RUN = "Final_Parallel_Tester.py"  # Ensure this matches your filename exactly
NUM_RUNS = 10

def run_benchmark():
    execution_times = []
    throughputs = []
    
    print(f"🚀 STARTING STABILITY BENCHMARK ({NUM_RUNS} Runs)...")
    print("=" * 60)

    for i in range(1, NUM_RUNS + 1):
        print(f"   🔄 Run {i}/{NUM_RUNS}: Executing...", end="", flush=True)
        
        try:
            # Run the command and capture output
            result = subprocess.run(
                ["python3", SCRIPT_TO_RUN], 
                capture_output=True, 
                text=True
            )
            
            output = result.stdout
            
            # Regex to find the numbers in your specific output format
            # Looks for: "Real Time:       20.26 sec"
            time_match = re.search(r"Real Time:\s+([0-9.]+)\s+sec", output)
            # Looks for: "Throughput:      2450.50 pkts/sec"
            tput_match = re.search(r"Throughput:\s+([0-9.]+)\s+pkts/sec", output)
            
            if time_match and tput_match:
                exec_time = float(time_match.group(1))
                tput = float(tput_match.group(1))
                
                execution_times.append(exec_time)
                throughputs.append(tput)
                print(f" ✅ Done. (Time: {exec_time}s | Speed: {tput} p/s)")
            else:
                print(" ❌ Error parsing output. (Did the script crash?)")
                print("--- DEBUG OUTPUT ---")
                print(output[-500:]) # Show last 500 chars
                
        except Exception as e:
            print(f" ❌ System Error: {e}")

    return execution_times, throughputs

def generate_graphs(times, tputs):
    if not times:
        print("No data collected.")
        return

    runs = np.arange(1, len(times) + 1)
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Execution Time
    ax1.plot(runs, times, marker='o', linestyle='-', color='red', label='Real Time')
    ax1.set_title(f'Execution Time Stability ({len(times)} Runs)')
    ax1.set_ylabel('Time (Seconds)')
    ax1.set_xlabel('Run Number')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add average line
    avg_time = np.mean(times)
    ax1.axhline(y=avg_time, color='darkred', linestyle='--', label=f'Avg: {avg_time:.2f}s')
    ax1.legend()

    # Plot 2: Throughput
    ax2.plot(runs, tputs, marker='s', linestyle='-', color='green', label='Throughput')
    ax2.set_title(f'System Throughput Stability ({len(times)} Runs)')
    ax2.set_ylabel('Packets Per Second')
    ax2.set_xlabel('Run Number')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add average line
    avg_tput = np.mean(tputs)
    ax2.axhline(y=avg_tput, color='darkgreen', linestyle='--', label=f'Avg: {avg_tput:.0f} pkts/s')
    ax2.legend()

    plt.tight_layout()
    plt.savefig("Stability_Benchmark_Graph.png")
    print(f"\n📸 Graph saved as 'Stability_Benchmark_Graph.png'")

if __name__ == "__main__":
    times, tputs = run_benchmark()
    generate_graphs(times, tputs)
    
    # Final Stats for your Slides
    if times:
        print("\n📊 FINAL STATISTICS FOR PRESENTATION")
        print("-" * 40)
        print(f"Average Time:       {np.mean(times):.2f} seconds")
        print(f"Time Std Dev:       {np.std(times):.2f} (Lower is more stable)")
        print(f"Average Throughput: {np.mean(tputs):.2f} pkts/sec")
        print("-" * 40)
