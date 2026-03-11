import numpy as np
import time
import matplotlib.pyplot as plt

from .utils import _fast_assign_2d_scalar, _fast_assign_2d_non_scalar, _fast_assign_3d_scalar, _fast_assign_3d_non_scalar

def benchmark_functions(test_sizes, num_runs=10):
    results = {
        '2d_scalar': [],
        '2d_non_scalar': [],
        '3d_scalar': [],
        '3d_scalar_third_axis_idx': [],
        '3d_scalar_third_axis_idx_scalar': [],
        '3d_non_scalar': [],
        '3d_non_scalar_third_axis_idx': [],
        '3d_non_scalar_third_axis_idx_scalar': [],
    }
        
    for size1, size2, size3 in test_sizes:
        idx_size = int(size2 * 0.25)
        idx = np.sort(np.random.choice(np.arange(size2), size=idx_size, replace=False))
        third_axis_idx = np.arange(size3)
        third_axis_idx_scalar = -1
        
        # Prepare test data using np.random.randint
        x_2d = np.zeros((size1, size2), dtype=np.int32)
        x_3d = np.zeros((size1, size2, size3+1), dtype=np.int32) # simulate num_classes + 1
        x_3d_2 = np.zeros((size1, size2, size3), dtype=np.int32)
        
        y_1d = 1
        y_2d = np.random.randint(0, 3, (size1, len(idx)), dtype=np.int32)
        y_3d = np.random.randint(0, 3, (size1, len(idx), size3), dtype=np.int32)
        
        # Benchmark 2D scalar
        if '2d_scalar' in results:
            times_2d_scalar = []
            for _ in range(num_runs):
                x = x_2d.copy()
                start = time.time()
                _fast_assign_2d_scalar(x, y_1d, idx)
                times_2d_scalar.append(time.time() - start)
            results['2d_scalar'].append(np.mean(times_2d_scalar))
        
        # Benchmark 2D non-scalar
        if '2d_non_scalar' in results:
            times_2d_non_scalar = []
            for _ in range(num_runs):
                x = x_2d.copy()
                start = time.time()
                _fast_assign_2d_non_scalar(x, y_2d, idx)
                times_2d_non_scalar.append(time.time() - start)
            results['2d_non_scalar'].append(np.mean(times_2d_non_scalar))
        
        # Benchmark 3D scalar
        if '3d_scalar' in results:
            times_3d_scalar = []
            for _ in range(num_runs):
                x = x_3d.copy()
                start = time.time()
                _fast_assign_3d_scalar(x, y_1d, idx)
                times_3d_scalar.append(time.time() - start)
            results['3d_scalar'].append(np.mean(times_3d_scalar))
        
        # Benchmark 3D scalar with third_axis_idx
        if '3d_scalar_third_axis_idx' in results:
            times_3d_scalar_third_axis_idx = []
            for _ in range(num_runs):
                x = x_3d.copy()
                start = time.time()
                _fast_assign_3d_scalar(x, y_1d, idx, third_axis_idx)
                times_3d_scalar_third_axis_idx.append(time.time() - start)
            results['3d_scalar_third_axis_idx'].append(np.mean(times_3d_scalar_third_axis_idx))
        
        # Benchmark 3D scalar with third_axis_idx scalar
        if '3d_scalar_third_axis_idx_scalar' in results:
            times_3d_scalar_third_axis_idx_scalar = []
            for _ in range(num_runs):
                x = x_3d.copy()
                third_axis_idx_scalar_ok = np.array([third_axis_idx_scalar]) if np.isscalar(third_axis_idx_scalar) else third_axis_idx_scalar
                start = time.time()
                _fast_assign_3d_scalar(x, y_1d, idx, third_axis_idx_scalar_ok)
                times_3d_scalar_third_axis_idx_scalar.append(time.time() - start)
            results['3d_scalar_third_axis_idx_scalar'].append(np.mean(times_3d_scalar_third_axis_idx_scalar))
        
        # Benchmark 3D non-scalar
        if '3d_non_scalar' in results:
            times_3d_non_scalar = []
            for _ in range(num_runs):
                x = x_3d_2.copy()
                start = time.time()
                _fast_assign_3d_non_scalar(x, y_3d, idx)
                times_3d_non_scalar.append(time.time() - start)
            results['3d_non_scalar'].append(np.mean(times_3d_non_scalar))
        
        # Benchmark 3D non-scalar with third_axis_idx
        if '3d_non_scalar_third_axis_idx' in results:
            times_3d_non_scalar_third_axis_idx = []
            for _ in range(num_runs):
                x = x_3d.copy()
                start = time.time()
                _fast_assign_3d_non_scalar(x, y_3d, idx, third_axis_idx)
                times_3d_non_scalar_third_axis_idx.append(time.time() - start)
            results['3d_non_scalar_third_axis_idx'].append(np.mean(times_3d_non_scalar_third_axis_idx))
        
        # Benchmark 3D non-scalar with third_axis_idx scalar
        if '3d_non_scalar_third_axis_idx_scalar' in results:
            times_3d_non_scalar_third_axis_idx_scalar = []
            for _ in range(num_runs):
                x = x_3d.copy()
                third_axis_idx_scalar_ok = np.array([third_axis_idx_scalar]) if np.isscalar(third_axis_idx_scalar) else third_axis_idx_scalar
                start = time.time()
                _fast_assign_3d_non_scalar(x, y_3d, idx, third_axis_idx_scalar_ok)
                times_3d_non_scalar_third_axis_idx_scalar.append(time.time() - start)
            results['3d_non_scalar_third_axis_idx_scalar'].append(np.mean(times_3d_non_scalar_third_axis_idx_scalar))
    
    return results

def plot_benchmark_results(test_sizes, results):
    plt.figure(figsize=(12, 6))
    for func_name, times in results.items():
        plt.plot(str(test_sizes), times, marker='o', label=func_name)
    
    plt.title('Numba Function Performance')
    plt.xlabel('Array Size')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Run benchmarks
    test_sizes = [[10000, 20000, 3]]

    # Warmup numba functions
    warmup_test_sizes = [[4, test_sizes[0][1], test_sizes[0][2]]]
    _ = benchmark_functions(warmup_test_sizes, num_runs=1)
    benchmark_results = benchmark_functions(test_sizes, num_runs=5)

    # Print results
    for func, times in benchmark_results.items():
        print(f"{func} performance:")
        for size, time_taken in zip(test_sizes, times):
            print(f"  Size {size}: {time_taken:.6f} seconds")

    # Optional: Plot results
    plot_benchmark_results(test_sizes, benchmark_results)