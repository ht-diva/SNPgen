import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
import random

try:
    import amp_C
    APEX_IS_AVAILABLE = True
except Exception:
    APEX_IS_AVAILABLE = False

'''
Benchmark the performance of different implementations of the exponential moving average (EMA) operation.
'''

# Lets define a helpful benchmarking function:
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


def average_runs(N, f, *args, **kwargs):
    total_time = 0.
    for n in range(N):
        total_time += benchmark_torch_function_in_microseconds(f, *args, **kwargs)
    average_time = total_time / N
    return average_time


def get_random_w(device=None):
    dims = random.randint(1, 3)
    if dims == 1:
        dim_1_choice = [1, 2, 3, 32, 64, 128, 192, 256, 384, 512, 1000, 1024, 2048]
        w = torch.randn(random.choice(dim_1_choice))
    elif dims == 2:
        dim_1_choice = [2, 64, 128, 256, 512, 1024, 2048]
        dim_2_choice = [64, 128, 256, 512, 1024]
        w = torch.randn(random.choice(dim_1_choice), random.choice(dim_2_choice))
    elif dims == 3:
        dim_1_choice = [1, 2, 3, 32, 64, 128, 256]
        dim_2_choice = [1, 3, 32, 64, 128, 192, 256, 384, 512]
        dim_3_choice = [1, 3]
        w = torch.randn(random.choice(dim_1_choice), random.choice(dim_3_choice))
    if device is not None:
        return w.to(device)
    else:
        return w

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    num_w = 500
    model_weights = [get_random_w(device) for _ in range(num_w)]
    ema_model_weights = [p.detach().clone() * torch.randn_like(p) for p in model_weights]

    _overflow_buf = torch.IntTensor([0]).to(device)
    decay = 0.995

    N_runs = 5

    def base_implementation(ema_w, model_w, decay):
        for ema_v, model_v in zip(ema_w, model_w):
            ema_v.copy_(decay * ema_v + (1. - decay) * model_v)

    def lerp_implementation(ema_w, model_w, decay):
        for ema_v, model_v in zip(ema_w, model_w):
            ema_v.lerp_(model_v, weight=1. - decay)

    def foreach_mul_add(ema_w, model_w, decay):
        torch._foreach_mul_(ema_w, scalar=decay)
        torch._foreach_add_(ema_w, model_w, alpha=1. - decay)

    try:
        ema_model_weights1 = [p.clone() for p in ema_model_weights]
        print(f"The base implementation runs in {average_runs(N_runs, base_implementation, ema_model_weights1, model_weights, decay):.3f} microseconds")
    except RuntimeError:
        print("Error in base implementation")

    try:
        ema_model_weights2 = [p.clone() for p in ema_model_weights]
        print(f"The lerp_ implementation runs in {average_runs(N_runs, lerp_implementation, ema_model_weights2, model_weights, decay):.3f} microseconds")
    except RuntimeError:
        print("Error in base implementation")

    try:
        ema_model_weights3 = [p.clone() for p in ema_model_weights]
        print(f"The foreach_mul_add implementation runs in {average_runs(N_runs, foreach_mul_add, ema_model_weights3, model_weights, decay):.3f} microseconds")
    except RuntimeError:
        print("Error in foreach_mul_add implementation")

    if hasattr(torch, '_foreach_lerp_'):
        try:
            ema_model_weights4 = [p.clone() for p in ema_model_weights]
            print(f"The foreach_lerp implementation runs in {average_runs(N_runs, torch._foreach_lerp_, ema_model_weights4, model_weights, 1. - decay):.3f} microseconds")
        except RuntimeError:
            print("Error in foreach_lerp implementation")
    else:
        print("torch._foreach_lerp_ is not available")

    # Also test xformers implementation if available
    if APEX_IS_AVAILABLE:
        if torch.cuda.is_available():
            try:
                ema_model_weights5 = [p.clone() for p in ema_model_weights]
                print(f"The amp_C (apex) implementation runs in {average_runs(N_runs, amp_C.multi_tensor_axpby, 65536, _overflow_buf, [ema_model_weights5, model_weights, ema_model_weights5], decay, 1. - decay, -1):.3f} microseconds")
            except RuntimeError:
                print("Error in amp_C (apex) implementation")
        else:
            print("The amp_C (apex) implementation runs only on cuda devices.")
    else:
        print("apex is not installed: https://github.com/NVIDIA/apex#installation")

