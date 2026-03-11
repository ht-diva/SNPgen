import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from torch.backends.cuda import SDPBackend

from functools import lru_cache
from typing import Optional, List, Union
    
try:
    from torch.backends.cuda import sdp_kernel
    SDP_IS_AVAILABLE = True
except:
    SDP_IS_AVAILABLE = False
    
try:
    from torch.nn.attention import sdpa_kernel
    SDPA_IS_AVAILABLE = True
except:
    SDPA_IS_AVAILABLE = False
    
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    
    
try:
    from tabulate import tabulate
    from torch.nn.attention.flex_attention import (
        _DEFAULT_SPARSE_BLOCK_SIZE,
        create_block_mask,
        create_mask,
        flex_attention,
        _score_mod_signature,
        _mask_mod_signature,
    )
    from triton.testing import do_bench
    IS_FLEX_ATTENTION_AVAILABLE = True
except:
    IS_FLEX_ATTENTION_AVAILABLE = False
    



# Adapted from https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html

# Lets define a helpful benchmarking function:
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


def test_attention_implementations(B=32, H=32, S=1024, D=32, dtype=torch.float32):
    print(f"\n\n########## Testing Attention Implementations [B: {B}, H: {H}, S: {S}, D: {D}, dtype: {dtype}] ##########")
    
    # Lets define the hyper-parameters of our input
    batch_size = B
    num_heads = H
    max_sequence_len = S
    embed_dimension = D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    query = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
    key = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
    value = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)

    query_xformers = query.permute(0, 2, 1, 3).contiguous()
    key_xformers = key.permute(0, 2, 1, 3).contiguous()
    value_xformers = value.permute(0, 2, 1, 3).contiguous()

    print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")

    # Helpful arguments mapper
    backend_map = {
        SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False, "enable_cudnn": False},
        SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False, "enable_cudnn": False},
        SDPBackend.EFFICIENT_ATTENTION: {"enable_math": False, "enable_flash": False, "enable_mem_efficient": True, "enable_cudnn": False},
        SDPBackend.CUDNN_ATTENTION: {"enable_math": False, "enable_flash": False, "enable_mem_efficient": False, "enable_cudnn": True}
    }
    
    if SDP_IS_AVAILABLE:

        with sdp_kernel(**backend_map[SDPBackend.MATH]):
            print(f"[SDP] The math implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")

        with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
            try:
                print(f"[SDP] The flash attention implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
            except RuntimeError:
                print("[SDP] FlashAttention is not supported. See warnings for reasons.")

        with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
            try:
                print(f"[SDP] The memory efficient implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
            except RuntimeError:
                print("[SDP] EfficientAttention is not supported. See warnings for reasons.")
                
        with sdp_kernel(**backend_map[SDPBackend.CUDNN_ATTENTION]):
            try:
                print(f"[SDP] The cuDNN implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
            except RuntimeError:
                print("[SDP] CudnnAttention is not supported. See warnings for reasons.")
                
        with sdp_kernel():
            try:
                print(f"[SDP] The automatic selection implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
            except RuntimeError:
                print("[SDP] Automatic attention selection is not supported. See warnings for reasons.")

    # Also test xformers implementation if available
    if XFORMERS_IS_AVAILABLE:
        try:
            print(f"[XFORMERS] The xformers implementation runs in {benchmark_torch_function_in_microseconds(xformers.ops.memory_efficient_attention, query_xformers, key_xformers, value_xformers):.3f} microseconds")
        except RuntimeError:
            print("[XFORMERS] xformers is not supported. See warnings for reasons.")
    else:
        print("[XFORMERS] xformers is not installed")
        
        
        
    if SDPA_IS_AVAILABLE:
        # Test the sdpa_kernel, which is the new API for the scaled dot product attention

        with sdpa_kernel(SDPBackend.MATH):
            print(f"[SDPA] The math implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            try:
                print(f"[SDPA] The flash attention implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
            except RuntimeError:
                print("[SDPA] FlashAttention is not supported. See warnings for reasons.")

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            try:
                print(f"[SDPA] The memory efficient implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
            except RuntimeError:
                print("[SDPA] EfficientAttention is not supported. See warnings for reasons.")
                
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            try:
                print(f"[SDPA] The cudnn implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
            except RuntimeError:
                print("[SDPA] CudnnAttention is not supported. See warnings for reasons.")
                
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
            try:
                print(f"[SDPA] The automatic selection implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")
            except RuntimeError:
                print("[SDPA] Automatic attention selection is not supported. See warnings for reasons.")
                
        torch.cuda.empty_cache()
                
def test_masked_attention_implementations(B=16, H=8, S=2048, D=32, dtype=torch.float32):
    print(f"\n\n########## Testing Masked Attention Implementations [B: {B}, H: {H}, S: {S}, D: {D}, dtype: {dtype}] ##########")
           
    def generate_sliding_window(window_size: int) -> _mask_mod_signature:
        """Generates a symmetric sliding window attention mask with a given window size.
        Args:
            window_size: The total size of the sliding window.
                        Each position can attend to (window_size//2) tokens before and after it.
        """
        def sliding_window(b, h, q_idx, kv_idx):
            distance = torch.abs(q_idx - kv_idx)
            return distance <= window_size // 2

        sliding_window_mask = sliding_window
        sliding_window_mask.__name__ = f"sliding_window_{window_size}"
        return sliding_window_mask

    torch.set_default_device("cuda")
    torch.manual_seed(0)

    torch._dynamo.reset()
    torch._dynamo.config.cache_size_limit = 1000

    # Compile the flex_attention function
    flex_attention_c = torch.compile(flex_attention, dynamic=False)

    # For better performance, you can use:
    # flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

    data_type = dtype

    # The kernels will utilize block sparsity to increase performance
    print(f"Using the default sparsity block size: {_DEFAULT_SPARSE_BLOCK_SIZE}")


    @lru_cache
    def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
        block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
        return block_mask
    
    create_block_mask_cached.cache_clear()


    def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
        return multiplier * flops * (1e3 / time_ms) / 1e12


    def print_header(text):
        width = 91
        print("╔" + "═" * (width - 2) + "╗")
        print(f"║ {text.center(width - 4)} ║")
        print("╚" + "═" * (width - 2) + "╝")
        

    def test_mask(
        score_mod: Optional[_score_mod_signature] = None,
        mask_mod: Optional[_mask_mod_signature] = None,
        B: int = B,
        H: int = H,
        S: int = S,
        D: int = D,
        skip_correctness: bool = False,
        print_mask: bool = True,
        device: str = "cuda",
    ):
        assert score_mod is not None or mask_mod is not None, "Must provide a score_mod or mask_mod"
        if mask_mod is not None:
            block_mask = create_block_mask_cached(mask_mod, 1, 1, S, S, device=device)
        else:
            block_mask = None
        sdpa_mask_fn = mask_mod if mask_mod is not None else score_mod
        mask = create_mask(sdpa_mask_fn, 1, 1, S, S, device=device)

        qkv = [
            torch.randn(B, H, S, D, device=device, dtype=data_type, requires_grad=True)
            for _ in range(3)
        ]
        gradOut = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        sdpa_mask = lambda: F.scaled_dot_product_attention(*qkv, attn_mask=mask)
        flex_attention_call = lambda: flex_attention_c(*qkv, score_mod=score_mod, block_mask=block_mask)

        results = []
        if block_mask is not None:
            density = (100 - block_mask.sparsity()) / 100
        else:
            density = 1.0
        flops = density * B * H * D * S * S

        times = []
        for attn in (sdpa_mask, flex_attention_call):
            fwd_time = do_bench(attn)
            fwd_out = attn()
            bwd_time = do_bench(lambda: fwd_out.backward(gradOut, retain_graph=True))  # noqa: F821
            times.append((fwd_time, bwd_time))

            del fwd_out
            torch.cuda.empty_cache()

        print_header(
            f"{score_mod.__name__ if score_mod is not None else mask_mod.__name__}".replace(
                "_", " "
            ).title()
        )
        # Inline correctness check
        if not skip_correctness:
            sdpa_mask_outs = []
            flex_outs = []

            for tensor in qkv:
                tensor.grad = None

            out1 = sdpa_mask()
            sdpa_mask_outs.append(out1)
            out1.backward(gradOut)
            sdpa_mask_outs += [tensor.grad for tensor in qkv]

            for tensor in qkv:
                tensor.grad = None

            out2 = flex_attention_call()
            flex_outs.append(out2)
            out2.backward(gradOut)
            flex_outs += [tensor.grad for tensor in qkv]
            for flex, sdpa_mask in zip(flex_outs, sdpa_mask_outs):
                torch.testing.assert_close(flex, sdpa_mask, atol=1e-1, rtol=1e-2)

            print("Correctness check passed ✅")

        (
            (sdpa_mask_time, sdpa_mask_bw_time),
            (flex_ms, flex_bw_ms),
        ) = times
        # Usage in your results formatting:
        results = [
            [
                "F.sdpa + mask",
                f"{sdpa_mask_time:.4f}",
                f"{calculate_tflops(flops, sdpa_mask_time, 4):.2f}",
                f"{sdpa_mask_bw_time:.4f}",
                f"{calculate_tflops(flops, sdpa_mask_bw_time, 10):.2f}",
            ],
            [
                "flexattention",
                f"{flex_ms:.4f}",
                f"{calculate_tflops(flops, flex_ms, 4):.2f}",
                f"{flex_bw_ms:.4f}",
                f"{calculate_tflops(flops, flex_bw_ms, 10):.2f}",
            ],
        ]
        print(
            tabulate(
                results,
                headers=[
                    "Operation",
                    "FW Time (ms)",
                    "FW FLOPS (TF/s)",
                    "BW Time (ms)",
                    "BW FLOPS (TF/s)",
                ],
                tablefmt="grid",
            )
        )
        if print_mask:
            print(f"\nBlock Mask:\n{block_mask}")
      
    AVAILABLE_EXAMPLES = {
        "sliding_window": lambda: test_mask(mask_mod=generate_sliding_window(window_size=5), S=20000),
    }
            
    for example in AVAILABLE_EXAMPLES.values():
        example()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    test_attention_implementations(dtype=torch.float32)
    test_attention_implementations(dtype=torch.float16)

    if IS_FLEX_ATTENTION_AVAILABLE:
        test_masked_attention_implementations(dtype=torch.float32)
        # test_masked_attention_implementations(dtype=torch.float16)
    else:
        print("\n\n[FlexAttention] FlexAttention is not available. You should probably install a more recent version of PyTorch.")
    



