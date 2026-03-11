from functools import lru_cache
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    create_block_mask,
    create_mask,
)

@lru_cache
def create_block_mask_cached(mask_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(mask_mod, B, H, M, N, device=device)
    return block_mask

def generate_sliding_window(window_size: int) -> _mask_mod_signature:
    """Generates a sliding window attention mask.
    
    For even window_size, each token attends to (window_size//2) tokens before
    and after it. For odd window_size, each token attends to one extra previous token.
    
    Args:
        window_size: Total size of the sliding window.
    """
    # Determine number of previous and future tokens to attend to
    if window_size % 2 == 0:
        prev = window_size // 2
        fut = window_size // 2
    else:
        prev = window_size // 2 + 1  # one extra previous token
        fut = window_size // 2

    def sliding_window(b, h, q_idx, kv_idx):
        distance = q_idx - kv_idx
        return ((distance) <= prev) & ((-distance) <= fut)

    sliding_window.__name__ = f"sliding_window_{window_size}"
    return sliding_window

def print_sliding_window_mask(window_size, Q_LEN, KV_LEN, device='cpu'):
    from rich.console import Console
    console = Console(width=300)
    
    mask = generate_sliding_window(window_size=window_size)
    mask = create_mask(mask, 1, 1, Q_LEN, KV_LEN, device=device)
    
    mask_to_print = mask.squeeze().cpu().tolist()

    for i, row in enumerate(mask_to_print):
        row_str = []
        for j, element in enumerate(row):
            if i == j:
                row_str.append(f"[bold blue]{element}[/bold blue]")
            else:
                row_str.append(str(element))
        console.print(" ".join(row_str))