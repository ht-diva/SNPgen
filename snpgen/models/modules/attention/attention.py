from packaging import version
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint

from einops import rearrange

from snpgen.utils import get_pylogger, print_once, exists, default, supports_flash_attention
from snpgen.models.modules.utils import (
    conv_nd,
    zero_module,
    normalization,
    Normalize
)

logpy = get_pylogger(__name__)

try:
    from flash_attn.ops.rms_norm import RMSNorm # type: ignore
except:
    logpy.warn("FlashAttention RMSNorm not available, loading default RMSNorm")
    from ..utils import RMSNorm

# Check, import and setup attention backends
if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},
    }
    
    BACKEND_NAME_MAP = {
        'math': SDPBackend.MATH,
        'flash': SDPBackend.FLASH_ATTENTION,
        'mem_efficient': SDPBackend.EFFICIENT_ATTENTION,
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}
    BACKEND_NAME_MAP = {}
    logpy.warn(
        f"No SDP backend available, likely because you are running in pytorch "
        f"versions < 2.0. In fact, you are using PyTorch {torch.__version__}. "
        f"You might want to consider upgrading."
    )
      
try:
    from torch.nn.attention.flex_attention import flex_attention
    from .utils import create_block_mask_cached, generate_sliding_window
    torch._dynamo.config.cache_size_limit = 1000
    flex_attention_c = torch.compile(flex_attention, dynamic=False)
    FLEX_ATTENTION_IS_AVAILABLE = True
except:
    logpy.warn("FlexAttention not available")
    FLEX_ATTENTION_IS_AVAILABLE = False
    
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
    BACKEND_NAME_MAP.update({'xformers': 'xformers'})
except:
    logpy.warn("xformers not available, using default PyTorch attention")
    XFORMERS_IS_AVAILABLE = False


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        dropout=0.,
        xformers=True,
        use_bias=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1, bias=use_bias)

        if xformers and XFORMERS_IS_AVAILABLE:
            self.attention = AttentionXformers(self.num_heads, dropout=dropout, channel_last=False)
        else:
            self.attention = AttentionTorch(self.num_heads, dropout=dropout, channel_last=False)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1, bias=use_bias))

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)
        
    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim = 1)
        h = self.attention(q,k,v)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class Attention2D(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., xformers = True, use_bias=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads

        self.norm = RMSNorm(dim) # nn.LayerNorm(dim)

        if xformers and XFORMERS_IS_AVAILABLE:
            self.attention = AttentionXformers(heads=heads, dropout=dropout, channel_last=False)
        else:
            self.attention = AttentionTorch(heads=heads, dropout=dropout, channel_last=False)

        #self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        #self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=use_bias)
        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias=use_bias)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim = 1)
        out = self.attention(q,k,v)
        return self.to_out(out)
    

#################################
#       Attention Classes       #
#################################
class Attention(nn.Module):
    def __new__(cls, *args, **kwargs):
        if cls is Attention:
            raise TypeError(f"You are trying to instantiate an abstract class {cls.__name__}. Please use a concrete subclass.")
        return super().__new__(cls)
    
    def __init__(self, heads, dropout=0., channel_last=False):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.channel_last = channel_last
        self.dims_pattern = None
        
    @property
    def attend(self):
        # Lazy instantiation: cache the module after first creation.
        if not hasattr(self, '_attend'):
            self._attend = self.getAttend()
        return self._attend
        
    def getAttend(self):
        raise NotImplementedError("getAttend method of abstract base class called")

    def forward(self, q, k, v, mask=None):
        '''
        q, k, v: [B, C, ...] or [B, ..., C] depending on self.channel_last
        q/k/v can have and arbitrary number of spatial dimensions
        '''
        
        if self.channel_last:
            b, *spatial_dims, c  = q.shape
        else:
            b, c, *spatial_dims = q.shape

        if self.dims_pattern is None:
            self.dims_pattern = ' '.join([f'x{i}' for i in range(len(spatial_dims))])

        # Reshape q, k, v to the expected format for the actual attention operation
        q, k, v = map(lambda t: rearrange(t, self.pre_attend_pattern, h = self.heads), (q,k,v))
        # Compute attention
        out = self.attend(q, k, v, mask=mask)
        # Reshape the output to the original input format
        out = rearrange(out, self.post_attend_pattern.format(dims_pattern=self.dims_pattern), **{f'x{i}': spatial_dims[i] for i in range(len(spatial_dims))})

        return out
        
class AttentionTorch(Attention):
    def __init__(self, heads, dropout=0., channel_last=False, sdp_backend=None):
        super().__init__(heads, dropout, channel_last)
        self.sdp_backend = sdp_backend
        
        # whether the num_heads*head_dim dimension is the first or the last
        if self.channel_last:
            self.pre_attend_pattern = 'b ... (h d) -> b h (...) d'
            self.post_attend_pattern = 'b h ({dims_pattern}) d -> b {dims_pattern} (h d)'
        else:
            self.pre_attend_pattern = 'b (h d) ... -> b h (...) d'
            self.post_attend_pattern = 'b h ({dims_pattern}) d -> b (h d) {dims_pattern}'
            
    def getAttend(self):
        return AttendTorch(dropout=self.dropout, sdp_backend=self.sdp_backend)
    
    
class AttentionFlex(AttentionTorch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_mod = None
        self.mask_mod = None
    
    def getAttend(self):
        return AttendFlex(score_mod=self.score_mod, mask_mod=self.mask_mod, dropout=self.dropout)

class WindowedAttention(AttentionFlex):
    def __init__(self, window_size=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_mod = generate_sliding_window(window_size=window_size)


class AttentionXformers(Attention):
    def __init__(self, heads, dropout=0., channel_last=False):
        super().__init__(heads, dropout, channel_last)

        # whether the num_heads*head_dim dimension is the first or the last
        if self.channel_last:
            self.pre_attend_pattern = 'b ... (h d) -> b (...) h d'
            self.post_attend_pattern = 'b ({dims_pattern}) h d -> b {dims_pattern} (h d)'
        else:
            self.pre_attend_pattern = 'b (h d) ... -> b (...) h d'
            self.post_attend_pattern = 'b ({dims_pattern}) h d -> b (h d) {dims_pattern}'
            
    def getAttend(self):
        return AttendXformers(dropout=self.dropout)


##################################
#         Attend Classes         #
##################################
class Attend(nn.Module):
    def __init__(self, dropout = 0.):
        super().__init__()
        self.dropout = dropout

    def attend(self, q, k, v, mask=None):
        raise NotImplementedError("`attend` method of abstract base class called")

    def forward(self, q, k, v, mask=None):
        return self.attend(q, k, v, mask=mask)
 
       
class AttendMath(Attend):
    def __init__(self, dropout = 0.):
        super().__init__(dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
    
    def attend(self, q, k, v, mask=None):
        """
        Manual implementation of scaled dot product attention
        
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        
        if exists(mask):
            raise NotImplementedError("Masking is not implemented for MathAttention")
    
        scale = q.shape[-1] ** -0.5
        # similarity
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        # attention
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)
        # aggregate values
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        return out


class AttendTorch(Attend):
    def __init__(self, dropout = 0., sdp_backend=None):
        super().__init__(dropout=dropout)
        self.sdp_backend = sdp_backend

        assert sdp_backend is None or isinstance(sdp_backend, SDPBackend), "sdp_backend must be either None or an instance of SDPBackend"
        
        # default to all backends enabled (also, it's the config for CPU)
        self.backend_config = BACKEND_MAP[None] 
        self.cpu_backend_config = BACKEND_MAP[None]

        if SDP_IS_AVAILABLE and torch.cuda.is_available():
            self.backend_config = BACKEND_MAP[sdp_backend]

            # Check if we can actually use Flash Attention
            if self.backend_config['enable_flash']:
                if supports_flash_attention(device_id=0):
                    print_once('Supported GPU detected, using Flash Attention')
                else:
                    print_once('Unsupported GPU detected, Flash Attention can not be used. Falling back to Mem Efficient Attention')
                    self.backend_config = BACKEND_MAP[SDPBackend.EFFICIENT_ATTENTION]

    def attend(self, q, k, v, mask=None):
        '''
        Input tensors must be in format [B, H, M, K], where
            - B is the batch size,
            - M the sequence length,
            - H the number of heads,
            - K the embeding size per head
        If inputs have dimension 3, it is assumed that the dimensions are [B, M, K] and H=1
        '''
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        
        backend_config = self.backend_config if q.is_cuda else self.cpu_backend_config

        with sdp_kernel(**backend_config):
            # print("dispatching into backend", self.sdp_backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.
            ) # scale is dim_head ** -0.5 per default
        del q, k, v

        return out
    
class AttendFlex(Attend):
    def __init__(self, score_mod=None, mask_mod=None, dropout = 0.):
        super().__init__(dropout=dropout)
        if dropout != 0:
            logpy.warn("Dropout is not supported for the FlexAttention implementation")
        self.score_mod = score_mod
        self.mask_mod = mask_mod
            
    def attend(self, q, k, v, mask=None):
        if exists(mask):
            raise NotImplementedError("Manual masking is not implemented for FlexAttention. Please use the mask_mod argument")
        
        Q_LEN = q.shape[-2] # q: [B, H, Q_LEN, D]
        KV_LEN = k.shape[-2] # k: [B, H, KV_LEN, D]
        block_mask = create_block_mask_cached(self.mask_mod, 1, 1, Q_LEN, KV_LEN, device=q.device)
        
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        
        # TODO: necessary otherwise the compilation fails. Probably a bug in the FlexAttention implementation which eventually will be fixed
        q, k, v = map(lambda t: t.to(torch.float32), (q, k, v))
        
        out = flex_attention_c(q, k, v, score_mod=self.score_mod, block_mask=block_mask)
        del q, k, v

        return out


class AttendXformers(Attend):
    def __init__(self, dropout = 0.):
        super().__init__(dropout=dropout)

    def attend(self, q, k, v, mask=None):
        '''
        Input tensors must be in format [B, M, H, K], where
            - B is the batch size,
            - M the sequence length,
            - H the number of heads,
            - K the embeding size per head
        If inputs have dimension 3, it is assumed that the dimensions are [B, M, K] and H=1
        '''
        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError("Masking is not implemented for xformers attention")
        
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
            
        # If batch size >= 65536 (unlikely...), use this workaround:
        # https://github.com/Stability-AI/generative-models/blob/9d759324e914de6c96dbd1468b3a4a50243c6528/sgm/modules/attention.py#L417
        out = xformers.ops.memory_efficient_attention(
            q, k, v,
            attn_bias=None,
            p=self.dropout if self.training else 0.
        ) # scale is dim_head ** -0.5 per default
        del q, k, v
        
        return out
    

class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
        window_size=None,
    ):
        super().__init__()
        logpy.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, "
            f"context_dim is {context_dim} and using {heads} heads with a "
            f"dimension of {dim_head}."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.inner_dim = inner_dim   

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend
        self.window_size = window_size
        
        if isinstance(backend, str):
            backend = BACKEND_NAME_MAP[backend]
            
        if window_size is not None:
            if not FLEX_ATTENTION_IS_AVAILABLE:
                raise NotImplementedError("FlexAttention is not available")
            self.attention = WindowedAttention(window_size=window_size, heads=heads, dropout=0.0, channel_last=True)
        elif backend is None or isinstance(backend, SDPBackend):
            self.attention = AttentionTorch(heads=heads, dropout=0.0, channel_last=True, sdp_backend=backend)
        elif backend == 'xformers' and XFORMERS_IS_AVAILABLE:
            self.attention = AttentionXformers(heads=heads, dropout=0.0, channel_last=True)
        else:
            raise ValueError(f"Unknown attention backend '{backend}'")

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        # print(f"query dim: {self.query_dim}, context dim: {self.context_dim}, inner dim: {self.inner_dim}")
        # print(f"x shape: {x.shape}, context shape: {context.shape}")
        # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        # print("")
        out = self.attention(q, k, v, mask=mask)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_BACKENDS = [
        None,
        'xformers',
        SDPBackend.MATH,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
    ] + list(BACKEND_NAME_MAP.keys())

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        single_layer=False,
        backend=None,
        window_size=None,
    ):
        super().__init__()
        assert backend in self.ATTENTION_BACKENDS, f"Invalid attention backend '{backend}'"
        if backend == "xformers" and not XFORMERS_IS_AVAILABLE:
            logpy.warn(
                f"Backend '{backend}' is not available. Falling "
                f"back to native attention. This is not a problem in "
                f"Pytorch >= 2.0. FYI, you are running with PyTorch "
                f"version {torch.__version__}."
            )
            backend = None
        elif backend != "xformers" and not SDP_IS_AVAILABLE:
            logpy.warn(
                "We do not support vanilla attention anymore, as it is too "
                "expensive. Sorry."
            )
            if not XFORMERS_IS_AVAILABLE:
                assert (
                    False
                ), "Please install xformers via e.g. 'pip install xformers'"
            else:
                logpy.info("Falling back to xformers efficient attention.")
                backend = "xformers"

        self.single_layer = single_layer
        self.disable_self_attn = disable_self_attn

        self.attn1 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim if self.disable_self_attn else None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            backend=backend,
            window_size=window_size
        )  # is a self-attention if self.disable_self_attn is False
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        if not self.single_layer:
            self.attn2 = CrossAttention(
                query_dim=dim,
                context_dim=context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                backend=backend,
                window_size=window_size
            )  # is self-attn if context is none
            self.norm3 = nn.LayerNorm(dim)

        self.checkpoint = checkpoint
        if self.checkpoint:
            logpy.debug(f"{self.__class__.__name__} is using checkpointing")

    def forward(self, x, context=None):
        if self.checkpoint:
            return checkpoint(self._forward, x, context, use_reentrant=False)
        else:
            return self._forward(x, context)

    def _forward(self, x, context=None):
        if self.single_layer:
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
            x = self.ff(self.norm2(x)) + x
        else:
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x)) + x
        return x



class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        dims=2,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
        single_layer=False,
        backend=None,
        window_size=None,
    ):
        super().__init__()
        logpy.debug(
            f"constructing {self.__class__.__name__} of depth {depth} w/ "
            f"{in_channels} channels and {n_heads} heads."
        )

        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                logpy.warn(
                    f"{self.__class__.__name__}: Found context dims "
                    f"{context_dim} of depth {len(context_dim)}, which does not "
                    f"match the specified 'depth' of {depth}. Setting context_dim "
                    f"to {depth * [context_dim[0]]} now."
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth

        self.use_linear = use_linear
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        if not use_linear:
            self.proj_in = conv_nd(
                dims, in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    single_layer=single_layer,
                    backend=backend,
                    window_size=window_size,
                )
                for d in range(depth)
            ]
        )

        if not use_linear:
            self.proj_out = zero_module(
                conv_nd(dims, inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            # self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]

        b, c, *spatial_dims = x.shape # let's support any number of spatial dimensions (1D, 2D, 3D, ...)
        dims_pattern = ' '.join([f'x{i}' for i in range(len(spatial_dims))])
        #print(f"[SpatialTransformer] Input shape: {x.shape}, spatial_dims: {spatial_dims}")

        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)

        x = rearrange(x, "b c ... -> b (...) c").contiguous()
        #print(f"[SpatialTransformer] After flatten: {x.shape} (seq_len: {x.shape[1]})")
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, f"b ({dims_pattern}) c -> b c {dims_pattern}", **{f'x{i}': spatial_dims[i] for i in range(len(spatial_dims))}).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in