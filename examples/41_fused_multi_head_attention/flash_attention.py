
import math
import torch
import sys
import os
import ctypes

FWD_LIB_PATH="/home/yuqxia/repo/cutlass/build/examples/41_fused_multi_head_attention/libfmha_fwd.so"
BWD_LIB_PATH="/home/yuqxia/repo/cutlass/build/examples/41_fused_multi_head_attention/libfmha_bwd.so"

class FlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, bias, causal=False, softmax_scale=None):
        """
            q: (batch_size, seqlen_q, nheads, headdim)
            k, v: (batch_size, seqlen_k, nheads, headdim)
            bias: (batch, nheads, seqlen_q, seqlen_k).
        """
        # Make sure that the last dimension is contiguous

        q, k, v, bias = [x.contiguous() for x in [q, k, v, bias]]
        batch_size, seqlen_q, nheads, head_dim = q.shape
        seqlen_kv = k.shape[1]
        head_dim_value = v.shape[-1]
        softmax_scale = 1 / q.shape[-1] ** 0.5
        lse = torch.zeros((batch_size, nheads, seqlen_q), device=q.device, dtype=torch.float32)
        o = torch.empty_like(q)
        lib = ctypes.cdll.LoadLibrary(FWD_LIB_PATH)
        lib.flash_attn_fwd.argstypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_bool]   
        lib.flash_attn_fwd.restype = ctypes.c_int32
        torch_arrs = [o, q, k, v, bias, lse]
        attr_arrs = [batch_size, seqlen_q, seqlen_kv, nheads, head_dim, head_dim_value]
        stats = lib.flash_attn_fwd(*[ctypes.cast(arr.data_ptr(), ctypes.c_void_p) for arr in torch_arrs], *[ctypes.c_int32(arr) for arr in attr_arrs],ctypes.c_float(softmax_scale), ctypes.c_bool(causal) )
        ctx.save_for_backward(q, k, v, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, bias = ctx.saved_tensors
        causal = ctx.causal
        batch_size, seqlen_q, nheads, head_dim = q.shape
        seqlen_kv = k.shape[1]
        head_dim_value = v.shape[-1]   
        # delta = (do * o).sum(-1).transpose(-2, -1).float()
        delta = torch.zeros([batch_size, nheads, seqlen_q], dtype=torch.float).cuda()
        softmax_scale = (1 / q.shape[-1] ** 0.5)
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        db = torch.zeros_like(bias)
        lib = ctypes.cdll.LoadLibrary(BWD_LIB_PATH)
        lib.flash_attn_bwd.argstypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_bool]   
        lib.flash_attn_bwd.restype = ctypes.c_int32
        torch_arrs = [dq, dk, dv, db, o, q, k, v, bias, lse, do, delta]
        attr_arrs = [batch_size, seqlen_q, seqlen_kv, nheads, head_dim, head_dim_value]
        stats = lib.flash_attn_bwd(*[ctypes.cast(arr.data_ptr(), ctypes.c_void_p) for arr in torch_arrs], *[ctypes.c_int32(arr) for arr in attr_arrs],ctypes.c_float(softmax_scale), ctypes.c_bool(causal))
        return dq, dk, dv, db, None, None


flash_attn_func = FlashAttnFunc.apply