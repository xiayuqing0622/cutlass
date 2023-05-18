import argparse
import torch
import sys
import os
from piped_subprocess import PipedSubprocess, TORCH_DTYPE_NAME
import math


parser = argparse.ArgumentParser()
parser.add_argument("example_exe", type=str, help="Path to the 41_fused_multi_head_attention_backward executable")
args = parser.parse_args()

torch.manual_seed(0)
dtype = torch.float16
B, Mq, Mkv, H, K, Kv = 2, 2048, 2048, 8, 64, 64
causal = False
repeat_count = 100

ATOL = {
    torch.float: 5e-4,
    torch.half: 9.5e-2,
    torch.bfloat16: 7e-1,
}[dtype]

RTOL = {
    torch.float: 1e-4,
    torch.half: 2e-2,
    torch.bfloat16: 1e-1,
}[dtype]


assert not (causal and Mq < Mkv), "causal only supports seqlenK <= seqlenQ"

fmha_fw_binary = args.example_exe
if not os.path.isfile(fmha_fw_binary):
    print(f"""No such file: `{fmha_fw_binary}`\nDid you forget to run "make 41_fused_multi_head_attention"?""")
    sys.exit(1)

def create_lower_triangular_mask():
    return torch.triu(torch.full(  # type: ignore
        [1, Mq, Mkv],
        dtype=dtype,
        fill_value=float("-inf"),
    ), diagonal=1)

def ref_mha_bmk(q, k, v, bias, mask):
    # Multi-head attention with inputs/outputs in BMK format
    q = q.float()
    k = k.float()
    v = v.float()
    bias = bias.float()
    q = q * (1 / q.shape[-1] ** 0.5)
    attn = q @ k.transpose(-2, -1)
    if mask is not None:
        attn += mask
    if bias is not None:
        attn += bias
    attn_max = attn.max(-1, True).values
    attn_norm = (attn - attn_max).exp().sum(-1, True)
    attn = attn.softmax(-1)
    lse = attn_max + attn_norm.log()
    lse = lse.squeeze(2)
    return attn @ v, lse


def bmhk2bmk(t):
    return t.permute((0, 2, 1, 3)).reshape(
        [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
    )
def bhmn2bmn(t):
    return t.permute((0,1,2,3)).reshape([t.shape[0] * t.shape[1], t.shape[2], t.shape[3]])
def ref_mha_bmhk(q, k, v, bias, mask):
    # Multi-head attention with inputs/outputs in BMHK format
    assert q.ndim == 4

    out, lse = ref_mha_bmk(bmhk2bmk(q), bmhk2bmk(k), bmhk2bmk(v), bhmn2bmn(bias),mask=mask)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3)), lse.reshape([q.shape[0], q.shape[2], q.shape[1]])

def ref_mha_bw_bmhk(q, k, v, bias, mask, lse, out, grad_out, delta):
    lse = lse[:, :, :q.shape[1]]  #BMH, unpad Q dimension
    delta = delta.reshape([-1, delta.shape[-1], 1])

    # bmhk -> bmk
    q, k, v, out, grad_out = [bmhk2bmk(x).float() for x in (q, k, v, out, grad_out)]
    bias = bhmn2bmn(bias).float()
    attn_T = k @ q.transpose(-2, -1)
    if mask is not None:
        attn_T += mask.transpose(-2, -1)
    attn_T = attn_T * (1 / q.shape[-1] ** 0.5)
    if bias is not None:
        attn_T += bias.transpose(-2, -1)
    attn_T = attn_T - lse.reshape([-1, 1, lse.shape[-1]])
    attn_T = attn_T.exp()

    grad_v = attn_T @ grad_out

    dov = grad_out @ v.transpose(-2, -1)
    tmp = (dov - delta) * attn_T.transpose(-2, -1)
    grad_b = tmp
    tmp = tmp / (q.shape[-1] ** 0.5)
    grad_q = tmp @ k
    grad_k = tmp.transpose(-2, -1) @ q

    return [x.reshape([B, H, x.shape[1], x.shape[-1]]).permute([0, 2, 1, 3]) for x in [grad_q, grad_k, grad_v]] + [grad_b.reshape(B, H, Mq, Mkv).permute(0, 1, 2, 3)]


print("initializing tensors...")
query = torch.randn([B, Mq, H, K], dtype=dtype)
key = 3 * torch.randn([B, Mkv, H, K], dtype=dtype)
value = 3 * torch.randn([B, Mkv, H, Kv], dtype=dtype)
bias = torch.randn([B, H, Mq, Mkv], dtype=dtype)
mask = create_lower_triangular_mask() if causal else None


# let PyTorch compute gradients
query.requires_grad_(True)
key.requires_grad_(True)
value.requires_grad_(True)
bias.requires_grad_(True)

print("computing fw...")
outr, lser = ref_mha_bmhk(query, key, value, bias, mask=mask)


scale = (1 / query.shape[-1] ** 0.5)


with PipedSubprocess(fmha_fw_binary) as fw_kernel:
    # Send kernel arguments
    fw_kernel.write(
        TORCH_DTYPE_NAME[query.dtype],
        "scale", scale,
        "head_dim", K,
        "head_dim_value", Kv,
        "num_queries", Mq,
        "num_keys", Mkv,
        "num_heads", H,
        "custom_mask_type", (1 if causal else 0),
        "num_batches", B,
        "repeat_count", repeat_count,
    )
    fw_kernel.writeTensor(query, "query", ["q_strideB", "q_strideM", "q_strideH"])
    fw_kernel.writeTensor(key, "key", ["k_strideB", "k_strideM", "k_strideH"])
    fw_kernel.writeTensor(value, "value", ["v_strideB", "v_strideM", "v_strideH"])
    fw_kernel.writeTensor(bias, "bias", ["bias_strideB", "bias_strideH", "bias_strideM"])

    if fw_kernel.read() != "OK":
        print("Got unexpected output")
        print(fw_kernel.subp.communicate()[0])
        sys.exit(0)

    # Read kernel output
    output = fw_kernel.readTensor("output", ["o_strideB", "o_strideM", "o_strideH"], outr.shape).float()
    lse = fw_kernel.readTensor("lse", ["lse_strideB", "lse_strideH"], lser.shape).float()
    runtime_ms = float(fw_kernel.readNamed("runtime_ms"))

print(f"""
Fused multi-head attention - backward
    batch_size={B}
    num_queries={Mq}
    num_keys={Mkv}
    num_heads={H}
    head_dim={K}
    head_dim_value={Kv}

    Correctness:
        output: {"PASS" if torch.allclose(output, outr, rtol=RTOL, atol=ATOL) else "FAIL"} (delta: {(output - outr).abs().max()})
        lse:   {"PASS" if torch.allclose(lse, lser, rtol=RTOL, atol=ATOL) else "FAIL"} (delta: {(lse - lser).abs().max()})
        (atol={ATOL} / rtol={RTOL})
    Runtime: {runtime_ms}ms
""")


