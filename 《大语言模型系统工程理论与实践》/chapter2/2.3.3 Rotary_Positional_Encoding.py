import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
      #本函数要与计算旋转角度
      # dim: 词嵌入维度 (应为偶数)
      # end: 序列最大长度
      # theta: RoPE中的常数，通常为10000
      #计算每个偶数维度对应的频率值（要分为2i和2i＋1，总数是偶数）
      freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
      #创建每个位置索引张量
      t = torch.arange(end, device=freqs.device)  # type: ignore
      freqs = torch.outer(t, freqs).float()  # type: ignore
      # 将角度转换为复数形式（cis(θ) = cos(θ) + i*sin(θ)）
      freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
      # freqs_cis.shape = (end, dim // 2)
      return freqs_cis
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
      #将旋转位置编码应用到Q和K张量
      # xq, xk: Query and Key tensors, shape (batch_size, seq_len, num_heads, head_dim)
      # freqs_cis: Precomputed cos/sin values in complex form, shape (seq_len, head_dim // 2)
      # Reshape xq and xk to (batch_size, seq_len, num_heads, head_dim // 2, 2)

      #将张量转化为复数形式：1.先将最后一维重塑为[..., head_dim//2, 2]（实部和虚部）
      xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
      xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)
      # Convert to complex numbers； 2.使用view_as_complex将其转换为复数张量
      xq_c = torch.view_as_complex(xq_r)
      xk_c = torch.view_as_complex(xk_r)
      # freqs_cis needs to be reshaped/broadcasted to match xq_c and xk_c for multiplication
      # freqs_cis shape: (seq_len, head_dim // 2) -> (1, seq_len, 1, head_dim // 2)
      freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # Add batch and head_num dimension
      # Apply rotation in complex plane: z_rotated = z * e^(i*theta)
      xq_out = torch.view_as_real(xq_c * freqs_cis).flatten(3)
      xk_out = torch.view_as_real(xk_c * freqs_cis).flatten(3) 
      return xq_out.type_as(xq), xk_out.type_as(xk)

# # 示例参数
B, S, N, H = 1, 10, 8, 64 # Batch, SeqLen, NumHeads, HeadDim
D = N * H # ModelDim (实际上RoPE作用于HeadDim)
freqs_cis_example = precompute_freqs_cis(H, S)
q_example = torch.randn(B, S, N, H)
k_example = torch.randn(B, S, N, H)
q_rotated, k_rotated = apply_rotary_emb(q_example, k_example, freqs_cis_example)
print("Query shape after RoPE:", q_rotated.shape)
print("Key shape after RoPE:", k_rotated.shape)
