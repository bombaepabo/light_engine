use crate::state::{RunState, KVCache};
use crate::math;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub shared_classifier: bool,
}

impl Config {
    pub fn from_raw(raw: [i32; 7]) -> Self {
        let raw_vocab_size = raw[5];
        let vocab_size = raw_vocab_size.abs() as usize;
        let shared_classifier = raw_vocab_size > 0;

        Self {
            dim: raw[0] as usize,
            hidden_dim: raw[1] as usize,
            n_layers: raw[2] as usize,
            n_heads: raw[3] as usize,
            n_kv_heads: raw[4] as usize,
            vocab_size,
            seq_len: raw[6] as usize,
            shared_classifier,
        }
    }
}

// 1. Define QuantizedTensor for q8_0 blocks of size 32
#[derive(Clone, Copy)]
pub struct QuantizedTensor<'a> {
    pub scales: &'a [f32],   // Scale factor for each block of 32
    pub weights: &'a [i8],   // 8-bit quantized weights
    pub scales_device: *const f32, // GPU VRAM pointer for scales
    pub weights_device: *const i8, // GPU VRAM pointer for weights
}

unsafe impl<'a> Send for QuantizedTensor<'a> {}
unsafe impl<'a> Sync for QuantizedTensor<'a> {}

// 2. Update LayerWeights to use QuantizedTensor for projection layers
#[derive(Clone)]
pub struct LayerWeights<'a> {
    pub rms_att_weight: &'a [f32], // Still f32
    pub rms_ffn_weight: &'a [f32], // Still f32

    pub wq: QuantizedTensor<'a>,   // Quantized
    pub wk: QuantizedTensor<'a>,   // Quantized
    pub wv: QuantizedTensor<'a>,   // Quantized
    pub wo: QuantizedTensor<'a>,   // Quantized

    pub w1: QuantizedTensor<'a>,   // Quantized
    pub w2: QuantizedTensor<'a>,   // Quantized
    pub w3: QuantizedTensor<'a>,   // Quantized
}

#[derive(Clone)]
pub struct TransformerWeights<'a> {
    pub token_embedding_table: &'a [f32], // Still f32
    pub layer: Vec<LayerWeights<'a>>,
    pub rms_final_weight: &'a [f32],       // Still f32
    pub freq_cis_real: &'a [f32],          // Still f32
    pub freq_cis_imag: &'a [f32],          // Still f32
    pub wcls: Option<&'a [f32]>,           // Still f32
}

impl<'a> TransformerWeights<'a> {
    pub fn forward(
        &self,
        token: usize,
        pos: usize,
        config: &Config,
        state: &mut RunState,
        kv_cache: &mut KVCache,
    ) {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let n_layers = config.n_layers;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let seq_len = config.seq_len;
        let head_size = dim / n_heads;
        let kv_mul = n_heads / n_kv_heads;

        // 1. Token Embedding Lookup
        let embed_offset = token * dim;
        state.x.copy_from_slice(&self.token_embedding_table[embed_offset .. embed_offset + dim]);

        // 2. Loop through all 6 transformer layers
        for l in 0..n_layers {
            let layer_weights = &self.layer[l];

            // --- Block 1: Attention ---
            math::rmsnorm(&mut state.xb, &state.x, layer_weights.rms_att_weight, 1e-5);

            // Call math::matmul_q8 for quantized projections
            math::matmul_q8(&mut state.q, &state.xb, &layer_weights.wq, dim, dim);
            math::matmul_q8(&mut state.k[0 .. n_kv_heads * head_size], &state.xb, &layer_weights.wk, n_kv_heads * head_size, dim);
            math::matmul_q8(&mut state.v[0 .. n_kv_heads * head_size], &state.xb, &layer_weights.wv, n_kv_heads * head_size, dim);
            math::rope(&mut state.q, pos, self.freq_cis_real, self.freq_cis_imag, n_heads, head_size);
            math::rope(&mut state.k, pos, self.freq_cis_real, self.freq_cis_imag, n_kv_heads, head_size);

            let kv_dim = n_kv_heads * head_size;
            let layer_offset = l * seq_len * kv_dim;
            let pos_offset = layer_offset + pos * kv_dim;
            
            kv_cache.key_cache[pos_offset .. pos_offset + kv_dim].copy_from_slice(&state.k[0 .. kv_dim]);
            kv_cache.value_cache[pos_offset .. pos_offset + kv_dim].copy_from_slice(&state.v[0 .. kv_dim]);

            for h in 0..n_heads {
                let q_offset = h * head_size;
                let att_offset = h * seq_len;
                let kv_h = h / kv_mul;

                for t in 0..=pos {
                    let cache_offset = l * seq_len * kv_dim + t * kv_dim + kv_h * head_size;
                    let mut score = 0.0;
                    for i in 0..head_size {
                        score += state.q[q_offset + i] * kv_cache.key_cache[cache_offset + i];
                    }
                    score /= (head_size as f32).sqrt();
                    state.att[att_offset + t] = score;
                }

                let att_slice = &mut state.att[att_offset .. att_offset + pos + 1];
                math::softmax(att_slice);

                let xb2_offset = h * head_size;
                for i in 0..head_size {
                    state.xb2[xb2_offset + i] = 0.0;
                }

                for t in 0..=pos {
                    let cache_offset = l * seq_len * kv_dim + t * kv_dim + kv_h * head_size;
                    let prob = state.att[att_offset + t];
                    for i in 0..head_size {
                        state.xb2[xb2_offset + i] += prob * kv_cache.value_cache[cache_offset + i];
                    }
                }
            }

            // Call math::matmul_q8 for quantized attention output projection
            math::matmul_q8(&mut state.xb, &state.xb2, &layer_weights.wo, dim, dim);

            for i in 0..dim {
                state.x[i] += state.xb[i];
            }

            // --- Block 2: Feed-Forward Network ---
            math::rmsnorm(&mut state.xb, &state.x, layer_weights.rms_ffn_weight, 1e-5);

            // Call math::matmul_q8 for quantized FFN projection layers
            math::matmul_q8(&mut state.hb, &state.xb, &layer_weights.w1, hidden_dim, dim);
            math::matmul_q8(&mut state.hb2, &state.xb, &layer_weights.w3, hidden_dim, dim);

            math::silu_and_mul(&mut state.hb, &state.hb2);

            math::matmul_q8(&mut state.xb, &state.hb, &layer_weights.w2, dim, hidden_dim);

            for i in 0..dim {
                state.x[i] += state.xb[i];
            }
        }

        // 3. Final layer normalization
        math::rmsnorm(&mut state.xb, &state.x, self.rms_final_weight, 1e-5);

        // 4. Classifier Head
        // Note: wcls remains f32 because it shares token_embedding_table (f32)
        let wcls = self.wcls.unwrap_or(self.token_embedding_table);
        math::matmul(&mut state.logits, &state.xb, wcls, config.vocab_size, dim);
    }
}