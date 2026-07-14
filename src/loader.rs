use memmap2::Mmap;
use crate::model::{Config, TransformerWeights, LayerWeights, QuantizedTensor};

pub fn read_config(mmap: &Mmap) -> Config {
    let header_bytes = &mmap[0..28];
    let mut header_ints = [0i32; 7];

    for i in 0..7 {
        let start = i * 4;
        let byte = [
            header_bytes[start],
            header_bytes[start + 1],
            header_bytes[start + 2],
            header_bytes[start + 3],
        ];
        header_ints[i] = i32::from_le_bytes(byte);
    }

    Config::from_raw(header_ints)
}

// 1. Helper to slice a standard contiguous f32 block from the weights
fn slice_next<'a>(data: &'a [f32], offset: &mut usize, size: usize) -> &'a [f32] {
    let start = *offset;
    *offset += size;
    &data[start..*offset]
}

// 2. Helper to slice a contiguous q8_0 QuantizedTensor (scales block, then weights block)
fn slice_q8<'a>(data: &'a [f32], offset: &mut usize, size: usize) -> QuantizedTensor<'a> {
    let b = size / 32;
    
    // Load scales
    let start_scales = *offset;
    *offset += b;
    let scales = &data[start_scales..*offset];
    
    // Load weights (size elements occupy size/4 float slots)
    let start_weights = *offset;
    *offset += size / 4;
    let weights_f32 = &data[start_weights..*offset];
    
    let weights = unsafe {
        std::slice::from_raw_parts(weights_f32.as_ptr() as *const i8, size)
    };
    
    QuantizedTensor { scales, weights }
}

pub fn load_weights<'a>(raw_data: &'a [u8], config: &Config) -> TransformerWeights<'a> {
    let header_size = 28;
    let weights_f32 = unsafe {
        let ptr = raw_data.as_ptr().add(header_size) as *const f32;
        let len = (raw_data.len() - header_size) / 4;
        std::slice::from_raw_parts(ptr, len)
    };

    let dim = config.dim;
    let hidden_dim = config.hidden_dim;
    let n_layers = config.n_layers;
    let n_heads = config.n_heads;
    let n_kv_heads = config.n_kv_heads;
    let vocab_size = config.vocab_size;
    let seq_len = config.seq_len;
    let head_size = dim / n_heads;

    let mut offset = 0;

    // 1. Token Embedding Table (still f32)
    let token_embedding_table = slice_next(weights_f32, &mut offset, vocab_size * dim);

    // 2. Load the massive grouped tensor blocks for all layers (RMS norms are f32, projections are q8)
    let rms_att_weight_block = slice_next(weights_f32, &mut offset, n_layers * dim);
    
    let wq_block = slice_q8(weights_f32, &mut offset, n_layers * dim * (n_heads * head_size));
    let wk_block = slice_q8(weights_f32, &mut offset, n_layers * dim * (n_kv_heads * head_size));
    let wv_block = slice_q8(weights_f32, &mut offset, n_layers * dim * (n_kv_heads * head_size));
    let wo_block = slice_q8(weights_f32, &mut offset, n_layers * (n_heads * head_size) * dim);
    
    let rms_ffn_weight_block = slice_next(weights_f32, &mut offset, n_layers * dim);
    
    let w1_block = slice_q8(weights_f32, &mut offset, n_layers * hidden_dim * dim);
    let w2_block = slice_q8(weights_f32, &mut offset, n_layers * dim * hidden_dim);
    let w3_block = slice_q8(weights_f32, &mut offset, n_layers * hidden_dim * dim);

    // 3. Final RMS weight (still f32)
    let rms_final_weight = slice_next(weights_f32, &mut offset, dim);

    // 4. RoPE Frequency tables (still f32)
    let freq_cis_real = slice_next(weights_f32, &mut offset, seq_len * (head_size / 2));
    let freq_cis_imag = slice_next(weights_f32, &mut offset, seq_len * (head_size / 2));

    // 5. Optional classifier weights (still f32)
    let wcls = if !config.shared_classifier {
        Some(slice_next(weights_f32, &mut offset, vocab_size * dim))
    } else {
        None
    };

    assert_eq!(
        offset, weights_f32.len(),
        "Weights binary size mismatch! Read {}/{} floats.",
        offset, weights_f32.len()
    );

    // 6. Slice the massive blocks into per-layer weights
    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let rms_att_weight = &rms_att_weight_block[l * dim .. (l + 1) * dim];
        
        let q_size = dim * (n_heads * head_size);
        let q_blocks = q_size / 32;
        let wq = QuantizedTensor {
            scales: &wq_block.scales[l * q_blocks .. (l + 1) * q_blocks],
            weights: &wq_block.weights[l * q_size .. (l + 1) * q_size],
        };
        
        let k_size = dim * (n_kv_heads * head_size);
        let k_blocks = k_size / 32;
        let wk = QuantizedTensor {
            scales: &wk_block.scales[l * k_blocks .. (l + 1) * k_blocks],
            weights: &wk_block.weights[l * k_size .. (l + 1) * k_size],
        };
        let wv = QuantizedTensor {
            scales: &wv_block.scales[l * k_blocks .. (l + 1) * k_blocks],
            weights: &wv_block.weights[l * k_size .. (l + 1) * k_size],
        };
        
        let o_size = (n_heads * head_size) * dim;
        let o_blocks = o_size / 32;
        let wo = QuantizedTensor {
            scales: &wo_block.scales[l * o_blocks .. (l + 1) * o_blocks],
            weights: &wo_block.weights[l * o_size .. (l + 1) * o_size],
        };
        
        let rms_ffn_weight = &rms_ffn_weight_block[l * dim .. (l + 1) * dim];
        
        let ffn_size = hidden_dim * dim;
        let ffn_blocks = ffn_size / 32;
        let w1 = QuantizedTensor {
            scales: &w1_block.scales[l * ffn_blocks .. (l + 1) * ffn_blocks],
            weights: &w1_block.weights[l * ffn_size .. (l + 1) * ffn_size],
        };
        let w2 = QuantizedTensor {
            scales: &w2_block.scales[l * ffn_blocks .. (l + 1) * ffn_blocks],
            weights: &w2_block.weights[l * ffn_size .. (l + 1) * ffn_size],
        };
        let w3 = QuantizedTensor {
            scales: &w3_block.scales[l * ffn_blocks .. (l + 1) * ffn_blocks],
            weights: &w3_block.weights[l * ffn_size .. (l + 1) * ffn_size],
        };

        layers.push(LayerWeights {
            rms_att_weight,
            rms_ffn_weight,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
        });
    }

    println!("Weights loaded successfully!");

    TransformerWeights {
        token_embedding_table,
        layer: layers,
        rms_final_weight,
        freq_cis_real,
        freq_cis_imag,
        wcls,
    }
}