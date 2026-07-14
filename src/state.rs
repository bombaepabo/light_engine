use crate::model::Config;
// 1. RunState holds all the pre-allocated temporary buffers for a single forward pass.
pub struct RunState {
    pub x: Vec<f32>,      // Current activation state (size: dim)
    pub xb: Vec<f32>,     // Helper buffer inside a layer (size: dim)
    pub xb2: Vec<f32>,    // Second helper buffer (size: dim)
    pub hb: Vec<f32>,     // FFN hidden layer buffer (size: hidden_dim)
    pub hb2: Vec<f32>,    // FFN second hidden buffer (size: hidden_dim)
    pub q: Vec<f32>,      // Query vector (size: dim)
    pub k: Vec<f32>,      // Key vector (size: dim)
    pub v: Vec<f32>,      // Value vector (size: dim)
    pub att: Vec<f32>,    // Attention scores buffer (size: n_heads * seq_len)
    pub logits: Vec<f32>, // Output logits (size: vocab_size)
}
impl RunState {
    pub fn new(config: &Config) -> Self {
        Self {
            x: vec![0.0; config.dim],
            xb: vec![0.0; config.dim],
            xb2: vec![0.0; config.dim],
            hb: vec![0.0; config.hidden_dim],
            hb2: vec![0.0; config.hidden_dim],
            q: vec![0.0; config.dim],
            k: vec![0.0; config.dim],
            v: vec![0.0; config.dim],
            // For attention, each head looks at up to `seq_len` tokens
            att: vec![0.0; config.n_heads * config.seq_len],
            logits: vec![0.0; config.vocab_size],
        }
    }
}
// 2. KVCache holds the memory of past keys and values for all layers.
pub struct KVCache {
    pub key_cache: Vec<f32>,   // Holds past keys
    pub value_cache: Vec<f32>, // Holds past values
}
impl KVCache {
    pub fn new(config: &Config) -> Self {
        // Size: n_layers * seq_len * (n_kv_heads * head_size)
        // In our model: 6 * 256 * (6 * 48) = 442,368 floats
        let cache_size = config.n_layers
            * config.seq_len
            * (config.n_kv_heads * (config.dim / config.n_heads));
        Self {
            key_cache: vec![0.0; cache_size],
            value_cache: vec![0.0; cache_size],
        }
    }
}