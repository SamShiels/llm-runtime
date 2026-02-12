use std::io::{self, Write};

use crate::{engine::llm, types::LayerType};

mod loader;
mod engine;
pub mod types;  // Common types accessible throughout

#[tokio::main]
async fn main() {
    let model_result = loader::gguf::load().await;

    let model = match model_result {
        Ok(model) => model,
        Err(err) =>  {
            println!("Error loading GGUF: {:?}", err);
            return;
        },
    };

    println!("\n=== Model Architecture ===");
    println!("Architecture: {}", model.config.arch);
    println!("Context Length: {}", model.config.context_length);
    println!("Embedding Length: {}", model.config.embedding_length);
    println!("Total Layers: {}", model.layers.len());

    println!("\n=== Layer Weights ===");
    for (i, layer) in model.layers.iter().enumerate() {
        match &layer.layer_type {
            LayerType::Attention(attn) => {
                println!("Layer {i}: q={} k={} v={} out={} ffn_gate={}",
                    attn.q.len(), attn.k.len(), attn.v.len(), attn.output.len(),
                    layer.ffn.gate.len());
            },
            LayerType::ShortConv(conv) => {
                println!("Layer {i} (shortconv): in_proj={} conv={} out_proj={} ffn_gate={}",
                    conv.in_proj.len(), conv.conv.len(), conv.out_proj.len(),
                    layer.ffn.gate.len());
            }
        }
    }

    print!("\nEnter text: ");
    io::stdout().flush().unwrap();  // Force print before read

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    llm::infer(&model, input);
}
