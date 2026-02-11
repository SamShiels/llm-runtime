use std::io::{self, Write};

use crate::engine::llm;

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
    println!("Total Tensors: {}", model.tensors.len());

    println!("\n=== Tensor Information ===");
    for (i, info) in model.tensors.iter().enumerate() {
        println!("{}. {} - dims: {:?}", i, info.name, info.dims);
    }

    print!("\nEnter text: ");
    io::stdout().flush().unwrap();  // Force print before read

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    llm::infer(&model, input);
}
