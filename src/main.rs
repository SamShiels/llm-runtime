use std::io::{self, Write};

use crate::tokenizer::Tokenizer;

mod loader;
mod tokenizer;

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

    let tokeizer = Tokenizer::new(&model);

    print!("Enter text: ");
    io::stdout().flush().unwrap();  // Force print before read
    
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    tokeizer.tokenize(&input.trim());
}
