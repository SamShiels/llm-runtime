mod loader;

#[tokio::main]
async fn main() {
    let model_result = loader::gguf::load().await;

    match model_result {
        Ok(model) => println!("{}", model.config.arch),
        Err(err) => println!("Error loading GGUF: {:?}", err),
    }
}
