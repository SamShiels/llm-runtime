mod loader;

#[tokio::main]
async fn main() {
    println!("Hello, world!");
    loader::gguf::load().await
}
