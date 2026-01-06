mod loader;

#[tokio::main]
async fn main() {
    loader::gguf::load().await
}
