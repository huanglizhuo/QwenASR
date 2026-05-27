fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // Check if blas feature is enabled via CARGO_FEATURE_BLAS env var
    if std::env::var("CARGO_FEATURE_BLAS").is_ok() {
        match target_os.as_str() {
            "macos" => {
                println!("cargo:rustc-link-lib=framework=Accelerate");
            }
            "linux" => {
                println!("cargo:rustc-link-lib=openblas");
            }
            "windows" => {
                if let Ok(dir) = std::env::var("OPENBLAS_LIB_DIR") {
                    println!("cargo:rustc-link-search=native={dir}");
                }
                println!("cargo:rustc-link-lib=openblas_64");
            }
            _ => {
                // No BLAS available, will use fallback matmul
            }
        }
    }
}
