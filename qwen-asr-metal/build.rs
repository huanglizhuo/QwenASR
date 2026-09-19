fn main() {
    if std::env::var_os("CARGO_FEATURE_GPU").is_some() {
        let path =
            std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("build");
        println!("cargo:rustc-link-search=native={}", path.display());
        println!(
            "cargo:rerun-if-changed={}",
            path.join("libqwen_metal.a").display()
        );
    }
}
