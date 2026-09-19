.PHONY: reference gpu probe microbench runtime-check encoder-check decoder-check frontend-check fusion-check profile-check ops-check test baseline prepare perf-probe

gpu:
	mkdir -p build
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=26.4 -c native/runtime.mm -o build/runtime.o
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=26.4 -c native/encoder.mm -o build/encoder.o
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=26.4 -c native/decoder.mm -o build/decoder.o
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=26.4 -c native/bridge.mm -o build/bridge.o
	ar rcs build/libqwen_metal.a build/runtime.o build/encoder.o build/decoder.o build/bridge.o
	cargo build --locked --release --features gpu --bin qwen-asr-gpu

reference:
	cargo build --locked --release --manifest-path Cargo.toml

probe:
	mkdir -p build
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=26.4 native/probe.mm -framework Foundation -framework Metal -o build/metal-probe
	./build/metal-probe

test:
	python3 -m unittest discover -s bench -p 'test_*.py' -v
	cargo fmt --manifest-path Cargo.toml --check

microbench:
	mkdir -p build
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=26.4 native/microbench.mm -framework Foundation -framework Metal -o build/metal-microbench
	./build/metal-microbench native/quantized.metal build/microbench.json

runtime-check:
	mkdir -p build
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=26.4 native/runtime.mm native/runtime_check.mm -framework Foundation -framework Metal -o build/runtime-check

encoder-check:
	mkdir -p build
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=26.4 native/runtime.mm native/encoder.mm native/encoder_check.mm -framework Foundation -framework Metal -o build/encoder-check

decoder-check:
	mkdir -p build
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=26.4 native/runtime.mm native/decoder.mm native/decoder_check.mm -framework Foundation -framework Metal -o build/decoder-check

frontend-check:
	mkdir -p build
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=26.4 native/runtime.mm native/frontend_check.mm -framework Foundation -framework Metal -o build/frontend-check

fusion-check:
	mkdir -p build
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=26.4 native/runtime.mm native/fusion_check.mm -framework Foundation -framework Metal -o build/fusion-check
	./build/fusion-check models/asr-hybrid native

profile-check:
	mkdir -p build
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=26.4 native/runtime.mm native/decoder.mm native/profile_check.mm -framework Foundation -framework Metal -o build/profile-check

ops-check:
	mkdir -p build
	xcrun clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=26.4 native/runtime.mm native/ops_check.mm -framework Foundation -framework Metal -o build/ops-check
	./build/ops-check native

prepare:
	python3 bench/benchmark.py prepare

baseline: reference
	python3 bench/benchmark.py run --output bench/runs/cpu-$$(date +%Y%m%d-%H%M%S)

perf-probe:
	mkdir -p build
	xcrun clang++ -std=c++20 -O2 -fobjc-arc -mmacosx-version-min=26.4 native/runtime.mm native/encoder.mm native/decoder.mm native/perf_probe.mm -framework Foundation -framework Metal -o build/perf-probe
