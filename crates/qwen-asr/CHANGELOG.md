# Changelog

## [0.11.1](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.11.0...qwen-asr-v0.11.1) (2026-08-27)


### Bug Fixes

* **review:** address PR [#52](https://github.com/huanglizhuo/QwenASR/issues/52) review — 4th BLAS site, spin-join, flag parsing ([b71f338](https://github.com/huanglizhuo/QwenASR/commit/b71f33866ae5da4311a805c97960d67e838b685e))
* **stream:** keep stable-token cursor in sync on degen/re-anchor reset ([1d1906d](https://github.com/huanglizhuo/QwenASR/commit/1d1906da1383241c9b612dae614be637a7af19ef))
* **x86:** avoid pooled OpenBLAS calls ([ee27b7c](https://github.com/huanglizhuo/QwenASR/commit/ee27b7cacdce162f01c97b1f15a0b5b26021d538))
* **x86:** avoid pooled OpenBLAS calls ([52d73c6](https://github.com/huanglizhuo/QwenASR/commit/52d73c64d59789e67c9b316276d7b00fcfe37542))


### Performance Improvements

* **cli:** multi-file parallel transcription + macOS build tuning ([a932504](https://github.com/huanglizhuo/QwenASR/commit/a9325047805ca4a3776e15325678ed2f79b0a70f))

## [0.11.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.10.0...qwen-asr-v0.11.0) (2026-07-30)


### ⚠ BREAKING CHANGES

* **qwen-asr:** transcribe_segmented and the listed constants are removed from the published qwen-asr API surface.

### Code Refactoring

* **qwen-asr:** remove unused public API and dead code ([7fea8f2](https://github.com/huanglizhuo/QwenASR/commit/7fea8f22fb20a77fce1730eacf59a3e4e9d94707))

## [0.10.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.9.0...qwen-asr-v0.10.0) (2026-07-25)


### Features

* **decoder:** offline prompt-lookup speculative decode (opt-in, default off) ([6519c23](https://github.com/huanglizhuo/QwenASR/commit/6519c23e1513bf1ca65f9d084ab5eea1ca14868c))
* **flutter:** Android device support — dotprod rustflags, release permissions, metric hooks ([62a8894](https://github.com/huanglizhuo/QwenASR/commit/62a8894876e295c9dd9c334d914d206bcd80cc2f))
* **mobile:** gate INT8 decoder-prefill behind int8-prefill cargo feature; Android default-ON (full-set WER +3.3%) ([3aa1491](https://github.com/huanglizhuo/QwenASR/commit/3aa1491a6e749f24a03fcad6f42a45d9f7c4cb22))
* **mobile:** INT8 encoder weight GEMMs behind int8-encoder cargo feature (R13 stage 2) ([b11b1a7](https://github.com/huanglizhuo/QwenASR/commit/b11b1a7b3e818b484918b9b5bdb0226551c4205e))
* **mobile:** QASR_METRIC active-INT8-path log + CLI no-BLAS feature forwarding (R13 stage 2 diagnostics) ([e4d952d](https://github.com/huanglizhuo/QwenASR/commit/e4d952df98fe54d3610e48498313795af34a3a42))
* multi-token greedy verifier core (R13-A) ([1ff66ff](https://github.com/huanglizhuo/QwenASR/commit/1ff66ff75311b8cf225153c2d55bf49ffc8ef4e5))
* **stream:** decouple VAD segment reset from multilingual; 2s-only chunk ([f06a0de](https://github.com/huanglizhuo/QwenASR/commit/f06a0dec861b83b5a53dc68576120ef8ba1f4171))
* **streaming:** expose provisional (unfixed) tail for lower-latency UX ([e70c387](https://github.com/huanglizhuo/QwenASR/commit/e70c387d1d80cfcf8ac1397cecac7f9b42fa3009))
* **streaming:** opt-in multilingual per-utterance language re-detection ([d3b18c1](https://github.com/huanglizhuo/QwenASR/commit/d3b18c12099cd69e4bfa6be16f84f0cf4103d639))


### Bug Fixes

* **ci:** resolve clippy lints on linux/stable toolchain ([4c11cb7](https://github.com/huanglizhuo/QwenASR/commit/4c11cb7b2da667477e49b2cede8ad56c3d9b2925))
* **decoder:** align prompt-lookup drafts with the verify window ([fec0072](https://github.com/huanglizhuo/QwenASR/commit/fec0072c66f3a1876758185aad2ac9577dd2f8ee))


### Performance Improvements

* **audio:** reuse thread-local scratch buffers in mel_spectrogram ([cea1278](https://github.com/huanglizhuo/QwenASR/commit/cea12781c2c2084dd8a02d49a6c3fa635f1493aa))
* **decoder:** fuse final-norm + INT8 lm_head argmax into the decode region ([8fa008d](https://github.com/huanglizhuo/QwenASR/commit/8fa008dd65365cef87ab14af6304f5884256c335))
* **decoder:** remove per-token heap allocations from hot decode kernels ([fccf84e](https://github.com/huanglizhuo/QwenASR/commit/fccf84e9fdcb3c319fbda0c24e2ddaf48ed55580))
* **encoder:** cache sinusoidal PE table across chunks ([1504b5e](https://github.com/huanglizhuo/QwenASR/commit/1504b5ea09415946d08950145cbbad3410e6a93d))
* **encoder:** parallelize conv-output reshape, vectorize bias adds ([81df750](https://github.com/huanglizhuo/QwenASR/commit/81df7507ab259abe4d4d1cd045c9e51d1bf03fd6))
* **kernels:** INT8 decoder-prefill GEMM for no-BLAS Android (R13 stage 1, default off) ([a276e90](https://github.com/huanglizhuo/QwenASR/commit/a276e90d35a3d7e3fff68acbd9c9744868a62683))
* **kernels:** NEON pool-parallel no-BLAS GEMM fallback (Android) ([b1fa840](https://github.com/huanglizhuo/QwenASR/commit/b1fa8406f9bdfda9a4d239bd4254ff8299fceac3))
* streaming overlap-draft speculative decode via multi-token verifier (R13-B) ([f187897](https://github.com/huanglizhuo/QwenASR/commit/f187897e66b820fc51b4523d9841cf0c8904909e))
* **transcribe:** parallel segment scheduling for timestamped paths ([32b41f8](https://github.com/huanglizhuo/QwenASR/commit/32b41f8a98e747cdc8cb04516fe641d23585f14b))
* x86_64 AVX2 INT8 decode kernels and encoder conv workspace reuse ([da3df9f](https://github.com/huanglizhuo/QwenASR/commit/da3df9f0f6c25c1bdb1fbaa0b2170baaf4a91891))

## [0.9.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.8.1...qwen-asr-v0.9.0) (2026-07-16)


### Features

* batched INT8 decode kernels for lockstep decode (R12-E2 stage) ([930dc1c](https://github.com/huanglizhuo/QwenASR/commit/930dc1c23346bd04f48f1b623e9c91fcfea071f5))


### Bug Fixes

* revert INT4 decode FFN weights to INT8 to restore WER (R12-B5) ([04fb434](https://github.com/huanglizhuo/QwenASR/commit/04fb4341a9adb91c0740350608eda88af61e967e))


### Performance Improvements

* drop aarch64 runtime copy of fused gate_up bf16 weights (R12-A) ([97b521c](https://github.com/huanglizhuo/QwenASR/commit/97b521c92cdfc0397eab5a78a2279aa99fcf0eff))
* dynamic chunk scheduling across P/E cores ([cc7f49d](https://github.com/huanglizhuo/QwenASR/commit/cc7f49d48a5a9361070438dc004e90d5910a06fb))
* INT4 group-quantized decode weights (tier 1) ([6ed3952](https://github.com/huanglizhuo/QwenASR/commit/6ed3952646ba0d9e710a63305952a1f1405c3dd3))
* lockstep batched segment decode amortizes decode weight stream (R12-E3) ([f80fe8f](https://github.com/huanglizhuo/QwenASR/commit/f80fe8fdb05af48da70ce4d4cb179f773d0e84d4))
* mmap-backed INT8 weight sidecar cuts startup quantization (R12-H1) ([2e49dcc](https://github.com/huanglizhuo/QwenASR/commit/2e49dccb40a7db72cc837db8c7cce1412642ca28))
* pool-parallel GEMM slices and E-core-aware default threads ([9a0c19e](https://github.com/huanglizhuo/QwenASR/commit/9a0c19ea26c94a8989e77ecf7f32219055911335))
* raise default thread count for dynamic-scheduling era ([963a404](https://github.com/huanglizhuo/QwenASR/commit/963a404106730a334a25556b75c924313bdfe3ce))

## [0.8.1](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.8.0...qwen-asr-v0.8.1) (2026-07-09)


### Performance Improvements

* use spare CPUs for long segment workers ([43e4626](https://github.com/huanglizhuo/QwenASR/commit/43e4626bcd3115fdbdc919e2231d8e7c66b3c311))

## [0.8.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.7.5...qwen-asr-v0.8.0) (2026-07-08)


### Features

* add structured subtitle outputs ([14c39c0](https://github.com/huanglizhuo/QwenASR/commit/14c39c05103be978b8d0e374520b9965d8eaf6ee))


### Bug Fixes

* restore default decode preamble and correct cue-grouping rules ([757647c](https://github.com/huanglizhuo/QwenASR/commit/757647c211172df4a063a8af9abbdc8b792c4b31))


### Performance Improvements

* fuse single-token decoder layer stages into one parallel region ([307016b](https://github.com/huanglizhuo/QwenASR/commit/307016b07112144a93f5ff4b11097398ac8eef0c))
* load encoder weights via pread to keep bf16 pages nonresident ([620751e](https://github.com/huanglizhuo/QwenASR/commit/620751ecf6120a8626d21d0c07951008b239d1ee))
* pair GQA query heads in single-token attention KV scan ([31713a9](https://github.com/huanglizhuo/QwenASR/commit/31713a93413c54af8e92171f4a4acd4ffcee4ddf))
* parallelize bf16 weight scratch conversion in prefill GEMMs ([522c5cb](https://github.com/huanglizhuo/QwenASR/commit/522c5cbb1d3f79eac8c24e5dbb86e5a6c00a9f16))
* parallelize independent segment decode for long audio ([aea0cc2](https://github.com/huanglizhuo/QwenASR/commit/aea0cc27cb2771cb23a66c195d1ce9ca690d85ef))
* prepack encoder transformer weights to f32 at load ([9141b78](https://github.com/huanglizhuo/QwenASR/commit/9141b788bab0cee1836617d73af7e23fada3658b))

## [0.7.5](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.7.4...qwen-asr-v0.7.5) (2026-07-03)


### Bug Fixes

* remove remaining transcript truncation causes (fixes [#31](https://github.com/huanglizhuo/QwenASR/issues/31)) ([1d677db](https://github.com/huanglizhuo/QwenASR/commit/1d677dbd97d0f1b7d381cf069531198cd8c7df56))

## [0.7.4](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.7.3...qwen-asr-v0.7.4) (2026-07-02)


### Bug Fixes

* remove LONG_AUDIO_FAST token cap and overlap argmax with next-token prep (fixes [#31](https://github.com/huanglizhuo/QwenASR/issues/31)) ([b142e3a](https://github.com/huanglizhuo/QwenASR/commit/b142e3a4de641e35454dbd0fe1cfc2ae6792d856))


### Performance Improvements

* prefault safetensors mmap pages ([76cfa2a](https://github.com/huanglizhuo/QwenASR/commit/76cfa2a5bbd73e5fac099b819599fcbcaa436732))

## [0.7.3](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.7.2...qwen-asr-v0.7.3) (2026-06-27)


### Bug Fixes

* harden encoder BF16 scratch sizing for proj2 ([#33](https://github.com/huanglizhuo/QwenASR/issues/33)) ([4078f51](https://github.com/huanglizhuo/QwenASR/commit/4078f51c3c22a0800e02fa708d530bae73bcbea4))


### Performance Improvements

* keep weights as BF16 mmap views to fit on iOS ([#32](https://github.com/huanglizhuo/QwenASR/issues/32)) ([64fff76](https://github.com/huanglizhuo/QwenASR/commit/64fff76c1154318a211657d0f5076cc6aa5f79cc)) — streams BF16 weights through a shared f32 scratch buffer instead of upconverting every tensor to `Vec<f32>` at load. Cuts peak memory **~6.4× (4.54 GB → 0.70 GB)** on the 0.6B forced aligner and **~42–57%** across ASR offline/segmented/streaming, with model load **~4× faster** and alignment output bit-identical.

## [0.7.2](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.7.1...qwen-asr-v0.7.2) (2026-06-13)


### Performance Improvements

* add startup-phase profile breakdown (A6) ([8eba33f](https://github.com/huanglizhuo/QwenASR/commit/8eba33fb91003fcf65ef7474df412b5fd1ec66eb))
* batched GEMM prefill causal attention (E8) ([b2dac19](https://github.com/huanglizhuo/QwenASR/commit/b2dac19735828b11f97be3fb45e58f75678a55e6))
* **decoder:** allocate hot decoder weights with 2 MB superpages ([1dd48aa](https://github.com/huanglizhuo/QwenASR/commit/1dd48aa844253fb7cd8d8bd07920392f085c0120))
* **decoder:** allocate kv cache with superpages ([eee2396](https://github.com/huanglizhuo/QwenASR/commit/eee23969151fa5738a6ccbdf8bc45e13a6ddca75))
* default thread count to performance cores (post-E8) ([44d6bad](https://github.com/huanglizhuo/QwenASR/commit/44d6bade14ca26fa2fa1daf2928cc0d579b45dc6))
* parallelize model load conversions (E2) ([7a8ed02](https://github.com/huanglizhuo/QwenASR/commit/7a8ed024dda75fd93d960c2026a3601f907a9e18))
* prefault mmap'd model weights with MADV_WILLNEED (A5) ([f1d3596](https://github.com/huanglizhuo/QwenASR/commit/f1d3596c84708b9f73a386b827aedcb0afef9e3a))

## [0.7.1](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.7.0...qwen-asr-v0.7.1) (2026-05-14)


### Bug Fixes

* document qwen-asr WER benchmark scripts ([e49a138](https://github.com/huanglizhuo/QwenASR/commit/e49a138990422f0bda9d368bec7b7bf84683bd63))

## [0.7.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.6.0...qwen-asr-v0.7.0) (2026-05-14)


### Features

* improve ASR WER and streaming performance ([c41d2fc](https://github.com/huanglizhuo/QwenASR/commit/c41d2fcc5ada21aa3b64920cff8166255752e464))

## [0.6.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.5.0...qwen-asr-v0.6.0) (2026-05-10)


### Features

* document benchmarked ASR performance ([f3b33a0](https://github.com/huanglizhuo/QwenASR/commit/f3b33a0863c815fedbdac18650e2b673af53efbe))

## [0.5.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.4.2...qwen-asr-v0.5.0) (2026-03-20)


### Features

* optimize the speed with auto research pattern ([9c13daa](https://github.com/huanglizhuo/QwenASR/commit/9c13daaadd964a6c2d79bed99364b26a368c305f))

## [0.4.2](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.4.1...qwen-asr-v0.4.2) (2026-03-14)


### Bug Fixes

* install OpenBLAS on CI to support build ([e995cad](https://github.com/huanglizhuo/QwenASR/commit/e995cad2c18999a39e74c56c26a2d6526020fd53))

## [0.4.1](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.4.0...qwen-asr-v0.4.1) (2026-03-14)


### Bug Fixes

* clean up the code ([346d112](https://github.com/huanglizhuo/QwenASR/commit/346d112595c0d93f58c54339a7f32ca6e1e648d8))

## [0.4.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.3.0...qwen-asr-v0.4.0) (2026-03-13)


### Features

* expose streaming C API for native integration ([a58befe](https://github.com/huanglizhuo/QwenASR/commit/a58befe991393405867ece18621d60ad5dc7cfd6))
* expose streaming C API for native macOS/iOS integration ([51dc917](https://github.com/huanglizhuo/QwenASR/commit/51dc917fc214a38ab6a85db55e2cc53ecdfabb6d))


### Bug Fixes

* handle split UTF-8 sequences in BPE token decoding ([75c3e31](https://github.com/huanglizhuo/QwenASR/commit/75c3e3172b33e412fdde0bee6f8009277e9bb35e))
* handle split UTF-8 sequences in BPE token decoding ([c86fae2](https://github.com/huanglizhuo/QwenASR/commit/c86fae2918c50f5ef74097fb4ad1e97a19387a18))

## [0.3.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.2.4...qwen-asr-v0.3.0) (2026-02-23)


### Features

* add missing parameter to qwen asr offline model ([f56e8b1](https://github.com/huanglizhuo/QwenASR/commit/f56e8b1e58731344fad92a7ed38c59a9f09267f6))
* add missing parameter to qwen asr offline model ([6d1e38d](https://github.com/huanglizhuo/QwenASR/commit/6d1e38da19cbae46c2afe2e1af03a5d437679ef8))
* improve the live stream performace on macos ([ba47230](https://github.com/huanglizhuo/QwenASR/commit/ba47230403f897bde2486b3235cfd3f5ca24e293))
* refine the stream mode ([7ac71a9](https://github.com/huanglizhuo/QwenASR/commit/7ac71a9a4f30b13ff155ff2eb345db85bce1b91c))
* support live from blackhold for macos for qwen-asr-cli ([724ead1](https://github.com/huanglizhuo/QwenASR/commit/724ead1fe121d0ed0a0f7ef874142f665e7d0da3))
* update readme ([cde2178](https://github.com/huanglizhuo/QwenASR/commit/cde21787bb545e12c154045562883b9ced00514d))


### Bug Fixes

* publish 0.2.3 with tag-driven flow ([3637ec8](https://github.com/huanglizhuo/QwenASR/commit/3637ec80f5519ecbd0a034f6c1f23f78156cd0fe))
* publish 0.2.3 with tag-driven flow ([e7bbd18](https://github.com/huanglizhuo/QwenASR/commit/e7bbd18dc009c3bd87f32e2346c196f65c618b19))
* trigger patch release 0.2.1 for flutter ([b5785f9](https://github.com/huanglizhuo/QwenASR/commit/b5785f9e0a6e4cab3a4796bbd1bd401876ea5926))
* update the both library readme to mention this is WIP project ([139a591](https://github.com/huanglizhuo/QwenASR/commit/139a5915205083abc4b87fd0228ccf4c725c99c0))
* update the release flow to support PAT ([2b9be6c](https://github.com/huanglizhuo/QwenASR/commit/2b9be6c21b7e74e51bf1d1f15e6959679db70542))

## [0.2.3](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.2.2...qwen-asr-v0.2.3) (2026-02-22)


### Bug Fixes

* update the release flow to support PAT ([2b9be6c](https://github.com/huanglizhuo/QwenASR/commit/2b9be6c21b7e74e51bf1d1f15e6959679db70542))

## [0.2.2](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.2.1...qwen-asr-v0.2.2) (2026-02-22)


### Bug Fixes

* publish 0.2.3 with tag-driven flow ([3637ec8](https://github.com/huanglizhuo/QwenASR/commit/3637ec80f5519ecbd0a034f6c1f23f78156cd0fe))
* publish 0.2.3 with tag-driven flow ([e7bbd18](https://github.com/huanglizhuo/QwenASR/commit/e7bbd18dc009c3bd87f32e2346c196f65c618b19))

## [0.2.1](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.2.0...qwen-asr-v0.2.1) (2026-02-22)


### Bug Fixes

* trigger patch release 0.2.1 for flutter ([b5785f9](https://github.com/huanglizhuo/QwenASR/commit/b5785f9e0a6e4cab3a4796bbd1bd401876ea5926))
* update the both library readme to mention this is WIP project ([139a591](https://github.com/huanglizhuo/QwenASR/commit/139a5915205083abc4b87fd0228ccf4c725c99c0))

## [0.2.0](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.1.2...qwen-asr-v0.2.0) (2026-02-22)


### Features

* add missing parameter to qwen asr offline model ([f56e8b1](https://github.com/huanglizhuo/QwenASR/commit/f56e8b1e58731344fad92a7ed38c59a9f09267f6))
* add missing parameter to qwen asr offline model ([6d1e38d](https://github.com/huanglizhuo/QwenASR/commit/6d1e38da19cbae46c2afe2e1af03a5d437679ef8))
