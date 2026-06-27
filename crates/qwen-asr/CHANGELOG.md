# Changelog

## [0.7.3](https://github.com/huanglizhuo/QwenASR/compare/qwen-asr-v0.7.2...qwen-asr-v0.7.3) (2026-06-27)


### Bug Fixes

* harden encoder BF16 scratch sizing for proj2 ([#33](https://github.com/huanglizhuo/QwenASR/issues/33)) ([4078f51](https://github.com/huanglizhuo/QwenASR/commit/4078f51c3c22a0800e02fa708d530bae73bcbea4))


### Performance Improvements

* keep weights as BF16 mmap views to fit on iOS ([#32](https://github.com/huanglizhuo/QwenASR/issues/32)) ([64fff76](https://github.com/huanglizhuo/QwenASR/commit/64fff76c1154318a211657d0f5076cc6aa5f79cc))

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
