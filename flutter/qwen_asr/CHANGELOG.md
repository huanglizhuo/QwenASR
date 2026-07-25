# Changelog

## [0.4.0](https://github.com/huanglizhuo/QwenASR/compare/qwen_asr-v0.3.0...qwen_asr-v0.4.0) (2026-07-25)


### Features

* **example:** device file picker on the File tab ([c82a13a](https://github.com/huanglizhuo/QwenASR/commit/c82a13a87ed53c3d0154c7f9112845c74aa437db))
* **example:** push-to-talk Record tab ([e0e4292](https://github.com/huanglizhuo/QwenASR/commit/e0e42928227c4bff9dd79bd30088ac3b8a713f6c))
* **example:** redesign VAD Live as Silero-VAD + one-shot offline decode ([aecc60a](https://github.com/huanglizhuo/QwenASR/commit/aecc60a298f7df4d9d8ea45f97140b25e49aab5b))
* **flutter:** Android device support — dotprod rustflags, release permissions, metric hooks ([62a8894](https://github.com/huanglizhuo/QwenASR/commit/62a8894876e295c9dd9c334d914d206bcd80cc2f))
* **flutter:** AUTO_SIM_WAV accepts http(s) URLs; record Android device results ([59912e6](https://github.com/huanglizhuo/QwenASR/commit/59912e6046eb3a9d3ab40697a9eb68cea37eec63))
* **flutter:** iOS simulator E2E tests, static-link loader fix, cargokit dotprod fix ([ecc9acc](https://github.com/huanglizhuo/QwenASR/commit/ecc9accc60496d2a94389025b3e6df499229f496))
* **flutter:** model download + live mic streaming in example app ([2654d25](https://github.com/huanglizhuo/QwenASR/commit/2654d25ed3f43029598cee5e52b46dcd00442d8c))
* **flutter:** precompiled binary distribution and package rename to com.clothpath ([a41498f](https://github.com/huanglizhuo/QwenASR/commit/a41498f5ce7f0e8d214d94e44b9d94a4ba3745ab))
* **mobile:** gate INT8 decoder-prefill behind int8-prefill cargo feature; Android default-ON (full-set WER +3.3%) ([3aa1491](https://github.com/huanglizhuo/QwenASR/commit/3aa1491a6e749f24a03fcad6f42a45d9f7c4cb22))
* **mobile:** INT8 encoder weight GEMMs behind int8-encoder cargo feature (R13 stage 2) ([b11b1a7](https://github.com/huanglizhuo/QwenASR/commit/b11b1a7b3e818b484918b9b5bdb0226551c4205e))
* **stream:** decouple VAD segment reset from multilingual; 2s-only chunk ([f06a0de](https://github.com/huanglizhuo/QwenASR/commit/f06a0dec861b83b5a53dc68576120ef8ba1f4171))
* **streaming:** expose provisional (unfixed) tail for lower-latency UX ([e70c387](https://github.com/huanglizhuo/QwenASR/commit/e70c387d1d80cfcf8ac1397cecac7f9b42fa3009))
* **streaming:** opt-in multilingual per-utterance language re-detection ([d3b18c1](https://github.com/huanglizhuo/QwenASR/commit/d3b18c12099cd69e4bfa6be16f84f0cf4103d639))


### Bug Fixes

* **example:** flush finalize drain + record R13-Android-UX latencies ([d62a9f1](https://github.com/huanglizhuo/QwenASR/commit/d62a9f19ffadce67fbad3d7c9af6c6809f1e6856))
* **flutter:** live mic uses 2s engine chunks for real-time partials ([1a779d5](https://github.com/huanglizhuo/QwenASR/commit/1a779d5a8b812c278c32d7ba9444cf2f93239c4d))
* **flutter:** word-level offline reference compare + serialize test suites ([92a6866](https://github.com/huanglizhuo/QwenASR/commit/92a686634d7b3f9e01e61cc3c76d53f95fe423b5))


### Performance Improvements

* **mobile:** device RTF sweep — revert live chunk to 2s, keep threads=auto (INT8 winner 0.94x) ([9500748](https://github.com/huanglizhuo/QwenASR/commit/95007486e664e9190059760957fa8af995849936))

## [0.3.0](https://github.com/huanglizhuo/QwenASR/compare/qwen_asr-v0.2.6...qwen_asr-v0.3.0) (2026-02-23)


### Features

* update readme ([cde2178](https://github.com/huanglizhuo/QwenASR/commit/cde21787bb545e12c154045562883b9ced00514d))

## [0.2.6](https://github.com/huanglizhuo/QwenASR/compare/qwen_asr-v0.2.5...qwen_asr-v0.2.6) (2026-02-22)


### Bug Fixes

* update the release flow to support PAT ([2b9be6c](https://github.com/huanglizhuo/QwenASR/commit/2b9be6c21b7e74e51bf1d1f15e6959679db70542))

## [0.2.5](https://github.com/huanglizhuo/QwenASR/compare/qwen_asr-v0.2.4...qwen_asr-v0.2.5) (2026-02-22)


### Bug Fixes

* publish 0.2.3 with tag-driven flow ([3637ec8](https://github.com/huanglizhuo/QwenASR/commit/3637ec80f5519ecbd0a034f6c1f23f78156cd0fe))
* publish 0.2.3 with tag-driven flow ([e7bbd18](https://github.com/huanglizhuo/QwenASR/commit/e7bbd18dc009c3bd87f32e2346c196f65c618b19))

## [0.2.4](https://github.com/huanglizhuo/QwenASR/compare/qwen_asr-v0.2.3...qwen_asr-v0.2.4) (2026-02-22)


### Bug Fixes

* remove hardcoded version for local path dependency ([f9cf0d0](https://github.com/huanglizhuo/QwenASR/commit/f9cf0d0f83d179d0782c620a7ea34496bbb8522d))

## [0.2.3](https://github.com/huanglizhuo/QwenASR/compare/qwen_asr-v0.2.2...qwen_asr-v0.2.3) (2026-02-22)


### Bug Fixes

* remove non-existent flutter_rust_bridge_codegen dependency ([176cfa7](https://github.com/huanglizhuo/QwenASR/commit/176cfa7aad0a775bb6b8db487ea8c6c7b39f7758))

## [0.2.2](https://github.com/huanglizhuo/QwenASR/compare/qwen_asr-v0.2.1...qwen_asr-v0.2.2) (2026-02-22)


### Bug Fixes

* add flutter_rust_bridge_codegen to dev_dependencies for github actions ([c78a437](https://github.com/huanglizhuo/QwenASR/commit/c78a437a19baf586b036302f597957d91ac82510))

## [0.2.1](https://github.com/huanglizhuo/QwenASR/compare/qwen_asr-v0.2.0...qwen_asr-v0.2.1) (2026-02-22)


### Bug Fixes

* trigger patch release 0.2.1 for flutter ([b5785f9](https://github.com/huanglizhuo/QwenASR/commit/b5785f9e0a6e4cab3a4796bbd1bd401876ea5926))
* update the both library readme to mention this is WIP project ([139a591](https://github.com/huanglizhuo/QwenASR/commit/139a5915205083abc4b87fd0228ccf4c725c99c0))

## [0.2.0](https://github.com/huanglizhuo/QwenASR/compare/qwen_asr-v0.1.0...qwen_asr-v0.2.0) (2026-02-22)


### Features

* add missing parameter to qwen asr offline model ([f56e8b1](https://github.com/huanglizhuo/QwenASR/commit/f56e8b1e58731344fad92a7ed38c59a9f09267f6))
* add missing parameter to qwen asr offline model ([6d1e38d](https://github.com/huanglizhuo/QwenASR/commit/6d1e38da19cbae46c2afe2e1af03a5d437679ef8))

## 0.1.0

* Initial release.
* `QAsrEngine.load` — load a Qwen3-ASR model from a directory.
* `transcribeFile` — transcribe a WAV file by path.
* `transcribePcm` — transcribe raw Float32 PCM samples (16 kHz, mono).
* `transcribeWavBuffer` — transcribe from an in-memory WAV buffer.
* `setLanguage` — force a specific language or auto-detect.
* `setSegmentSec` — enable segmented mode for long audio.
* `perfStats` — retrieve timing stats from the last transcription.
* Platform support: iOS, Android, macOS.
