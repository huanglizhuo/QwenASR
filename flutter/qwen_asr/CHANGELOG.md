# Changelog

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
