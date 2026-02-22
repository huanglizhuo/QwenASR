# Changelog

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
