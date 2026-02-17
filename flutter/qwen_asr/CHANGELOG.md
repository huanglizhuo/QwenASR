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
