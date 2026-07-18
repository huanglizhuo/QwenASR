import 'dart:typed_data';

/// Sample rate the ASR engine expects.
const int kEngineSampleRate = 16000;

/// Parse a 16-bit PCM WAV to **16 kHz mono** float samples (-1..1).
///
/// Reads the `fmt ` chunk for the source sample rate and channel count,
/// downmixes to mono, and linearly resamples to [kEngineSampleRate]. Mirrors
/// the Rust `audio::parse_wav_buffer` so the simulated-mic path accepts
/// arbitrary-rate WAVs (e.g. the 44.1 kHz bench clip). Returns an empty list
/// on malformed input.
Float32List parseWavTo16kMono(Uint8List bytes) {
  if (bytes.lengthInBytes < 44) return Float32List(0);
  final bd = ByteData.view(
    bytes.buffer,
    bytes.offsetInBytes,
    bytes.lengthInBytes,
  );
  if (bd.getUint32(0, Endian.big) != 0x52494646 /*RIFF*/ ||
      bd.getUint32(8, Endian.big) != 0x57415645 /*WAVE*/ ) {
    return Float32List(0);
  }

  var channels = 1;
  var srcRate = kEngineSampleRate;
  var bits = 16;
  var pos = 12;
  Float32List? mono;

  while (pos + 8 <= bytes.lengthInBytes) {
    final id = bd.getUint32(pos, Endian.big);
    final size = bd.getUint32(pos + 4, Endian.little);
    final body = pos + 8;
    if (id == 0x666d7420 /*fmt */ && body + 16 <= bytes.lengthInBytes) {
      channels = bd.getUint16(body + 2, Endian.little).clamp(1, 8);
      srcRate = bd.getUint32(body + 4, Endian.little);
      bits = bd.getUint16(body + 14, Endian.little);
    } else if (id == 0x64617461 /*data*/ ) {
      if (bits != 16) return Float32List(0); // only PCM16 supported
      final totalSamples = size ~/ 2;
      final frames = totalSamples ~/ channels;
      mono = Float32List(frames);
      for (var f = 0; f < frames; f++) {
        var acc = 0.0;
        for (var c = 0; c < channels; c++) {
          acc +=
              bd.getInt16(body + (f * channels + c) * 2, Endian.little) /
              32768.0;
        }
        mono[f] = acc / channels;
      }
      break;
    }
    pos = body + size + (size & 1);
  }

  if (mono == null) return Float32List(0);
  if (srcRate == kEngineSampleRate) return mono;
  return _resampleLinear(mono, srcRate, kEngineSampleRate);
}

Float32List _resampleLinear(Float32List input, int srcRate, int dstRate) {
  if (input.isEmpty) return input;
  final outLen = (input.length * dstRate / srcRate).floor();
  final out = Float32List(outLen);
  final ratio = srcRate / dstRate;
  for (var i = 0; i < outLen; i++) {
    final srcPos = i * ratio;
    final i0 = srcPos.floor();
    final i1 = (i0 + 1 < input.length) ? i0 + 1 : i0;
    final frac = srcPos - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}
