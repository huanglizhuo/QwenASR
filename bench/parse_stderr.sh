#!/usr/bin/env bash
# Parse q-asr stderr output into key=value pairs.
# Usage: bench/parse_stderr.sh < stderr_file
#
# Expected stderr lines:
#   Inference: 1624 ms, 26 text tokens (16.01 tok/s, encoding: 511ms, decoding: 1113ms)
#   Audio: 11.0 s processed in 1.6 s (6.77x realtime)
#   [profile] op_name: 123.4ms (56 calls, 2.20ms avg)

while IFS= read -r line; do
    case "$line" in
        Inference:*)
            total_ms=$(echo "$line" | sed -n 's/^Inference: \([0-9.]*\) ms.*/\1/p')
            tokens=$(echo "$line" | sed -n 's/.*, \([0-9]*\) text tokens.*/\1/p')
            tokens_per_sec=$(echo "$line" | sed -n 's/.*(\([0-9.]*\) tok\/s.*/\1/p')
            encode_ms=$(echo "$line" | sed -n 's/.*encoding: \([0-9.]*\)ms.*/\1/p')
            decode_ms=$(echo "$line" | sed -n 's/.*decoding: \([0-9.]*\)ms.*/\1/p')
            echo "total_ms=$total_ms"
            echo "tokens=$tokens"
            echo "tokens_per_sec=$tokens_per_sec"
            echo "encode_ms=$encode_ms"
            echo "decode_ms=$decode_ms"
            ;;
        Audio:*)
            audio_duration=$(echo "$line" | sed -n 's/^Audio: \([0-9.]*\) s processed.*/\1/p')
            realtime_factor=$(echo "$line" | sed -n 's/.*(\([0-9.]*\)x realtime).*/\1/p')
            echo "audio_duration_s=$audio_duration"
            echo "realtime_factor=$realtime_factor"
            ;;
        "[profile] "*)
            # [profile] op_name: 123.4ms (56 calls, 2.20ms avg)
            op=$(echo "$line" | sed -n 's/\[profile\] \([^:]*\):.*/\1/p')
            ms=$(echo "$line" | sed -n 's/.*: \([0-9.]*\)ms.*/\1/p')
            echo "profile_${op}_ms=$ms"
            ;;
    esac
done
