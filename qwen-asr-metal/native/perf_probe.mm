// Development-only performance probe. Never linked into the production library:
// it replicates the bridge ASR flow with extra command-buffer boundaries so each
// stage reports GPU time separately, and it calibrates machine rooflines.
#include "runtime.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

static std::vector<float> read_pcm(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot open pcm: ") + path);
    f.seekg(0, std::ios::end);
    size_t bytes = size_t(f.tellg());
    if (bytes == 0 || bytes % 4) throw std::runtime_error("invalid raw f32 pcm");
    std::vector<float> pcm(bytes / 4);
    f.seekg(0); f.read((char *)pcm.data(), bytes);
    if (!f) throw std::runtime_error("short pcm read");
    return pcm;
}
struct Phase { const char *name; double gpu_ms = 0, host_ms = 0; };
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
int main(int argc, const char **argv) { @autoreleasepool { try {
    if (argc < 5) throw std::runtime_error(
        "usage: perf_probe <package> <kernels> <pcm.f32> <phases|counters|roofline> [repeats]");
    std::string mode = argv[4];
    int repeats = argc > 5 ? std::stoi(argv[5]) : 3;
    if (repeats < 1 || repeats > 50) throw std::runtime_error("invalid repeats");
    MetalRuntime rt(argv[2]);
    ModelPackage model(rt.device, argv[1]);
    if (![model.index[@"task"] isEqual:@"asr"] || ![model.index[@"precision"] isEqual:@"hybrid"])
        throw std::runtime_error("probe expects the hybrid ASR package");
    const std::vector<int32_t> prefix{151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669};
    const std::vector<int32_t> suffix{151670, 151645, 198, 151644, 77091, 198, 11528, 6364, 151704};
    auto pcm = read_pcm(argv[3]);
    if (pcm.size() < 160 || pcm.size() > 480000) throw std::runtime_error("pcm out of range");
    size_t frames = pcm.size() / 160, audio_tokens = 0;
    for (size_t i = 0; i < frames; i += 100) audio_tokens += (std::min(size_t(100), frames - i) + 7) / 8;
    size_t total = prefix.size() + audio_tokens + suffix.size();
    DecoderCache cache(rt, total + 2048);

    if (mode == "roofline") {
        // Stream bandwidth via pure copy (read+write) and add (2R+1W) on ~256 MiB.
        size_t n = 64u << 20;
        auto a = rt.allocate({n}), b = rt.allocate({n});
        auto run = [&](const char *name, auto &&op) {
            for (int w = 0; w < 2; ++w) { rt.begin(); op(); rt.finish(); }
            double best = 1e9;
            for (int r = 0; r < 5; ++r) { rt.begin(); double g = op(); best = std::min(best, rt.finish()); }
            printf("\"%s\": {\"gpu_ms\": %.4f}, ", name, best);
        };
        // finish() returns GPU ms; reuse tensor leases by re-binding each call.
        printf("{");
        run("copy_512MB_rw", [&] { rt.copy(a, b); return 0; });
        run("add_768MB_rw", [&] { auto c = rt.add(a, b); return 0; });
        printf("}\n");
        return 0;
    }

    if (mode == "launch") {
        // Per-dispatch floor: N back-to-back trivial kernels in one encoder.
        Tensor a = rt.allocate({1024}), b = rt.allocate({1024});
        for (int w = 0; w < 2; ++w) { rt.begin(); for (int i = 0; i < 64; ++i) rt.copy(a, b); rt.finish(); }
        double best = 1e9;
        for (int r = 0; r < 10; ++r) {
            rt.begin(); double t = now_ms();
            for (int i = 0; i < 512; ++i) rt.copy(a, b);
            double g = rt.finish();
            best = std::min(best, g / 512 * 1000);
        }
        printf("{\"per_dispatch_us\": %.3f}\n", best);
        return 0;
    }

    if (mode == "kstorm") {
        // True in-stream per-kernel cost: each kernel type dispatched K times
        // back-to-back inside one serial encoder, mirroring the decode step.
        const std::string p0 = "thinker.model.layers.0.";
        Tensor x = rt.upload(std::vector<float>(1024, 0.1f).data(), 1, 1024);
        auto [qx, xs] = [&] { rt.begin(); auto r = rt.norm_quant(x, model.weight(p0 + "input_layernorm.weight"), 1e-6f); rt.finish(); return r; }();
        Tensor residual = rt.upload(std::vector<float>(1024, 0.2f).data(), 1, 1024);
        Tensor gate = rt.upload(std::vector<float>(3072, 0.1f).data(), 1, 3072);
        Tensor qkv_out = rt.upload(std::vector<float>(4096, 0.1f).data(), 1, 4096);
        Tensor q_row = rt.upload(std::vector<float>(2048, 0.1f).data(), 1, 2048);
        const int K = 112;
        auto storm = [&](const char *label, auto &&op) {
            for (int w = 0; w < 2; ++w) { rt.begin(); for (int i = 0; i < 8; ++i) op(); rt.finish(); }
            double best = 1e9;
            for (int r = 0; r < 5; ++r) {
                rt.begin(); double t = now_ms();
                for (int i = 0; i < K; ++i) op();
                double g = rt.finish();
                best = std::min(best, g * 1000 / K);
            }
            printf("  \"%s_us\": %.3f,\n", label, best);
        };
        printf("{\n");
        storm("copy_f32", [&] { rt.copy(x, residual); });
        storm("norm_quant", [&] { rt.norm_quant(x, model.weight(p0 + "input_layernorm.weight"), 1e-6f); });
        storm("qkv_w8a8", [&] { rt.qkv_w8a8(qx, xs, model, p0); });
        storm("rope_cache", [&] { rt.rope_cache(qkv_out, model, p0, cache.keys[0], cache.values[0], 0, 1e-6f, 10000.0f); });
        storm("causal_attention_quant", [&] { rt.causal_attention_quant(q_row, cache.keys[0], cache.values[0], 0); });
        auto [oq, os] = [&] { rt.begin(); auto r = rt.quantize_decode_input(q_row); rt.finish(); return r; }();
        storm("gemv_residual_o", [&] { rt.gemv_residual(oq, os, model, p0 + "self_attn.o_proj.weight", residual); });
        storm("gate_up_w8a8", [&] { rt.gate_up_w8a8(qx, xs, model, p0); });
        auto [gq, gs] = [&] { rt.begin(); auto r = rt.quantize_decode_input(gate); rt.finish(); return r; }();
        storm("quantize_3072", [&] { rt.quantize_decode_input(gate); });
        storm("gemv_residual_down", [&] { rt.gemv_residual(gq, gs, model, p0 + "mlp.down_proj.weight", residual); });
        storm("lm_head_gemv", [&] { rt.gemv_w8a8(qx, xs, model, "thinker.lm_head.weight"); });
        printf("  \"_k\": %d\n}\n", K);
        return 0;
    }

    auto run_flow = [&](std::vector<Phase> &phases, std::vector<double> &steps, bool split) {
        cache.reset();
        auto input = rt.upload(pcm.data(), 1, pcm.size());
        rt.begin();
        double t0 = now_ms();
        auto mel = rt.mel_spectrogram(input);
        if (split) phases.push_back({"mel", rt.finish(), now_ms() - t0}), rt.begin(), t0 = now_ms();
        auto encoded = encode_mel(rt, model, mel);
        if (split) phases.push_back({"encode", rt.finish(), now_ms() - t0}), rt.begin(), t0 = now_ms();
        auto before = rt.embedding(model, prefix), after = rt.embedding(model, suffix);
        auto prompt = rt.allocate({total, 1024});
        auto dest = prompt; dest.shape = before.shape; rt.copy(before, dest);
        dest.offset = prefix.size() * 1024 * 4; dest.shape = encoded.shape; rt.copy(encoded, dest);
        dest.offset = (prefix.size() + audio_tokens) * 1024 * 4; dest.shape = after.shape; rt.copy(after, dest);
        if (split) phases.push_back({"prompt", rt.finish(), now_ms() - t0}), rt.begin(), t0 = now_ms();
        auto prefill = prompt; prefill.shape[0] = total - 1;
        decoder_forward(rt, model, cache, prefill);
        if (split) phases.push_back({"prefill", rt.finish(), now_ms() - t0}), rt.begin(), t0 = now_ms();
        auto last = prompt; last.offset += (total - 1) * 1024 * 4; last.shape[0] = 1;
        auto hidden = decoder_forward(rt, model, cache, last);
        auto ids = rt.argmax(decoder_logits(rt, model, hidden));
        double g = rt.finish();
        if (split) phases.push_back({"first_token", g, now_ms() - t0});
        int32_t token = *(const int32_t *)ids.buffer.contents;
        for (int step = 0; step < 2047; ++step) {
            if (token == 151645 || token == 151643) break;
            double h0 = now_ms();
            rt.begin();
            auto embed = rt.embedding(model, {token});
            auto h = decoder_forward(rt, model, cache, embed);
            ids = rt.argmax(decoder_logits(rt, model, h));
            steps.push_back(rt.finish());
            token = *(const int32_t *)ids.buffer.contents;
            if (split) phases.push_back({"step_host", 0, now_ms() - h0});
        }
        return token;
    };

    if (mode == "difftest") {
        // Byte-level diff of old vs fused single-token layer pieces on real
        // post-prefill state. Host snapshots are taken right after each finish.
        std::vector<Phase> ph; std::vector<double> st;
        run_flow(ph, st, false);
        cache.reset();
        auto input = rt.upload(pcm.data(), 1, pcm.size());
        rt.begin();
        auto mel = rt.mel_spectrogram(input);
        auto encoded = encode_mel(rt, model, mel);
        auto before = rt.embedding(model, prefix), after = rt.embedding(model, suffix);
        auto prompt = rt.allocate({total, 1024});
        auto dest = prompt; dest.shape = before.shape; rt.copy(before, dest);
        dest.offset = prefix.size() * 1024 * 4; dest.shape = encoded.shape; rt.copy(encoded, dest);
        dest.offset = (prefix.size() + audio_tokens) * 1024 * 4; dest.shape = after.shape; rt.copy(after, dest);
        auto prefill = prompt; prefill.shape[0] = total - 1;
        decoder_forward(rt, model, cache, prefill);
        rt.finish();
        auto last = prompt; last.offset += (total - 1) * 1024 * 4; last.shape[0] = 1;
        const std::string p0 = "thinker.model.layers.0.";
        float epsilon = [model.index[@"config"][@"thinker_config"][@"text_config"][@"rms_norm_eps"] floatValue];
        float theta = [model.index[@"config"][@"thinker_config"][@"text_config"][@"rope_theta"] floatValue];
        size_t slot = cache.length;
        auto snap_f = [&](const Tensor &t) { const float *p=(const float *)t.buffer.contents + t.offset/4; return std::vector<float>(p, p + t.elements()); };
        auto snap_i8 = [&](const Tensor &t) { const uint8_t *p=(const uint8_t *)t.buffer.contents + t.offset; return std::vector<uint8_t>(p, p + t.elements()); };
        auto diff_f = [](const std::vector<float> &a, const std::vector<float> &b) {
            double maxd=0; size_t idx=0;
            for(size_t i=0;i<a.size() && i<b.size();++i){double d=fabs(a[i]-b[i]);if(d>maxd){maxd=d;idx=i;}}
            return std::pair<double,size_t>{maxd,idx};
        };
        auto diff_b = [](const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) {
            size_t n=0; for(size_t i=0;i<a.size() && i<b.size();++i) if(a[i]!=b[i])++n; return n;
        };
        // ---- OLD path, full layer 0, snapshots after each finish.
        std::vector<float> k_old, v_old, q_old, o_old, x1_old, g_old, xnew_old; std::vector<uint8_t> q8a_old, a8_old, g8_old; float xs_old, as_old, gs_old;
        {
            rt.begin();
            auto n = rt.norm(last, model.weight(p0 + "input_layernorm.weight"), nullptr, epsilon);
            auto qa = rt.quantize_decode_input(n);
            auto qkv = rt.qkv_w8a8(qa.first, qa.second, model, p0);
            auto q0 = qkv, k0 = qkv, v0 = qkv;
            q0.shape = {1,2048}; k0.offset = 2048*4; k0.shape = {1,1024}; v0.offset = 3072*4; v0.shape = {1,1024};
            auto rq = rt.head_norm_rope(q0, model.weight(p0+"self_attn.q_norm.weight"), 16, unsigned(slot), epsilon, theta);
            auto rk = rt.head_norm_rope(k0, model.weight(p0+"self_attn.k_norm.weight"), 8, unsigned(slot), epsilon, theta);
            auto kd = cache.keys[0], vd = cache.values[0];
            kd.offset += slot*1024*4; vd.offset += slot*1024*4;
            kd.shape = rk.shape; vd.shape = v0.shape;
            rt.copy(rk, kd); rt.copy(v0, vd);
            auto a = rt.causal_attention(rq, cache.keys[0], cache.values[0], unsigned(slot));
            auto aq = rt.quantize_decode_input(a);
            auto o = rt.gemv_w8a8(aq.first, aq.second, model, p0+"self_attn.o_proj.weight");
            auto x1 = rt.add(last, o);
            auto n2 = rt.norm(x1, model.weight(p0+"post_attention_layernorm.weight"), nullptr, epsilon);
            auto g = rt.decoder_gate(n2, model, p0);
            auto gq = rt.quantize_decode_input(g);
            auto d = rt.gemv_w8a8(gq.first, gq.second, model, p0+"mlp.down_proj.weight");
            auto xnew = rt.add(x1, d);
            rt.finish();
            q8a_old = snap_i8(qa.first); xs_old = *(const float *)((uint8_t *)qa.second.buffer.contents + qa.second.offset);
            q_old = snap_f(rq); k_old = snap_f(rk); v_old = snap_f(v0);
            a8_old = snap_i8(aq.first); as_old = *(const float *)((uint8_t *)aq.second.buffer.contents + aq.second.offset);
            o_old = snap_f(o); x1_old = snap_f(x1); g_old = snap_f(g);
            g8_old = snap_i8(gq.first); gs_old = *(const float *)((uint8_t *)gq.second.buffer.contents + gq.second.offset);
            xnew_old = snap_f(xnew);
        }
        // ---- FUSED path, same input, same cache slot (overwrites the slot).
        {
            rt.begin();
            auto fq = rt.norm_quant(last, model.weight(p0+"input_layernorm.weight"), epsilon);
            auto fqkv = rt.qkv_w8a8(fq.first, fq.second, model, p0);
            auto frq = rt.rope_cache(fqkv, model, p0, cache.keys[0], cache.values[0], unsigned(slot), epsilon, theta);
            auto faq = rt.causal_attention_quant(frq, cache.keys[0], cache.values[0], unsigned(slot));
            auto fo = rt.gemv_residual(faq.first, faq.second, model, p0+"self_attn.o_proj.weight", last);
            auto fnq = rt.norm_quant(fo, model.weight(p0+"post_attention_layernorm.weight"), epsilon);
            auto fg = rt.gate_up_w8a8(fnq.first, fnq.second, model, p0);
            auto fgq = rt.quantize_decode_input(fg);
            auto fx = rt.gemv_residual(fgq.first, fgq.second, model, p0+"mlp.down_proj.weight", fo);
            rt.finish();
            auto r1=diff_b(q8a_old, snap_i8(fq.first));
            float fxs = *(const float *)((uint8_t *)fq.second.buffer.contents + fq.second.offset);
            auto r2=diff_f(q_old, snap_f(frq));
            const float *kb=(const float *)cache.keys[0].buffer.contents + slot*1024;
            const float *vb=(const float *)cache.values[0].buffer.contents + slot*1024;
            auto r3=diff_f(k_old, std::vector<float>(kb,kb+1024));
            auto r4=diff_f(v_old, std::vector<float>(vb,vb+1024));
            auto r5=diff_b(a8_old, snap_i8(faq.first));
            float fas = *(const float *)((uint8_t *)faq.second.buffer.contents + faq.second.offset);
            auto r6=diff_f(o_old, snap_f(fo));
            auto r7=diff_f(x1_old, snap_f(fo)); // residual-inclusive compare
            auto r8=diff_f(g_old, snap_f(fg));
            auto r9=diff_b(g8_old, snap_i8(fgq.first));
            float fgs = *(const float *)((uint8_t *)fgq.second.buffer.contents + fgq.second.offset);
            auto r10=diff_f(xnew_old, snap_f(fx));
            printf("norm_q8 byte diffs=%zu  scale old=%.9g new=%.9g\n", r1, xs_old, fxs);
            printf("roped_q max|d|=%.9g @%zu   cache_k max|d|=%.9g @%zu   cache_v max|d|=%.9g @%zu\n", r2.first,r2.second,r3.first,r3.second,r4.first,r4.second);
            printf("attn_q8 byte diffs=%zu scale old=%.9g new=%.9g\n", r5, as_old, fas);
            printf("o_proj max|d|=%.9g @%zu   o+residual(x1) max|d|=%.9g @%zu\n", r6.first,r6.second,r7.first,r7.second);
            printf("gate max|d|=%.9g @%zu  gate_q8 diffs=%zu scale old=%.9g new=%.9g\n", r8.first,r8.second,r9,gs_old,fgs);
            printf("x_after_layer max|d|=%.9g @%zu\n", r10.first,r10.second);
        }
        return 0;
    }

    if (mode == "stepstorm") {
        // One full decode step per iteration cycling through ALL layer weights
        // (defeats L2 reuse that kstorm's single-layer repetition enjoys).
        std::vector<Phase> ph; std::vector<double> st;
        run_flow(ph, st, false);
        const float epsilon = [model.index[@"config"][@"thinker_config"][@"text_config"][@"rms_norm_eps"] floatValue];
        const float theta = [model.index[@"config"][@"thinker_config"][@"text_config"][@"rope_theta"] floatValue];
        Tensor x = rt.upload(std::vector<float>(1024, 0.05f).data(), 1, 1024);
        size_t slot = cache.length;
        const int STEPS = 4;
        for (int w = 0; w < 2; ++w) { rt.begin(); x = decoder_forward(rt, model, cache, x); rt.finish(); cache.length = slot; }
        double best = 1e9, encode_best = 1e9;
        for (int r = 0; r < 5; ++r) {
            cache.length = slot;
            double t0 = now_ms();
            rt.begin();
            x = decoder_forward(rt, model, cache, x);
            double t1 = now_ms();   // all dispatches encoded, not yet committed
            rt.finish();
            best = std::min(best, now_ms() - t0);
            encode_best = std::min(encode_best, t1 - t0);
            cache.length = slot;
        }
        printf("{\"layer_cold_step_ms\": %.4f, \"cpu_encode_ms\": %.4f}\n", best, encode_best);
        return 0;
    }

    if (mode == "phases") {
        run_flow(*(new std::vector<Phase>), *(new std::vector<double>), false); // warmup
        std::vector<Phase> phases; std::vector<double> steps;
        for (int r = 0; r < repeats; ++r) run_flow(phases, steps, true);
        double encode = 0, mel = 0, prompt = 0, prefill = 0, first = 0, step_host = 0;
        for (auto &p : phases) {
            std::string n = p.name;
            if (n == "mel") mel += p.gpu_ms; else if (n == "encode") encode += p.gpu_ms;
            else if (n == "prompt") prompt += p.gpu_ms; else if (n == "prefill") prefill += p.gpu_ms;
            else if (n == "first_token") first += p.gpu_ms; else step_host += p.host_ms;
        }
        double d = repeats;
        std::sort(steps.begin(), steps.end());
        printf("{\n");
        printf("  \"pcm_samples\": %zu, \"audio_tokens\": %zu, \"decode_steps\": %zu,\n",
               pcm.size(), audio_tokens, steps.size() / size_t(repeats));
        printf("  \"per_run_gpu_ms\": {\"mel\": %.3f, \"encode\": %.3f, \"prompt\": %.3f, \"prefill\": %.3f, \"first_token\": %.3f, \"decode_total\": %.3f},\n",
               mel / d, encode / d, prompt / d, prefill / d, first / d,
               std::accumulate(steps.begin(), steps.end(), 0.0) / d);
        printf("  \"decode_step_gpu_ms\": {\"p50\": %.4f, \"p90\": %.4f, \"min\": %.4f, \"max\": %.4f},\n",
               steps[steps.size() / 2], steps[size_t(steps.size() * 0.9)], steps.front(), steps.back());
        printf("  \"decode_step_host_ms\": %.4f,\n", step_host / double(steps.size()));
        printf("  \"total_gpu_ms\": %.3f\n", (mel + encode + prompt + prefill + first) / d +
               std::accumulate(steps.begin(), steps.end(), 0.0) / d);
        printf("}\n");
        return 0;
    }

    if (mode == "counters") {
        // Warm the full flow, then record per-kernel stage counters for the
        // encoder, the prefill, and a few hot decode steps. The profiling path
        // puts every kernel in its own pass: ticks locate hot kernels, the
        // phase mode above certifies end-to-end time.
        std::vector<Phase> p; std::vector<double> s;
        run_flow(p, s, false);
        rt.enable_profiling("build/perf-counters.jsonl");
        cache.reset();
        auto input = rt.upload(pcm.data(), 1, pcm.size());
        rt.begin();
        auto mel = rt.mel_spectrogram(input);
        auto encoded = encode_mel(rt, model, mel);
        printf("{\"encoder_gpu_ms\": %.4f}\n", rt.finish());
        rt.begin();
        auto before = rt.embedding(model, prefix), after = rt.embedding(model, suffix);
        auto prompt = rt.allocate({total, 1024});
        auto dest = prompt; dest.shape = before.shape; rt.copy(before, dest);
        dest.offset = prefix.size() * 1024 * 4; dest.shape = encoded.shape; rt.copy(encoded, dest);
        dest.offset = (prefix.size() + audio_tokens) * 1024 * 4; dest.shape = after.shape; rt.copy(after, dest);
        auto prefill = prompt; prefill.shape[0] = total - 1;
        decoder_forward(rt, model, cache, prefill);
        printf("{\"prefill_gpu_ms\": %.4f}\n", rt.finish());
        rt.begin();
        auto last = prompt; last.offset += (total - 1) * 1024 * 4; last.shape[0] = 1;
        auto hidden = decoder_forward(rt, model, cache, last);
        auto ids = rt.argmax(decoder_logits(rt, model, hidden));
        printf("{\"first_token_gpu_ms\": %.4f}\n", rt.finish());
        int32_t token = *(const int32_t *)ids.buffer.contents;
        for (int step = 0; step < 3 && token != 151645 && token != 151643; ++step) {
            rt.begin();
            auto embed = rt.embedding(model, {token});
            auto h = decoder_forward(rt, model, cache, embed);
            ids = rt.argmax(decoder_logits(rt, model, h));
            printf("{\"decode_step_gpu_ms\": %.4f}\n", rt.finish());
            token = *(const int32_t *)ids.buffer.contents;
        }
        puts("counters written to build/perf-counters.jsonl");
        return 0;
    }
    throw std::runtime_error("unknown mode: " + mode);
} catch (const std::exception &e) { fprintf(stderr, "perf probe failed: %s\n", e.what()); return 2; } } }
