// orchestrator-test/main.cpp — minimal end-to-end smoke test of the
// predictive expert prefetch orchestrator wired into a real llama.cpp run.
//
// Two-phase init pattern (model first, orchestrator+context second) so we
// can read n_layers/n_experts/hidden_dim from the model and set cb_eval on
// the context BEFORE the first llama_decode.
//
// Usage:
//   orchestrator-test -m model.gguf --orchestrator-predictor predictor_mlp_v1.bin \
//                     -p "Why is the sky blue?" -n 30

#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include "orchestrator.h"
#include "predictor.h"
#include "mode_b.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

struct test_args {
    std::string model_path;
    std::string prompt           = "Why is the sky blue?";
    std::string predictor_path;
    std::string hopfield_path;       // --orchestrator-hopfield PATH (L2)
    int         l2_tier2_top_n = 32; // --l2-tier2-top-n N
    int         tier2_size      = 0; // --tier2-size N (0 = disabled)
    bool        mode_b_graceful = false; // --mode-b-graceful: drop -INF mask
    bool        warmup_prompt_centroid = false; // --warmup-prompt-centroid (opt-in; needs layer-0 Hopfield)
    int         n_predict        = 30;
    int         n_gpu_layers      = 99;
    int         n_threads         = 24;
    int         ctx_size          = 4096;
    bool        override_exps_cpu = true;
    // Modes:
    //   --orchestrator-predictor PATH    => Mode B (predictor + recency)
    //   --orchestrator-recency-only      => Mode A (recency-only, no MLP)
    //   neither                          => no orchestrator (baseline)
    bool        orchestrator_recency_only = false;
    int         l1_capacity                = 32;
    int         l2_capacity                = 80;
    int         prefetch_top_k             = 16;
    bool        global_l1                  = false;
    float       l2_promote_threshold       = -1.0f;   // < 0 = disabled
    std::string dump_trace_path;          // --dump-trace PATH
    std::string prompts_file;             // --prompts-file PATH (1 prompt per line)
    // SPIKE: between prompt 0 and prompt 1, zero an expert tensor.
    // Tests whether ggml respects mid-execution tensor data mutation.
    // Format: "L,E" where L=layer index, E=expert index (within that layer's
    // ffn_*_exps), e.g. "5,0" zeros the FIRST expert of layer 5.
    // Empty/unset = no mutation.
    std::string spike_mutate;
    // Stage 7 mode selection. Determines where expert weights physically live.
    //   "auto"     — pick at runtime (commit #3 will implement detection;
    //                until then, defaults to "slot" behavior).
    //   "resident" — experts loaded to CUDA buffer (VRAM). Mode A.
    //                Requires VRAM >= expert footprint. For DGX Spark / H100 /
    //                M1 Max-class hardware. Equivalent to --no-override-exps.
    //   "slot"     — experts pinned to CPU buffer; orchestrator's tiered
    //                cache manages a VRAM subset. Mode B. Required on
    //                consumer GPUs where total expert footprint exceeds VRAM.
    std::string orchestrator_mode = "auto";
};

static void print_usage(const char * prog) {
    std::printf(
        "usage: %s -m MODEL.gguf [options]\n"
        "  -m PATH                          model GGUF\n"
        "  -p TEXT                          prompt (default: 'Why is the sky blue?')\n"
        "  -n N                             tokens to predict (default 30)\n"
        "  -ngl N                           layers on GPU (default 99 = all)\n"
        "  -t N                             CPU threads (default 24)\n"
        "  -c N                             context size (default 4096)\n"
        "  --orchestrator-predictor PATH    .bin predictor file (Mode B: MLP + recency)\n"
        "  --orchestrator-hopfield PATH     L2 Hopfield predictor .bin (per-window prefetch)\n"
        "  --l2-tier2-top-n N               experts per layer Hopfield prefetches (default 32)\n"
        "  --tier2-size N                   Mode B: pinned host Tier 2 capacity per layer (0 = off)\n"
        "  --mode-b-graceful                drop -INF mask (gate selects freely; pair with Hopfield)\n"
        "  --orchestrator-recency-only      enable orchestrator without predictor (Mode A)\n"
        "  --l1 N                           L1 (VRAM) cache capacity (per layer; total if --global-l1) (default 32)\n"
        "  --l2 N                           L2 (RAM)  cache capacity per layer (default 80)\n"
        "  --prefetch-top-k N               experts the MLP prefetches per call (default 16)\n"
        "  --global-l1                      L1 is a global pool of slots across all layers\n"
        "                                   (live-mode physical-VRAM model; --l1 = total slots)\n"
        "  --l2-promote-threshold X         enable L2->L1 promotion when predictor confidence >= X\n"
        "                                   (X in (0,1]; default disabled = skip-if-resident)\n"
        "  --dump-trace PATH                write per-(layer, token) trace to PATH (binary)\n"
        "  --prompts-file PATH              run a list of prompts (one per line) sequentially\n"
        "  --no-override-exps               do NOT pin experts to CPU (default: do pin)\n"
        "  --orchestrator-mode MODE         live-mode strategy: auto | resident | slot (default auto)\n"
        "                                   resident = experts in VRAM (Mode A; needs big VRAM)\n"
        "                                   slot     = experts on CPU, orchestrator manages VRAM cache\n"
        "                                   auto     = pick at runtime (today: behaves as slot)\n",
        prog);
}

static bool parse_args(int argc, char ** argv, test_args & a) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto need = [&](const char * msg) {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", msg); return false; }
            return true;
        };
        if      (arg == "-m"   && need("-m"))   a.model_path     = argv[++i];
        else if (arg == "-p"   && need("-p"))   a.prompt         = argv[++i];
        else if (arg == "-n"   && need("-n"))   a.n_predict      = std::atoi(argv[++i]);
        else if (arg == "-ngl" && need("-ngl")) a.n_gpu_layers   = std::atoi(argv[++i]);
        else if (arg == "-t"   && need("-t"))   a.n_threads      = std::atoi(argv[++i]);
        else if (arg == "-c"   && need("-c"))   a.ctx_size       = std::atoi(argv[++i]);
        else if (arg == "--orchestrator-predictor" && need("--orchestrator-predictor"))
                                                a.predictor_path = argv[++i];
        else if (arg == "--orchestrator-hopfield" && need("--orchestrator-hopfield"))
                                                a.hopfield_path = argv[++i];
        else if (arg == "--l2-tier2-top-n" && need("--l2-tier2-top-n"))
                                                a.l2_tier2_top_n = std::atoi(argv[++i]);
        else if (arg == "--tier2-size" && need("--tier2-size"))
                                                a.tier2_size = std::atoi(argv[++i]);
        else if (arg == "--mode-b-graceful")    a.mode_b_graceful = true;
        else if (arg == "--warmup-prompt-centroid") a.warmup_prompt_centroid = true;
        else if (arg == "--orchestrator-recency-only") a.orchestrator_recency_only = true;
        else if (arg == "--l1" && need("--l1")) a.l1_capacity = std::atoi(argv[++i]);
        else if (arg == "--l2" && need("--l2")) a.l2_capacity = std::atoi(argv[++i]);
        else if (arg == "--prefetch-top-k" && need("--prefetch-top-k"))
                                                a.prefetch_top_k = std::atoi(argv[++i]);
        else if (arg == "--global-l1")          a.global_l1 = true;
        else if (arg == "--l2-promote-threshold" && need("--l2-promote-threshold"))
                                                a.l2_promote_threshold = (float)std::atof(argv[++i]);
        else if (arg == "--spike-mutate" && need("--spike-mutate"))
                                                a.spike_mutate = argv[++i];
        else if (arg == "--orchestrator-mode" && need("--orchestrator-mode")) {
            a.orchestrator_mode = argv[++i];
            if (a.orchestrator_mode != "auto" && a.orchestrator_mode != "resident" && a.orchestrator_mode != "slot") {
                std::fprintf(stderr, "invalid --orchestrator-mode (must be auto|resident|slot)\n");
                return false;
            }
        }
        else if (arg == "--dump-trace" && need("--dump-trace"))
                                                a.dump_trace_path = argv[++i];
        else if (arg == "--prompts-file" && need("--prompts-file"))
                                                a.prompts_file = argv[++i];
        else if (arg == "--no-override-exps")    a.override_exps_cpu = false;
        else if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return false; }
        else { std::fprintf(stderr, "unknown arg: %s\n", arg.c_str()); return false; }
    }
    if (a.model_path.empty()) { print_usage(argv[0]); return false; }
    return true;
}

// Compute a hidden_dim-sized centroid by averaging token_embd rows for the
// given token IDs. Handles quantized embedding tables via the ggml type
// traits to_float helper.
//
// Returns true on success. Used for pre-prefill Hopfield warmup so the
// cache is populated BEFORE the first llama_decode runs.
static bool compute_prompt_centroid(llama_model * model,
                                    const std::vector<llama_token> & tokens,
                                    int32_t hidden_dim,
                                    std::vector<float> & out_centroid) {
    ggml_tensor * te = llama_model_get_tensor(model, "token_embd.weight");
    if (!te) {
        // Some archs use a different name (e.g., output.weight as input embed).
        te = llama_model_get_tensor(model, "tok_embeddings.weight");
    }
    if (!te) return false;
    if (te->ne[0] != (int64_t)hidden_dim) return false;

    const auto * traits = ggml_get_type_traits(te->type);
    if (!traits || !traits->to_float) return false;

    const size_t row_bytes = ggml_row_size(te->type, te->ne[0]);
    std::vector<uint8_t> raw(row_bytes);
    std::vector<float>   row_f((size_t)hidden_dim);
    out_centroid.assign((size_t)hidden_dim, 0.0f);

    int32_t kept = 0;
    for (llama_token t : tokens) {
        if (t < 0 || (int64_t)t >= te->ne[1]) continue;
        ggml_backend_tensor_get(te, raw.data(), (size_t)t * row_bytes, row_bytes);
        traits->to_float(raw.data(), row_f.data(), (int64_t)hidden_dim);
        for (int32_t i = 0; i < hidden_dim; i++) out_centroid[i] += row_f[i];
        kept++;
    }
    if (kept == 0) return false;
    const float inv = 1.0f / (float)kept;
    for (int32_t i = 0; i < hidden_dim; i++) out_centroid[i] *= inv;
    return true;
}

static int read_n_experts_from_model(llama_model * model) {
    // Try to read the qwen3moe.expert_count metadata. Default 128 if missing.
    char buf[64];
    int len = llama_model_meta_val_str(model, "qwen3moe.expert_count", buf, sizeof(buf));
    if (len > 0) return std::atoi(buf);
    // Try alt key.
    len = llama_model_meta_val_str(model, "general.expert_count", buf, sizeof(buf));
    if (len > 0) return std::atoi(buf);
    std::fprintf(stderr, "WARN: could not read expert count from model metadata; defaulting to 128\n");
    return 128;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    test_args args;
    if (!parse_args(argc, argv, args)) return 1;

    llama_backend_init();

    // ---- Phase 1: load model ----
    // Stage 7 mode selection: resident = experts to VRAM, slot = experts on CPU.
    // "auto" today means slot (commit #3 will add VRAM detection). Resident mode
    // overrides --no-override-exps to off (don't pin to CPU); slot mode overrides
    // it on (do pin, current default behavior).
    if (args.orchestrator_mode == "resident") {
        if (args.override_exps_cpu) {
            std::printf("[stage7] mode=resident: experts will load to GPU buffer (overriding default CPU pin)\n");
        }
        args.override_exps_cpu = false;
        std::fprintf(stderr,
            "[stage7] WARNING: resident mode currently has no live-mode handling for\n"
            "[stage7]   expert footprint > VRAM. If the model's expert weights exceed\n"
            "[stage7]   available VRAM minus dense+KV+compute headroom, model load will\n"
            "[stage7]   fail with cudaMalloc OOM. For consumer GPUs running large MoE\n"
            "[stage7]   (e.g., 235B Q3 on 16 GB), use --orchestrator-mode slot instead.\n");
    } else if (args.orchestrator_mode == "slot") {
        args.override_exps_cpu = true;
    }
    // mode == "auto": leave override_exps_cpu at its current value (default true,
    // i.e. behaves as slot). Auto-select with runtime VRAM detection deferred to
    // post-v0.1 polish per the focused-push plan (Chris, 2026-05-02).
    std::printf("[stage7] orchestrator-mode=%s  experts on %s\n",
                args.orchestrator_mode.c_str(),
                args.override_exps_cpu ? "CPU (slot-managed)" : "GPU (resident)");

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
    // Spike test mutates expert tensors via ggml_backend_tensor_set, which
    // segfaults on mmap'd-read-only model storage. Disable mmap when spike
    // is requested. Has no effect in production (live mode owns the buffer).
    if (!args.spike_mutate.empty()) {
        mparams.use_mmap = false;
        std::printf("[SPIKE] mmap disabled for spike test\n");
    }
    // Apply the same expert-on-CPU override the baseline benchmark uses.
    // This is via the tensor-buft override mechanism. The CLI-equivalent of
    // --override-tensor "exps.=CPU" maps to this struct.
    // Pin all expert tensors to CPU buffer type — same effect as
    // `--override-tensor "\.ffn_(up|down|gate|gate_up)_(ch|)exps=CPU"` on
    // llama-cli. Required for big MoE models that don't fit in VRAM whole.
    std::vector<llama_model_tensor_buft_override> overrides;
    static const char * EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
    if (args.override_exps_cpu) {
        overrides.push_back({EXPS_REGEX, ggml_backend_cpu_buffer_type()});
        overrides.push_back({nullptr, nullptr});  // sentinel
        mparams.tensor_buft_overrides = overrides.data();
        std::printf("Override: experts pinned to CPU buffer type.\n");
    }

    std::printf("Loading model: %s\n", args.model_path.c_str());
    llama_model * model = llama_model_load_from_file(args.model_path.c_str(), mparams);
    if (!model) { std::fprintf(stderr, "failed to load model\n"); return 1; }

    const int32_t n_layers   = llama_model_n_layer(model);
    const int32_t hidden_dim = llama_model_n_embd(model);
    const int32_t n_experts  = read_n_experts_from_model(model);
    std::printf("Model dims: n_layers=%d, hidden_dim=%d, n_experts=%d\n",
                n_layers, hidden_dim, n_experts);

    // ---- Phase 1.5: Mode B context (slot-mapped live mode) ----
    // Initialize the singleton ModeBContext so qwen3moe.cpp's graph builder
    // sees an active slot_map at graph build time. For commit #4 the slot
    // weight tensors aren't allocated yet (commit #5 follows), so the
    // slot_map defaults to identity (slot_map[e] = e for e in [0, n_slot),
    // others = 0), which is a no-op behaviorally — same outputs as baseline.
    // The full Mode B path lights up in commit #5 + #6.
    // BUGFIX (2026-05-02): auto mode used to spawn mode_b_ctx unconditionally,
    // which broke baseline runs (no --orchestrator-mode flag). Until commit #3
    // adds runtime VRAM-vs-expert detection for auto mode, only explicit "slot"
    // engages Mode B. "auto" without explicit flag = baseline path.
    std::unique_ptr<moe_orch::ModeBContext> mode_b_ctx;
    if (args.orchestrator_mode == "slot") {
        // Mode B's n_slot reuses the existing --l1 capacity flag for natural
        // continuity with the shadow-mode sweeps (Phases 5/6). Default 32.
        const int n_slot = args.l1_capacity;
        mode_b_ctx = std::make_unique<moe_orch::ModeBContext>();
        if (!mode_b_ctx->init(model, n_layers, n_experts, n_slot, /*device_id=*/0)) {
            std::fprintf(stderr, "[stage7] ModeBContext init FAILED — falling back to no Mode B\n");
            mode_b_ctx.reset();
        } else {
            mode_b_ctx->attach_model(model);
            mode_b_ctx->set_graceful_mask(args.mode_b_graceful);
            moe_orch::set_mode_b_context(mode_b_ctx.get());
            std::printf("[stage7] Mode B context active for graph build (graceful_mask=%s)\n",
                        args.mode_b_graceful ? "ON" : "OFF");
            if (args.tier2_size > 0) {
                if (!mode_b_ctx->init_tier2(args.tier2_size)) {
                    std::fprintf(stderr, "[stage7] Tier 2 init FAILED — continuing without\n");
                }
            }
            // When Mode B is active, force orchestrator-recency-only so the
            // cb_eval bridge gets wired (we use it to observe routing for
            // Mode B's slot cache). Predictor still optional via the existing
            // --orchestrator-predictor flag for warm-up acceleration.
            if (args.predictor_path.empty() && !args.orchestrator_recency_only) {
                args.orchestrator_recency_only = true;
                std::printf("[stage7]   (auto-enabled --orchestrator-recency-only for cb_eval observation)\n");
            }
        }
    }

    // ---- Phase 2: build orchestrator (if requested) ----
    std::unique_ptr<moe_orch::Orchestrator> orchestrator;
    const bool orch_enabled = !args.predictor_path.empty()
                            || args.orchestrator_recency_only
                            || !args.dump_trace_path.empty();
    const char * mode_label = "BASELINE (no orchestrator)";
    if (orch_enabled) {
        moe_orch::OrchestratorConfig cfg;
        cfg.shadow_mode = true;
        cfg.l1_capacity = args.l1_capacity;
        cfg.l2_capacity = args.l2_capacity;
        cfg.l1_prefetch_top_k = args.prefetch_top_k;
        cfg.global_l1 = args.global_l1;
        cfg.l2_promote_threshold = args.l2_promote_threshold;
        cfg.dump_trace_path = args.dump_trace_path;
        cfg.l2_tier2_top_n = args.l2_tier2_top_n;
        orchestrator = std::make_unique<moe_orch::Orchestrator>(
            cfg, args.predictor_path, args.hopfield_path,
            n_layers, n_experts, hidden_dim);
        if (!args.dump_trace_path.empty() && args.predictor_path.empty() && !args.orchestrator_recency_only) {
            mode_label = "TRACE-CAPTURE (no cache stats meaningful)";
        } else {
            mode_label = args.predictor_path.empty()
                ? "MODE A (recency-only)"
                : "MODE B (recency + MLP predictor)";
        }
        std::printf("Orchestrator: %s  l1=%d%s  l2=%d%s\n",
                    mode_label,
                    args.l1_capacity, args.global_l1 ? " (global)" : "/layer",
                    args.l2_capacity,
                    args.l2_promote_threshold >= 0.0f ? "  l2->l1 gated" : "");
        if (args.l2_promote_threshold >= 0.0f) {
            std::printf("  l2->l1 promote threshold: %.3f\n", args.l2_promote_threshold);
        }
        if (!args.predictor_path.empty()) {
            std::printf("  L1 predictor (MLP): %s\n", args.predictor_path.c_str());
        }
        if (!args.hopfield_path.empty()) {
            std::printf("  L2 predictor (Hopfield): %s  tier2_top_n=%d\n",
                        args.hopfield_path.c_str(), args.l2_tier2_top_n);
        }
        if (!args.dump_trace_path.empty()) {
            std::printf("  trace dump: %s\n", args.dump_trace_path.c_str());
        }
    } else {
        std::printf("Orchestrator: %s\n", mode_label);
    }

    // ---- Phase 3: build context with cb_eval wired ----
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = args.ctx_size;
    cparams.n_batch     = 512;
    cparams.n_threads   = args.n_threads;
    cparams.n_threads_batch = args.n_threads;
    if (orchestrator) {
        cparams.cb_eval           = ggml_eval_callback_orchestrator;
        cparams.cb_eval_user_data = orchestrator.get();
        std::printf("Eval callback wired to orchestrator.\n");
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::fprintf(stderr, "failed to init context\n"); llama_model_free(model); return 1; }

    // ---- Phase 4: tokenize + run ----
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    // Per-prompt latency record. Captured for each prompt and aggregated at end.
    struct PromptLatency {
        int    n_prompt_tokens;
        int    n_generated_tokens;
        double ttft_ms;                 // prefill llama_decode + argmax for first token
        double prefill_decode_ms;       // just the prefill llama_decode (no argmax/sampling)
        double generation_total_ms;     // from after first token to end of generation
        double steady_state_tok_per_s;  // (n_generated - 1) / generation_total
        std::vector<double> per_token_ms; // post-first-token decode latencies (size = n_generated - 1)
    };
    std::vector<PromptLatency> latencies;
    latencies.reserve(64);

    auto run_one_prompt = [&](const std::string & prompt) -> bool {
        std::vector<llama_token> tokens(prompt.size() + 16);
        int n_tok = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                    tokens.data(), (int)tokens.size(),
                                    add_bos, /*parse_special=*/true);
        if (n_tok < 0) { std::fprintf(stderr, "tokenize failed\n"); return false; }
        tokens.resize(n_tok);

        std::printf(">>> [%d toks in] ", n_tok);
        std::fflush(stdout);

        PromptLatency lat;
        lat.n_prompt_tokens = n_tok;
        lat.n_generated_tokens = 0;
        lat.per_token_ms.reserve(args.n_predict);

        // ---- Pre-prefill Hopfield warmup (opt-in via --warmup-prompt-centroid) ----
        // Computes prompt centroid from token_embd rows and runs Hopfield
        // retrieval against it. Only useful when Hopfield was TRAINED on
        // tok_embd-output space (early layer hidden states). For Hopfield
        // trained on mid-network layer hidden states (e.g., layer 12 input),
        // this warmup uses a different representation than the training keys
        // and produces noisy retrieval — leave OFF in that case.
        if (args.warmup_prompt_centroid && orchestrator && !args.hopfield_path.empty()) {
            std::vector<float> centroid;
            if (compute_prompt_centroid(model, tokens, hidden_dim, centroid)) {
                orchestrator->warm_up_from_prompt_centroid(centroid.data());
                if (mode_b_ctx) mode_b_ctx->refresh_slots();
            } else {
                std::fprintf(stderr, "[stage7] prompt-centroid warmup skipped "
                                     "(token_embd not found or shape mismatch)\n");
            }
        }

        // ---- Prefill (= TTFT-dominant work) ----
        const auto t_prefill_start = std::chrono::steady_clock::now();
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tok))) {
            std::fprintf(stderr, "prefill failed\n"); return false;
        }
        // Mode B: refresh slots based on what experts the prefill observed.
        if (mode_b_ctx) mode_b_ctx->refresh_slots();
        const auto t_prefill_end = std::chrono::steady_clock::now();
        lat.prefill_decode_ms =
            std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

        // ---- Generation loop ----
        // First-iteration timing: prefill_end → first_token_picked ⇒ TTFT.
        // Subsequent iterations: each timed individually for steady-state percentiles.
        std::chrono::steady_clock::time_point t_first_token = t_prefill_end;  // updated below
        std::chrono::steady_clock::time_point t_gen_start;                    // = t_first_token

        for (int i = 0; i < args.n_predict; i++) {
            const auto t_iter_start = std::chrono::steady_clock::now();

            const float * logits = llama_get_logits_ith(ctx, -1);
            int n_vocab = llama_vocab_n_tokens(vocab);
            llama_token best = 0;
            float       best_l = logits[0];
            for (int t = 1; t < n_vocab; t++) {
                if (logits[t] > best_l) { best_l = logits[t]; best = t; }
            }
            if (llama_vocab_is_eog(vocab, best)) break;

            char piece[128];
            int n = llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, true);
            if (n > 0) { std::fwrite(piece, 1, n, stdout); std::fflush(stdout); }

            if (i == 0) {
                // After argmax + emit of the first token = TTFT moment.
                t_first_token = std::chrono::steady_clock::now();
                t_gen_start   = t_first_token;
                lat.ttft_ms = std::chrono::duration<double, std::milli>(
                    t_first_token - t_prefill_start).count();
            }

            if (llama_decode(ctx, llama_batch_get_one(&best, 1))) {
                std::fprintf(stderr, "decode step %d failed\n", i); return false;
            }
            // Mode B: refresh slots between tokens. This is the crux of
            // commit #6 — page in any newly-seen experts that the slot
            // cache wants but doesn't yet have.
            if (mode_b_ctx) mode_b_ctx->refresh_slots();

            const auto t_iter_end = std::chrono::steady_clock::now();
            if (i > 0) {
                // Steady-state token: time from end of previous iter to end of this iter
                // includes argmax, emit, and decode. That's what the user perceives as
                // "time per token after the first one."
                lat.per_token_ms.push_back(std::chrono::duration<double, std::milli>(
                    t_iter_end - t_iter_start).count());
            }
            lat.n_generated_tokens++;
        }
        const auto t_gen_end = std::chrono::steady_clock::now();
        lat.generation_total_ms = std::chrono::duration<double, std::milli>(
            t_gen_end - t_gen_start).count();
        lat.steady_state_tok_per_s = (lat.n_generated_tokens > 1)
            ? ((double)(lat.n_generated_tokens - 1) / (lat.generation_total_ms / 1000.0))
            : 0.0;

        std::printf("\n<<<\n");
        std::printf("    [latency] TTFT=%.0fms  steady=%.2f tok/s  gen=%d toks  prefill=%.0fms  prompt=%d toks\n",
                    lat.ttft_ms, lat.steady_state_tok_per_s,
                    lat.n_generated_tokens, lat.prefill_decode_ms, lat.n_prompt_tokens);
        std::fflush(stdout);

        latencies.push_back(std::move(lat));
        return true;
    };

    // Build prompt list. If --prompts-file is set, read one prompt per non-empty
    // line; otherwise the single -p prompt.
    std::vector<std::string> prompts;
    if (!args.prompts_file.empty()) {
        std::ifstream f(args.prompts_file);
        if (!f) {
            std::fprintf(stderr, "failed to open prompts file: %s\n", args.prompts_file.c_str());
            llama_free(ctx); llama_model_free(model); return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) prompts.push_back(line);
        }
        std::printf("Loaded %d prompts from %s\n", (int)prompts.size(), args.prompts_file.c_str());
    } else {
        prompts.push_back(args.prompt);
    }

    // SPIKE state — populated when --spike-mutate is set.
    // We capture the original tensor contents before mutation so we can
    // restore + verify ggml_backend_tensor_set is what's being read.
    struct SpikeState {
        bool            active = false;
        int             layer = -1;
        int             expert = -1;
        struct ggml_tensor * t = nullptr;
        std::vector<uint8_t> orig_bytes;
    };
    SpikeState spike;
    if (!args.spike_mutate.empty()) {
        size_t comma = args.spike_mutate.find(',');
        if (comma != std::string::npos) {
            spike.layer  = std::atoi(args.spike_mutate.substr(0, comma).c_str());
            spike.expert = std::atoi(args.spike_mutate.substr(comma + 1).c_str());
            char tname[64];
            std::snprintf(tname, sizeof(tname), "blk.%d.ffn_up_exps.weight", spike.layer);
            spike.t = llama_model_get_tensor(model, tname);
            if (!spike.t) {
                std::fprintf(stderr, "[SPIKE] tensor %s not found — disabled\n", tname);
            } else {
                spike.active = true;
                size_t total = ggml_nbytes(spike.t);
                spike.orig_bytes.resize(total);
                ggml_backend_tensor_get(spike.t, spike.orig_bytes.data(), 0, total);
                std::printf("[SPIKE] armed: layer=%d expert=%d tensor=%s size=%zu bytes\n",
                            spike.layer, spike.expert, tname, total);
            }
        }
    }

    for (size_t pi = 0; pi < prompts.size(); pi++) {
        if (prompts.size() > 1) {
            std::printf("\n--- prompt %d/%d ---\n", (int)(pi + 1), (int)prompts.size());
            llama_memory_clear(llama_get_memory(ctx), true);
            if (orchestrator) orchestrator->clear_cache_events();
        }

        // SPIKE: before prompt index 1, mutate the expert tensor.
        // Strategy: zero the WHOLE expert weights tensor for layer L. This
        // affects ALL experts at that layer, so generated tokens will be
        // garbage if the mutation takes effect. If the mutation is ignored
        // (cached pointers), prompt 1 output will match prompt 0.
        if (spike.active && pi == 1) {
            std::vector<uint8_t> zeros(spike.orig_bytes.size(), 0);
            ggml_backend_tensor_set(spike.t, zeros.data(), 0, zeros.size());
            std::printf("[SPIKE] mutated tensor to all-zeros before prompt %d\n", (int)pi);
        }

        if (!run_one_prompt(prompts[pi])) break;

        // SPIKE: restore the tensor after prompt 1 runs.
        if (spike.active && pi == 1) {
            ggml_backend_tensor_set(spike.t, spike.orig_bytes.data(), 0, spike.orig_bytes.size());
            std::printf("[SPIKE] restored original tensor contents after prompt %d\n", (int)pi);
        }
    }
    std::printf("\n");

    llama_perf_context_print(ctx);

    if (orchestrator) {
        auto s = orchestrator->cache_stats();
        std::printf("\n=== Orchestrator cache stats - %s ===\n", mode_label);
        std::printf("L1 hits:           %llu\n", (unsigned long long)s.hits_L1);
        std::printf("L2 hits:           %llu\n", (unsigned long long)s.hits_L2);
        std::printf("L3 misses:         %llu\n", (unsigned long long)s.misses_L3);
        std::printf("L1 hit rate:       %.3f\n", s.l1_hit_rate);
        const uint64_t total = s.hits_L1 + s.hits_L2 + s.misses_L3;
        const double l12_rate = total ? (double)(s.hits_L1 + s.hits_L2) / (double)total : 0.0;
        std::printf("L1+L2 hit rate:    %.3f\n", l12_rate);
        std::printf("Promotions to L1:  %llu\n", (unsigned long long)s.promotions_to_L1);
        std::printf("Evictions L1->L2:  %llu\n", (unsigned long long)s.evictions_L1_to_L2);
        std::printf("Evictions L2->L3:  %llu\n", (unsigned long long)s.evictions_L2_to_L3);
        std::printf("L1 predictor calls: %llu\n", (unsigned long long)orchestrator->l1_predictor_calls());
        std::printf("L1 predictor avg:   %.3f ms\n", orchestrator->l1_predictor_avg_ms());

        // Per-layer breakdown — find the bottleneck layers (lowest L1+L2 rates).
        std::printf("\n--- Per-layer L1 / L1+L2 hit rate (lowest 8 + highest 8) ---\n");
        std::vector<std::tuple<double, double, int, uint64_t>> rows;  // (l1_rate, l12_rate, layer, total)
        rows.reserve(s.per_layer_hits_L1.size());
        for (size_t L = 0; L < s.per_layer_hits_L1.size(); L++) {
            const uint64_t h1 = s.per_layer_hits_L1[L];
            const uint64_t h2 = s.per_layer_hits_L2[L];
            const uint64_t m3 = s.per_layer_misses_L3[L];
            const uint64_t tot = h1 + h2 + m3;
            if (tot == 0) continue;
            const double l1r  = (double)h1 / (double)tot;
            const double l12r = (double)(h1 + h2) / (double)tot;
            rows.emplace_back(l1r, l12r, (int)L, tot);
        }
        // Sort ascending by L1+L2 rate to surface bottleneck layers.
        std::sort(rows.begin(), rows.end(),
                  [](const auto & a, const auto & b) { return std::get<1>(a) < std::get<1>(b); });
        const size_t n_show = std::min<size_t>(8, rows.size());
        std::printf("Layer  L1     L1+L2  accesses   (lowest L1+L2)\n");
        for (size_t i = 0; i < n_show; i++) {
            std::printf("L%-3d   %.3f  %.3f  %llu\n",
                        std::get<2>(rows[i]),
                        std::get<0>(rows[i]),
                        std::get<1>(rows[i]),
                        (unsigned long long)std::get<3>(rows[i]));
        }
        if (rows.size() > n_show) {
            std::printf("...\n");
            const size_t tail_start = rows.size() > n_show ? rows.size() - n_show : 0;
            std::printf("Layer  L1     L1+L2  accesses   (highest L1+L2)\n");
            for (size_t i = tail_start; i < rows.size(); i++) {
                std::printf("L%-3d   %.3f  %.3f  %llu\n",
                            std::get<2>(rows[i]),
                            std::get<0>(rows[i]),
                            std::get<1>(rows[i]),
                            (unsigned long long)std::get<3>(rows[i]));
            }
        }
    }

    // ---- Aggregate per-prompt latency report (TTFT + steady-state) ----
    if (!latencies.empty()) {
        auto pct = [](std::vector<double> v, double p) -> double {
            if (v.empty()) return 0.0;
            std::sort(v.begin(), v.end());
            const double idx = p * (v.size() - 1);
            const size_t lo = (size_t)idx;
            const size_t hi = std::min(lo + 1, v.size() - 1);
            const double frac = idx - (double)lo;
            return v[lo] * (1.0 - frac) + v[hi] * frac;
        };

        std::vector<double> ttft_ms, prefill_ms, ss_tok_per_s;
        std::vector<double> all_per_token_ms;
        int total_gen = 0, total_prompt = 0;
        for (const auto & lat : latencies) {
            ttft_ms.push_back(lat.ttft_ms);
            prefill_ms.push_back(lat.prefill_decode_ms);
            if (lat.steady_state_tok_per_s > 0) ss_tok_per_s.push_back(lat.steady_state_tok_per_s);
            for (double t : lat.per_token_ms) all_per_token_ms.push_back(t);
            total_gen    += lat.n_generated_tokens;
            total_prompt += lat.n_prompt_tokens;
        }

        std::printf("\n=== Latency aggregate (%zu prompts) ===\n", latencies.size());
        std::printf("TTFT (s):           p50=%.2f  p90=%.2f  p99=%.2f  min=%.2f  max=%.2f\n",
                    pct(ttft_ms, 0.50)/1000.0, pct(ttft_ms, 0.90)/1000.0,
                    pct(ttft_ms, 0.99)/1000.0,
                    *std::min_element(ttft_ms.begin(), ttft_ms.end())/1000.0,
                    *std::max_element(ttft_ms.begin(), ttft_ms.end())/1000.0);
        std::printf("Prefill decode (s): p50=%.2f  p90=%.2f  p99=%.2f\n",
                    pct(prefill_ms, 0.50)/1000.0, pct(prefill_ms, 0.90)/1000.0,
                    pct(prefill_ms, 0.99)/1000.0);
        std::printf("Steady-state tok/s: p50=%.2f  p90=%.2f  p10=%.2f  best=%.2f  worst=%.2f\n",
                    pct(ss_tok_per_s, 0.50), pct(ss_tok_per_s, 0.90), pct(ss_tok_per_s, 0.10),
                    ss_tok_per_s.empty() ? 0.0 : *std::max_element(ss_tok_per_s.begin(), ss_tok_per_s.end()),
                    ss_tok_per_s.empty() ? 0.0 : *std::min_element(ss_tok_per_s.begin(), ss_tok_per_s.end()));
        if (!all_per_token_ms.empty()) {
            std::printf("Per-token decode (ms): p50=%.1f  p90=%.1f  p99=%.1f  (n=%zu samples)\n",
                        pct(all_per_token_ms, 0.50), pct(all_per_token_ms, 0.90),
                        pct(all_per_token_ms, 0.99), all_per_token_ms.size());
        }
        std::printf("Tokens: %d prompt-tokens total / %d generated total\n",
                    total_prompt, total_gen);
    }

    moe_orch::set_mode_b_context(nullptr);
    mode_b_ctx.reset();

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
