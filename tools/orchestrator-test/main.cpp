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

#include "ggml-backend.h"

#include "orchestrator.h"
#include "predictor.h"

#include <algorithm>
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
    std::string dump_trace_path;          // --dump-trace PATH
    std::string prompts_file;             // --prompts-file PATH (1 prompt per line)
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
        "  --orchestrator-recency-only      enable orchestrator without predictor (Mode A)\n"
        "  --l1 N                           L1 (VRAM) cache capacity per layer (default 32)\n"
        "  --l2 N                           L2 (RAM)  cache capacity per layer (default 80)\n"
        "  --prefetch-top-k N               experts the MLP prefetches per call (default 16)\n"
        "  --dump-trace PATH                write per-(layer, token) trace to PATH (binary)\n"
        "  --prompts-file PATH              run a list of prompts (one per line) sequentially\n"
        "  --no-override-exps               do NOT pin experts to CPU (default: do pin)\n",
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
        else if (arg == "--orchestrator-recency-only") a.orchestrator_recency_only = true;
        else if (arg == "--l1" && need("--l1")) a.l1_capacity = std::atoi(argv[++i]);
        else if (arg == "--l2" && need("--l2")) a.l2_capacity = std::atoi(argv[++i]);
        else if (arg == "--prefetch-top-k" && need("--prefetch-top-k"))
                                                a.prefetch_top_k = std::atoi(argv[++i]);
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
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.n_gpu_layers;
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
        cfg.dump_trace_path = args.dump_trace_path;
        orchestrator = std::make_unique<moe_orch::Orchestrator>(
            cfg, args.predictor_path, /*hopfield=*/"",
            n_layers, n_experts, hidden_dim);
        if (!args.dump_trace_path.empty() && args.predictor_path.empty() && !args.orchestrator_recency_only) {
            mode_label = "TRACE-CAPTURE (no cache stats meaningful)";
        } else {
            mode_label = args.predictor_path.empty()
                ? "MODE A (recency-only)"
                : "MODE B (recency + MLP predictor)";
        }
        std::printf("Orchestrator: %s  l1=%d  l2=%d\n",
                    mode_label, args.l1_capacity, args.l2_capacity);
        if (!args.predictor_path.empty()) {
            std::printf("  predictor: %s\n", args.predictor_path.c_str());
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

    auto run_one_prompt = [&](const std::string & prompt) -> bool {
        std::vector<llama_token> tokens(prompt.size() + 16);
        int n_tok = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                    tokens.data(), (int)tokens.size(),
                                    add_bos, /*parse_special=*/true);
        if (n_tok < 0) { std::fprintf(stderr, "tokenize failed\n"); return false; }
        tokens.resize(n_tok);

        std::printf(">>> [%d toks in] ", n_tok);
        std::fflush(stdout);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tok))) {
            std::fprintf(stderr, "prefill failed\n"); return false;
        }

        for (int i = 0; i < args.n_predict; i++) {
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

            if (llama_decode(ctx, llama_batch_get_one(&best, 1))) {
                std::fprintf(stderr, "decode step %d failed\n", i); return false;
            }
        }
        std::printf("\n<<<\n");
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

    for (size_t pi = 0; pi < prompts.size(); pi++) {
        if (prompts.size() > 1) {
            std::printf("\n--- prompt %d/%d ---\n", (int)(pi + 1), (int)prompts.size());
            // Clear KV cache so prompts are independent (important for trace
            // capture so the predictor doesn't pick up cross-prompt artifacts).
            llama_memory_clear(llama_get_memory(ctx), true);
            // Drop the orchestrator's audit event log between prompts. Stats
            // live in scalar counters now (see ThreeTierCache::_log) so they
            // accumulate correctly across the run; this just prevents events_
            // from growing toward its 4M cap. At L1=8/L2=32 Mode B that cap
            // was reachable mid-run and the surrounding heap pressure tripped
            // the prompt-14 abort. Cumulative summary still reports correctly
            // at the end of the run.
            if (orchestrator) orchestrator->clear_cache_events();
        }
        if (!run_one_prompt(prompts[pi])) break;
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

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
