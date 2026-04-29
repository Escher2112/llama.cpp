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

#include "orchestrator/orchestrator.h"
#include "orchestrator/predictor.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
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
        "  --orchestrator-predictor PATH    .bin predictor file (enables orchestrator)\n"
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
    std::vector<llama_model_tensor_buft_override> overrides;
    if (args.override_exps_cpu) {
        // The user-side string is "exps.=CPU"; the tensor-buft override API
        // takes a regex pattern + buffer-type pointer. We match by pattern.
        // (See llama-cli's --override-tensor handling.)
        // For v0 simplicity we skip this here — caller can use the same flag
        // pattern via the prebuilt llama-cli for the baseline; this tool's
        // smoke test just confirms the orchestrator wiring works.
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
    if (!args.predictor_path.empty()) {
        moe_orch::OrchestratorConfig cfg;
        cfg.shadow_mode = true;
        orchestrator = std::make_unique<moe_orch::Orchestrator>(
            cfg, args.predictor_path, /*hopfield=*/"",
            n_layers, n_experts, hidden_dim);
        std::printf("Orchestrator: built (predictor=%s)\n", args.predictor_path.c_str());
    } else {
        std::printf("Orchestrator: not enabled (pass --orchestrator-predictor PATH)\n");
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

    std::vector<llama_token> tokens(args.prompt.size() + 16);
    int n_tok = llama_tokenize(vocab, args.prompt.c_str(), (int)args.prompt.size(),
                                tokens.data(), (int)tokens.size(),
                                add_bos, /*parse_special=*/true);
    if (n_tok < 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
    tokens.resize(n_tok);
    std::printf("Tokenized %d input tokens.\n", n_tok);

    // Prefill
    std::printf(">>> ");
    std::fflush(stdout);
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tok))) {
        std::fprintf(stderr, "prefill failed\n"); return 1;
    }

    // Greedy decode N tokens
    for (int i = 0; i < args.n_predict; i++) {
        // Greedy: take argmax of logits.
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
            std::fprintf(stderr, "decode step %d failed\n", i); break;
        }
    }
    std::printf("\n<<<\n\n");

    llama_perf_context_print(ctx);

    if (orchestrator) {
        auto s = orchestrator->cache_stats();
        std::printf("\n=== Orchestrator cache stats ===\n");
        std::printf("L1 hits:           %llu\n", (unsigned long long)s.hits_L1);
        std::printf("L2 hits:           %llu\n", (unsigned long long)s.hits_L2);
        std::printf("L3 misses:         %llu\n", (unsigned long long)s.misses_L3);
        std::printf("L1 hit rate:       %.3f\n", s.l1_hit_rate);
        std::printf("Promotions to L1:  %llu\n", (unsigned long long)s.promotions_to_L1);
        std::printf("Evictions L1->L2:  %llu\n", (unsigned long long)s.evictions_L1_to_L2);
        std::printf("Evictions L2->L3:  %llu\n", (unsigned long long)s.evictions_L2_to_L3);
        std::printf("L1 predictor calls: %llu\n", (unsigned long long)orchestrator->l1_predictor_calls());
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
