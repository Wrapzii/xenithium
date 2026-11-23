# Executables. SYCL unless HOST (cmake/Xenithium.cmake: xenithium_tool).

# The OpenAI-compatible server (docs/server-env.md): options, then the arch's serving backend.
xenithium_tool(xenithium-server SOURCES server/main.cpp server/options.cpp LIBS xenithium_gpu_engine)

# Checkpoint catalog, arch validation and the byte budget.
xenithium_tool(xenithium-inspect HOST SOURCES inspect/main.cpp LIBS xenithium_model)

# A chat-format golden from the engine's Jinja interpreter over a checkpoint's chat template (or --check one).
xenithium_tool(xenithium-gen-chat-golden HOST SOURCES gen_chat_golden/main.cpp LIBS xenithium_model xenithium_text)

# The interactive launcher (scripts/launch.sh, docs/launcher.md): server.env settings, model pick, launch and live
# status. Host only, no curses: its own terminal layer. A library so tests/test_launch.cpp links the same code.
add_library(xenithium_launch STATIC launch/term.cpp launch/settings.cpp launch/proc.cpp launch/gpu.cpp launch/chat.cpp launch/bench.cpp launch/job.cpp launch/app.cpp)
xenithium_host(xenithium_launch)
target_include_directories(xenithium_launch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/launch)
target_link_libraries(xenithium_launch PUBLIC xenithium_model xenithium_util Threads::Threads)
xenithium_tool(xenithium-launch HOST SOURCES launch/main.cpp LIBS xenithium_launch)

# The benchmark (tools/bench/main.cpp, docs/bench.md): scenarios against any OpenAI-compatible endpoint. Host only, no
# engine code: JSON and TLS from xenithium_util. The corpus (bench/corpus/<set>/<kind>/<file>, sets easy and hard) is
# embedded through a generator built first, so the tool is one binary. A library so tests/test_bench.cpp and the launcher's Bench tab share the code.
xenithium_tool(xenithium-bench-embed HOST SOURCES bench/embed/main.cpp)
file(GLOB_RECURSE XENITHIUM_BENCH_CORPUS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bench/corpus/*/*/*)
set(XENITHIUM_BENCH_CORPUS_CPP ${CMAKE_CURRENT_BINARY_DIR}/bench_corpus_data.cpp)
add_custom_command(OUTPUT ${XENITHIUM_BENCH_CORPUS_CPP}
  COMMAND xenithium-bench-embed ${CMAKE_CURRENT_SOURCE_DIR}/bench/corpus ${XENITHIUM_BENCH_CORPUS_CPP}
  DEPENDS xenithium-bench-embed ${XENITHIUM_BENCH_CORPUS}
  COMMENT "Embedding the benchmark corpus")
add_library(xenithium_bench STATIC
  bench/stats.cpp bench/format.cpp bench/http_stream.cpp bench/client.cpp bench/corpus.cpp bench/prompts.cpp bench/params.cpp
  bench/counters.cpp bench/power.cpp bench/image.cpp bench/quality.cpp bench/session.cpp bench/report.cpp bench/svg_charts.cpp
  bench/html_assets.cpp bench/html_report.cpp bench/runner.cpp bench/registry.cpp
  bench/scenarios/common.cpp bench/scenarios/decode.cpp bench/scenarios/concurrency.cpp bench/scenarios/prefill.cpp
  bench/scenarios/deep_decode.cpp bench/scenarios/turns.cpp bench/scenarios/burst.cpp bench/scenarios/late_arrival.cpp
  bench/scenarios/mixed.cpp bench/scenarios/needle.cpp bench/scenarios/repeat.cpp bench/scenarios/structured.cpp
  bench/scenarios/arrivals.cpp bench/scenarios/vision.cpp bench/scenarios/quality.cpp bench/scenarios/long_agents.cpp
  bench/scenarios/prefill_decode.cpp bench/scenarios/chat_sessions.cpp bench/scenarios/rag.cpp bench/scenarios/autocomplete.cpp
  bench/scenarios/tool_calls.cpp bench/scenarios/long_reply.cpp
  ${XENITHIUM_BENCH_CORPUS_CPP})
xenithium_host(xenithium_bench)
target_include_directories(xenithium_bench PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/bench)
target_link_libraries(xenithium_bench PUBLIC xenithium_util Threads::Threads)
xenithium_tool(xenithium-bench HOST SOURCES bench/main.cpp LIBS xenithium_bench)

# Downstream gate: GSM8K / HumanEval / needle against a running server.
xenithium_tool(xenithium-evalrun HOST SOURCES evalrun/main.cpp LIBS xenithium_quant xenithium_util Threads::Threads)

# The host side of the KL gate: the reference dumps, dump vs dump comparison.
xenithium_tool(xenithium-kl HOST SOURCES kl/main.cpp LIBS xenithium_eval)

xenithium_tool(xenithium-quant HOST SOURCES quant/main.cpp LIBS xenithium_model)

# Offline bake (docs: references/offline-bake.md in the engine skill).
xenithium_tool(xenithium-bake
  SOURCES bake/main.cpp bake/synth.cpp bake/experts.cpp bake/ple.cpp bake/calib_run.cpp bake/assemble.cpp bake/mtp.cpp bake/head.cpp bake/prompts.cpp bake/dense.cpp
  LIBS xenithium_quant_gpu xenithium_quant xenithium_arch xenithium_model xenithium_eval)

# Host reference of the MTP drafter, scored on the target's hiddens (probe --hidden-out).
xenithium_tool(xenithium-mtp-ref SOURCES mtp_ref/main.cpp LIBS xenithium_kernels xenithium_model)
target_compile_options(xenithium-mtp-ref PRIVATE -O3)

# Offline scoring of prefill routing predictors on a probe --pf-route-trace capture.
xenithium_tool(xenithium-lookahead HOST SOURCES lookahead/main.cpp LIBS xenithium_model)
target_compile_options(xenithium-lookahead PRIVATE -O3)

# Benchmarks and gates on the cards: one subcommand per op / model (tools/probe/main.cpp lists them).
xenithium_tool(xenithium-probe
  SOURCES probe/main.cpp probe/flags.cpp probe/moe.cpp probe/gdn.cpp probe/qsa.cpp probe/ple.cpp probe/model.cpp probe/mtp.cpp probe/seqs.cpp probe/serve.cpp
          probe/prefix.cpp probe/qwen35.cpp probe/qwen35_gdn.cpp probe/qwen35_attn.cpp probe/qwen35_model.cpp probe/qwen35_dflash.cpp
          probe/qwen35_prefix.cpp probe/qwen35_select.cpp probe/qwen35_headx.cpp probe/llama.cpp probe/flint.cpp probe/rank.cpp probe/dflash38.cpp
          xrv/qsa_capture.cpp)
target_include_directories(xenithium-probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(xenithium-probe PRIVATE xenithium_kernels xenithium_arch xenithium_eval xenithium_gpu_engine)

# Kernel Replay Validation: direct replay of the production QSA top-512 selector from a validated XRV fixture.
xenithium_tool(xenithium-xrv-qsa-select SOURCES xrv/qsa_select_xpu.cpp LIBS xenithium_kernels)
xenithium_tool(xenithium-xrv-qsa-sparse-attention SOURCES xrv/qsa_sparse_attention_xpu.cpp LIBS xenithium_kernels)
