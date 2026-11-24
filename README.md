# Xenithium

**A native C++20 LLM inference engine and OpenAI-compatible server for Intel Arc Pro B-series GPUs (Battlemage / Xe2).**

- **Native all the way down.** Kernels are SYCL + ESIMD, compiled ahead of time for `bmg`. Memory and queues go
  through Level Zero directly, and the tensor-parallel exchange between cards is the engine's own (no oneCCL).
- **No Python at runtime.** Tokenizer, chat-template renderer (an in-engine Jinja interpreter), reasoning and
  tool-call parsers and sampling are all C++.
- **Built for speed.** Speculative decoding with block drafters (DFlash 2, DSpine b16) and MTP, a prefix cache, step
  graphs, and 1-, 2- or 4-card tensor parallelism.
- **Drop-in API.** `/v1/chat/completions`, `/v1/completions` and `/v1/responses`, with streaming, reasoning, tool
  calls and image / video input.

**Contents:** [Performance](#performance) · [Supported models](#supported-models) · [Quick start](#quick-start) ·
[Requirements](#requirements) · [Install](#install-dependencies) · [Build](#build) · [Run on the host](#run-on-the-host) ·
[Run with Docker](#run-with-docker-compose) · [API](#api) · [Documentation](#documentation) ·
[Repository layout](#repository-layout) · [Licence](#licence)

---

## Performance

Measured with the project's own benchmark, `xenithium-bench`, in its **easy mode**: the way the usual public
benchmarks measure (tasks that repeat or reshape their input, replies that end where the model ends them, a code-led
weighted mean). Every row is one standard-suite run against a live server on one machine with 4x Arc Pro B70,
**power-capped to 220 W per card**.

> **Looking for the strict numbers?** The [hard-mode results](docs/performance-hard-mode.md) measure the same
> models on this project's own code and documentation, every reply run to its full length. Their decode numbers are
> lower, and they are the numbers to compare builds with.

| Model | Cards | Slots | Drafter | Decode (tok/s) | Throughput at 1 / 4 / 8 / 16 streams (tok/s) | Prefill 8K / 32K / 128K (tok/s) | Decode at 32K / 109K (tok/s) | TTFT 8K | Quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Qwen3.8-Flash-Next](docs/benchmark/qwen38-flash-next-4-card-easy.html) | 4 | 16 | DFlash 2 | **353** | 319 / 592 / 693 / 757 | 7,736 / 7,922 / 7,744 | 217 / 186 | 1.0 s | 100 % |
| [Qwen3.8-Flash-Next](docs/benchmark/qwen38-flash-next-2-card-easy.html) | 2 | 16 | DFlash 2 | **145** | 120 / 183 / 205 / 214 | 2,676 / 2,717 / 2,577 | 123 / 99 | 2.9 s | 100 % |
| [Qwen3.8-27B AutoRound](docs/benchmark/qwen38-27b-autoround-2-card-easy.html) | 2 | 16 | DSpine b16 | **273** | 224 / 448 / 652 / 783 | 4,557 / 4,119 / 2,793 | 173 / 135 | 1.7 s | 97 % |
| [Qwen3.8-27B AutoRound](docs/benchmark/qwen38-27b-autoround-1-card-easy.html) | 1 | 8 | DSpine b16 | **155** | 132 / 266 / 389 / - | 2,470 / 2,151 / 1,394 | 97 / 71 | 3.1 s | 97 % |
| [Qwen3.8-27B Swift-1.5](docs/benchmark/qwen38-27b-swift15-2-card-easy.html) | 2 | 16 | DSpine b16 | **272** | 227 / 461 / 687 / 769 | 4,542 / 4,081 / 2,775 | 174 / 151 | 1.7 s | 97 % |
| [Qwen3.8-27B Swift-1.5](docs/benchmark/qwen38-27b-swift15-1-card-easy.html) | 1 | 8 | none | **94** | 88 / 250 / 314 / - | 2,518 / 2,216 / 1,406 | 75 / 61 | 3.1 s | 100 % |
| [Recursive Flint INT4](docs/benchmark/flint-int4-2-card-easy.html) | 2 | 8 | DFlash 2 | **502** | 466 / 704 / 829 / - | 13,010 / 11,184 / 6,551 | 361 / 272 | 0.4 s | 100 % |
| [Recursive Flint INT4](docs/benchmark/flint-int4-1-card-easy.html) | 1 | 8 | DFlash 2 | **334** | 312 / 456 / 534 / - | 8,017 / 6,633 / 3,606 | 219 / 168 | 0.7 s | 94 % |

**How to read the table**

| Column | Meaning |
| --- | --- |
| Cards / Slots | The cards the model runs on, and the requests it serves at once (`XEN_SLOTS`). |
| Drafter | The speculative drafter the server ran: DFlash 2 (a block drafter) or DSpine b16 (16-token blocks); `none` = plain decoding. |
| Decode | The headline: single-stream decode tok/s, the easy mix of the categories (code 30 %, reasoning 20 %, prose 15 %, JSON 15 %, edits 10 %, summaries 10 %). |
| Throughput | Tokens per second of all streams together at 1 / 4 / 8 / 16 requests in flight. `-` = more streams than the setup has slots, so not measured. |
| Prefill | Prompt tokens read per second at 8K / 32K / 128K tokens, prefix cache defeated. |
| Decode at depth | Decode tok/s with a 32K / 109K-token document in context. |
| TTFT 8K | Time to the first token of an 8K-token prompt. |
| Quality | Share of 34 checkable questions answered correctly (greedy). |

Each model name links its self-contained HTML report in `docs/benchmark/`;
[docs/benchmark/README.md](docs/benchmark/README.md) names the run and the settings behind every report. To measure
your own cards, start a server and run:

```bash
build/xenithium-bench run --mode easy                                     # the standard suite
build/xenithium-bench run --mode easy --set concurrency.levels=1,2,4,8    # a server with 8 slots
```

The [benchmarking guide](docs/bench-guide.md) explains modes, suites and scenarios; [docs/bench.md](docs/bench.md) is
the full reference.

## Supported models

| Model | Arch | Cards | Checkpoint (`~/models/...`) | Drafter (`~/models/...`) |
| --- | --- | --- | --- | --- |
| Qwen3.8 Flash-Next | `qwen38` | 4 / 2 / 1 | `qwen38-w4a16`, `qwen38-w4a16-intel` (W4A16): [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | `qwen38-flash-next-dflash2`: [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) |
| Qwen3.8-27B (Swift-1.5) | `qwen35` | 2 / 1 | `swift15-w4a16-autoround` (W4A16): [ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AutoRound](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AutoRound/tree/main) | `qwen38-27b-dspine-b16`: [tencent/Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16)<br>`qwen38-27b-dflash2`: [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)<br>`qwen38-27b-dflash2-int4-gptq` (GPTQ int4): [Launch80/Qwen3.8-27B-DFlash2-Int4-baked-drafter](https://huggingface.co/Launch80/Qwen3.8-27B-DFlash2-Int4-baked-drafter) |
| Qwen3.8-27B (W4A16 AutoRound) | `qwen35` | 2 / 1 | `Qwen3.8-27B-W4A16-AutoRound`: [dbirks/Qwen3.8-27B-W4A16-AutoRound](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound) (a copy of the same export; Intel's repository is no longer public) | the same three as the row above |
| Qwen3.8-27B (Swift-1.5 Uncensored, GPTQ int4 baked) | `qwen35` | 2 / 1| `swift15-27b-int4-baked` (GPTQ int4 g128, int4 head and MTP layer): [Launch80/Swift-1.5-Qwen3.8-27B-Uncensored-Int4-baked](https://huggingface.co/Launch80/Swift-1.5-Qwen3.8-27B-Uncensored-Int4-baked) | `qwen38-27b-dflash2-int4-gptq` or `qwen38-27b-dflash2` (DFlash 2 only: DSpine needs a bf16 head) |
| Nanbeige4.3-33B | `llama` | 2 / 1 | `nanbeige43-33b-int4` (int4 g32 zero-point): [Nanbeige/Nanbeige4.3-33B-INT4](https://huggingface.co/Nanbeige/Nanbeige4.3-33B-INT4) | - |
| Recursive Flint INT4 | `flint` | 2 / 1 | `flint-int4` (MLX affine int4, 8-bit embedding): [RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4) | `flint-dflash2` |

What each model brings:

- **Qwen3.8 Flash-Next** (MoE + Gated DeltaNet hybrid): MTP or DFlash 2 drafter, prefix cache, step graphs, KV
  lending, images.
- **Qwen3.8-27B**, all three checkpoints (dense, Gated DeltaNet hybrid): MTP layer + DSpine / DFlash 2 drafters, W4A8
  prefill, prefix cache, page-selected attention, images. `models/presets/qwen38-27b-intel.env` starts the AutoRound
  checkpoint. The GPTQ int4 baked checkpoint (AutoGPTQ config, int4 `lm_head`) runs with its MTP layer or a DFlash 2
  drafter with `XEN_SEG_ROWS` 8; DSpine b16 is refused at load (it needs a bf16 head). DFlash 2 drafters load as bf16
  or as GPTQ int4 (4 bits, group 128, symmetric, no act order); the GPTQ drafter is a 1.4 GiB download instead of
  3.6 GiB at the same speed.
- **Nanbeige4.3-33B** (dense Llama): many-row prefill, prefix cache; no drafter in the checkpoint.
- **Recursive Flint INT4** (MoE, 256 experts top-6, sliding-window + full attention with sinks): many-row int4
  prefill, prefix cache, DFlash 2 drafter, step graphs, chat, reasoning and tool calls.

The arch is picked from the checkpoint's `config.json` (`model_type`). Every model is checked against stored goldens:
tokenizer and chat-render output (byte-equal) and the KL divergence of the next-token distribution on fixed token
sets. Each preset in `models/presets/` starts its model with the settings it was tuned for
([docs/server-env.md](docs/server-env.md)).

> **Tip: the fastest Qwen3.8-27B setup runs both drafters.** DSpine b16 (or DFlash 2 with `XEN_SEG_ROWS` 8 if you have
> no 16-token drafter) guesses through 8 streams, and the checkpoint's own MTP layer takes over at 9 and up
> (`XEN_DRAFTER_MTP_FROM` 9, the default). The shipped presets do this; a manual start gets it by setting
> `XEN_DRAFTER_MODEL` and leaving those two settings at their defaults.

> **Model licences.** Every checkpoint and drafter above is a third-party download with a licence of its own, shown on
> the page you download it from (and, for a quantised export, on the licence of the base model behind it). None of
> them is distributed with this repository, and Xenithium's own licence covers none of them: read the terms on each
> page before you download, and keep the licence file that comes with the weights (see [Licence](#licence)).

## Quick start

On Ubuntu 26.04 with the `xe` kernel driver, from the repository root:

```bash
scripts/install-deps.sh          # compiler, oneDNN, Level Zero, GPU driver (apt, via sudo)
scripts/build-host.sh            # build everything (Release, icpx, AOT for bmg)
scripts/launch.sh                # the launcher: pick a model, launch it, benchmark it
```

Or with Docker, no oneAPI on the host: `docker compose build && docker compose run --rm launch`. Then send a request:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "flint", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Requirements

**Hardware**

- 1, 2 or 4 Intel Arc Pro B-series cards (developed on Arc Pro B70). A two-card setup needs both cards visible; four
  cards serve Qwen3.8 Flash-Next.
- VRAM: weights per card at TP2 are about 6.5 GiB (Qwen3.8-27B), 8.8 GiB (Nanbeige) and 10.5 GiB (Flint); the rest
  goes to the KV pool. Flint fits on one card at its full 262K context with int8 KV.

**Software** (the versions the engine is built and tested with)

| Component | Version |
| --- | --- |
| OS | Linux with the `xe` kernel driver (Ubuntu 26.04, kernel 7.0) |
| GPU compute runtime | `libze-intel-gpu1` / `intel-opencl-icd` 26.35, `libigc2` 2.41; Level Zero loader `libze1` + `libze-dev` 1.32 |
| Compiler | Intel oneAPI 2026.2: the DPC++/C++ compiler (`icpx`, under `/opt/intel/oneapi`) and its SYCL runtime. Other compilers are refused. |
| oneDNN | oneAPI's package (3.13.2): the W4A8 prefill GEMMs and the opt-in oneDNN prefill attention |
| Build tools | CMake >= 3.28, a C++20 standard library, POSIX threads. No other third-party libraries. |

`scripts/oneapi-env.sh` sets up oneAPI and its oneDNN. Check the cards with `xpu-smi discovery` (or `sycl-ls`): the
B70s must show up as Level Zero GPUs.

## Install dependencies

Only needed to build and run on the host; the Docker image brings its own (see
[Run with Docker Compose](#run-with-docker-compose)). On Ubuntu 26.04 with the `xe` kernel driver loaded (the stock
26.04 kernel has it):

```bash
scripts/install-deps.sh          # apt (via sudo): oneAPI 2026.2 DPC++ compiler, oneDNN, Level Zero, the GPU driver
                                 # (Level Zero + OpenCL), cmake, ninja, ffmpeg
sudo usermod -aG render,video "$USER"   # access to /dev/dri/renderD*; log out and in again
```

<details>
<summary>What <code>install-deps.sh</code> sets up</summary>

`install-deps.sh` adds two apt sources when they are not configured yet, with their keys from `docker/keys/` (checked
by sha256): Intel's oneAPI repository (`apt.repos.intel.com/oneapi`) and the `kobuk-team/intel-graphics` PPA (the Level
Zero loader `libze1` / `libze-dev`, the user-mode driver `libze-intel-gpu1` and `intel-opencl-icd`, `intel-ocloc`;
oneDNN builds its GPU kernels through the OpenCL device, so without `intel-opencl-icd` every oneDNN call fails with
"OpenCL device not found"). The compiler is pinned to `2026.2.0-863` and oneDNN to `2026.1.0-617`
(`ONEAPI_VERSION=...` / `DNNL_VERSION=...` for others); the PPA keeps only its newest driver build, so the driver is
not pinned. `scripts/install-deps.sh --runtime` installs only what running needs (SYCL runtime, oneDNN, Level Zero,
driver, ffmpeg).

`scripts/oneapi-env.sh` (sourced by the build and test scripts) sets up oneAPI from `/opt/intel/oneapi/2026.2` and its
oneDNN; `XEN_ONEAPI=DIR` / `XEN_DNNL_ROOT=DIR` point elsewhere (then delete `build/CMakeCache.txt` and rebuild). To
have `icpx` and the runtime libraries in every shell, add `source ~/xenithium-beta/scripts/oneapi-env.sh` to
`~/.bashrc`.

</details>

## Build

```bash
scripts/build-host.sh                       # configure (Release, icpx, AOT for bmg) + build everything
scripts/build-host.sh xenithium-server      # or one target (also: xenithium-launch, xenithium-probe, ...)
```

`build-host.sh` sources `scripts/oneapi-env.sh` when `icpx` is not on `PATH`. The manual equivalent:

```bash
source scripts/oneapi-env.sh
cmake -S . -B build -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"
```

**Tests:** `scripts/test.sh` runs ctest: the `host` suites, then the `gpu` gates when the cards are idle
(`BUILD_DIR=DIR` for another build tree; extra arguments go to ctest, e.g. `-R gqa`).

## Run on the host

Build first (above). Three ways, from the simplest:

**1. Launcher (TUI).** Pick a model, edit the settings, start and watch the server, run benchmarks. See
[docs/launcher.md](docs/launcher.md).

```bash
scripts/build-host.sh xenithium-launch
scripts/build-host.sh xenithium-server
scripts/launch.sh                          # --models DIR to scan another folder for one run
```

**2. Start script with a settings file.** `scripts/start-server.sh` reads `KEY=value` lines from `./server.env` (or
`SERVE_ENV=FILE`), fills in the defaults and starts `build/xenithium-server`. Example `server.env` for Flint:

```bash
XEN_MODEL=$HOME/models/flint-int4
XEN_SERVED_NAME=flint
XEN_SLOTS=4            # concurrent sequences
XEN_CONTEXT=32768      # max context: one request's most tokens (up to 1048576; YaRN past the trained length, Flint needs none)
XEN_TP=2               # 2 = both cards, 1 = one card (XEN_CARD picks which)
XEN_KV=int8
XEN_MODELS_DIR=$HOME/models   # launcher only: the folder its m list scans (default: the project's models/)
```

```bash
scripts/start-server.sh                    # SERVE_DRY_RUN=1 prints the command instead
```

**3. Directly.**

```bash
./build/xenithium-server --model ~/models/flint-int4 --host 0.0.0.0 --port 8000 --slots 4 --context 32768 --tp 2
```

The server binds `0.0.0.0:8000` by default; loading takes ~10-15 s plus a few seconds of warm-up, and it prints
`serving on http://...` when ready. Every setting (environment variable, flag, default, which arch reads it, the
measurements behind the default) is in [docs/server-env.md](docs/server-env.md).

**Models folder.** The launcher's `m` list scans a models folder for the checkpoints you can serve: by default the
project's `models/` folder, which also holds the per-model settings files (the shipped presets in `models/presets/`,
your own beside them, git-ignored). To serve checkpoints from your own folder, set `XEN_MODELS_DIR=<folder>` in
`server.env` (e.g. `XEN_MODELS_DIR=$HOME/models`): the launcher scans it for the model list and the DFlash drafter
lists; the server itself never reads it. `--models DIR` to `scripts/launch.sh` overrides the folder for one run.

## Run with Docker Compose

The image (`docker/Dockerfile`) builds the engine with the same toolchain as the host (oneAPI 2026.2 with its oneDNN,
Level Zero and the GPU driver from the same apt sources, via `scripts/install-deps.sh`), runs the host test gates during
the build, and ships a runtime image without the compiler. The host needs only Docker with Compose v2, the `xe` kernel
driver and `/dev/dri`; no oneAPI.

```bash
mkdir -p .cache build/launch         # folders the containers write to (else Docker creates them owned by root)
docker compose build                 # image xenithium:beta; the first build ~6 min on 32 cores, later ones ~3 min
docker compose up -d server          # start the server with ./server.env (+ the model's settings file), as scripts/start-server.sh
docker compose run --rm launch       # or: the launcher, as on the host (pick a model, launch, stop, relaunch)
docker compose logs -f server        # load progress; "serving on http://..." when ready (healthy once /health answers)
docker compose down                  # stop it and release the cards
```

`scripts/docker.sh` takes the same arguments (`scripts/docker.sh up -d server`, `scripts/docker.sh run --rm launch`)
and fills in what plain `docker compose` takes from defaults: your user and group ids, the groups of `/dev/dri` and of
the Docker socket, and mounts for checkpoint folders outside your home (below). Use it when your ids differ from the
defaults (uid / gid 1000, render 991, video 44, docker 983) or your checkpoints live outside your home.

<details>
<summary>How the containers see the host</summary>

The containers see the host the way the native tools do, so the native and the Docker paths are interchangeable:

- **Settings are the host's files**: `server.env` (or `XEN_SETTINGS=/work/<file>`: `/work` is this folder),
  `models/` (your per-model settings and the presets), `.cache/` (hot-expert rankings) and `build/launch/` (the
  launcher's state). The server container's command is `scripts/start-server.sh`.
- **Folders**: your home folder is mounted at the same path and `HOME` is yours, so every `~/...` path works:
  checkpoints, the launcher's models folder (`XEN_MODELS_DIR`), drafters. `scripts/docker.sh` also mounts the folders
  of `XEN_MODELS_DIR`, `XEN_MODEL` and `XEN_DRAFTER_MODEL` that lie outside your home (read from the settings file and
  the model's settings file; written to `.cache/compose.mounts.yaml`). A models folder outside your home picked
  later in the launcher needs a new `scripts/docker.sh run --rm launch`.
- **Processes** (`pid: host`): the launcher in Docker sees, stops and relaunches any xenithium server, native or in a
  container, and the host's launcher sees the container's; `scripts/start-server.sh` refuses while one runs, as natively.
- **Network**: the host's, so `XEN_HOST` / `XEN_PORT` from `server.env` apply directly (default `0.0.0.0:8000`). Set
  `XEN_PORT` in the shell too when it is not 8000 (the server container's health check reads it).
- **Cards**: `/dev/dri` is passed in; the containers run as your user with the render and video groups. The host's
  Docker socket is mounted (with its group) so the "containers hold the cards" checks see the host's containers; this
  image's own containers are not counted (their servers are seen as processes). `ZE_AFFINITY_MASK` is passed through.
- **Other tools**: `docker compose run --rm tools xenithium-inspect ~/models/flint-int4` (every `xenithium-*` binary
  of the build is on `PATH`; no argument = a shell). The GPU gates: run `scripts/test.sh` on a host build. The
  launcher's Bench tab runs the image's own `xenithium-bench` ([docs/bench.md](docs/bench.md)).
- Plain `docker compose` takes `XEN_UID`, `XEN_GID`, `XEN_RENDER_GID`, `XEN_VIDEO_GID`, `XEN_DOCKER_GID` from the shell
  or a `./.env` file (else the defaults above) and mounts nothing outside your home.

</details>

## API

OpenAI-compatible endpoints:

| Endpoint | What it does |
| --- | --- |
| `/v1/chat/completions` | Chat: streaming, reasoning in `reasoning`, tool calls |
| `/v1/completions` | Plain completions |
| `/v1/responses` | Stateless Responses: `input` text or items, `instructions`, function tools, streaming events (no `previous_response_id` / stored responses) |
| `/v1/models` | The served model |
| `/health`, `/version`, `/metrics` | Health check, build version, live metrics |

With `XEN_VISION=1` (qwen38, the 27B) chat and Responses messages take images (`image_url` / `input_image`: base64
`data:` URLs or `http(s)` URLs the server downloads) and videos (`video_url`, decoded by an ffmpeg subprocess:
`XEN_VISION_FFMPEG`).

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "flint",
  "messages": [{"role": "user", "content": "What is the capital of Australia?"}],
  "max_tokens": 400
}'
```

Chat templates: by default the checkpoint's own `chat_template.jinja` is rendered by the in-engine Jinja interpreter
(`--chat-template auto|builtin|jinja|FILE`, `XEN_CHAT_TEMPLATE`).

## Documentation

| Document | What is in it |
| --- | --- |
| [docs/server-env.md](docs/server-env.md) | Every server setting: environment variable, flag, default, the measurements behind it |
| [docs/launcher.md](docs/launcher.md) | The launcher TUI: screens, keys, files |
| [docs/bench-guide.md](docs/bench-guide.md) | How to benchmark a server with `xenithium-bench` |
| [docs/bench.md](docs/bench.md) | The benchmark reference: options, scenarios, how each number is computed |
| [docs/performance-hard-mode.md](docs/performance-hard-mode.md) | The hard-mode performance table |
| [docs/benchmark/README.md](docs/benchmark/README.md) | The HTML reports behind both performance tables, with the run and settings of each |

## Repository layout

| Path | Contents |
| --- | --- |
| `src/arch/<arch>/`, `include/xenithium/arch/<arch>/` | everything model-specific: the arch module (`arch.cpp`), the runtime (`model.cpp`), its `core::StepModel` adapter and serving backend (`serve.cpp`); `qwen35`'s runtime also serves `llama` and `flint` |
| `src/kernels/<family>/` | SYCL / ESIMD kernels by family: `gemm`, `attention`, `linear_attention`, `moe`, `residual`, `embed`, `head`, `drafter`, `comm` (+ the `tile` headers) |
| `src/model/`, `src/load/`, `src/rt/` | checkpoint reader, config, placement, the arch registry; the dense-weight loader; Level Zero runtime and the two-card fabric |
| `src/core/`, `src/server/`, `src/serve/` | batching engine and sampler, HTTP server, the serving-backend registry |
| `src/text/` | tokenizer, Unicode tables, Jinja interpreter, chat formats and parsers (`src/text/chat/`) |
| `tools/server/` | `xenithium-server`: options, then the arch's serving backend |
| `tools/launch/` | the launcher TUI |
| `tools/probe/`, `tools/kl/`, `tools/lookahead/`, `tools/bake/`, `tools/quant/` | kernel probes, log-prob dumps and KL compare, bake and quantisation tools |
| `tools/bench/` | `xenithium-bench`: the benchmark suites against any OpenAI-compatible endpoint (`docs/bench.md`) |
| `tests/`, `tests/data/` | gates (ctest labels `host` / `gpu`) and goldens |
| `cmake/Xenithium.cmake` | the target helpers every `CMakeLists.txt` uses |
| `docs/` | settings reference, launcher guide, benchmarking guide and reference, the hard-mode performance table |
| `docs/benchmark/` | the HTML reports behind the performance tables, one run per model and card count |
| `docker/Dockerfile`, `compose.yaml`, `scripts/docker.sh` | the image (build + runtime stages), the Compose services `server`, `launch`, `tools`, and the compose wrapper |
| `scripts/install-deps.sh` | system packages (apt: compiler, oneDNN, Level Zero, GPU driver), for the host and the image |

## Licence

Xenithium is licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)); all rights not expressly granted
by that licence are reserved by Niluthon, who develops and owns the engine (see [NOTICE](NOTICE)).

That covers what this repository holds: the engine's code, its documentation, its tests and its benchmark text. The
models it runs are not part of it. Every checkpoint and drafter in the Supported models table is downloaded from its
own source under the terms that source sets — including the licence of the base model a quantised export was made
from — and this project neither grants nor changes those terms. Check the licence on each download page and meet it
before using the weights or passing their output on; [NOTICE](NOTICE) names the third-party parts of the engine itself.

## Kernel replay validation

For QSA selector fixture replay, exact top-k checks, and replay-first kernel workflows, see [docs/XRV.md](docs/XRV.md). Current adapter coverage and pending work are listed there explicitly.
