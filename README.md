# Xenithium

A native C++20 LLM inference engine and OpenAI-compatible server for Intel Arc Pro B-series GPUs
(Battlemage / Xe2). Kernels are SYCL + ESIMD compiled ahead of time for `bmg`; memory and queues go through Level Zero
directly; the two-card tensor-parallel exchange is the engine's own (no oneCCL). Tokenizer, chat-template renderer
(an in-engine Jinja interpreter), reasoning / tool-call parsers and sampling are all native C++: no Python at runtime.

## Supported models

| Model | Checkpoint (`~/models/...`) | Download | Drafter download (`~/models/...`) | Arch | Type | Cards | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8 Flash-Next | `qwen38-w4a16`, `qwen38-w4a16-intel` (W4A16) | [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | `qwen38-flash-next-dflash2`: [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) | `qwen38` | MoE + Gated DeltaNet hybrid | 4 / 2 / 1 | MTP or DFlash 2 drafter, prefix cache, step graphs, KV lending, images |
| Qwen3.8-27B (Swift-1.5) | `swift15-w4a16-autoround` (W4A16) | [ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AutoRound](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AutoRound/tree/main) | `qwen38-27b-dspine-b16`: [tencent/Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16)<br>`qwen38-27b-dflash2`: [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) | `qwen35` | dense, Gated DeltaNet hybrid | 2 / 1 | MTP layer + DSpine / DFlash 2 drafters, W4A8 prefill, prefix cache, page-selected attention, images |
| Qwen3.8-27B (W4A16 AutoRound) | `Qwen3.8-27B-W4A16-AutoRound` | [dbirks/Qwen3.8-27B-W4A16-AutoRound](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound) (a copy of the same export; Intel's repository is no longer public) | the same two as the row above | `qwen35` | dense, Gated DeltaNet hybrid | 2 / 1 | as above; `models/presets/qwen38-27b-intel.env` starts it |
| Nanbeige4.3-33B | `nanbeige43-33b-int4` (int4 g32 zero-point) | [Nanbeige/Nanbeige4.3-33B-INT4](https://huggingface.co/Nanbeige/Nanbeige4.3-33B-INT4) | - | `llama` | dense Llama | 2 / 1 | no drafter in the checkpoint; many-row prefill, prefix cache |
| Recursive Flint INT4 | `flint-int4` (MLX affine int4, 8-bit embedding) | [RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4) | `flint-dflash2` | `flint` | MoE 256 experts top-6, sliding-window + full attention with sinks | 2 / 1 | many-row int4 prefill, prefix cache, DFlash 2 drafter, step graphs, chat, reasoning and tool calls |

**Every checkpoint and drafter in the two download columns is a third-party download with a licence of its own**, shown
on the page you download it from (and, for a quantised export, on the licence of the base model behind it). None of them is
distributed with this repository, and Xenithium's own licence covers none of them — read the terms on each page before
you download, and keep the licence file that comes with the weights (see [Licence](#licence)).

The arch is picked from the checkpoint's `config.json` (`model_type`). Every model is checked against stored goldens:
tokenizer and chat-render output (byte-equal) and the KL divergence of the next-token distribution on fixed token sets.
Each preset in `models/presets/` starts its model with the settings it was tuned for (docs/server-env.md).

For the fastest Qwen3.8-27B setup run both drafters: DSpine b16 — DFlash 2 with `XEN_SEG_ROWS` 8 if you have no
16-token drafter — guesses through 8 streams, and the checkpoint's own MTP layer takes over at 9 and up
(`XEN_DRAFTER_MTP_FROM` 9, the default). The shipped presets do this; a manual start gets it by setting
`XEN_DRAFTER_MODEL` and leaving those two settings at their defaults.

## Performance(Xenithium-Bench)

`xenithium-bench` (`docs/bench.md`) against a live server started from each preset, on one machine: 2x Arc Pro B70 for
the two-card and one-card rows, 4x B70 for the four-card one. **The GPUs are power-capped to 220 W per card** (every
report's power section records the cap). "Decode" is the agent mix of the sampled scenarios
(harmonic mean over chat, code, reasoning, maths, prose, JSON, edit and summary), `xenithium-bench`'s headline.
"All streams" is the closed-loop concurrency sweep, tokens per second over every stream together. Prefill is the
prompt-read rate of real code text at the depth named, with the prefix cache defeated; TTFT is the 8K prompt's.
Decode at depth is the same document sent once, then decoded with that context. A cell shows `-` where the endpoint's
context cannot hold that depth. Every row links its `xenithium-bench` report (self-contained HTML in
`docs/benchmark/`): the name is the quick run of decode, concurrency, prefill to 32K and decode at 32K; the
long-context link is the same preset's run of the 64K / 128K depths. "Drafter" is the speculative drafter the row's
preset ran: DFlash 2 (a block drafter, up to 7 guesses per step) or DSpine b16 (16-token blocks, up to 15 guesses
per step after a good one); every row ran with its drafter on, and the reports show the accepted tokens per step.

> **These rows are a strict measurement of the machine, not the easiest one.** The measured text is this project's own:
> **real code** and **real documentation**, repeated nothing. That matters most for Decode, where a drafter earns its speed by
> guessing right, and real text is guessed far less well than synthetic, repetitive or empty-prompt benchmarks — the
> accepted tokens per step of the easy ones would let any engine post a higher decode number on the same hardware.
> Compare rows against each other, and against another setup only corpus for corpus.

| Server (preset) | Drafter | Decode, 1 stream (tok/s) | All streams 1 / 4 / 8 (tok/s) | Prefill 2K, 8K, 32K (tok/s) | Prefill 64K / 128K (tok/s) | Decode at 32K / 64K / 128K (tok/s) | TTFT at 8K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Qwen3.8-Flash-Next, four cards](docs/benchmark/qwen38-flash-next-4-card.html) · [long-context](docs/benchmark/qwen38-flash-next-4-card-long-context.html) | DFlash 2 | 281 | 264/532/733 | 5,916, 7,936, 7,897 | 7,520 / 7,451 | 208 / 203 / 211 | 1.0 s |
| [Qwen3.8-Flash-Next, two cards](docs/benchmark/qwen38-flash-next-2-card.html) · [long-context](docs/benchmark/qwen38-flash-next-2-card-long-context.html) | DFlash 2 | 130 | 132/248/299 | 2,610, 2,994, 2,835 | 2,686 / 2,447 | 102 / 99 / 103 | 2.7 s |
| [Qwen3.8-Flash-Next, one card](docs/benchmark/qwen38-flash-next-1-card.html) · [long-context](docs/benchmark/qwen38-flash-next-1-card-long-context.html) | DFlash 2 | 59 | 54/58/65 | 493, 631, 605 | 577 / 545 | 48 / 46 / 51 | 12.7 s |
| [Qwen3.8-27B Swift-1.5, two cards](docs/benchmark/qwen38-27b-swift15-2-card.html) · [long-context](docs/benchmark/qwen38-27b-swift15-2-card-long-context.html) | DSpine b16 | 187 | 180/433/605 | 4,162, 4,581, 4,103 | 3,661 / 2,893 | 165 / 141 / 130 | 1.8 s |
| [Qwen3.8-27B Swift-1.5, one card](docs/benchmark/qwen38-27b-swift15-1-card.html) · [long-context](docs/benchmark/qwen38-27b-swift15-1-card-long-context.html) | DSpine b16 | 108 | 103/256/353 | 2,456, 2,512, 2,184 | 1,909 / - | 87 / 75 / - | 3.2 s |
| [Qwen3.8-27B AutoRound, two cards](docs/benchmark/qwen38-27b-autoround-2-card.html) · [long-context](docs/benchmark/qwen38-27b-autoround-2-card-long-context.html) | DSpine b16 | 173 | 164/418/601 | 4,441, 4,627, 4,049 | 3,512 / 2,706 | 146 / 130 / 123 | 1.7 s |
| [Qwen3.8-27B AutoRound, one card](docs/benchmark/qwen38-27b-autoround-1-card.html) · [long-context](docs/benchmark/qwen38-27b-autoround-1-card-long-context.html) | DSpine b16 | 102 | 98/259/367 | 2,479, 2,521, 2,208 | 1,920 / - | 85 / 80 / - | 3.2 s |
| [Recursive Flint INT4, two cards](docs/benchmark/flint-int4-2-card.html) · [long-context](docs/benchmark/flint-int4-2-card-long-context.html) | DFlash 2 | 422 | 406/670/831 | 12,498, 11,745, 9,878 | 7,915 / 5,470 | 372 / 259 / 237 | 0.7 s |
| [Recursive Flint INT4, one card](docs/benchmark/flint-int4-1-card.html) · [long-context](docs/benchmark/flint-int4-1-card-long-context.html) | DFlash 2 | 278 | 273/445/536 | 8,166, 7,499, 5,859 | 4,532 / 2,963 | 201 / 175 / 140 | 1.1 s |

Every row is one `xenithium-bench run` of a preset (`models/presets/`), the same command on your cards measures
yours. The four-card setup of Qwen3.8-Flash-Next is the fastest; two cards are the default of its preset. Flint's
one-card row uses `models/presets/flint-int4.tp1.env` (8 slots, a 262,144-token max context: the whole model is on
one card). `docs/benchmark/README.md` names the run directory and the settings file of every linked report.

## Stack

Hardware
- 1 or 2 Intel Arc Pro B-series cards (developed on 2x Arc Pro B70). TP2 needs both cards visible.
- VRAM: weights per card at TP2 are about 6.5 GiB (Qwen3.8-27B), 8.8 GiB (Nanbeige) and 10.5 GiB (Flint); the rest goes
  to the KV pool. Flint fits on one card at its full 262K context with int8 KV.

Software (the versions the engine is built and tested with)
- Linux with the `xe` kernel driver (Ubuntu 26.04, kernel 7.0).
- Intel GPU compute runtime: `libze-intel-gpu1` / `intel-opencl-icd` 26.35, `libigc2` 2.41; Level Zero loader
  `libze1` + `libze-dev` 1.32.
- Intel oneAPI 2026.2: the DPC++/C++ compiler (`icpx`, under `/opt/intel/oneapi`) and its SYCL runtime. Other compilers are
  refused.
- oneDNN (the W4A8 prefill GEMMs and the opt-in oneDNN prefill attention): oneAPI's package (3.13.2).
  `scripts/oneapi-env.sh` sets up oneAPI and its oneDNN.
- CMake >= 3.28, a C++20 standard library, POSIX threads. No other third-party libraries.

Check the cards with `xpu-smi discovery` (or `sycl-ls`): the B70s must show up as Level Zero GPUs.

## Install dependencies

Only needed to build and run on the host; the Docker image brings its own (see "Run with Docker Compose"). On Ubuntu
26.04 with the `xe` kernel driver loaded (the stock 26.04 kernel has it):

```bash
scripts/install-deps.sh          # apt (via sudo): oneAPI 2026.2 DPC++ compiler, oneDNN, Level Zero, the GPU driver
                                 # (Level Zero + OpenCL), cmake, ninja, ffmpeg
sudo usermod -aG render,video "$USER"   # access to /dev/dri/renderD*; log out and in again
```

`install-deps.sh` adds two apt sources when they are not configured yet, with their keys from `docker/keys/` (checked by
sha256): Intel's oneAPI repository (`apt.repos.intel.com/oneapi`) and the `kobuk-team/intel-graphics` PPA (the Level
Zero loader `libze1` / `libze-dev`, the user-mode driver `libze-intel-gpu1` and `intel-opencl-icd`, `intel-ocloc`; oneDNN
builds its GPU kernels through the OpenCL device, so without `intel-opencl-icd` every oneDNN call fails with "OpenCL
device not found"). The compiler is pinned to `2026.2.0-863` and oneDNN to `2026.1.0-617` (`ONEAPI_VERSION=...` /
`DNNL_VERSION=...` for others); the PPA keeps only its newest driver build, so the driver is not pinned.
`scripts/install-deps.sh --runtime` installs only what running needs (SYCL runtime, oneDNN, Level Zero, driver, ffmpeg).

`scripts/oneapi-env.sh` (sourced by the build and test scripts) sets up oneAPI from `/opt/intel/oneapi/2026.2` and
its oneDNN; `XEN_ONEAPI=DIR` / `XEN_DNNL_ROOT=DIR` point elsewhere (then delete `build/CMakeCache.txt` and rebuild). To have `icpx` and the runtime libraries in every shell, add
`source ~/xenithium-beta/scripts/oneapi-env.sh` to `~/.bashrc`.

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

Tests: `scripts/test.sh` (ctest: the `host` suites, then the `gpu` gates when the cards are idle; `BUILD_DIR=DIR` for another
build tree, extra arguments go to ctest, e.g. `-R gqa`).

## Run on the host

Build first (above). Three ways, from the simplest:

1. **Launcher (TUI)**: pick a model, edit the settings, start and watch the server.
   ```bash
   scripts/build-host.sh xenithium-launch
   scripts/build-host.sh xenithium-server
   scripts/launch.sh                          # --models DIR to scan another folder for one run
   ```
   See `docs/launcher.md`.

2. **Start script** with a settings file: `scripts/start-server.sh` reads `KEY=value` lines from `./server.env` (or
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

3. **Directly**:
   ```bash
   ./build/xenithium-server --model ~/models/flint-int4 --host 0.0.0.0 --port 8000 --slots 4 --context 32768 --tp 2
   ```

The launcher's `m` list scans a **models folder** for the checkpoints you can serve: by default the project's
`models/` folder (it also holds the per-model settings files: the shipped presets in `models/presets/`, your own
beside them, git-ignored). To serve checkpoints from your own folder, set
`XEN_MODELS_DIR=<folder>` in `server.env` (e.g. `XEN_MODELS_DIR=$HOME/models`) — the launcher scans it for the
model list and the DFlash drafter lists; the server itself never reads it. `--models DIR` to `scripts/launch.sh`
overrides the folder for one run.

Every setting (environment variable, flag, default, which arch reads it, the measurements behind the default) is in
`docs/server-env.md`. The server binds `0.0.0.0:8000` by default; loading takes ~10-15 s plus a few seconds of warm-up,
and it prints `serving on http://...` when ready.

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

The containers see the host the way the native tools do, so the native and the Docker paths are interchangeable:

- **Settings are the host's files**: `server.env` (or `XEN_SETTINGS=/work/<file>`: `/work` is this folder),
  `models/` (your per-model settings and the presets), `.cache/` (hot-expert rankings) and `build/launch/` (the launcher's state). The server container's
  command is `scripts/start-server.sh`.
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

## API

Endpoints (OpenAI-compatible): `/v1/chat/completions` (streaming, reasoning in `reasoning`, tool calls),
`/v1/completions`, `/v1/responses` (stateless: `input` text or items, `instructions`, function tools, streaming
events; no `previous_response_id` / stored responses), `/v1/models`, plus `/health`, `/version` and `/metrics`. With
`XEN_VISION=1` (qwen38, the 27B) chat and Responses messages take images (`image_url` / `input_image`: base64 `data:`
URLs or `http(s)` URLs the server downloads) and videos (`video_url`, decoded by an ffmpeg subprocess: `XEN_VISION_FFMPEG`).

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "flint",
  "messages": [{"role": "user", "content": "What is the capital of Australia?"}],
  "max_tokens": 400
}'
```

Chat templates: by default the checkpoint's own `chat_template.jinja` is rendered by the in-engine Jinja interpreter
(`--chat-template auto|builtin|jinja|FILE`, `XEN_CHAT_TEMPLATE`).

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
| `docs/server-env.md`, `docs/launcher.md`, `docs/bench.md` | settings reference, launcher guide, benchmark guide |
| `docs/benchmark/` | the HTML reports behind the README's performance table, one run per preset and card count |
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
