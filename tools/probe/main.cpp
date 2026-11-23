// xenithium-probe: hardware and kernel measurements on the cards.
// devices Level Zero topology (cards, EU geometry, peer access)
// bw [--device N] [--mib M] [--iters I] plain VRAM read: the floor every kernel is judged by
// w4 [--device N] [--n N] [--k K] [--iters I] [--ksplit S] int4 W4A16 M=4 GEMM vs its floor
// dense [--device N] [--iters I] the dense decode GEMM (int8 W8A16 and bf16) on this model's
// per-card shapes, against the read floor
// pf [--device N] [--iters I] the prefill GEMM (pf_dense) on the same shapes at 256 / 1024 /
// 2048 rows: us and TFLOPS
// load --model DIR [--layer L] [--device N] [--threads T]
// one layer's routed experts end to end: O_DIRECT read, pack into pinned staging,
// H2D in 64 MiB copies, read-back compare, and W4A16 on real experts vs the CPU
// reference computed from the raw checkpoint tensors
// head --model DIR [--iters I] the step's ends on real weights, both cards: embedding
// gather -> final HC mixer -> lm_head (int8) -> greedy pick across the cards; checks
// against host references (same int8 weights, and the bf16 checkpoint), then per-stage times
// hc --model DIR [--iters I] [--rows N] every hyper-connection mixer on real weights (N rows, default 4), identity blocks:
// embed -> 48 x (attn, mlp mixer) -> final mixer -> lm_head -> pick on both cards;
// teacher-forced checks of selected mixers vs the host reference, then step timing
// qsa --model DIR [--layers 3,23,47] [--kv-bytes 1|2] [--kv-v4 0|1] [--context N] [--iters I] QSA decode on
// real weights, both cards: a ~2,400-token history on the chain input with teacher-forced
// stage checks, then chained timing of the 12 attention layers at 8K / 262K / 1M (or N)
// moeflint --model DIR [--layers L] [--iters I] arch flint: one MoE layer (256 experts, TP2 halves, shared expert) vs the
// dequantised reference, then launch timings vs the byte floor of the step's unique experts
// gqaflint [--iters I] [--context N] arch flint: the attention decode at one card's TP2 share (synthetic weights),
// sliding (window 513, KV ring) and full layers, contexts 8K-262K, 1 / 4 / 8 rows, int8 / bf16 KV vs the KV byte floor
// w4load --model DIR [--layers 0,3] [--iters I] qwen35 (Qwen3.8-27B): one layer's pack-quantized Linears through
// DenseStore (Store:w4) on both cards, w4a16_gemm_m4 vs w4a16_ref on independent host slices, then timing
// head35 --model DIR [--iters I] qwen35: the step's ends on real weights, both cards: embedding gather -> final
// (1 + w) norm -> lm_head (int8) -> greedy pick across the cards, vs host references, then per-stage times
// mlp35 --model DIR [--layers 0,3] [--rows N] [--iters I] qwen35: the dense SwiGLU MLP (gate | up W4 -> swiglu -> down
// -> mailbox all-reduce) per layer on both cards vs dense_mlp_ref on independent host slices, then per-stage times
// gdn35 --model DIR [--layers 0,1,2,62] [--iters I] qwen35: the Gated DeltaNet decode block (W4 qkv|z and out_proj, swish
// gate) on real weights, both cards: steps of 1-4 rows / accept 0-4 vs the host reference, then chained timing
// gqa35 --model DIR [--layers 3,31,63] [--kv-bytes 1|2] [--kv-v4 0|1] [--context N] [--iters I] [--dec-attn exact|fast|int8|int8-fast] qwen35: the gated full-attention
// decode block (W4 q|k|v and o_proj, dense softmax over the whole context in splits) on real weights, both cards:
// a ~4,200-token history with teacher-forced stage checks, then chained timing of the 16 layers at 8K-262K (or N)
// model35 --model DIR [--tp 1|2] [--queue-prof 0|1] [--ppl a.bin,b.bin] [--rows 1,4,8,pf] [--lp-ab CASES] [--pf-checks 0|1] [--generate N] [--at P] [--context N] [--kv-bytes 1|2] [--kv-v4 0|1] [--lp-out DIR] [--lp-ref F,F] [--pf-chunk N] [--pf-slices 0|1] [--pf-attn-slices 0|1] [--pf-attn 0|1|2|3] [--pf-p-lo 0|1] [--pf-w4dq 0|1] [--pf-split 0|1] [--pf-ab CASES] [--pf-ab-prof 0|1] [--pf-reps N] [--pf-lens N,N] [--pf-calls R,R]
// prefix --model DIR --ppl F [--mtp DIR] [--prompt-len P] [--generate G] [--seqs TURNS] [--hot-rank F] [--snap-prepin 0|1] (the
// qwen38 prefix cache: call boundaries, snapshot restore / fork / live resume vs cold, costs, an agent session's TTFT
// per turn with the cache off / on / on + live; tools/probe/prefix.cpp)
// prefix35 --model DIR [--tp 1|2] --ppl F [--prompt-len P] [--generate G] [--context N] [--dflash DIR] [--seqs TURNS] [--snap-prepin 0|1] (qwen38-27b
// the prefix cache: snapshot restore / fork / live resume bit-identity, their timings, an agent session's
// TTFT per turn with the cache off and on; tools/probe/qwen35_prefix.cpp)
// select35 --model DIR [--ppl F [--pf-lens P,P]] [--doc F,F] [--sel-budgets B,B] [--generate T] [--step-reps N] [--dec-attn exact|int8-fast] (
// audit C1: page-selected decode attention vs dense at long context: KL, a needle or a document's reply, step times;
// --sel-pf 1 [--pf-a8 1] [--pf-chunk R]: the page-selected prefill instead;
// tools/probe/qwen35_select.cpp)
// qwen35: the whole model on both cards (of qwen38-27b): teacher-forced ppl per row count, greedy
// generation with the decoded text, then 1- and 4-row steps at position P (default 8,192) with the module profile
// store --model DIR... [--dense 1] with --dense 1 the non-expert weights load first (the
// arch's dense plan, int8 trunk) and the hot-expert count comes from the free VRAM left
// after them minus the KV at the asked context and the reserve; then the dense spot checks
// store --model DIR [--context N] [--seqs N] [--kv-bytes 1|2] [--drafter 0|1] [--hot N] [--verify 0|1]
// every routed expert onto both cards (EP2, hot count from the arch budget unless --hot),
// streamed layer by layer; prints the load phases and the per-card ledger, verifies every
// blob by hash, and runs W4A16 on a hot and a cold (pinned host, read over PCIe) expert
// per card against the raw-tensor reference
// rank --model DIR --prompts F.jsonl --out F.xrc [--gen N] [--context N] a hot-expert ranking (XRC1) for a MoE
// checkpoint (qwen38) from the decode routes of chat prompts (rank.cpp; scripts/make-hot-rank.sh)
// model --model DIR [--ppl FILE] [--rows 1,4] [--generate N] [--at P] [--context N] [--kv-bytes 1|2] [--kv-v4 0|1] [--hot N] [--verify 0|1]
// [--routes-out F] [--hot-rank F] [--reserve GIB] [--lp-out DIR] [--lp-ref F,F] [--bf16 PAT,PAT] [--moe-down-bf16 0|1] [--moe-dn-ks N] [--graphs 0|1]
// [--moe-tp 0|1] [--hidden-out DIR] [--gen-out F] [--pf-chunk N] [--pf-moe-a8 0|1] [--pf-dense-a8 0|1|2] [--pf-wire fp32|bf16] [--dec-moe-a8 0|1] [--kv-pool N] [--kv-scramble 0|1] [--kv-lend 0|1] [--kv-lend-keep N] [--kv-lend-idle 0|1] [--moe-gu-ks N] [--gdn-scan 0|1] [--pf-cold-stage N] [--pf-route-trace F] [--pf-time-only 0|1] [--pf-sweep N,N]
// the whole model on both cards (model.cpp): teacher-forced perplexity on a token file at
// each row count, then greedy decode timing; one load for everything
// Every timed loop warms the card by wall time first (warmup.hpp), uses full-entropy data and
// (w4) rotates weight copies totalling > 4x the 24 MiB L2 so the number is a DRAM number.
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <chrono>
#include <thread>

#include "warmup.hpp"
#include "xenithium/arch/qwen38/expert_store.hpp"
#include "xenithium/load/dense_store.hpp"
#include "xenithium/arch/qwen38/head.hpp"
#include "xenithium/arch/qwen38/hc.hpp"
#include "xenithium/arch/qwen38/moe.hpp"
#include "xenithium/kernels/moe/moe.hpp"
#include <functional>
#include <set>
#include <thread>
#include "xenithium/arch/qwen38/experts.hpp"
#include "xenithium/model/arch.hpp"
#include "xenithium/rt/card.hpp"
#include "xenithium/model/bulk_read.hpp"
#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/gemm/pf_dense.hpp"
#include "xenithium/kernels/gemm/w4a16_gemm.hpp"
#include "xenithium/rt/device.hpp"

#include "probe.hpp"
#include "xrv/qsa_capture.hpp"

namespace {

struct Options {
  int device = 0;
  int mib = 512;
  int iters = 100;
  int n = 1280, k = 2560;
  int ksplit = 0;
  std::string model;
  int layer = 0;
  int threads = 16;
  int64_t context = 8192;
  int seqs = 1;
  int kv_bytes = 2;
  int drafter = 0;
  int64_t hot = -1;
  int verify = 1;
  int fill_free = 0;
  std::string layers = "0,24,47";
  int dense = 0;  // MiB left free: after the load, allocate + fill all other VRAM, then verify
  bool context_set = false, kv_bytes_set = false;  // qsa: its own defaults (all three contexts, 8-bit KV)
  std::string ppl, rows = "1,4";  // model
  int generate = 0;
  int64_t at = 0;
  bool verify_set = false;
  std::string routes_out, hot_rank, lp_out, lp_ref;  // model
  double reserve = -1;
  std::string bf16;  // model: numerics switches
  bool moe_down_bf16 = false;
  int moe_dn_ks = 0;
  bool graphs = true;
  bool moe_tp = true;
  std::string hidden_out, gen_out;  // model: MTP drafter inputs, greedy text
  int pf_chunk = 0;  // model: prefill chunk rows (0 = ModelOptions default)
  bool pf_slices = false;  // model35: every prefill component on its decode kernels per 4-row slice
  bool pf_attn_slices = false;  // model35: only the attention on decode slices (A/B)
  int pf_attn = 3;  // model35: gqa_rows attention body (0 decode per 64-row batch, 1 block, 2 staged, 3 fa, 4 oneDNN)
  bool pf_p_lo = true;  // model35: the fa body's P as hi + lo bf16 (1) or hi only (0, numerics switch)
  bool pf_w4dq = false;  // model35: the prefill W4 GEMMs dequantised (1, numerics switch) or exact (0)
  std::string pf_ab;  // model35: prefill A/B cases on one load
  int pf_reps = 3;  // model35: --pf-ab repetitions per case
  std::string pf_lens;  // model35: --pf-ab prompt lengths (prefixes of each file's first chunk); select35: prompt lengths
  std::string sel_budgets;  // select35: page budgets
  std::string docs;  // select35: text files, each one user turn (a document and its question)
  bool pf_split = true;  // model35: split two-card sums in prefill (0 = the chunk mailbox)
  bool pf_time_only = false;  // model: --rows pf times the prefill only (no ppl pass, no gates)
  std::string pf_sweep;  // model: --rows pf, prefill calls of these row counts timed
  std::string mtp, drafts_ref;  // mtp: the drafter checkpoint, mtp-ref --drafts-out records
  int prompt_len = 0, depth = 3, mtp_kv_bytes = 2;
  int prompts = 0, dflash_shift = 0;
  // model35: --spec N speculative tokens per prompt with the MTP drafter; --mtp-dense-bytes its
  // matrices (1 int8, 2 bf16); --mtp-vocab its draft vocabulary (0 = the target's lm_head); --mtp-kv-bytes its KV when
  // given (else the model's default, int8)
  int spec = 0, mtp_dense_bytes = 1, mtp_vocab = 0;
  bool mtp_kv_bytes_set = false;
  // model35: --seqs N sequences on N slots (the first N prompts), --seq-tokens T each; --seq-depths D,D..
  // drafter depths for the speculative batch runs (needs --spec > 0 for the drafter; empty = plain batches only)
  int seq_tokens = 32;  // (--seqs is the shared option; model35 runs the section from 2)
  std::string seq_depths;
  // model35: --step-ab CASES decode step A/B, --step-shapes SxR.., --step-reps N, --profile-card C
  std::string step_ab, step_shapes, step_opts;
  int step_reps = 3, profile_card = 0;
  // model35: --kv-pool N tokens of one KV pool shared by the slots (0 = fixed per slot, -1 = auto from the
  // free VRAM), --kv-scramble 1 hands its pages out in a shuffled order (results must not change)
  long long kv_pool = 0;
  bool kv_scramble = false;
  std::string mtp_dump;  // model35 --spec: teacher-forced drafter outputs of the first prompt for a host reference
  int pf_prompt_len = 0;  // mtp: section 4, the drafter in prefill; 0 = skip
  bool pf_only = false;  // mtp: run section 4 only
  double stall_ms = 100;  // serve: BatchConfig:prefill_stall_ms
};

double event_ms(const sycl::event &e) {
  const auto s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto t = e.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(t - s) / 1e6;
}

struct Stats {
  double best, median, p90;
};
Stats stats(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return {v.front(), v[v.size() / 2], v[std::min(v.size() - 1, v.size() * 9 / 10)]};
}

int cmd_devices() {
  const auto loader = xenithium::l0::Loader::open();
  std::printf("Level Zero API %s, %zu device(s) after ZE_AFFINITY_MASK\n", loader.api_version().c_str(), loader.size());
  std::printf("%-4s %-34s %-13s %9s %5s %10s\n", "dev", "name", "pci", "VRAM GiB", "EUs", "clock MHz");
  for (size_t i = 0; i < loader.size(); ++i) {
    const auto in = loader.info(i);
    std::printf("%-4zu %-34s %-13s %9.2f %5u %10u\n", i, in.name.c_str(), in.pci_bdf().c_str(),
                static_cast<double>(in.vram_bytes) / (1ull << 30), in.eu_total(), in.clock_mhz);
  }
  for (size_t i = 0; i < loader.size(); ++i) {  // the engine's identification (rt/card.hpp), XEN_EMU_CARD applied
    const auto in = loader.info(i);
    std::printf("card %zu: pci id 0x%04x: %s\n", i, in.device_id, xenithium::rt::card_view(in).describe().c_str());
  }
  for (size_t i = 0; i < loader.size(); ++i)
    for (size_t j = 0; j < loader.size(); ++j)
      if (i != j) std::printf("peer %zu -> %zu: %s (driver answer)\n", i, j, loader.can_access_peer(i, j) ? "yes" : "no");
  return 0;
}

class BwRead;

int cmd_bw(const Options &o) {
  const auto loader = xenithium::l0::Loader::open();
  auto dev = xenithium::Device::open(loader, static_cast<size_t>(o.device));
  const size_t bytes = static_cast<size_t>(o.mib) << 20, vecs = bytes / sizeof(sycl::uint4), items = vecs / 4;
  auto *buf = dev.alloc<sycl::uint4>(vecs, xenithium::l0::MemKind::Device);
  auto *sink = dev.alloc<uint32_t>(items, xenithium::l0::MemKind::Device);
  {
    auto *host = dev.alloc<uint32_t>(bytes / 4, xenithium::l0::MemKind::Host);
    std::mt19937 rng(4242);
    for (size_t i = 0; i < bytes / 4; ++i) host[i] = rng();  // full entropy: no compression
    dev.copy_queue().copy(buf, host, bytes, true);
    dev.release(host);
  }
  auto launch = [&] {
    return dev.compute().submit([&](sycl::handler &h) {
      h.parallel_for<BwRead>(sycl::range<1>(items), [=](sycl::id<1> id) {
        const size_t b = id[0] * 4;
        const sycl::uint4 v0 = buf[b], v1 = buf[b + 1], v2 = buf[b + 2], v3 = buf[b + 3];
        const sycl::uint4 x = v0 ^ v1 ^ v2 ^ v3;
        const uint32_t s = x.x() ^ x.y() ^ x.z() ^ x.w();
        if (s == 0xdeadbeefu) sink[id[0]] = s;  // practically never: keeps the loads
      });
    });
  };
  xenithium::probe::warm_for([&] { launch().wait(); });
  std::vector<double> ms;
  for (int i = 0; i < o.iters; ++i) {
    auto e = launch();
    e.wait_and_throw();
    ms.push_back(event_ms(e));
  }
  const Stats s = stats(ms);
  std::printf("plain read, device %d (%s), %d MiB full-entropy buffer, %d iters after a 100 ms warm-up\n", o.device,
              dev.info().pci_bdf().c_str(), o.mib, o.iters);
  std::printf("%-10s %10s %10s\n", "", "ms", "GB/s");
  for (auto [name, v] : {std::pair{"best", s.best}, std::pair{"median", s.median}, std::pair{"p90", s.p90}})
    std::printf("%-10s %10.3f %10.1f\n", name, v, static_cast<double>(bytes) / (v * 1e6));
  dev.release(buf);
  dev.release(sink);
  return 0;
}

int cmd_w4(const Options &o) {
  namespace k = xenithium::kernels;
  using bf16 = sycl::ext::oneapi::bfloat16;
  const k::W4Problem p{o.n, o.k};
  if (!p.valid()) {
    std::fprintf(stderr, "w4: n %% 16 and k %% 128 must be 0\n");
    return 2;
  }
  const auto loader = xenithium::l0::Loader::open();
  auto dev = xenithium::Device::open(loader, static_cast<size_t>(o.device));
  auto &q = dev.compute();
  const size_t wb = p.packed_bytes() + p.scale_count() * 2;
  const int copies = static_cast<int>(std::max<size_t>(1, (96ull << 20) / wb + 1));  // > 4x L2
  std::mt19937 rng(11);
  std::vector<uint32_t> qw(static_cast<size_t>(o.k / 8) * o.n);
  std::vector<uint8_t> packed(p.packed_bytes());
  std::vector<sycl::half> sc(p.scale_count());
  std::vector<uint8_t *> dw(copies);
  std::vector<sycl::half *> ds(copies);
  for (int c = 0; c < copies; ++c) {
    for (auto &v : qw) v = rng();
    for (auto &v : sc) v = sycl::half(std::ldexp(1.f, -static_cast<int>(rng() % 8)));
    k::pack_w4_gptq(qw.data(), o.n, o.k, packed.data());
    dw[c] = dev.alloc<uint8_t>(packed.size(), xenithium::l0::MemKind::Device);
    ds[c] = dev.alloc<sycl::half>(sc.size(), xenithium::l0::MemKind::Device);
    q.memcpy(dw[c], packed.data(), packed.size());
    q.memcpy(ds[c], sc.data(), sc.size() * 2).wait();
  }
  std::vector<bf16> a(4 * static_cast<size_t>(o.k));
  for (auto &v : a) v = bf16(static_cast<float>(static_cast<int>(rng() % 9) - 4));
  auto *da = dev.alloc<bf16>(a.size(), xenithium::l0::MemKind::Device);
  auto *dc = dev.alloc<float>(4 * static_cast<size_t>(o.n), xenithium::l0::MemKind::Device);
  q.memcpy(da, a.data(), a.size() * sizeof(bf16)).wait();
  int it = 0;
  auto launch = [&] {
    const int c = it++ % copies;
    return k::w4a16_gemm_m4(q, da, dw[c], ds[c], dc, o.n, o.k, o.ksplit);
  };
  xenithium::probe::warm_for([&] { launch().wait(); });
  std::vector<double> ms;
  for (int i = 0; i < o.iters; ++i) {
    auto e = launch();
    e.wait_and_throw();
    ms.push_back(event_ms(e));
  }
  const Stats s = stats(ms);
  const double floor_ms = static_cast<double>(wb) / 604e6;
  std::printf("w4a16 M=4 N=%d K=%d ksplit=%d, %.2f MiB weights+scales, %d rotating copies, device %d\n", o.n, o.k,
              o.ksplit ? o.ksplit : k::w4a16_auto_ksplit(o.n, o.k), wb / 1048576.0, copies, o.device);
  std::printf("%-10s %10s %10s %12s\n", "", "us", "GB/s", "x floor");
  for (auto [name, v] : {std::pair{"best", s.best}, std::pair{"median", s.median}, std::pair{"p90", s.p90}})
    std::printf("%-10s %10.2f %10.1f %12.2f\n", name, v * 1e3, static_cast<double>(wb) / (v * 1e6), v / floor_ms);
  for (int c = 0; c < copies; ++c) {
    dev.release(dw[c]);
    dev.release(ds[c]);
  }
  dev.release(da);
  dev.release(dc);
  return 0;
}

int cmd_dense(const Options &o) {
  namespace k = xenithium::kernels;
  using bf16 = sycl::ext::oneapi::bfloat16;
  const auto loader = xenithium::l0::Loader::open();
  auto dev = xenithium::Device::open(loader, static_cast<size_t>(o.device));
  auto &q = dev.compute();
  struct Shape {
    const char *what;
    int n, k;
  };
  const Shape shapes[] = {
      {"GDN in_proj qkv (TP2 half)", 5120, 2560}, {"GDN in_proj z (half)", 3072, 2560},  {"GDN out_proj (half k)", 2560, 3072},
      {"QSA q_proj (half)", 6144, 2560},          {"QSA o_proj (half k)", 2560, 3072},   {"QSA k_proj (half)", 256, 2560},
      {"shared gate|up", 1280, 2560},             {"shared down", 2560, 640},            {"HC down (half)", 160, 10240},
      {"HC up (half)", 5120, 320},                {"router (bf16 only)", 512, 2560},     {"lm_head (half)", 124160, 2560},
  };
  std::printf("dense decode GEMM, M=4, device %d (%s); rotating copies > 96 MiB; floor = bytes / 604 GB/s\n", o.device,
              dev.info().pci_bdf().c_str());
  std::printf("%-30s %7s %6s | %9s %9s %8s | %9s %9s %8s\n", "shape", "n", "k", "i8 us", "GB/s", "x floor", "bf16 us", "GB/s", "x floor");
  std::mt19937 rng(3);
  for (const Shape &s : shapes) {
    double res[2][3] = {};
    for (const bool i8 : {true, false}) {
      const size_t wbytes = static_cast<size_t>(s.n) * s.k * (i8 ? 1 : 2);
      const size_t bytes = wbytes + (i8 ? s.n * 4ull : 0);
      const int copies = static_cast<int>(std::max<size_t>(1, (96ull << 20) / bytes + 1));
      std::vector<uint8_t> host(wbytes);
      std::vector<void *> ws(copies);
      std::vector<float *> ss(copies);
      std::vector<float> sc(s.n);
      for (auto &v : sc) v = 0.001f * (1 + rng() % 100);
      for (int c = 0; c < copies; ++c) {
        for (size_t i = 0; i < host.size(); i += 4) {
          const uint32_t r = rng();
          std::memcpy(&host[i], &r, std::min<size_t>(4, host.size() - i));
        }
        if (!i8)  // keep bf16 finite: clear the exponent's top bits
          for (size_t i = 1; i < host.size(); i += 2) host[i] &= 0xBF;
        ws[c] = dev.alloc<uint8_t>(wbytes, xenithium::l0::MemKind::Device);
        ss[c] = dev.alloc<float>(s.n, xenithium::l0::MemKind::Device);
        q.memcpy(ws[c], host.data(), wbytes);
        q.memcpy(ss[c], sc.data(), s.n * 4ull).wait();
      }
      std::vector<bf16> x(4 * static_cast<size_t>(s.k));
      for (auto &v : x) v = bf16(static_cast<float>(static_cast<int>(rng() % 9) - 4));
      auto *dx = dev.alloc<bf16>(x.size(), xenithium::l0::MemKind::Device);
      auto *dy = dev.alloc<float>(4 * static_cast<size_t>(s.n), xenithium::l0::MemKind::Device);
      q.memcpy(dx, x.data(), x.size() * sizeof(bf16)).wait();
      int it = 0;
      auto launch = [&] {
        const int c = it++ % copies;
        k::DenseWeights w;
        w.kind = i8 ? k::DenseKind::i8 : k::DenseKind::bf16;
        w.w = ws[c];
        w.scale = i8 ? ss[c] : nullptr;
        w.n = s.n;
        w.k = s.k;
        return k::dense_gemm_m4(q, dx, w, dy);
      };
      xenithium::probe::warm_for([&] { launch().wait(); });
      std::vector<double> ms;
      for (int i = 0; i < o.iters; ++i) {
        auto e = launch();
        e.wait_and_throw();
        ms.push_back(event_ms(e));
      }
      const Stats st = stats(ms);
      res[i8 ? 0 : 1][0] = st.median * 1e3;
      res[i8 ? 0 : 1][1] = static_cast<double>(bytes) / (st.median * 1e6);
      res[i8 ? 0 : 1][2] = st.median / (static_cast<double>(bytes) / 604e6);
      for (int c = 0; c < copies; ++c) {
        dev.release(ws[c]);
        dev.release(ss[c]);
      }
      dev.release(dx);
      dev.release(dy);
    }
    std::printf("%-30s %7d %6d | %9.2f %9.1f %8.2f | %9.2f %9.1f %8.2f\n", s.what, s.n, s.k, res[0][0], res[0][1], res[0][2], res[1][0],
                res[1][1], res[1][2]);
  }
  return 0;
}

double since_ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Prefill GEMM: pf_dense on the same per-card shapes as `dense, at chunk sizes
// 256 / 1024 / 2048 rows. Random int8 / bf16 weights (rotating copies > 96 MiB), random bf16
// activations; TFLOPS = 2 m n k / time. The last column is one 4-row decode launch
// (dense_gemm_m4) on the same weights, for scale.
int cmd_pf(const Options &o) {
  namespace k = xenithium::kernels;
  using bf16 = sycl::ext::oneapi::bfloat16;
  const auto loader = xenithium::l0::Loader::open();
  auto dev = xenithium::Device::open(loader, static_cast<size_t>(o.device));
  auto &q = dev.compute();
  struct Shape {
    const char *what;
    int n, k;
  };
  const Shape shapes[] = {
      {"GDN in_proj qkv (TP2 half)", 5120, 2560}, {"GDN in_proj z (half)", 3072, 2560}, {"GDN out_proj (half k)", 2560, 3072},
      {"QSA q_proj (half)", 6144, 2560},          {"QSA o_proj (half k)", 2560, 3072},  {"shared gate|up", 1280, 2560},
      {"shared down", 2560, 640},                 {"HC down|inject", 336, 10240},       {"HC up", 10240, 320},
      {"PLE kv", 6400, 2560},                     {"router (bf16)", 528, 2560},
  };
  const int ms_list[] = {256, 512, 1024, 2048};
  std::printf("prefill GEMM (pf_dense), device %d (%s); int8 weights unless named bf16; TFLOPS = 2mnk / t; host wall per call (W8A8 rows,\n"
              "round 75: the quantiser launch + the GEMM; W8A8 row: round 119's per-row scales)\n", o.device, dev.info().pci_bdf().c_str());
  std::printf("%-34s %6s %6s", "shape", "n", "k");
  for (int m : ms_list) std::printf(" | m=%-4d %8s %6s", m, "us", "TFLOPS");
  std::printf(" | decode m4 us/4 rows\n");
  std::mt19937 rng(5);
  for (const Shape &s : shapes) {
    const bool i8 = std::strstr(s.what, "bf16") == nullptr;
    const size_t wbytes = static_cast<size_t>(s.n) * s.k * (i8 ? 1 : 2);
    const int copies = static_cast<int>(std::max<size_t>(1, (96ull << 20) / wbytes + 1));
    std::vector<uint8_t> host(wbytes);
    std::vector<void *> ws(copies);
    std::vector<float> sc(s.n);
    for (auto &v : sc) v = 0.001f * (1 + rng() % 100);
    auto *ds = dev.alloc<float>(s.n, xenithium::l0::MemKind::Device);
    q.memcpy(ds, sc.data(), s.n * 4ull).wait();
    for (int c = 0; c < copies; ++c) {
      for (size_t i = 0; i < host.size(); i += 4) {
        const uint32_t r = rng();
        std::memcpy(&host[i], &r, std::min<size_t>(4, host.size() - i));
      }
      if (!i8)
        for (size_t i = 1; i < host.size(); i += 2) host[i] &= 0xBF;
      ws[c] = dev.alloc<uint8_t>(wbytes, xenithium::l0::MemKind::Device);
      q.memcpy(ws[c], host.data(), wbytes).wait();
    }
    const int mmax = k::kPfRows;
    std::vector<bf16> x(static_cast<size_t>(mmax) * s.k);
    for (auto &v : x) v = bf16(static_cast<float>(static_cast<int>(rng() % 17) - 8) * 0.125f);
    auto *dx = dev.alloc<bf16>(x.size(), xenithium::l0::MemKind::Device);
    auto *dy = dev.alloc<bf16>(static_cast<size_t>(mmax) * s.n, xenithium::l0::MemKind::Device);
    auto *dy4 = dev.alloc<float>(4ull * s.n, xenithium::l0::MemKind::Device);
    auto *dq = dev.alloc<int8_t>(static_cast<size_t>(mmax) * s.k, xenithium::l0::MemKind::Device);
    auto *dqs = dev.alloc<float>(k::pf_a8_scales(mmax, s.k), xenithium::l0::MemKind::Device);
    const k::PfA8 a8{dq, dqs};
    k::PfA8 a8r{dq, dqs};  // per-row scales
    a8r.row = true;
    q.memcpy(dx, x.data(), x.size() * sizeof(bf16)).wait();
    int it = 0;
    auto weights = [&] {
      k::DenseWeights w;
      w.kind = i8 ? k::DenseKind::i8 : k::DenseKind::bf16;
      w.w = ws[it++ % copies];
      w.scale = i8 ? ds : nullptr;
      w.n = s.n;
      w.k = s.k;
      return w;
    };
    for (const int mode : {0, 1, 2}) {  // W8A16, W8A8 per group, W8A8 per row
      const bool use8 = mode != 0;
      if (use8 && !i8) continue;
      char what[64];
      std::snprintf(what, sizeof what, "%s%s", s.what, mode == 1 ? " W8A8" : mode == 2 ? " W8A8 row" : "");
      std::printf("%-34s %6d %6d", what, s.n, s.k);
      for (int m : ms_list) {
        // Both variants by the host wall around submit + wait (W8A8 is two launches, the quantiser and the GEMM;
        // the GEMM's event alone would hide the quantiser). Rounds before 75 printed the GEMM's event time: the
        // wall adds ~10-20 us per launch to those numbers.
        std::vector<double> t;
        auto launch = [&] { k::pf_dense(q, m, weights(), dx, s.k, k::PfOut{nullptr, dy, 0}, mode == 1 ? &a8 : mode == 2 ? &a8r : nullptr).wait_and_throw(); };
        xenithium::probe::warm_for([&] { launch(); });
        for (int i = 0; i < o.iters; ++i) {
          const auto t0 = std::chrono::steady_clock::now();
          launch();
          t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const double med = stats(t).median;
        std::printf(" | %6s %8.1f %6.1f", "", med * 1e3, 2.0 * m * s.n * s.k / (med * 1e9));
      }
      if (use8) {
        std::printf(" | %8s\n", "");
        continue;
      }
      auto launch = [&] { return k::dense_gemm_m4(q, dx, weights(), dy4); };
      xenithium::probe::warm_for([&] { launch().wait(); });
      std::vector<double> t;
      for (int i = 0; i < o.iters; ++i) {
        auto e = launch();
        e.wait_and_throw();
        t.push_back(event_ms(e));
      }
      std::printf(" | %8.1f\n", stats(t).median * 1e3);
    }
    for (int c = 0; c < copies; ++c) dev.release(ws[c]);
    dev.release(ds);
    dev.release(dx);
    dev.release(dy);
    dev.release(dy4);
    dev.release(dq);
    dev.release(dqs);
  }
  return 0;
}

int cmd_load(const Options &o) {
  namespace m = xenithium::model;
  namespace q38 = xenithium::arch::qwen38;
  namespace k = xenithium::kernels;
  using bf16 = sycl::ext::oneapi::bfloat16;
  if (o.model.empty()) {
    std::fprintf(stderr, "load: --model DIR is required\n");
    return 2;
  }
  const auto mc = m::ModelConfig::load(o.model);
  const q38::Config cfg = q38::Config::parse(mc);
  const m::Checkpoint ck = m::Checkpoint::open(o.model);
  const auto lay = q38::ExpertLayout::of(cfg);
  const int E = cfg.experts, L = o.layer;
  if (L < 0 || L >= cfg.layers) {
    std::fprintf(stderr, "load: layer %d out of range\n", L);
    return 2;
  }

  // 1. read: every tensor of the layer's experts, O_DIRECT, merged ranges.
  std::vector<const m::TensorInfo *> ts;
  for (int e = 0; e < E; ++e)
    for (const auto *t : q38::expert_tensors(ck, L, e)) ts.push_back(t);
  const m::BulkBuffer buf = m::bulk_read(ck, ts);
  const auto &rs = buf.stats();

  // 2. pack into pinned host staging (where the device copy starts).
  const auto loader = xenithium::l0::Loader::open();
  auto dev = xenithium::Device::open(loader, static_cast<size_t>(o.device));
  const size_t total = static_cast<size_t>(E) * lay.stride;
  auto *stage = dev.alloc<uint8_t>(total, xenithium::l0::MemKind::Host);
  auto t0 = std::chrono::steady_clock::now();
  {
    std::vector<std::thread> th;
    std::vector<std::string> err(o.threads);
    for (int t = 0; t < o.threads; ++t)
      th.emplace_back([&, t] {
        try {
          for (int e = t; e < E; e += o.threads) q38::pack_expert(lay, ck, buf, L, e, stage + static_cast<size_t>(e) * lay.stride);
        } catch (const std::exception &x) {
          err[t] = x.what();
        }
      });
    for (auto &x : th) x.join();
    for (const auto &x : err)
      if (!x.empty()) throw std::runtime_error(x);
  }
  const double pack_ms = since_ms(t0);

  // 3. H2D: 64 MiB copies in ONE submit on the copy engine.
  auto *blobs = dev.alloc<uint8_t>(total, xenithium::l0::MemKind::Device);
  constexpr size_t kChunk = 64ull << 20;
  auto &cq = dev.copy_queue();
  std::vector<xenithium::l0::Queue::CopyOp> ops;
  for (size_t off = 0; off < total; off += kChunk) ops.push_back({blobs + off, stage + off, std::min(kChunk, total - off)});
  t0 = std::chrono::steady_clock::now();
  cq.copy_batch(ops, true);
  const double h2d_ms = since_ms(t0);
  t0 = std::chrono::steady_clock::now();
  cq.copy_batch(ops, true);  // same bytes again: the first pass also pays first-touch of fresh VRAM
  const double h2d2_ms = since_ms(t0);
  // Pre-touch alternative: a fresh buffer filled on the card first (fill ~700 GB/s), then the copy.
  auto *blobs2 = dev.alloc<uint8_t>(total, xenithium::l0::MemKind::Device);
  for (auto &op : ops) op.dst = blobs2 + (static_cast<uint8_t *>(op.dst) - blobs);
  t0 = std::chrono::steady_clock::now();
  cq.fill(blobs2, 0, total, true);
  const double fill_ms = since_ms(t0);
  t0 = std::chrono::steady_clock::now();
  cq.copy_batch(ops, true);
  const double h2d3_ms = since_ms(t0);
  dev.release(blobs2);

  // 4. read back and compare, expert by expert.
  std::vector<uint8_t> back(total);
  cq.copy(back.data(), blobs, total, true);
  int bad_blobs = 0;
  for (int e = 0; e < E; ++e)
    if (std::memcmp(back.data() + static_cast<size_t>(e) * lay.stride, stage + static_cast<size_t>(e) * lay.stride, lay.stride))
      ++bad_blobs;

  std::printf("layer %d, %d experts, device %d (%s)\n", L, E, o.device, dev.info().pci_bdf().c_str());
  std::printf("expert blob %zu B (stride %zu): gate|up codes @%zu, down codes @%zu, gate|up scales @%zu, down scales @%zu\n",
              lay.bytes, lay.stride, lay.gu_codes, lay.dn_codes, lay.gu_scales, lay.dn_scales);
  std::printf("%-34s %10s %10s %10s\n", "stage", "MiB", "ms", "GB/s");
  std::printf("%-34s %10.1f %10.1f %10.2f\n", rs.direct ? "read (O_DIRECT, 8 threads)" : "read (buffered)",
              rs.read_bytes / 1048576.0, rs.seconds * 1e3, rs.gbs());
  std::printf("%-34s %10.1f %10.1f %10.2f\n", ("pack into pinned host (" + std::to_string(o.threads) + " threads)").c_str(),
              total / 1048576.0, pack_ms, total / (pack_ms * 1e6));
  std::printf("%-34s %10.1f %10.1f %10.2f\n", "H2D, fresh VRAM (64 MiB copies)", total / 1048576.0, h2d_ms, total / (h2d_ms * 1e6));
  std::printf("%-34s %10.1f %10.1f %10.2f\n", "H2D, same VRAM again", total / 1048576.0, h2d2_ms, total / (h2d2_ms * 1e6));
  std::printf("%-34s %10.1f %10.1f %10.2f\n", "fill fresh VRAM (copy engine)", total / 1048576.0, fill_ms, total / (fill_ms * 1e6));
  std::printf("%-34s %10.1f %10.1f %10.2f\n", "H2D after that fill", total / 1048576.0, h2d3_ms, total / (h2d3_ms * 1e6));
  std::printf("read: %zu ranges, %zu pieces, %d buffered retries; tensors %.1f MiB of %.1f MiB read\n", rs.ranges, rs.pieces,
              rs.buffered_retries, rs.tensor_bytes / 1048576.0, rs.read_bytes / 1048576.0);
  std::printf("[%s] device read-back: %d of %d expert blobs differ from the host pack\n", bad_blobs ? "FAIL" : "PASS", bad_blobs, E);

  // 5. W4A16 on real experts vs the CPU reference from the raw checkpoint tensors.
  auto &qc = dev.compute();
  const int H = cfg.hidden, I = cfg.moe_inter;
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.f, 1.f);
  auto act = [&](int kdim, std::vector<bf16> &a, std::vector<float> &af) {
    a.resize(4 * static_cast<size_t>(kdim));
    af.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] = bf16(nd(rng));
      af[i] = static_cast<float>(a[i]);
    }
  };
  std::vector<bf16> a_gu, a_dn;
  std::vector<float> af_gu, af_dn;
  act(H, a_gu, af_gu);
  act(I, a_dn, af_dn);
  auto *d_a = dev.alloc<bf16>(4 * static_cast<size_t>(H), xenithium::l0::MemKind::Device);
  auto *d_c = dev.alloc<float>(4 * static_cast<size_t>(std::max(2 * I, H)), xenithium::l0::MemKind::Device);
  auto scales_f = [&](const m::TensorInfo &t) {
    const auto *h = buf.as<sycl::half>(t);
    std::vector<float> f(t.bytes / 2);
    for (size_t i = 0; i < f.size(); ++i) f[i] = static_cast<float>(h[i]);
    return f;
  };
  double worst = 0;
  int fails = 0;
  for (const int e : {0, 1, E / 2 - 1, E / 2, E - 1}) {
    const uint8_t *blob = blobs + static_cast<size_t>(e) * lay.stride;
    struct Part {
      const char *what;
      int n, kdim;
      size_t codes, scales;
      const std::vector<bf16> *a;
      const std::vector<float> *af;
    };
    for (const Part p : {Part{"gate|up", 2 * I, H, lay.gu_codes, lay.gu_scales, &a_gu, &af_gu},
                         Part{"down", H, I, lay.dn_codes, lay.dn_scales, &a_dn, &af_dn}}) {
      qc.memcpy(d_a, p.a->data(), p.a->size() * sizeof(bf16)).wait();
      qc.fill(d_c, -12345.f, 4 * static_cast<size_t>(p.n)).wait();  // poison
      k::w4a16_gemm_m4(qc, d_a, blob + p.codes, reinterpret_cast<const sycl::half *>(blob + p.scales), d_c, p.n, p.kdim)
          .wait_and_throw();
      std::vector<float> got(4 * static_cast<size_t>(p.n));
      qc.memcpy(got.data(), d_c, got.size() * sizeof(float)).wait();
      // Reference per checkpoint matrix (gate and up separately: proves the concatenation).
      std::vector<double> want(got.size());
      const std::vector<const char *> projs = p.n == 2 * I ? std::vector<const char *>{"gate_proj", "up_proj"}
                                                           : std::vector<const char *>{"down_proj"};
      const int pn = p.n / static_cast<int>(projs.size());
      for (size_t j = 0; j < projs.size(); ++j) {
        const auto &qt = ck.at(q38::expert_name(L, e, projs[j], "qweight"));
        const auto sf = scales_f(ck.at(q38::expert_name(L, e, projs[j], "scales")));
        std::vector<double> c(4 * static_cast<size_t>(pn));
        k::w4a16_ref(p.af->data(), buf.as<uint32_t>(qt), sf.data(), 4, pn, p.kdim, c.data());
        for (int r = 0; r < 4; ++r)
          for (int col = 0; col < pn; ++col)
            want[static_cast<size_t>(r) * p.n + j * pn + col] = c[static_cast<size_t>(r) * pn + col];
      }
      double maxref = 0, maxdiff = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        maxref = std::max(maxref, std::fabs(want[i]));
        maxdiff = std::max(maxdiff, std::fabs(got[i] - want[i]));
      }
      const double rel = maxdiff / maxref;
      worst = std::max(worst, rel);
      const bool ok = rel <= 1e-5;
      fails += !ok;
      std::printf("[%s] expert %3d %-7s N=%4d K=%4d: max |dev - ref| / max |ref| = %.2e\n", ok ? "PASS" : "FAIL", e, p.what, p.n,
                  p.kdim, rel);
    }
  }
  std::printf("%s: worst relative error %.2e (bound 1e-5: fp32 group sums vs the double reference)\n",
              (fails || bad_blobs) ? "FAILED" : "OK", worst);
  dev.release(d_a);
  dev.release(d_c);
  dev.release(blobs);
  dev.release(stage);
  return (fails || bad_blobs) ? 1 : 0;
}

// W4A16 of one expert's gate|up and down on `q` from the blob at `blob` (device or pinned
// host), against w4a16_ref on the raw checkpoint tensors. Returns the worst relative error.
double check_expert(sycl::queue &q, xenithium::Fabric &fab, size_t card, const xenithium::model::Checkpoint &ck,
                    const xenithium::arch::qwen38::Config &cfg, const xenithium::arch::qwen38::ExpertLayout &lay, int layer,
                    int e, const uint8_t *blob) {
  namespace k = xenithium::kernels;
  namespace q38 = xenithium::arch::qwen38;
  using bf16 = sycl::ext::oneapi::bfloat16;
  const int H = cfg.hidden, I = cfg.moe_inter;
  const auto buf = xenithium::model::bulk_read(ck, q38::expert_tensors(ck, layer, e));
  std::mt19937 rng(static_cast<unsigned>(layer * 1000 + e));
  std::normal_distribution<float> nd(0.f, 1.f);
  auto *d_a = static_cast<bf16 *>(fab.alloc(card, 4 * static_cast<size_t>(H) * sizeof(bf16)));
  auto *d_c = static_cast<float *>(fab.alloc(card, 4 * static_cast<size_t>(H) * sizeof(float)));
  double worst = 0;
  for (const bool gu : {true, false}) {
    const int n = gu ? 2 * I : H, kd = gu ? H : I;
    std::vector<bf16> a(4 * static_cast<size_t>(kd));
    std::vector<float> af(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
      a[i] = bf16(nd(rng));
      af[i] = static_cast<float>(a[i]);
    }
    q.memcpy(d_a, a.data(), a.size() * sizeof(bf16)).wait();
    q.fill(d_c, -12345.f, 4 * static_cast<size_t>(n)).wait();
    k::w4a16_gemm_m4(q, d_a, blob + (gu ? lay.gu_codes : lay.dn_codes),
                     reinterpret_cast<const sycl::half *>(blob + (gu ? lay.gu_scales : lay.dn_scales)), d_c, n, kd)
        .wait_and_throw();
    std::vector<float> got(4 * static_cast<size_t>(n));
    q.memcpy(got.data(), d_c, got.size() * sizeof(float)).wait();
    const std::vector<const char *> projs = gu ? std::vector<const char *>{"gate_proj", "up_proj"} : std::vector<const char *>{"down_proj"};
    const int pn = n / static_cast<int>(projs.size());
    double maxref = 0, maxdiff = 0;
    for (size_t j = 0; j < projs.size(); ++j) {
      const auto &st = ck.at(q38::expert_name(layer, e, projs[j], "scales"));
      const auto *hs = buf.as<sycl::half>(st);
      std::vector<float> sf(st.bytes / 2);
      for (size_t i = 0; i < sf.size(); ++i) sf[i] = static_cast<float>(hs[i]);
      std::vector<double> ref(4 * static_cast<size_t>(pn));
      k::w4a16_ref(af.data(), buf.as<uint32_t>(ck.at(q38::expert_name(layer, e, projs[j], "qweight"))), sf.data(), 4, pn, kd, ref.data());
      for (int r = 0; r < 4; ++r)
        for (int col = 0; col < pn; ++col) {
          const double w = ref[static_cast<size_t>(r) * pn + col];
          maxref = std::max(maxref, std::fabs(w));
          maxdiff = std::max(maxdiff, std::fabs(got[static_cast<size_t>(r) * n + j * pn + col] - w));
        }
    }
    worst = std::max(worst, maxdiff / maxref);
  }
  fab.release(d_a);
  fab.release(d_c);
  return worst;
}

// Device dense GEMM on card `card`'s slice of `name` vs the double reference on the same slice
// built here from the raw tensor (rows: explicit per-section ranges; cols: column range).
// Returns {rel err vs the quantised reference, rel err vs bf16 (the quantisation)}.
std::pair<double, double> check_dense(xenithium::Fabric &fab, const xenithium::load::DenseStore &ds, const xenithium::model::Checkpoint &ck,
                                      const xenithium::model::DenseSpec &sp, int card) {
  namespace k = xenithium::kernels;
  namespace m = xenithium::model;
  using bf16 = sycl::ext::oneapi::bfloat16;
  const auto &t = ck.at(sp.name);
  const auto buf = m::bulk_read(ck, {&t});
  const auto *w = buf.as<uint16_t>(t);
  const int64_t d0 = t.dim(0), cols_all = t.numel() / d0;
  std::vector<int64_t> rows;
  int64_t c0 = 0, cols = cols_all;
  if (sp.split == m::DenseSpec::Split::rows) {
    std::vector<int64_t> secs = sp.sections.empty() ? std::vector<int64_t>{d0} : sp.sections;
    int64_t off = 0;
    for (int64_t s : secs) {
      for (int64_t r = off + s / 2 * card; r < off + s / 2 * (card + 1); ++r) rows.push_back(r);
      off += s;
    }
  } else {
    for (int64_t r = 0; r < d0; ++r) rows.push_back(r);
    if (sp.split == m::DenseSpec::Split::cols) {
      cols = cols_all / 2;
      c0 = cols * card;
    }
  }
  const int kd = static_cast<int>(cols);
  std::vector<uint16_t> slice(rows.size() * static_cast<size_t>(kd));
  for (size_t r = 0; r < rows.size(); ++r) std::memcpy(&slice[r * kd], w + static_cast<size_t>(rows[r]) * cols_all + c0, kd * 2ull);
  for (const auto &an : sp.append) {  // appended tensors' rows follow (their own rows split, if any)
    const auto &at = ck.at(an);
    const auto abuf = m::bulk_read(ck, {&at});
    const auto *aw = abuf.as<uint16_t>(at);
    const int64_t a0 = sp.split == m::DenseSpec::Split::rows ? at.dim(0) / 2 * card : 0;
    const int64_t a1 = sp.split == m::DenseSpec::Split::rows ? at.dim(0) / 2 * (card + 1) : at.dim(0);
    slice.insert(slice.end(), aw + static_cast<size_t>(a0) * kd, aw + static_cast<size_t>(a1) * kd);
  }
  const int n = static_cast<int>(slice.size() / kd);
  std::mt19937 rng(77);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<bf16> x(4 * static_cast<size_t>(kd));
  std::vector<float> xf(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = bf16(nd(rng));
    xf[i] = static_cast<float>(x[i]);
  }
  std::vector<double> ref_q(4 * static_cast<size_t>(n)), ref_b(ref_q.size());
  k::dense_ref_bf16(xf.data(), slice.data(), 4, n, kd, ref_b.data());
  if (sp.store == m::DenseSpec::Store::i8) {
    std::vector<int8_t> codes(slice.size());
    std::vector<float> sc(n);
    k::quantize_i8_rows(slice.data(), n, kd, codes.data(), sc.data(), 1);
    k::dense_ref_i8(xf.data(), codes.data(), sc.data(), 4, n, kd, ref_q.data());
  } else {
    ref_q = ref_b;
  }
  const auto &dt = ds.at(card, sp.name);
  const auto dw = dt.dense();
  auto &q = fab.compute(card);
  auto *dx = static_cast<bf16 *>(fab.alloc(card, x.size() * sizeof(bf16)));
  auto *dy = static_cast<float *>(fab.alloc(card, 4ull * dw.n * sizeof(float)));
  q.memcpy(dx, x.data(), x.size() * sizeof(bf16)).wait();
  q.fill(dy, -12345.f, 4ull * dw.n).wait();
  k::dense_gemm_m4(q, dx, dw, dy).wait_and_throw();
  std::vector<float> got(4ull * dw.n);
  q.memcpy(got.data(), dy, got.size() * 4).wait();
  fab.release(dx);
  fab.release(dy);
  double mr = 0, dq = 0, db = 0, nb = 0;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < n; ++c) {
      const double g = got[static_cast<size_t>(r) * dw.n + c], a = ref_q[static_cast<size_t>(r) * n + c], b = ref_b[static_cast<size_t>(r) * n + c];
      mr = std::max(mr, std::fabs(a));
      dq = std::max(dq, std::fabs(g - a));
      db += (g - b) * (g - b);
      nb += b * b;
    }
  return {dq / mr, std::sqrt(db / nb)};
}

int cmd_head(const Options &o) {
  namespace m = xenithium::model;
  namespace k = xenithium::kernels;
  namespace q38 = xenithium::arch::qwen38;
  using bf16 = sycl::ext::oneapi::bfloat16;
  if (o.model.empty()) {
    std::fprintf(stderr, "head: --model DIR is required\n");
    return 2;
  }
  const auto mc = m::ModelConfig::load(o.model);
  const auto arch = m::open_arch(mc);
  const q38::Config cfg = q38::Config::parse(mc);
  const m::Checkpoint ck = m::Checkpoint::open(o.model);
  const std::string P = "model.language_model.", EMB = P + "embed_tokens.weight", NW = P + "hyper_connection_mixer.hc_norm.weight",
                    DW = P + "hyper_connection_mixer.input_mix_weight_down.weight", UW = P + "hyper_connection_mixer.input_mix_weight_up.weight",
                    LM = "lm_head.weight";
  std::vector<m::DenseSpec> plan;
  for (const auto &sp : arch->dense_plan(ck, m::DeployParams{}))
    if (sp.name == EMB || sp.name == NW || sp.name == DW || sp.name == UW || sp.name == LM) plan.push_back(sp);
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, {0, 1});
  xenithium::load::DenseStore dense(fab);
  dense.load(ck, plan, o.threads);
  std::printf("head: %zu tensors loaded in %.2f s (embedding %.2f GiB pinned host)\n", plan.size(), dense.stats().total_s,
              dense.stats().host_bytes / 1073741824.0);
  q38::Head head(fab, dense, cfg);
  const auto &hs = head.hc_shape();
  const int rows = q38::Head::kRows, H = hs.hidden, W = hs.width(), R = hs.lowrank, V = head.vocab_per_card();
  const int32_t ids[rows] = {11, 1000, 50000, 248043};
  head.set_ids(ids, rows);
  head.embed(rows);
  head.final_mix(rows);
  head.lm_pick(rows);
  head.wait();
  int rc = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    rc |= !ok;
  };
  // ---- device results ----
  auto read_bf = [&](int c, const bf16 *p, size_t n) {
    std::vector<uint16_t> v(n);
    fab.compute(c).memcpy(v.data(), p, n * 2).wait();
    return v;
  };
  std::vector<uint16_t> res_d[2], out_d[2];
  std::vector<float> lg[2];
  std::vector<int32_t> tok[2];
  for (int c = 0; c < 2; ++c) {
    res_d[c] = read_bf(c, head.residual(c), static_cast<size_t>(rows) * W);
    out_d[c] = read_bf(c, head.card(c).out, static_cast<size_t>(rows) * H);
    lg[c].resize(static_cast<size_t>(rows) * V);
    fab.compute(c).memcpy(lg[c].data(), head.card(c).logits, lg[c].size() * 4).wait();
    tok[c] = head.tokens(c, rows);
  }
  // ---- host references from the raw checkpoint ----
  const auto *table = reinterpret_cast<const uint16_t *>(dense.at(0, EMB).ptr);
  std::vector<uint16_t> res_ref(static_cast<size_t>(rows) * W);
  for (int r = 0; r < rows; ++r)
    for (int s = 0; s < hs.streams; ++s) std::memcpy(&res_ref[(static_cast<size_t>(r) * hs.streams + s) * H], table + static_cast<size_t>(ids[r]) * H, H * 2);
  check(res_d[0] == res_ref && res_d[1] == res_ref, "embedding gather (pinned host table): residual == table rows on both cards");
  const auto &tn = ck.at(NW), &td = ck.at(DW), &tu = ck.at(UW), &tl = ck.at(LM);
  const auto raw = m::bulk_read(ck, {&tn, &td, &tu, &tl});
  auto widen = [](const uint16_t *b, size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = k::bf16_value(b[i]);
    return v;
  };
  auto quant = [](const uint16_t *b, int n, int kk, std::vector<float> &vals, std::vector<float> &scale) {
    std::vector<int8_t> codes(static_cast<size_t>(n) * kk);
    scale.resize(n);
    k::quantize_i8_rows(b, n, kk, codes.data(), scale.data(), 1);
    vals.assign(codes.begin(), codes.end());
  };
  std::vector<float> dq, ds, uq, us;
  quant(raw.as<uint16_t>(td), R, W, dq, ds);
  quant(raw.as<uint16_t>(tu), W, R, uq, us);
  const auto db = widen(raw.as<uint16_t>(td), static_cast<size_t>(R) * W), ub = widen(raw.as<uint16_t>(tu), static_cast<size_t>(W) * R);
  std::vector<uint16_t> out_i8(static_cast<size_t>(rows) * H), out_bf(out_i8.size()), tmp = res_ref;
  k::hc_mix_ref(hs, rows, tmp.data(), nullptr, nullptr, raw.as<uint16_t>(tn), {dq.data(), ds.data(), R, W}, {uq.data(), us.data(), W, R},
                nullptr, nullptr, out_i8.data());
  tmp = res_ref;
  k::hc_mix_ref(hs, rows, tmp.data(), nullptr, nullptr, raw.as<uint16_t>(tn), {db.data(), nullptr, R, W}, {ub.data(), nullptr, W, R},
                nullptr, nullptr, out_bf.data());
  auto ulp_stats = [](const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, int &mx, size_t &off) {
    mx = 0;
    off = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      auto lin = [](uint16_t v) { return (v & 0x8000) ? -static_cast<int>(v & 0x7fff) : static_cast<int>(v); };
      const int u = std::abs(lin(a[i]) - lin(b[i]));
      mx = std::max(mx, u);
      off += u != 0;
    }
  };
  auto rel = [](const std::vector<uint16_t> &a, const std::vector<uint16_t> &b) {
    double d = 0, n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      const double x = k::bf16_value(a[i]), y = k::bf16_value(b[i]);
      d += (x - y) * (x - y);
      n += y * y;
    }
    return std::sqrt(d / n);
  };
  int mx;
  size_t off;
  ulp_stats(out_d[0], out_i8, mx, off);
  // An ulp count misleads both ways: near-cancelling sums (4 gated streams with signed (1 + w)
  // affines) show tens of ulps on tiny values, while one ulp of a value at 4x the row rms is
  // large in absolute terms. Bound each element by 2 bf16 ulps of max(|value|, row rms):
  // |dev - ref| <= 2^-7 * max(|ref|, rms).
  size_t beyond = 0;
  double worst_scaled = 0;
  for (int r = 0; r < rows; ++r) {
    double ss = 0;
    for (int i = 0; i < H; ++i) ss += std::pow(k::bf16_value(out_i8[static_cast<size_t>(r) * H + i]), 2);
    const double rms = std::sqrt(ss / H);
    for (int i = 0; i < H; ++i) {
      const size_t at = static_cast<size_t>(r) * H + i;
      const double y = k::bf16_value(out_i8[at]), d = k::bf16_value(out_d[0][at]);
      const double sc = std::fabs(d - y) / std::max(std::fabs(y), rms);
      worst_scaled = std::max(worst_scaled, sc);
      beyond += sc > 1.0 / 128;
    }
  }
  char line[256];
  std::snprintf(line, sizeof line,
                "final mixer vs host reference on the same int8 weights: rel err %.2e, worst |diff| / max(|ref|, rms) %.2e (2 ulps = 7.8e-03), "
                "%zu / %zu differ",
                rel(out_d[0], out_i8), worst_scaled, off, out_i8.size());
  check(rel(out_d[0], out_i8) <= 1e-3 && beyond == 0, line);
  check(out_d[0] == out_d[1], "final mixer: both cards bit-identical (replicated)");
  std::printf("       final mixer int8 vs bf16 weights (host references): rel err %.4f; device vs bf16 reference %.4f\n", rel(out_i8, out_bf),
              rel(out_d[0], out_bf));
  // Pick vs the device's own logits (exact), and both cards agree.
  bool pick_ok = true;
  for (int r = 0; r < rows; ++r) {
    std::vector<float> full(lg[0].begin() + static_cast<size_t>(r) * V, lg[0].begin() + static_cast<size_t>(r + 1) * V);
    full.insert(full.end(), lg[1].begin() + static_cast<size_t>(r) * V, lg[1].begin() + static_cast<size_t>(r + 1) * V);
    pick_ok &= tok[0][r] == k::pick_ref(full.data(), 2 * V, 0) && tok[1][r] == tok[0][r];
  }
  check(pick_ok, "greedy pick == host argmax over both cards' logits; both cards hold the same tokens");
  check(head.wire_error(0) == 0 && head.wire_error(1) == 0, "pick wire: no mailbox timeout");
  // lm_head logits vs the double reference on the same int8 rows (1,024 sampled rows per card),
  // input = the device's own mixer output, so this isolates the GEMM.
  const auto *lmw = raw.as<uint16_t>(tl);
  std::vector<float> x(static_cast<size_t>(rows) * H);
  for (size_t i = 0; i < x.size(); ++i) x[i] = k::bf16_value(out_d[0][i]);
  double worst = 0;
  std::mt19937 rng(9);
  for (int c = 0; c < 2; ++c)
    for (int sidx = 0; sidx < 1024; ++sidx) {
      const int col = static_cast<int>(rng() % V);
      const uint16_t *wr = lmw + (static_cast<size_t>(c) * V + col) * H;
      std::vector<int8_t> code(H);
      float sc;
      k::quantize_i8_rows(wr, 1, H, code.data(), &sc, 1);
      double yr[rows], mag = 0;
      k::dense_ref_i8(x.data(), code.data(), &sc, rows, 1, H, yr);
      for (int r = 0; r < rows; ++r) mag = std::max(mag, std::fabs(yr[r]));
      for (int r = 0; r < rows; ++r) worst = std::max(worst, std::fabs(lg[c][static_cast<size_t>(r) * V + col] - yr[r]) / std::max(mag, 1e-3));
    }
  std::snprintf(line, sizeof line, "lm_head logits vs double reference on the same int8 rows (2 x 1,024 sampled columns): worst rel err %.2e", worst);
  check(worst <= 1e-5, line);
  // Quantisation effect end to end: the whole path in bf16 (host, threads) vs the device tokens.
  std::vector<double> lbf(static_cast<size_t>(rows) * 2 * V);
  std::vector<float> xb(static_cast<size_t>(rows) * H);
  for (size_t i = 0; i < xb.size(); ++i) xb[i] = k::bf16_value(out_bf[i]);
  {
    std::vector<std::thread> th;
    const int T = std::max(1, o.threads);
    for (int t = 0; t < T; ++t)
      th.emplace_back([&, t] {
        for (int64_t col = t; col < 2LL * V; col += T) {
          const uint16_t *wr = lmw + static_cast<size_t>(col) * H;
          double acc[rows] = {0, 0, 0, 0};
          for (int i = 0; i < H; ++i) {
            const double w = k::bf16_value(wr[i]);
            for (int r = 0; r < rows; ++r) acc[r] += w * xb[static_cast<size_t>(r) * H + i];
          }
          for (int r = 0; r < rows; ++r) lbf[static_cast<size_t>(r) * 2 * V + col] = acc[r];
        }
      });
    for (auto &t : th) t.join();
  }
  int agree = 0;
  for (int r = 0; r < rows; ++r) {
    std::vector<float> row(lbf.begin() + static_cast<size_t>(r) * 2 * V, lbf.begin() + static_cast<size_t>(r + 1) * 2 * V);
    const int64_t tb = k::pick_ref(row.data(), 2 * V, 0);
    agree += tb == tok[0][r];
    const float dev_top = static_cast<float>(bf16(r < rows ? row[static_cast<size_t>(tok[0][r])] : 0.f));
    std::printf("       row %d id %6d -> device token %6d, all-bf16 reference token %6lld (bf16 logit %.3f vs %.3f at the device pick)\n", r, ids[r],
                tok[0][r], static_cast<long long>(tb), static_cast<float>(bf16(row[static_cast<size_t>(tb)])), dev_top);
  }
  std::printf("       int8 path vs all-bf16 reference: %d / %d rows pick the same token\n", agree, rows);

  // ---- timing: the whole chain per step (both cards), and each stage's kernel time on card 0 ----
  std::vector<double> wall, st[7];
  xenithium::probe::warm_for([&] {
    head.embed(rows);
    head.final_mix(rows);
    head.lm_pick(rows);
    head.wait();
  });
  for (int i = 0; i < o.iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    head.embed(rows);
    head.final_mix(rows);
    head.lm_pick(rows);
    head.wait();
    wall.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    const auto &e = head.events(0);
    const sycl::event *evs[7] = {&e.embed, &e.norm, &e.down, &e.mix, &e.lm, &e.chunks, &e.pick};
    for (int j = 0; j < 7; ++j) st[j].push_back(event_ms(*evs[j]) * 1e3);
  }
  const char *names[7] = {"embed gather (host table)", "HC combine+norm (ESIMD)", "HC down 320x10240 i8", "HC silu+up+gate mix",
                          "lm_head 124160x2560 i8", "pick chunk keys", "pick merge + wire"};
  const double bytes[7] = {0, 0, 320.0 * 10240 + 1280, 10240.0 * 320 + 40960, 124160.0 * 2560 + 496640, 4.0 * 124160 * 4, 0};
  std::printf("\nper-stage kernel time, card 0, %d iterations, both cards running the chain\n%-28s %10s %10s %10s\n", o.iters, "stage", "median us",
              "p90 us", "x floor");
  double sum = 0;
  for (int j = 0; j < 7; ++j) {
    const Stats s = stats(st[j]);
    sum += s.median;
    if (bytes[j] > 0) std::printf("%-28s %10.2f %10.2f %10.2f\n", names[j], s.median, s.p90, s.median / (bytes[j] / 604e3));
    else std::printf("%-28s %10.2f %10.2f %10s\n", names[j], s.median, s.p90, "-");
  }
  const Stats sw = stats(wall);
  std::printf("%-28s %10.2f\n%-28s %10.2f %10.2f   (host wall, submit + wait of both cards)\n", "sum of stages", sum, "whole chain", sw.median * 1e3,
              sw.p90 * 1e3);
  check(head.wire_error(0) == 0 && head.wire_error(1) == 0, "pick wire after timing: no mailbox timeout");
  return rc;
}

// ---- probe hc: every hyper-connection mixer of the model on real weights ----------------------
// The layers' blocks (attention / GDN, MoE, PLE) do not exist yet, so each block is the
// identity: block output = block input. That keeps every mixer's real data path - combine with
// the previous block's output and its injection logits, norm, merged down|inject GEMM, silu,
// up, gate mix - and the real weights, in the model's order: embed -> 48 x (attn mixer, mlp
// mixer) -> final mixer -> lm_head -> pick, on both cards. At the PLE layer the pending
// combine is materialised first (hc_combine) as the model does before adding PLE.
int cmd_hc(const Options &o) {
  namespace m = xenithium::model;
  namespace k = xenithium::kernels;
  namespace q38 = xenithium::arch::qwen38;
  using bf16 = sycl::ext::oneapi::bfloat16;
  if (o.model.empty()) {
    std::fprintf(stderr, "hc: --model DIR is required\n");
    return 2;
  }
  const auto mc = m::ModelConfig::load(o.model);
  const auto arch = m::open_arch(mc);
  const q38::Config cfg = q38::Config::parse(mc);
  const m::Checkpoint ck = m::Checkpoint::open(o.model);
  const std::string P = "model.language_model.";
  std::vector<m::DenseSpec> plan;
  for (const auto &sp : arch->dense_plan(ck, m::DeployParams{}))
    if (sp.comp == m::Component::hyper_connection || sp.name == P + "embed_tokens.weight" || sp.name == "lm_head.weight") plan.push_back(sp);
  const auto loader = xenithium::l0::Loader::open();
  xenithium::Fabric fab(loader, {0, 1});
  xenithium::load::DenseStore dense(fab);
  dense.load(ck, plan, o.threads);
  std::printf("hc: %zu tensors loaded in %.2f s; hyper-connections %.3f GiB per card (replicated)\n", plan.size(), dense.stats().total_s,
              dense.stats().card_bytes[0].count(m::Component::hyper_connection)
                  ? dense.stats().card_bytes[0].at(m::Component::hyper_connection) / 1073741824.0
                  : 0.0);
  q38::Head head(fab, dense, cfg);
  const q38::HcSet hs = q38::HcSet::from(dense, cfg, 2);
  const auto &sh = hs.shape;
  // --rows N (1..8): the decode rows per mixer (default 4 = the figure; it was kRows = 8 from to 71)
  const int rows = o.rows.find(',') == std::string::npos ? std::max(1, std::min(q38::Head::kRows, std::atoi(o.rows.c_str()))) : 4;
  const int H = sh.hidden, W = sh.width(), R = sh.lowrank, LD = hs.down_ld, L = cfg.layers;
  if (head.lora_ld() < LD) throw std::runtime_error("hc: head lora buffer too narrow");
  using Path = k::HcPath;
  bf16 *outA[2], *outB[2];
  for (int c = 0; c < 2; ++c) {
    outA[c] = static_cast<bf16 *>(fab.alloc(c, static_cast<size_t>(rows) * H * 2));
    outB[c] = static_cast<bf16 *>(fab.alloc(c, static_cast<size_t>(rows) * H * 2));
    fab.compute(c).memset(outA[c], 0, static_cast<size_t>(rows) * H * 2);
    fab.compute(c).memset(outB[c], 0, static_cast<size_t>(rows) * H * 2).wait();
  }
  const int32_t ids[q38::Head::kRows] = {11, 1000, 50000, 248043, 7, 4242, 123456, 99};
  head.set_ids(ids, rows);

  // One step. hook(id, after) runs on the host around card 0's mixer `id` (2l attn, 2l+1 mlp,
  // 2L final) when given; ev (card 0) gets every mixer's events.
  using Hook = std::function<void(int, bool)>;
  auto step = [&](Path path, int ks, std::vector<k::HcEvents> *ev, const Hook &hook, int uks = 0) {
    if (ev) ev->clear();
    head.embed(rows);
    for (int l = 0; l < L; ++l) {
      for (int part = 0; part < 2; ++part) {
        const int id = 2 * l + part;
        if (hook) hook(id, false);
        for (int c = 0; c < 2; ++c) {
          auto &q = fab.compute(c);
          auto &st = head.state(c);
          const bf16 *blk = part == 0 ? (l > 0 ? outB[c] : nullptr) : outA[c];
          if (part == 0 && l == hs.ple_layer && blk) {  // materialise before the (future) PLE add
            k::hc_combine(q, sh, rows, st, blk);
            blk = nullptr;
          }
          const auto e = k::hc_mixer(q, sh, rows, st, blk, part == 0 ? hs.attn[c][l] : hs.mlp[c][l], head.scratch(c),
                                     part == 0 ? outA[c] : outB[c], path, ks, uks);
          if (c == 0 && ev) ev->push_back(e);
        }
        if (hook) hook(id, true);
      }
    }
    if (hook) hook(2 * L, false);
    head.final_mix(rows, {outB[0], outB[1]}, path);
    if (ev) ev->push_back({head.events(0).norm, head.events(0).down, head.events(0).mix});
    if (hook) hook(2 * L, true);
    head.lm_pick(rows);
  };

  int rc = 0;
  auto check = [&](bool ok, const std::string &what) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    rc |= !ok;
  };
  // ---- teacher-forced check of selected mixers on card 0 against the host reference ----
  std::set<int> checked;
  for (int l : {0, hs.ple_layer, hs.ple_layer + 1, L / 2, L - 1}) {
    checked.insert(2 * l);
    checked.insert(2 * l + 1);
  }
  checked.insert(2 * L);
  auto name_of = [&](int id) {
    if (id == 2 * L) return P + "hyper_connection_mixer.";
    return P + "layers." + std::to_string(id / 2) + (id % 2 ? ".mlp_hyper_connection." : ".attn_hyper_connection.");
  };
  std::vector<const m::TensorInfo *> want;
  for (int id : checked) {
    for (const char *t : {"hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight"}) want.push_back(&ck.at(name_of(id) + t));
    if (id != 2 * L) want.push_back(&ck.at(name_of(id) + "block_inject_weight.weight"));
  }
  const auto raw = m::bulk_read(ck, want);
  auto &q0 = fab.compute(0);
  auto rd16 = [&](const void *p, size_t n) {
    std::vector<uint16_t> v(n);
    q0.memcpy(v.data(), p, n * 2).wait();
    return v;
  };
  auto rdf = [&](const float *p, size_t n) {
    std::vector<float> v(n);
    q0.memcpy(v.data(), p, n * 4).wait();
    return v;
  };
  struct Snap {
    std::vector<uint16_t> res, blk;
    std::vector<float> inj;
    bool comb = false;
  } before;
  double worst_out = 0, worst_res = 0;
  int worst_inj_ulp = 0, n_checked = 0;
  size_t off_out = 0, n_out = 0;
  auto scaled = [&](const std::vector<uint16_t> &dv, const std::vector<uint16_t> &rv, int row, size_t &off) {
    double w = 0;
    for (size_t r0 = 0; r0 < rv.size(); r0 += row) {
      double ss = 0;
      for (int i = 0; i < row; ++i) ss += std::pow(k::bf16_value(rv[r0 + i]), 2);
      const double rms = std::sqrt(ss / row);
      for (int i = 0; i < row; ++i) {
        const double y = k::bf16_value(rv[r0 + i]), d = k::bf16_value(dv[r0 + i]);
        w = std::max(w, std::fabs(d - y) / std::max({std::fabs(y), rms, 1e-30}));
        off += dv[r0 + i] != rv[r0 + i];
      }
    }
    return w;
  };
  Hook hook = [&](int id, bool after) {
    if (!checked.count(id)) return;
    const bool final_m = id == 2 * L;
    const int l = id / 2, part = id % 2;
    // What this mixer consumes: the pending block (previous mixer's output) and its injection.
    const bool comb = final_m || part == 1 || l > 0;
    const bf16 *blk = final_m ? outB[0] : part == 1 ? outA[0] : outB[0];
    if (!after) {
      fab.compute(0).wait();
      fab.compute(1).wait();
      before.res = rd16(head.residual(0), static_cast<size_t>(rows) * W);
      before.comb = comb;
      if (comb) {
        before.blk = rd16(blk, static_cast<size_t>(rows) * H);
        const auto lora = rdf(head.state(0).down_out(), static_cast<size_t>(rows) * LD);
        before.inj.assign(static_cast<size_t>(rows) * sh.streams, 0.f);
        for (int r = 0; r < rows; ++r)
          for (int s = 0; s < sh.streams; ++s) before.inj[r * sh.streams + s] = lora[static_cast<size_t>(r) * LD + R + s];
      }
      return;
    }
    fab.compute(0).wait();
    fab.compute(1).wait();
    const auto res_d = rd16(head.residual(0), static_cast<size_t>(rows) * W);
    const auto out_d = rd16(final_m ? head.card(0).out : part == 0 ? outA[0] : outB[0], static_cast<size_t>(rows) * H);
    const int dn = final_m ? R : LD;
    const auto lora_d = rdf(head.state(0).down_out(), static_cast<size_t>(rows) * dn);
    // Reference weights: int8 rows exactly as the store quantised them.
    const std::string nm = name_of(id);
    const auto &tn = ck.at(nm + "hc_norm.weight"), &td = ck.at(nm + "input_mix_weight_down.weight"), &tu = ck.at(nm + "input_mix_weight_up.weight");
    const int RI = final_m ? R : R + sh.streams;
    std::vector<uint16_t> dsrc(static_cast<size_t>(RI) * W);
    std::memcpy(dsrc.data(), raw.as<uint16_t>(td), static_cast<size_t>(R) * W * 2);
    if (!final_m) std::memcpy(dsrc.data() + static_cast<size_t>(R) * W, raw.as<uint16_t>(ck.at(nm + "block_inject_weight.weight")), static_cast<size_t>(sh.streams) * W * 2);
    std::vector<int8_t> dc(dsrc.size()), uc(static_cast<size_t>(W) * R);
    std::vector<float> dsc(RI), usc(W);
    k::quantize_i8_rows(dsrc.data(), RI, W, dc.data(), dsc.data(), 1);
    k::quantize_i8_rows(raw.as<uint16_t>(tu), W, R, uc.data(), usc.data(), 1);
    const std::vector<float> dv(dc.begin(), dc.end()), uv(uc.begin(), uc.end());
    std::vector<uint16_t> res_r = before.res, out_r(static_cast<size_t>(rows) * H);
    std::vector<float> lora_r(static_cast<size_t>(rows) * RI);
    k::hc_mix_ref(sh, rows, res_r.data(), before.comb ? before.blk.data() : nullptr, before.comb ? before.inj.data() : nullptr, raw.as<uint16_t>(tn),
                  {dv.data(), dsc.data(), RI, W}, {uv.data(), usc.data(), W, R}, nullptr, lora_r.data(), out_r.data());
    size_t off = 0, off_r = 0;
    const double wo = scaled(out_d, out_r, H, off), wr = scaled(res_d, res_r, H, off_r);
    int inj_ulp = 0;
    if (!final_m)
      for (int r = 0; r < rows; ++r)
        for (int s = 0; s < sh.streams; ++s) {
          const uint16_t a = k::bf16_bits(lora_d[static_cast<size_t>(r) * LD + R + s]), b = k::bf16_bits(lora_r[static_cast<size_t>(r) * RI + R + s]);
          auto lin = [](uint16_t v) { return (v & 0x8000) ? -static_cast<int>(v & 0x7fff) : static_cast<int>(v); };
          inj_ulp = std::max(inj_ulp, std::abs(lin(a) - lin(b)));
        }
    std::printf("       mixer %3d %-44s out worst %.2e (%zu/%zu differ)  res worst %.2e  inj %d ulp\n", id,
                (final_m ? std::string("final") : nm.substr(P.size())).c_str(), wo, off, out_r.size(), wr, inj_ulp);
    worst_out = std::max(worst_out, wo);
    worst_res = std::max(worst_res, wr);
    worst_inj_ulp = std::max(worst_inj_ulp, inj_ulp);
    off_out += off;
    n_out += out_r.size();
    ++n_checked;
  };
  step(Path::three, 0, nullptr, hook);
  head.wait();
  char line[256];
  std::snprintf(line, sizeof line,
                "%d mixers (teacher-forced, real int8 weights) vs host reference: out worst %.2e, residual worst %.2e of max(|ref|, rms) "
                "(bound 7.8e-03), %.3f %% of outputs differ, injection logits within %d ulp",
                n_checked, worst_out, worst_res, 100.0 * off_out / std::max<size_t>(1, n_out), worst_inj_ulp);
  check(worst_out <= 1.0 / 128 && worst_res <= 1.0 / 128 && off_out * 100 <= n_out && worst_inj_ulp <= 2, line);
  // Replication: both cards end the step with the same bits.
  std::vector<uint16_t> fin[2], fres[2];
  std::vector<int32_t> tk[2];
  auto read_card = [&](int c) {
    std::vector<uint16_t> a(static_cast<size_t>(rows) * H), b(static_cast<size_t>(rows) * W);
    fab.compute(c).memcpy(a.data(), head.card(c).out, a.size() * 2);
    fab.compute(c).memcpy(b.data(), head.residual(c), b.size() * 2).wait();
    fin[c] = a;
    fres[c] = b;
    tk[c] = head.tokens(c, rows);
  };
  read_card(0);
  read_card(1);
  check(fin[0] == fin[1] && fres[0] == fres[1] && tk[0] == tk[1], "after 97 mixers both cards hold bit-identical residuals, outputs and tokens");
  check(head.wire_error(0) == 0 && head.wire_error(1) == 0, "pick wire: no mailbox timeout");
  // Fused vs unfused over the whole chain (free-running, so differences compound).
  const auto fused_out = fin[0];
  const auto fused_tok = tk[0];
  for (const Path pth : {Path::unfused}) {
    step(pth, 0, nullptr, nullptr);
    head.wait();
    read_card(0);
    size_t offu = 0;
    const double wu = scaled(fused_out, fin[0], H, offu);
    std::printf("       whole chain, fused vs %s (free-running 97 mixers): final mixer out worst %.2e of max(|ref|, rms), %zu / %zu differ; tokens %s\n",
                pth == Path::unfused ? "unfused" : "?", wu, offu, fin[0].size(), fused_tok == tk[0] ? "equal" : "DIFFER");
  }
  {
    bool det = true;
    for (int i = 0; i < 3; ++i) {
      step(Path::three, 0, nullptr, nullptr);
      head.wait();
      read_card(0);
      det &= fin[0] == fused_out;
    }
    check(det, "fused chain run-to-run bit-identical (3 more steps)");
  }

  // ---- timing ----
  const double layer_bytes = static_cast<double>(hs.attn[0][0].down.bytes() + hs.attn[0][0].up.bytes()) + 2.0 * W;
  const double final_bytes = static_cast<double>(hs.final_mixer[0].down.bytes() + hs.final_mixer[0].up.bytes()) + 2.0 * W;
  const double floor_us = (2.0 * L * layer_bytes + final_bytes) / 604e3;
  std::printf("\nmixer weight floor: layer mixer %.2f MB -> %.2f us, 97 mixers %.1f us per step (int8, 604 GB/s)\n", layer_bytes / 1e6,
              layer_bytes / 604e3, floor_us);
  std::printf("%-30s %9s %9s %9s %8s %8s %9s %7s\n", "variant (both cards)", "wall med", "wall p90", "GPU span", "norm us", "down us",
              "upmix us", "x floor");
  struct V {
    const char *name;
    Path path;
    int ks, uks;
  };
  for (const V v : {V{"3 launches, up ks 1 (default)", Path::three, 0, 0}, V{"3 launches, up ks 2", Path::three, 0, 2},
                    V{"3 launches, up ks 4", Path::three, 0, 4}, V{"5 launches (unfused)", Path::unfused, 0, 0},
                    V{"3 launches, down ks 8", Path::three, 8, 0}, V{"3 launches, down ks 32", Path::three, 32, 0},
                    V{"3 launches, up ks 1 (again)", Path::three, 0, 0}}) {
    std::vector<k::HcEvents> ev;
    xenithium::probe::warm_for([&] {
      step(v.path, v.ks, nullptr, nullptr, v.uks);
      head.wait();
    });
    std::vector<double> wall, tn, td, tu, span, g1, g2, g3;  // gaps: norm -> down, down -> up, up -> next norm (us)
    for (int i = 0; i < o.iters; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      step(v.path, v.ks, &ev, nullptr, v.uks);
      head.wait();
      wall.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
      span.push_back(static_cast<double>(head.events(0).pick.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                         head.events(0).embed.get_profiling_info<sycl::info::event_profiling::command_start>()) /
                     1e3);
      double a = 0, b = 0, c = 0;
      for (size_t j = 0; j + 1 < ev.size(); ++j) {  // layer mixers only (the last is the final mixer)
        a += event_ms(ev[j].norm) * 1e3;
        b += event_ms(ev[j].down) * 1e3;
        c += event_ms(ev[j].up) * 1e3;
      }
      const double n = static_cast<double>(ev.size() - 1);
      auto t_s = [](const sycl::event &e) { return static_cast<double>(e.get_profiling_info<sycl::info::event_profiling::command_start>()); };
      auto t_e = [](const sycl::event &e) { return static_cast<double>(e.get_profiling_info<sycl::info::event_profiling::command_end>()); };
      double x1 = 0, x2 = 0, x3 = 0;
      for (size_t j = 0; j + 1 < ev.size(); ++j) {
        x1 += t_s(ev[j].down) - t_e(ev[j].norm);
        x2 += t_s(ev[j].up) - t_e(ev[j].down);
        if (j + 2 < ev.size()) x3 += t_s(ev[j + 1].norm) - t_e(ev[j].up);
      }
      g1.push_back(x1 / n / 1e3);
      g2.push_back(x2 / n / 1e3);
      g3.push_back(x3 / (n - 1) / 1e3);
      tn.push_back(a / n);
      td.push_back(b / n);
      tu.push_back(c / n);
    }
    const Stats w = stats(wall), a = stats(tn), b = stats(td), c = stats(tu), sp = stats(span);
    std::printf("%-30s %9.1f %9.1f %9.1f %8.2f %8.2f %9.2f %7.2f   gaps %5.2f %5.2f %5.2f\n", v.name, w.median, w.p90, sp.median, a.median,
                b.median, c.median, (a.median + b.median + c.median) / (layer_bytes / 604e3), stats(g1).median, stats(g2).median, stats(g3).median);
  }
  std::printf("(wall = embed + 97 mixers + lm_head + pick, host submit + wait, us; GPU span = card 0 embed start -> pick end;\n"
              " per-launch columns = card 0 kernel time per layer mixer; x floor = their sum / the mixer's weight floor;\n"
              " unfused: the upmix column is its gate mix only; gaps = card 0 median us norm -> down, down -> up, up -> next mixer's norm\n"
              " (the next mixer's combine waits for nothing else here: identity blocks))\n");
  check(head.wire_error(0) == 0 && head.wire_error(1) == 0, "pick wire after timing: no mailbox timeout");
  for (int c = 0; c < 2; ++c) {
    fab.release(outA[c]);
    fab.release(outB[c]);
  }
  return rc;
}


// The MoE block on every layer with the loader's real placement (hot experts in VRAM, cold ones
// read in place from pinned host over PCIe): 48 layers chained like a step, on a synthetic input
// (routing is not the model's; this checks the placement path, the tables and the wire).
int moe_on_store(xenithium::Fabric &fab, const xenithium::load::DenseStore &dense, const xenithium::arch::qwen38::ExpertStore &store,
                 const xenithium::arch::qwen38::Config &cfg, int iters) {
  namespace k = xenithium::kernels;
  namespace q38 = xenithium::arch::qwen38;
  using bf16 = sycl::ext::oneapi::bfloat16;
  const q38::MoeSet ms = q38::MoeSet::from(dense, cfg, 2);
  const int L = cfg.layers, per = store.per_card(), H = cfg.hidden, rows = 4;
  int rc = 0;
  const uint8_t **tab[2];
  k::MoeScratch sc[2];
  k::MoeWire wire[2];
  bf16 *x[2], *out[2];
  k::MoeRoute *rec[2];
  std::vector<uint16_t> xh(static_cast<size_t>(rows) * H);
  {
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto &v : xh) v = k::bf16_bits(nd(rng));
  }
  for (int c = 0; c < 2; ++c) {
    const auto s = ms.shape_of(c);
    std::vector<const uint8_t *> t(static_cast<size_t>(L) * per);
    for (int l = 0; l < L; ++l)
      for (int i = 0; i < per; ++i) t[static_cast<size_t>(l) * per + i] = store.blob(l, c * per + i);
    tab[c] = static_cast<const uint8_t **>(fab.alloc(c, t.size() * 8));
    fab.compute(c).memcpy(tab[c], t.data(), t.size() * 8).wait();
    sc[c].logits = static_cast<float *>(fab.alloc(c, 4ull * ms.w[c][0].router.n * 4));
    sc[c].route = static_cast<k::MoeRoute *>(fab.alloc(c, sizeof(k::MoeRoute)));
    sc[c].h = static_cast<bf16 *>(fab.alloc(c, k::moe_h_elems(s) * 2));
    sc[c].partial = static_cast<float *>(fab.alloc(c, k::moe_partial_elems(s) * 4));
    wire[c].inbox = static_cast<uint32_t *>(fab.alloc_mailbox(c, k::moe_wire_words(s) * 4));
    wire[c].error = static_cast<uint32_t *>(fab.alloc(c, 4096));
    x[c] = static_cast<bf16 *>(fab.alloc(c, xh.size() * 2));
    out[c] = static_cast<bf16 *>(fab.alloc(c, static_cast<size_t>(L) * rows * H * 2));
    rec[c] = static_cast<k::MoeRoute *>(fab.alloc(c, static_cast<size_t>(L) * sizeof(k::MoeRoute)));
    fab.compute(c).memcpy(x[c], xh.data(), xh.size() * 2).wait();
  }
  for (int c = 0; c < 2; ++c) {
    wire[c].peer_inbox = wire[1 - c].inbox;
    k::moe_wire_init(fab.compute(c), ms.shape_of(c), wire[c].inbox).wait();
    fab.compute(c).memset(wire[c].error, 0, 4).wait();
  }
  int parity = 0;
  auto step = [&](bool keep, std::vector<sycl::event> *first_last) {
    for (int l = 0; l < L; ++l) {
      for (int c = 0; c < 2; ++c) {
        k::MoeWeights w = ms.w[c][l];
        w.blobs = tab[c] + static_cast<size_t>(l) * per;
        const auto e = k::moe_decode(fab.compute(c), ms.shape_of(c), rows, x[c], w, sc[c], &wire[c], parity, out[c] + static_cast<size_t>(l) * rows * H);
        if (keep) fab.compute(c).memcpy(rec[c] + l, sc[c].route, sizeof(k::MoeRoute));
        if (first_last && c == 0) {
          if (l == 0) (*first_last)[0] = e.router;
          if (l == L - 1) (*first_last)[1] = e.combine;
        }
      }
      parity ^= 1;
    }
  };
  step(true, nullptr);
  for (int c = 0; c < 2; ++c) fab.compute(c).wait_and_throw();
  std::vector<uint16_t> o[2];
  std::vector<k::MoeRoute> r[2];
  for (int c = 0; c < 2; ++c) {
    o[c].resize(static_cast<size_t>(L) * rows * H);
    r[c].resize(L);
    fab.compute(c).memcpy(o[c].data(), out[c], o[c].size() * 2);
    fab.compute(c).memcpy(r[c].data(), rec[c], L * sizeof(k::MoeRoute)).wait();
  }
  int64_t routes = 0, cold = 0, uniq[2] = {0, 0};
  bool ids_same = true, finite = true;
  for (int l = 0; l < L; ++l) {
    for (int c = 0; c < 2; ++c) {
      uniq[c] += r[c][l].count;
      for (int u = 0; u < r[c][l].count; ++u) {
        bool hot = false;
        store.blob(l, r[c][l].expert[u], &hot);
        routes += __builtin_popcount(r[c][l].mask[u]);
        cold += hot ? 0 : __builtin_popcount(r[c][l].mask[u]);
      }
    }
    for (int rr = 0; rr < rows; ++rr)
      for (int j = 0; j < cfg.topk; ++j) ids_same &= r[0][l].ids[rr][j] == r[1][l].ids[rr][j];
  }
  for (uint16_t b : o[0]) finite &= (b & 0x7f80) != 0x7f80;
  std::printf("[%s] moe x %d layers on the loaded placement (synthetic input, %d rows): both cards bit-identical outputs and routing, all finite; "
              "%lld routes, %lld cold (%.1f %%, read in place over PCIe), unique experts per layer card 0 %.1f card 1 %.1f\n",
              o[0] == o[1] && ids_same && finite ? "PASS" : "FAIL", L, rows, static_cast<long long>(routes), static_cast<long long>(cold),
              100.0 * cold / std::max<int64_t>(1, routes), uniq[0] / double(L), uniq[1] / double(L));
  rc |= !(o[0] == o[1] && ids_same && finite);
  std::vector<double> span;
  for (int rep = 0; rep < std::max(3, iters / 10) + 2; ++rep) {
    std::vector<sycl::event> fl(2);
    step(false, &fl);
    for (int c = 0; c < 2; ++c) fab.compute(c).wait();
    if (rep < 2) continue;
    span.push_back(static_cast<double>(fl[1].get_profiling_info<sycl::info::event_profiling::command_end>() -
                                       fl[0].get_profiling_info<sycl::info::event_profiling::command_start>()) *
                   1e-6);
  }
  std::sort(span.begin(), span.end());
  uint32_t err[2];
  for (int c = 0; c < 2; ++c) fab.compute(c).memcpy(&err[c], wire[c].error, 4).wait();
  std::printf("moe x %d layers chained: GPU span card 0 median %.2f ms per step (synthetic routing, this placement's cold share)\n", L, span[span.size() / 2]);
  std::printf("[%s] moe mailbox: no timeout\n", err[0] || err[1] ? "FAIL" : "PASS");
  rc |= err[0] || err[1];
  for (int c = 0; c < 2; ++c)
    for (void *p : {(void *)tab[c], (void *)sc[c].logits, (void *)sc[c].route, (void *)sc[c].h, (void *)sc[c].partial, (void *)wire[c].inbox,
                    (void *)wire[c].error, (void *)x[c], (void *)out[c], (void *)rec[c]})
      fab.release(p);
  return rc;
}

int cmd_store(const Options &o) {
  namespace m = xenithium::model;
  namespace q38 = xenithium::arch::qwen38;
  if (o.model.empty()) {
    std::fprintf(stderr, "store: --model DIR is required\n");
    return 2;
  }
  const auto mc = m::ModelConfig::load(o.model);
  const auto arch = m::open_arch(mc);
  const q38::Config cfg = q38::Config::parse(mc);
  const m::Checkpoint ck = m::Checkpoint::open(o.model);
  m::DeployParams p;
  p.context = o.context;
  p.seqs = o.seqs;
  p.kv_bytes = o.kv_bytes;
  p.drafter = o.drafter != 0;
  const auto loader = xenithium::l0::Loader::open();
  // Placement from the live free VRAM of the tighter card (sysman), not a constant.
  double free_gib = 1e30;
  for (size_t c = 0; c < 2; ++c) {
    const auto v = xenithium::l0::vram_state(loader.info(c));
    if (v.ok) free_gib = std::min(free_gib, v.free / 1073741824.0);
  }
  if (free_gib < 1e29) p.vram_gib = free_gib;
  const m::Budget b = arch->budget(ck, p);
  int64_t hot = o.hot >= 0 ? o.hot : b.hot_experts_fit;

  auto vram_line = [&](const char *when) {
    for (size_t c = 0; c < loader.size(); ++c) {
      const auto v = xenithium::l0::vram_state(loader.info(c));
      if (v.ok) std::printf("vram %s: card %zu %.2f GiB free of %.2f GiB\n", when, c, v.free / 1073741824.0, v.size / 1073741824.0);
      else std::printf("vram %s: card %zu sysman unavailable\n", when, c);
    }
  };
  vram_line("before");
  xenithium::Fabric fab(loader, {0, 1});
  std::unique_ptr<xenithium::load::DenseStore> dense;
  std::vector<m::DenseSpec> plan;
  if (o.dense) {
    plan = arch->dense_plan(ck, p);
    dense = std::make_unique<xenithium::load::DenseStore>(fab);
    dense->load(ck, plan, o.threads);
    const auto &d = dense->stats();
    std::printf("dense: %zu tensors in %.2f s (drive wait %.2f, slice+quantise+pack %.2f, H2D wait %.2f); read %.2f GiB; host pinned %.2f GiB\n",
                plan.size(), d.total_s, d.read_wait_s, d.pack_s, d.h2d_wait_s, d.read_bytes / 1073741824.0, d.host_bytes / 1073741824.0);
    std::printf("%-22s %14s %14s\n", "component", "card 0 GiB", "card 1 GiB");
    for (const auto &kv : d.card_bytes[0])
      std::printf("%-22s %14.3f %14.3f\n", std::string(m::component_name(kv.first)).c_str(), kv.second / 1073741824.0,
                  d.card_bytes[1].at(kv.first) / 1073741824.0);
    std::printf("%-22s %14.3f %14.3f\n", "allocated (aligned)", d.card_alloc[0] / 1073741824.0, d.card_alloc[1] / 1073741824.0);
    vram_line("after dense");
    if (o.hot < 0) {
      // Experts get what is really free now, minus the KV at the asked context and the reserve.
      double free_now = 1e30;
      for (size_t c = 0; c < 2; ++c) {
        const auto v = xenithium::l0::vram_state(loader.info(c));
        if (v.ok) free_now = std::min(free_now, static_cast<double>(v.free));
      }
      const double kv = b.kv_bytes_per_token * static_cast<double>(p.context) * p.seqs;
      const double room = free_now - kv - p.reserve_gib * 1073741824.0;
      hot = std::max<int64_t>(0, std::min<int64_t>(b.experts_per_card, static_cast<int64_t>(room / q38::ExpertLayout::of(cfg).stride)));
      std::printf("placement (measured): %.3f GiB free - KV %.3f GiB - reserve %.2f GiB -> %lld hot experts per card (budget said %lld)\n",
                  free_now / 1073741824.0, kv / 1073741824.0, p.reserve_gib, static_cast<long long>(hot),
                  static_cast<long long>(b.hot_experts_fit));
    }
  }
  q38::ExpertStore store(fab, cfg, o.moe_tp);
  q38::ExpertStoreOptions so;
  so.hot_per_card = hot;
  so.pack_threads = o.threads;
  so.hash = o.verify != 0;
  if (o.moe_tp && std::filesystem::exists(o.model + "/native/experts-L00.tp2")) so.native_dir = o.model + "/native";
  std::printf("experts: %s, %s\n", o.moe_tp ? "TP (1/2 of every expert per card)" : "EP", so.native_dir.empty() ? "packed at load" : "prepacked (native/)");
  std::printf("placement: %.2f GiB free (live), reserve %.2f GiB, context %lld x %d slot(s), KV %s%s -> budget: %.3f GiB non-expert per card, %lld of %lld experts hot per card%s\n",
              p.vram_gib, p.reserve_gib, static_cast<long long>(p.context), p.seqs, p.kv_bytes == 1 ? "8-bit" : "bf16", p.drafter ? " + drafter" : "",
              b.resident_non_expert / 1073741824.0, static_cast<long long>(hot), static_cast<long long>(b.experts_per_card),
              o.hot >= 0 ? " (--hot)" : "");
  store.load(ck, so);
  const auto &s = store.stats();
  std::printf("%-40s %10s\n", "load phase", "s");
  std::printf("%-40s %10.2f\n", "total (wall)", s.total_s);
  std::printf("%-40s %10.2f\n", "  allocations (VRAM + pinned host)", s.alloc_s);
  std::printf("%-40s %10.2f\n", "  waiting for the drive (reader)", s.read_wait_s);
  std::printf("%-40s %10.2f\n", "    of which layer 0 (not overlapped)", s.first_read_s);
  std::printf("%-40s %10.2f\n", "  packing (16 threads)", s.pack_s);
  std::printf("%-40s %10.2f\n", "  waiting for H2D", s.h2d_wait_s);
  std::printf("%-40s %10.2f\n", "  freeing read buffers", s.free_s);
  std::printf("drive: %.2f GiB in %.2f s = %.2f GB/s end to end (O_DIRECT floor ~5.3 GB/s)\n", s.read_bytes / 1073741824.0,
              s.total_s, s.read_bytes / s.total_s / 1e9);
  std::printf("H2D: %.2f GiB to the cards; pinned host held: %.2f GiB (cold experts)\n", s.h2d_bytes / 1073741824.0,
              s.pinned_bytes / 1073741824.0);
  for (int c = 0; c < store.cards(); ++c)
    std::printf("card %d: %lld hot experts (%.2f GiB VRAM), %lld cold; per layer %d..%d hot of %d\n", c,
                static_cast<long long>(store.hot_total(c)), store.hot_total(c) * double(store.layout().stride) / 1073741824.0,
                static_cast<long long>(int64_t(cfg.layers) * store.per_card() - store.hot_total(c)), store.hot_count(c, cfg.layers - 1),
                store.hot_count(c, 0), store.per_card());
  vram_line("after load");
  std::vector<void *> extra;
  if (o.fill_free > 0) {
    // Is the sysman figure real? Take everything but fill_free MiB in 256 MiB pieces, write
    // every byte on the copy engine, then verify the experts (which must be untouched).
    constexpr size_t kPiece = 256ull << 20;
    for (size_t c = 0; c < 2; ++c) {
      size_t got = 0;
      for (;;) {
        const auto v = xenithium::l0::vram_state(loader.info(c));
        if (!v.ok || v.free < kPiece + (static_cast<uint64_t>(o.fill_free) << 20)) break;
        void *p = fab.alloc(c, kPiece);
        fab.copy(c).fill(p, 0x5A, kPiece, true);
        extra.push_back(p);
        got += kPiece;
      }
      std::printf("card %zu: allocated and filled %.2f GiB more\n", c, got / 1073741824.0);
    }
    vram_line("after fill");
  }
  int rc = 0;
  if (o.verify) {
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t bad = store.verify();
    std::printf("[%s] verify: %lld of %lld blobs differ from their pack-time hash (%.1f s)\n", bad ? "FAIL" : "PASS",
                static_cast<long long>(bad), static_cast<long long>(int64_t(cfg.layers) * cfg.experts), since_ms(t0) / 1e3);
    rc |= bad != 0;
  }
  // Spot checks: a hot and a cold expert on each card, at the first and last layer.
  for (int c = 0; c < 2; ++c)
    for (const int layer : {0, cfg.layers - 1}) {
      const int first = c * store.per_card(), h = store.hot_count(c, layer);
      for (const int e : {first, first + store.per_card() - 1}) {
        bool is_hot = false;
        const uint8_t *blob = store.blob(layer, e, &is_hot);
        if (!is_hot && e == first && h > 0) continue;
        const double rel = check_expert(fab.compute(c), fab, c, ck, cfg, store.layout(), layer, e, blob);
        const bool ok = rel <= 1e-5;
        rc |= !ok;
        std::printf("[%s] card %d layer %2d expert %3d (%s): W4A16 vs raw-tensor reference, rel err %.2e\n", ok ? "PASS" : "FAIL", c, layer,
                    e, is_hot ? "hot, VRAM" : "cold, pinned host over PCIe", rel);
      }
    }
  if (dense) {
    // Every split kind on both cards: rows (+sections, padded rows), cols, replicate, bf16, k tail.
    const std::vector<std::string> names = {"lm_head.weight",
                                            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                                            "model.language_model.layers.3.self_attn.o_proj.weight",
                                            "model.language_model.layers.3.self_attn.q_proj.weight",
                                            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
                                            "model.language_model.layers.0.mlp.gate.weight",  // + shared gate row
                                            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",  // + up rows
                                            "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                                            "model.language_model.layers.5.attn_hyper_connection.input_mix_weight_down.weight"};  // + inject rows
    for (const auto &nm : names) {
      const auto it = std::find_if(plan.begin(), plan.end(), [&](const m::DenseSpec &s) { return s.name == nm; });
      for (int c = 0; c < 2; ++c) {
        const auto [rel, quant] = check_dense(fab, *dense, ck, *it, c);
        const bool ok = rel <= 1e-5;
        rc |= !ok;
        const char *split = it->split == m::DenseSpec::Split::rows ? "rows" : it->split == m::DenseSpec::Split::cols ? "cols" : "repl";
        std::printf("[%s] card %d %-4s %-4s %-70s rel err %.2e (quantisation vs bf16: %.4f)\n", ok ? "PASS" : "FAIL", c, split,
                    it->store == m::DenseSpec::Store::i8 ? "i8" : "bf16", nm.c_str(), rel, quant);
      }
    }
    // Raw slices: GDN A_log, in_proj_a (rows) and the conv taps (rows by sections) read back byte-equal.
    for (const std::string nm : {"model.language_model.layers.0.linear_attn.A_log", "model.language_model.layers.0.linear_attn.in_proj_a.weight",
                                 "model.language_model.layers.0.linear_attn.conv1d.weight"}) {
      const auto &t = ck.at(nm);
      const auto buf = m::bulk_read(ck, {&t});
      const auto it = std::find_if(plan.begin(), plan.end(), [&](const m::DenseSpec &s) { return s.name == nm; });
      for (int c = 0; c < 2; ++c) {
        const auto &dt = dense->at(c, nm);
        std::vector<uint8_t> back(dt.bytes);
        fab.copy(c).copy(back.data(), dt.ptr, back.size(), true);
        std::vector<uint8_t> want;
        const int64_t rowb = t.bytes / t.dim(0);
        for (const auto &rg : xenithium::load::slice_rows(t.dim(0), it->sections, c, 2))
          want.insert(want.end(), buf.data(t) + rg.first * rowb, buf.data(t) + rg.second * rowb);
        const bool ok = back == want;
        rc |= !ok;
        std::printf("[%s] card %d raw slice of %s byte-equal (%zu B)\n", ok ? "PASS" : "FAIL", c, nm.c_str(), back.size());
      }
    }
  }
  if (dense) rc |= moe_on_store(fab, *dense, store, cfg, o.iters);
  for (void *p : extra) fab.release(p);
  return rc;
}

}  // namespace



int main(int argc, char **argv) {
  if (!xenithium::xrv::install_probe_qsa_capture_from_env()) return 2;
  if (argc >= 2 && std::string(argv[1]) == "rank") {  // a hot-expert ranking for a MoE checkpoint (rank.cpp)
    try {
      return cmd_rank(argc - 2, argv + 2);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
  }
  if (argc < 2) {
    std::fprintf(stderr, "usage: xenithium-probe devices|bw|w4|dense|pf|load|store|head|hc|moe|gdn|qsa|ple|model|mtp|seqs|serve [--model DIR] [--mtp DIR] [--prompt-len N] [--depth D] [--drafts-ref F] [--mtp-kv-bytes 1|2] [--ppl FILE] [--rows 1,4] [--generate N] [--layers L,L] [--context N] [--seqs N] [--kv-bytes 1|2] [--drafter 0|1] [--hot N] [--verify 0|1] [--routes-out F] [--hot-rank F] [--reserve GIB] [--lp-out DIR] [--lp-ref F,F] [--bf16 PAT,PAT] [--moe-down-bf16 0|1] [--moe-dn-ks N] [--graphs 0|1] [--moe-tp 0|1] [--hidden-out DIR] [--gen-out F] [--pf-chunk N] [--pf-moe-a8 0|1] [--pf-dense-a8 0|1|2] [--pf-wire fp32|bf16] [--dec-moe-a8 0|1] [--kv-pool N] [--kv-scramble 0|1] [--kv-lend 0|1] [--kv-lend-keep N] [--kv-lend-idle 0|1] [--moe-gu-ks N] [--gdn-scan 0|1] [--pf-cold-stage N] [--pf-route-trace F] [--pf-time-only 0|1] [--pf-sweep N,N] [--lp-time N] [--step-rows 4|8] [--two-deep 0|1] [--spec-cap F] [--bucket-gate N] [--bucket-pre 0|1] [--p1-kv-gate A,B] [--layer L] [--threads T] [--device N] [--mib M] [--iters I] [--n N] [--k K] [--ksplit S]\n");
    return 2;
  }
  const std::string cmd = argv[1];
  Options o;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) {
      std::fprintf(stderr, "%s needs a value\n", a.c_str());
      return 2;
    }
    const char *val = argv[++i];
    const int v = std::atoi(val);
    if (a == "--model") o.model = val;
    else if (a == "--layer") o.layer = v;
    else if (a == "--layers") o.layers = val;
    else if (a == "--threads") o.threads = v;
    else if (a == "--context") o.context = std::atoll(val), o.context_set = true;
    else if (a == "--seqs") o.seqs = v;
    else if (a == "--kv-bytes") o.kv_bytes = v, o.kv_bytes_set = true;
    else if (a == "--drafter") o.drafter = v;
    else if (a == "--hot") o.hot = std::atoll(val);
    else if (a == "--verify") o.verify = v, o.verify_set = true;
    else if (a == "--fill-free") o.fill_free = v;
    else if (a == "--dense") o.dense = v;
    else if (a == "--device") o.device = v;
    else if (a == "--mib") o.mib = v;
    else if (a == "--iters") o.iters = v;
    else if (a == "--n") o.n = v;
    else if (a == "--k") o.k = v;
    else if (a == "--ksplit") o.ksplit = v;
    else if (a == "--ppl") o.ppl = val;
    else if (a == "--rows") o.rows = val;
    else if (a == "--generate") o.generate = v;
    else if (a == "--at") o.at = std::atoll(val);
    else if (a == "--routes-out") o.routes_out = val;
    else if (a == "--hot-rank") o.hot_rank = val;
    else if (a == "--reserve") o.reserve = std::atof(val);
    else if (a == "--lp-out") o.lp_out = val;
    else if (a == "--mtp-dump") o.mtp_dump = val;
    else if (a == "--seq-depths") o.seq_depths = val;
    else if (a == "--step-ab") o.step_ab = val;
    else if (a == "--step-opts") o.step_opts = val;
    else if (a == "--kv-pool") o.kv_pool = g_kv_pool = std::atoll(val);  // The qwen38 commands' too (the line
    else if (a == "--kv-scramble") g_kv_scramble = o.kv_scramble = std::atoi(val) != 0;  // below never matched)
    else if (a == "--step-shapes") o.step_shapes = val;
    else if (a == "--step-reps") o.step_reps = std::stoi(val);
    else if (a == "--profile-card") o.profile_card = std::stoi(val);
    else if (a == "--lp-ref") o.lp_ref = val;
    else if (a == "--bf16") o.bf16 = val;
    else if (a == "--moe-down-bf16") o.moe_down_bf16 = v != 0;
    else if (a == "--moe-dn-ks") o.moe_dn_ks = v, g_q38_moe_dn_ks = static_cast<int>(v);
    else if (a == "--graphs") o.graphs = v != 0;
    else if (a == "--moe-tp") o.moe_tp = v != 0;
    else if (a == "--hidden-out") o.hidden_out = val;
    else if (a == "--gen-out") o.gen_out = val;
    else if (a == "--pf-chunk") o.pf_chunk = v;
    else if (a == "--pf-slices") o.pf_slices = v != 0;
    else if (a == "--pf-attn-slices") o.pf_attn_slices = v != 0;
    else if (a == "--pf-attn") o.pf_attn = v;
    else if (a == "--pf-p-lo") o.pf_p_lo = v != 0;
    else if (a == "--pf-w4dq") o.pf_w4dq = v != 0;
    else if (a == "--pf-ab") o.pf_ab = val;
    else if (a == "--pf-reps") o.pf_reps = v;
    else if (a == "--pf-lens") o.pf_lens = val;
    else if (a == "--sel-budgets") o.sel_budgets = val;
    else if (a == "--doc") o.docs = val;
    else if (a == "--pf-split") o.pf_split = v != 0;
    else if (a == "--pf-moe-a8") g_pf_moe_a8 = v != 0;
    else if (a == "--pf-dense-a8") g_pf_dense_a8 = static_cast<int>(v);
    else if (a == "--pf-wire") g_q38_pf_wire_bf16 = std::string(val) == "bf16";
    else if (a == "--dec-moe-a8") g_dec_moe_a8 = v != 0 ? 1 : 0;
    else if (a == "--kv-lend") g_kv_lend = std::atoi(val) != 0;
    else if (a == "--kv-lend-keep") g_kv_lend_keep = std::atoll(val);
    else if (a == "--kv-lend-idle") g_kv_lend_idle = std::atoi(val) != 0;
    else if (a == "--moe-gu-ks") g_moe_gu_ks = v;
    else if (a == "--w4") g_w4_tensors = val;
    else if (a == "--w4-mse") g_w4_mse = v;
    else if (a == "--gdn-w4") g_gdn_w4 = v;
    else if (a == "--gdn-cap") g_gdn_cap = val;
    else if (a == "--gdn-cap-rows") g_gdn_cap_rows = v;
    else if (a == "--gdn-cap-every") g_gdn_cap_every = v;
    else if (a == "--gdn-scan") g_gdn_scan = v != 0;
    else if (a == "--pf-cold-stage") g_pf_cold_stage = static_cast<int>(v);
    else if (a == "--moe-cpu") g_moe_cpu = val;
    else if (a == "--moe-cpu-pct") g_moe_cpu_pct = static_cast<int>(v);
    else if (a == "--moe-cpu-threads") g_moe_cpu_threads = static_cast<int>(v);
    else if (a == "--moe-cpu-min-cold") g_moe_cpu_min_cold = static_cast<int>(v);
    else if (a == "--pf-route-trace") g_pf_route_trace = val;
    else if (a == "--lp-time") g_lp_time = static_cast<int>(v);
    else if (a == "--step-rows") g_step_rows = static_cast<int>(v);
    else if (a == "--two-deep") g_two_deep = static_cast<int>(v);
    else if (a == "--spec-cap") g_spec_cap = val;
    else if (a == "--bucket-gate") g_bucket_gate = static_cast<int>(v);
    else if (a == "--p1-kv-gate") {
      if (std::sscanf(val, "%d,%d", &g_p1_kv_a, &g_p1_kv_b) != 2) g_p1_kv_a = g_p1_kv_b = -1;
    }
    else if (a == "--bucket-pre") g_bucket_pre = v != 0;
    else if (a == "--pf-time-only") o.pf_time_only = v != 0;
    else if (a == "--pf-sweep") o.pf_sweep = val;
    else if (a == "--mtp") o.mtp = val;
    else if (a == "--drafts-ref") o.drafts_ref = val;
    else if (a == "--prompt-len") o.prompt_len = v;
    else if (a == "--prompts") o.prompts = v;
    else if (a == "--dflash-shift") o.dflash_shift = v;
    else if (a == "--depth") o.depth = v;
    else if (a == "--mtp-kv-bytes") o.mtp_kv_bytes = v, o.mtp_kv_bytes_set = true;
    else if (a == "--spec") o.spec = v;
    else if (a == "--mtp-dense-bytes") o.mtp_dense_bytes = v;
    else if (a == "--mtp-vocab") o.mtp_vocab = v, g_q38_mtp_vocab = v;
    else if (a == "--mtp-head-w4") g_q38_mtp_head_w4 = v;
    else if (a == "--dflash") g_dflash = val;
    else if (a == "--head-w4") g_head_w4 = static_cast<int>(v);
    else if (a == "--head-rescore") g_head_rescore = static_cast<int>(v);
    else if (a == "--draft-head") g_draft_head = static_cast<int>(v);
    else if (a == "--pf-a8") g_pf_a8 = static_cast<int>(v);
    else if (a == "--pf-a8-keep") g_pf_a8_keep = val;
    else if (a == "--head-x-out") g_head_x_out = val;
    else if (a == "--dflash-bytes") g_dflash_bytes = static_cast<int>(v);
    else if (a == "--dflash-check") g_dflash_check = static_cast<int>(v);
    else if (a == "--dflash-prof") g_dflash_prof = v != 0;
    else if (a == "--step-reset") g_step_reset = v != 0;
    else if (a == "--dflash-defer") g_dflash_defer = static_cast<int>(v);
    else if (a == "--dflash-cut") g_dflash_cut = std::atof(val);
    else if (a == "--dflash-tree") g_dflash_tree = std::atoi(val);
    else if (a == "--dflash-tree-cut") g_dflash_tree_cut = std::atof(val);
    else if (a == "--dflash-tree-seqs") g_dflash_tree_seqs = std::atoi(val);
    else if (a == "--dflash-tree-temp") g_dflash_tree_temp = std::atof(val);
    else if (a == "--dflash-wide") g_dflash_wide = std::atoi(val);
    else if (a == "--dflash-tree-wide") g_dflash_tree_wide = std::atoi(val);
    else if (a == "--mtp-from") g_mtp_from = std::atoi(val);
    else if (a == "--tp") g_tp = std::atoi(val);
    else if (a == "--queue-prof") g_queue_prof = std::atoi(val) != 0;
    else if (a == "--sel-kv") g_sel_kv = std::atoi(val) != 0;
    else if (a == "--attn-select") g_attn_select = std::atoi(val);
    else if (a == "--pf-select") g_pf_select = std::atoi(val);
    else if (a == "--sel-pf") g_sel_pf = std::atoi(val) != 0;
    else if (a == "--sel-raw") g_sel_raw = std::atoi(val) != 0;
    else if (a == "--sel-dump") g_sel_dump = val;
    else if (a == "--sel-self") g_sel_self = std::atoi(val);
    else if (a == "--dec-attn") g_dec_attn = val;
    else if (a == "--lp-ab") g_lp_ab = val;
    else if (a == "--pf-checks") g_pf_checks = std::atoi(val) != 0;
    else if (a == "--pf-ab-prof") g_pf_ab_prof = std::atoi(val) != 0;
    else if (a == "--lend-check") g_lend_check = std::atoi(val) != 0;
    else if (a == "--snap-prepin") g_snap_prepin = std::atoi(val) != 0;
    else if (a == "--pf-calls") g_pf_calls = val;
    else if (a == "--pf-merge") g_pf_merge = val;
    else if (a == "--pf-merge-at") g_pf_merge_at = std::atoi(val);
    else if (a == "--mtp-defer") g_mtp_defer = std::atoi(val);
    else if (a == "--kv-v4") g_kv_v4 = std::atoi(val) != 0;
    else if (a == "--draft-conf") g_draft_conf = true, g_draft_conf_path = val;
    else if (a == "--seq-tokens") o.seq_tokens = v;
    else if (a == "--pf-prompt-len") o.pf_prompt_len = v;
    else if (a == "--pf-only") o.pf_only = v != 0;
    else if (a == "--stall-ms") o.stall_ms = std::atof(val);
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  try {
    if (cmd == "devices") return cmd_devices();
    if (cmd == "bw") return cmd_bw(o);
    if (cmd == "w4") return cmd_w4(o);
    if (cmd == "load") return cmd_load(o);
    if (cmd == "dense") return cmd_dense(o);
    if (cmd == "pf") return cmd_pf(o);
    if (cmd == "head") return cmd_head(o);
    if (cmd == "hc") return cmd_hc(o);
    if (cmd == "moe") return cmd_moe_run(o.model, o.layers, o.iters, o.threads);
    if (cmd == "w4pf") return cmd_w4pf_llama(o.model, o.layers, o.rows, o.iters, o.threads, g_tp);
    if (cmd == "gqaflint") return cmd_gqa_flint(o.iters, o.context_set ? static_cast<int>(o.context) : 0);
    if (cmd == "moeflint") return cmd_moe_flint(o.model, o.layers == "0,24,47" ? 24 : std::atoi(o.layers.c_str()), o.iters);
    if (cmd == "w4load") {
      const auto aid = xenithium::model::open_arch(xenithium::model::ModelConfig::load(o.model))->id();
      if (aid == "flint") return cmd_w4load_flint(o.model, o.layers == "0,24,47" ? "0,2,3" : o.layers, o.iters, o.threads, g_tp);
      if (aid == "llama")
        return cmd_w4load_llama(o.model, o.layers == "0,24,47" ? "0,26" : o.layers, o.iters, o.threads, g_tp);
      return cmd_w4load(o.model, o.layers == "0,24,47" ? "0,3" : o.layers, o.iters, o.threads);
    }
    if (cmd == "head35x") return cmd_head35x(o.model, o.ppl, o.generate > 0 ? o.generate : 128, o.threads);
    if (cmd == "head35") return cmd_head35(o.model, o.iters, o.threads);
    if (cmd == "mlp35") return cmd_mlp35(o.model, o.layers == "0,24,47" ? "0,3" : o.layers, o.rows == "1,4" ? 4 : std::stoi(o.rows), o.iters, o.threads);
    if (cmd == "gdn35") return cmd_gdn35(o.model, o.layers == "0,24,47" ? "0,1,2,62" : o.layers, o.iters, o.threads);
    if (cmd == "model35") return cmd_model35(o.model, o.ppl, o.rows, o.generate, o.at, o.context_set ? o.context : 0, o.kv_bytes_set ? o.kv_bytes : 1, o.threads, o.lp_out, o.lp_ref, o.pf_chunk, o.pf_slices, o.pf_attn_slices, o.pf_attn, o.pf_p_lo, o.pf_w4dq, o.pf_split, o.pf_ab, o.pf_reps, [&] {
      std::vector<int> v;
      std::stringstream ss(o.pf_lens);
      std::string t;
      while (std::getline(ss, t, ','))
        if (!t.empty()) v.push_back(std::stoi(t));
      return v;
    }(), o.spec, o.depth, o.mtp_kv_bytes_set ? o.mtp_kv_bytes : 1, o.mtp_dense_bytes, o.mtp_vocab, o.mtp_dump, o.seqs > 1 ? o.seqs : 0, o.seq_tokens, o.seq_depths, o.step_ab, o.step_shapes, o.step_reps, o.profile_card, o.step_opts, o.kv_pool, o.kv_scramble);
    if (cmd == "prefix")
      return cmd_prefix_run(o.model, o.ppl, o.mtp, o.prompt_len, o.generate, o.seqs > 1 ? o.seqs : 6, o.hot_rank, o.threads);
    if (cmd == "prefix35")
      return cmd_prefix35(o.model, o.ppl, o.prompt_len, o.generate, o.context_set ? o.context : 0, o.threads, o.seqs > 1 ? o.seqs : 6);
    if (cmd == "select35") return cmd_select35(o.model, o.ppl, o.pf_lens, o.docs, o.sel_budgets, o.generate, o.step_reps, o.threads, static_cast<int>(o.pf_chunk));
    if (cmd == "gqa35")
      return cmd_gqa35(o.model, o.layers == "0,24,47" ? "3,31,63" : o.layers, o.iters, o.threads, o.kv_bytes_set ? o.kv_bytes : 1, o.context_set ? o.context : 0);
    if (cmd == "gdn") return cmd_gdn_run(o.model, o.layers == "0,24,47" ? "0,1,2,44" : o.layers, o.iters, o.threads);
    if (cmd == "qsa")
      return cmd_qsa_run(o.model, o.layers == "0,24,47" ? "3,23,47" : o.layers, o.iters, o.threads, o.kv_bytes_set ? o.kv_bytes : 1, o.context_set ? o.context : 0, o.verify != 0);
    if (cmd == "ple") return cmd_ple_run(o.model, o.iters, o.threads);
    if (cmd == "store") return cmd_store(o);
    if (cmd == "model")
      return cmd_model_run(o.model, o.ppl, o.rows, o.generate, o.at, o.context, o.kv_bytes_set ? o.kv_bytes : 1, o.hot, o.threads, o.verify_set ? o.verify != 0 : false,
                           o.routes_out, o.hot_rank, o.reserve, o.lp_out, o.lp_ref, o.bf16, o.moe_down_bf16, o.moe_dn_ks, o.graphs, o.moe_tp, o.hidden_out, o.gen_out, o.pf_chunk, o.pf_time_only,
                           o.pf_sweep);
    if (cmd == "dflash")
      return cmd_dflash38(o.model, g_dflash, o.ppl, o.prompt_len, o.prompts, o.generate > 0 ? o.generate : 256, o.depth, o.context, o.graphs, o.hot_rank, o.reserve,
                          o.threads, o.dflash_shift);
    if (cmd == "mtp")
      return cmd_mtp_run(o.model, o.mtp, o.ppl, o.prompt_len, o.generate, o.depth, o.drafts_ref, o.context, o.graphs, o.hot_rank, o.reserve, o.threads, o.mtp_kv_bytes,
                         o.pf_prompt_len, o.pf_only, o.pf_chunk);
    if (cmd == "seqs")
      return cmd_seqs_run(o.model, o.ppl, o.seqs, o.prompt_len, o.generate, o.context_set ? o.context : 0, o.graphs, o.hot_rank, o.reserve, o.threads,
                          o.kv_bytes_set ? o.kv_bytes : 1, o.hot, o.pf_chunk);
    if (cmd == "serve")
      return cmd_serve_run(o.model, o.ppl, o.seqs, o.prompt_len, o.generate, o.context_set ? o.context : 0, o.pf_chunk, o.graphs, o.hot_rank, o.reserve,
                           o.threads, o.kv_bytes_set ? o.kv_bytes : 1, o.stall_ms, o.mtp);
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "xenithium-probe: %s\n", e.what());
    return 1;
  }
}
