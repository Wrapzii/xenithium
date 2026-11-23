// Xenithium · QSA (the full-attention layers): gated GQA over a sparse, indexer-selected set of
// tokens, decode (rows = 1..4 speculative tokens of one sequence), one card's heads.
//
// Semantics (qwen4_exp `full_attention` layers, from config.json and the checkpoint), per token at
// position p (rows r = 0..R-1 sit at p = pos + r):
//   [q_h | gate_h] (per q head), k, v = bf16(x . W^T)                         head_dim 256
//   q, k      = bf16(rope(bf16(gemma_rmsnorm(.))))   gemma: x rsqrt(mean x^2 + eps) (1 + w), fp32
//   rope      NeoX halves over the first 64 dims, cos | sin from the host table (bf16 values,
//             fp32 math, one bf16 rounding); text positions: M-RoPE sections collapse to 1-D
//   index     [iq_0..3 | ik] = bf16(x . Widx^T)                               dim 128
//             iq = bf16(rope(bf16(gemma_rmsnorm(iq))))  (q_layernorm), position p
//             every 4th token (p % 4 == 3) closes block b = p / 4:
//             kc[b] = bf16(rope(bf16(gemma_rmsnorm(bf16(mean ik[4b .. 4b+3])))))  at position 4b
//   score(b)  = sum_h relu(iq_h . kc[b]) / sqrt(128) over the complete blocks b < (p + 1) / 4
//   select    the top 512 blocks (ties: lowest block; all of them when there are <= 512),
//             ascending, then the open block's tail tokens 4 nblk .. p: <= 2,051 tokens
//   attn      softmax(q . K_sel / 16) . V_sel per q head (fp32; q head h reads kv head h / 12); a card's work-group
//             takes its q heads of one kv head: 12 (one or two cards), 6 (four cards: the kv head on
//             two cards, each with 6 of its q heads; the same bits per head)
//   out       bf16(bf16(attn) bf16(sigmoid(gate))) . Wo^T: per card a k slice -> fp32 partials,
//             two-card sum (wire.hpp).
//
// KV cache: pages of 64 tokens named by a per-sequence device page table (logical page ->
// physical page of the pools); selection works on logical positions, only cache reads / writes
// map through the table (an identity table = one contiguous cache). One page id names the K, V,
// scale and compressed-key rows of its 64 tokens / 16 blocks. KV dtype: int8 or bf16.
// int8: one fp32 scale per (token, kv head) over its 256 dims, s = amax / 127, code = rne(x (127 /
// amax)) clamped to +-127 (both divisions IEEE: fdiv_rn on the card, so codes are exact on host and
// card; an all-zero row stores s = 0). The codes widen to bf16 exactly, so attention runs the bf16
// DPAS path and applies the token's scale to its scores / probabilities in fp32.
// int8-v4: K as int8 (as above), V as int4: one bf16 scale per (token, kv head, 32 dims), s =
// bf16(amax / 7), code = rne(x (1 / s)) clamped to +-7 (IEEE divisions), two codes per byte (dim 2j in the low nibble,
// 2j + 1 in the high one): a V row is 128 bytes + 16 bytes of scales (int8: 256 + 4). The codes widen to bf16 exactly;
// each attention item's 16 dims share one scale per token, folded into its P (fp32) before P's hi / lo split.
//
// Speculative rows need no accept word: everything is addressed by position, so a rejected row's
// K / V / raw index key / compressed key is rewritten when the real token at that position comes.
// Rows read only positions <= their own, all written before they are read (prep runs first).
//
// Draft trees (QsaStep::tree, records kSegTree / kSegTreeHi, rows.hpp): row j of a tree segment at
// p0 sits at the virtual position vp = p0 + depth(j) for every rope / count / block rule and writes its K / V / scales
// and raw index key at its ROW slot p0 + j (ring (p0 + j) % 16), so siblings never collide. It reads the committed
// positions (< p0) and its ancestors: position t >= p0 of its path is row anc(j, t - p0) (tree_anc_word), stored at row
// slot p0 + anc. A row with (vp + 1) % 4 == 0 closes block vp / 4 from its path's raw keys into QsaCache::tkc[row] (and
// a main-chain row, depth(j) == j, also into the pool, where a chain writes it); the score of a block closed inside
// the step (b >= p0 / 4) is taken per row from its closing ancestor's tkc. A tree row's output is, bit for bit, that of
// row depth(j) of a chain step of its own path; qsa_kv_move then moves the accepted path to the chain's slots. Segments
// without tree words in a tree step run as chains; QsaStep::tree false runs the chain kernels.
//
// Launches: qkv GEMM (int8), idx GEMM (bf16, replicated), prep (norms, rope, cache writes, block
// compression), score (DPAS, every complete block), select (radix top-k; above 4,096
// blocks in chunks of max(4,096, bound / 16) + one merge), attn (64-token tiles), merge (LSE over tiles, gate), o GEMM (int8, fp32
// partial), pair_reduce.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>
#include <sycl/sycl.hpp>

#include "xenithium/kernels/gemm/dense.hpp"
#include "xenithium/kernels/attention/qsa_ref.hpp"
#include "xenithium/kernels/comm/wire.hpp"
#include "xenithium/kernels/rows.hpp"
#include "xenithium/kernels/stage_trace.hpp"

namespace xenithium::kernels {

using qsa_bf16 = sycl::ext::oneapi::bfloat16;

enum class QsaKv : uint8_t { bf16, i8, i8v4 };  // i8v4: K int8, V int4
inline bool qsa_kv_scaled(QsaKv kv) { return kv != QsaKv::bf16; }  // 8-bit (4-bit V) codes + scale pools
inline bool qsa_kv_v4(QsaKv kv) { return kv == QsaKv::i8v4; }

struct QsaShape {
  int hidden = 0, q_heads = 0, kv_heads = 0;  // the whole model
  int cards = 1, card = 0;
  float eps = 1e-6f;
  QsaKv kv = QsaKv::i8;
  int hq() const { return q_heads / cards; }   // this card's q heads
  // this card's kv heads: kv_heads / cards, or (TP4: 2 kv heads over 4 cards) one kv head that
  // cards / kv_heads cards hold alike (cards 2k and 2k + 1 hold kv head k, each with half its q heads)
  int hkv() const { return kv_heads >= cards ? kv_heads / cards : 1; }
  int kv0() const { return kv_heads >= cards ? card * hkv() : card * kv_heads / cards; }  // this card's first kv head
  int grp() const { return q_heads / kv_heads; }  // the model's q heads per kv head
  int lgrp() const { return hq() / hkv(); }       // this card's q heads per kv head (grp, or grp x kv_heads / cards)
  // the heads split over the cards: whole kv heads per card, or whole cards per kv head with its q heads split evenly
  bool split_ok() const {
    return cards >= 1 && card >= 0 && card < cards && kv_heads > 0 && q_heads % kv_heads == 0 && q_heads % cards == 0 &&
           (kv_heads % cards == 0 || (cards % kv_heads == 0 && grp() % (cards / kv_heads) == 0));
  }
  int q_n() const { return hq() * 2 * kQsaDim; }               // q | gate per head
  int qkv_n() const { return q_n() + 2 * hkv() * kQsaDim; }    // ... | k | v
  int idx_n() const { return (kQsaIdxHeads + 1) * kQsaIdxDim; }
  int o_k() const { return hq() * kQsaDim; }
  int kv_elem() const { return kv == QsaKv::bf16 ? 2 : 1; }                  // K bytes per element
  int v_row_bytes() const { return qsa_kv_v4(kv) ? kQsaDim / 2 : kQsaDim * kv_elem(); }  // one token's V of one head
  int v_scale_bytes() const { return qsa_kv_v4(kv) ? kQsaV4Scales * 2 : 4; }  // V scale bytes per (token, kv head)
};

struct QsaWeights {
  DenseWeights qkv;  // int8 [q|gate of each head | k | v] of this card's heads x hidden
  DenseWeights idx;  // bf16 [iq 4 x 128 | ik 128] x hidden (replicated)
  DenseWeights o;    // int8 [hidden][hq x 256]: this card's k slice of o_proj
  const qsa_bf16 *q_norm = nullptr, *k_norm = nullptr;    // [256]
  const qsa_bf16 *iq_norm = nullptr, *ik_norm = nullptr;  // [128]
};

// ---- cache --------------------------------------------------------------------------------------
// The sequences' state for one layer on this card. Pools are indexed by physical page and shared
// by every slot; each slot (rows.hpp) has its own page table row and raw-key ring.
struct QsaCache {
  uint8_t *k = nullptr, *v = nullptr;   // [pages][hkv][64][256] x kv_elem bytes; v: [pages][hkv][64][v_row_bytes]
  float *ks = nullptr, *vs = nullptr;   // [pages][hkv][64] fp32 (8-bit KV scales; unused for bf16); int8-v4's vs holds
                                        // bf16 [pages][hkv][64][kQsaV4Scales] (one per 32 dims)
  qsa_bf16 *kc = nullptr;               // [pages][16][128] compressed index keys
  qsa_bf16 *ring = nullptr;             // [slots][16][128] raw index keys, slot position % 16
  const int32_t *pages = nullptr;       // [slots][slot_pages]: logical page -> physical page (device)
  int max_tokens = 0;                   // capacity per slot: logical pages x 64
  int slot_pages = 0;                   // page table ints per slot (0: one slot)
  // this layer's per-row compressed keys of a tree step, [kMaxRows][128] (kQsaTreeKcElems; a step
  // row's key of the block it closes, read by its descendants' scores and by qsa_kv_move); needed by tree steps only
  qsa_bf16 *tkc = nullptr;
};
inline constexpr size_t kQsaTreeKcElems = static_cast<size_t>(kMaxRows) * kQsaIdxDim;
// Rows a tree segment holds at most: its row slots p0 .. p0 + 12 keep the ring entries of p0 - 3 .. p0 - 1 (the
// committed raw keys a block straddling p0 reads) out of the 16-entry ring.
inline constexpr int kQsaTreeRows = kQsaRing - (kQsaCompress - 1);
size_t qsa_page_kv_bytes(const QsaShape &s);  // K bytes of one page (all hkv heads)
size_t qsa_page_v_bytes(const QsaShape &s);   // V bytes of one page (== K's but with int8-v4)
size_t qsa_page_scale_floats(const QsaShape &s);   // K's scales of one page (and V's but with int8-v4)
size_t qsa_page_vscale_floats(const QsaShape &s);  // V's scales of one page, in floats (int8-v4: bf16 pairs)
size_t qsa_page_kc_elems();

// Decode scratch for R rows (a step's rows, every segment together; the GEMM
// buffers y / yi / attn / partial hold the row shape's rows).
struct QsaScratch {
  float *y = nullptr;        // [R][qkv_n] projections
  float *yi = nullptr;       // [R][idx_n]
  qsa_bf16 *q = nullptr;     // [R][hq][256] normed, roped
  qsa_bf16 *iq = nullptr;    // [R][4][128]
  int32_t *nblk = nullptr;   // [R] complete blocks per row (written by prep)
  uint32_t *score = nullptr; // [R][max_blocks] score bits (non-negative floats: bit order = order)
  uint32_t *cand_key = nullptr;  // [R][16][512] chunk candidates (chunked select only)
  int32_t *cand_id = nullptr;    // [R][16][512]
  int32_t *cand_n = nullptr;     // [R][16]
  int32_t *sel = nullptr;    // [R][512] selected blocks, ascending
  int32_t *sel_n = nullptr;  // [R]
  float *part = nullptr;     // [R][hkv][33][lgrp][2 + 256]: tile max, sum, o
  qsa_bf16 *attn = nullptr;  // [R][hq x 256] gated attention = o_proj input
  float *partial = nullptr;  // [R][hidden]
  // Decode profiling (XEN_DEC_PROFILE, qsa_decode only): a cycle stamp (comm/wire.hpp stamp) after each stage before
  // QsaStage::out, word s for stage s (kQsaStamped words), so a profile splits the module into its stages; out (o_proj +
  // the cross-card sum) runs from the last stamp to the caller's next one. Null = no stamps.
  uint64_t *stamps = nullptr;
  // Four cards: with sel_share set and sel_hi > sel_lo (this card's owner_rows of
  // the step, comm/wire.hpp), score + select run for those rows only, then owner_share sends their sel rows to the other
  // cards (sel as bf16 rows of 2 x 512: its stride; the mailbox owner_share_words(1024), parity sel_parity) and every
  // row's sel_n is min(nblk, 512) - the same bits on every card as every card selecting every row (an exact top-k).
  int sel_lo = 0, sel_hi = 0;
  const OwnerShare *sel_share = nullptr;
  int sel_parity = 0;
};
// The stages of a QSA decode layer in launch order (QsaScratch::stamps).
enum class QsaStage : int { projections, prep, score, select, attention, out, kCount };
inline constexpr int kQsaStamped = static_cast<int>(QsaStage::out);  // stamp words qsa_decode writes
const char *qsa_stage_name(QsaStage s);
// qsa_rows' launches by stage (QsaRowsScratch::trace): select includes the chunked select's merge, attention the
// tiles' merge, out the o_proj parts, the split's send and the sum.
using QsaTrace = StageTrace<static_cast<int>(QsaStage::kCount)>;
// Scratch sizes for a cache capacity (tokens), for `rows` rows.
size_t qsa_score_words(int max_tokens, int rows = kQsaRows);
size_t qsa_part_floats(const QsaShape &s, int rows = kQsaRows);

struct QsaStep {
  const int32_t *pos = nullptr;  // device word: position of row 0; or segment records (rows.hpp)
  int pos_bound = 0;             // host bound: every row's position < pos_bound <= cache.max_tokens (grid sizes)
  int stride = 0;                // ints between segment records (kSegRec), 0 = a bare position word
  int seg_rows = 0;              // rows per segment (0 = one segment)
  bool tree = false;  // some segment's record is a draft tree (needs records and cache.tkc)
  // the caller knows every row has at most kQsaTopBlocks complete blocks (pos_bound <= kQsaTopBlocks *
  // kQsaCompress): its selection is every block, which the attention reads from the position alone, so qsa_decode
  // launches no score and no select (score, sel and sel_n are not written)
  bool every = false;
};

struct QsaEvents {
  sycl::event qkv, idx, prep, score, select, merge_sel, attn, merge, o, reduce;
  sycl::event send;  // the split's peer copy (qsa_rows with a send)
};

// x bf16 [shape][hidden] (row_shape(rows) rows readable, rows.hpp); out bf16 [shape][hidden] (rows < `rows`
// written). Rows 1..8, in segments of any size.
// rope: device bf16 [>= cache.max_tokens][64]. wire: null with one card. Several sequences
// (step.stride, step.seg_rows): row r reads its segment's position and slot; every row computes
// what a step of its own sequence alone computes for it, bit for bit.
QsaEvents qsa_decode(sycl::queue &q, const QsaShape &s, int rows, const qsa_bf16 *x, const QsaWeights &w, const QsaCache &c,
                     const qsa_bf16 *rope, const QsaStep &step, const QsaScratch &sc, const PairWire *wire, int wire_parity,
                     qsa_bf16 *out, int qkv_ks = 0, int o_ks = 0);
// After a tree step, the accepted path's per-position state to the chain's slots. from[x] = the step
// row of accepted position p0 + x (from[0] = 0, strictly increasing, < kQsaTreeRows; the segment's first step row is
// r0). In each listed cache (one card's QSA layers, sequence slot `slot`), for every x with from[x] != x: K / V codes and
// their scales of every kv head of this card from slot p0 + from[x] to p0 + x, the raw index key ring (p0 + from[x]) % 16
// -> (p0 + x) % 16, and, when position p0 + x closes a block, the pool's compressed key of block (p0 + x) / 4 =
// tkc[r0 + from[x]]. x ascending: a destination p0 + x is only ever the source of an earlier x, so every source is read
// before it is overwritten. One launch per kQsaMoveCaches caches; a path that is rows 0 .. n - 1 launches nothing.
inline constexpr int kQsaMoveCaches = 16;
struct QsaMove {
  int n = 0;                  // accepted positions
  int from[kTreeRows] = {};   // step row (in the segment) of accepted position x
};
sycl::event qsa_kv_move(sycl::queue &q, const QsaShape &s, const QsaCache *caches, int n_caches, int slot, int p0, int r0, const QsaMove &mv);
// qsa_decode's cache writes only (the qkv / idx projections and prep: K / V,
// scales, the raw-key ring and compressed keys), bit for bit what qsa_decode stores for these rows;
// no attention, no output, no wire. For a pass whose output nobody reads (the drafter's pass 1 on a
// step that drafts nothing next) but whose KV later passes attend to. Refuses tree steps.
QsaEvents qsa_store(sycl::queue &q, const QsaShape &s, int rows, const qsa_bf16 *x, const QsaWeights &w, const QsaCache &c,
                    const qsa_bf16 *rope, const QsaStep &step, const QsaScratch &sc, int qkv_ks = 0);

// ---- many rows (prefill) ----------------------------------------------------------------
// The projections (qkv, idx, o) over the chunk's m rows on pf_dense and the two-card sum on the
// chunk-sized wire (stage A). The middle (stage B): one prep over all m rows (the raw-key ring is
// written afterwards from the chunk's last 16 rows: a row writing the ring inside prep could
// overwrite a slot the chunk's first block still reads), then per batch of kQsaRowsBatch rows score
// + select (only when a row of the batch has more than 512 complete blocks: below that the selection
// is every block, which attn / merge take directly) and attn + merge. Every row computes what the
// decode kernels compute for it, bit for bit. words: per 4-row slice a device word with the slice's
// first position, word_stride ints apart (slice 0's = pos0); pos0: the chunk's first position. With
// word_stride >= kSegRec the words are segment records (rows.hpp) and slice 0's names the slot. Chains only:
// the records' tree words are not read (its kernels are the chain instances), so a caller never passes a
// tree step's records here.
inline constexpr int kQsaRowsBatch = 64;  // rows per score / select / attn batch (a multiple of 4)
struct QsaRowsScratch {
  float *y = nullptr;        // [rows_max + 4][qkv.n]
  float *yi = nullptr;       // [rows_max + 4][idx.n]
  qsa_bf16 *attn = nullptr;  // [rows_max + 4][o_k]
  float *partial = nullptr;  // [rows_max][hidden]
  qsa_bf16 *q = nullptr;     // [rows_max + 4][hq][256]
  qsa_bf16 *iq = nullptr;    // [rows_max + 4][4][128]
  int32_t *nblk = nullptr;   // [rows_max + 4]
  int32_t *sel = nullptr;    // [rows_max + 4][512]
  int32_t *sel_n = nullptr;  // [rows_max + 4]
  // per batch (reused):
  uint32_t *score = nullptr;     // [kQsaRowsBatch][qsa_score_words(cap) / 4]
  uint32_t *cand_key = nullptr;  // [kQsaRowsBatch][16][512]
  int32_t *cand_id = nullptr;    // [kQsaRowsBatch][16][512]
  int32_t *cand_n = nullptr;     // [kQsaRowsBatch][16]
  float *part = nullptr;         // qsa_part_floats(s) / 4 x kQsaRowsBatch
  PfA8 a8;                       // set = the qkv and o GEMMs run W8A8 (pf_dense.hpp; idx stays bf16)
  QsaTrace *trace = nullptr;     // prefill profiling: every launch's event by stage (stage_trace.hpp); null = none
  // The indexer by owner rows. With `between` set and sel_hi > sel_lo,
  // score + select run for the call's rows [sel_lo, sel_hi) only (sel_lo a multiple of kQsaRows; every row's selection is
  // the same bits whichever rows share its batch: the select is an exact top-k), then `between` (the caller sends this
  // card's rows of sel / sel_n to the other cards and waits for theirs; its argument: the last select launch, null when
  // none ran), then attention for every row.
  int sel_lo = 0, sel_hi = 0;
  const std::function<void(const sycl::event *)> *between = nullptr;
};
QsaEvents qsa_rows(sycl::queue &q, const QsaShape &s, int m, const qsa_bf16 *x, const QsaWeights &w, const QsaCache &c, const qsa_bf16 *rope,
                   const int32_t *words, int word_stride, int64_t pos0, const QsaScratch &sc, const QsaRowsScratch &rs, const PairRowsWire *wire,
                   int wire_parity, qsa_bf16 *out);

// Opt-in host callback for a selector-boundary capture. Register once before launching work; with no callback the
// selector only performs a null function-pointer check. A capture callback may enqueue copies after `complete` and wait;
// it must not throw and must not retain device pointers after returning. The normal server never registers this hook.
struct QsaSelectCaptureView {
  int rows = 0, stride = 0, bound = 0, chunk = 0, tp_rank = 0, tp_size = 1;
  const uint32_t *scores = nullptr;
  const int32_t *nblk = nullptr, *selected_ids = nullptr, *selected_counts = nullptr;
  sycl::event first, complete;
};
using QsaSelectCaptureHook = void (*)(sycl::queue &q, const QsaSelectCaptureView &view, void *user) noexcept;
void qsa_set_select_capture_hook(QsaSelectCaptureHook hook, void *user = nullptr) noexcept;

// Opt-in capture of one decode attention boundary. The callback may wait for complete, copy device inputs/output,
// and must not retain device pointers or throw. Production does not register this hook.
struct QsaSparseAttentionCaptureView {
  int rows = 0, projection_stride = 0, tp_rank = 0, tp_size = 1;
  QsaShape shape;
  QsaCache cache;
  QsaStep step;
  QsaScratch scratch;
  sycl::event complete;
};
using QsaSparseAttentionCaptureHook = void (*)(sycl::queue &q, const QsaSparseAttentionCaptureView &view, void *user) noexcept;
void qsa_set_sparse_attention_capture_hook(QsaSparseAttentionCaptureHook hook, void *user = nullptr) noexcept;

// The select stage alone: rows' keys in score [rows][stride], nblk[r] keys per row (device words), top-512 -> sel / sel_n
// (candidate buffers sized for `rows`). bound = the largest nblk (grid size). Optional TP identity is for capture metadata.
sycl::event qsa_select(sycl::queue &q, int rows, const uint32_t *score, int stride, const int32_t *nblk, int bound, const QsaScratch &sc,
                       sycl::event *merge = nullptr, int chunk = 0, int tp_rank = 0, int tp_size = 1);
// Blocks per select chunk for a bound: max(4,096, bound / 16) up to 16,384; one launch
// (no merge) while bound <= the chunk. qsa_select's chunk 0 = this (decode); qsa_rows passes 16,384.
int qsa_select_chunk(int bound);

}  // namespace xenithium::kernels
