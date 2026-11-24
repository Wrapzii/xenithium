// Xenithium · QSA decode (see qsa.hpp for the semantics, the cache layout and the row protocol).
#include "xenithium/kernels/attention/qsa.hpp"
#include "xenithium/kernels/attention/qsa_xrv.hpp"
#include "xenithium/kernels/gemm/pf_dense.hpp"

#include <algorithm>
#include <atomic>
#include <iterator>
#include <type_traits>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/math.hpp>

#include "xenithium/kernels/rows.hpp"
#include "attention/topk_wg.hpp"

namespace xenithium::kernels {
namespace {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = esimd::xmx;
using bf16 = qsa_bf16;

constexpr int kD = kQsaDim, kID = kQsaIdxDim, kM = kQsaRows, kT = kQsaTiles, kTile = kQsaTile;
constexpr int kPartStride = kD + 16;  // o[256] | max, sum (16-float aligned rows)
constexpr int kSelItems = 1024;       // select work-group
constexpr int kSelStride = 8, kSelFiltMin = 8192, kSelFiltMax = 8192;  // the select's pre-filter
constexpr int kScoreItems = 16, kScoreBlocks = 64;  // blocks per score work-item (8 DPAS groups of 8)
// Up to 4,096 blocks (16K tokens) an item takes 16 blocks; above it 64.
constexpr int kScoreBlocksShort = 16, kScoreShortBound = 4096;

inline float bfr(float v) { return static_cast<float>(bf16(v)); }

// The rows' positions and slots (rows.hpp): a bare position word (stride 0) or segment records.
struct SegW {
  const int32_t *rec = nullptr;
  int stride = 0, seg_rows = 0;
};

// Draft trees (qsa.hpp): a row's place in its segment's tree. vp = the virtual position (the segment's
// first position + depth), anc = the step row (in the segment) of each depth of its path, 4 bits per depth
// (tree_anc_word). A chain row: vp = its position, depth = its row, anc = the identity (chain segments of a tree step
// hold <= kQsaTreeRows rows, so every depth fits a nibble).
inline constexpr uint64_t kAncIdentity = 0xFEDCBA9876543210ull;
struct TreeRow {
  int vp, depth;
  uint64_t anc;
  bool tree;
};
inline TreeRow tree_row(const int32_t *rec, int stride, const SegRow &sr) {
  TreeRow t{sr.pos, sr.j, kAncIdentity, false};
  if (stride >= kSegRec) {
    const uint64_t tw = rec_word(rec + sr.seg * stride, kSegTree, kSegTreeHi);
    if (tw & kTreeFlag) {
      t.anc = tree_anc_word(tw, sr.j, &t.depth);
      t.vp = sr.pos - sr.j + t.depth;
      t.tree = true;
    }
  }
  return t;
}
// the step row (in the segment) at depth t of a path word
inline int anc_row(uint64_t anc, int t) { return static_cast<int>((anc >> (4 * t)) & 15u); }

// ---- prep: norms, rope, cache writes, block compression (SIMT, one work-group per (row, unit)) ----
// Units: q heads [0, hq), kv heads [hq, hq + hkv), the indexer (last). 256 items, item d = dim d.
// TREE: the step may hold tree segments (qsa.hpp); TREE false is the chain instance.
template <QsaKv KV, bool TREE>
class QsaPrepK;

// int8 KV code of v for a row with amax > 0 (qsa.hpp): rne(v (127 / amax)) clamped to +-127, IEEE
// divisions (fdiv_rn: the plain operator is approximate on the card), == qsa_ref_i8.
inline int8_t i8_enc(float v, float inv) {
  const float t = sycl::rint(v * inv);
  return static_cast<int8_t>(t > 127.f ? 127.f : (t < -127.f ? -127.f : t));
}
// int4 code (int8-v4's V): rne(v (1 / s)) clamped to +-7, == qsa_ref_v4.
inline int i4_enc(float v, float inv) {
  const float t = sycl::rint(v * inv);
  return static_cast<int>(t > 7.f ? 7.f : (t < -7.f ? -7.f : t));
}

// ring_from: rows >= it write the raw index key into the ring (decode: 0; many rows: none, see ring_tail).
template <QsaKv KV, bool TREE>
sycl::event prep(sycl::queue &q, const QsaShape &s, int rows, const QsaWeights &w, const QsaCache &c, const bf16 *rope, const SegW &sw,
                 const QsaScratch &sc, int ring_from = 0) {
  const int hq = s.hq(), hkv = s.hkv(), units = hq + hkv + 1, ldy = w.qkv.n, ldi = w.idx.n, qn = s.q_n();
  const float eps = s.eps;
  const float *y = sc.y, *yi = sc.yi;
  bf16 *qo = sc.q, *iqo = sc.iq, *ring0 = c.ring, *kc = c.kc, *tkc = c.tkc;
  uint8_t *kp = c.k, *vp = c.v;
  float *ksp = c.ks, *vsp = c.vs;
  int32_t *nblk = sc.nblk;
  const int32_t *pages0 = c.pages, *rec = sw.rec;
  const int slot_pages = c.slot_pages, rstride = sw.stride, seg_rows = sw.seg_rows;
  const bf16 *qnw = w.q_norm, *knw = w.k_norm, *iqw = w.iq_norm, *ikw = w.ik_norm;
  return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> buf(sycl::range<1>(kD), h);
    sycl::local_accessor<float, 1> qd(sycl::range<1>(4), h);  // int8 KV: the K / V reciprocals and scales
    sycl::local_accessor<float, 1> vg(sycl::range<1>(KV == QsaKv::i8v4 ? kD : 1), h);  // int8-v4: |v| per dim
    sycl::local_accessor<float, 1> qv(sycl::range<1>(KV == QsaKv::i8v4 ? kQsaV4Scales : 1), h);  // int8-v4: 1 / s per group
    h.parallel_for<QsaPrepK<KV, TREE>>(sycl::nd_range<2>({static_cast<size_t>(rows), static_cast<size_t>(units) * kD}, {1, kD}), [=](sycl::nd_item<2> it) {
      // a (row, unit) grid: the ids come from the launch, not from a division by the run-time unit count in every item
      const int r = static_cast<int>(it.get_group(0)), u = static_cast<int>(it.get_group(1)), d = static_cast<int>(it.get_local_id(1));
      // the row's segment: its first position pos0 and first row rbase; the slot's page table / ring
      const SegRow sr = seg_row(rec, rstride, seg_rows, r);
      const int p = sr.pos, pos0 = p - sr.j, rbase = r - sr.j;
      // a tree row ropes, counts and closes blocks at its virtual position; its slots stay at p
      TreeRow tr{p, sr.j, kAncIdentity, false};
      if constexpr (TREE) tr = tree_row(rec, rstride, sr);
      const int vpos = tr.vp;
      const int32_t *pages = pages0 + static_cast<size_t>(sr.slot) * slot_pages;
      bf16 *ring = ring0 + static_cast<size_t>(sr.slot) * kQsaRing * kID;
      // the segment's rope delta (records: word kSegRope, a YaRN tier's first table row; a bare position word: 0)
      const int rdelta = rstride >= kSegRec ? rec[sr.seg * rstride + kSegRope] : 0;
      const auto grp = it.get_group();
      auto norm_rope = [&](float v, int L, const bf16 *nw, int rp) {
        const float ss = sycl::reduce_over_group(grp, d < L ? v * v : 0.f, sycl::plus<float>());
        const float wv = d < L ? static_cast<float>(nw[d]) : 0.f;
        const float n = bfr(v * sycl::rsqrt(ss / static_cast<float>(L) + eps) * (1.f + wv));
        sycl::group_barrier(grp);
        buf[d] = n;
        sycl::group_barrier(grp);
        float o = n;
        if (d < kQsaRot) {
          const int i = d & 31;
          const size_t row = static_cast<size_t>(rp + rdelta) * kQsaRopeRow;
          const float cs = static_cast<float>(rope[row + i]);
          const float sn = static_cast<float>(rope[row + 32 + i]);
          o = d < 32 ? bfr(n * cs - buf[d + 32] * sn) : bfr(n * cs + buf[d - 32] * sn);
        }
        return o;
      };
      const float *yr = y + static_cast<size_t>(r) * ldy;
      if (u < hq) {
        const float o = norm_rope(bfr(yr[u * 2 * kD + d]), kD, qnw, vpos);
        qo[(static_cast<size_t>(r) * hq + u) * kD + d] = bf16(o);
      } else if (u < hq + hkv) {
        const int kh = u - hq;
        const float kn = norm_rope(bfr(yr[qn + kh * kD + d]), kD, knw, vpos);
        const float vv = bfr(yr[qn + (hkv + kh) * kD + d]);
        const size_t slot = (static_cast<size_t>(pages[p >> 6]) * hkv + kh) * kQsaPage + (p & 63);
        if constexpr (KV != QsaKv::bf16) {
          // The four IEEE divisions (a software sequence, fdiv_rn) once per work-group, each in its own
          // sub-group, broadcast through SLM (not once per lane).
          namespace imf = sycl::ext::intel::math;
          constexpr bool V4 = KV == QsaKv::i8v4;
          constexpr float vq = V4 ? 7.f : 127.f;  // V's largest code
          const float ka = sycl::reduce_over_group(grp, sycl::fabs(kn), sycl::maximum<float>());
          const float va = sycl::reduce_over_group(grp, sycl::fabs(vv), sycl::maximum<float>());
          const bool kz = !(ka > 0.f), vz = !(va > 0.f);
          if (d == 0) qd[0] = kz ? 0.f : imf::fdiv_rn(127.f, ka);
          if (!V4 && d == 64) qd[1] = vz ? 0.f : imf::fdiv_rn(vq, va);
          if (d == 128) qd[2] = kz ? 0.f : imf::fdiv_rn(ka, 127.f);
          if (!V4 && d == 192) qd[3] = vz ? 0.f : imf::fdiv_rn(va, vq);
          sycl::group_barrier(grp);
          reinterpret_cast<int8_t *>(kp)[slot * kD + d] = kz ? int8_t(0) : i8_enc(kn, qd[0]);
          if constexpr (V4) {  // two codes per byte: dim d (even) low, d + 1 high; the odd lane hands its code over
            // one bf16 scale per kQsaV4Group dims (== qsa_ref_v4): each sub-group's amax to SLM, then lane 32 g + 16 takes
            // group g: its two IEEE divisions (software sequences) run in the group's own sub-group, beside K's (lanes 0 /
            // 128), not one after another in sub-group 0
            const auto sg = it.get_sub_group();
            const int S = static_cast<int>(sg.get_max_local_range()[0]);  // 8 / 16 / 32: divides the group
            const float sm = sycl::reduce_over_group(sg, sycl::fabs(vv), sycl::maximum<float>());
            if (sg.get_local_linear_id() == 0) vg[d / S] = sm;
            sycl::group_barrier(grp);
            if (d % kQsaV4Group == kQsaV4Group / 2) {
              const int gq = d / kQsaV4Group;
              float am = 0.f;
              for (int j = gq * kQsaV4Group / S; j < (gq + 1) * kQsaV4Group / S; ++j) am = sycl::fmax(am, vg[j]);
              const bf16 sb = am > 0.f ? bf16(imf::fdiv_rn(am, 7.f)) : bf16(0.f);
              const float sf = static_cast<float>(sb);
              qv[gq] = sf > 0.f ? imf::fdiv_rn(1.f, sf) : 0.f;
              reinterpret_cast<bf16 *>(vsp)[slot * kQsaV4Scales + gq] = sb;
            }
            sycl::group_barrier(grp);
            const int cv = i4_enc(vv, qv[d / kQsaV4Group]);  // inv 0 (an all-zero group): code 0
            const int c1 = sycl::permute_group_by_xor(it.get_sub_group(), cv, 1);
            if ((d & 1) == 0) vp[slot * (kD / 2) + d / 2] = static_cast<uint8_t>((cv & 0xF) | ((c1 & 0xF) << 4));
          } else {
            reinterpret_cast<int8_t *>(vp)[slot * kD + d] = vz ? int8_t(0) : i8_enc(vv, qd[1]);
          }
          if (d == 0) {
            ksp[slot] = qd[2];
            if constexpr (!V4) vsp[slot] = qd[3];
          }
        } else {
          reinterpret_cast<bf16 *>(kp)[slot * kD + d] = bf16(kn);
          reinterpret_cast<bf16 *>(vp)[slot * kD + d] = bf16(vv);
        }
      } else {
        const float *yir = yi + static_cast<size_t>(r) * ldi;
        for (int hh = 0; hh < kQsaIdxHeads; ++hh) {
          const float o = norm_rope(d < kID ? bfr(yir[hh * kID + d]) : 0.f, kID, iqw, vpos);
          if (d < kID) iqo[(static_cast<size_t>(r) * kQsaIdxHeads + hh) * kID + d] = bf16(o);
        }
        if (d < kID && r >= ring_from) ring[(p % kQsaRing) * kID + d] = bf16(bfr(yir[kQsaIdxHeads * kID + d]));
        if (d == 0) nblk[r] = (vpos + 1) / kQsaCompress;
        if ((vpos + 1) % kQsaCompress == 0) {
          const int b = vpos / kQsaCompress;
          float sum = 0.f;
          if (d < kID) {
#pragma unroll
            for (int j = 0; j < kQsaCompress; ++j) {
              const int pj = b * kQsaCompress + j;
              // the path's raw key at pj: a tree row's ancestor (TREE), the row pj - pos0 of a chain
              const int prow = TREE ? anc_row(tr.anc, pj - pos0) : pj - pos0;
              const float v = pj >= pos0 ? bfr(yi[static_cast<size_t>(rbase + prow) * ldi + kQsaIdxHeads * kID + d])
                                         : static_cast<float>(ring[(pj % kQsaRing) * kID + d]);
              sum = j == 0 ? v : sum + v;
            }
          }
          const float o = norm_rope(bfr(sum / static_cast<float>(kQsaCompress)), kID, ikw, b * kQsaCompress);
          // the pool's key where a chain puts it: every chain row and a tree's main-chain rows (one per block: their
          // virtual positions differ); a tree row's own key in tkc (its descendants' scores, qsa_kv_move)
          if (d < kID && tr.depth == sr.j) kc[(static_cast<size_t>(pages[b / kQsaPageBlocks]) * kQsaPageBlocks + (b % kQsaPageBlocks)) * kID + d] = bf16(o);
          if constexpr (TREE)
            if (d < kID && tr.tree) tkc[static_cast<size_t>(r) * kID + d] = bf16(o);
        }
      }
    });
  });
}

// ---- score: every complete block of every row on DPAS (A = 8 compressed keys, B = the 16 query
// columns row x indexer head of a 4-row group). One work-item = 64 blocks; keys = score bits with
// the sign cleared. Spills 512 B at 128 GRF. Rows beyond 4 run as further 4-row groups; a group never spans two
// segments (each segment reads its own slot's keys): segments of seg rows take ceil(seg / 4) groups each.
// TREE: in a tree segment the blocks closed inside the step (b >= p0 / 4: their keys differ per
// branch) are rescored per row: the same DPAS body on the pool keys with those blocks' rows of A taken from the row's
// closing ancestor's tkc, the row's own columns kept. A chain step of the row's path computes the same products in the
// same order on the same key values, so the scores are its bits; the pool's in-step keys are never used for a tree row.
template <bool TREE>
class QsaScoreK;
// [r_lo, r_hi) = the rows whose keys are wanted (the groups of 4 rows outside it return; a group across a bound
// computes all its rows: the same bits). The default is every row (a prefill batch may have more rows than a decode
// step).
template <bool TREE>
sycl::event score(sycl::queue &q, int rows, const QsaCache &c, const SegW &sw, int blk_bound, const QsaScratch &sc, int stride, int r_lo = 0,
                  int r_hi = std::numeric_limits<int>::max()) {
  const int bpi = blk_bound <= kScoreShortBound ? kScoreBlocksShort : kScoreBlocks;
  const int wgs = std::max(1, (blk_bound + kScoreItems * bpi - 1) / (kScoreItems * bpi));
  const int sr_rows = seg_size(rows, sw.seg_rows), nseg = seg_count(rows, sw.seg_rows), gps = (sr_rows + kM - 1) / kM;
  const bf16 *iq0 = sc.iq, *kc = c.kc, *tkc = c.tkc;
  const int32_t *pages0 = c.pages, *rec = sw.rec;
  const int slot_pages = c.slot_pages, rstride = sw.stride, seg_rows = sw.seg_rows;
  uint32_t *out0 = sc.score;
  return q.parallel_for<QsaScoreK<TREE>>(
      sycl::nd_range<3>({static_cast<size_t>(nseg), static_cast<size_t>(gps), static_cast<size_t>(wgs) * kScoreItems}, {1, 1, kScoreItems}),
      [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
        // a (segment, group, work-group) grid: the ids come from the launch, not from divisions by run-time counts in every item
        const int sg = static_cast<int>(it.get_group(0)), gi = static_cast<int>(it.get_group(1));
        const int b0 = (static_cast<int>(it.get_group(2)) * kScoreItems + static_cast<int>(it.get_local_id(2))) * bpi;
        int sg0 = 0, seg_n = 0;  // the segment's first row and rows (any layout)
        seg_span(seg_rows, rows, sg, sg0, seg_n);
        const int r0 = sg0 + gi * kM;  // the group's first row
        const int rows_g = seg_n - gi * kM < kM ? seg_n - gi * kM : kM;
        if (rows_g <= 0 || r0 + rows_g <= r_lo || r0 >= r_hi) return;
        const SegRow srw = seg_row(rec, rstride, seg_rows, r0);
        const int32_t *pages = pages0 + static_cast<size_t>(srw.slot) * slot_pages;
        const int nmax = (srw.pos + rows_g) / kQsaCompress;
        if (b0 >= nmax) return;
        const bf16 *iq = iq0 + static_cast<size_t>(r0) * kQsaIdxHeads * kID;
        uint32_t *out = out0 + static_cast<size_t>(r0) * stride;
        // B: the indexer queries packed for DPAS: dword (t, lane) at t * 16 + lane, t = k / 2.
        esimd::simd<uint32_t, 1024> bq;
        {
          esimd::simd<uint32_t, 1024> raw;
          const uint32_t *src = reinterpret_cast<const uint32_t *>(iq);
#pragma unroll
          for (int j = 0; j < 8; ++j) raw.template select<128, 1>(j * 128) = esimd::block_load<uint32_t, 128>(src + j * 128);
#pragma unroll
          for (int L = 0; L < 16; ++L) {
            esimd::simd<uint32_t, 64> col = raw.template select<64, 1>(L * 64);
            if (L >= rows_g * kQsaIdxHeads) col = 0u;
            bq.template select<64, 16>(L) = col;
          }
        }
        const float inv = 1.f / sycl::sqrt(static_cast<float>(kID));
        // C (8 blocks x 16 query columns) = A (8 keys) . B
        auto dpas8 = [&](esimd::simd<bf16, 1024> &A) SYCL_ESIMD_FUNCTION {
          esimd::simd<float, 128> acc(0.f);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            esimd::simd<bf16, 128> a;
#pragma unroll
            for (int m = 0; m < 8; ++m) a.template select<16, 1>(m * 16) = A.template select<16, 1>(m * kID + j * 16);
            esimd::simd<uint32_t, 128> bj = bq.template select<128, 1>(j * 128);
            esimd::simd<bf16, 256> b = bj.template bit_cast_view<bf16>();
            acc = xmx::dpas<8, 8, float, float, bf16, bf16>(acc, b, a);
          }
          return acc;
        };
        // row r's keys of the 8 blocks from bg: sum_h relu / sqrt(128), the sign bit cleared
        auto store_keys = [&](esimd::simd<float, 128> &acc, int r, int bg) SYCL_ESIMD_FUNCTION {
          esimd::simd<float, 8> s(0.f);
#pragma unroll
          for (int hh = 0; hh < kQsaIdxHeads; ++hh) {
            esimd::simd<float, 8> v = acc.template select<8, 16>(r * kQsaIdxHeads + hh);
            v.merge(esimd::simd<float, 8>(0.f), v < 0.f);
            s += v;
          }
          s *= inv;
          esimd::simd<uint32_t, 8> key = s.template bit_cast_view<uint32_t>();
          key &= 0x7FFFFFFFu;
          esimd::block_store<uint32_t, 8>(out + static_cast<size_t>(r) * stride + bg, key);
        };
        // the group's segment's tree word (0 = a chain) and its first position
        uint64_t tw = 0;
        if constexpr (TREE) tw = rstride >= kSegRec ? rec_word(rec + srw.seg * rstride, kSegTree, kSegTreeHi) : 0ull;
        const int p0 = srw.pos - srw.j;
        for (int g = 0; g < bpi / 8; ++g) {
          const int bg = b0 + g * 8;
          if (bg >= nmax) break;
          const size_t row = static_cast<size_t>(pages[bg / kQsaPageBlocks]) * kQsaPageBlocks + (bg % kQsaPageBlocks);
          esimd::simd<bf16, 1024> A;
#pragma unroll
          for (int j = 0; j < 4; ++j) A.template select<256, 1>(j * 256) = esimd::block_load<bf16, 256>(kc + row * kID + j * 256);
          esimd::simd<float, 128> acc = dpas8(A);
          for (int r = 0; r < rows_g; ++r) store_keys(acc, r, bg);
          if constexpr (TREE) {
            if ((tw & kTreeFlag) && bg + 8 > p0 / kQsaCompress) {  // the group holds blocks closed inside the step
              for (int r = 0; r < rows_g; ++r) {
                int dep = 0;
                const uint64_t anc = tree_anc_word(tw, srw.j + r, &dep);
                const int nb = (p0 + dep + 1) / kQsaCompress;  // the row's complete blocks
                esimd::simd<bf16, 1024> At = A;
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                  const int b = bg + m, t = b * kQsaCompress + kQsaCompress - 1 - p0;  // the depth that closes block b
                  if (t >= 0 && b < nb)
                    At.template select<kID, 1>(m * kID) = esimd::block_load<bf16, kID>(tkc + static_cast<size_t>(sg0 + anc_row(anc, t)) * kID);
                }
                esimd::simd<float, 128> acct = dpas8(At);
                store_keys(acct, r, bg);
              }
            }
          }
        }
      });
}

using topk::SelSlm;
using topk::topk_wg;

template <bool FILT>
class QsaSelChunkK;
class QsaSelMergeK;
class QsaSelNK;
}  // namespace

namespace {
std::atomic<QsaSelectCaptureHook> g_qsa_select_capture_hook{nullptr};
std::atomic<void *> g_qsa_select_capture_user{nullptr};
std::atomic<QsaSparseAttentionCaptureHook> g_qsa_sparse_attention_capture_hook{nullptr};
std::atomic<void *> g_qsa_sparse_attention_capture_user{nullptr};

void capture_qsa_select(sycl::queue &q, int rows, const uint32_t *score, int stride, const int32_t *nblk, int bound, int chunk,
                        const QsaScratch &sc, int tp_rank, int tp_size, const sycl::event &first, const sycl::event &complete) noexcept {
  const auto hook = g_qsa_select_capture_hook.load(std::memory_order_acquire);
  if (!hook) return;
  const QsaSelectCaptureView view{rows, stride, bound, chunk, tp_rank, tp_size, score, nblk, sc.sel, sc.sel_n, first, complete};
  hook(q, view, g_qsa_select_capture_user.load(std::memory_order_relaxed));
}
}  // namespace

void qsa_set_select_capture_hook(QsaSelectCaptureHook hook, void *user) noexcept {
  g_qsa_select_capture_user.store(user, std::memory_order_relaxed);
  g_qsa_select_capture_hook.store(hook, std::memory_order_release);
}

void qsa_set_sparse_attention_capture_hook(QsaSparseAttentionCaptureHook hook, void *user) noexcept {
  g_qsa_sparse_attention_capture_user.store(user, std::memory_order_relaxed);
  g_qsa_sparse_attention_capture_hook.store(hook, std::memory_order_release);
}

namespace {
void capture_qsa_sparse_attention(sycl::queue &q, const QsaShape &shape, int rows, const QsaCache &cache, const QsaStep &step,
                                  const QsaScratch &scratch, int projection_stride, const sycl::event &complete) noexcept {
  const auto hook = g_qsa_sparse_attention_capture_hook.load(std::memory_order_acquire);
  if (!hook || step.tree || shape.kv != QsaKv::i8v4) return;
  const QsaSparseAttentionCaptureView view{rows, projection_stride, shape.card, shape.cards, shape, cache, step, scratch, complete};
  hook(q, view, g_qsa_sparse_attention_capture_user.load(std::memory_order_relaxed));
}
}  // namespace

int qsa_select_chunk(int bound) {
  // Up to 16 chunk work-groups per row (one per 4,096 blocks at least), as wide as the cap
  // of 16 candidate lists allows.
  const int c = ((bound + kQsaMaxChunks - 1) / kQsaMaxChunks + 63) / 64 * 64;
  return std::min(kQsaChunk, std::max(kQsaMinChunk, c));
}

sycl::event qsa_select(sycl::queue &q, int rows, const uint32_t *score, int stride, const int32_t *nblk, int bound, const QsaScratch &sc,
                       sycl::event *merge, int chunk, int tp_rank, int tp_size) {
  if (tp_size < 1 || tp_rank < 0 || tp_rank >= tp_size) throw std::invalid_argument("qsa_select: invalid tensor-parallel rank");
  if (chunk <= 0) chunk = qsa_select_chunk(bound);
  if (chunk < kQsaMinChunk || chunk > kQsaPfChunk) throw std::invalid_argument("qsa_select: chunk out of range");
  const int nch = std::max(1, (bound + chunk - 1) / chunk);
  if (nch > kQsaMaxBlocks / kQsaChunk) throw std::invalid_argument("qsa_select: more blocks than the kernel supports");
  const bool direct = nch == 1;
  int32_t *sel = sc.sel, *sel_n = sc.sel_n, *cid = sc.cand_id, *cn = sc.cand_n;
  uint32_t *ckey = sc.cand_key;
  constexpr int K = kQsaTopBlocks, kCh = kQsaMaxBlocks / kQsaChunk;
  auto launch = [&](auto filt_t) {
    constexpr bool FILT = decltype(filt_t)::value;
    return q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> hist(sycl::range<1>(512), h);
    sycl::local_accessor<int32_t, 1> sh(sycl::range<1>(8), h);
    sycl::local_accessor<uint32_t, 1> fkey(sycl::range<1>(FILT ? kSelFiltMax : 1), h);  // the pre-filter's candidates
    sycl::local_accessor<int32_t, 1> fidx(sycl::range<1>(FILT ? kSelFiltMax : 1), h);
    sycl::local_accessor<uint32_t, 1> fsh(sycl::range<1>(4), h);
    h.parallel_for<QsaSelChunkK<FILT>>(sycl::nd_range<1>(static_cast<size_t>(rows) * nch * kSelItems, kSelItems),
                                 [=](sycl::nd_item<1> it) {
      const int g = static_cast<int>(it.get_group(0)), r = g / nch, ch = g % nch;
      const int c0 = ch * chunk;
      const int n = sycl::clamp(nblk[r] - c0, 0, chunk);
      const uint32_t *keys = score + static_cast<size_t>(r) * stride + c0;
      SelSlm m{&hist[0], &sh[0]};
      int32_t *o = direct ? sel + r * K : cid + (r * kCh + ch) * K;
      int32_t *on = direct ? sel_n + r : cn + r * kCh + ch;
      const auto grp = it.get_group();
      const int li = static_cast<int>(it.get_local_id(0));
      // A provable pre-filter. The K-th largest key of every kSelStride-th key is <= the
      // K-th largest of all, so every key of the exact top-K (ties included) is >= it. The keys >= it go to SLM in index
      // order and the exact top-K runs on them only: the same set, ties to the lowest index, the same bits. Too many
      // candidates (heavy ties) fall back to the full select.
      bool filt = false;
      int nc = 0;
      if (FILT && n >= kSelFiltMin) {
        const int ns = n / kSelStride;
        topk_wg(
            it, m, ns, [](int) { return true; }, [&](int i) { return keys[i * kSelStride]; }, [&](int i) { return i; }, K, o, on);
        sycl::group_barrier(grp, sycl::memory_scope::device);
        uint32_t tmin = 0xFFFFFFFFu;
        for (int j = li; j < K; j += kSelItems) {
          const uint32_t kk = keys[o[j] * kSelStride];
          tmin = kk < tmin ? kk : tmin;
        }
        const uint32_t Ts = sycl::reduce_over_group(grp, tmin, sycl::minimum<uint32_t>());
        const int per = (n + kSelItems - 1) / kSelItems, i0 = sycl::min(n, li * per), i1 = sycl::min(n, i0 + per);
        int cnt = 0;
        for (int i = i0; i < i1; ++i) cnt += keys[i] >= Ts;
        int at = sycl::exclusive_scan_over_group(grp, cnt, sycl::plus<int>());
        nc = sycl::reduce_over_group(grp, cnt, sycl::plus<int>());
        filt = nc <= kSelFiltMax;
        if (filt)
          for (int i = i0; i < i1; ++i)
            if (keys[i] >= Ts) fkey[at] = keys[i], fidx[at] = i, ++at;
        sycl::group_barrier(grp, sycl::memory_scope::device);  // o's sample ids read, the candidates stored
      }
      auto kf = [&](int i) { return filt ? fkey[i] : keys[i]; };
      auto xf = [&](int i) { return filt ? fidx[i] : i; };
      const int nn = filt ? nc : n;
      if (!direct) {  // keep the candidates' keys too
        uint32_t *ok = ckey + (r * kCh + ch) * K;
        topk_wg(
            it, m, nn, [](int) { return true; }, kf, xf, K, o, on);
        sycl::group_barrier(grp);
        const int cnt = *on;
        for (int i = li; i < cnt; i += kSelItems) {
          const int b = o[i];
          ok[i] = keys[b];
          o[i] = c0 + b;
        }
      } else {
        topk_wg(
            it, m, nn, [](int) { return true; }, kf, [&](int i) { return c0 + xf(i); }, K, o, on);
      }
    });
    });
  };
  // the pre-filter runs in the chunked selects only (chunks of kSelFiltMin blocks or more); a one-chunk select is the
  // plain select
  const bool filt = nch > 1 && chunk >= kSelFiltMin;
  sycl::event e = filt ? launch(std::true_type{}) : launch(std::false_type{});
  if (direct) {
    if (merge) *merge = e;
    capture_qsa_select(q, rows, score, stride, nblk, bound, chunk, sc, tp_rank, tp_size, e, e);
    return e;
  }
  sycl::event e2 = q.submit([&](sycl::handler &h) {
    sycl::local_accessor<uint32_t, 1> hist(sycl::range<1>(512), h);
    sycl::local_accessor<int32_t, 1> sh(sycl::range<1>(8), h);
    h.parallel_for<QsaSelMergeK>(sycl::nd_range<1>(static_cast<size_t>(rows) * kSelItems, kSelItems),
                                 [=](sycl::nd_item<1> it) {
      const int r = static_cast<int>(it.get_group(0));
      const int32_t *cnr = cn + r * kCh;
      const uint32_t *kr = ckey + r * kCh * K;
      const int32_t *ir = cid + r * kCh * K;
      SelSlm m{&hist[0], &sh[0]};
      topk_wg(
          it, m, nch * K, [&](int i) { return (i % K) < cnr[i / K]; }, [&](int i) { return kr[i]; }, [&](int i) { return ir[i]; }, K, sel + r * K,
          sel_n + r);
    });
  });
  if (merge) *merge = e2;
  capture_qsa_select(q, rows, score, stride, nblk, bound, chunk, sc, tp_rank, tp_size, e, e2);
  return e;
}

namespace {

// ---- attention: one work-group per (row, kv head, 64-token tile of the row's selection).
// 16 items x 16 dims: each K / V byte is read once; partial q.k per item, summed in SLM in item
// order; softmax per head over the tile; P V per item; (o, max, sum) per head to `part`.
// kLines (int8 K, six q heads per kv head = four cards; decode steps and the prefill's tile groups, TPW > 1): the
// scores' K by whole cache lines. An item takes a quarter of the dims (64 bytes = one line per token) of a quarter of
// the tile's tokens and computes that quarter's four 16-dim partials; the partials are then summed over the 16 dim
// groups in the same order as without kLines, so every score has the same bits, and the tile needs two barriers for
// its scores instead of eight.
template <QsaKv KV, int G, int TPW, bool TREE>
class QsaAttnK;

// TPW tiles per work-group, walked in order with an online softmax (one partial per TPW tiles:
// the merge reads ceil(tiles / TPW) partials). TPW 1 is the decode's shape and its bits (alpha = 0, beta = 1 exactly).
// TREE: a tree row's counts from its virtual position, its path's tokens >= p0 at its ancestors'
// row slots (decode only: TPW 1).
template <QsaKv KV, int G, int TPW = 1, bool TREE = false>
sycl::event attn(sycl::queue &q, const QsaShape &s, int rows, const QsaCache &c, const SegW &sw, const QsaScratch &sc) {
  constexpr bool I8 = KV != QsaKv::bf16;  // int8 K codes (V int8 or int4) + per-token scales
  constexpr bool V4 = KV == QsaKv::i8v4;  // V as int4 codes, two per byte
  constexpr int kI = 16, kE = I8 ? 1 : 2;
  constexpr int kTokOff = 0, kKsOff = kTokOff + kTile, kVsOff = kKsOff + kTile;            // floats / ints
  constexpr bool kLines = I8 && G == 6;  // the scores' K by cache lines (above): decode steps and the prefill's tile groups
  constexpr int kLq = 4;                               // kLines: an item = 1 / kLq of the dims x 1 / kLq of the tile's tokens
  constexpr int kSubs = kTile / 16;
  static_assert(kI == kLq * kLq && kD == kLq * 64 && kSubs == kLq, "kLines: 16 items = 4 dim quarters (a 64-byte line each) x 4 x 16 tokens");
  constexpr int kSpOff = kVsOff + kTile, kSpN = (kLines ? kSubs : 1) * kI * G * 16 + 16;    // partial dots: [sub (kLines)][dim group][head][token]
  constexpr int kScOff = kSpOff + kSpN, kPvOff = kScOff + G * kTile, kMlOff = kPvOff + G * kTile;
  constexpr int kAlOff = kMlOff + 2 * G + 16;  // the online softmax's per-head rescale of o
  constexpr int kSlm = (kAlOff + 16) * 4;
  constexpr int kTG = (kT + TPW - 1) / TPW;  // tile groups per (row, kv head)
  const int hq = s.hq(), hkv = s.hkv();
  const uint8_t *kp = c.k, *vp = c.v;
  const float *ksp = c.ks, *vsp = c.vs;
  const int32_t *pages0 = c.pages, *sel = sc.sel, *sel_n = sc.sel_n, *rec = sw.rec;
  const int slot_pages = c.slot_pages, rstride = sw.stride, seg_rows = sw.seg_rows;
  const bf16 *qb = sc.q;
  float *part = sc.part;
  static_assert(!TREE || TPW == 1, "tree rows run in decode steps (one tile per work-group)");
  return q.parallel_for<QsaAttnK<KV, G, TPW, TREE>>(
      sycl::nd_range<3>({static_cast<size_t>(rows), static_cast<size_t>(hkv), static_cast<size_t>(kTG) * kI}, {1, 1, kI}),
      [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
        esimd::slm_init<kSlm>();
        // a (row, kv head, tile group) grid: the ids come from the launch, not from divisions by the run-time head count in every item
        const int r = static_cast<int>(it.get_group(0)), kvh = static_cast<int>(it.get_group(1)), tg = static_cast<int>(it.get_group(2));
        const int i = static_cast<int>(it.get_local_id(2));
        // nblk <= 512: the selection is every block (select would write 0 .. nblk - 1), so token
        // lt of the selection is position lt and sel is not read (many-row batches skip select).
        const SegRow srw = seg_row(rec, rstride, seg_rows, r);
        const int32_t *pages = pages0 + static_cast<size_t>(srw.slot) * slot_pages;
        TreeRow tr{srw.pos, srw.j, kAncIdentity, false};
        if constexpr (TREE) tr = tree_row(rec, rstride, srw);
        const int p = tr.vp, nblk = (p + 1) / kQsaCompress, p0 = srw.pos - srw.j;
        const bool every = nblk <= kQsaTopBlocks;
        const int nsel = every ? nblk : sel_n[r];
        const int ntok = nsel * kQsaCompress + (p + 1 - nblk * kQsaCompress);
        if (tg * TPW * kTile >= ntok) return;
        auto so = [](int f) SYCL_ESIMD_FUNCTION { return static_cast<uint32_t>(f * 4); };
        // the DPAS split: 12 heads = 8 + 4 rows; 6 heads (TP4) = one 8-row call, rows 6-7 zero (a DPAS
        // row's sums do not depend on the other rows: each head's bits are the 12-head kernel's)
        static_assert(G == 12 || G == 6, "attention DPAS split: 8 + 4 heads, or 6 in 8 rows");
        constexpr int kQA = G > 8 ? G : 8;  // A operand rows
        // q of this item's dims, [head][dims]: its 16-dim group, or (kLines) its quarter of the dims = four groups
        constexpr int kQd = kLines ? kD / kLq : 16;
        esimd::simd<bf16, G * kQd> qi;
#pragma unroll
        for (int h = 0; h < G; ++h)
          qi.template select<kQd, 1>(h * kQd) =
              esimd::block_load<bf16, kQd>(qb + (static_cast<size_t>(r) * hq + kvh * G + h) * kD + (kLines ? (i % kLq) * kQd : i * 16));
        // the scores' A operand: [head][16 dims] of q's dim group d4 of qi (rows past G stay zero)
        auto q_group = [&](int d4) SYCL_ESIMD_FUNCTION {
          esimd::simd<bf16, kQA * 16> a(0.f);
#pragma unroll
          for (int h = 0; h < G; ++h) a.template select<16, 1>(h * 16) = qi.template select<16, 1>(h * kQd + d4 * 16);
          return a;
        };
        esimd::simd<float, 128> o8(0.f);
        esimd::simd<float, 64> o4(0.f);
        float Mrun = -INFINITY, Lrun = 0.f;  // item i < G: head i's running max and sum over the group's tiles
#pragma unroll 1
        for (int tt = 0; tt < TPW; ++tt) {
        const int tile = tg * TPW + tt;
        const int t0 = tile * kTile;
        if (t0 >= ntok) break;
        const int nt = ntok - t0 < kTile ? ntok - t0 : kTile;
        if (tt > 0) esimd::barrier();  // the previous tile's SLM reads are done
        // token slots of the tile (dead slots repeat the last token)
        {
          constexpr int kPer = kTile / kI;
          esimd::simd<int32_t, kPer> sl;
          esimd::simd<float, kPer> kss(1.f), vss(1.f);
#pragma unroll
          for (int j = 0; j < kPer; ++j) {
            int t = i * kPer + j;
            t = t < nt ? t : nt - 1;
            const int lt = t0 + t;
            int tok = every                       ? lt
                      : lt < nsel * kQsaCompress ? sel[r * kQsaTopBlocks + lt / kQsaCompress] * kQsaCompress + (lt % kQsaCompress)
                                                 : nblk * kQsaCompress + (lt - nsel * kQsaCompress);
            if constexpr (TREE) tok = tok >= p0 ? p0 + anc_row(tr.anc, tok - p0) : tok;  // the path's row slot
            const int slot = (pages[tok / kQsaPage] * hkv + kvh) * kQsaPage + (tok % kQsaPage);
            sl[j] = slot;
            if constexpr (I8) {
              kss[j] = ksp[slot];
              if constexpr (!V4) vss[j] = vsp[slot];  // int8-v4: per 16-dim slot, applied per item in P V
            }
          }
          esimd::slm_block_store<int32_t, kPer>(so(kTokOff + i * kPer), sl);
          esimd::slm_block_store<float, kPer>(so(kKsOff + i * kPer), kss);
          esimd::slm_block_store<float, kPer>(so(kVsOff + i * kPer), vss);
        }
        esimd::barrier();
        // int8 codes widen to bf16 exactly (|code| <= 127 has <= 7 significant bits)
        auto row16b = [&](const uint8_t *base, int slot) SYCL_ESIMD_FUNCTION {
          const uint8_t *src = base + (static_cast<size_t>(slot) * kD + i * 16) * kE;
          esimd::simd<bf16, 16> v;
          if constexpr (I8) {
            esimd::simd<uint8_t, 16> raw = esimd::block_load<uint8_t, 16>(src);
            const esimd::simd<int8_t, 16> sb = raw.template bit_cast_view<int8_t>();
            const esimd::simd<float, 16> f = sb;
            v = f;
          } else {
            v = esimd::block_load<bf16, 16>(reinterpret_cast<const bf16 *>(src));
          }
          return v;
        };
        // 16 tokens' rows (this item's 16 dims each) as B operand values: int8 codes widen as one
        // 256-element batch (exact)
        auto tile16 = [&](const uint8_t *base, esimd::simd<int32_t, 16> sl) SYCL_ESIMD_FUNCTION {
          esimd::simd<bf16, 256> out;
          if constexpr (I8) {
            esimd::simd<uint8_t, 256> raw;
#pragma unroll
            for (int tl = 0; tl < 16; ++tl)
              raw.template select<16, 1>(tl * 16) = esimd::block_load<uint8_t, 16>(base + static_cast<size_t>(sl[tl]) * kD + i * 16);
            esimd::simd<int8_t, 256> sb = raw.template bit_cast_view<int8_t>();
            const esimd::simd<float, 256> f = sb;
            out = f;
          } else {
#pragma unroll
            for (int tl = 0; tl < 16; ++tl) out.template select<16, 1>(tl * 16) = row16b(base, sl[tl]);
          }
          return out;
        };
        // V's tile: int4 codes (V4) = 8 bytes of this item's 16 dims per token, one 32-dword gather; nibbles widen
        // exactly (|code| <= 7) to the same [token][dim] order as tile16's
        auto tile16v = [&](esimd::simd<int32_t, 16> sl) SYCL_ESIMD_FUNCTION {
          if constexpr (V4) {
            esimd::simd<uint32_t, 32> off;
            const esimd::simd<uint32_t, 16> rb = esimd::simd<uint32_t, 16>(sl) * static_cast<uint32_t>(kD / 2) + static_cast<uint32_t>(i * 8);
            off.template select<16, 2>(0) = rb;
            off.template select<16, 2>(1) = rb + 4u;
            esimd::simd<uint32_t, 32> raw = esimd::gather<uint32_t, 32>(reinterpret_cast<const uint32_t *>(vp), off);
            const esimd::simd<int8_t, 128> sb = raw.template bit_cast_view<int8_t>();
            const esimd::simd<int32_t, 128> w = sb;  // sign-extended bytes
            const esimd::simd<int32_t, 128> l4 = w & 0xF;
            const esimd::simd<float, 128> lo = l4 - ((l4 & 8) << 1), hi = w >> 4;
            esimd::simd<bf16, 256> out;
            out.template select<128, 2>(0) = esimd::simd<bf16, 128>(lo);
            out.template select<128, 2>(1) = esimd::simd<bf16, 128>(hi);
            return out;
          } else {
            return tile16(vp, sl);
          }
        };
        const float kmul = 1.f / 16.f;  // 1 / sqrt(256)
        // ---- scores ----
        // Partial q.k of 16 dims x 16 tokens on DPAS: B = K^T (16 dims x 16 tokens, VNNI: dword (dim pair p, token t)
        // at p * 16 + t = a transpose of the K rows' dwords), A = q of 8 + 4 heads (6: one 8-row call). Products are
        // exact (bf16 x bf16-exact K), sums fp32.
        auto partial = [&](esimd::simd<bf16, 256> kb, esimd::simd<bf16, kQA * 16> a) SYCL_ESIMD_FUNCTION {
          esimd::simd<uint32_t, 128> bd;
          {
            esimd::simd<uint32_t, 128> kd = kb.template bit_cast_view<uint32_t>();
#pragma unroll
            for (int pp = 0; pp < 8; ++pp) bd.template select<16, 1>(pp * 16) = kd.template select<16, 8>(pp);
          }
          const esimd::simd<bf16, 256> bk = bd.template bit_cast_view<bf16>();
          esimd::simd<float, G * 16> sp;
          const esimd::simd<bf16, 128> a8 = a.template select<128, 1>(0);
          esimd::simd<float, 128> c8(0.f);
          c8 = xmx::dpas<8, 8, float, float, bf16, bf16>(c8, bk, a8);
          if constexpr (G > 8) {
            const esimd::simd<bf16, 64> a4 = a.template select<64, 1>(128);
            esimd::simd<float, 64> c4(0.f);
            c4 = xmx::dpas<8, 4, float, float, bf16, bf16>(c4, bk, a4);
            sp.template select<128, 1>(0) = c8;
            sp.template select<64, 1>(128) = c4;
          } else {
            sp = c8.template select<G * 16, 1>(0);
          }
          return sp;
        };
        // item i sums outputs [i G, i G + G) of sub's (h, tl) over the 16 dim groups' partials at sp0, in group order
        auto sum_scores = [&](int sub, int sp0) SYCL_ESIMD_FUNCTION {
          esimd::simd<float, 16> acc(0.f);
#pragma unroll
          for (int dg = 0; dg < kI; ++dg) acc += esimd::slm_block_load<float, 16>(so(sp0 + dg * G * 16 + i * G));
          const esimd::simd<float, 16> ks = esimd::slm_block_load<float, 16>(so(kKsOff + sub * 16));
#pragma unroll
          for (int j = 0; j < G; ++j) {
            const int o = i * G + j, h = o / 16, tl = o % 16;
            float sv = acc[j] * kmul;
            if constexpr (I8) sv *= static_cast<float>(ks[tl]);
            esimd::slm_scalar_store<float>(so(kScOff + h * kTile + sub * 16 + tl), sv);
          }
        };
        if constexpr (kLines) {
          // item = (the 16 tokens of sub ts, dims quarter dq): one 64-byte line of each token's K row, then its four dim groups
          const int ts = i / kLq, dq = i % kLq;
          if (ts * 16 < nt) {
            const esimd::simd<int32_t, 16> sl = esimd::slm_block_load<int32_t, 16>(so(kTokOff + ts * 16));
            esimd::simd<uint8_t, 16 * 64> lines;  // [token][64 dims]
#pragma unroll
            for (int tl = 0; tl < 16; ++tl)
              lines.template select<64, 1>(tl * 64) = esimd::block_load<uint8_t, 64>(kp + static_cast<size_t>(sl[tl]) * kD + dq * 64);
#pragma unroll
            for (int d4 = 0; d4 < kLq; ++d4) {
              const int dg = dq * kLq + d4;
              esimd::simd<uint8_t, 256> raw;  // [token][the group's 16 dims]
#pragma unroll
              for (int tl = 0; tl < 16; ++tl) raw.template select<16, 1>(tl * 16) = lines.template select<16, 1>(tl * 64 + d4 * 16);
              const esimd::simd<int8_t, 256> codes = raw.template bit_cast_view<int8_t>();
              const esimd::simd<float, 256> f = codes;
              esimd::slm_block_store<float, G * 16>(so(kSpOff + (ts * kI + dg) * G * 16), partial(esimd::simd<bf16, 256>(f), q_group(d4)));
            }
          }
          esimd::barrier();
#pragma unroll
          for (int sub = 0; sub < kSubs; ++sub)
            if (sub * 16 < nt) sum_scores(sub, kSpOff + sub * kI * G * 16);
          esimd::barrier();
        } else {
          // 16 tokens at a time: every item its 16 dims of them
#pragma unroll 1
          for (int sub = 0; sub < kSubs; ++sub) {
            if (sub * 16 >= nt) break;
            const esimd::simd<int32_t, 16> sl = esimd::slm_block_load<int32_t, 16>(so(kTokOff + sub * 16));
            esimd::slm_block_store<float, G * 16>(so(kSpOff + i * G * 16), partial(tile16(kp, sl), q_group(0)));
            esimd::barrier();
            sum_scores(sub, kSpOff);
            esimd::barrier();
          }
        }
        // ---- softmax per head over the tile ----
        if (i < G) {
          esimd::simd<float, kTile> sv = esimd::slm_block_load<float, kTile>(so(kScOff + i * kTile));
          esimd::simd<int32_t, kTile> ti(0, 1);
          sv.merge(esimd::simd<float, kTile>(-INFINITY), ti >= nt);
          const float mx = esimd::hmax<float>(sv);
          esimd::simd<float, kTile> pe = esimd::exp(sv - mx);
          pe.merge(esimd::simd<float, kTile>(0.f), ti >= nt);
          const float l = esimd::reduce<float>(pe, std::plus<>());
          if constexpr (TPW > 1) {  // online: rescale the group's running sums to the new max
            const float Mn = Mrun > mx ? Mrun : mx;
            esimd::simd<float, 2> d2;
            d2[0] = Mrun - Mn;
            d2[1] = mx - Mn;
            const esimd::simd<float, 2> e2 = esimd::exp(d2);
            const float al = e2[0], be = e2[1];
            pe *= be;
            Lrun = Lrun * al + l * be;
            Mrun = Mn;
            esimd::slm_scalar_store<float>(so(kAlOff + i), al);
          } else {
            Mrun = mx;
            Lrun = l;
          }
          if constexpr (I8 && !V4) pe *= esimd::slm_block_load<float, kTile>(so(kVsOff));
          esimd::slm_block_store<float, kTile>(so(kPvOff + i * kTile), pe);
        }
        esimd::barrier();
        if constexpr (TPW > 1) {
          const esimd::simd<float, 16> al = esimd::slm_block_load<float, 16>(so(kAlOff));
#pragma unroll
          for (int h = 0; h < G; ++h) {
            if (h < 8)
              o8.template select<16, 1>(h * 16) *= al[h];
            else
              o4.template select<16, 1>((h - 8) * 16) *= al[h];
          }
        }
        // ---- P V on DPAS: C (heads x 16 dims) += P (heads x 16 tokens) . V (16 tokens x 16 dims),
        // V as B in token-pair VNNI order; P = hi + lo (two bf16 parts, ~16 significant bits).
#pragma unroll 1
        for (int sub = 0; sub < kSubs; ++sub) {
          if (sub * 16 >= nt) break;
          const esimd::simd<int32_t, 16> sl = esimd::slm_block_load<int32_t, 16>(so(kTokOff + sub * 16));
          esimd::simd<bf16, 256> vb = tile16v(sl);
          esimd::simd<bf16, 256> bv;
#pragma unroll
          for (int qq = 0; qq < 8; ++qq) {
            bv.template select<16, 2>(qq * 32) = vb.template select<16, 1>(2 * qq * 16);
            bv.template select<16, 2>(qq * 32 + 1) = vb.template select<16, 1>((2 * qq + 1) * 16);
          }
          esimd::simd<float, kQA * 16> pw(0.f);  // rows past G stay zero (6 heads: the 8-row call's rows 6-7)
#pragma unroll
          for (int h = 0; h < G; ++h) pw.template select<16, 1>(h * 16) = esimd::slm_block_load<float, 16>(so(kPvOff + h * kTile + sub * 16));
          if constexpr (V4) {  // this item's dims share one scale per token: fold it into P (fp32) before the hi / lo split
            static_assert(kQsaV4Group % 16 == 0 && kQsaV4Scales % 2 == 0, "an item's 16 dims in one group; scales in dword pairs");
            constexpr int kIg = kQsaV4Group / 16;  // items per group
            const int gi = i / kIg;
            const esimd::simd<uint32_t, 16> so4 = (esimd::simd<uint32_t, 16>(sl) * static_cast<uint32_t>(kQsaV4Scales / 2) + static_cast<uint32_t>(gi / 2)) * 4u;
            const esimd::simd<uint32_t, 16> dw = esimd::gather<uint32_t, 16>(reinterpret_cast<const uint32_t *>(vsp), so4);
            esimd::simd<uint32_t, 16> fb = (gi & 1 ? dw >> 16 : dw & 0xFFFFu) << 16;  // the bf16 scale's bits as fp32
            const esimd::simd<float, 16> vsg = fb.template bit_cast_view<float>();
#pragma unroll
            for (int h = 0; h < G; ++h) pw.template select<16, 1>(h * 16) *= vsg;
          }
          esimd::simd<bf16, kQA * 16> ph = pw;
          esimd::simd<bf16, kQA * 16> pl = pw - esimd::simd<float, kQA * 16>(ph);
          const esimd::simd<bf16, 128> h8 = ph.template select<128, 1>(0), l8 = pl.template select<128, 1>(0);
          o8 = xmx::dpas<8, 8, float, float, bf16, bf16>(o8, bv, h8);
          o8 = xmx::dpas<8, 8, float, float, bf16, bf16>(o8, bv, l8);
          if constexpr (G > 8) {
            const esimd::simd<bf16, 64> h4 = ph.template select<64, 1>(128), l4 = pl.template select<64, 1>(128);
            o4 = xmx::dpas<8, 4, float, float, bf16, bf16>(o4, bv, h4);
            o4 = xmx::dpas<8, 4, float, float, bf16, bf16>(o4, bv, l4);
          }
        }
        }  // the group's tiles
        if (i < G) {
          float *pm = part + ((static_cast<size_t>(r * hkv + kvh) * kT + tg) * G + i) * kPartStride + kD;
          esimd::simd<float, 2> ml;
          ml[0] = Mrun;
          ml[1] = Lrun;
          esimd::block_store<float, 2>(pm, ml);
        }
#pragma unroll
        for (int h = 0; h < G; ++h) {
          const esimd::simd<float, 16> oh = h < 8 ? esimd::simd<float, 16>(o8.template select<16, 1>((h % 8) * 16))
                                                  : esimd::simd<float, 16>(o4.template select<16, 1>((h % 4) * 16));
          esimd::block_store<float, 16>(part + ((static_cast<size_t>(r * hkv + kvh) * kT + tg) * G + h) * kPartStride + i * 16, oh);
        }
      });
}

// ---- merge: LSE over the tiles, bf16, output gate (SIMT, one work-group per (row, q head)) ----
// TREE: a tree row's token count from its virtual position.
template <bool TREE>
class QsaMergeK;
template <bool TREE = false>
sycl::event merge(sycl::queue &q, const QsaShape &s, int rows, int ldy, const SegW &sw, const QsaScratch &sc, int tpw = 1) {
  const int hq = s.hq(), hkv = s.hkv(), G = s.lgrp();  // this card's q heads per kv head (6 at TP4)
  const float *part = sc.part, *y = sc.y;
  const int32_t *sel_n = sc.sel_n, *rec = sw.rec;
  const int rstride = sw.stride, seg_rows = sw.seg_rows;
  bf16 *out = sc.attn;
  return q.parallel_for<QsaMergeK<TREE>>(sycl::nd_range<3>({static_cast<size_t>(rows), static_cast<size_t>(hkv), static_cast<size_t>(G) * kD}, {1, 1, kD}), [=](sycl::nd_item<3> it) {
    // a (row, kv head, q head of it) grid: the ids come from the launch, not from divisions by run-time head counts in every item
    const int r = static_cast<int>(it.get_group(0)), kvh = static_cast<int>(it.get_group(1)), hh = static_cast<int>(it.get_group(2));
    const int d = static_cast<int>(it.get_local_id(2)), h = kvh * G + hh;
    const SegRow srw = seg_row(rec, rstride, seg_rows, r);
    int p = srw.pos;
    if constexpr (TREE) p = tree_row(rec, rstride, srw).vp;
    const int nblk = (p + 1) / kQsaCompress;
    const int ntok = (nblk <= kQsaTopBlocks ? nblk : sel_n[r]) * kQsaCompress + (p + 1 - nblk * kQsaCompress);
    const int nt = ((ntok + kTile - 1) / kTile + tpw - 1) / tpw;  // partials: one per tpw tiles
    const float *pb = part + (static_cast<size_t>(r * hkv + kvh) * kT * G + hh) * kPartStride;
    const size_t ts = static_cast<size_t>(G) * kPartStride;
    // loads first (8 tiles per round, dead slots re-read the last tile with weight 0): the
    // sum order is fixed, the loads overlap.
    float M = -INFINITY;
#pragma unroll 8
    for (int t = 0; t < nt; ++t) M = sycl::fmax(M, pb[t * ts + kD]);
    float L = 0.f, O = 0.f;
    for (int t0 = 0; t0 < nt; t0 += 8) {
      float mv[8], lv[8], ov[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int t = t0 + j < nt ? t0 + j : nt - 1;
        mv[j] = pb[t * ts + kD];
        lv[j] = pb[t * ts + kD + 1];
        ov[j] = pb[t * ts + d];
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float e = t0 + j < nt ? sycl::exp(mv[j] - M) : 0.f;
        L += lv[j] * e;
        O += ov[j] * e;
      }
    }
    const float a = bfr(O / L);
    const float gt = bfr(y[static_cast<size_t>(r) * ldy + h * 2 * kD + kD + d]);
    const float sg = bfr(1.f / (1.f + sycl::exp(-gt)));
    out[(static_cast<size_t>(r) * hq + h) * kD + d] = bf16(a * sg);
  });
}

// ---- the raw-key ring after a many-row prep: the chunk's last min(m, 16) rows ----
class QsaRingK;
sycl::event ring_tail(sycl::queue &q, int m, int ldi, const SegW &sw, const float *yi, bf16 *ring0) {
  const int r0 = m > kQsaRing ? m - kQsaRing : 0;
  const int32_t *rec = sw.rec;
  const int rstride = sw.stride;
  return q.parallel_for<QsaRingK>(sycl::nd_range<1>(static_cast<size_t>(m - r0) * kID, kID), [=](sycl::nd_item<1> it) {
    const int r = r0 + static_cast<int>(it.get_group(0)), d = static_cast<int>(it.get_local_id(0));
    const SegRow srw = seg_row(rec, rstride, 0, r);  // one segment: the chunk
    bf16 *ring = ring0 + static_cast<size_t>(srw.slot) * kQsaRing * kID;
    ring[(srw.pos % kQsaRing) * kID + d] = bf16(bfr(yi[static_cast<size_t>(r) * ldi + kQsaIdxHeads * kID + d]));
  });
}

// Tiles per prefill attention work-group; XEN_QSA_PF_TILES=1 selects the decode's tiling (the bits of
// qsa_rows == the decode path, test-qsa's contract).
int pf_tiles() {
  const char *e = std::getenv("XEN_QSA_PF_TILES");  // read per call (host, once per layer): a test switches it
  return e && std::atoi(e) == 1 ? 1 : kQsaPfTiles;
}

// The KV dtype's prep / attention instances (TREE: decode steps with tree segments).
template <bool TREE = false>
sycl::event prep_kv(sycl::queue &q, const QsaShape &s, int rows, const QsaWeights &w, const QsaCache &c, const bf16 *rope, const SegW &sw,
                    const QsaScratch &sc, int ring_from = 0) {
  switch (s.kv) {
    case QsaKv::i8: return prep<QsaKv::i8, TREE>(q, s, rows, w, c, rope, sw, sc, ring_from);
    case QsaKv::i8v4: return prep<QsaKv::i8v4, TREE>(q, s, rows, w, c, rope, sw, sc, ring_from);
    default: return prep<QsaKv::bf16, TREE>(q, s, rows, w, c, rope, sw, sc, ring_from);
  }
}
template <int G, bool TREE>
sycl::event attn_kv_g(sycl::queue &q, const QsaShape &s, int rows, const QsaCache &c, const SegW &sw, const QsaScratch &sc, int tpw) {
  if constexpr (!TREE) {
    if (tpw == kQsaPfTiles) {
      switch (s.kv) {
        case QsaKv::i8: return attn<QsaKv::i8, G, kQsaPfTiles>(q, s, rows, c, sw, sc);
        case QsaKv::i8v4: return attn<QsaKv::i8v4, G, kQsaPfTiles>(q, s, rows, c, sw, sc);
        default: return attn<QsaKv::bf16, G, kQsaPfTiles>(q, s, rows, c, sw, sc);
      }
    }
  }
  switch (s.kv) {
    case QsaKv::i8: return attn<QsaKv::i8, G, 1, TREE>(q, s, rows, c, sw, sc);
    case QsaKv::i8v4: return attn<QsaKv::i8v4, G, 1, TREE>(q, s, rows, c, sw, sc);
    default: return attn<QsaKv::bf16, G, 1, TREE>(q, s, rows, c, sw, sc);
  }
}
// a card's q heads per kv head: 12 (one / two cards), 6 (four cards)
template <bool TREE = false>
sycl::event attn_kv(sycl::queue &q, const QsaShape &s, int rows, const QsaCache &c, const SegW &sw, const QsaScratch &sc, int tpw = 1) {
  return s.lgrp() == 6 ? attn_kv_g<6, TREE>(q, s, rows, c, sw, sc, tpw) : attn_kv_g<12, TREE>(q, s, rows, c, sw, sc, tpw);
}
// XEN_DEC_PROFILE: the stage's stamp word (QsaScratch::stamps), when the caller asked for stamps (stamp_stage)
// sel_n_of_blocks: every row's selected-block count, min(complete blocks, 512)
sycl::event sel_n_of_blocks(sycl::queue &q, int rows, const int32_t *nblk, int32_t *sel_n) {
  return q.parallel_for<QsaSelNK>(sycl::range<1>(static_cast<size_t>(rows)),
                                  [=](sycl::id<1> i) { sel_n[i] = sycl::min(nblk[i], static_cast<int32_t>(kQsaTopBlocks)); });
}
void stamp_stage(sycl::queue &q, const QsaScratch &sc, QsaStage stage) {
  if (sc.stamps) stamp(q, sc.stamps + static_cast<int>(stage));
}
// qsa_decode's middle: prep, score, select, attention, merge (TREE: the step holds tree segments)
// every (QsaStep::every): no row selects (each has at most kQsaTopBlocks complete blocks), so the score
// and the select are not launched.
template <bool TREE>
void decode_middle(sycl::queue &q, const QsaShape &s, int rows, const QsaWeights &w, const QsaCache &c, const bf16 *rope, const SegW &sw,
                   const QsaScratch &sc, int blk_bound, int stride, bool every, QsaEvents &e) {
  e.prep = prep_kv<TREE>(q, s, rows, w, c, rope, sw, sc);
  stamp_stage(q, sc, QsaStage::prep);
  if (every) {
    e.score = e.select = e.merge_sel = e.prep;
    stamp_stage(q, sc, QsaStage::score);
    stamp_stage(q, sc, QsaStage::select);
  } else if (sc.sel_share && sc.sel_hi > sc.sel_lo) {
    // this card's rows of the step (owner_rows) scored and selected, their sel rows
    // shared, sel_n of every row from its complete blocks (a top-512 over n blocks keeps min(n, 512))
    const int lo = sc.sel_lo, n = sc.sel_hi - sc.sel_lo;
    e.score = score<TREE>(q, rows, c, sw, blk_bound, sc, stride, lo, sc.sel_hi);
    stamp_stage(q, sc, QsaStage::score);
    QsaScratch u = sc;
    u.nblk += lo, u.sel += static_cast<size_t>(lo) * kQsaTopBlocks, u.sel_n += lo;
    u.cand_key += static_cast<size_t>(lo) * kQsaMaxChunks * kQsaTopBlocks, u.cand_id += static_cast<size_t>(lo) * kQsaMaxChunks * kQsaTopBlocks;
    u.cand_n += static_cast<size_t>(lo) * kQsaMaxChunks;
    e.select = qsa_select(q, n, sc.score + static_cast<size_t>(lo) * stride, stride, u.nblk, std::max(1, blk_bound), u, &e.merge_sel, 0, s.card, s.cards);
    sel_n_of_blocks(q, rows, sc.nblk, sc.sel_n);
    owner_share(q, rows, 2 * kQsaTopBlocks, reinterpret_cast<bf16 *>(sc.sel), *sc.sel_share, sc.sel_parity);
    stamp_stage(q, sc, QsaStage::select);
  } else {
    e.score = score<TREE>(q, rows, c, sw, blk_bound, sc, stride);
    stamp_stage(q, sc, QsaStage::score);
    e.select = qsa_select(q, rows, sc.score, stride, sc.nblk, std::max(1, blk_bound), sc, &e.merge_sel, 0, s.card, s.cards);
    stamp_stage(q, sc, QsaStage::select);
  }
  e.attn = attn_kv<TREE>(q, s, rows, c, sw, sc);
  e.merge = merge<TREE>(q, s, rows, w.qkv.n, sw, sc);
  stamp_stage(q, sc, QsaStage::attention);
}
// The heads' split and the attention kernel's groups (12 or 6 q heads per card and kv head).
void check_heads(const QsaShape &s, const char *who) {
  if (!s.split_ok() || s.grp() != 12 || (s.lgrp() != 12 && s.lgrp() != 6))
    throw std::invalid_argument(std::string(who) + ": heads must split over the cards (whole kv heads per card, or a kv head's q heads over its cards); "
                                "the attention kernel is built for 12 q heads per kv head, 12 or 6 per card");
}

}  // namespace

const char *qsa_stage_name(QsaStage s) {
  static const char *n[] = {"projections", "prep", "score", "select", "attention", "o_proj + sum"};
  static_assert(std::size(n) == static_cast<size_t>(QsaStage::kCount), "a name per stage");
  const int i = static_cast<int>(s);
  return i >= 0 && i < static_cast<int>(QsaStage::kCount) ? n[i] : "?";
}

size_t qsa_page_kv_bytes(const QsaShape &s) { return static_cast<size_t>(s.hkv()) * kQsaPage * kD * s.kv_elem(); }
size_t qsa_page_v_bytes(const QsaShape &s) { return static_cast<size_t>(s.hkv()) * kQsaPage * s.v_row_bytes(); }
size_t qsa_page_scale_floats(const QsaShape &s) { return static_cast<size_t>(s.hkv()) * kQsaPage; }
size_t qsa_page_vscale_floats(const QsaShape &s) { return static_cast<size_t>(s.hkv()) * kQsaPage * s.v_scale_bytes() / 4; }
size_t qsa_page_kc_elems() { return static_cast<size_t>(kQsaPageBlocks) * kID; }
size_t qsa_score_words(int max_tokens, int rows) {
  const size_t blocks = (static_cast<size_t>(max_tokens) / kQsaCompress + 63) / 64 * 64;
  return static_cast<size_t>(rows) * blocks;
}
size_t qsa_part_floats(const QsaShape &s, int rows) { return static_cast<size_t>(rows) * s.hkv() * kT * s.lgrp() * kPartStride; }

QsaEvents qsa_decode(sycl::queue &q, const QsaShape &s, int rows, const bf16 *x, const QsaWeights &w, const QsaCache &c, const bf16 *rope,
                     const QsaStep &step, const QsaScratch &sc, const PairWire *wire, int wire_parity, bf16 *out, int qkv_ks, int o_ks) {
  if (rows < 1 || rows > kMaxRows) throw std::invalid_argument("qsa_decode: rows 1..8");
  check_heads(s, "qsa_decode");
  if (s.hidden % 64) throw std::invalid_argument("qsa_decode: hidden % 64 == 0");
  if (w.qkv.n < s.qkv_n() || w.qkv.k != s.hidden || w.idx.n < s.idx_n() || w.idx.k != s.hidden || w.o.n != s.hidden || w.o.k != s.o_k() ||
      !w.q_norm || !w.k_norm || !w.iq_norm || !w.ik_norm)
    throw std::invalid_argument("qsa_decode: weight shapes do not match the shape");
  if (c.max_tokens % kQsaPage || c.max_tokens > kQsaMaxTokens || step.pos_bound < rows || step.pos_bound > c.max_tokens)
    throw std::invalid_argument("qsa_decode: pos_bound must lie in [rows, cache.max_tokens], capacity a multiple of 64 up to 1,048,576");
  if ((s.cards > 1) != (wire != nullptr)) throw std::invalid_argument("qsa_decode: a wire is needed exactly with more than one card");
  if (qsa_kv_scaled(s.kv) && (!c.ks || !c.vs)) throw std::invalid_argument("qsa_decode: 8-bit KV needs scale pools");
  if (!seg_rows_ok(rows, step.seg_rows, step.stride) || (step.seg_rows > 0 && step.stride < kSegRec))
    throw std::invalid_argument("qsa_decode: segments need records (stride >= kSegRec) and 1..rows rows each");
  if (step.tree && (step.stride < kSegRec || !c.tkc || seg_size(rows, step.seg_rows) > kQsaTreeRows))
    throw std::invalid_argument("qsa_decode: a tree step needs segment records (stride >= kSegRec), the per-row compressed keys (cache.tkc) and "
                                "at most " + std::to_string(kQsaTreeRows) + " rows per segment (got " + std::to_string(seg_size(rows, step.seg_rows)) +
                                "): the 16-entry raw-key ring holds the row slots and the 3 committed keys a straddling block reads");
  if (step.every && step.pos_bound > kQsaTopBlocks * kQsaCompress)
    throw std::invalid_argument("qsa_decode: a step without a select holds rows of at most kQsaTopBlocks complete blocks (pos_bound <= 2,048)");
  const int stride = static_cast<int>(qsa_score_words(c.max_tokens) / kM), blk_bound = step.pos_bound / kQsaCompress;
  const SegW sw{step.pos, step.stride, step.seg_rows};
  QsaEvents e;
  e.qkv = dense_gemm(q, rows, x, w.qkv, sc.y, qkv_ks);
  e.idx = dense_gemm(q, rows, x, w.idx, sc.yi);
  stamp_stage(q, sc, QsaStage::projections);
  if (step.tree) decode_middle<true>(q, s, rows, w, c, rope, sw, sc, blk_bound, stride, step.every, e);
  else decode_middle<false>(q, s, rows, w, c, rope, sw, sc, blk_bound, stride, step.every, e);
  capture_qsa_sparse_attention(q, s, rows, c, step, sc, w.qkv.n, e.merge);
  e.o = dense_gemm(q, rows, sc.attn, w.o, sc.partial, o_ks);
  e.reduce = pair_reduce(q, rows, s.hidden, sc.partial, wire, wire_parity, out);
  return e;
}

namespace xrv {

QsaEvents sparse_attention(sycl::queue &q, const QsaShape &s, int rows, const QsaCache &c, const QsaStep &step, const QsaScratch &sc,
                           int projection_stride, int tiles_per_group) {
  if (rows < 1 || rows > kMaxRows) throw std::invalid_argument("qsa XRV sparse_attention: rows 1..8");
  check_heads(s, "qsa XRV sparse_attention");
  if (s.kv != QsaKv::i8v4) throw std::invalid_argument("qsa XRV sparse_attention: fixture adapter requires packed K8V4 cache");
  if (step.tree) throw std::invalid_argument("qsa XRV sparse_attention: tree rows are not supported by this fixture adapter");
  if (!step.pos || c.max_tokens % kQsaPage || c.max_tokens > kQsaMaxTokens || step.pos_bound < rows || step.pos_bound > c.max_tokens ||
      c.slot_pages < c.max_tokens / kQsaPage)
    throw std::invalid_argument("qsa XRV sparse_attention: invalid cache capacity or position bound");
  if (!c.k || !c.v || !c.pages || (qsa_kv_scaled(s.kv) && (!c.ks || !c.vs)))
    throw std::invalid_argument("qsa XRV sparse_attention: cache tensors are incomplete");
  if (!seg_rows_ok(rows, step.seg_rows, step.stride) || (step.seg_rows > 0 && step.stride < kSegRec))
    throw std::invalid_argument("qsa XRV sparse_attention: segments need records (stride >= kSegRec)");
  if (projection_stride < s.q_n() || tiles_per_group < 1 || tiles_per_group > kQsaTiles)
    throw std::invalid_argument("qsa XRV sparse_attention: invalid projection stride or tile grouping");
  if (!sc.q || !sc.y || !sc.sel || !sc.sel_n || !sc.part || !sc.attn)
    throw std::invalid_argument("qsa XRV sparse_attention: query, gate, selection, partial and output buffers are required");
  const SegW sw{step.pos, step.stride, step.seg_rows};
  QsaEvents e;
  e.attn = attn_kv(q, s, rows, c, sw, sc, tiles_per_group);
  e.merge = merge(q, s, rows, projection_stride, sw, sc, tiles_per_group);
  return e;
}

}  // namespace xrv

QsaEvents qsa_store(sycl::queue &q, const QsaShape &s, int rows, const bf16 *x, const QsaWeights &w, const QsaCache &c, const bf16 *rope,
                    const QsaStep &step, const QsaScratch &sc, int qkv_ks) {
  if (rows < 1 || rows > kMaxRows) throw std::invalid_argument("qsa_store: rows 1..8");
  if (w.qkv.n < s.qkv_n() || w.qkv.k != s.hidden || w.idx.n < s.idx_n() || w.idx.k != s.hidden || !w.q_norm || !w.k_norm || !w.iq_norm || !w.ik_norm)
    throw std::invalid_argument("qsa_store: weight shapes do not match the shape");
  if (c.max_tokens % kQsaPage || c.max_tokens > kQsaMaxTokens || step.pos_bound < rows || step.pos_bound > c.max_tokens)
    throw std::invalid_argument("qsa_store: pos_bound must lie in [rows, cache.max_tokens]");
  if (qsa_kv_scaled(s.kv) && (!c.ks || !c.vs)) throw std::invalid_argument("qsa_store: 8-bit KV needs scale pools");
  if (!seg_rows_ok(rows, step.seg_rows, step.stride) || (step.seg_rows > 0 && step.stride < kSegRec))
    throw std::invalid_argument("qsa_store: segments need records (stride >= kSegRec) and 1..rows rows each");
  if (step.tree) throw std::invalid_argument("qsa_store: tree steps are not stored without attention (the drafter's passes are chains)");
  const SegW sw{step.pos, step.stride, step.seg_rows};
  QsaEvents e;
  // the same launches qsa_decode starts with, on the same inputs: the same stored bits
  e.qkv = dense_gemm(q, rows, x, w.qkv, sc.y, qkv_ks);
  e.idx = dense_gemm(q, rows, x, w.idx, sc.yi);
  e.prep = prep_kv(q, s, rows, w, c, rope, sw, sc);
  return e;
}

// ---- the accepted path of a tree step to the chain's slots ----
namespace {
class QsaKvMoveK;
// one launch's caches (kernel arguments by value)
struct QsaMoveBufs {
  uint8_t *k[kQsaMoveCaches], *v[kQsaMoveCaches];
  float *ks[kQsaMoveCaches], *vs[kQsaMoveCaches];
  bf16 *kc[kQsaMoveCaches], *ring[kQsaMoveCaches];
  const bf16 *tkc[kQsaMoveCaches];
  const int32_t *pages[kQsaMoveCaches];
};
constexpr int kMoveChunk = 16;                                   // bytes per item and row
constexpr int kMoveItems = kD * 2 / kMoveChunk;                  // items per (cache, kv head): a bf16 K / V row
static_assert(kID * 2 / kMoveChunk <= kMoveItems, "a raw / compressed index key in one item group");
}  // namespace

sycl::event qsa_kv_move(sycl::queue &q, const QsaShape &s, const QsaCache *caches, int n_caches, int slot, int p0, int r0, const QsaMove &mv) {
  if (n_caches < 1 || !caches || slot < 0 || p0 < 0 || r0 < 0 || mv.n < 1 || mv.n > kQsaTreeRows)
    throw std::invalid_argument("qsa_kv_move: caches, a slot, p0 >= 0, r0 >= 0 and 1.." + std::to_string(kQsaTreeRows) + " accepted rows (got " +
                                std::to_string(mv.n) + ")");
  bool moves = false;
  for (int x = 0; x < mv.n; ++x) {
    if (mv.from[x] < 0 || mv.from[x] >= kQsaTreeRows || (x == 0 ? mv.from[0] != 0 : mv.from[x] <= mv.from[x - 1]))
      throw std::invalid_argument("qsa_kv_move: from[0] = 0, then strictly increasing rows below " + std::to_string(kQsaTreeRows) + " (from[" +
                                  std::to_string(x) + "] = " + std::to_string(mv.from[x]) + ")");
    moves |= mv.from[x] != x;
  }
  if (r0 + mv.from[mv.n - 1] >= kMaxRows) throw std::invalid_argument("qsa_kv_move: the path's step rows past kMaxRows (tkc)");
  for (int l = 0; l < n_caches; ++l) {
    const QsaCache &c = caches[l];
    if (!c.k || !c.v || !c.kc || !c.ring || !c.pages || !c.tkc || (qsa_kv_scaled(s.kv) && (!c.ks || !c.vs)) || p0 + mv.from[mv.n - 1] >= c.max_tokens)
      throw std::invalid_argument("qsa_kv_move: cache " + std::to_string(l) + " lacks a pool / ring / page table / tkc, or the path passes its capacity");
  }
  if (!moves) return {};  // the path is the main chain: every row already sits at its position
  const int hkv = s.hkv(), krow = kD * s.kv_elem(), vrow = s.v_row_bytes();
  const bool scaled = qsa_kv_scaled(s.kv), v4 = qsa_kv_v4(s.kv);
  sycl::event e;
  for (int l0 = 0; l0 < n_caches; l0 += kQsaMoveCaches) {
    const int nl = std::min(kQsaMoveCaches, n_caches - l0);
    QsaMoveBufs b{};
    for (int l = 0; l < nl; ++l) {
      const QsaCache &c = caches[l0 + l];
      b.k[l] = c.k, b.v[l] = c.v, b.ks[l] = c.ks, b.vs[l] = c.vs, b.kc[l] = c.kc, b.tkc[l] = c.tkc;
      b.ring[l] = c.ring + static_cast<size_t>(slot) * kQsaRing * kID;
      b.pages[l] = c.pages + static_cast<size_t>(slot) * c.slot_pages;
    }
    // item (cache, kv head, chunk): chunk c's 16 bytes of each moved row's K and V, its scales (c 0), and on kv head 0
    // the ring entry's / compressed key's chunk c. One item walks the rows in order (x ascending, see qsa.hpp).
    e = q.parallel_for<QsaKvMoveK>(sycl::range<3>(static_cast<size_t>(nl), static_cast<size_t>(hkv), kMoveItems), [=](sycl::item<3> it) {
      const int l = static_cast<int>(it[0]), h = static_cast<int>(it[1]), c = static_cast<int>(it[2]);
      const int32_t *pages = b.pages[l];
      auto slot_of = [&](int p) { return (static_cast<size_t>(pages[p / kQsaPage]) * hkv + h) * kQsaPage + p % kQsaPage; };
      using chunk = sycl::uint4;
      for (int x = 1; x < mv.n; ++x) {
        const int fx = mv.from[x];
        if (fx == x) continue;
        const size_t src = slot_of(p0 + fx), dst = slot_of(p0 + x);
        if (c * kMoveChunk < krow) reinterpret_cast<chunk *>(b.k[l] + dst * krow)[c] = reinterpret_cast<const chunk *>(b.k[l] + src * krow)[c];
        if (c * kMoveChunk < vrow) reinterpret_cast<chunk *>(b.v[l] + dst * vrow)[c] = reinterpret_cast<const chunk *>(b.v[l] + src * vrow)[c];
        if (scaled && c == 0) {
          b.ks[l][dst] = b.ks[l][src];
          if (v4) reinterpret_cast<chunk *>(b.vs[l])[dst] = reinterpret_cast<const chunk *>(b.vs[l])[src];  // 8 bf16 scales
          else b.vs[l][dst] = b.vs[l][src];
        }
        if (h == 0 && c * kMoveChunk < kID * 2) {
          auto *ring = reinterpret_cast<chunk *>(b.ring[l]);
          constexpr int kRow = kID * 2 / kMoveChunk;  // chunks per index key
          ring[((p0 + x) % kQsaRing) * kRow + c] = ring[((p0 + fx) % kQsaRing) * kRow + c];
          if ((p0 + x + 1) % kQsaCompress == 0) {
            const int blk = (p0 + x) / kQsaCompress;
            const size_t kr = static_cast<size_t>(b.pages[l][blk / kQsaPageBlocks]) * kQsaPageBlocks + blk % kQsaPageBlocks;
            reinterpret_cast<chunk *>(b.kc[l])[kr * kRow + c] = reinterpret_cast<const chunk *>(b.tkc[l])[static_cast<size_t>(r0 + fx) * kRow + c];
          }
        }
      }
    });
  }
  return e;
}

QsaEvents qsa_rows(sycl::queue &q, const QsaShape &s, int m, const bf16 *x, const QsaWeights &w, const QsaCache &c, const bf16 *rope,
                   const int32_t *words, int word_stride, int64_t pos0, const QsaScratch &sc, const QsaRowsScratch &rs, const PairRowsWire *wire,
                   int wire_parity, bf16 *out) {
  if (m < 1 || pos0 + m > c.max_tokens) throw std::invalid_argument("qsa_rows: rows / capacity");
  check_heads(s, "qsa_rows");
  if ((s.cards > 1) != (wire != nullptr)) throw std::invalid_argument("qsa_rows: a wire is needed exactly with more than one card");
  if (wire && wire->split && s.cards != 2 && s.cards != kWireCardsMax) throw std::invalid_argument("qsa_rows: the split sum is two- or four-card");
  if (wire && wire->split && wire->send && wire->send->sp && wire->send->sp->cards != s.cards)
    throw std::invalid_argument("qsa_rows: the split's cards differ from the shape's");
  if (!rs.q || !rs.iq || !rs.nblk || !rs.sel || !rs.sel_n || !rs.score || !rs.cand_key || !rs.cand_id || !rs.cand_n || !rs.part)
    throw std::invalid_argument("qsa_rows: the stage-B scratch is not allocated");
  static_assert(kQsaRowsBatch % kM == 0, "batches start on a 4-row slice");
  if (qsa_kv_scaled(s.kv) && (!c.ks || !c.vs)) throw std::invalid_argument("qsa_rows: 8-bit KV needs scale pools");
  // score rows spaced by this call's blocks, not the capacity's (the same bits)
  const int stride = static_cast<int>(qsa_score_words(static_cast<int>(pos0 + m)) / kM);
  QsaEvents e;
  // one launch when both bodies pair (m <= 16: direct qkv + one-tile idx; the same bits)
  e.idx = pf_dense_pair(q, m, w.qkv, x, s.hidden, PfOut{rs.y, nullptr, w.qkv.n}, w.idx, x, s.hidden, PfOut{rs.yi, nullptr, w.idx.n}, true, nullptr,
                        rs.a8.on());  // qkv on W8A8 when set, idx on bf16 weights
  e.qkv = e.idx;
  // prefill profiling (rs.trace): each launch's event under its stage
  auto traced = [&](QsaStage stage, const sycl::event &ev) {
    if (rs.trace) rs.trace->add(static_cast<int>(stage), ev);
    return ev;
  };
  traced(QsaStage::projections, e.idx);
  // prep over the chunk: every row's K / V / compressed keys are in the cache before any row reads
  QsaScratch v = sc;
  v.y = rs.y;
  v.yi = rs.yi;
  v.q = rs.q;
  v.iq = rs.iq;
  v.nblk = rs.nblk;
  // the chunk is one segment: slice 0's word (a record with word_stride >= kSegRec) names the slot
  const int rec_stride = word_stride >= kSegRec ? word_stride : 0;
  const SegW all{words, rec_stride, 0};
  e.prep = traced(QsaStage::prep, prep_kv(q, s, m, w, c, rope, all, v, m));
  traced(QsaStage::prep, ring_tail(q, m, w.idx.n, all, rs.yi, c.ring));
  // the batch's scratch: its rows of the per-row buffers, the shared per-batch ones
  auto batch_of = [&](int b) {
    QsaScratch u = v;
    u.y = rs.y + static_cast<size_t>(b) * w.qkv.n;
    u.q = rs.q + static_cast<size_t>(b) * s.hq() * kD;
    u.iq = rs.iq + static_cast<size_t>(b) * kQsaIdxHeads * kID;
    u.nblk = rs.nblk + b;
    u.sel = rs.sel + static_cast<size_t>(b) * kQsaTopBlocks;
    u.sel_n = rs.sel_n + b;
    u.attn = rs.attn + static_cast<size_t>(b) * s.o_k();
    u.score = rs.score;
    u.cand_key = rs.cand_key;
    u.cand_id = rs.cand_id;
    u.cand_n = rs.cand_n;
    u.part = rs.part;
    return u;
  };
  // score + select of rows [b, b + rows) (only when a row of them has more than 512 complete blocks: below that the
  // selection is every block, which attn / merge take directly); returns whether it launched
  auto select_rows = [&](int b, int rows) {
    const SegW pos{words + static_cast<size_t>(b / kM) * word_stride, rec_stride, 0};  // == pos0 + b
    const QsaScratch u = batch_of(b);
    const int blk_bound = static_cast<int>((pos0 + b + rows) / kQsaCompress);  // the batch's largest nblk
    if (blk_bound <= kQsaTopBlocks) return false;
    e.score = traced(QsaStage::score, score<false>(q, rows, c, pos, blk_bound, u, stride));
    // prefill batches: one select launch up to kQsaPfChunk blocks (no merge launch); past it kQsaChunk-block chunks.
    // An exact top-k: the same bits either way.
    const int pf_chunk = blk_bound <= kQsaPfChunk ? kQsaPfChunk : kQsaChunk;
    e.select = traced(QsaStage::select, qsa_select(q, rows, u.score, stride, u.nblk, blk_bound, u, &e.merge_sel, pf_chunk, s.card, s.cards));
    if (e.merge_sel != e.select) traced(QsaStage::select, e.merge_sel);  // the chunked select's merge (direct: the same launch)
    return true;
  };
  // the selects of this card's rows first, then the caller's share, then every row's attention
  const bool owner = rs.between && rs.sel_hi > rs.sel_lo;
  if (owner) {
    if (rs.sel_lo % kM || rs.sel_lo < 0 || rs.sel_hi > m) throw std::invalid_argument("qsa_rows: the owner rows [sel_lo, sel_hi) of the call (sel_lo on a 4-row slice)");
    bool any = false;
    for (int b = rs.sel_lo; b < rs.sel_hi; b += kQsaRowsBatch) any |= select_rows(b, std::min(kQsaRowsBatch, rs.sel_hi - b));
    (*rs.between)(any ? &e.merge_sel : nullptr);
  }
  for (int b = 0; b < m; b += kQsaRowsBatch) {
    const int rows = std::min(kQsaRowsBatch, m - b);
    const SegW pos{words + static_cast<size_t>(b / kM) * word_stride, rec_stride, 0};  // == pos0 + b
    if (!owner) select_rows(b, rows);
    const QsaScratch u = batch_of(b);
    // the prefill walks pf_tiles() tiles per work-group (online softmax)
    const int tpw = pf_tiles();
    e.attn = traced(QsaStage::attention, attn_kv(q, s, rows, c, pos, u, tpw));
    e.merge = traced(QsaStage::attention, merge(q, s, rows, w.qkv.n, pos, u, tpw));
  }
  if (wire && wire->split && wire->send) {
    // the peer's rows first, their copy beside the own rows
    const int H = s.hidden, K = s.o_k();
    // (four cards = every other card's rows, [0, lo) and [hi, m); two: the peer's, the same launch)
    auto o_rows = [&](int r0, int rn) {
      return traced(QsaStage::out, pf_dense(q, rn, w.o, rs.attn + static_cast<size_t>(r0) * K, K, PfOut{rs.partial + static_cast<size_t>(r0) * H, nullptr, H}, rs.a8.on()));
    };
    int lo = 0, hi = 0;
    pair_split_owner_rows(s.card, m, &lo, &hi, wire->send->sp->cards);
    if (lo > 0) o_rows(0, lo);
    if (hi < m) o_rows(hi, m - hi);
    e.send = traced(QsaStage::out, pair_split_send(q, wire->send->side, s.card, m, H, rs.partial, *wire->send->sp, wire->send->seq));
    if (hi > lo) e.o = o_rows(lo, hi - lo);
  } else {
    e.o = traced(QsaStage::out, pf_dense(q, m, w.o, rs.attn, s.o_k(), PfOut{rs.partial, nullptr, s.hidden}, rs.a8.on()));
  }
  if (!wire || !wire->split) e.reduce = traced(QsaStage::out, pair_reduce_rows(q, m, s.hidden, rs.partial, wire, wire_parity, out));
  return e;
}

}  // namespace xenithium::kernels
