#include "qsa_capture.hpp"

#include <atomic>
#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <set>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include "xenithium/kernels/attention/qsa.hpp"

namespace xenithium::xrv {
namespace {
namespace k = xenithium::kernels;

struct CaptureConfig {
  std::filesystem::path root;
  int min_bound = 1;
  int rank = 0;
  int expected_tp_size = 1;
  size_t max_bytes = 256u * 1024u * 1024u;
  std::string model_fingerprint;
  std::string engine_revision;
  std::string kernel = "qsa.select";
  std::atomic<bool> captured{false};
};

bool env_int(const char *name, int fallback, int min, int max, int &out) {
  const char *v = std::getenv(name);
  if (!v || !*v) { out = fallback; return true; }
  char *end = nullptr;
  const long n = std::strtol(v, &end, 10);
  if (!end || *end || n < min || n > max) return false;
  out = static_cast<int>(n);
  return true;
}

bool env_size(const char *name, size_t fallback, size_t &out) {
  const char *v = std::getenv(name);
  if (!v || !*v) { out = fallback; return true; }
  char *end = nullptr;
  const unsigned long long n = std::strtoull(v, &end, 10);
  if (!end || *end || n == 0 || n > std::numeric_limits<size_t>::max()) return false;
  out = static_cast<size_t>(n);
  return true;
}

std::string json_escape(const std::string &s) {
  std::string out;
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) { char b[7]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
        else out += static_cast<char>(c);
    }
  }
  return out;
}

template <typename T>
void write_raw(const std::filesystem::path &path, const std::vector<T> &values) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.exceptions(std::ios::badbit | std::ios::failbit);
  f.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
}

std::string utc_now() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char b[32];
  std::strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return b;
}

void capture_qsa_select(sycl::queue &q, const k::QsaSelectCaptureView &v, void *opaque) noexcept {
  auto &cfg = *static_cast<CaptureConfig *>(opaque);
  if (v.tp_rank != cfg.rank || v.tp_size != cfg.expected_tp_size || v.bound < cfg.min_bound) return;
  bool expected = false;
  if (!cfg.captured.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;
  try {
    if (std::endian::native != std::endian::little) throw std::runtime_error("XRV raw capture currently requires a little-endian host");
    if (v.rows < 1 || v.rows > 64 || v.bound < 1 || v.bound > k::kQsaMaxBlocks || v.stride < v.bound ||
        !v.scores || !v.nblk || !v.selected_ids || !v.selected_counts)
      throw std::runtime_error("selector dimensions or buffers are outside XRV capture limits");
    const uint64_t score_bytes = static_cast<uint64_t>(v.rows) * static_cast<uint64_t>(v.bound) * sizeof(uint32_t);
    const uint64_t output_bytes = static_cast<uint64_t>(v.rows) * (k::kQsaTopBlocks + 1) * sizeof(int32_t);
    if (score_bytes + output_bytes > cfg.max_bytes) throw std::runtime_error("capture exceeds XEN_XRV_CAPTURE_MAX_BYTES");
    // The queue may be out-of-order; capture is opt-in and outside timing, so wait for selection and merge before readback.
    auto complete = v.complete;
    complete.wait_and_throw();
    std::vector<int32_t> counts(static_cast<size_t>(v.rows));
    q.memcpy(counts.data(), v.nblk, counts.size() * sizeof(int32_t)).wait_and_throw();
    for (int r = 0; r < v.rows; ++r)
      if (counts[static_cast<size_t>(r)] < 0 || counts[static_cast<size_t>(r)] > v.bound)
        throw std::runtime_error("selector nblk exceeds its bound");
    std::vector<uint32_t> scores(static_cast<size_t>(v.rows) * v.bound, 0);
    std::vector<int32_t> ids(static_cast<size_t>(v.rows) * k::kQsaTopBlocks, -1);
    std::vector<int32_t> selected_counts(static_cast<size_t>(v.rows), 0);
    q.memcpy(selected_counts.data(), v.selected_counts, selected_counts.size() * sizeof(int32_t)).wait_and_throw();
    for (int r = 0; r < v.rows; ++r) {
      const int n = counts[static_cast<size_t>(r)];
      const int selected = selected_counts[static_cast<size_t>(r)];
      if (selected != std::min(n, k::kQsaTopBlocks))
        throw std::runtime_error("selector output count differs from min(nblk,512)");
      if (n > 0) q.memcpy(scores.data() + static_cast<size_t>(r) * v.bound,
                          v.scores + static_cast<size_t>(r) * v.stride, static_cast<size_t>(n) * sizeof(uint32_t));
      if (selected > 0) q.memcpy(ids.data() + static_cast<size_t>(r) * k::kQsaTopBlocks,
                                 v.selected_ids + static_cast<size_t>(r) * k::kQsaTopBlocks,
                                 static_cast<size_t>(selected) * sizeof(int32_t));
    }
    q.wait_and_throw();

    std::filesystem::create_directories(cfg.root);
    const auto dir = cfg.root / "capture-000001";
    std::filesystem::create_directories(dir);
    write_raw(dir / "scores.u32le", scores);
    write_raw(dir / "nblk.i32le", counts);
    write_raw(dir / "candidate_ids.i32le", ids);
    write_raw(dir / "candidate_counts.i32le", selected_counts);
    const auto device = q.get_device();
    const std::string name = device.get_info<sycl::info::device::name>();
    const std::string driver = device.get_info<sycl::info::device::driver_version>();
    const std::string version = device.get_info<sycl::info::device::version>();
    const uint32_t vendor = device.get_info<sycl::info::device::vendor_id>();
#ifdef __INTEL_LLVM_COMPILER
    const int compiler_version = __INTEL_LLVM_COMPILER;
#else
    const int compiler_version = 0;
#endif
    std::ofstream meta(dir / "capture.json");
    meta.exceptions(std::ios::badbit | std::ios::failbit);
    meta << "{\n  \"rows\": " << v.rows << ",\n  \"stride\": " << v.bound << ",\n  \"source_stride\": " << v.stride << ",\n  \"bound\": " << v.bound
         << ",\n  \"chunk\": " << v.chunk << ",\n  \"tp_rank\": " << v.tp_rank << ",\n  \"tp_size\": " << v.tp_size << ",\n  \"position\": null,\n  \"captured_utc\": \"" << utc_now()
         << "\",\n  \"engine_revision\": \"" << json_escape(cfg.engine_revision) << "\",\n  \"model_fingerprint\": \"" << json_escape(cfg.model_fingerprint)
         << "\",\n  \"runtime_fingerprint\": {\"backend\": \"SYCL\", \"device\": \"" << json_escape(name) << "\", \"driver\": \""
         << json_escape(driver) << "\", \"device_version\": \"" << json_escape(version) << "\", \"vendor_id\": " << vendor
         << ", \"compiler\": \"IntelLLVM\", \"compiler_version_code\": " << compiler_version << "}\n}\n";
    std::fprintf(stderr, "XRV captured qsa.select rows=%d bound=%d TP=%d/%d at %s\n", v.rows, v.bound, v.tp_rank, v.tp_size, dir.c_str());
  } catch (const std::exception &e) {
    cfg.captured.store(false, std::memory_order_release);
    std::fprintf(stderr, "XRV qsa.select capture failed: %s\n", e.what());
  } catch (...) {
    cfg.captured.store(false, std::memory_order_release);
    std::fprintf(stderr, "XRV qsa.select capture failed: unknown error\n");
  }
}


void capture_qsa_sparse_attention(sycl::queue &q, const k::QsaSparseAttentionCaptureView &v, void *opaque) noexcept {
  auto &cfg = *static_cast<CaptureConfig *>(opaque);
  if (v.tp_rank != cfg.rank || v.tp_size != cfg.expected_tp_size || v.shape.kv != k::QsaKv::i8v4) return;
  bool expected = false;
  if (!cfg.captured.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;
  try {
    if (std::endian::native != std::endian::little) throw std::runtime_error("XRV raw capture requires a little-endian host");
    if (v.step.tree || v.rows < 1 || v.rows > 8 || !v.step.pos || !v.cache.k || !v.cache.v || !v.cache.ks || !v.cache.vs ||
        !v.cache.pages || v.cache.max_tokens < k::kQsaPage ||
        (v.cache.slot_pages != 0 && v.cache.slot_pages < v.cache.max_tokens / k::kQsaPage) ||
        !v.scratch.q || !v.scratch.y || !v.scratch.sel || !v.scratch.sel_n || !v.scratch.attn)
      throw std::runtime_error("attention capture supports non-tree K8V4 decode rows with complete cache and scratch buffers");
    const int effective_slot_pages = v.cache.slot_pages == 0 ? v.cache.max_tokens / k::kQsaPage : v.cache.slot_pages;
    auto complete = v.complete;
    complete.wait_and_throw();

    const int record_stride = v.step.stride;
    const int segments = k::seg_count(v.rows, v.step.seg_rows);
    const size_t record_words = record_stride >= k::kSegRec ? static_cast<size_t>(segments) * record_stride : 1;
    std::vector<int32_t> records(record_words);
    q.memcpy(records.data(), v.step.pos, records.size() * sizeof(int32_t)).wait_and_throw();
    std::vector<int32_t> positions(v.rows), original_slots(v.rows);
    std::vector<int32_t> selection(static_cast<size_t>(v.rows) * k::kQsaTopBlocks, -1);
    std::vector<int32_t> selection_counts(v.rows, 0);
    std::vector<int32_t> device_counts(v.rows, 0);
    std::vector<sycl::event> selector_copies;

    int max_blocks = 0;
    std::set<int> unique_slots;
    for (int r = 0; r < v.rows; ++r) {
      const k::SegRow sr = k::seg_row(records.data(), record_stride, v.step.seg_rows, r);
      if (sr.pos < 0 || sr.pos >= v.cache.max_tokens || sr.slot < 0 || (v.cache.slot_pages == 0 && sr.slot != 0))
        throw std::runtime_error("attention capture row position or slot is invalid for this cache layout");
      positions[r] = sr.pos;
      original_slots[r] = sr.slot;
      unique_slots.insert(sr.slot);
      const int nblk = (sr.pos + 1) / k::kQsaCompress;
      max_blocks = std::max(max_blocks, nblk);
      const int count = std::min(nblk, k::kQsaTopBlocks);
      selection_counts[r] = count;
      const size_t row_offset = static_cast<size_t>(r) * k::kQsaTopBlocks;
      if (nblk <= k::kQsaTopBlocks) {
        for (int j = 0; j < count; ++j) selection[row_offset + j] = j;
      } else {
        selector_copies.push_back(q.memcpy(selection.data() + row_offset, v.scratch.sel + row_offset,
                                           static_cast<size_t>(count) * sizeof(int32_t)));
        selector_copies.push_back(q.memcpy(&device_counts[r], v.scratch.sel_n + r, sizeof(int32_t)));
      }
    }
    for (auto &copy : selector_copies) copy.wait_and_throw();
    for (int r = 0; r < v.rows; ++r) {
      if (selection_counts[r] < k::kQsaTopBlocks) continue;
      if (device_counts[r] != selection_counts[r]) throw std::runtime_error("attention capture selector count differs from min(nblk,512)");
      const size_t row_offset = static_cast<size_t>(r) * k::kQsaTopBlocks;
      for (int j = 0; j < selection_counts[r]; ++j) {
        const int id = selection[row_offset + j];
        if (id < 0 || id >= (positions[r] + 1) / k::kQsaCompress)
          throw std::runtime_error("attention capture selector ID is outside the row block bound");
        if (j > 0 && selection[row_offset + j - 1] >= id)
          throw std::runtime_error("attention capture selector IDs are not unique ascending logical blocks");
      }
    }
    if (max_blocks < cfg.min_bound) {
      cfg.captured.store(false, std::memory_order_release);
      return;
    }

    std::vector<int32_t> compact_slots(unique_slots.begin(), unique_slots.end());
    std::map<int, int> slot_map, page_map;
    for (size_t i = 0; i < compact_slots.size(); ++i) slot_map.emplace(compact_slots[i], static_cast<int>(i));
    std::vector<std::vector<int32_t>> source_tables(compact_slots.size(), std::vector<int32_t>(effective_slot_pages));
    for (size_t i = 0; i < compact_slots.size(); ++i)
      q.memcpy(source_tables[i].data(), v.cache.pages + static_cast<size_t>(compact_slots[i]) * effective_slot_pages,
               static_cast<size_t>(effective_slot_pages) * sizeof(int32_t)).wait_and_throw();

    std::set<int> physical_pages;
    for (int r = 0; r < v.rows; ++r) {
      const int nblk = (positions[r] + 1) / k::kQsaCompress;
      const int count = selection_counts[r];
      std::set<int> logical;
      for (int j = 0; j < count; ++j) {
        const int block = selection[static_cast<size_t>(r) * k::kQsaTopBlocks + j];
        for (int t = 0; t < k::kQsaCompress; ++t) logical.insert((block * k::kQsaCompress + t) / k::kQsaPage);
      }
      for (int t = nblk * k::kQsaCompress; t <= positions[r]; ++t) logical.insert(t / k::kQsaPage);
      const int slot_index = slot_map.at(original_slots[r]);
      for (int lp : logical) {
        if (lp < 0 || lp >= effective_slot_pages) throw std::runtime_error("selected logical page is outside the slot page table");
        const int physical = source_tables[static_cast<size_t>(slot_index)][static_cast<size_t>(lp)];
        if (physical < 0) throw std::runtime_error("attention page table contains a negative physical page");
        physical_pages.insert(physical);
      }
    }
    if (physical_pages.empty()) throw std::runtime_error("attention capture selected no cache pages");
    int compact_page = 0;
    for (int page : physical_pages) page_map.emplace(page, compact_page++);
    const int pool_pages = static_cast<int>(physical_pages.size());
    const int table_slots = static_cast<int>(compact_slots.size());
    const size_t page_k_bytes = k::qsa_page_kv_bytes(v.shape), page_v_bytes = k::qsa_page_v_bytes(v.shape);
    const size_t page_ks_count = k::qsa_page_scale_floats(v.shape);
    const size_t page_vs_u16_count = k::qsa_page_vscale_floats(v.shape) * sizeof(float) / sizeof(uint16_t);
    std::vector<uint8_t> packed_k(static_cast<size_t>(pool_pages) * page_k_bytes);
    std::vector<uint8_t> packed_v(static_cast<size_t>(pool_pages) * page_v_bytes);
    std::vector<float> k_scales(static_cast<size_t>(pool_pages) * page_ks_count);
    std::vector<uint16_t> v_scales(static_cast<size_t>(pool_pages) * page_vs_u16_count);
    for (const auto &[physical, compact] : page_map) {
      q.memcpy(packed_k.data() + static_cast<size_t>(compact) * page_k_bytes, v.cache.k + static_cast<size_t>(physical) * page_k_bytes, page_k_bytes);
      q.memcpy(packed_v.data() + static_cast<size_t>(compact) * page_v_bytes, v.cache.v + static_cast<size_t>(physical) * page_v_bytes, page_v_bytes);
      q.memcpy(k_scales.data() + static_cast<size_t>(compact) * page_ks_count, v.cache.ks + static_cast<size_t>(physical) * page_ks_count,
               page_ks_count * sizeof(float));
      q.memcpy(v_scales.data() + static_cast<size_t>(compact) * page_vs_u16_count,
               reinterpret_cast<const uint16_t *>(v.cache.vs) + static_cast<size_t>(physical) * page_vs_u16_count,
               page_vs_u16_count * sizeof(uint16_t));
    }

    const int slot_pages = effective_slot_pages;
    std::vector<int32_t> page_table(static_cast<size_t>(table_slots) * slot_pages, 0);
    for (int slot = 0; slot < table_slots; ++slot) {
      for (int lp = 0; lp < slot_pages; ++lp) {
        const int physical = source_tables[static_cast<size_t>(slot)][static_cast<size_t>(lp)];
        const auto mapped = page_map.find(physical);
        if (mapped != page_map.end()) page_table[static_cast<size_t>(slot) * slot_pages + lp] = mapped->second;
      }
    }
    std::vector<int32_t> slots(v.rows);
    for (int r = 0; r < v.rows; ++r) slots[r] = slot_map.at(original_slots[r]);

    const int hq = v.shape.hq(), qdim = k::kQsaDim;
    std::vector<uint16_t> query(static_cast<size_t>(v.rows) * hq * qdim);
    std::vector<uint16_t> gate(query.size());
    std::vector<uint16_t> candidate(query.size());
    q.memcpy(query.data(), v.scratch.q, query.size() * sizeof(uint16_t));
    q.memcpy(candidate.data(), v.scratch.attn, candidate.size() * sizeof(uint16_t)).wait_and_throw();
    std::vector<float> gate_projection(static_cast<size_t>(v.rows) * v.projection_stride);
    q.memcpy(gate_projection.data(), v.scratch.y, gate_projection.size() * sizeof(float));
    q.wait_and_throw();
    for (int r = 0; r < v.rows; ++r)
      for (int h = 0; h < hq; ++h)
        for (int d = 0; d < qdim; ++d) {
          const float value = gate_projection[static_cast<size_t>(r) * v.projection_stride + h * 2 * qdim + qdim + d];
          const uint32_t bits = std::bit_cast<uint32_t>(value);
          gate[(static_cast<size_t>(r) * hq + h) * qdim + d] =
              static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
        }

    uint64_t bytes = query.size() * sizeof(uint16_t) * 3 + positions.size() * sizeof(int32_t) * 2 +
                     page_table.size() * sizeof(int32_t) + selection.size() * sizeof(int32_t) +
                     selection_counts.size() * sizeof(int32_t) + packed_k.size() + packed_v.size() +
                     k_scales.size() * sizeof(float) + v_scales.size() * sizeof(uint16_t);
    if (bytes > cfg.max_bytes) throw std::runtime_error("compact attention capture exceeds XEN_XRV_CAPTURE_MAX_BYTES");

    std::filesystem::create_directories(cfg.root);
    const auto dir = cfg.root / "capture-000001";
    std::filesystem::create_directories(dir);
    write_raw(dir / "query.u16le", query);
    write_raw(dir / "gate.u16le", gate);
    write_raw(dir / "positions.i32le", positions);
    write_raw(dir / "slots.i32le", slots);
    write_raw(dir / "page_table.i32le", page_table);
    write_raw(dir / "keys.u8", packed_k);
    write_raw(dir / "values.u8", packed_v);
    write_raw(dir / "key_scales.f32le", k_scales);
    write_raw(dir / "value_scales.u16le", v_scales);
    write_raw(dir / "selected_ids.i32le", selection);
    write_raw(dir / "selected_counts.i32le", selection_counts);
    write_raw(dir / "candidate_output.u16le", candidate);

    const auto device = q.get_device();
    const std::string name = device.get_info<sycl::info::device::name>();
    const std::string driver = device.get_info<sycl::info::device::driver_version>();
    const uint32_t vendor = device.get_info<sycl::info::device::vendor_id>();
#ifdef __INTEL_LLVM_COMPILER
    const int compiler_version = __INTEL_LLVM_COMPILER;
#else
    const int compiler_version = 0;
#endif
    std::ofstream meta(dir / "capture.json");
    meta.exceptions(std::ios::badbit | std::ios::failbit);
    meta << "{\n  \"kernel\": \"qsa.sparse_attention.k8v4\",\n  \"rows\": " << v.rows
         << ",\n  \"q_heads\": " << v.shape.q_heads << ",\n  \"kv_heads\": " << v.shape.kv_heads
         << ",\n  \"cards\": " << v.shape.cards << ",\n  \"tp_rank\": " << v.tp_rank << ",\n  \"tp_size\": " << v.tp_size
         << ",\n  \"capacity\": " << v.cache.max_tokens << ",\n  \"pool_pages\": " << pool_pages
         << ",\n  \"table_slots\": " << table_slots << ",\n  \"slot_pages\": " << slot_pages
         << ",\n  \"tiles_per_group\": 1,\n  \"source_projection_stride\": " << v.projection_stride
         << ",\n  \"runtime_fingerprint\": {\"backend\":\"SYCL\",\"device\":\"" << json_escape(name)
         << "\",\"driver\":\"" << json_escape(driver) << "\",\"vendor_id\":" << vendor << ",\"compiler\":\"IntelLLVM\",\"compiler_version_code\":"
         << compiler_version << "},\n  \"engine_revision\":\"" << json_escape(cfg.engine_revision)
         << "\",\n  \"model_fingerprint\":\"" << json_escape(cfg.model_fingerprint)
         << "\",\n  \"captured_utc\":\"" << utc_now() << "\"\n}\n";
    std::fprintf(stderr,"XRV captured qsa.sparse_attention.k8v4 rows=%d max_blocks=%d pages=%d TP=%d/%d at %s\n",
                 v.rows,max_blocks,pool_pages,v.tp_rank,v.tp_size,dir.c_str());
  } catch (const std::exception &e) {
    cfg.captured.store(false,std::memory_order_release);
    std::fprintf(stderr,"XRV sparse attention capture failed: %s\n",e.what());
  } catch (...) {
    cfg.captured.store(false,std::memory_order_release);
    std::fprintf(stderr,"XRV sparse attention capture failed: unknown error\n");
  }
}
}  // namespace

bool install_probe_qsa_capture_from_env() noexcept {
  const char *dir = std::getenv("XEN_XRV_CAPTURE_DIR");
  if (!dir || !*dir) return true;
  static CaptureConfig cfg;
  try {
    cfg.root = std::filesystem::path(dir);
    if (!env_int("XEN_XRV_CAPTURE_MIN_BOUND", 1, 0, k::kQsaMaxBlocks, cfg.min_bound) ||
        !env_int("XEN_XRV_CAPTURE_TP_RANK", 0, 0, 64, cfg.rank) ||
        !env_int("XEN_XRV_CAPTURE_TP_SIZE", 1, 1, 64, cfg.expected_tp_size) ||
        !env_size("XEN_XRV_CAPTURE_MAX_BYTES", 256u * 1024u * 1024u, cfg.max_bytes))
      throw std::runtime_error("invalid XEN_XRV_CAPTURE_* numeric setting");
    if (const char *x = std::getenv("XEN_XRV_MODEL_FINGERPRINT")) cfg.model_fingerprint = x;
    if (const char *x = std::getenv("XEN_XRV_ENGINE_REVISION")) cfg.engine_revision = x;
    if (const char *x = std::getenv("XEN_XRV_CAPTURE_KERNEL")) cfg.kernel = x;
    if (cfg.kernel == "qsa.select") {
      k::qsa_set_select_capture_hook(&capture_qsa_select, &cfg);
      std::fprintf(stderr, "XRV selector capture enabled: rank=%d min_bound=%d max_bytes=%zu\n", cfg.rank, cfg.min_bound, cfg.max_bytes);
    } else if (cfg.kernel == "qsa.sparse_attention.k8v4") {
      k::qsa_set_sparse_attention_capture_hook(&capture_qsa_sparse_attention, &cfg);
      std::fprintf(stderr, "XRV K8V4 attention capture enabled: rank=%d min_bound=%d max_bytes=%zu\n", cfg.rank, cfg.min_bound, cfg.max_bytes);
    } else {
      throw std::runtime_error("unsupported XEN_XRV_CAPTURE_KERNEL");
    }
    return true;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "XRV capture setup failed: %s\n", e.what());
    return false;
  }
}

}  // namespace xenithium::xrv
