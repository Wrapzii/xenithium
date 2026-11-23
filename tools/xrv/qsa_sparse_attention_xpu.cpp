#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

#include "xenithium/kernels/attention/qsa_xrv.hpp"
#include "xenithium/kernels/rows.hpp"

namespace k = xenithium::kernels;
using clock_type = std::chrono::steady_clock;

struct Args {
  std::string query, gate, positions, slots, page_table, keys, values, key_scales, value_scales, selected_ids, selected_counts, output;
  int rows = 0, q_heads = 0, kv_heads = 0, card = 0, cards = 0, capacity = 0, pool_pages = 0, table_slots = 0, slot_pages = 0;
  int projection_stride = 0, tiles = 1, tp_rank = 0, tp_size = 1, device = 0, warmup = 5, repeat = 30;
};

int parse_int(const char *text, const char *name) {
  size_t used = 0;
  const int value = std::stoi(text, &used);
  if (text[used] != '\0') throw std::invalid_argument(std::string("invalid integer for ") + name);
  return value;
}

Args parse(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&]() -> const char * {
      if (++i >= argc) throw std::invalid_argument("missing value for " + key);
      return argv[i];
    };
    if (key == "--query") a.query = value();
    else if (key == "--gate") a.gate = value();
    else if (key == "--positions") a.positions = value();
    else if (key == "--slots") a.slots = value();
    else if (key == "--page-table") a.page_table = value();
    else if (key == "--keys") a.keys = value();
    else if (key == "--values") a.values = value();
    else if (key == "--key-scales") a.key_scales = value();
    else if (key == "--value-scales") a.value_scales = value();
    else if (key == "--selected-ids") a.selected_ids = value();
    else if (key == "--selected-counts") a.selected_counts = value();
    else if (key == "--output") a.output = value();
    else if (key == "--rows") a.rows = parse_int(value(), "--rows");
    else if (key == "--q-heads") a.q_heads = parse_int(value(), "--q-heads");
    else if (key == "--kv-heads") a.kv_heads = parse_int(value(), "--kv-heads");
    else if (key == "--card") a.card = parse_int(value(), "--card");
    else if (key == "--cards") a.cards = parse_int(value(), "--cards");
    else if (key == "--capacity") a.capacity = parse_int(value(), "--capacity");
    else if (key == "--pool-pages") a.pool_pages = parse_int(value(), "--pool-pages");
    else if (key == "--table-slots") a.table_slots = parse_int(value(), "--table-slots");
    else if (key == "--slot-pages") a.slot_pages = parse_int(value(), "--slot-pages");
    else if (key == "--projection-stride") a.projection_stride = parse_int(value(), "--projection-stride");
    else if (key == "--tiles") a.tiles = parse_int(value(), "--tiles");
    else if (key == "--tp-rank") a.tp_rank = parse_int(value(), "--tp-rank");
    else if (key == "--tp-size") a.tp_size = parse_int(value(), "--tp-size");
    else if (key == "--device") a.device = parse_int(value(), "--device");
    else if (key == "--warmup") a.warmup = parse_int(value(), "--warmup");
    else if (key == "--repeat") a.repeat = parse_int(value(), "--repeat");
    else if (key == "--help") {
      std::puts("xenithium-xrv-qsa-sparse-attention --query FILE --gate FILE --positions FILE --slots FILE --page-table FILE --keys FILE --values FILE --key-scales FILE --value-scales FILE --selected-ids FILE --selected-counts FILE --output DIR --rows N --q-heads N --kv-heads N --card N --cards N --capacity N --pool-pages N --table-slots N --slot-pages N [--projection-stride N --tiles N --tp-rank N --tp-size N --device N --warmup N --repeat N]");
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + key);
    }
  }
  const std::string paths[] = {a.query, a.gate, a.positions, a.slots, a.page_table, a.keys, a.values, a.key_scales,
                               a.value_scales, a.selected_ids, a.selected_counts, a.output};
  for (const auto &path : paths)
    if (path.empty()) throw std::invalid_argument("all fixture paths and --output are required");
  if (a.rows < 1 || a.rows > k::kMaxRows || a.q_heads < 1 || a.kv_heads < 1 || a.q_heads % a.kv_heads || a.q_heads % a.cards)
    throw std::invalid_argument("invalid rows or head topology");
  if (a.cards < 1 || a.card < 0 || a.card >= a.cards || a.tp_size < 1 || a.tp_rank < 0 || a.tp_rank >= a.tp_size)
    throw std::invalid_argument("invalid TP/card rank");
  if (a.capacity < 64 || a.capacity % 64 || a.pool_pages < 1 || a.table_slots < 1 ||
      a.slot_pages < a.capacity / 64 || a.pool_pages > 65536)
    throw std::invalid_argument("invalid cache/page-table dimensions");
  if (a.projection_stride < 0 || a.tiles < 1 || a.tiles > k::kQsaTiles || a.device < 0 || a.warmup < 0 || a.repeat < 1)
    throw std::invalid_argument("invalid projection stride, tile grouping or timing options");
  return a;
}

template <class T>
std::vector<T> read_raw(const std::string &path, size_t count, const char *name) {
  if (count > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::overflow_error(std::string(name) + " size overflow");
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error(std::string("cannot open ") + name);
  const auto bytes = file.tellg();
  if (bytes < 0 || static_cast<uint64_t>(bytes) != count * sizeof(T)) throw std::runtime_error(std::string(name) + " byte count mismatch");
  std::vector<T> data(count);
  file.seekg(0);
  file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (!file) throw std::runtime_error(std::string("failed reading ") + name);
  return data;
}

template <class T> void write_raw(const std::string &path, const std::vector<T> &data) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.exceptions(std::ios::badbit | std::ios::failbit);
  file.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
}

struct DeviceAlloc {
  sycl::queue &queue;
  std::vector<void *> pointers;
  explicit DeviceAlloc(sycl::queue &q) : queue(q) {}
  template <class T> T *make(size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::overflow_error("device allocation size overflow");
    T *pointer = sycl::malloc_device<T>(std::max<size_t>(count, 1), queue);
    if (!pointer) throw std::bad_alloc();
    pointers.push_back(pointer);
    return pointer;
  }
  ~DeviceAlloc() {
    for (void *pointer : pointers) sycl::free(pointer, queue);
  }
};

double event_ms(const sycl::event &first, const sycl::event &last) {
  const auto begin = first.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end = last.get_profiling_info<sycl::info::event_profiling::command_end>();
  if (end < begin) throw std::runtime_error("non-monotonic event timestamps");
  return static_cast<double>(end - begin) / 1e6;
}

double quantile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  return values[std::min(values.size() - 1, static_cast<size_t>(std::ceil(p * values.size())) - 1)];
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values.size() % 2 ? values[values.size() / 2] : (values[values.size() / 2 - 1] + values[values.size() / 2]) * 0.5;
}

float bf16_float(uint16_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

std::string json_escape(const std::string &value) {
  std::string out;
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out.push_back('\\'), out.push_back(static_cast<char>(c));
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else if (c < 0x20) out += ' ';
    else out.push_back(static_cast<char>(c));
  }
  return out;
}

int main(int argc, char **argv) try {
  const Args a = parse(argc, argv);
  if (std::endian::native != std::endian::little) throw std::runtime_error("XRV adapter requires a little-endian host");
  const k::QsaShape shape{.hidden = 2560, .q_heads = a.q_heads, .kv_heads = a.kv_heads, .cards = a.cards,
                          .card = a.card, .kv = k::QsaKv::i8v4};
  if (!shape.split_ok() || shape.hq() != 12 || shape.hkv() != 1 || shape.lgrp() != 12)
    throw std::invalid_argument("K8V4 attention adapter requires 12 local query heads sharing one local KV head");
  if (a.tp_size != a.cards || a.tp_rank != a.card)
    throw std::invalid_argument("TP identity must match the QSA head partition");
  const int projection_stride = a.projection_stride ? a.projection_stride : shape.q_n();
  if (projection_stride < shape.q_n()) throw std::invalid_argument("projection stride is smaller than local Q|gate storage");
  const int hq = shape.hq(), hkv = shape.hkv(), record_stride = k::kSegRec;
  const size_t query_count = static_cast<size_t>(a.rows) * hq * k::kQsaDim;
  const size_t records_count = static_cast<size_t>(a.rows) * record_stride;
  const size_t table_count = static_cast<size_t>(a.table_slots) * a.slot_pages;
  const size_t key_bytes = static_cast<size_t>(a.pool_pages) * hkv * k::kQsaPage * k::kQsaDim;
  const size_t value_bytes = static_cast<size_t>(a.pool_pages) * hkv * k::kQsaPage * (k::kQsaDim / 2);
  const size_t key_scale_count = static_cast<size_t>(a.pool_pages) * hkv * k::kQsaPage;
  const size_t value_scale_count = static_cast<size_t>(a.pool_pages) * hkv * k::kQsaPage * k::kQsaV4Scales;
  const auto query = read_raw<uint16_t>(a.query, query_count, "query");
  const auto gate = read_raw<uint16_t>(a.gate, query_count, "gate");
  const auto positions = read_raw<int32_t>(a.positions, a.rows, "positions");
  const auto slots = read_raw<int32_t>(a.slots, a.rows, "slots");
  const auto pages = read_raw<int32_t>(a.page_table, table_count, "page table");
  const auto keys = read_raw<uint8_t>(a.keys, key_bytes, "K8 cache");
  const auto values = read_raw<uint8_t>(a.values, value_bytes, "V4 cache");
  const auto key_scales = read_raw<float>(a.key_scales, key_scale_count, "K scales");
  const auto value_scales = read_raw<uint16_t>(a.value_scales, value_scale_count, "V4 scales");
  const auto selected = read_raw<int32_t>(a.selected_ids, static_cast<size_t>(a.rows) * k::kQsaTopBlocks, "selected IDs");
  const auto selected_counts = read_raw<int32_t>(a.selected_counts, a.rows, "selected counts");
  std::vector<int32_t> records(records_count, 0);
  for (int r = 0; r < a.rows; ++r) {
    if (positions[r] < 0 || positions[r] >= a.capacity || positions[r] + 1 > a.capacity ||
        slots[r] < 0 || slots[r] >= a.table_slots)
      throw std::runtime_error("row position or page-table slot is outside the fixture");
    records[static_cast<size_t>(r) * record_stride + k::kSegPos] = positions[r];
    records[static_cast<size_t>(r) * record_stride + k::kSegSlot] = slots[r];
    const int blocks = (positions[r] + 1) / k::kQsaCompress;
    if (selected_counts[r] < 0 || selected_counts[r] > k::kQsaTopBlocks || (blocks > k::kQsaTopBlocks && selected_counts[r] == 0))
      throw std::runtime_error("selected count out of range for the row position");
    int previous = -1;
    for (int j = 0; j < k::kQsaTopBlocks; ++j) {
      const int id = selected[static_cast<size_t>(r) * k::kQsaTopBlocks + j];
      if (j < selected_counts[r]) {
        if (id <= previous || id < 0 || id >= blocks) throw std::runtime_error("selected IDs must be ascending and below the row block bound");
        previous = id;
      } else if (id != -1) {
        throw std::runtime_error("selected ID padding must be -1");
      }
    }
  }
  for (int page : pages)
    if (page < 0 || page >= a.pool_pages) throw std::runtime_error("page table refers outside the K/V pools");
  for (float scale : key_scales)
    if (!std::isfinite(scale) || scale < 0) throw std::runtime_error("invalid K scale");
  for (uint16_t bits : value_scales)
    if (!std::isfinite(bf16_float(bits)) || bf16_float(bits) < 0) throw std::runtime_error("invalid V4 scale");

  const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (a.device >= static_cast<int>(devices.size())) throw std::runtime_error("requested GPU is unavailable");
  const sycl::device device = devices.at(static_cast<size_t>(a.device));
  const uint32_t vendor = device.get_info<sycl::info::device::vendor_id>();
  if (vendor != 0x8086u) throw std::runtime_error("selected device is not Intel GPU");
  sycl::queue queue(device, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
  DeviceAlloc memory(queue);
  auto *device_query = memory.make<k::qsa_bf16>(query_count);
  auto *device_gate = memory.make<float>(static_cast<size_t>(a.rows) * projection_stride);
  auto *device_records = memory.make<int32_t>(records_count);
  auto *device_pages = memory.make<int32_t>(table_count);
  auto *device_keys = memory.make<uint8_t>(key_bytes);
  auto *device_values = memory.make<uint8_t>(value_bytes);
  auto *device_key_scales = memory.make<float>(key_scale_count);
  auto *device_value_scales = memory.make<k::qsa_bf16>(value_scale_count);
  auto *device_selected = memory.make<int32_t>(selected.size());
  auto *device_selected_counts = memory.make<int32_t>(selected_counts.size());
  auto *device_part = memory.make<float>(k::qsa_part_floats(shape, a.rows));
  auto *device_attention = memory.make<k::qsa_bf16>(static_cast<size_t>(a.rows) * shape.o_k());

  std::vector<float> gate_projection(static_cast<size_t>(a.rows) * projection_stride, 0.f);
  for (int r = 0; r < a.rows; ++r)
    for (int h = 0; h < hq; ++h)
      for (int d = 0; d < k::kQsaDim; ++d)
        gate_projection[static_cast<size_t>(r) * projection_stride + h * 2 * k::kQsaDim + k::kQsaDim + d] =
            bf16_float(gate[(static_cast<size_t>(r) * hq + h) * k::kQsaDim + d]);

  queue.memcpy(device_query, query.data(), query.size() * sizeof(uint16_t));
  queue.memcpy(device_gate, gate_projection.data(), gate_projection.size() * sizeof(float));
  queue.memcpy(device_records, records.data(), records.size() * sizeof(int32_t));
  queue.memcpy(device_pages, pages.data(), pages.size() * sizeof(int32_t));
  queue.memcpy(device_keys, keys.data(), keys.size());
  queue.memcpy(device_values, values.data(), values.size());
  queue.memcpy(device_key_scales, key_scales.data(), key_scales.size() * sizeof(float));
  queue.memcpy(device_value_scales, value_scales.data(), value_scales.size() * sizeof(uint16_t));
  queue.memcpy(device_selected, selected.data(), selected.size() * sizeof(int32_t));
  queue.memcpy(device_selected_counts, selected_counts.data(), selected_counts.size() * sizeof(int32_t)).wait_and_throw();

  k::QsaCache cache{};
  cache.k = device_keys;
  cache.v = device_values;
  cache.ks = device_key_scales;
  cache.vs = reinterpret_cast<float *>(device_value_scales);
  cache.pages = device_pages;
  cache.max_tokens = a.capacity;
  cache.slot_pages = a.slot_pages;
  k::QsaScratch scratch{};
  scratch.q = device_query;
  scratch.y = device_gate;
  scratch.sel = device_selected;
  scratch.sel_n = device_selected_counts;
  scratch.part = device_part;
  scratch.attn = device_attention;
  k::QsaStep step{};
  step.pos = device_records;
  step.pos_bound = a.capacity;
  step.stride = record_stride;
  step.seg_rows = 1;

  std::vector<double> device_ms, host_ms;
  auto once = [&](bool record_timing) {
    const auto host_begin = clock_type::now();
    auto events = k::xrv::sparse_attention(queue, shape, a.rows, cache, step, scratch, projection_stride, a.tiles);
    events.merge.wait_and_throw();
    const auto host_end = clock_type::now();
    if (record_timing) {
      device_ms.push_back(event_ms(events.attn, events.merge));
      host_ms.push_back(std::chrono::duration<double, std::milli>(host_end - host_begin).count());
    }
  };
  for (int i = 0; i < a.warmup; ++i) once(false);
  for (int i = 0; i < a.repeat; ++i) once(true);
  std::vector<uint16_t> output(static_cast<size_t>(a.rows) * shape.o_k());
  queue.memcpy(output.data(), device_attention, output.size() * sizeof(uint16_t)).wait_and_throw();
  std::filesystem::create_directories(a.output);
  write_raw(a.output + "/output.u16le", output);

  const std::string device_name = device.get_info<sycl::info::device::name>();
  const std::string driver = device.get_info<sycl::info::device::driver_version>();
  const char *isolation = std::getenv("XRV_WORKLOAD_ISOLATION");
  const bool attested = isolation && std::string(isolation) == "attested-idle-after-process-check-and-exclusive-lock";
#ifdef __INTEL_LLVM_COMPILER
  const int compiler_version = __INTEL_LLVM_COMPILER;
#else
  const int compiler_version = 0;
#endif
  auto emit_samples = [](const std::vector<double> &samples) {
    for (size_t i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
  };
  const auto device_minmax = std::minmax_element(device_ms.begin(), device_ms.end());
  const auto host_minmax = std::minmax_element(host_ms.begin(), host_ms.end());
  std::cout << "{\"hardware\":{\"device\":\"" << json_escape(device_name) << "\",\"driver\":\"" << json_escape(driver)
            << "\",\"vendor_id\":" << vendor << ",\"device_index\":" << a.device
            << ",\"frequency_power_state\":\"unmeasured\",\"concurrent_workload\":\"unknown\"},\"workload_isolation\":{\"status\":\""
            << (attested ? "caller-attested-idle" : "unknown") << "\",\"evidence\":\""
            << (attested ? "process-check+exclusive-lock-attested" : "not-reported")
            << "\"},\"runtime\":{\"backend\":\"SYCL\",\"compiler\":\"IntelLLVM\",\"compiler_version_code\":" << compiler_version
            << "},\"tp\":{\"rank\":" << a.tp_rank << ",\"size\":" << a.tp_size
            << "},\"timing\":{\"measurement_kind\":\"sycl_event_span_and_host_wall\",\"device_event_ms\":{\"median\":"
            << median(device_ms) << ",\"p90\":" << quantile(device_ms, .90) << ",\"p99\":" << quantile(device_ms, .99)
            << ",\"jitter_max_minus_min\":" << (*device_minmax.second - *device_minmax.first) << ",\"samples\":[";
  emit_samples(device_ms);
  std::cout << "]},\"host_wall_ms\":{\"median\":" << median(host_ms) << ",\"p90\":" << quantile(host_ms, .90)
            << ",\"p99\":" << quantile(host_ms, .99) << ",\"jitter_max_minus_min\":" << (*host_minmax.second - *host_minmax.first)
            << ",\"samples\":[";
  emit_samples(host_ms);
  std::cout << "]}},\"execution\":{\"rows\":" << a.rows << ",\"capacity\":" << a.capacity << ",\"pool_pages\":" << a.pool_pages
            << ",\"tiles_per_group\":" << a.tiles << ",\"record_stride\":" << record_stride << ",\"projection_stride\":" << projection_stride
            << ",\"warmup\":" << a.warmup << ",\"repeat\":" << a.repeat << "}}\n";
  return 0;
} catch (const sycl::exception &error) {
  std::cerr << "SYCL: " << error.what() << '\n';
  return 2;
} catch (const std::exception &error) {
  std::cerr << "xrv qsa attention: " << error.what() << '\n';
  return 2;
}
