#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>
#include <limits>

#include "xenithium/kernels/attention/qsa.hpp"

namespace k = xenithium::kernels;
using clock_type = std::chrono::steady_clock;

struct Args {
  std::string scores, nblk, output;
  int rows = 0, stride = 0, bound = 0, chunk = 0, tp_rank = 0, tp_size = 1, device = 0, warmup = 5, repeat = 1;
};

std::string json_escape(const std::string &s) {
  std::string out;
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: if (c < 0x20) { char b[7]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; } else out += static_cast<char>(c);
    }
  }
  return out;
}

int parse_int(const char *s, const char *name) {
  size_t used = 0;
  const int v = std::stoi(s, &used);
  if (s[used] != '\0') throw std::invalid_argument(std::string("invalid integer for ") + name);
  return v;
}

Args parse(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto value = [&]() -> const char * { if (++i >= argc) throw std::invalid_argument("missing value for " + key); return argv[i]; };
    if (key == "--scores") a.scores = value();
    else if (key == "--nblk") a.nblk = value();
    else if (key == "--output") a.output = value();
    else if (key == "--rows") a.rows = parse_int(value(), "--rows");
    else if (key == "--stride") a.stride = parse_int(value(), "--stride");
    else if (key == "--bound") a.bound = parse_int(value(), "--bound");
    else if (key == "--chunk") a.chunk = parse_int(value(), "--chunk");
    else if (key == "--tp-rank") a.tp_rank = parse_int(value(), "--tp-rank");
    else if (key == "--tp-size") a.tp_size = parse_int(value(), "--tp-size");
    else if (key == "--device") a.device = parse_int(value(), "--device");
    else if (key == "--warmup") a.warmup = parse_int(value(), "--warmup");
    else if (key == "--repeat") a.repeat = parse_int(value(), "--repeat");
    else if (key == "--help") { std::puts("xenithium-xrv-qsa-select --scores FILE --nblk FILE --rows N --stride N --bound N --chunk N --output DIR [--device N --warmup N --repeat N --tp-rank N --tp-size N]"); std::exit(0); }
    else throw std::invalid_argument("unknown argument: " + key);
  }
  if (a.scores.empty() || a.nblk.empty() || a.output.empty()) throw std::invalid_argument("--scores, --nblk and --output are required");
  if (a.rows < 1 || a.rows > 64 || a.stride < 1 || a.stride > k::kQsaMaxBlocks || a.bound < 1 || a.bound > a.stride || a.bound > k::kQsaMaxBlocks)
    throw std::invalid_argument("invalid rows/stride/bound (expected rows 1..64 and bound<=stride<=262144)");
  if (a.chunk && (a.chunk < k::kQsaMinChunk || a.chunk > k::kQsaPfChunk)) throw std::invalid_argument("chunk must be 0 or 4096..32768");
  if (a.tp_size < 1 || a.tp_rank < 0 || a.tp_rank >= a.tp_size) throw std::invalid_argument("invalid tensor-parallel rank/size");
  if (a.device < 0 || a.warmup < 0 || a.repeat < 1) throw std::invalid_argument("device/warmup/repeat out of range");
  return a;
}

template <typename T>
std::vector<T> read_raw(const std::string &path, size_t expected, const char *name) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error(std::string("cannot open ") + name + ": " + path);
  const auto bytes = f.tellg();
  if (bytes < 0 || static_cast<uint64_t>(bytes) != expected * sizeof(T)) throw std::runtime_error(std::string(name) + " byte count does not match fixture dimensions");
  std::vector<T> v(expected);
  f.seekg(0);
  f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(expected * sizeof(T)));
  if (!f) throw std::runtime_error(std::string("failed reading ") + name);
  return v;
}

template <typename T>
void write_raw(const std::string &path, const std::vector<T> &v) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.exceptions(std::ios::badbit | std::ios::failbit);
  f.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
}

struct DeviceAlloc {
  sycl::queue &q;
  std::vector<void *> ptrs;
  explicit DeviceAlloc(sycl::queue &queue) : q(queue) {}
  template <typename T> T *make(size_t n) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::overflow_error("device allocation size overflow");
    T *p = sycl::malloc_device<T>(std::max<size_t>(1, n), q);
    if (!p) throw std::bad_alloc();
    ptrs.push_back(p);
    return p;
  }
  ~DeviceAlloc() { for (void *p : ptrs) sycl::free(p, q); }
};

double event_ms(const sycl::event &first, const sycl::event &last) {
  const auto start = first.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end = last.get_profiling_info<sycl::info::event_profiling::command_end>();
  if (end < start) throw std::runtime_error("SYCL profiling timestamps were not monotonic");
  return static_cast<double>(end - start) / 1e6;
}

double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  const size_t i = std::min(v.size() - 1, static_cast<size_t>(std::ceil(p * v.size())) - 1);
  return v[i];
}

double median_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char **argv) try {
  const Args a = parse(argc, argv);
  if (std::endian::native != std::endian::little) throw std::runtime_error("XRV QSA adapter currently requires a little-endian host");
  const size_t score_count = static_cast<size_t>(a.rows) * a.stride;
  const auto scores = read_raw<uint32_t>(a.scores, score_count, "scores");
  const auto nblk = read_raw<int32_t>(a.nblk, static_cast<size_t>(a.rows), "nblk");
  int actual_bound = 0;
  for (int r = 0; r < a.rows; ++r) {
    if (nblk[r] < 0 || nblk[r] > a.bound) throw std::runtime_error("nblk row exceeds fixture bound");
    actual_bound = std::max(actual_bound, nblk[r]);
    for (int i = 0; i < nblk[r]; ++i) {
      const uint32_t b = scores[static_cast<size_t>(r) * a.stride + i];
      if ((b & 0x80000000u) || ((b >> 23) & 0xffu) == 0xffu) throw std::runtime_error("score is negative, NaN or infinity; qsa_select requires finite non-negative score bits");
    }
  }
  if (actual_bound > a.bound) throw std::runtime_error("actual nblk exceeds bound");

  auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (a.device >= static_cast<int>(devices.size())) throw std::runtime_error("requested GPU device index is unavailable");
  const sycl::device device = devices.at(static_cast<size_t>(a.device));
  const uint32_t vendor_id = device.get_info<sycl::info::device::vendor_id>();
  if (vendor_id != 0x8086u) throw std::runtime_error("selected XPU is not an Intel GPU (vendor id is not 0x8086)");
  sycl::queue q(device, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
  DeviceAlloc mem(q);
  auto *dscore = mem.make<uint32_t>(score_count);
  auto *dnblk = mem.make<int32_t>(static_cast<size_t>(a.rows));
  auto *dsel = mem.make<int32_t>(static_cast<size_t>(a.rows) * k::kQsaTopBlocks);
  auto *dsel_n = mem.make<int32_t>(static_cast<size_t>(a.rows));
  auto *cand_key = mem.make<uint32_t>(static_cast<size_t>(a.rows) * k::kQsaMaxChunks * k::kQsaTopBlocks);
  auto *cand_id = mem.make<int32_t>(static_cast<size_t>(a.rows) * k::kQsaMaxChunks * k::kQsaTopBlocks);
  auto *cand_n = mem.make<int32_t>(static_cast<size_t>(a.rows) * k::kQsaMaxChunks);
  q.memcpy(dscore, scores.data(), score_count * sizeof(uint32_t));
  q.memcpy(dnblk, nblk.data(), static_cast<size_t>(a.rows) * sizeof(int32_t));
  q.wait_and_throw();
  k::QsaScratch sc{};
  sc.score=dscore; sc.nblk=dnblk; sc.sel=dsel; sc.sel_n=dsel_n; sc.cand_key=cand_key; sc.cand_id=cand_id; sc.cand_n=cand_n;

  std::vector<double> device_ms, wall_ms;
  sycl::event last;
  auto once = [&](bool measure) {
    q.memset(dsel, 0xff, static_cast<size_t>(a.rows) * k::kQsaTopBlocks * sizeof(int32_t));
    q.memset(dsel_n, 0, static_cast<size_t>(a.rows) * sizeof(int32_t));
    q.wait_and_throw();  // keep output initialization outside the measured span
    const auto wall0 = clock_type::now();
    sycl::event merge;
    const sycl::event first = k::qsa_select(q, a.rows, dscore, a.stride, dnblk, a.bound, sc, &merge, a.chunk);
    last = merge;
    last.wait_and_throw();
    const auto wall1 = clock_type::now();
    if (measure) {
      device_ms.push_back(event_ms(first, last));
      wall_ms.push_back(std::chrono::duration<double, std::milli>(wall1-wall0).count());
    }
  };
  for (int i=0;i<a.warmup;++i) once(false);
  for (int i=0;i<a.repeat;++i) once(true);

  std::vector<int32_t> ids(static_cast<size_t>(a.rows) * k::kQsaTopBlocks), counts(static_cast<size_t>(a.rows));
  q.memcpy(ids.data(),dsel,ids.size()*sizeof(int32_t));
  q.memcpy(counts.data(),dsel_n,counts.size()*sizeof(int32_t));
  q.wait_and_throw();
  write_raw(a.output + "/selected_ids.i32le",ids);
  write_raw(a.output + "/selected_counts.i32le",counts);

  const auto ms = device.get_info<sycl::info::device::name>();
  const auto driver = device.get_info<sycl::info::device::driver_version>();
  const auto device_version = device.get_info<sycl::info::device::version>();
  const char *isolation_env = std::getenv("XRV_WORKLOAD_ISOLATION");
  const bool workload_attested = isolation_env != nullptr &&
      std::string(isolation_env) == "attested-idle-after-process-check-and-exclusive-lock";
#ifdef __INTEL_LLVM_COMPILER
  const int compiler_version = __INTEL_LLVM_COMPILER;
#else
  const int compiler_version = 0;
#endif
  std::cout << "{\"hardware\":{\"device\":\"" << json_escape(ms) << "\",\"driver\":\"" << json_escape(driver)
            << "\",\"device_version\":\"" << json_escape(device_version) << "\",\"vendor_id\":" << vendor_id
            << ",\"device_index\":" << a.device << ",\"frequency_power_state\":\"unmeasured\",\"concurrent_workload\":\"unknown\"},\"workload_isolation\":{\"status\":\""
            << (workload_attested ? "caller-attested-idle" : "unknown") << "\",\"evidence\":\""
            << (workload_attested ? "process-check+exclusive-lock-attested" : "not-reported") << "\"},\"runtime\":{\"backend\":\"SYCL\",\"xrv_adapter\":1,\"compiler\":\"IntelLLVM\",\"compiler_version_code\":" << compiler_version << "},\"tp\":{\"rank\":" << a.tp_rank
            << ",\"size\":" << a.tp_size << "},\"timing\":{\"measurement_kind\":\"sycl_event_span_and_host_wall\",\"device_event_ms\":{\"median\":"
            << median_of(device_ms) << ",\"p90\":" << percentile(device_ms,.90) << ",\"p99\":" << percentile(device_ms,.99) << ",\"jitter_max_minus_min\":"
            << (*std::max_element(device_ms.begin(),device_ms.end())-*std::min_element(device_ms.begin(),device_ms.end())) << ",\"samples\":[";
  for (size_t i=0;i<device_ms.size();++i) std::cout << (i ? "," : "") << device_ms[i];
  std::cout << "]},\"host_wall_ms\":{\"median\":" << median_of(wall_ms) << ",\"p90\":" << percentile(wall_ms,.90) << ",\"p99\":"
            << percentile(wall_ms,.99) << ",\"jitter_max_minus_min\":" << (*std::max_element(wall_ms.begin(),wall_ms.end())-*std::min_element(wall_ms.begin(),wall_ms.end())) << ",\"samples\":[";
  for (size_t i=0;i<wall_ms.size();++i) std::cout << (i ? "," : "") << wall_ms[i];
  std::cout << "]}},\"execution\":{\"rows\":" << a.rows << ",\"bound\":" << a.bound << ",\"chunk\":" << a.chunk << ",\"warmup\":" << a.warmup << ",\"repeat\":" << a.repeat << "}}\n";
  return 0;
} catch (const sycl::exception &e) {
  std::cerr << "xrv xpu: SYCL error: " << e.what() << '\n';
  return 2;
} catch (const std::exception &e) {
  std::cerr << "xrv xpu: " << e.what() << '\n';
  return 2;
}
