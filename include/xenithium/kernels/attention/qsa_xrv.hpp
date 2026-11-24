#pragma once

#include "xenithium/kernels/attention/qsa.hpp"

// XRV's low-level QSA attention replay boundary. This launches the production packed-cache
// sparse attention and LSE/gate merge from already projected queries, selections and cache state.
// It intentionally excludes projections, scoring/selection, and output GEMM.
namespace xenithium::kernels::xrv {

QsaEvents sparse_attention(sycl::queue &q, const QsaShape &shape, int rows, const QsaCache &cache,
                           const QsaStep &step, const QsaScratch &scratch, int projection_stride, int tiles_per_group = 1);

}  // namespace xenithium::kernels::xrv
