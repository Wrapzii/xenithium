#pragma once

namespace xenithium::xrv {

// Installs the qsa.select capture hook only when XEN_XRV_CAPTURE_DIR is set. Returns false for invalid
// capture settings so an explicitly requested capture cannot silently proceed without a hook.
bool install_probe_qsa_capture_from_env() noexcept;

}  // namespace xenithium::xrv
