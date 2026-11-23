# XRV CI setup

The CPU workflow is usable immediately. GPU jobs remain skipped until repository variables, protected environments, and dedicated isolated runners are configured.

## Tier 1: automatic portable CI

.github/workflows/xrv-cpu.yml runs on pull requests and pushes. Its portable job uses GitHub-hosted Ubuntu, Python plus the jsonschema CI dependency, checked-in synthetic fixtures, and no model/GPU. It checks PR patch whitespace, runs Python syntax checks and validates the versioned JSON Schemas against all checked-in manifests, runs all XRV tests, validates fixture checksums and independent goldens, then runs selector CPU replay and attention CPU-reference reproduction. A separate GitHub-hosted CPU build job uses Intel's oneAPI 2026.2 development container to compile the probe and replay adapters without GPU access. The portable job uploads only a small JSON summary.

This tier compiles SYCL C++ but does not execute XPU kernels or validate GPU correctness or performance. Those require Tier 2. CPU/synthetic success is not GPU validation.

## Tier 2: manually approved Intel B60 replay

.github/workflows/xrv-b60.yml is workflow_dispatch only and requires explicit authorization. It only accepts dispatch from the repository default branch and checks that the revision is already an ancestor of that branch, so unreviewed fork/PR code is never executed on the self-hosted GPU runner. It also requires XRV_B60_ENABLED=true and protected environment xrv-b60. Concurrency is serialized. The job checks devices and fails if a Xenithium server/probe/bake process is running; it never stops anything. It replays both fixtures under one exclusive host lock. The attention adapter runs one rank per process, so its summary is per-rank kernel evidence, not a TP2 critical-path measurement.

Owner setup required in GitHub:

1. In a dedicated isolated Linux host/VM with B60, install the repository toolchain. Current source of truth is README.md and scripts/install-deps.sh: oneAPI 2026.2 / IntelLLVM, CMake 3.28+, Ninja, Level Zero loader and B60-compatible driver, and oneDNN from that oneAPI setup. Source scripts/oneapi-env.sh in validation shells.
2. Register a self-hosted Actions runner under a dedicated account/workspace. Assign labels linux, x64, intel-xpu, b60, xrv. Never label a production inference host as an XRV runner.
3. Set repository variable XRV_B60_ENABLED=true only after isolation and runner verification. Set XRV_GPU_RUNNER_ISOLATED=true. Create protected environment xrv-b60 and require owner/reviewer approval.
4. Restrict runner group access to this repository. Use a dedicated work directory and mount only source and the checked-in small fixture. Do not mount model directories, user homes, credentials, production sockets, or private capture storage.
5. Give the runner account read/write only to its work directory. Do not add SSH/cloud/model credentials. Workflow token is contents:read only. Prefer ephemeral runners and remove checkout credentials after checkout.
6. Configure machine scheduling so only the xrv-b60 environment can claim the B60 runner. Workflow concurrency is a second guard, not isolation. Keep serving containers off this runner.
7. Before enabling the repository variable, verify approval, labels, device visibility, cleanup, and retention with synthetic selector and sparse-attention replay. Retain only JSON/Markdown summaries; never upload tensors or logs containing prompt/model data.

Repository implementation in xrv-b60.yml includes manual dispatch, boolean authorization, trusted-ref ancestry check, protected environment, concurrency serialization, timeouts, server/probe refusal, Intel device check, targeted build, XPU replay, and summary-only upload. It remains skipped until a matching runner and variable exist.

## Tier 3: gated full-model validation

.github/workflows/xrv-full-model.yml is separate, manual, and requires explicit authorization, XRV_FULL_MODEL_ENABLED=true, a protected xrv-full-model environment, a trusted merged revision, and a dedicated xrv-model runner. It is skipped by default and refuses to run without the future SMR executable. SMR is not implemented; do not enable this workflow until that adapter and its correctness gate exist and the owner separately authorizes isolated model access.

Owner setup for Tier 3:

- Use a separate runner host/group, protected environment, and restricted model mount. Never run untrusted PR code on it.
- Store no model weights, prompts, capture tensors, or secrets in Actions artifacts. If model credentials are essential, give only read-only scope and never echo them.
- Configure memory headroom, timeout, cleanup, log redaction, and concurrency limits. The job must fail rather than stop or preempt another server.
- Establish independent known-good state, output acceptance, and baseline criteria before enabling model performance gates.

## Self-hosted runner security

A self-hosted runner executes repository code with the runner account's access. Untrusted PR code can exfiltrate files/credentials, persist background processes, or attack other services. Never use pull_request_target to check out and run PR code on these runners. The workflows accept only revisions already merged into the default branch and require a human approval environment. Use ephemeral runners where possible, wipe workspaces between jobs, rotate registration tokens, and remove runners after testing. Do not grant write permissions, package publishing, or long-lived secrets.
