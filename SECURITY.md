<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Security Policy

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.

To report a potential security vulnerability in any NVIDIA product, use one of
the following channels:

1. **NVIDIA Vulnerability Disclosure Program** (preferred):
   <https://www.nvidia.com/en-us/security/>
2. **Email:** [psirt@nvidia.com](mailto:psirt@nvidia.com). Please encrypt
   sensitive reports with NVIDIA's public PGP key:
   <https://www.nvidia.com/en-us/security/pgp-key>
3. **GitHub Private Vulnerability Reporting:** use the **Security** tab of this
   repository, if enabled.

OEM partners should contact their NVIDIA Customer Program Manager.

Please include:

1. Product name and version or branch that contains the vulnerability
2. Type of vulnerability (for example code execution, denial of service,
   buffer overflow)
3. Instructions to reproduce the vulnerability
4. Proof-of-concept or exploit code, if available
5. Potential impact, including how an attacker could exploit it

NVIDIA PSIRT acknowledges reports, assesses them, and coordinates fixes and
disclosure with the reporter. See <https://www.nvidia.com/en-us/security/> for
past security bulletins and notices.

## Security Architecture and Context

**Project:** the Triton Inference Server Identity Backend, a small C++ backend
loaded by Triton Inference Server through the `TRITONBACKEND_*` plugin API. It
copies input tensors to the matching output tensors and is used primarily for
testing and as a reference for backend authors.

**Classification:** Library (a shared object, `libtriton_identity.so`, loaded
in-process by the server). It is not a standalone service.

**Repository Exposure Classification:** Public (the repository is publicly
visible on GitHub).

**Service Exposure Classification:** Internal-Isolated (low confidence). The
backend is intended for testing and development, has no network listener of its
own, and handles no secrets. Deployments that expose it through a Triton server
inherit that server's exposure.

**Primary security responsibility:** correct handling of request and response
buffers supplied by the Triton core, so a malformed request cannot cause memory
corruption or crash the hosting server process.

**Key interfaces and boundaries:**

- The Triton backend API in `src/identity.cc` (initialize, model and instance
  lifecycle, `TRITONBACKEND_ModelInstanceExecute`). This is the only runtime
  interface; the backend opens no sockets and parses no files.
- Input and output buffers passed by the core, which can reside in CPU or GPU
  memory, copied by the `CopyBuffer` helper.
- Backend configuration and model configuration supplied by the server. The
  backend logs the backend configuration and also acts on model configuration:
  optional-input shapes, a model-load delay (`creation_delay_sec`), an execution
  delay (`execute_delay_ms`, `instance_wise_delay_multiplier`) and custom
  tracing settings (`enable_custom_tracing`, `nested_span_count`,
  `single_activity_frequency`). These parameters control runtime behavior, so
  permission to change a model configuration is a runtime control.
- Optional metrics registered through the Triton metrics API.

## Threat Model

1. **Buffer size mismatch during copy:** in `TRITONBACKEND_ModelInstanceExecute`
   the output buffer is allocated from the reported total input byte size and
   filled by copying each input buffer at a running offset. A core or backend
   API inconsistency between the per-buffer sizes and the total could lead to
   an out-of-bounds write in the hosting process.
2. **Crash or hang of the hosting server:** the backend runs in the Triton
   server process, so an unhandled error, exception, or invalid pointer in
   request handling affects every model served by that process (denial of
   service).
3. **Unexpected shapes and sizes:** the shape-derived byte size computed by
   `GetByteSize` and tensor shapes taken from requests could be extreme or
   zero-sized; overflow or large allocations could exhaust memory.
4. **GPU memory handling:** when buffers are in GPU memory the backend issues
   device copies on a CUDA stream owned by the model instance. Errors in
   stream synchronization or memory-type handling could expose stale data
   between requests or fail unpredictably.
5. **Information exposure through logging:** the backend logs the full
   backend configuration as JSON at startup and request details (identifiers,
   input sizes) during execution. If configuration or request metadata contains
   sensitive values, they appear in server logs.
6. **Supply chain and build integrity:** the backend is built from CMake with
   dependencies fetched from sibling Triton repositories (`common`, `core`,
   `backend`). A compromised dependency or an unpinned fetch affects the
   resulting library.

## Critical Security Assumptions

- **Request protections are deployment requirements.** The backend performs no
  authentication, authorization or request size check of its own, and it sizes
  output buffers from the reported input byte size. It assumes the deployment
  authenticates clients and enforces request size limits (through server
  options or a gateway), and that the core has validated tensor shapes and data
  types before the request reaches the backend. Where those controls are not
  configured, requests reach the backend unauthenticated and unbounded.
- **No transport security in this component.** The backend has no network
  surface; TLS and access control are the responsibility of the Triton server
  endpoints and the deployment around them.
- **Buffers returned by the core are consistent.** The reported buffer sizes,
  counts, and memory types are assumed to be accurate and the buffers valid for
  the duration of the call.
- **Model repository is trusted.** Anyone who can place a model configuration
  or load this library into the server can run code in the server process.
- **The backend is not intended for production inference.** It is a testing and
  reference component and has not been hardened for hostile workloads.
- **Build inputs are trusted.** Dependencies and the build environment are
  assumed to come from the official Triton repositories at the matching
  release.
