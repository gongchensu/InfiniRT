# Owning Tensor Migration Plan

Status: active construction plan

Last updated: 2026-11-23

Owner: InfiniRT/InfiniLM runtime maintainers

This document is the source of truth for migrating the owning tensor and memory
foundation currently implemented in InfiniLM into InfiniRT. It is deliberately
written as a handoff document: every work session must update the task ledger,
evidence log, and decisions before leaving the work in a clean state.

## 1. Purpose and Outcome

The end state is:

1. `infini::rt::Device`, `infini::rt::DataType`, shape/stride metadata, and
   `infini::rt::TensorView` remain the canonical non-owning tensor description.
2. InfiniRT owns the reusable owning tensor implementation, including storage
   ownership, metadata validation, view construction, and lifetime semantics.
3. InfiniOps continues to accept `infini::rt::TensorView` and does not define a
   competing tensor type.
4. InfiniLM replaces its private `infinilm::runtime::Tensor` implementation
   with RT aliases or thin adapters while preserving its Python API, worker
   runtime rules, graph behavior, and inference semantics.
5. Memory allocation, stream ordering, graph retention, and external ownership
   remain explicit and testable across RT, Ops, LM, and CCL.

This is a runtime-boundary migration, not an attempt to move operators into RT
or redesign LM scheduling.

## 2. Current Baseline

The plan starts from these coherent revisions. Do not mix them with older local
checkouts while reproducing baseline evidence.

| Repository | Branch | Revision |
| --- | --- | --- |
| InfiniLM | `main` | `ebb081ef129ee8405e2e0863f40441367a40f85c` |
| InfiniCore | `main` | `64362228212c77422124574af4d7339d728610c7` |
| InfiniRT | `master` | `c0810cfef8f4e4c00a3bc404c2831428ac37b9bc` |
| InfiniOps | `master` | `f52af91a83f19610b16120ab1be4f5c884c02c1c` |
| InfiniCCL | `master` | `51cc9db31d58aac21643d72f0f77cf5c9ecfaee5` |

### 2.1 Current Implementation Inventory

InfiniRT currently provides:

- `DataType` in `src/data_type.h`, including distinct `kBool`.
- `Device` in `src/device.h`.
- Non-owning `TensorView` in `src/tensor_view.h` and `src/tensor_view.cc`.
- Inline shape/stride storage in `src/common/shape_strides_storage.h`.
- `ReusableMemoryPool` in `include/infini/rt/reusable_memory_pool.h` and
  `src/reusable_memory_pool.cc`.
- `DeviceGuard` in `include/infini/rt/device_guard.h`.

InfiniLM currently provides:

- Owning `Tensor` and `TensorImpl` in `csrc/runtime/tensor.hpp`,
  `csrc/runtime/tensor/tensor.cc`, `csrc/runtime/tensor/view.cc`, and
  `csrc/runtime/tensor/copy.cc`.
- `Memory` in `csrc/runtime/memory.hpp` and `csrc/runtime/memory.cc`.
- Worker `Runtime` in `csrc/runtime/runtime.hpp` and `csrc/runtime/runtime.cc`.
- Graph tensor capture behavior in `csrc/execution/graph/*`.
- Python bindings in `csrc/pybind11/runtime/tensor.hpp` and
  `python/infinilm/_core/tensor.py`.

InfiniOps currently aliases:

- `infini::ops::Tensor` to `infini::rt::TensorView` in `src/tensor.h`.
- `DataType`, `Device`, and runtime templates to their RT equivalents.
- `src/tensor.cc` is obsolete and excluded by CMake. Remove it during this
  migration rather than reviving it.

InfiniCCL does not use a tensor object. Its C API accepts raw pointers, element
counts, `infinicclDataType_t`, communicators, and streams. InfiniLM performs
dtype conversion at the communication boundary.

A baseline search of current InfiniLM found approximately:

- 207 source files referencing the LM tensor type.
- 1,693 lines naming `infinilm::runtime::Tensor` or `runtime::Tensor`.
- 1,815 common tensor method calls.

Those numbers are why the migration uses a compatibility alias and small,
verifiable stages. A one-shot replacement is not acceptable.

## 3. Architecture Rules

### 3.1 Dependency Direction

The only allowed dependency direction is:

```text
InfiniLM -> InfiniOps -> InfiniRT
InfiniLM -> InfiniRT
InfiniLM -> InfiniCCL
```

RT must not include InfiniOps, InfiniLM, or InfiniCCL headers. In particular,
RT owning tensor code must not call `Fill`, `Copy`, or any other operator.

Some current LM tensor convenience functions call Ops:

- `Tensor::ones()` calls `infini::ops::Fill`.
- `Tensor::contiguous()` and several copy paths call `infini::ops::Copy`.

Those functions cannot move verbatim into RT. They remain LM adapters until RT
has genuine runtime primitives or the APIs are deliberately relocated with a
non-Ops implementation.

### 3.2 Owning and Non-Owning Split

`TensorView` remains the non-owning interchange type:

```text
data pointer + shape + strides + dtype + device
```

The RT owning tensor is a value type over a shared implementation:

```text
shared TensorImpl -> metadata + storage/offset
shared Memory    -> raw bytes + device + deleter + ownership flags
```

The owning tensor must expose a cheap `view()` conversion. Ops continues to
receive views, not owning objects, unless a separate explicit API decision is
made later.

### 3.3 Metadata Representation

The canonical shape and stride types are the existing `TensorView` vector-like
types with inline capacity eight. The new owning tensor reuses them rather than
introducing another `std::vector` metadata layout.

Migration costs are expected at LM call sites that currently assume:

```cpp
const std::vector<size_t>&
```

The update must preserve range iteration, indexed access, `.size()`, conversion
to Python sequences, explicit construction from common ranges, and view lifetime
guarantees. If a staged port needs a temporary conversion boundary, record it
and remove it before completion.

### 3.4 Allocation Owner Identity

RT storage must not refer to `infinilm::runtime::Runtime`. RT `Memory` may store
an opaque owner token such as `std::weak_ptr<void>`. An LM runtime can create
that token from its `shared_ptr<Runtime>`, and a typed LM helper may recover it
with `std::static_pointer_cast` only at the LM boundary.

RT treats the token as identity, not as a stream-synchronization or producer
guarantee. This preserves the existing distinction between allocation identity,
last-writer completion, stream ownership, and graph retention.

### 3.5 External Storage and Lifetime

`from_blob` with an optional owner remains required. The owning tensor must keep
the external owner alive. Safetensors, NumPy, Torch, and graph blob paths depend
on this behavior.

Never publish a raw pointer beyond a call unless an owner is retained by a
tensor or graph object, the caller has proven completion, or the API is
explicitly unsafe and named accordingly.

### 3.6 Graph and Stream Safety

LM graph capture relies on pinned and graph-pinnable memory flags,
`ReusableMemoryPool::retain_for_capture`, blob conversion and reinstantiation,
worker-thread ownership, stream-ordered release, and graph lifetime leases.

Storage flags and opaque identity may move to RT. The LM graph manager and
worker policy remain in LM during this migration. Any graph behavior change
needs NVIDIA graph tests before the Core manifest changes.

### 3.7 CCL Boundary

CCL remains a C-style raw-pointer API. Do not make CCL depend on RT tensor
objects in this migration. LM continues to validate contiguous storage,
matching dtype and runtime device, supported CCL dtype mapping, stream, and
communicator ownership. Centralize and test the RT-to-CCL dtype conversion.

## 4. Target Public Surface

The initial target is an RT surface similar to the following. Exact names may
change only through a decision-log entry before implementation.

```cpp
namespace infini::rt {

class Memory {
 public:
  using Deleter = std::function<void(std::byte*)>;

  Memory(std::byte* data, std::size_t size, Device device, Deleter deleter,
         bool pinned = false, bool graph_pinnable = false,
         std::weak_ptr<void> allocation_owner = {});

  std::byte* data();
  const std::byte* data() const;
  Device device() const;
  std::size_t size() const;
  bool is_pinned() const;
  bool is_graph_pinnable() const;
  std::shared_ptr<void> allocation_owner() const;
};

class Tensor {
 public:
  using Size = TensorView::Size;
  using Stride = TensorView::Stride;
  using Shape = TensorView::Shape;
  using Strides = TensorView::Strides;

  Tensor() = default;

  static Tensor from_blob(void* data, const Shape& shape,
                          const DataType& dtype, const Device& device);
  static Tensor from_blob(void* data, const Shape& shape,
                          const DataType& dtype, const Device& device,
                          std::shared_ptr<void> owner);
  static Tensor strided_from_blob(/* same plus strides */);

  TensorView view() const;
  Tensor squeeze(std::size_t dim) const;
  Tensor unsqueeze(std::size_t dim) const;
  Tensor narrow(std::vector<Slice> slices) const;
  Tensor permute(const Shape& order) const;
  Tensor reshape(const Shape& shape) const;
  Tensor as_strided(const Shape& shape, const Strides& strides) const;

  void* data();
  const void* data() const;
  ShapeView shape() const;
  StridesView strides() const;
  DataType dtype() const;
  Device device() const;
  bool is_contiguous() const;
  Size numel() const;
  Size ndim() const;
  Size size(std::size_t dim) const;
  Stride stride(std::size_t dim) const;
  std::size_t element_size() const;
  std::size_t nbytes() const;
};

}  // namespace infini::rt
```

Allocation factories enter through an explicit allocator/context boundary
rather than by making RT depend on LM Runtime. Computation convenience APIs
stay in LM until a separate design adds runtime-native implementations.

## 5. Construction Stages

Do not start the next stage while the current stage has failed evidence or an
unresolved blocking decision.

### M0. Plan and Baseline

Status: complete

Deliverables:

- This document.
- The coherent revision table.
- Baseline inventory and dependency rules.

Exit criteria:

- The document is committed on the RT migration branch.
- Participating repositories are clean or contain only documented changes.
- A reviewer can reproduce the starting tree from the revision table.

### M1. Contract and Branch Setup

Status: ready

Tasks:

- **T1.1** Create implementation branches: RT `feat/rt-owning-tensor`, Ops
  `feat/rt-owning-tensor`, LM `feat/adopt-rt-owning-tensor`, and Core manifest
  changes only after component commits exist.
- **T1.2** Add an API contract or compile-only consumer for `Memory`, owning
  `Tensor`, `Tensor::view()`, and all metadata/view operations.
- **T1.3** Confirm generated public-header installation rules for every new
  header.
- **T1.4** Record final names for `Memory`, `Tensor`, slices, allocator/context,
  and owner-token semantics in the decision log.

Exit criteria:

- RT CPU build and tests pass with no runtime behavior change beyond the new
  contract surface.
- `test_install_consumer` consumes the new headers from an installed prefix.
- No production caller has yet been migrated.

### M2. RT Storage and Tensor Core

Status: not started

Tasks:

- **T2.1** Move the storage-object concept from LM into RT without LM types.
- **T2.2** Move metadata validation, storage-span calculation, zero-size
  handling, negative-stride rejection, and overflow checks.
- **T2.3** Implement owning tensor construction from external blobs and owners.
- **T2.4** Implement pure view operations: squeeze, unsqueeze, narrow, permute,
  reshape/view, and as_strided.
- **T2.5** Test rank 0, empty tensors, ranks above eight, broadcast strides,
  custom non-negative strides, overflow, shared storage, owner lifetime, and
  data-pointer offset.
- **T2.6** Add ABI/layout checks or static assertions that document object
  boundaries and prevent accidental inline-field churn.

Forbidden in this stage: allocating through LM Runtime, calling Ops, changing
Python behavior, or changing graph capture behavior.

Exit criteria:

- RT tests cover the complete metadata/view matrix.
- RT tests pass on CPU and at least one accelerator build.
- Ownership tests prove deleter execution and external-owner retention.

### M3. Allocator Bridge

Status: not started

Tasks:

- **T3.1** Define an RT allocator/context interface for ordinary device memory,
  parameter memory, and pinned host memory.
- **T3.2** Keep the interface non-operator-specific and non-LM-specific.
- **T3.3** Adapt LM worker Runtime as the first implementation, using its
  existing `ReusableMemoryPool`.
- **T3.4** Move empty and strided-empty factories to RT through the allocator.
- **T3.5** Test stream-ordered allocation and release.
- **T3.6** Decide whether `empty_parameter` remains LM-specific or becomes a
  general storage policy.

Exit criteria:

- LM constructs RT owning tensors through the bridge while preserving worker and
  stream rules.
- RT does not include LM or Ops headers.
- CPU and NVIDIA allocation tests pass.

### M4. InfiniLM Compatibility Adoption

Status: not started

Tasks:

- **T4.1** Introduce temporary LM aliases:

  ```cpp
  namespace infinilm::runtime {
  using Tensor = infini::rt::Tensor;
  using Memory = infini::rt::Memory;
  }
  ```

- **T4.2** Keep LM-specific runtime helpers in a clearly named adapter file.
- **T4.3** Mechanically update shape/stride signatures and call sites.
- **T4.4** Update pybind holders, return policies, Python wrapper ownership,
  NumPy conversion, Torch conversion, and safetensors borrowing.
- **T4.5** Preserve the public Python API unless a change is explicitly approved.
- **T4.6** Run native tensor tests, Python runtime tests, checkpoint tests, and
  model smoke tests.

Exit criteria:

- LM compiles and runs with RT owning tensors as its only LM tensor alias.
- Python users see no unintended API regression.
- No duplicate LM tensor implementation remains active.

### M5. Copy, Fill, and Compute Boundary

Status: not started

Tasks:

- **T5.1** Move no Ops-backed method into RT.
- **T5.2** Move current copy, contiguous, and ones behavior into LM adapter
  functions if class membership must change.
- **T5.3** Update LM call sites to those adapters.
- **T5.4** Evaluate which contiguous copies can use RT runtime memcpy directly.
- **T5.5** Design a separate RT-native fill/copy proposal only after the main
  migration is stable.

Exit criteria:

- RT contains no Ops dependency.
- LM transfer semantics and error handling are unchanged on CPU and NVIDIA.
- Graph-recording refusal rules for prohibited host observation remain intact.

### M6. Graph Retention and Blob Semantics

Status: not started

Tasks:

- **T6.1** Move storage flags and opaque owner identity into RT Memory.
- **T6.2** Keep `GraphTensor`, graph segmentation, and worker policy in LM.
- **T6.3** Express blob conversion and reinstantiation through the allocator
  bridge or a documented LM adapter.
- **T6.4** Test capture retention, logical ownership count, graph lifetime,
  replay failure retirement, and cleanup drain.
- **T6.5** Compare graph execution reports and memory budgets before and after.

Exit criteria:

- NVIDIA graph capture/replay and eager output equivalence tests pass.
- No graph outlives allocations on which it depends.
- No non-capturable allocation is reused during capture.

### M7. InfiniOps Integration

Status: not started

Tasks:

- **T7.1** Build Ops against an installed RT migration prefix.
- **T7.2** Confirm `infini::ops::Tensor` remains an alias to
  `infini::rt::TensorView`.
- **T7.3** Remove obsolete `src/tensor.cc` and dead old-class symbols.
- **T7.4** Audit NineToothed and Torch adapters for shape pointer lifetime and
  metadata assumptions.
- **T7.5** Run CPU operator tests and an NVIDIA operator subset.

Exit criteria:

- Ops compiles and passes tests with no local tensor implementation.
- Generated wrappers consume canonical RT metadata.
- No accidental owner copy occurs at the Ops boundary.

### M8. CCL, Distributed LM, and Dtype Mapping

Status: not started

Tasks:

- **T8.1** Centralize RT-to-CCL dtype conversion in one LM helper.
- **T8.2** Add exhaustive dtype mapping tests, including unsupported `kBool`.
- **T8.3** Run single-process communication validation and multi-process TP
  smoke where hardware is available.
- **T8.4** Validate runtime owner and thread fail-stop behavior before CCL calls.
- **T8.5** Confirm no non-contiguous or wrong-device tensor reaches CCL.

Exit criteria: all supported mappings are tested, TP inference smoke passes, and
CCL remains independent of RT object layout.

### M9. End-to-End Validation

Status: not started

Minimum matrix:

1. RT CPU full tests and installed-consumer test.
2. RT NVIDIA full tests, memory-pool graph test, and installed-consumer test.
3. Ops CPU tests against installed RT.
4. Ops NVIDIA representative operator tests against installed RT.
5. LM CPU native tensor and runtime tests.
6. LM NVIDIA model inference smoke test.
7. LM graph capture/replay equivalence test.
8. LM Python API and checkpoint tests.
9. LM TP/CCL smoke test where available.
10. Performance comparison against the pre-migration baseline.

Record model shape, dtype, batch, input length, output length, hardware, graph
mode, and comparison method. Local model paths may be omitted.

Exit criteria: every required test passes, no forbidden dependency is found,
performance is within the approved threshold or has an approved follow-up, and
public docs and compatibility notes are updated.

### M10. Core Pin and Release Handoff

Status: not started

Tasks:

- **T10.1** Update Core submodule gitlinks only after RT and Ops commits exist.
- **T10.2** Update LM's Core submodule only after the Core manifest commit exists.
- **T10.3** Rebuild clean prefixes in RT, Ops, CCL, LM order.
- **T10.4** Run the final matrix on the exact pinned revisions.
- **T10.5** Mark all ledger entries complete and archive temporary compatibility
  decisions.

Exit criteria: a fresh machine reproduces the result using recorded commits,
submodule status is clean, and no migration branch is needed to build the stack.

## 6. Task Ledger

Status values are `Ready`, `In progress`, `Blocked`, `Complete`, and `Dropped`.
Every completed task must have an owner, revision or commit, evidence, and
remaining work.

| ID | Status | Scope | Exit evidence | Notes |
| --- | --- | --- | --- | --- |
| M0 | Complete | Plan and baseline | This document on `feat/rt-owning-tensor` | Baseline revisions recorded |
| T1.1 | Complete | Branch setup | Branches listed in evidence log | Local integration branches created |
| T1.2 | Ready | API contract tests | RT CPU `ctest` output | No production caller migration yet |
| T1.3 | Ready | Public header/install rules | `test_install_consumer` output | Coordinate with generated headers |
| T1.4 | Ready | API naming decision | Decision-log entry | Resolve before M2 |
| T2.* | Ready | RT storage and tensor core | M2 exit tests | Do after M1 |
| T3.* | Not started | Allocator bridge | M3 exit tests | Depends on M2 |
| T4.* | Not started | LM adoption | LM native and Python tests | Depends on M3 |
| T5.* | Not started | Copy/fill boundary | Dependency scan and transfer tests | No RT-to-Ops dependency |
| T6.* | Not started | Graph and blob semantics | NVIDIA graph tests | Highest-risk stage |
| T7.* | Not started | Ops integration | Ops CPU and NVIDIA tests | Depends on installed RT |
| T8.* | Not started | CCL, dtype, distributed | Mapping and TP tests | Keep CCL C ABI |
| T9 | Not started | End-to-end matrix | Complete evidence table | Includes performance |
| T10 | Not started | Core pin and release | Clean submodule build | Last stage only |

## 7. Evidence Log

Append new entries at the top. Never edit old evidence to make a failed run look
successful.

### 2026-11-23 - Plan initialization

- Branch: RT `feat/rt-owning-tensor`, based on `c0810cfe`.
- Related branches:
  - Ops `feat/rt-owning-tensor`, based on `f52af91a`.
  - LM `feat/adopt-rt-owning-tensor`, based on `ebb081ef`.
- Work completed:
  - Reviewed current RT, LM, Ops, and CCL tensor boundaries.
  - Recorded dependency, graph, ownership, ABI, and dtype risks.
  - Created this staged migration plan.
  - Created local RT, Ops, and LM integration branches.
- Evidence:
  - This commit contains the plan and documentation link.
- Remaining:
  - Implement M1 contract tests and generated-header rules.
  - Push or bundle the local branches before handing off to another machine.

## 8. Decision Log

Append decisions with new `D#` numbers. A decision changes only when a later
decision supersedes it.

### D1. TensorView remains the Ops interchange type

Date: 2026-11-23

Decision: Ops APIs continue to receive `infini::rt::TensorView`.

Rationale: this preserves operator ABI, avoids ownership ambiguity, and keeps
the compute boundary narrow.

### D2. RT must not depend on Ops or LM

Date: 2026-11-23

Decision: migrated RT source may not include InfiniOps, InfiniLM, or InfiniCCL.

Rationale: RT is the bottom runtime layer. LM convenience operations that call
Ops remain in LM until a separate RT-native primitive design exists.

### D3. Canonical metadata is the RT inline-capacity representation

Date: 2026-11-23

Decision: the owning tensor uses existing RT `TensorView` metadata types.

Rationale: this avoids two tensor layouts and preserves the metadata allocation
optimization.

### D4. CCL remains a raw-pointer C API

Date: 2026-11-23

Decision: do not couple CCL to the RT owning tensor in this migration.

Rationale: storage lifetime and stream correctness already dominate the risk;
changing the CCL ABI adds unnecessary blast radius.

## 9. Risk Register

| Risk | Impact | Early warning | Mitigation |
| --- | --- | --- | --- |
| RT depends on Ops | Architecture collapse | forbidden include scan matches | Dependency gate after every RT stage |
| One-shot LM replacement | Long unbuildable period | Multiple milestones broken in LM | Compatibility alias and mechanical batches |
| Shape representation mismatch | Widespread compile failures | Python or vector-reference errors | Contract tests before migration |
| Dangling external owner | Torch or safetensors crashes | Use-after-free in Python tests | Owner-retention tests and sanitizer where available |
| Graph lifetime error | Replay corruption or late free | Flaky graph tests | Retain and reinstantiate tests; no capture-time cache miss |
| Stream-order regression | Silent data races | Intermittent output mismatch | Worker tests and graph/eager comparisons |
| ABI drift | Downstream layout failures | Installed-consumer failures | Install test for every public-header change |
| Dtype drift | CCL or Python conversion errors | Unsupported or wrong dtype | Exhaustive mapping tests including `kBool` |
| Version mixing | Unrelated build failures | Dirty submodule status | Exact revisions and clean-prefix builds |
| Performance regression | Migration blocked after correctness | Tensor-heavy benchmark slowdown | Baseline comparison before redesign |

## 10. Required Verification Commands

Record exact adapted paths and results in the evidence log.

### RT CPU

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CPU=ON \
  -DINFINI_RT_BUILD_TESTING=ON
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

### RT NVIDIA

```bash
cmake -S . -B build-nvidia \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CPU=ON \
  -DWITH_NVIDIA=ON \
  -DINFINI_RT_BUILD_TESTING=ON
cmake --build build-nvidia -j
ctest --test-dir build-nvidia --output-on-failure
```

### RT dependency-direction audit

```bash
rg -n 'infini/ops|infinilm|infiniccl' include src tests
```

Expected result before an explicit documented exception: no implementation-code
matches.

### Ops against installed RT

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/path/to/prefix \
  -DINFINI_RT_ROOT=/path/to/prefix \
  -DWITH_CPU=ON
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
cmake --install build-cpu
```

### LM native and Python tests

Use one coherent installation prefix and matching C++ ABI:

```bash
export INFINI_ROOT=/absolute/path/to/coherent-prefix
xmake f -y -m release --cxx11-abi=1
xmake build -j _infinilm
python -m pytest test/runtime test/scripts -q
```

Model smoke evidence must additionally record device, model shape, dtype,
batch, input/output lengths, graph mode, and comparison method.

## 11. Session and Handoff Protocol

Every work session must:

1. Read this document before changing code.
2. Confirm repository revisions and `git status` output.
3. Select one Ready task, or continue exactly one In-progress task.
4. Record session intent in the evidence log before implementation.
5. Make small commits that compile independently whenever practical.
6. Run the milestone's required tests.
7. Update task status, evidence, decisions, risks, and remaining work.
8. Leave each repository clean or with a precisely documented branch and state.
9. Push branches, or create bundles or patches if the network is unavailable.
10. Mark Blocked only after the blocker is specific and attempted mitigations
    are recorded.

Suggested evidence entry:

```markdown
### YYYY-MM-DD - Short session title

- Branches and revisions:
  - RT: `<branch>` / `<commit>`
  - Ops: `<branch>` / `<commit>`
  - LM: `<branch>` / `<commit>`
- Work completed:
  - ...
- Commands and results:
  - ...
- Current blockers:
  - none, or a precise blocker.
- Next handoff step:
  - ...
```

## 12. Stop Conditions

Stop and stabilize instead of continuing when:

1. RT needs an InfiniOps or InfiniLM include to progress.
2. A graph or transfer test becomes intermittently wrong.
3. Ownership lifetime cannot be proven for a raw-pointer path.
4. Public ABI changes are discovered outside compatibility documentation.
5. A submodule pointer must target an unavailable or uncommitted revision.
6. LM behavior changes without an explicit test and decision-log entry.
7. More than one major milestone is simultaneously broken.

## 13. Completion Definition

The migration is complete only when:

1. every Section 6 task is Complete or explicitly Dropped with rationale;
2. RT owns the canonical Memory and Tensor implementation;
3. Ops exposes no competing tensor implementation;
4. LM has no active private Tensor or Memory implementation;
5. RT, Ops, and LM public docs describe the final boundary;
6. clean-prefix RT, Ops, CCL, and LM builds pass;
7. CPU and NVIDIA end-to-end tests pass;
8. graph and TP smoke tests pass where applicable;
9. performance evidence is recorded;
10. Core and LM pins target committed, fetchable revisions;
11. a fresh contributor can continue or audit the work using this file,
    recorded commits, and standard repository tooling.
