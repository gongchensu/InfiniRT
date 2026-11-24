# Owning Tensor 迁移施工计划

状态：施工中

最后更新：2026-11-23

负责人：InfiniRT / InfiniLM runtime 维护者

本文档是把 InfiniLM 中现有的 owning Tensor / Memory 基础迁移到 InfiniRT 的唯一施工依据。它同时是断联交接文档：每次实际开发后，必须同步更新任务台账、证据日志和决策记录，让别人可以只依赖本文档和已记录的 commit 继续工作。

## 1. 目标和最终形态

迁移完成后的目标形态是：

1. `infini::rt::Device`、`infini::rt::DataType`、shape/stride 元数据以及 `infini::rt::TensorView` 继续作为统一的非拥有张量描述。
2. InfiniRT 拥有可复用的 owning Tensor 实现，包括存储所有权、元数据校验、视图构造和生命周期语义。
3. InfiniOps 继续接收 `infini::rt::TensorView`，不重新定义或复制一套 Tensor 类型。
4. InfiniLM 用 RT alias 或薄适配层替换私有 `infinilm::runtime::Tensor` 实现，同时保持 Python API、worker Runtime 规则、图行为和推理语义不变。
5. 内存分配、stream 顺序、graph retention 和外部存储所有权在 RT、Ops、LM、CCL 边界上都有显式定义和测试。

这是一次 runtime 边界迁移，不是把算子搬进 RT，也不是重做 LM 调度器。

## 2. 当前基线

计划从下面这组互相匹配的 revision 开始。复现基线证据时，不要把它们和本地旧 checkout 混用。

| 仓库 | 分支 | Revision |
| --- | --- | --- |
| InfiniLM | `main` | `ebb081ef129ee8405e2e0863f40441367a40f85c` |
| InfiniCore | `main` | `64362228212c77422124574af4d7339d728610c7` |
| InfiniRT | `master` | `c0810cfef8f4e4c00a3bc404c2831428ac37b9bc` |
| InfiniOps | `master` | `f52af91a83f19610b16120ab1be4f5c884c02c1c` |
| InfiniCCL | `master` | `51cc9db31d58aac21643d72f0f77cf5c9ecfaee5` |

### 2.1 现状盘点

InfiniRT 当前提供：

- `src/data_type.h` 中的 `DataType`，包含独立的 `kBool`。
- `src/device.h` 中的 `Device`。
- `src/tensor_view.h` 和 `src/tensor_view.cc` 中的非拥有 `TensorView`。
- `src/common/shape_strides_storage.h` 中的 inline shape/stride 存储。
- `include/infini/rt/reusable_memory_pool.h` 和 `src/reusable_memory_pool.cc` 中的 `ReusableMemoryPool`。
- `include/infini/rt/device_guard.h` 中的 `DeviceGuard`。

InfiniLM 当前提供：

- `csrc/runtime/tensor.hpp`、`csrc/runtime/tensor/tensor.cc`、`csrc/runtime/tensor/view.cc`、`csrc/runtime/tensor/copy.cc` 中的 owning `Tensor` 和 `TensorImpl`。
- `csrc/runtime/memory.hpp` 和 `csrc/runtime/memory.cc` 中的 `Memory`。
- `csrc/runtime/runtime.hpp` 和 `csrc/runtime/runtime.cc` 中的 worker `Runtime`。
- `csrc/execution/graph/*` 中的 GraphTensor 和捕获行为。
- `csrc/pybind11/runtime/tensor.hpp` 和 `python/infinilm/_core/tensor.py` 中的 Python 绑定。

InfiniOps 当前只是 alias：

- `src/tensor.h` 中 `infini::ops::Tensor` alias 到 `infini::rt::TensorView`。
- `DataType`、`Device` 和 runtime 模板均 alias 到 RT 等价类型。
- `src/tensor.cc` 是旧实现残留，且已被 CMake 排除。迁移中应删除，而不是恢复。

InfiniCCL 不使用 Tensor 对象。它的 C API 接收裸指针、元素数、`infinicclDataType_t`、communicator 和 stream。InfiniLM 在通信边界完成 dtype 转换。

对当前 InfiniLM 的基线搜索结果约为：

- 207 个源码文件引用 LM Tensor 类型。
- 1693 行显式出现 `infinilm::runtime::Tensor` 或 `runtime::Tensor`。
- 1815 处常见 Tensor 方法调用。

这个规模决定了迁移必须使用兼容 alias 和小步验证，不能一次性替换。

## 3. 架构规则

### 3.1 依赖方向

唯一允许的依赖方向是：

```text
InfiniLM -> InfiniOps -> InfiniRT
InfiniLM -> InfiniRT
InfiniLM -> InfiniCCL
```

RT 不得 include InfiniOps、InfiniLM 或 InfiniCCL 头文件。尤其不允许 RT owning Tensor 代码调用 `Fill`、`Copy` 或任何其他算子。

当前 LM Tensor 便捷 API 中有算子依赖：

- `Tensor::ones()` 调用 `infini::ops::Fill`。
- `Tensor::contiguous()` 和多个 copy 路径调用 `infini::ops::Copy`。

这些函数不能原样搬进 RT。它们先留在 LM 适配层，直到 RT 后续有真正 runtime primitive，或另行设计非 Ops 实现。

### 3.2 拥有与非拥有边界

`TensorView` 继续作为非拥有交换类型：

```text
data pointer + shape + strides + dtype + device
```

新的 RT owning Tensor 是基于 shared implementation 的值类型：

```text
shared TensorImpl -> metadata + storage/offset
shared Memory    -> raw bytes + device + deleter + ownership flags
```

Owning Tensor 必须提供低开销的 `view()` 转换。Ops 继续接收 view，不接收 owning 对象；除非未来有单独且显式的 API 决策。

### 3.3 元数据表示

规范 shape/stride 类型继续使用现有 `TensorView` 的 vector-like 类型，inline capacity 为 8。新的 owning Tensor 复用它们，不引入另一套 `std::vector` 元数据布局。

LM 现有代码中大量签名假设：

```cpp
const std::vector<size_t>&
```

迁移时必须保留：

- range 迭代；
- 下标访问；
- `.size()`；
- Python sequence 转换；
- 从常见 range 显式构造；
- view accessor 的生命周期语义。

如果阶段性移植需要临时 conversion boundary，必须记录，并在完成前删除。

### 3.4 Allocation Owner Identity

RT storage 不得引用 `infinilm::runtime::Runtime`。RT `Memory` 可以保存 opaque owner token，例如：

```cpp
std::weak_ptr<void> allocation_owner_;
```

LM Runtime 可以从自己的 `shared_ptr<Runtime>` 创建该 token。只有 LM 边界可以用 `std::static_pointer_cast` 恢复具体类型。

RT 只把 token 当作 allocation identity，不把它当作 stream 同步或 producer 完成证明。这样保留现有语义区别：

- allocation identity；
- last writer / producer completion；
- stream ownership；
- graph retention。

### 3.5 外部存储和生命周期

带 optional owner 的 `from_blob` 是必须保留的语义。Owning Tensor 必须保活外部 owner。Safetensors、NumPy、Torch 和 graph blob 路径都依赖它。

除非满足以下条件之一，裸指针不得逃逸出当前调用：

1. Tensor 或 Graph 对象持有 owner；
2. 调用方已经明确证明外部写入完成；
3. API 本身显式声明 unsafe，并且命名能表达风险。

### 3.6 Graph 和 Stream 安全

LM graph capture 当前依赖：

- pinned / graph-pinnable 内存标记；
- `ReusableMemoryPool::retain_for_capture`；
- blob 转换和恢复；
- worker 线程所有权；
- stream-ordered release；
- graph lifetime lease。

Storage 标记和 opaque identity 可以移入 RT。LM GraphManager 和 worker 策略在本次迁移中留在 LM。任何 graph 行为变化都必须先通过 NVIDIA graph 测试，再更新 Core manifest。

### 3.7 CCL 边界

CCL 继续保持 C 风格裸指针 API。本次迁移不让 CCL 依赖 RT Tensor 对象。LM 继续校验：

- contiguous storage；
- dtype 匹配；
- runtime device 匹配；
- CCL dtype 支持；
- stream 和 communicator 所有权。

RT `DataType` 到 `infinicclDataType_t` 的转换应集中并测试。

## 4. 目标公共接口

初始目标接口如下。实现前如需修改命名，必须先在决策日志记录。

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
  static Tensor strided_from_blob(/* 同上，并带 strides */);

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

分配工厂通过显式 allocator/context 边界引入，不让 RT 依赖 LM Runtime。计算型便捷 API 先留在 LM，后续再单独设计 runtime-native 实现。

## 5. 施工阶段

当前阶段存在失败证据或未决阻塞决策时，不得进入下一阶段。

### M0. 计划和基线

状态：完成

交付物：

- 本文档。
- 一致 revision 表。
- 基线盘点和依赖规则。

退出标准：

- 文档已提交到 RT 迁移分支。
- 参与仓库干净，或只包含已记录变更。
- 审阅者能按 revision 表复现起始状态。

### M1. 契约和分支准备

状态：完成

任务：

- **T1.1** 创建实现分支：RT、Ops、LM、Core、CCL 均使用 `feat/rt-owning-tensor`，不使用额外 worktree。Core manifest 只有在组件 commit 存在后才修改。
- **T1.2** 为 `Memory`、owning `Tensor`、`Tensor::view()` 和全部 metadata/view 操作增加 API contract 或 compile-only consumer。
- **T1.3** 确认每个新头文件的 public header 生成和安装规则。
- **T1.4** 在决策日志中确定 `Memory`、`Tensor`、slice、allocator/context、owner token 的最终命名和语义。

退出标准：

- RT CPU 构建和测试通过，除新增 contract surface 外无运行时行为变化。
- `test_install_consumer` 能从安装前缀消费新头文件。
- 尚未迁移任何 production caller。

### M2. RT Storage 和 Tensor Core

状态：完成

任务：

- **T2.1** 将 LM 的 storage object 概念移入 RT，且不引入 LM 类型。
- **T2.2** 迁移 metadata 校验、storage-span 计算、zero-size 处理、负 stride 拒绝和溢出检查。
- **T2.3** 实现基于外部 blob 和 owner 的 owning Tensor 构造。
- **T2.4** 实现纯 view 操作：squeeze、unsqueeze、narrow、permute、reshape/view、as_strided。
- **T2.5** 测试 rank 0、empty tensor、rank 大于 8、broadcast stride、自定义非负 stride、溢出、共享存储、owner 生命周期和 data pointer offset。
- **T2.6** 增加 ABI/layout 检查或 static assertion，说明对象边界并避免无意的 inline field churn。

本阶段禁止：

- 通过 LM Runtime 分配；
- 调用 Ops；
- 改 Python 行为；
- 改 graph capture 行为。

退出标准：

- RT 测试覆盖完整 metadata/view 矩阵。
- CPU 和至少一个加速器构建的 RT 测试通过。
- ownership 测试证明 deleter 执行和 external owner 保活。

### M3. Allocator Bridge

状态：进行中

任务：

- **T3.1** 为普通 device memory、parameter memory 和 pinned host memory 定义 RT allocator/context 接口。
- **T3.2** 接口不得绑定具体算子或 LM 类型。
- **T3.3** 将 LM worker Runtime 适配为第一个实现，继续使用现有 `ReusableMemoryPool`。
- **T3.4** 通过 allocator 将 empty 和 strided-empty 工厂迁到 RT。
- **T3.5** 测试 stream-ordered allocation 和 release。
- **T3.6** 决定 `empty_parameter` 是保留 LM 专属工厂，还是变成通用 storage policy。

退出标准：

- LM 能通过 bridge 构造 RT owning Tensor，并保持 worker 和 stream 规则。
- RT 不 include LM 或 Ops 头文件。
- CPU 和 NVIDIA allocation 测试通过。

### M4. InfiniLM 兼容接入

状态：进行中

任务：

- **T4.1** 引入临时 LM alias：

  ```cpp
  namespace infinilm::runtime {
  using Tensor = infini::rt::Tensor;
  using Memory = infini::rt::Memory;
  }
  ```

- **T4.2** 将 LM 专属 runtime helper 放进命名清晰的适配文件。
- **T4.3** 机械式更新 shape/stride 签名和调用点。
- **T4.4** 更新 pybind holder、return policy、Python wrapper ownership、NumPy 转换、Torch 转换和 safetensors 借用。
- **T4.5** 除非明确批准，保持公共 Python API 不变。
- **T4.6** 运行 native tensor 测试、Python runtime 测试、checkpoint 测试和模型 smoke 测试。

退出标准：

- LM 编译并运行时，RT owning Tensor 是唯一 LM Tensor alias。
- Python 用户没有非预期 API 回归。
- LM 不再有活跃的私有 Tensor 实现。

### M5. Copy、Fill 和计算边界

状态：未开始

任务：

- **T5.1** 不把任何 Ops-backed 方法移入 RT。
- **T5.2** 如果必须改变类归属，将当前 copy、contiguous、ones 行为移入 LM adapter 函数。
- **T5.3** 将 LM 调用点更新到这些 adapter。
- **T5.4** 评估哪些 contiguous copy 可以直接使用 RT runtime memcpy。
- **T5.5** 主迁移稳定后，再单独设计 RT-native fill/copy 方案。

退出标准：

- RT 没有 Ops 依赖。
- LM transfer 语义和错误处理在 CPU 与 NVIDIA 上不变。
- 禁止 host observation 的 graph recording 规则不变。

### M6. Graph Retention 和 Blob 语义

状态：未开始

任务：

- **T6.1** 将 storage 标记和 opaque owner identity 移入 RT Memory。
- **T6.2** `GraphTensor`、graph segmentation 和 worker policy 留在 LM。
- **T6.3** 通过 allocator bridge 或明确记录的 LM adapter 表达 blob 转换和恢复。
- **T6.4** 测试 capture retention、logical ownership count、graph lifetime、replay failure retirement 和 cleanup drain。
- **T6.5** 对比迁移前后的 graph execution report 和 memory budget。

退出标准：

- NVIDIA graph capture/replay 和 eager 输出等价测试通过。
- Graph 不可能比其依赖 allocation 活得更久。
- capture 中不会重用不可捕获的 allocation。

### M7. InfiniOps 接入

状态：未开始

任务：

- **T7.1** 用安装后的 RT migration prefix 构建 Ops。
- **T7.2** 确认 `infini::ops::Tensor` 仍 alias 到 `infini::rt::TensorView`。
- **T7.3** 删除废弃的 `src/tensor.cc` 和旧 class 死符号。
- **T7.4** 审计 NineToothed 和 Torch adapter 对 shape 指针生命周期及 metadata 表示的假设。
- **T7.5** 运行 CPU 算子测试和一个 NVIDIA 算子子集。

退出标准：

- Ops 无本地 Tensor 实现，仍能编译并通过测试。
- generated wrapper 消费同一套 RT metadata。
- Ops 边界不发生无意的 owner 复制。

### M8. CCL、分布式 LM 和 Dtype 映射

状态：未开始

任务：

- **T8.1** 将 RT 到 CCL 的 dtype 转换集中到一个 LM helper。
- **T8.2** 增加穷举 dtype mapping 测试，包括不支持的 `kBool`。
- **T8.3** 在硬件可用时运行单进程通信验证和多进程 TP smoke。
- **T8.4** 在 CCL 调用前验证 runtime owner 和 thread fail-stop 行为。
- **T8.5** 确认非 contiguous 或错误 device 的 Tensor 不会进入 CCL。

退出标准：所有支持的 dtype mapping 有测试，TP 推理 smoke 通过，CCL 不依赖 RT 对象布局。

### M9. 端到端验证

状态：未开始

最低矩阵：

1. RT CPU 全量测试和 installed-consumer 测试。
2. RT NVIDIA 全量测试、memory-pool graph 测试和 installed-consumer 测试。
3. Ops 基于安装后 RT 的 CPU 测试。
4. Ops 基于安装后 RT 的 NVIDIA 代表性算子测试。
5. LM CPU native tensor/runtime 测试。
6. LM NVIDIA 模型推理 smoke。
7. LM graph capture/replay 输出等价测试。
8. LM Python API 和 checkpoint 测试。
9. 硬件可用时的 LM TP/CCL smoke。
10. 与迁移前基线的性能对比。

必须记录模型 shape、dtype、batch、input length、output length、硬件、graph mode 和结果对比方式。本地模型路径可以省略。

退出标准：必需测试全部通过；未发现禁止依赖；性能在批准阈值内或已有批准的后续处理；公共文档和 compatibility 说明已更新。

### M10. Core Pin 和发布交接

状态：未开始

任务：

- **T10.1** 只有 RT 和 Ops commit 存在后，才更新 Core submodule gitlink。
- **T10.2** 只有 Core manifest commit 存在后，才更新 LM 的 Core submodule。
- **T10.3** 按 RT、Ops、CCL、LM 顺序使用干净 prefix 重建。
- **T10.4** 在最终 pinned revision 上重跑完整矩阵。
- **T10.5** 将任务台账全部置为完成，并归档临时兼容决策。

退出标准：新机器能只根据记录的 commit 复现结果；submodule 状态干净；最终 stack 构建不依赖迁移分支。

## 6. 任务台账

状态取值：

- `Ready`
- `In progress`
- `Blocked`
- `Complete`
- `Dropped`

每个完成任务必须记录 owner、revision 或 commit、证据和剩余工作。

| ID | 状态 | 范围 | 退出证据 | 备注 |
| --- | --- | --- | --- | --- |
| M0 | Complete | 计划和基线 | `feat/rt-owning-tensor` 上的本文档 | 基线 revision 已记录 |
| T1.1 | Complete | 分支准备 | 证据日志中的分支 | 本地集成分支已创建 |
| T1.2 | Complete | API contract 测试 | 2026-11-23 证据日志 | 尚未迁移 production caller |
| T1.3 | Complete | Public header/install 规则 | 2026-11-23 证据日志 | `rt.h` 直接导出安装头 |
| T1.4 | Complete | API 命名决策 | 决策 D6 | allocator 签名在 M3 前不得再变名 |
| T2.* | Complete | RT storage/tensor core | 2026-11-23 证据日志 | 未引入 allocator 或 Ops 依赖 |
| T3.* | In progress | Allocator bridge | 2026-11-24 证据日志 | RT allocator 与 LM Runtime adapter 已完成；LM Tensor 工厂切换归入 M4 |
| T4.* | In progress | LM 接入 | 2026-11-24 证据日志 | dot API 与首批 runtime/execution 迁移完成；alias 未切换 |
| T5.* | Not started | Copy/fill 边界 | 依赖扫描和 transfer 测试 | 禁止 RT 依赖 Ops |
| T6.* | Not started | Graph/blob 语义 | NVIDIA graph 测试 | 最高风险阶段 |
| T7.* | Not started | Ops 接入 | Ops CPU/NVIDIA 测试 | 依赖安装后的 RT |
| T8.* | Not started | CCL、dtype、分布式 | mapping 和 TP 测试 | 保持 CCL C ABI |
| T9 | Not started | 端到端矩阵 | 完整证据表 | 包含性能 |
| T10 | Not started | Core pin 和发布 | 干净 submodule 构建 | 只能最后执行 |

## 7. 证据日志

新条目插入在最上方。不得修改旧证据来掩盖失败。

## 7.1 当前变更矩阵

更新时间：2026-11-24。

| 仓库 | 分支 | 基线 | 迁移 commit | 截至所列 commit 的行数 |
| --- | --- | --- | --- | --- |
| InfiniRT | `feat/rt-owning-tensor` | `c0810cfe` | `d0a9d2f`, `afb97f2`, `aacd0c1`, `f6a91b5`, `5b41739`, `5428970`, `9aab3b4` | 15 files, +2187/-62 |
| InfiniLM | `feat/rt-owning-tensor` | `ebb081ef` | `2490e047`, `7175d971` | 16 files, +306/-210 |
| InfiniOps | `feat/rt-owning-tensor` | `f52af91a` | 无 | 0 files |
| InfiniCore | `feat/rt-owning-tensor` | `64362228` | 无 | 0 files |
| InfiniCCL | `feat/rt-owning-tensor` | `51cc9db` | 无 | 0 files |

LM 两个 commit 的独立规模：

- `2490e047 feat: adapt runtime to rt tensor allocator`：9 files, +96/-91。
- `7175d971 refactor: prepare lm tensor for rt value api`：9 files, +210/-119。

当前只有 RT 与 LM 有源码迁移提交；Ops、Core、CCL 只保留同名分支和固定基线。Core gitlink 必须等 RT/Ops 最终 commit 存在后再更新。

### 2026-11-24 - M4 第一批 value API 准备

- 分支和提交：
  - LM `feat/rt-owning-tensor` / `7175d971`。
- 完成内容：
  - LM `Tensor` 增加 RT 兼容 dot-style API，包括 metadata、data、pure view、copy/transfer、graph blob 和 owner 访问。
  - 增加 `Tensor::reshape(shape)`；保留旧 `Tensor::view(shape)` 作为过渡 API。
  - 无参 `Tensor::view()` 返回 `infini::rt::TensorView`，对齐 RT 语义。
  - `csrc/runtime/io.cc`、`csrc/execution/submit.hpp`、graph 宏、tensor checks、tensor expressions 和 native view 测试从 `operator->` 迁移为 dot access。
  - shape reshape 调用从 `view(shape)` 改为 `reshape(shape)`。
- 验证：
  - LM `_infinilm` 完整构建成功。
  - `test_tensor_empty_view` 通过。
  - LM CPU 与 NVIDIA allocator bridge 测试通过。
  - LM Python `_core.empty/view/sync_stream` smoke 通过。
- 剩余工作：
  - LM 全库仍有约两千处 Tensor `operator->` 调用未迁移。
  - `infinilm::runtime::Tensor` 尚未 alias 到 RT Tensor。
  - copy/ones 等 Ops-backed 方法仍在 LM wrapper 上，后续需移到 LM adapter/free function。

### 2026-11-24 - M3 RT allocator 契约

- 分支：RT `feat/rt-owning-tensor`。
- 完成内容：
  - 新增公共 `include/infini/rt/tensor_allocator.h`。
  - 定义 `TensorStorageKind`：`kGeneral`、`kParameter`、`kPinnedHost`。
  - 定义抽象 `TensorAllocator` 接口，包含 device、opaque owner、普通分配、exact 分配和 pinned host 分配。
  - `Tensor::empty` 与 `Tensor::strided_empty` 通过 allocator 构造 RT owning Tensor。
  - 新增 `tests/test_tensor_allocator.cc`，覆盖三种 storage kind、owner token、分配计数和释放计数。
- 命令和结果：
  - RT CPU 本地测试 12/12 通过。
  - RT CPU `test_install` / `test_install_consumer` 2/2 通过。
- 当前边界：
  - RT 不 include LM / Ops / CCL。
  - 尚未适配 LM Runtime。
- 下一步：
  - 已由下方 LM Runtime adapter 记录继续。

### 2026-11-24 - M3 LM Runtime adapter

- 分支和提交：
  - RT `feat/rt-owning-tensor` / `5428970`。
  - LM `feat/rt-owning-tensor` / `2490e047`。
- 完成内容：
  - `infinilm::runtime::Memory` 改为 alias 到 `infini::rt::Memory`。
  - 删除 LM 私有 `memory.cc` 实现。
  - LM `TensorImpl::allocation_owner()` 通过 LM 边界 helper 将 opaque owner token 恢复为 typed `Runtime`。
  - LM `Runtime` 实现 `infini::rt::TensorAllocator`：
    - `device()`;
    - `allocation_owner()`;
    - `allocate()`;
    - `allocate_exact()`;
    - `allocate_pinned_host()`。
  - LM 继续复用现有 `ReusableMemoryPool`、stream 释放规则和 foreign-thread drain 逻辑。
  - 新增 `test/native/tensor_allocator_test.cpp`，通过 LM Runtime 构造 RT owning Tensor，并验证 owner identity、数据写入和 stream 同步。
- 命令和结果：
  - 独立 prefix：`/data/shared/zhuyue/codes/migration-m3-prefix`。
  - RT CPU+NVIDIA、Ops CPU+NVIDIA、CCL NVIDIA/NCCL 均安装到该 prefix。
  - LM `_infinilm` 完整构建成功。
  - LM CPU `test_tensor_allocator` 通过。
  - `CUDA_VISIBLE_DEVICES=0 INFINILM_TENSOR_ALLOCATOR_DEVICE=nvidia` 下 LM NVIDIA allocator bridge 测试通过。
  - LM `test_tensor_empty_view` 通过。
  - LM Python `_core.empty/sync_stream` smoke 通过。
  - RT CPU+NVIDIA 15/15 本地测试通过。
  - RT NVIDIA install consumer 2/2 通过。
- 构建环境记录：
  - Ops 完整 LM 编译需要 `Copy`、`W4MarlinGemm`、FlashAttention provider 和 Torch `upsample_bicubic2d`。
  - W4 Marlin 需要 CMake 3.26+；本轮使用 CMake 3.31.6。
  - Torch provider 链接时需要 `-Wl,-rpath-link,$TORCH_LIB` 和 CUDA toolkit lib64。
  - xmake 需要显式使用当前 migration prefix，避免回退到 `/home/zhuyue/.infini` 旧库。
- 当前边界：
  - RT owning Tensor 已能通过 LM Runtime allocator 构造。
  - LM 私有 `Tensor` 尚未整体 alias 到 RT Tensor；LM `empty/strided_empty` 仍保留私有实现。
- 下一步：
  - M4 引入 `infinilm::runtime::Tensor = infini::rt::Tensor`，逐步移除 LM 私有 TensorImpl。

### 2026-11-23 - M2 外部存储和纯视图实现

- 分支：RT `feat/rt-owning-tensor`，M2 提交为 `f6a91b5`，前一个提交为 M1 `aacd0c1`。
- 完成内容：
  - 在 `src/tensor.cc` 实现 `TensorImpl` PIMPL。
  - 实现 `from_blob`、带 owner 的 `from_blob`、`strided_from_blob` 和带 owner 的 `strided_from_blob`。
  - 实现 `view`、`squeeze`、`unsqueeze`、`narrow`、`permute`、`reshape`、`as_strided`。
  - 实现 dtype、device、shape、strides、data pointer、numel、stride、element size 和 byte size 访问。
  - 实现 rank 校验、负 stride 拒绝、storage span 校验、offset 校验、numel/stride/byte 溢出检查。
  - 外部 owner 通过 `Memory::Deleter` 保活；视图共享 `Memory`。
  - 新增 `tests/test_owning_tensor.cc`，覆盖外部存储、strided storage、视图共享、rank 0、empty tensor、rank 9、broadcast stride、owner 生命周期和错误路径。
  - `docs/api/core-types.md` 与 `docs/compatibility.md` 改为中文并记录 `Memory` / `Tensor` 公共边界。
- 命令和结果：
  - CPU 配置和构建成功；本地 11/11 测试通过。
  - CPU `test_install` / `test_install_consumer` 2/2 通过。
  - NVIDIA CUDA 12.9、A100、sm80 配置和构建成功；本地 14/14 测试通过。
  - NVIDIA `test_install` / `test_install_consumer` 2/2 通过。
  - `rg -n 'infini/ops|infinilm|infiniccl' include src tests` 无实现代码匹配。
- 环境注意事项：
  - 当前 shell 的 `LD_LIBRARY_PATH` 含 `/home/zhuyue/.infini/lib` 时会抢先进旧 RT 库；本地测试需显式指定当前 build 的 `src` 或 install prefix。
  - NVIDIA 实际 toolkit 在 Spack CUDA 12.9 路径，不是 `/usr/local/cuda`。
  - RT 的 `generated/include` 与 `generated/src` 位于源码树并在多个 build 目录间共享；从 NVIDIA 切回 CPU 或从 CPU 切到 NVIDIA 时必须重新 configure 对应 build 目录，否则会把上一后端的生成声明带入链接。
- 交接状态：
  - RT `feat/rt-owning-tensor` 已推送到 `origin`。
  - Ops、LM、Core、CCL 的同名本地分支处于各自基线提交，尚无独立变更。
- 当前阻塞：
  - 无。
- 下一步：
  - M3 定义 `TensorAllocator` 并让 LM worker Runtime 适配现有 `ReusableMemoryPool`。

### 2026-11-23 - M1 API 契约和安装验证

- 分支：
  - RT `feat/rt-owning-tensor`
  - Ops `feat/rt-owning-tensor`
  - LM `feat/rt-owning-tensor`
  - Core `feat/rt-owning-tensor`
  - CCL `feat/rt-owning-tensor`
- 完成内容：
  - 新增公共 `include/infini/rt/memory.h`。
  - 新增公共 `include/infini/rt/tensor.h`。
  - `<infini/rt.h>` 导出两个新头文件。
  - 新增 `tests/test_owning_tensor_contract.cc`，覆盖 `Memory` 生命周期和 owning `Tensor` API 签名。
  - `tests/install_consumer_smoke.cc` 增加新公共类型的安装消费检查。
  - 五个仓库统一分支名，均未使用额外 worktree。
- 命令和结果：
  - `cmake -S . -B build-migration-cpu -DCMAKE_BUILD_TYPE=Release -DWITH_CPU=ON -DINFINI_RT_BUILD_TESTING=ON`：配置成功。
  - `cmake --build build-migration-cpu -j8`：构建成功。
  - 首次完整 `ctest` 中 6 个运行时测试失败，原因是当前 shell 的 `LD_LIBRARY_PATH` 包含 `/home/zhuyue/.infini/lib`，其优先级高于测试 RUNPATH，实际加载了旧安装库。
  - `LD_LIBRARY_PATH="$PWD/build-migration-cpu/src" ctest ...`：10/10 通过。
  - `LD_LIBRARY_PATH="$PWD/build-migration-cpu/tests/install_consumer_prefix/lib" ctest ... -R '^(test_install|test_install_consumer)$'`：2/2 通过。
  - `clang-format 22.1.3` 已格式化新增和修改的 C++ 文件。
- 当前阻塞：
  - 无。
- 下一步：
  - M2 实现 `TensorImpl`、外部存储构造和纯 view 操作。

### 2026-11-23 - 文档语言修正

- 分支：RT `feat/rt-owning-tensor`。
- 完成内容：
  - 将迁移施工文档改为中文。
  - 增加项目文档语言约定。
- 验证：
  - `git diff --check`。
- 下一步：
  - T1.2 API contract 测试。

### 2026-11-23 - 计划初始化

- 分支：RT `feat/rt-owning-tensor`，基于 `c0810cfe`。
- 相关分支：
  - Ops `feat/rt-owning-tensor`，基于 `f52af91a`。
  - LM `feat/adopt-rt-owning-tensor`，基于 `ebb081ef`。
- 完成内容：
  - 盘点当前 RT、LM、Ops、CCL tensor 边界。
  - 记录依赖、graph、ownership、ABI、dtype 风险。
  - 创建分阶段施工计划。
  - 创建本地 RT、Ops、LM 集成分支。
- 剩余工作：
  - 实现 M1 contract 测试和 generated-header 规则。
  - 交接到其他机器前推送或 bundle 本地分支。

## 8. 决策日志

新增决策使用递增 `D#`。后续决策只能显式 supersede 旧决策。

### D1. TensorView 继续作为 Ops 交换类型

日期：2026-11-23

决策：Ops API 继续接收 `infini::rt::TensorView`。

理由：保持算子 ABI，避免 ownership 歧义，并维持窄计算边界。

### D2. RT 不得依赖 Ops 或 LM

日期：2026-11-23

决策：迁移后的 RT 源码不得 include InfiniOps、InfiniLM 或 InfiniCCL。

理由：RT 是底层 runtime。调用 Ops 的 LM 便捷操作先留在 LM，直到出现独立的 RT-native primitive 设计。

### D3. 规范 metadata 使用 RT inline-capacity 表示

日期：2026-11-23

决策：owning Tensor 使用现有 RT `TensorView` metadata 类型。

理由：避免维护两套 tensor layout，并保留现有 metadata 分配优化。

### D4. CCL 继续使用裸指针 C API

日期：2026-11-23

决策：本次迁移不让 CCL 耦合 RT owning Tensor。

理由：storage lifetime 和 stream 正确性已经构成主要风险，改变 CCL ABI 会不必要地扩大影响面。

### D5. 项目文档语言约定

日期：2026-11-23

决策：

- 工程设计、施工计划、交接文档、调研结论以中文为主。
- 代码注释、标识符、commit message、日志和错误消息使用英文。
- 面向国际社区的发布文档可以在中文定稿后提供英文翻译或双语版本。
- License、法律文本和上游必须英文的接口说明不翻译。

理由：中文文档降低跨 session 和跨机器施工的沟通成本；英文代码注释和标识符保持代码库国际可读性。

### D6. M1 公共命名和所有权语义

日期：2026-11-23

决策：

- 公共 storage 类型命名为 `infini::rt::Memory`。
- 公共 owning tensor 类型命名为 `infini::rt::Tensor`。
- 实现类型命名为 `infini::rt::TensorImpl`，只作为 PIMPL 实现细节，不在公共 API 中暴露成员。
- 切片参数命名为嵌套的 `Tensor::Slice`，字段为 `dim`、`start`、`length`。
- `Memory` 的 allocation owner 使用 `std::weak_ptr<void>`，RT 只把它作为 opaque identity。
- `Tensor::reset()` 只清空 owning handle，不尝试释放或同步底层存储。
- M3 allocator 接口命名为 `TensorAllocator`；接口签名在 M3 实现前需保持非 LM、非 Ops 方向。

理由：这些名称与现有 RT 命名空间和 LM 迁移方向一致；`Slice` 避免重新引入 LM 的 `TensorSliceParams` 命名。

### D7. Parameter storage 是通用 storage policy

日期：2026-11-24

决策：RT allocator 使用 `TensorStorageKind::kParameter` 表示 exact physical storage，而不是 LM 专属概念。

理由：参数物理布局只依赖 allocator 的 exact allocation 语义，可以服务 LM Parameter，也可以服务其他需要禁用复用策略的上层对象。

### D8. M4 采用 dot API 先行策略

日期：2026-11-24

决策：

- 先在 LM 私有 `Tensor` wrapper 上补齐 RT Tensor 兼容的 dot-style API。
- LM 的 shape reshape 统一命名为 `reshape(shape)`。
- 无参 `view()` 专用于转换为 `TensorView`。
- 先迁移 runtime、execution、graph 和 native 测试，再扩展到 layers/models。
- 全部调用点具备 RT 兼容形态后，才把 LM Tensor alias 到 RT Tensor。

理由：LM 当前约有 2170 处 Tensor 方法 `operator->` 调用，直接 alias 会造成长时间不可构建；dot API 先行可以让每个目录独立迁移、独立编译和回滚。

## 9. 风险清单

| 风险 | 影响 | 早期信号 | 缓解措施 |
| --- | --- | --- | --- |
| RT 依赖 Ops | 架构方向失败 | 禁止 include 扫描出现匹配 | 每个 RT 阶段后跑依赖门禁 |
| 一次性替换 LM | 长时间不可构建 | LM 多个 milestone 同时失败 | 兼容 alias 和机械分批 |
| Shape 表示不一致 | 大范围编译失败 | Python 或 vector reference 报错 | 迁移前先做 contract 测试 |
| 外部 owner 悬垂 | Torch 或 safetensors 崩溃 | Python 测试 use-after-free | owner 保活测试，可用 sanitizer |
| Graph 生命周期错误 | replay 数据损坏或延迟释放 | graph 测试偶发失败 | retain/reinstantiate 测试，capture 中禁止 cache miss |
| Stream 顺序回归 | 静默数据竞争 | 输出偶发不一致 | worker 测试和 graph/eager 对比 |
| ABI 漂移 | 下游 layout/link 失败 | installed-consumer 失败 | public header 每次变化都跑安装测试 |
| Dtype 漂移 | CCL 或 Python 转换错误 | dtype 不支持或映射错误 | 穷举 mapping，包括 `kBool` |
| 版本混用 | 无关构建失败 | submodule 状态不干净 | 精确 revision 和干净 prefix |
| 共享 generated 源码树 | CPU/NVIDIA 交叉构建带入错误后端符号 | CPU 链接出现 CUDA undefined reference | 切换后端时重新 configure 对应 build 目录 |
| 性能回退 | 正确性后迁移被阻塞 | tensor-heavy benchmark 变慢 | 重构前先做基线对比 |

## 10. 必需验证命令

证据日志必须记录实际适配后的路径和结果，不能只写“通过”。

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

### RT 依赖方向审计

```bash
rg -n 'infini/ops|infinilm|infiniccl' include src tests
```

在没有显式记录例外前，期望实现代码无匹配。

### Ops 基于安装后的 RT

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

### LM native 和 Python 测试

使用同一个一致安装前缀和匹配的 C++ ABI：

```bash
export INFINI_ROOT=/absolute/path/to/coherent-prefix
xmake f -y -m release --cxx11-abi=1
xmake build -j _infinilm
python -m pytest test/runtime test/scripts -q
```

模型 smoke 还必须记录 device、模型 shape、dtype、batch、input/output length、graph mode 和结果对比方式。

## 11. Session 和交接协议

每次开发 session 必须执行：

1. 修改代码前先阅读本文档。
2. 确认各仓库 revision 和 `git status`。
3. 只选择一个 Ready 任务，或继续唯一一个 In progress 任务。
4. 实现前把本次意图写入证据日志。
5. 尽量让每个 commit 独立可编译。
6. 运行当前 milestone 必需测试。
7. 更新任务状态、证据、决策、风险和剩余工作。
8. 每个仓库保持干净，或留下明确记录的分支和工作区状态。
9. 推送分支；网络不可用时生成 bundle 或 patch。
10. 只有阻塞点具体且已记录尝试过的缓解措施后，才标记 Blocked。

建议证据格式：

```markdown
### YYYY-MM-DD - 简短标题

- 分支和 revision：
  - RT：`<branch>` / `<commit>`
  - Ops：`<branch>` / `<commit>`
  - LM：`<branch>` / `<commit>`
- 完成内容：
  - ...
- 命令和结果：
  - ...
- 当前阻塞：
  - 无，或写明具体阻塞。
- 下一步：
  - ...
```

## 12. 停止条件

出现以下情况时，先停下来稳定，不要继续扩大改动：

1. RT 需要 include InfiniOps 或 InfiniLM 才能继续。
2. graph 或 transfer 测试出现偶发错误。
3. 裸指针路径无法证明 ownership 生命周期。
4. 发现 compatibility 文档之外的公共 ABI 变化。
5. submodule 需要指向不存在或未提交的 revision。
6. LM 行为变化没有对应测试和决策记录。
7. 多个主要 milestone 同时处于 broken 状态。

## 13. 完成定义

只有同时满足以下条件，迁移才算完成：

1. 第 6 节所有任务是 Complete，或有明确理由的 Dropped。
2. RT 拥有规范 Memory/Tensor 实现。
3. Ops 没有竞争性 Tensor 实现。
4. LM 没有活跃的私有 Tensor/Memory 实现。
5. RT、Ops、LM 公共文档描述最终边界。
6. 干净 prefix 下 RT、Ops、CCL、LM 构建通过。
7. CPU 和 NVIDIA 端到端测试通过。
8. 可用硬件下 graph 和 TP smoke 通过。
9. 性能证据已记录。
10. Core 和 LM pin 指向可获取的已提交 revision。
11. 新贡献者能只依靠本文档、已记录 commit 和标准仓库工具继续或审计该工作。
