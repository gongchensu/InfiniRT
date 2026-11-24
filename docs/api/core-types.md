# 核心类型

核心类型通过以下入口提供：

```cpp
#include <infini/rt.h>
```

## Device

`infini::rt::Device` 表示后端类型和设备编号。

```cpp
infini::rt::Device device{infini::rt::Device::Type::kCpu, 0};
auto type = device.type();
auto index = device.index();
```

当前已知设备类型包括：

- `Device::Type::kCpu`
- `Device::Type::kNvidia`
- `Device::Type::kIluvatar`
- `Device::Type::kMetax`
- `Device::Type::kMars`
- `Device::Type::kMoore`
- `Device::Type::kHygon`
- `Device::Type::kCambricon`
- `Device::Type::kAscend`
- `Device::Type::kThead`

## DataType

`infini::rt::DataType` 描述张量元素类型。当前支持：

- `kInt8`, `kInt16`, `kInt32`, `kInt64`
- `kUInt8`, `kUInt16`, `kUInt32`, `kUInt64`
- `kFloat16`, `kBFloat16`, `kFloat32`, `kFloat64`
- `kBool`

`kBool` 与 `kUInt8` 是不同枚举值，并且占用一个字节。

## TensorView

`infini::rt::TensorView` 是非拥有张量描述，负责保存：

- data pointer;
- shape;
- strides;
- dtype;
- device。

它不拥有 data pointer 指向的存储。rank 0 到 8 的 shape/stride 元数据内联保存，rank 9 及以上使用 heap fallback。

```cpp
std::vector<float> data(16);

infini::rt::TensorView tensor{
    data.data(),
    std::vector<std::size_t>{4, 4},
    infini::rt::DataType::kFloat32,
    infini::rt::Device{infini::rt::Device::Type::kCpu, 0},
    std::vector<std::ptrdiff_t>{4, 1}};
```

在 lvalue `TensorView` 上，`shape()` 和 `strides()` 返回轻量 borrowed view。该 view 不能比原 `TensorView` 活得更久；在 rvalue 上调用时会返回拥有式 metadata，避免临时对象悬垂。

## Memory

`infini::rt::Memory` 拥有一次 allocation 的释放义务，包含：

- data pointer;
- 字节数;
- device;
- deleter;
- pinned 标记;
- graph-pinnable 标记;
- optional opaque allocation owner。

`allocation_owner` 只表示 allocation identity，不表示 producer 已经完成，也不表示某个 stream 上的写入已经可见。外部存储可以通过 `std::shared_ptr<void>` owner 保活。

## Tensor

`infini::rt::Tensor` 是 shared-ownership 张量值类型。当前公共能力包括：

- `from_blob` 和 `strided_from_blob`;
- external owner 保活;
- `view()` 转换为 `TensorView`;
- `squeeze`, `unsqueeze`, `narrow`, `permute`, `reshape`, `as_strided`;
- metadata、data pointer 和 storage 生命周期访问。

视图操作共享底层 `Memory`，不复制元素数据。RT 当前不通过 Ops 提供填充或拷贝便捷函数；这些计算语义由上层 runtime 或后续 RT-native primitive 提供。

## TensorAllocator

`infini::rt::TensorAllocator` 是张量存储上下文边界。实现方负责：

- 返回与目标设备一致的普通存储；
- 返回参数物理布局使用的 exact storage;
- 返回 pinned host storage;
- 维护 allocation 释放规则和 stream 顺序；
- 提供 opaque allocation owner token。

RT 只通过该接口构造 `Tensor::empty` 和 `Tensor::strided_empty`，不感知 LM Runtime、Ops 或上层调度对象。
