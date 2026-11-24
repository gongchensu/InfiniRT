# 兼容性边界

本页描述 InfiniRT 面向下游消费者的兼容性约定。

## 稳定用户入口

使用：

```cpp
#include <infini/rt.h>
```

当前稳定用户面包括：

- `infini::rt::Device`
- `infini::rt::DataType`
- `infini::rt::TensorView`
- `infini::rt::Memory`
- `infini::rt::Tensor`
- `infini::rt::TensorAllocator`
- `infini::rt::set_runtime_device_type`
- `infini::rt::runtime_device_type`
- [Runtime API](api/runtime.md) 中的 `infini::rt::runtime` 类型、dispatch 函数和常量。

## 实现侧头文件

安装后的 `infini/rt/detail/*` 头文件用于支撑公共入口，不鼓励直接包含。

`infini/rt/cpu/runtime_.h` 等后端 wrapper 暴露具体设备的
`runtime::Runtime<Device::Type::...>` 特化，适用于后端测试和高级集成。普通消费者应优先使用 `<infini/rt.h>`。

## 生成头文件

生成公共头依赖 CMake 配置时选择的后端。消费者应使用同一次配置和安装产生的头文件与库，不要混合不同 prefix 或不同后端配置。

## ABI 说明

InfiniRT 当前暴露 C++ API。安装头文件和 `libinfinirt.so` 必须来自同一次构建或同一个 release。

`TensorView::Shape` 和 `TensorView::Strides` 是 concrete vector-like C++ 类型，inline capacity 为 8。rank 0 到 8 内联保存，rank 9 及以上使用 heap fallback。该布局相对旧的 `std::vector` alias 是 API/ABI 兼容性破坏；消费者必须重建，不能混用新旧头文件和库。

`Tensor` 是基于 `std::shared_ptr<TensorImpl>` 的 PIMPL 值类型。`TensorImpl` 是实现细节，不承诺布局稳定。`Memory` 当前在公共头文件中定义，其成员布局同样属于 C++ ABI 的一部分。

从 `std::vector` 或其他兼容 range 构造 metadata 仍受支持。需要精确使用 `std::vector` alias 的代码必须显式转换。在 lvalue 上，`shape()` 和 `strides()` 返回 borrowed contiguous view；需要所有权时应显式物化为 `TensorView::Shape` 或 `TensorView::Strides`。
