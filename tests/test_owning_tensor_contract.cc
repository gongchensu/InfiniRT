#include <infini/rt.h>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

namespace {

using infini::rt::DataType;
using infini::rt::Device;
using infini::rt::Memory;
using infini::rt::Tensor;
using infini::rt::TensorView;

static_assert(std::is_same_v<Tensor::Shape, TensorView::Shape>);
static_assert(std::is_same_v<Tensor::Strides, TensorView::Strides>);
static_assert(std::is_same_v<Tensor::ShapeView, TensorView::ShapeView>);
static_assert(std::is_same_v<Tensor::StridesView, TensorView::StridesView>);

static_assert(
    std::is_same_v<decltype(&Memory::device), Device (Memory::*)() const>);
static_assert(
    std::is_same_v<decltype(&Memory::size), std::size_t (Memory::*)() const>);
static_assert(
    std::is_same_v<decltype(&Memory::is_pinned), bool (Memory::*)() const>);
static_assert(std::is_same_v<decltype(&Memory::is_graph_pinnable),
                             bool (Memory::*)() const>);
static_assert(std::is_same_v<decltype(&Memory::allocation_owner),
                             std::shared_ptr<void> (Memory::*)() const>);

static_assert(
    std::is_same_v<decltype(&Tensor::view), TensorView (Tensor::*)() const>);
static_assert(std::is_same_v<decltype(&Tensor::squeeze),
                             Tensor (Tensor::*)(std::size_t) const>);
static_assert(std::is_same_v<decltype(&Tensor::unsqueeze),
                             Tensor (Tensor::*)(std::size_t) const>);
static_assert(std::is_same_v<
              decltype(&Tensor::narrow),
              Tensor (Tensor::*)(const std::vector<Tensor::Slice>&) const>);
static_assert(std::is_same_v<decltype(&Tensor::permute),
                             Tensor (Tensor::*)(const Tensor::Shape&) const>);
static_assert(std::is_same_v<decltype(&Tensor::reshape),
                             Tensor (Tensor::*)(const Tensor::Shape&) const>);
static_assert(std::is_same_v<decltype(&Tensor::as_strided),
                             Tensor (Tensor::*)(const Tensor::Shape&,
                                                const Tensor::Strides&) const>);
static_assert(
    std::is_same_v<decltype(&Tensor::dtype), DataType (Tensor::*)() const>);
static_assert(
    std::is_same_v<decltype(&Tensor::device), Device (Tensor::*)() const>);
static_assert(
    std::is_same_v<decltype(&Tensor::numel), Tensor::Size (Tensor::*)() const>);
static_assert(std::is_same_v<decltype(&Tensor::size),
                             Tensor::Size (Tensor::*)(std::size_t) const>);
static_assert(std::is_same_v<decltype(&Tensor::stride),
                             Tensor::Stride (Tensor::*)(std::size_t) const>);
static_assert(
    std::is_same_v<decltype(&Tensor::reset), void (Tensor::*)() noexcept>);
static_assert(sizeof(Tensor) ==
              sizeof(std::shared_ptr<::infini::rt::TensorImpl>));
static_assert(std::is_nothrow_move_constructible_v<Tensor>);
static_assert(std::is_nothrow_move_assignable_v<Tensor>);

static_assert(std::is_same_v<
              decltype(static_cast<Tensor (*)(void*, const Tensor::Shape&,
                                              const DataType&, const Device&)>(
                  &Tensor::from_blob)),
              Tensor (*)(void*, const Tensor::Shape&, const DataType&,
                         const Device&)>);
static_assert(
    std::is_same_v<decltype(static_cast<Tensor (*)(
                                void*, const Tensor::Shape&, const DataType&,
                                const Device&, std::shared_ptr<void>)>(
                       &Tensor::from_blob)),
                   Tensor (*)(void*, const Tensor::Shape&, const DataType&,
                              const Device&, std::shared_ptr<void>)>);
static_assert(std::is_same_v<
              decltype(static_cast<Tensor (*)(void*, const Tensor::Shape&,
                                              const Tensor::Strides&,
                                              const DataType&, const Device&)>(
                  &Tensor::strided_from_blob)),
              Tensor (*)(void*, const Tensor::Shape&, const Tensor::Strides&,
                         const DataType&, const Device&)>);

}  // namespace

int main() {
  const Device device{Device::Type::kCpu};
  const auto owner = std::make_shared<int>(7);
  std::byte bytes[4]{};
  bool released = false;

  {
    Memory memory{bytes,  sizeof(bytes),
                  device, [&released](std::byte*) { released = true; },
                  false,  false,
                  owner};
    if (memory.data() != bytes || memory.device() != device ||
        memory.size() != sizeof(bytes) || memory.is_pinned() ||
        memory.is_graph_pinnable() || !memory.allocation_owner() ||
        memory.allocation_owner().get() != owner.get()) {
      return 1;
    }
  }

  if (!released) {
    return 1;
  }

  const Tensor tensor;
  if (tensor || !tensor.empty() || tensor.use_count() != 0) {
    return 1;
  }

  return 0;
}
