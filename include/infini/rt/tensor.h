#ifndef INFINI_RT_TENSOR_H_
#define INFINI_RT_TENSOR_H_

#include <infini/rt/generated.h>
#include <infini/rt/memory.h>
#include <infini/rt/tensor_allocator.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace infini::rt {

class TensorImpl;

// A shared-ownership tensor over external or allocator-provided storage.
class Tensor {
 public:
  using Size = TensorView::Size;
  using Stride = TensorView::Stride;
  using Shape = TensorView::Shape;
  using Strides = TensorView::Strides;
  using ShapeView = TensorView::ShapeView;
  using StridesView = TensorView::StridesView;

  struct Slice {
    std::size_t dim;
    std::size_t start;
    Size length;
  };

  Tensor() noexcept = default;
  Tensor(const Tensor&) noexcept = default;
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(const Tensor&) noexcept = default;
  Tensor& operator=(Tensor&&) noexcept = default;

  static Tensor from_blob(void* data, const Shape& shape, const DataType& dtype,
                          const Device& device);
  static Tensor from_blob(void* data, const Shape& shape, const DataType& dtype,
                          const Device& device, std::shared_ptr<void> owner);
  static Tensor strided_from_blob(void* data, const Shape& shape,
                                  const Strides& strides, const DataType& dtype,
                                  const Device& device);
  static Tensor strided_from_blob(void* data, const Shape& shape,
                                  const Strides& strides, const DataType& dtype,
                                  const Device& device,
                                  std::shared_ptr<void> owner);
  static Tensor empty(TensorAllocator& allocator, const Shape& shape,
                      const DataType& dtype,
                      TensorStorageKind kind = TensorStorageKind::kGeneral);
  static Tensor strided_empty(
      TensorAllocator& allocator, const Shape& shape, const Strides& strides,
      const DataType& dtype,
      TensorStorageKind kind = TensorStorageKind::kGeneral);

  TensorView view() const;
  Tensor squeeze(std::size_t dim) const;
  Tensor unsqueeze(std::size_t dim) const;
  Tensor narrow(const std::vector<Slice>& slices) const;
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

  bool empty() const noexcept { return impl_ == nullptr; }
  void reset() noexcept { impl_.reset(); }
  std::size_t use_count() const noexcept {
    return impl_ ? impl_.use_count() : 0;
  }
  explicit operator bool() const noexcept { return impl_ != nullptr; }

 protected:
  friend class TensorImpl;

  explicit Tensor(std::shared_ptr<TensorImpl> impl) noexcept
      : impl_{std::move(impl)} {}

  std::shared_ptr<TensorImpl> impl_;

 private:
  void RequireImpl() const;
};

}  // namespace infini::rt

#endif
