#include <infini/rt/tensor.h>
#include <infini/rt/tensor_allocator.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infini::rt {
namespace {

constexpr std::size_t kMaxSize = std::numeric_limits<std::size_t>::max();
constexpr Tensor::Stride kMaxStride =
    std::numeric_limits<Tensor::Stride>::max();

std::size_t CheckedElementSize(const DataType& dtype) {
  const std::size_t size = kDataTypeToSize.at(dtype);
  if (size == 0) {
    throw std::length_error("tensor element size is zero");
  }
  return size;
}

Tensor::Size CheckedNumel(const Tensor::Shape& shape) {
  Tensor::Size result = 1;
  for (const Tensor::Size dim : shape) {
    if (dim != 0 && result > kMaxSize / dim) {
      throw std::length_error("tensor element count overflows");
    }
    result *= dim;
  }
  return result;
}

Tensor::Strides DefaultStrides(const Tensor::Shape& shape) {
  Tensor::Strides strides;
  strides.resize(shape.size());
  Tensor::Size stride = 1;
  for (std::size_t index = shape.size(); index > 0; --index) {
    if (stride > static_cast<Tensor::Size>(kMaxStride)) {
      throw std::length_error("tensor storage stride overflows");
    }
    strides[index - 1] = static_cast<Tensor::Stride>(stride);
    if (shape[index - 1] != 0 && stride > kMaxSize / shape[index - 1]) {
      throw std::length_error("tensor storage stride overflows");
    }
    stride *= shape[index - 1];
  }
  return strides;
}

std::size_t StorageSpanBytes(const Tensor::Shape& shape,
                             const Tensor::Strides& strides,
                             std::size_t element_size) {
  if (shape.size() != strides.size()) {
    throw std::invalid_argument(
        "tensor shape and strides must have the same rank");
  }

  std::size_t max_offset = 0;
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (strides[index] < 0) {
      throw std::invalid_argument("negative tensor strides are not supported");
    }
    if (shape[index] == 0) return 0;

    const std::size_t stride = static_cast<std::size_t>(strides[index]);
    const std::size_t extent = shape[index] - 1;
    if (extent != 0 && stride > kMaxSize / extent) {
      throw std::overflow_error("tensor storage span overflows");
    }
    const std::size_t contribution = extent * stride;
    if (contribution > kMaxSize - max_offset) {
      throw std::overflow_error("tensor storage span overflows");
    }
    max_offset += contribution;
  }

  if (max_offset == kMaxSize || max_offset + 1 > kMaxSize / element_size) {
    throw std::overflow_error("tensor storage size overflows");
  }
  return (max_offset + 1) * element_size;
}

std::string MetadataString(const Tensor::Shape& shape,
                           const Tensor::Strides& strides) {
  std::string result{"shape=["};
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) result += ',';
    result += std::to_string(shape[index]);
  }
  result += "] strides=[";
  for (std::size_t index = 0; index < strides.size(); ++index) {
    if (index != 0) result += ',';
    result += std::to_string(strides[index]);
  }
  return result + "]";
}

Tensor::Stride CheckedStrideProduct(Tensor::Stride stride,
                                    Tensor::Size factor) {
  if (stride < 0 || (stride != 0 &&
                     factor > static_cast<Tensor::Size>(kMaxStride / stride))) {
    throw std::length_error("tensor view stride overflows");
  }
  return stride == 0 ? 0 : stride * static_cast<Tensor::Stride>(factor);
}

}  // namespace

class TensorImpl final : public std::enable_shared_from_this<TensorImpl> {
 public:
  TensorImpl(Tensor::Shape shape, Tensor::Strides strides,
             const DataType& dtype, std::shared_ptr<Memory> memory,
             std::size_t offset = 0)
      : shape_{std::move(shape)},
        strides_{std::move(strides)},
        dtype_{dtype},
        memory_{std::move(memory)},
        offset_{offset} {
    if (!memory_) {
      throw std::invalid_argument("tensor storage must be non-null");
    }
    if (shape_.size() != strides_.size()) {
      throw std::invalid_argument(
          "tensor shape and strides must have the same rank");
    }
    if (std::any_of(strides_.begin(), strides_.end(),
                    [](Tensor::Stride stride) { return stride < 0; })) {
      throw std::invalid_argument("negative tensor strides are not supported");
    }
    const std::size_t span =
        StorageSpanBytes(shape_, strides_, CheckedElementSize(dtype_));
    if (offset_ > memory_->size() || span > memory_->size() - offset_) {
      throw std::out_of_range("tensor view exceeds its underlying storage: " +
                              MetadataString(shape_, strides_));
    }
  }

  std::byte* data() {
    return memory_->data() == nullptr ? nullptr : memory_->data() + offset_;
  }
  const std::byte* data() const {
    return memory_->data() == nullptr ? nullptr : memory_->data() + offset_;
  }
  Tensor::ShapeView shape() const { return {shape_.data(), shape_.size()}; }
  Tensor::StridesView strides() const {
    return {strides_.data(), strides_.size()};
  }
  DataType dtype() const { return dtype_; }
  Device device() const { return memory_->device(); }

  bool is_contiguous() const {
    Tensor::Stride expected_stride = 1;
    for (std::size_t reverse = shape_.size(); reverse > 0; --reverse) {
      const std::size_t index = reverse - 1;
      if (strides_[index] != expected_stride) return false;
      if (shape_[index] == 0) {
        expected_stride = 0;
      } else {
        if (expected_stride != 0 &&
            shape_[index] > static_cast<Tensor::Size>(kMaxStride) /
                                static_cast<Tensor::Size>(expected_stride)) {
          return false;
        }
        const Tensor::Size next =
            static_cast<Tensor::Size>(expected_stride) * shape_[index];
        if (next > static_cast<Tensor::Size>(kMaxStride)) {
          return false;
        }
        expected_stride = static_cast<Tensor::Stride>(next);
      }
    }
    return true;
  }

  Tensor::Size numel() const { return CheckedNumel(shape_); }
  Tensor::Size ndim() const { return shape_.size(); }

  Tensor::Size size(std::size_t dim) const {
    if (dim >= shape_.size()) {
      throw std::out_of_range("tensor dimension is out of range");
    }
    return shape_[dim];
  }

  Tensor::Stride stride(std::size_t dim) const {
    if (dim >= strides_.size()) {
      throw std::out_of_range("tensor stride dimension is out of range");
    }
    return strides_[dim];
  }

  std::size_t element_size() const { return CheckedElementSize(dtype_); }

  std::size_t nbytes() const {
    const Tensor::Size elements = numel();
    const std::size_t bytes_per_element = element_size();
    if (elements > kMaxSize / bytes_per_element) {
      throw std::overflow_error("tensor byte size overflows");
    }
    return elements * bytes_per_element;
  }

  Tensor squeeze(std::size_t dim) const {
    if (dim >= shape_.size()) {
      throw std::out_of_range("squeeze dimension is out of range");
    }
    if (shape_[dim] != 1) {
      throw std::runtime_error("squeeze dimension must have size one");
    }

    Tensor::Shape new_shape;
    Tensor::Strides new_strides;
    new_shape.reserve(shape_.size() - 1);
    new_strides.reserve(strides_.size() - 1);
    for (std::size_t index = 0; index < shape_.size(); ++index) {
      if (index == dim) continue;
      new_shape.push_back(shape_[index]);
      new_strides.push_back(strides_[index]);
    }
    return SharedStorageTensor(std::move(new_shape), std::move(new_strides),
                               offset_);
  }

  Tensor unsqueeze(std::size_t dim) const {
    if (dim > shape_.size()) {
      throw std::out_of_range("unsqueeze dimension is out of range");
    }

    Tensor::Shape new_shape;
    Tensor::Strides new_strides;
    new_shape.reserve(shape_.size() + 1);
    new_strides.reserve(strides_.size() + 1);
    for (std::size_t index = 0; index <= shape_.size(); ++index) {
      if (index == dim) {
        new_shape.push_back(1);
        new_strides.push_back(0);
      }
      if (index == shape_.size()) break;
      new_shape.push_back(shape_[index]);
      new_strides.push_back(strides_[index]);
    }
    return SharedStorageTensor(std::move(new_shape), std::move(new_strides),
                               offset_);
  }

  Tensor narrow(const std::vector<Tensor::Slice>& slices) const {
    Tensor::Shape new_shape = shape_;
    std::size_t offset = offset_;
    for (const Tensor::Slice& slice : slices) {
      if (slice.dim >= shape_.size() || slice.start > shape_[slice.dim] ||
          slice.length > shape_[slice.dim] - slice.start) {
        throw std::out_of_range("tensor slice is out of range");
      }
      const std::size_t element_bytes = CheckedElementSize(dtype_);
      const std::size_t stride = static_cast<std::size_t>(strides_[slice.dim]);
      if (stride != 0 && element_bytes > kMaxSize / stride) {
        throw std::overflow_error("tensor slice stride overflows");
      }
      const std::size_t stride_bytes = stride * element_bytes;
      if (slice.start != 0 &&
          stride_bytes > (kMaxSize - offset) / slice.start) {
        throw std::overflow_error("tensor slice offset overflows");
      }
      offset += slice.start * stride_bytes;
      new_shape[slice.dim] = slice.length;
    }
    return SharedStorageTensor(std::move(new_shape), strides_, offset);
  }

  Tensor permute(const Tensor::Shape& order) const {
    if (order.size() != shape_.size()) {
      throw std::invalid_argument(
          "permutation rank must match the tensor rank");
    }
    Tensor::Shape flags;
    flags.resize(shape_.size());
    std::fill(flags.begin(), flags.end(), Tensor::Size{0});
    for (const Tensor::Size axis : order) {
      if (axis >= order.size() || flags[axis] != 0) {
        throw std::invalid_argument("permutation must contain each axis once");
      }
      flags[axis] = 1;
    }

    Tensor::Shape new_shape;
    Tensor::Strides new_strides;
    new_shape.reserve(order.size());
    new_strides.reserve(order.size());
    for (const Tensor::Size axis : order) {
      new_shape.push_back(shape_[axis]);
      new_strides.push_back(strides_[axis]);
    }
    return SharedStorageTensor(std::move(new_shape), std::move(new_strides),
                               offset_);
  }

  Tensor reshape(const Tensor::Shape& new_shape) const {
    if (CheckedNumel(shape_) != CheckedNumel(new_shape)) {
      throw std::invalid_argument("incompatible shape for tensor reshape: " +
                                  MetadataString(shape_, strides_));
    }
    if (CheckedNumel(shape_) == 0 || shape_.empty()) {
      Tensor::Strides new_strides = DefaultStrides(new_shape);
      return SharedStorageTensor(Tensor::Shape{new_shape},
                                 std::move(new_strides), offset_);
    }

    Tensor::Shape merged_shape;
    Tensor::Strides merged_strides;
    merged_shape.push_back(shape_[0]);
    merged_strides.push_back(strides_[0]);
    for (std::size_t index = 1; index < shape_.size(); ++index) {
      if (CheckedStrideProduct(strides_[index], shape_[index]) ==
          merged_strides.back()) {
        if (merged_shape.back() > kMaxSize / shape_[index]) {
          throw std::overflow_error("tensor view extent overflows");
        }
        merged_shape.back() *= shape_[index];
        merged_strides.back() = strides_[index];
      } else {
        merged_shape.push_back(shape_[index]);
        merged_strides.push_back(strides_[index]);
      }
    }

    Tensor::Strides new_strides;
    new_strides.resize(new_shape.size());
    std::size_t merged_index = 0;
    Tensor::Stride current_stride = merged_strides[0];
    Tensor::Size remaining_size = merged_shape[0];
    for (std::size_t new_index = 0; new_index < new_shape.size(); ++new_index) {
      const Tensor::Size requested_size = new_shape[new_index];
      while (requested_size > remaining_size) {
        if (++merged_index == merged_shape.size()) {
          throw std::invalid_argument(
              "incompatible strides for tensor reshape: " +
              MetadataString(shape_, strides_));
        }
        current_stride = merged_strides[merged_index];
        remaining_size = merged_shape[merged_index];
      }
      if (remaining_size % requested_size != 0) {
        throw std::invalid_argument(
            "incompatible strides for tensor reshape: " +
            MetadataString(shape_, strides_));
      }
      new_strides[new_index] =
          CheckedStrideProduct(current_stride, remaining_size / requested_size);
      remaining_size /= requested_size;
    }
    return SharedStorageTensor(Tensor::Shape{new_shape}, std::move(new_strides),
                               offset_);
  }

  Tensor as_strided(const Tensor::Shape& new_shape,
                    const Tensor::Strides& new_strides) const {
    return SharedStorageTensor(Tensor::Shape{new_shape}, new_strides, offset_);
  }

  TensorView view() const {
    return TensorView{const_cast<std::byte*>(data()), shape_, dtype_, device(),
                      strides_};
  }

 private:
  Tensor SharedStorageTensor(Tensor::Shape shape, Tensor::Strides strides,
                             std::size_t offset) const {
    return Tensor{std::make_shared<TensorImpl>(
        std::move(shape), std::move(strides), dtype_, memory_, offset)};
  }

  Tensor::Shape shape_;
  Tensor::Strides strides_;
  DataType dtype_;
  std::shared_ptr<Memory> memory_;
  std::size_t offset_;
};

Tensor Tensor::from_blob(void* data, const Shape& shape, const DataType& dtype,
                         const Device& device) {
  return from_blob(data, shape, dtype, device, nullptr);
}

Tensor Tensor::from_blob(void* data, const Shape& shape, const DataType& dtype,
                         const Device& device, std::shared_ptr<void> owner) {
  const Strides strides = DefaultStrides(shape);
  const std::size_t size =
      StorageSpanBytes(shape, strides, CheckedElementSize(dtype));
  if (data == nullptr && size != 0) {
    throw std::invalid_argument(
        "from_blob requires a non-null pointer for non-empty tensors");
  }
  Memory::Deleter keep_owner;
  if (owner) {
    keep_owner = [owner = std::move(owner)](std::byte*) {};
  }
  auto memory = std::make_shared<Memory>(static_cast<std::byte*>(data), size,
                                         device, std::move(keep_owner));
  return Tensor{std::make_shared<TensorImpl>(Shape{shape}, std::move(strides),
                                             dtype, std::move(memory))};
}

Tensor Tensor::strided_from_blob(void* data, const Shape& shape,
                                 const Strides& strides, const DataType& dtype,
                                 const Device& device) {
  return strided_from_blob(data, shape, strides, dtype, device, nullptr);
}

Tensor Tensor::strided_from_blob(void* data, const Shape& shape,
                                 const Strides& strides, const DataType& dtype,
                                 const Device& device,
                                 std::shared_ptr<void> owner) {
  const std::size_t size =
      StorageSpanBytes(shape, strides, CheckedElementSize(dtype));
  if (data == nullptr && size != 0) {
    throw std::invalid_argument(
        "strided_from_blob requires a non-null pointer for non-empty tensors");
  }
  Memory::Deleter keep_owner;
  if (owner) {
    keep_owner = [owner = std::move(owner)](std::byte*) {};
  }
  auto memory = std::make_shared<Memory>(static_cast<std::byte*>(data), size,
                                         device, std::move(keep_owner));
  return Tensor{std::make_shared<TensorImpl>(Shape{shape}, Strides{strides},
                                             dtype, std::move(memory))};
}

Tensor Tensor::empty(TensorAllocator& allocator, const Shape& shape,
                     const DataType& dtype, TensorStorageKind kind) {
  return strided_empty(allocator, shape, DefaultStrides(shape), dtype, kind);
}

Tensor Tensor::strided_empty(TensorAllocator& allocator, const Shape& shape,
                             const Strides& strides, const DataType& dtype,
                             TensorStorageKind kind) {
  const std::size_t size =
      StorageSpanBytes(shape, strides, CheckedElementSize(dtype));
  auto memory = allocator.allocate(kind, size);
  if (!memory) {
    throw std::invalid_argument("tensor allocator returned null storage");
  }
  return Tensor{std::make_shared<TensorImpl>(Shape{shape}, Strides{strides},
                                             dtype, std::move(memory))};
}

TensorView Tensor::view() const {
  RequireImpl();
  return impl_->view();
}

Tensor Tensor::squeeze(std::size_t dim) const {
  RequireImpl();
  return impl_->squeeze(dim);
}

Tensor Tensor::unsqueeze(std::size_t dim) const {
  RequireImpl();
  return impl_->unsqueeze(dim);
}

Tensor Tensor::narrow(const std::vector<Slice>& slices) const {
  RequireImpl();
  return impl_->narrow(slices);
}

Tensor Tensor::permute(const Shape& order) const {
  RequireImpl();
  return impl_->permute(order);
}

Tensor Tensor::reshape(const Shape& shape) const {
  RequireImpl();
  return impl_->reshape(shape);
}

Tensor Tensor::as_strided(const Shape& shape, const Strides& strides) const {
  RequireImpl();
  return impl_->as_strided(shape, strides);
}

void* Tensor::data() {
  RequireImpl();
  return impl_->data();
}

const void* Tensor::data() const {
  RequireImpl();
  return impl_->data();
}

Tensor::ShapeView Tensor::shape() const {
  RequireImpl();
  return impl_->shape();
}

Tensor::StridesView Tensor::strides() const {
  RequireImpl();
  return impl_->strides();
}

DataType Tensor::dtype() const {
  RequireImpl();
  return impl_->dtype();
}

Device Tensor::device() const {
  RequireImpl();
  return impl_->device();
}

bool Tensor::is_contiguous() const {
  RequireImpl();
  return impl_->is_contiguous();
}

Tensor::Size Tensor::numel() const {
  RequireImpl();
  return impl_->numel();
}

Tensor::Size Tensor::ndim() const {
  RequireImpl();
  return impl_->ndim();
}

Tensor::Size Tensor::size(std::size_t dim) const {
  RequireImpl();
  return impl_->size(dim);
}

Tensor::Stride Tensor::stride(std::size_t dim) const {
  RequireImpl();
  return impl_->stride(dim);
}

std::size_t Tensor::element_size() const {
  RequireImpl();
  return impl_->element_size();
}

std::size_t Tensor::nbytes() const {
  RequireImpl();
  return impl_->nbytes();
}

void Tensor::RequireImpl() const {
  if (!impl_) {
    throw std::logic_error("tensor implementation is empty");
  }
}

}  // namespace infini::rt
