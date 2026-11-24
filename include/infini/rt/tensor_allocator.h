#ifndef INFINI_RT_TENSOR_ALLOCATOR_H_
#define INFINI_RT_TENSOR_ALLOCATOR_H_

#include <infini/rt/generated.h>
#include <infini/rt/memory.h>

#include <cstddef>
#include <memory>

namespace infini::rt {

enum class TensorStorageKind {
  kGeneral,
  kParameter,
  kPinnedHost,
};

// Context boundary for tensor storage. Implementations own allocation and
// release rules; RT never learns about a higher-level runtime class.
class TensorAllocator {
 public:
  TensorAllocator() = default;
  TensorAllocator(const TensorAllocator&) = delete;
  TensorAllocator& operator=(const TensorAllocator&) = delete;
  TensorAllocator(TensorAllocator&&) = delete;
  TensorAllocator& operator=(TensorAllocator&&) = delete;
  virtual ~TensorAllocator() = default;

  virtual Device device() const = 0;
  virtual std::weak_ptr<void> allocation_owner() const = 0;

  virtual std::shared_ptr<Memory> allocate(std::size_t size) = 0;
  virtual std::shared_ptr<Memory> allocate_exact(std::size_t size) = 0;
  virtual std::shared_ptr<Memory> allocate_pinned_host(std::size_t size) = 0;

  std::shared_ptr<Memory> allocate(TensorStorageKind kind, std::size_t size);
};

}  // namespace infini::rt

#endif
