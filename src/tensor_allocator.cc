#include <infini/rt/tensor_allocator.h>

#include <stdexcept>

namespace infini::rt {

std::shared_ptr<Memory> TensorAllocator::allocate(TensorStorageKind kind,
                                                  std::size_t size) {
  switch (kind) {
    case TensorStorageKind::kGeneral:
      return allocate(size);
    case TensorStorageKind::kParameter:
      return allocate_exact(size);
    case TensorStorageKind::kPinnedHost:
      return allocate_pinned_host(size);
  }
  throw std::invalid_argument("unsupported tensor storage kind");
}

}  // namespace infini::rt
