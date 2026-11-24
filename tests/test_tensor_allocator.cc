#include <infini/rt.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace {

using infini::rt::DataType;
using infini::rt::Device;
using infini::rt::Memory;
using infini::rt::Tensor;
using infini::rt::TensorAllocator;
using infini::rt::TensorStorageKind;

class TestAllocator final : public TensorAllocator,
                            public std::enable_shared_from_this<TestAllocator> {
 public:
  Device device() const override { return Device{Device::Type::kCpu, 0}; }

  std::weak_ptr<void> allocation_owner() const override {
    return std::const_pointer_cast<TestAllocator>(shared_from_this());
  }

  std::shared_ptr<Memory> allocate(std::size_t size) override {
    ++general_count;
    return MakeMemory(size, false, false);
  }

  std::shared_ptr<Memory> allocate_exact(std::size_t size) override {
    ++exact_count;
    return MakeMemory(size, false, true);
  }

  std::shared_ptr<Memory> allocate_pinned_host(std::size_t size) override {
    ++pinned_count;
    return MakeMemory(size, true, false, Device{Device::Type::kCpu, 0});
  }

  int general_count = 0;
  int exact_count = 0;
  int pinned_count = 0;
  int live_allocations = 0;

 private:
  std::shared_ptr<Memory> MakeMemory(std::size_t size, bool pinned,
                                     bool graph_pinnable,
                                     Device allocation_device = Device{
                                         Device::Type::kCpu, 0}) {
    ++live_allocations;
    auto storage = std::make_shared<std::vector<std::byte>>(size);
    std::byte* data = storage->data();
    auto allocator = shared_from_this();
    return std::make_shared<Memory>(
        data, size, allocation_device,
        [allocator, storage = std::move(storage)](std::byte*) {
          --allocator->live_allocations;
        },
        pinned, graph_pinnable, allocator);
  }
};

}  // namespace

int main() {
  auto allocator = std::make_shared<TestAllocator>();

  Tensor general = Tensor::empty(*allocator, {2, 3}, DataType::kFloat32,
                                 TensorStorageKind::kGeneral);
  if (!general || !general.is_contiguous() || general.numel() != 6 ||
      general.nbytes() != 24 || general.device() != allocator->device() ||
      general.data() == nullptr || allocator->general_count != 1 ||
      allocator->live_allocations != 1) {
    return 1;
  }

  Tensor exact = Tensor::empty(*allocator, {4}, DataType::kInt32,
                               TensorStorageKind::kParameter);
  if (!exact || exact.nbytes() != 16 || allocator->exact_count != 1 ||
      allocator->live_allocations != 2) {
    return 1;
  }

  Tensor pinned =
      Tensor::strided_empty(*allocator, {2, 2}, {3, 1}, DataType::kFloat32,
                            TensorStorageKind::kPinnedHost);
  if (!pinned || pinned.is_contiguous() || pinned.nbytes() != 16 ||
      pinned.device().type() != Device::Type::kCpu ||
      allocator->pinned_count != 1 || allocator->live_allocations != 3) {
    return 1;
  }

  exact.reset();
  if (exact || allocator->live_allocations != 2) {
    return 1;
  }

  const auto owner = allocator->allocation_owner().lock();
  if (!owner || owner.get() != allocator.get()) {
    return 1;
  }

  return 0;
}
