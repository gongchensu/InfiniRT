#ifndef INFINI_RT_MEMORY_H_
#define INFINI_RT_MEMORY_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

namespace infini::rt {

// Owns the release obligation for one allocation, not producer completion.
// The optional owner token preserves allocation identity without making RT
// depend on a higher-level runtime class.
class Memory {
 public:
  using Deleter = std::function<void(std::byte*)>;

  Memory(std::byte* data, std::size_t size, Device device, Deleter deleter,
         bool pinned = false, bool graph_pinnable = false,
         std::weak_ptr<void> allocation_owner = {})
      : data_{data},
        size_{size},
        device_{device},
        deleter_{std::move(deleter)},
        pinned_{pinned},
        graph_pinnable_{graph_pinnable},
        allocation_owner_{std::move(allocation_owner)} {}

  Memory(const Memory&) = delete;
  Memory& operator=(const Memory&) = delete;
  Memory(Memory&&) = delete;
  Memory& operator=(Memory&&) = delete;

  ~Memory() {
    if (deleter_) {
      deleter_(data_);
    }
  }

  std::byte* data() { return data_; }
  const std::byte* data() const { return data_; }
  Device device() const { return device_; }
  std::size_t size() const { return size_; }
  bool is_pinned() const { return pinned_; }
  bool is_graph_pinnable() const { return graph_pinnable_; }
  std::shared_ptr<void> allocation_owner() const {
    return allocation_owner_.lock();
  }

 private:
  std::byte* data_;
  std::size_t size_;
  Device device_;
  Deleter deleter_;
  bool pinned_;
  bool graph_pinnable_;
  std::weak_ptr<void> allocation_owner_;
};

}  // namespace infini::rt

#endif
