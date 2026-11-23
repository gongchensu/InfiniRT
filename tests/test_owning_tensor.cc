#include <infini/rt.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using infini::rt::DataType;
using infini::rt::Device;
using infini::rt::Tensor;

const Device kCpu{Device::Type::kCpu, 0};

template <typename Exception, typename Function>
bool Throws(Function&& function) {
  try {
    function();
  } catch (const Exception&) {
    return true;
  } catch (...) {
  }
  return false;
}

bool SameData(const Tensor& left, const Tensor& right) {
  return left.data() == right.data();
}

std::string ShapeString(const Tensor& tensor) {
  std::ostringstream out;
  out << '[';
  for (std::size_t index = 0; index < tensor.ndim(); ++index) {
    if (index != 0) out << ',';
    out << tensor.size(index);
  }
  return std::string{out.str() + ']'};
}

void TestExternalStorage() {
  std::int32_t values[6]{};
  const Tensor tensor =
      Tensor::from_blob(values, {2, 3}, DataType::kInt32, kCpu);

  if (tensor.data() != values || tensor.ndim() != 2 ||
      tensor.shape() != std::vector<std::size_t>{2, 3} ||
      tensor.strides() != std::vector<std::ptrdiff_t>{3, 1} ||
      tensor.dtype() != DataType::kInt32 || tensor.device() != kCpu ||
      !tensor.is_contiguous() || tensor.numel() != 6 || tensor.size(0) != 2 ||
      tensor.size(1) != 3 || tensor.stride(0) != 3 || tensor.stride(1) != 1 ||
      tensor.element_size() != 4 || tensor.nbytes() != 24) {
    throw std::runtime_error{"contiguous from_blob metadata mismatch"};
  }

  const auto view = tensor.view();
  if (view.data() != values || view.shape() != tensor.shape() ||
      view.strides() != tensor.strides() || view.dtype() != tensor.dtype() ||
      view.device() != tensor.device()) {
    throw std::runtime_error{"owning Tensor view mismatch"};
  }
}

void TestStridedExternalStorage() {
  float values[4]{};
  const Tensor tensor = Tensor::strided_from_blob(values, {2, 2}, {3, 1},
                                                  DataType::kFloat32, kCpu);

  if (!SameData(tensor, tensor) || tensor.is_contiguous() ||
      tensor.strides() != std::vector<std::ptrdiff_t>{3, 1} ||
      tensor.nbytes() != 16) {
    throw std::runtime_error{"strided from_blob metadata mismatch"};
  }
}

void TestViewsShareStorage() {
  std::uint8_t values[24]{};
  Tensor tensor = Tensor::from_blob(values, {2, 3, 4}, DataType::kUInt8, kCpu);

  const Tensor squeezed = tensor.unsqueeze(0).squeeze(0);
  const Tensor unsqueezed = tensor.unsqueeze(0);
  const Tensor narrowed = tensor.narrow({{2, 1, 2}});
  const Tensor permuted = tensor.permute({2, 1, 0});
  const Tensor reshaped = tensor.reshape({6, 4});
  const Tensor strided = tensor.as_strided({4, 6}, {1, 4});
  const Tensor broadcast = tensor.as_strided({2, 1, 4}, {0, 0, 1});

  if (squeezed.shape() != std::vector<std::size_t>{2, 3, 4} ||
      unsqueezed.shape() != std::vector<std::size_t>{1, 2, 3, 4} ||
      narrowed.shape() != std::vector<std::size_t>{2, 3, 2} ||
      permuted.shape() != std::vector<std::size_t>{4, 3, 2} ||
      permuted.strides() != std::vector<std::ptrdiff_t>{1, 4, 12} ||
      reshaped.shape() != std::vector<std::size_t>{6, 4} ||
      strided.shape() != std::vector<std::size_t>{4, 6}) {
    throw std::runtime_error{
        "view shape mismatch: squeezed=" + ShapeString(squeezed) +
        " unsqueezed=" + ShapeString(unsqueezed) + " narrowed=" +
        ShapeString(narrowed) + " permuted=" + ShapeString(permuted) +
        " reshaped=" + ShapeString(reshaped) +
        " strided=" + ShapeString(strided)};
  }

  if (static_cast<const std::uint8_t*>(narrowed.data()) - values != 1 ||
      !SameData(tensor, squeezed) || !SameData(tensor, unsqueezed) ||
      !SameData(tensor, permuted) || !SameData(tensor, reshaped) ||
      !SameData(tensor, strided)) {
    throw std::runtime_error{"views must share storage"};
  }

  if (broadcast.shape() != std::vector<std::size_t>{2, 1, 4} ||
      broadcast.strides() != std::vector<std::ptrdiff_t>{0, 0, 1} ||
      broadcast.numel() != 8 || broadcast.nbytes() != 8 ||
      broadcast.is_contiguous() || !SameData(tensor, broadcast)) {
    throw std::runtime_error{"broadcast view metadata mismatch"};
  }
}

void TestScalarAndEmptyTensors() {
  float scalar{};
  const Tensor scalar_tensor =
      Tensor::from_blob(&scalar, {}, DataType::kFloat32, kCpu);
  if (scalar_tensor.ndim() != 0 || scalar_tensor.numel() != 1 ||
      scalar_tensor.nbytes() != 4 || !scalar_tensor.is_contiguous() ||
      scalar_tensor.data() != &scalar) {
    throw std::runtime_error{"scalar tensor metadata mismatch"};
  }

  const Tensor empty_tensor =
      Tensor::from_blob(nullptr, {2, 0}, DataType::kFloat32, kCpu);
  if (empty_tensor.data() != nullptr || empty_tensor.numel() != 0 ||
      empty_tensor.nbytes() != 0 || !empty_tensor.is_contiguous()) {
    throw std::runtime_error{"empty tensor metadata mismatch"};
  }

  const Tensor reshaped_empty = empty_tensor.reshape({0, 4});
  if (reshaped_empty.shape() != std::vector<std::size_t>{0, 4} ||
      reshaped_empty.numel() != 0 || reshaped_empty.data() != nullptr) {
    throw std::runtime_error{"empty tensor reshape mismatch"};
  }
}

void TestHighRankMetadata() {
  std::int8_t value{};
  const Tensor tensor = Tensor::from_blob(&value, {1, 1, 1, 1, 1, 1, 1, 1, 1},
                                          DataType::kInt8, kCpu);
  if (tensor.ndim() != 9 || tensor.numel() != 1 || tensor.shape().size() != 9 ||
      tensor.strides().size() != 9) {
    throw std::runtime_error{"rank-nine tensor metadata mismatch"};
  }
}

void TestExternalOwnerLifetime() {
  auto owner = std::make_shared<std::int32_t>(7);
  const std::weak_ptr<void> weak_owner = owner;
  std::uint8_t value{};
  Tensor tensor =
      Tensor::from_blob(&value, {1}, DataType::kUInt8, kCpu, std::move(owner));

  owner.reset();
  if (weak_owner.expired() || weak_owner.lock() == nullptr) {
    throw std::runtime_error{"owning Tensor dropped its external owner"};
  }

  tensor.reset();
  if (tensor || !weak_owner.expired()) {
    throw std::runtime_error{"external owner outlived owning Tensor"};
  }
}

void TestValidation() {
  std::int8_t value{};
  if (!Throws<std::invalid_argument>(
          [&] { Tensor::from_blob(nullptr, {1}, DataType::kInt8, kCpu); })) {
    throw std::runtime_error{"non-empty null blob was accepted"};
  }

  Tensor tensor = Tensor::from_blob(&value, {1}, DataType::kInt8, kCpu);
  std::int8_t two_values[2]{};
  Tensor two_element_tensor =
      Tensor::from_blob(two_values, {2}, DataType::kInt8, kCpu);
  if (!Throws<std::out_of_range>([&] { tensor.as_strided({2}, {1}); })) {
    throw std::runtime_error{"out-of-range view was accepted"};
  }

  if (!Throws<std::runtime_error>([&] { two_element_tensor.squeeze(0); }) ||
      !Throws<std::out_of_range>([&] { tensor.size(1); }) ||
      !Throws<std::invalid_argument>([&] { tensor.permute({0, 0}); }) ||
      !Throws<std::invalid_argument>([&] { tensor.reshape({2}); })) {
    throw std::runtime_error{"invalid view operation was accepted"};
  }

  if (!Throws<std::invalid_argument>([&] {
        Tensor::strided_from_blob(&value, {1}, {-1}, DataType::kInt8, kCpu);
      })) {
    throw std::runtime_error{"negative stride was accepted"};
  }

  if (!Throws<std::length_error>([&] {
        Tensor::from_blob(&value, {std::numeric_limits<std::size_t>::max(), 2},
                          DataType::kInt8, kCpu);
      })) {
    throw std::runtime_error{"stride overflow was accepted"};
  }

  const Tensor empty;
  if (!Throws<std::logic_error>([&] { empty.shape(); })) {
    throw std::runtime_error{"empty Tensor accessor was accepted"};
  }
}

}  // namespace

int main() {
  TestExternalStorage();
  TestStridedExternalStorage();
  TestViewsShareStorage();
  TestScalarAndEmptyTensors();
  TestHighRankMetadata();
  TestExternalOwnerLifetime();
  TestValidation();
  return 0;
}
