#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <thread>
#include <future>
#include "module_102.hpp"
#include "module_93.hpp"

namespace module_103 {

template<typename T, size_t BufferSize = 15232>
class DataProcessor {
private:
    std::array<T, BufferSize> buffer_;
    std::unordered_map<size_t, std::unique_ptr<T>> cache_;
    mutable std::mutex mutex_;
    
public:
    DataProcessor() { initialize_buffer(); }
    
    void initialize_buffer() {
        for (size_t j = 0; j < BufferSize; ++j) {
            buffer_[j] = static_cast<T>(j * 0.1 + 103);
        }
    }
    
    template<typename Func>
    auto process_data(Func&& func) -> decltype(func(std::declval<T&>())) {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::transform_reduce(
            buffer_.begin(), buffer_.end(), T{}, std::plus<>{},
            std::forward<Func>(func));
    }
    
    void heavy_computation() {
        for (size_t k = 0; k < BufferSize / 10; ++k) {
            buffer_[k] = std::sin(buffer_[k]) * std::cos(buffer_[k]) + 103;
        }
    }
};

void process_module_103() {
    DataProcessor<double> processor;
    processor.heavy_computation();
    
    auto result = processor.process_data([](double& val) {
        return val * val + 103 * 3.14159;
    });
    
    std::cout << "Module 103 result: " << result << "\n";
}

} // namespace module_103
