#include "ring_buffer.h"

namespace ml2 {

RingBuffer::RingBuffer(size_t capacity)
    : slots_(capacity), capacity_(capacity) {}

void RingBuffer::push(const uint8_t* data, size_t size, int64_t timestamp_ns) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (count_ == capacity_) {
        // Buffer full — drop oldest (advance tail)
        slots_[tail_].occupied = false;
        tail_ = (tail_ + 1) % capacity_;
        --count_;
        ++drops_;
    }

    auto& slot = slots_[head_];
    slot.data.assign(data, data + size);
    slot.timestamp_ns = timestamp_ns;
    slot.occupied = true;

    head_ = (head_ + 1) % capacity_;
    ++count_;
}

bool RingBuffer::pop(std::function<void(const uint8_t*, size_t, int64_t)> callback) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (count_ == 0) return false;

    auto& slot = slots_[tail_];
    callback(slot.data.data(), slot.data.size(), slot.timestamp_ns);
    slot.occupied = false;
    slot.data.clear();

    tail_ = (tail_ + 1) % capacity_;
    --count_;
    return true;
}

size_t RingBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
}

size_t RingBuffer::drops() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return drops_;
}

bool RingBuffer::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_ == 0;
}

void RingBuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& slot : slots_) {
        slot.data.clear();
        slot.occupied = false;
    }
    head_ = 0;
    tail_ = 0;
    count_ = 0;
}

} // namespace ml2
