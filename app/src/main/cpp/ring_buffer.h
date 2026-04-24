#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace ml2 {

// A fixed-capacity ring buffer for sensor frame data. When full, the oldest
// frame is silently dropped (overwritten). Thread-safe for single-producer
// single-consumer usage with the mutex, or multi-producer with lock.
//
// Each slot holds a byte buffer (e.g., a JPEG-encoded frame) plus a timestamp.
struct FrameSlot {
    std::vector<uint8_t> data;
    int64_t timestamp_ns = 0;
    bool occupied = false;
};

class RingBuffer {
public:
    explicit RingBuffer(size_t capacity);

    // Push a frame. If the buffer is full, overwrites the oldest frame
    // and increments the drop counter.
    void push(const uint8_t* data, size_t size, int64_t timestamp_ns);

    // Pop the oldest frame. Returns false if buffer is empty.
    // Calls the callback with a pointer to the data and its size.
    // The data is valid only for the duration of the callback.
    bool pop(std::function<void(const uint8_t* data, size_t size, int64_t timestamp_ns)> callback);

    // Number of frames currently in the buffer.
    size_t size() const;

    // Number of frames dropped since construction.
    size_t drops() const;

    // Whether the buffer is empty.
    bool empty() const;

    // Reset to empty state.
    void clear();

private:
    std::vector<FrameSlot> slots_;
    size_t capacity_;
    size_t head_ = 0; // next write position
    size_t tail_ = 0; // next read position
    size_t count_ = 0;
    size_t drops_ = 0;
    mutable std::mutex mutex_;
};

} // namespace ml2
