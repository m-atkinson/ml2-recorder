#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ml2 {

/// A task submitted to the WriteQueue for async encoding + disk write.
struct WriteTask {
    std::vector<uint8_t> data;      // Raw pixel/sensor data
    int64_t timestamp_ns = 0;       // Common-clock timestamp
    int64_t sensor_timestamp_ns = 0; // Sensor-native timestamp
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t stride = 0;            // Row stride in bytes (0 = tightly packed)
    uint32_t frame_index = 0;       // Sequential frame number (assigned at submit)
    int stream_id = 0;              // Opaque caller context (e.g., camera index)

    // Per-frame camera pose (world_from_camera), captured synchronously at
    // frame-arrival time. Quaternion order is SDK-native: x, y, z, w.
    // pose_valid == 0 means the encoder should write camera_pose_valid=0.
    uint8_t pose_valid = 0;
    float pose_position[3] = {0.0f, 0.0f, 0.0f};
    float pose_orientation[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

/// Stats returned by the WriteQueue.
struct WriteQueueStats {
    size_t queued = 0;    // Currently waiting in queue
    size_t written = 0;   // Successfully written to disk
    size_t dropped = 0;   // Dropped due to backpressure
};

/// Multi-threaded async write queue.  Capture threads submit raw frame data;
/// a pool of writer threads encodes and writes to disk in the background.
///
/// The caller provides an encode function that handles the actual encoding
/// and file I/O.  The WriteQueue handles threading, backpressure, and
/// graceful draining on shutdown.
class WriteQueue {
public:
    /// Callback type: receives a WriteTask, encodes it, writes to disk.
    /// Must return true on success.
    using EncodeFn = std::function<bool(const WriteTask& task)>;

    /// @param max_queue_depth  Maximum number of pending tasks before dropping.
    /// @param num_writers      Number of background writer threads.
    /// @param encode_fn        User-provided encode + write function.
    WriteQueue(size_t max_queue_depth, int num_writers, EncodeFn encode_fn);
    ~WriteQueue();

    WriteQueue(const WriteQueue&) = delete;
    WriteQueue& operator=(const WriteQueue&) = delete;

    /// Submit a frame for async writing.  Returns false if the queue is full
    /// (frame is dropped and the drop counter is incremented).
    bool submit(WriteTask task);

    /// Drain all remaining tasks and join writer threads.  Blocks until
    /// every enqueued frame has been processed.  Call this during shutdown.
    void drain_and_stop();

    /// Get current stats (thread-safe).
    WriteQueueStats stats() const;

    /// Number of frames successfully written.
    size_t written() const { return written_.load(); }

    /// Number of frames dropped due to full queue.
    size_t dropped() const { return dropped_.load(); }

    /// Current number of tasks waiting in the queue.
    size_t pending() const;

private:
    void writer_loop();

    EncodeFn encode_fn_;
    size_t max_depth_;

    std::deque<WriteTask> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;          // wakes writers when work arrives
    std::condition_variable drain_cv_;    // wakes drain_and_stop() when queue empties

    std::vector<std::thread> writers_;
    std::atomic<bool> running_{false};
    std::atomic<bool> draining_{false};

    std::atomic<size_t> written_{0};
    std::atomic<size_t> dropped_{0};
};

} // namespace ml2
