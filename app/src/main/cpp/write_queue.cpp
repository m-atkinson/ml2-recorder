#include "write_queue.h"

#include <android/log.h>

#define LOG_TAG "ML2Recorder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml2 {

WriteQueue::WriteQueue(size_t max_queue_depth, int num_writers, EncodeFn encode_fn)
    : encode_fn_(std::move(encode_fn)),
      max_depth_(max_queue_depth) {
    running_.store(true);
    writers_.reserve(num_writers);
    for (int i = 0; i < num_writers; ++i) {
        writers_.emplace_back(&WriteQueue::writer_loop, this);
    }
}

WriteQueue::~WriteQueue() {
    // If the caller forgot to drain, do it now.
    if (running_.load()) {
        drain_and_stop();
    }
}

bool WriteQueue::submit(WriteTask task) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (queue_.size() >= max_depth_) {
        dropped_.fetch_add(1);
        return false;
    }

    queue_.push_back(std::move(task));
    cv_.notify_one();
    return true;
}

void WriteQueue::drain_and_stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        draining_.store(true);
        // Don't set running_ = false yet; writers need to keep processing.
        cv_.notify_all();
    }

    // Wait for the queue to empty.
    {
        std::unique_lock<std::mutex> lock(mutex_);
        drain_cv_.wait(lock, [this] { return queue_.empty(); });
    }

    // Now signal writers to exit.
    running_.store(false);
    cv_.notify_all();

    for (auto& t : writers_) {
        if (t.joinable()) t.join();
    }

    LOGI("WriteQueue drained: written=%zu, dropped=%zu",
         written_.load(), dropped_.load());
}

WriteQueueStats WriteQueue::stats() const {
    WriteQueueStats s;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        s.queued = queue_.size();
    }
    s.written = written_.load();
    s.dropped = dropped_.load();
    return s;
}

size_t WriteQueue::pending() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

void WriteQueue::writer_loop() {
    while (true) {
        WriteTask task;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] {
                return !queue_.empty() || !running_.load();
            });

            if (queue_.empty()) {
                if (!running_.load()) break;
                continue;
            }

            task = std::move(queue_.front());
            queue_.pop_front();

            // If draining and queue is now empty, notify the drain waiter.
            if (draining_.load() && queue_.empty()) {
                drain_cv_.notify_all();
            }
        }

        // Encode and write outside the lock.
        if (encode_fn_) {
            if (encode_fn_(task)) {
                written_.fetch_add(1);
            } else {
                LOGE("WriteQueue: encode_fn failed for frame %u (stream %d)",
                     task.frame_index, task.stream_id);
            }
        }
    }
}

} // namespace ml2
