// Minimal shim replacing boost::interprocess::interprocess_semaphore for
// Android NDK builds where Boost is not available.
// Implements the same API used by vrs/os/Semaphore.h:
//   - constructor(unsigned int count)
//   - post()
//   - wait()
//   - timed_wait(abs_time_point) -> bool
#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>

namespace boost {
namespace interprocess {

class interprocess_semaphore {
 public:
  explicit interprocess_semaphore(unsigned int count) : count_(count) {}

  void post() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++count_;
    cv_.notify_one();
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return count_ > 0u; });
    --count_;
  }

  // VRS calls this as: interprocess_semaphore::timed_wait(steady_clock::time_point)
  template <class TimePoint>
  bool timed_wait(const TimePoint& abs_time) {
    std::unique_lock<std::mutex> lock(mutex_);
    bool ok = cv_.wait_until(lock, abs_time, [this] { return count_ > 0u; });
    if (ok) {
      --count_;
    }
    return ok;
  }

 private:
  unsigned int count_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

} // namespace interprocess
} // namespace boost
