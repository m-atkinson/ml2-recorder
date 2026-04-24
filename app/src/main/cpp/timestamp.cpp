#include "timestamp.h"

#ifdef __ANDROID__
#include <time.h>
#endif

namespace ml2 {

int64_t now_ns() {
#ifdef __ANDROID__
    struct timespec ts;
    clock_gettime(CLOCK_BOOTTIME, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
#else
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
#endif
}

} // namespace ml2
