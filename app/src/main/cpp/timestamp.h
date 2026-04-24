#pragma once

#include <cstdint>
#include <chrono>

namespace ml2 {

// Returns the current monotonic clock time in nanoseconds.
// On Android, this wraps SystemClock.elapsedRealtimeNanos() via clock_gettime(CLOCK_BOOTTIME).
// On host (for testing), falls back to steady_clock.
int64_t now_ns();

// Converts a sensor-provided timestamp to nanoseconds if needed.
// Identity function for now — exists as an extension point if sensor clocks differ.
inline int64_t sensor_ts_to_ns(int64_t sensor_ts) { return sensor_ts; }

} // namespace ml2
