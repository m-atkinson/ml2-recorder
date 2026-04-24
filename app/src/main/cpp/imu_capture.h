#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "timestamp.h"

// Forward-declare ML2 sensor types.
struct ASensorManager;
struct ASensorEventQueue;
struct ASensor;
struct ALooper;

namespace ml2 {

class VrsWriter;  // forward declaration — no VRS headers needed here

struct ImuCaptureConfig {
    VrsWriter* vrs_writer = nullptr;
    int sample_rate_us = 5000;  // 200 Hz default (5000 µs period)
};

/// Captures accelerometer and gyroscope data from the Android Sensor API.
/// Writes to the VrsWriter IMU stream.  One VRS record per sensor event,
/// with fields: common_timestamp_ns, sensor_timestamp_ns, unit_id,
/// accel[3], gyro[3].
class ImuCapture {
public:
    ImuCapture();
    ~ImuCapture();

    ImuCapture(const ImuCapture&) = delete;
    ImuCapture& operator=(const ImuCapture&) = delete;

    bool init(const ImuCaptureConfig& config);
    bool start();
    void stop();

    bool   is_running()   const { return running_.load(); }
    size_t rows_written() const { return rows_written_.load(); }

private:
    void sensor_loop();

    ImuCaptureConfig config_;

    std::thread sensor_thread_;
    std::atomic<bool>   running_{false};
    std::atomic<size_t> rows_written_{0};

    struct SensorUnit {
        const ASensor* accel = nullptr;
        const ASensor* gyro  = nullptr;
        int unit_id = 0;
    };
    std::vector<SensorUnit> units_;
};

}  // namespace ml2
