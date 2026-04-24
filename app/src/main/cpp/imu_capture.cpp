#include "imu_capture.h"

#include <android/log.h>
#include <android/looper.h>
#include <android/sensor.h>

#include <unordered_map>

#include "vrs_writer.h"

#define LOG_TAG "ML2Recorder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml2 {

ImuCapture::ImuCapture() = default;
ImuCapture::~ImuCapture() { stop(); }

bool ImuCapture::init(const ImuCaptureConfig& config) {
    config_ = config;

    if (!config_.vrs_writer) {
        LOGE("ImuCapture: vrs_writer is null");
        return false;
    }

    ASensorManager* mgr =
        ASensorManager_getInstanceForPackage("com.ml2.recorder");
    if (!mgr) {
        LOGE("ASensorManager_getInstanceForPackage returned null");
        return false;
    }

    ASensorList sensor_list = nullptr;
    int count = ASensorManager_getSensorList(mgr, &sensor_list);

    std::vector<const ASensor*> accels, gyros;
    for (int i = 0; i < count; ++i) {
        int type = ASensor_getType(sensor_list[i]);
        if (type == ASENSOR_TYPE_ACCELEROMETER)  accels.push_back(sensor_list[i]);
        else if (type == ASENSOR_TYPE_GYROSCOPE) gyros.push_back(sensor_list[i]);
    }

    LOGI("IMU discovery: %zu accelerometers, %zu gyroscopes",
         accels.size(), gyros.size());

    size_t n = std::max(accels.size(), gyros.size());
    if (n == 0) {
        LOGE("No IMU sensors found");
        return false;
    }

    units_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        units_[i].unit_id = static_cast<int>(i);
        if (i < accels.size()) units_[i].accel = accels[i];
        if (i < gyros.size())  units_[i].gyro  = gyros[i];
    }

    LOGI("IMU init: %zu units, sample period %d µs", n, config_.sample_rate_us);
    return true;
}

bool ImuCapture::start() {
    if (running_.load()) return true;
    running_.store(true);
    sensor_thread_ = std::thread(&ImuCapture::sensor_loop, this);
    LOGI("IMU capture started");
    return true;
}

void ImuCapture::stop() {
    if (!running_.load()) return;
    running_.store(false);
    if (sensor_thread_.joinable()) sensor_thread_.join();
    LOGI("IMU capture stopped. Rows written: %zu", rows_written_.load());
}

void ImuCapture::sensor_loop() {
    LOGI("IMU sensor thread started");

    ASensorManager* mgr =
        ASensorManager_getInstanceForPackage("com.ml2.recorder");
    if (!mgr) { LOGE("IMU thread: no sensor manager"); return; }

    ALooper* looper = ALooper_prepare(ALOOPER_PREPARE_ALLOW_NON_CALLBACKS);
    if (!looper) { LOGE("IMU thread: ALooper_prepare failed"); return; }

    ASensorEventQueue* queue =
        ASensorManager_createEventQueue(mgr, looper, 0, nullptr, nullptr);
    if (!queue) { LOGE("IMU thread: no event queue"); return; }

    for (auto& unit : units_) {
        if (unit.accel) {
            ASensorEventQueue_enableSensor(queue, unit.accel);
            ASensorEventQueue_setEventRate(queue, unit.accel,
                                            config_.sample_rate_us);
        }
        if (unit.gyro) {
            ASensorEventQueue_enableSensor(queue, unit.gyro);
            ASensorEventQueue_setEventRate(queue, unit.gyro,
                                            config_.sample_rate_us);
        }
    }

    struct ImuState { float ax=0,ay=0,az=0, gx=0,gy=0,gz=0; };
    std::unordered_map<int, ImuState> states;

    ASensorEvent events[16];

    while (running_.load()) {
        ALooper_pollOnce(10, nullptr, nullptr, nullptr);

        int n;
        while ((n = ASensorEventQueue_getEvents(queue, events, 16)) > 0) {
            for (int i = 0; i < n; ++i) {
                const ASensorEvent& ev = events[i];

                int unit_idx = -1;
                bool is_gyro = false;

                if (ev.type == ASENSOR_TYPE_ACCELEROMETER) {
                    for (size_t u = 0; u < units_.size(); ++u) {
                        if (units_[u].accel &&
                            ASensor_getHandle(units_[u].accel) == ev.sensor) {
                            unit_idx = static_cast<int>(u);
                            is_gyro = false;
                            break;
                        }
                    }
                } else if (ev.type == ASENSOR_TYPE_GYROSCOPE) {
                    for (size_t u = 0; u < units_.size(); ++u) {
                        if (units_[u].gyro &&
                            ASensor_getHandle(units_[u].gyro) == ev.sensor) {
                            unit_idx = static_cast<int>(u);
                            is_gyro = true;
                            break;
                        }
                    }
                }

                if (unit_idx < 0) continue;

                auto& st = states[unit_idx];
                if (is_gyro) {
                    st.gx = ev.data[0]; st.gy = ev.data[1]; st.gz = ev.data[2];
                } else {
                    st.ax = ev.data[0]; st.ay = ev.data[1]; st.az = ev.data[2];
                }

                int64_t common_ts  = now_ns();
                int64_t sensor_ts  = ev.timestamp;

                config_.vrs_writer->write_imu(
                    common_ts, sensor_ts, unit_idx,
                    st.ax, st.ay, st.az,
                    st.gx, st.gy, st.gz);

                rows_written_.fetch_add(1);
            }
        }
    }

    for (auto& unit : units_) {
        if (unit.accel) ASensorEventQueue_disableSensor(queue, unit.accel);
        if (unit.gyro)  ASensorEventQueue_disableSensor(queue, unit.gyro);
    }
    ASensorManager_destroyEventQueue(mgr, queue);

    LOGI("IMU sensor thread exiting. Rows: %zu", rows_written_.load());
}

}  // namespace ml2
