#include "perception_capture.h"

#include <android/log.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <ml_perception.h>
#include <ml_snapshot.h>
#include <ml_head_tracking.h>
#include <ml_eye_tracking.h>
#include <ml_hand_tracking.h>
#pragma GCC diagnostic pop

#include "vrs_writer.h"

#define LOG_TAG "ML2Recorder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml2 {

const char* const kHandKeypointNames[kHandKeypointCount] = {
    "thumb_tip",    "thumb_ip",     "thumb_mcp",    "thumb_cmc",
    "index_tip",    "index_dip",    "index_pip",    "index_mcp",
    "middle_tip",   "middle_dip",   "middle_pip",   "middle_mcp",
    "ring_tip",     "ring_dip",     "ring_pip",     "ring_mcp",
    "pinky_tip",    "pinky_dip",    "pinky_pip",    "pinky_mcp",
    "wrist_center", "wrist_ulnar",  "wrist_radial", "hand_center",
    "index_meta",   "middle_meta",  "ring_meta",    "pinky_meta",
};

static void quat_to_forward(const MLQuaternionf& q,
                             float& dx, float& dy, float& dz) {
    dx = -2.0f * (q.x * q.z + q.w * q.y);
    dy = -2.0f * (q.y * q.z - q.w * q.x);
    dz = -(1.0f - 2.0f * (q.x * q.x + q.y * q.y));
}

// ---------------------------------------------------------------------------

PerceptionCapture::PerceptionCapture() = default;
PerceptionCapture::~PerceptionCapture() { stop(); }

bool PerceptionCapture::init(const PerceptionCaptureConfig& config) {
    config_ = config;

    if (!config_.vrs_writer) {
        LOGE("PerceptionCapture: vrs_writer is null");
        return false;
    }

    // Start perception system.
    MLPerceptionSettings settings = {};
    MLResult result = MLPerceptionInitSettings(&settings);
    if (result != MLResult_Ok) {
        LOGE("MLPerceptionInitSettings failed: %d", static_cast<int>(result));
        return false;
    }
    result = MLPerceptionStartup(&settings);
    if (result != MLResult_Ok) {
        LOGE("MLPerceptionStartup failed: %d", static_cast<int>(result));
        return false;
    }
    perception_started_ = true;

    // Head tracker.
    if (config_.enable_head_pose) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        result = MLHeadTrackingCreate(&head_tracker_);
        if (result == MLResult_Ok) {
            MLHeadTrackingStaticData sd = {};
            if (MLHeadTrackingGetStaticData(head_tracker_, &sd) == MLResult_Ok) {
                std::memcpy(head_cfuid_, &sd.coord_frame_head, 16);
                LOGI("Head tracker created");
            } else {
                config_.enable_head_pose = false;
            }
        } else {
            LOGE("MLHeadTrackingCreate failed: %d", static_cast<int>(result));
            config_.enable_head_pose = false;
        }
#pragma GCC diagnostic pop
    }

    // Eye tracker.
    if (config_.enable_eye_tracking) {
        result = MLEyeTrackingCreate(&eye_tracker_);
        if (result == MLResult_Ok) {
            MLEyeTrackingStaticData sd = {};
            if (MLEyeTrackingGetStaticData(eye_tracker_, &sd) == MLResult_Ok) {
                std::memcpy(eye_vergence_cfuid_, &sd.vergence,     16);
                std::memcpy(eye_left_cfuid_,     &sd.left_center,  16);
                std::memcpy(eye_right_cfuid_,    &sd.right_center, 16);
                LOGI("Eye tracker created");
            } else {
                config_.enable_eye_tracking = false;
            }
        } else {
            LOGE("MLEyeTrackingCreate failed: %d", static_cast<int>(result));
            config_.enable_eye_tracking = false;
        }
    }

    // Hand tracker.
    if (config_.enable_hand_tracking) {
        MLHandTrackingSettings hs = {};
        MLHandTrackingSettingsInit(&hs);
        result = MLHandTrackingCreateEx(&hs, &hand_tracker_);
        if (result == MLResult_Ok) {
            MLHandTrackingStaticData sd = {};
            MLHandTrackingStaticDataInit(&sd);
            if (MLHandTrackingGetStaticData(hand_tracker_, &sd) == MLResult_Ok) {
                for (int i = 0; i < kHandKeypointCount; ++i) {
                    std::memcpy(hand_left_cfuids_[i],
                        &sd.hand_cfuids[MLHandTrackingHandType_Left].keypoint_cfuids[i], 16);
                    std::memcpy(hand_right_cfuids_[i],
                        &sd.hand_cfuids[MLHandTrackingHandType_Right].keypoint_cfuids[i], 16);
                }
                LOGI("Hand tracker created (28 keypoints per hand)");
            } else {
                config_.enable_hand_tracking = false;
            }
        } else {
            LOGE("MLHandTrackingCreateEx failed: %d", static_cast<int>(result));
            config_.enable_hand_tracking = false;
        }
    }

    bool any = config_.enable_head_pose || config_.enable_eye_tracking ||
               config_.enable_hand_tracking;
    if (!any) {
        LOGE("PerceptionCapture: all streams failed to init");
        return false;
    }

    LOGI("PerceptionCapture: head=%d eye=%d hand=%d @ %d Hz",
         config_.enable_head_pose, config_.enable_eye_tracking,
         config_.enable_hand_tracking, config_.poll_rate_hz);
    return true;
}

bool PerceptionCapture::start() {
    if (running_.load()) return true;
    running_.store(true);
    poll_thread_ = std::thread(&PerceptionCapture::poll_loop, this);
    return true;
}

void PerceptionCapture::stop() {
    if (!running_.load() && !perception_started_) return;
    running_.store(false);
    if (poll_thread_.joinable()) poll_thread_.join();

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if (head_tracker_ != 0xFFFFFFFFFFFFFFFF) {
        MLHeadTrackingDestroy(head_tracker_);
        head_tracker_ = 0xFFFFFFFFFFFFFFFF;
    }
#pragma GCC diagnostic pop
    if (eye_tracker_  != 0xFFFFFFFFFFFFFFFF) {
        MLEyeTrackingDestroy(eye_tracker_);
        eye_tracker_ = 0xFFFFFFFFFFFFFFFF;
    }
    if (hand_tracker_ != 0xFFFFFFFFFFFFFFFF) {
        MLHandTrackingDestroy(hand_tracker_);
        hand_tracker_ = 0xFFFFFFFFFFFFFFFF;
    }
    if (perception_started_) {
        MLPerceptionShutdown();
        perception_started_ = false;
    }

    LOGI("PerceptionCapture stopped: head=%zu eye=%zu hand=%zu rows",
         head_pose_rows_.load(), eye_tracking_rows_.load(),
         hand_tracking_rows_.load());
}

void PerceptionCapture::poll_loop() {
    LOGI("Perception poll thread @ %d Hz", config_.poll_rate_hz);
    const auto interval =
        std::chrono::microseconds(1000000 / config_.poll_rate_hz);

    while (running_.load()) {
        auto frame_start = std::chrono::steady_clock::now();
        int64_t common_ts = now_ns();

        MLSnapshot* snapshot = nullptr;
        if (MLPerceptionGetSnapshot(&snapshot) != MLResult_Ok || !snapshot) {
            std::this_thread::sleep_until(frame_start + interval);
            continue;
        }

        if (config_.enable_head_pose)     sample_head_pose(snapshot, common_ts);
        if (config_.enable_eye_tracking)  sample_eye_tracking(snapshot, common_ts);
        if (config_.enable_hand_tracking) sample_hand_tracking(snapshot, common_ts);

        MLPerceptionReleaseSnapshot(snapshot);
        std::this_thread::sleep_until(frame_start + interval);
    }
    LOGI("Perception poll thread exiting");
}

void PerceptionCapture::sample_head_pose(const MLSnapshot* snapshot,
                                          int64_t common_ts) {
    MLTransform t = {};
    MLCoordinateFrameUID cfuid;
    std::memcpy(&cfuid, head_cfuid_, 16);

    if (MLSnapshotGetTransform(snapshot, &cfuid, &t) != MLResult_Ok) return;

    config_.vrs_writer->write_head_pose(
        common_ts, common_ts,  // head pose has no separate sensor timestamp
        t.position.x, t.position.y, t.position.z,
        t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z);
    head_pose_rows_.fetch_add(1);
}

void PerceptionCapture::sample_eye_tracking(const MLSnapshot* snapshot,
                                             int64_t common_ts) {
    MLEyeTrackingStateEx state = {};
    MLEyeTrackingStateInit(&state);
    if (MLEyeTrackingGetStateEx(eye_tracker_, &state) != MLResult_Ok) return;
    if (state.timestamp == last_eye_ts_) return;
    last_eye_ts_ = state.timestamp;

    MLTransform left_t = {}, right_t = {}, verg_t = {};
    MLCoordinateFrameUID lc, rc, vc;
    std::memcpy(&lc, eye_left_cfuid_,     16);
    std::memcpy(&rc, eye_right_cfuid_,    16);
    std::memcpy(&vc, eye_vergence_cfuid_, 16);

    if (MLSnapshotGetTransform(snapshot, &lc, &left_t)  != MLResult_Ok) return;
    if (MLSnapshotGetTransform(snapshot, &rc, &right_t) != MLResult_Ok) return;
    MLSnapshotGetTransform(snapshot, &vc, &verg_t);  // vergence is optional

    float ldx, ldy, ldz, rdx, rdy, rdz;
    quat_to_forward(left_t.rotation,  ldx, ldy, ldz);
    quat_to_forward(right_t.rotation, rdx, rdy, rdz);

    config_.vrs_writer->write_eye_tracking(
        common_ts, static_cast<int64_t>(state.timestamp),
        left_t.position.x,  left_t.position.y,  left_t.position.z,
        ldx, ldy, ldz,
        right_t.position.x, right_t.position.y, right_t.position.z,
        rdx, rdy, rdz,
        verg_t.position.x,  verg_t.position.y,  verg_t.position.z);
    eye_tracking_rows_.fetch_add(1);
}

void PerceptionCapture::sample_hand_tracking(const MLSnapshot* snapshot,
                                              int64_t common_ts) {
    MLHandTrackingData hd = {};
    MLHandTrackingDataInit(&hd);
    if (MLHandTrackingGetData(hand_tracker_, &hd) != MLResult_Ok) return;
    if (hd.timestamp_ns == last_hand_ts_) return;
    last_hand_ts_ = hd.timestamp_ns;

    const auto& ls = hd.hand_state[MLHandTrackingHandType_Left];
    const auto& rs = hd.hand_state[MLHandTrackingHandType_Right];
    if (!ls.is_hand_detected && !rs.is_hand_detected) return;

    float left_kp[kHandKeypointCount * 3];
    float right_kp[kHandKeypointCount * 3];

    auto resolve = [&](const MLHandTrackingHandState& state,
                       const uint8_t cfuids[][16],
                       float* out) {
        for (int i = 0; i < kHandKeypointCount; ++i) {
            float* p = out + i * 3;
            if (state.is_hand_detected && state.keypoints_mask[i]) {
                MLCoordinateFrameUID cfuid;
                std::memcpy(&cfuid, cfuids[i], 16);
                MLTransform t = {};
                if (MLSnapshotGetTransform(snapshot, &cfuid, &t) == MLResult_Ok) {
                    p[0] = t.position.x;
                    p[1] = t.position.y;
                    p[2] = t.position.z;
                    continue;
                }
            }
            p[0] = p[1] = p[2] = std::numeric_limits<float>::quiet_NaN();
        }
    };

    resolve(ls, hand_left_cfuids_,  left_kp);
    resolve(rs, hand_right_cfuids_, right_kp);

    config_.vrs_writer->write_hand_tracking(
        common_ts, hd.timestamp_ns,
        ls.is_hand_detected, rs.is_hand_detected,
        ls.hand_confidence,  rs.hand_confidence,
        left_kp, right_kp);
    hand_tracking_rows_.fetch_add(1);

    // Cache thumb/index for gesture detection.
    {
        std::lock_guard<std::mutex> lock(gesture_mutex_);
        left_hand_valid_  = ls.is_hand_detected;
        right_hand_valid_ = rs.is_hand_detected;
        if (left_hand_valid_) {
            std::memcpy(left_thumb_tip_,  left_kp + 0 * 3, 12);
            std::memcpy(left_index_tip_,  left_kp + 4 * 3, 12);
        }
        if (right_hand_valid_) {
            std::memcpy(right_thumb_tip_,  right_kp + 0 * 3, 12);
            std::memcpy(right_index_tip_,  right_kp + 4 * 3, 12);
        }
    }
}

bool PerceptionCapture::is_both_hands_pinching(float threshold_m) const {
    std::lock_guard<std::mutex> lock(gesture_mutex_);
    if (!left_hand_valid_ || !right_hand_valid_) return false;

    auto dist = [](const float a[3], const float b[3]) {
        float dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    };
    return dist(left_thumb_tip_, left_index_tip_)   < threshold_m &&
           dist(right_thumb_tip_, right_index_tip_) < threshold_m;
}

}  // namespace ml2
