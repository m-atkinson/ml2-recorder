#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "timestamp.h"

typedef uint64_t MLHandle;
typedef int64_t  MLTime;
struct MLSnapshot;

namespace ml2 {

class VrsWriter;

constexpr int kHandKeypointCount = 28;
extern const char* const kHandKeypointNames[kHandKeypointCount];

struct PerceptionCaptureConfig {
    VrsWriter* vrs_writer = nullptr;
    int  poll_rate_hz = 30;
    bool enable_head_pose    = true;
    bool enable_eye_tracking = true;
    bool enable_hand_tracking = true;
};

/// Polls head pose, eye tracking, and hand tracking at a fixed rate, writing
/// each sample to the VrsWriter.  All three share the ML2 Perception system
/// snapshot, so they are combined into a single module.
class PerceptionCapture {
public:
    PerceptionCapture();
    ~PerceptionCapture();

    PerceptionCapture(const PerceptionCapture&) = delete;
    PerceptionCapture& operator=(const PerceptionCapture&) = delete;

    bool init(const PerceptionCaptureConfig& config);
    bool start();
    void stop();

    bool   is_running()         const { return running_.load(); }
    size_t head_pose_rows()     const { return head_pose_rows_.load(); }
    size_t eye_tracking_rows()  const { return eye_tracking_rows_.load(); }
    size_t hand_tracking_rows() const { return hand_tracking_rows_.load(); }

    bool is_both_hands_pinching(float threshold_m = 0.03f) const;

    /// Expose the head-tracker handle so other captures (e.g. RGB) can pass it
    /// to MLCVCameraGetFramePose. Returns 0xFFFFFFFFFFFFFFFF if not created.
    MLHandle head_tracking_handle() const { return head_tracker_; }

private:
    void poll_loop();
    void sample_head_pose(const MLSnapshot* snapshot, int64_t common_ts);
    void sample_eye_tracking(const MLSnapshot* snapshot, int64_t common_ts);
    void sample_hand_tracking(const MLSnapshot* snapshot, int64_t common_ts);

    PerceptionCaptureConfig config_;

    std::thread poll_thread_;
    std::atomic<bool> running_{false};

    std::atomic<size_t> head_pose_rows_{0};
    std::atomic<size_t> eye_tracking_rows_{0};
    std::atomic<size_t> hand_tracking_rows_{0};

    bool     perception_started_ = false;
    MLHandle head_tracker_ = 0xFFFFFFFFFFFFFFFF;
    MLHandle eye_tracker_  = 0xFFFFFFFFFFFFFFFF;
    MLHandle hand_tracker_ = 0xFFFFFFFFFFFFFFFF;

    uint8_t head_cfuid_[16]          = {};
    uint8_t eye_vergence_cfuid_[16]  = {};
    uint8_t eye_left_cfuid_[16]      = {};
    uint8_t eye_right_cfuid_[16]     = {};
    uint8_t hand_left_cfuids_[28][16]  = {};
    uint8_t hand_right_cfuids_[28][16] = {};

    MLTime last_eye_ts_  = 0;
    MLTime last_hand_ts_ = 0;

    mutable std::mutex gesture_mutex_;
    float left_thumb_tip_[3]  = {};
    float left_index_tip_[3]  = {};
    float right_thumb_tip_[3] = {};
    float right_index_tip_[3] = {};
    bool  left_hand_valid_  = false;
    bool  right_hand_valid_ = false;
};

}  // namespace ml2
