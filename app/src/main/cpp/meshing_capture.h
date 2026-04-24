#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

#include "timestamp.h"

typedef uint64_t MLHandle;

namespace ml2 {

class VrsWriter;

struct MeshingCaptureConfig {
    VrsWriter* vrs_writer = nullptr;
    float poll_interval_s  = 1.0f;
    float bounds_extents_m = 5.0f;
    bool  compute_normals  = true;
    bool  compute_confidence = false;
    int   lod              = 1;  // 0=min, 1=medium, 2=max
};

/// Captures spatial mesh snapshots from the ML2 meshing subsystem.
/// Each snapshot writes one VRS mesh record: raw vertex, index, and optional
/// normal arrays (no PLY encoding — the VRS record IS the container).
class MeshingCapture {
public:
    MeshingCapture();
    ~MeshingCapture();

    MeshingCapture(const MeshingCapture&) = delete;
    MeshingCapture& operator=(const MeshingCapture&) = delete;

    bool init(const MeshingCaptureConfig& config);
    bool start();
    void stop();

    bool     is_running()       const { return running_.load(); }
    uint32_t snapshots_written() const { return snapshots_written_.load(); }
    uint64_t total_vertices()    const { return total_vertices_.load(); }

private:
    void capture_loop();

    MeshingCaptureConfig  config_;
    std::thread           capture_thread_;
    std::atomic<bool>     running_{false};
    std::atomic<uint32_t> snapshot_index_{0};
    std::atomic<uint32_t> snapshots_written_{0};
    std::atomic<uint64_t> total_vertices_{0};

    MLHandle meshing_client_ = 0xFFFFFFFFFFFFFFFF;
};

}  // namespace ml2
