#include "audio_capture.h"

#include <android/log.h>

#include <cstring>

#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>
#include <SLES/OpenSLES_AndroidConfiguration.h>

#include "vrs_writer.h"

#define LOG_TAG "ML2Recorder"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml2 {

static void opensl_buffer_callback(SLAndroidSimpleBufferQueueItf, void* ctx) {
    static_cast<AudioCapture*>(ctx)->on_buffer(0);
}

AudioCapture::AudioCapture() = default;

AudioCapture::~AudioCapture() { stop(); }

bool AudioCapture::init(const AudioCaptureConfig& config) {
    config_ = config;

    if (!config_.vrs_writer) {
        LOGE("AudioCapture: vrs_writer is null");
        return false;
    }

    buffer_size_samples_ =
        (config_.sample_rate * config_.channels * kBufferDurationMs) / 1000;
    for (int i = 0; i < kNumBuffers; ++i)
        buffers_[i].resize(buffer_size_samples_);

    // Write stream config record so the reader knows the audio format.
    VrsWriter::AudioConfig ac;
    ac.sample_rate = config_.sample_rate;
    ac.channels    = config_.channels;
    config_.vrs_writer->write_audio_config(ac);

    // Create OpenSL ES engine.
    SLObjectItf engine_obj = nullptr;
    SLresult res = slCreateEngine(&engine_obj, 0, nullptr, 0, nullptr, nullptr);
    if (res != SL_RESULT_SUCCESS) {
        LOGE("slCreateEngine failed: %d", static_cast<int>(res));
        return false;
    }
    sl_engine_obj_ = static_cast<void*>(const_cast<SLObjectItf_**>(engine_obj));

    res = (*engine_obj)->Realize(engine_obj, SL_BOOLEAN_FALSE);
    if (res != SL_RESULT_SUCCESS) {
        LOGE("Engine Realize failed: %d", static_cast<int>(res));
        return false;
    }

    SLEngineItf engine_itf = nullptr;
    res = (*engine_obj)->GetInterface(engine_obj, SL_IID_ENGINE, &engine_itf);
    if (res != SL_RESULT_SUCCESS) {
        LOGE("GetInterface ENGINE failed: %d", static_cast<int>(res));
        return false;
    }
    sl_engine_ = static_cast<void*>(const_cast<SLEngineItf_**>(engine_itf));

    LOGI("AudioCapture init: %d ch, %d Hz", config_.channels, config_.sample_rate);
    return true;
}

bool AudioCapture::start() {
    if (running_.load()) return true;

    SLEngineItf engine = static_cast<SLEngineItf>(sl_engine_);
    SLresult res;

    SLDataLocator_IODevice loc_dev = {
        SL_DATALOCATOR_IODEVICE,
        SL_IODEVICE_AUDIOINPUT,
        SL_DEFAULTDEVICEID_AUDIOINPUT,
        nullptr
    };
    SLDataSource audio_src = {&loc_dev, nullptr};

    SLDataLocator_AndroidSimpleBufferQueue loc_bq = {
        SL_DATALOCATOR_ANDROIDSIMPLEBUFFERQUEUE,
        static_cast<SLuint32>(kNumBuffers)
    };

    SLuint32 channel_mask = 0;
    switch (config_.channels) {
        case 1: channel_mask = SL_SPEAKER_FRONT_CENTER; break;
        case 2: channel_mask = SL_SPEAKER_FRONT_LEFT | SL_SPEAKER_FRONT_RIGHT; break;
        case 4: channel_mask = SL_SPEAKER_FRONT_LEFT | SL_SPEAKER_FRONT_RIGHT |
                               SL_SPEAKER_BACK_LEFT  | SL_SPEAKER_BACK_RIGHT;  break;
        default: channel_mask = SL_SPEAKER_FRONT_LEFT | SL_SPEAKER_FRONT_RIGHT; break;
    }

    SLDataFormat_PCM format_pcm = {
        SL_DATAFORMAT_PCM,
        static_cast<SLuint32>(config_.channels),
        static_cast<SLuint32>(config_.sample_rate * 1000),  // milliHz
        SL_PCMSAMPLEFORMAT_FIXED_16,
        SL_PCMSAMPLEFORMAT_FIXED_16,
        channel_mask,
        SL_BYTEORDER_LITTLEENDIAN
    };
    SLDataSink audio_sink = {&loc_bq, &format_pcm};

    const SLInterfaceID ids[] = {SL_IID_ANDROIDSIMPLEBUFFERQUEUE,
                                  SL_IID_ANDROIDCONFIGURATION};
    const SLboolean req[] = {SL_BOOLEAN_TRUE, SL_BOOLEAN_FALSE};

    SLObjectItf recorder_obj = nullptr;
    res = (*engine)->CreateAudioRecorder(engine, &recorder_obj,
                                          &audio_src, &audio_sink, 2, ids, req);
    if (res != SL_RESULT_SUCCESS) {
        // Fallback to mono.
        LOGE("CreateAudioRecorder failed (%d) — trying mono", static_cast<int>(res));
        config_.channels = 1;
        format_pcm.numChannels = 1;
        format_pcm.channelMask = SL_SPEAKER_FRONT_CENTER;
        buffer_size_samples_ =
            (config_.sample_rate * 1 * kBufferDurationMs) / 1000;
        for (int i = 0; i < kNumBuffers; ++i) buffers_[i].resize(buffer_size_samples_);

        res = (*engine)->CreateAudioRecorder(engine, &recorder_obj,
                                              &audio_src, &audio_sink, 2, ids, req);
        if (res != SL_RESULT_SUCCESS) {
            LOGE("CreateAudioRecorder mono fallback failed: %d", static_cast<int>(res));
            return false;
        }
        LOGI("Audio: fell back to mono");
    }
    sl_recorder_obj_ = static_cast<void*>(const_cast<SLObjectItf_**>(recorder_obj));

    res = (*recorder_obj)->Realize(recorder_obj, SL_BOOLEAN_FALSE);
    if (res != SL_RESULT_SUCCESS) {
        LOGE("Recorder Realize failed: %d", static_cast<int>(res));
        return false;
    }

    res = (*recorder_obj)->GetInterface(recorder_obj, SL_IID_RECORD, &sl_recorder_);
    if (res != SL_RESULT_SUCCESS) {
        LOGE("GetInterface RECORD failed: %d", static_cast<int>(res));
        return false;
    }

    res = (*recorder_obj)->GetInterface(recorder_obj,
                                         SL_IID_ANDROIDSIMPLEBUFFERQUEUE,
                                         &sl_buffer_queue_);
    if (res != SL_RESULT_SUCCESS) {
        LOGE("GetInterface BufferQueue failed: %d", static_cast<int>(res));
        return false;
    }

    SLAndroidSimpleBufferQueueItf bq =
        static_cast<SLAndroidSimpleBufferQueueItf>(sl_buffer_queue_);
    (*bq)->RegisterCallback(bq, opensl_buffer_callback, this);

    for (int i = 0; i < kNumBuffers; ++i) {
        (*bq)->Enqueue(bq, buffers_[i].data(),
                       buffers_[i].size() * sizeof(int16_t));
    }

    SLRecordItf recorder = static_cast<SLRecordItf>(sl_recorder_);
    running_.store(true);
    res = (*recorder)->SetRecordState(recorder, SL_RECORDSTATE_RECORDING);
    if (res != SL_RESULT_SUCCESS) {
        LOGE("SetRecordState RECORDING failed: %d", static_cast<int>(res));
        running_.store(false);
        return false;
    }

    LOGI("AudioCapture started (%d ch)", config_.channels);
    return true;
}

void AudioCapture::on_buffer(uint32_t /*buffer_index*/) {
    if (!running_.load()) return;

    static int current_buf = 0;
    int filled = current_buf;
    current_buf = (current_buf + 1) % kNumBuffers;

    const size_t num_samples = buffers_[filled].size();
    const size_t byte_size   = num_samples * sizeof(int16_t);

    config_.vrs_writer->write_audio(
        now_ns(),
        buffers_[filled].data(),
        static_cast<uint32_t>(num_samples),
        static_cast<uint32_t>(config_.channels));
    bytes_written_.fetch_add(byte_size);

    SLAndroidSimpleBufferQueueItf bq =
        static_cast<SLAndroidSimpleBufferQueueItf>(sl_buffer_queue_);
    (*bq)->Enqueue(bq, buffers_[filled].data(), byte_size);
}

void AudioCapture::stop() {
    if (!running_.load() && !sl_engine_obj_) return;
    running_.store(false);

    if (sl_recorder_) {
        SLRecordItf rec = static_cast<SLRecordItf>(sl_recorder_);
        (*rec)->SetRecordState(rec, SL_RECORDSTATE_STOPPED);
    }
    if (sl_recorder_obj_) {
        SLObjectItf obj = static_cast<SLObjectItf>(sl_recorder_obj_);
        (*obj)->Destroy(obj);
        sl_recorder_obj_ = nullptr;
        sl_recorder_     = nullptr;
        sl_buffer_queue_ = nullptr;
    }
    if (sl_engine_obj_) {
        SLObjectItf obj = static_cast<SLObjectItf>(sl_engine_obj_);
        (*obj)->Destroy(obj);
        sl_engine_obj_ = nullptr;
        sl_engine_     = nullptr;
    }

    LOGI("AudioCapture stopped. Bytes written: %zu", bytes_written_.load());
}

}  // namespace ml2
