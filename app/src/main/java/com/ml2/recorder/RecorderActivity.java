package com.ml2.recorder;

import android.app.NativeActivity;
import android.media.AudioManager;
import android.media.ToneGenerator;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.Toast;

/**
 * Thin Java wrapper around NativeActivity that provides user-facing feedback:
 * - Toast messages for recording state (visible in the ML2 notification area)
 * - Audio tones for start/stop (audible without looking at a screen)
 *
 * Native code calls these via JNI: showToast(), playStartTone(), playStopTone().
 */
public class RecorderActivity extends NativeActivity {

    private Handler mainHandler;
    private ToneGenerator toneGenerator;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mainHandler = new Handler(Looper.getMainLooper());
        try {
            toneGenerator = new ToneGenerator(AudioManager.STREAM_NOTIFICATION, 80);
        } catch (Exception e) {
            // ToneGenerator may fail on some devices; non-fatal.
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (toneGenerator != null) {
            toneGenerator.release();
            toneGenerator = null;
        }
    }

    /**
     * Show a Toast message. Called from native code via JNI.
     */
    public void showToast(final String message) {
        mainHandler.post(() -> Toast.makeText(RecorderActivity.this, message, Toast.LENGTH_LONG).show());
    }

    /**
     * Play a short ascending tone to indicate recording start.
     */
    public void playStartTone() {
        if (toneGenerator != null) {
            // Two ascending beeps
            toneGenerator.startTone(ToneGenerator.TONE_PROP_BEEP, 200);
            mainHandler.postDelayed(() -> {
                if (toneGenerator != null) {
                    toneGenerator.startTone(ToneGenerator.TONE_PROP_BEEP2, 200);
                }
            }, 250);
        }
    }

    /**
     * Play a short descending tone to indicate recording stop.
     */
    public void playStopTone() {
        if (toneGenerator != null) {
            // Three short beeps
            toneGenerator.startTone(ToneGenerator.TONE_PROP_ACK, 150);
            mainHandler.postDelayed(() -> {
                if (toneGenerator != null) {
                    toneGenerator.startTone(ToneGenerator.TONE_PROP_ACK, 150);
                }
            }, 200);
            mainHandler.postDelayed(() -> {
                if (toneGenerator != null) {
                    toneGenerator.startTone(ToneGenerator.TONE_PROP_ACK, 150);
                }
            }, 400);
        }
    }
}
