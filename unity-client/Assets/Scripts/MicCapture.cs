using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;

/// <summary>Continuously sends fixed-size mono PCM16 microphone frames.</summary>
public class MicCapture : MonoBehaviour
{
    [Header("Audio Settings")]
    public int sampleRate = 16000;
    [Range(20, 200)] public int chunkSizeMs = 100;
    [Range(0.25f, 8f)] public float inputGain = 1f;
    [Tooltip("Optional exact or partial Unity microphone device name. Empty uses the first available input.")]
    public string preferredDeviceName = "";

    [Header("Input Conditioning")]
    public bool automaticGainControl = true;
    [Range(0.02f, 0.2f)] public float targetRms = 0.08f;
    [Range(1f, 12f)] public float maxAutoGain = 6f;
    [Range(0.001f, 0.03f)] public float autoGainNoiseFloor = 0.006f;
    [Tooltip("Prints microphone peak/RMS and active gain every two seconds.")]
    public bool logInputLevels = true;

    [Header("Turn Taking")]
    [Tooltip("Prevents the avatar's speaker audio from feeding back into VAD. Disable for barge-in with headphones/AEC.")]
    public bool muteWhileAssistantSpeaks = true;
    [Tooltip("Keeps the mic muted briefly after playback so laptop-speaker echo cannot start a new turn.")]
    [Range(0f, 2f)] public float speakerEchoTailSeconds = 0.65f;
    [Tooltip("Post-gain RMS required to open barge-in while avatar audio or a response is active.")]
    [Range(0.01f, 0.2f)] public float bargeInRmsThreshold = 0.035f;
    [Tooltip("Consecutive microphone chunks above threshold required before interruption audio is released.")]
    [Range(1, 5)] public int bargeInConfirmChunks = 2;
    [Tooltip("Recent chunks preserved so the beginning of a genuine interruption is not clipped.")]
    [Range(1, 5)] public int bargeInPreRollChunks = 3;

    [Header("References")]
    public WebSocketClient webSocket;

    private AudioClip _micClip;
    private string _micDevice;
    private int _readPosition;
    private int _sourceChunkFrames;
    private int _targetChunkSamples;
    private bool _recording;
    private bool _assistantSpeaking;
    private bool _assistantTurnActive;
    private float _muteUntil;
    private float _autoGain = 1f;
    private float _lastLevelLog;
    private float _lastRawRms;
    private bool _bargeInGateOpen;
    private int _bargeInLoudChunks;
    private readonly Queue<byte[]> _bargeInPreRoll = new Queue<byte[]>();
    private bool _conversationInputEnabled = true;

    public bool IsRecording => _recording;
    public bool ConversationInputEnabled => _conversationInputEnabled;

    public void SetConversationInputEnabled(bool enabled)
    {
        if (_conversationInputEnabled == enabled) return;
        _conversationInputEnabled = enabled;
        ResetBargeInGate();
        Debug.Log(enabled
            ? "[Mic] Gaze engaged: conversational input enabled."
            : "[Mic] Gaze disengaged: sending silence until the avatar is addressed.");
    }

    IEnumerator Start()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        while (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(
                   UnityEngine.Android.Permission.Microphone))
            yield return null;
#else
        yield return null;
#endif
        InitMicrophone();
    }

    public void SetAssistantSpeaking(bool speaking)
    {
        _assistantSpeaking = speaking;
        if (speaking)
            _muteUntil = float.PositiveInfinity;
        else
            _muteUntil = Time.unscaledTime + speakerEchoTailSeconds;
        if (!speaking && !_assistantTurnActive)
            ResetBargeInGate();
    }

    public void SetAssistantTurnActive(bool active)
    {
        _assistantTurnActive = active;
        if (!active && !_assistantSpeaking)
            ResetBargeInGate();
    }

    public void InitMicrophone()
    {
        if (_recording) return;
        if (Microphone.devices.Length == 0)
        {
            Debug.LogError("[Mic] No microphone found or permission not granted.");
            return;
        }

        Debug.Log($"[Mic] Available inputs: {string.Join(" | ", Microphone.devices)}");
        _micDevice = SelectMicrophoneDevice();
        _targetChunkSamples = Mathf.Max(1, sampleRate * chunkSizeMs / 1000);
        StartRecording();
    }

    private string SelectMicrophoneDevice()
    {
        if (!string.IsNullOrWhiteSpace(preferredDeviceName))
        {
            foreach (string device in Microphone.devices)
                if (device.IndexOf(preferredDeviceName, StringComparison.OrdinalIgnoreCase) >= 0)
                    return device;
            Debug.LogWarning($"[Mic] Preferred input '{preferredDeviceName}' was not found; using the first input.");
        }
        return Microphone.devices[0];
    }

    public void StartRecording()
    {
        if (string.IsNullOrEmpty(_micDevice)) return;
        _micClip = Microphone.Start(_micDevice, true, 10, sampleRate);
        _readPosition = 0;
        _recording = _micClip != null;
        if (_recording)
        {
            _sourceChunkFrames = Mathf.Max(1, _micClip.frequency * chunkSizeMs / 1000);
            Debug.Log(
                $"[Mic] Using '{_micDevice}': {_micClip.frequency} Hz, " +
                $"{_micClip.channels} channel(s) -> mono {sampleRate} Hz."
            );
        }
    }

    public void StopRecording()
    {
        if (!string.IsNullOrEmpty(_micDevice) && Microphone.IsRecording(_micDevice))
            Microphone.End(_micDevice);
        _recording = false;
    }

    void Update()
    {
        if (!_recording || _micClip == null || webSocket == null || !webSocket.IsConnected)
            return;

        int writePosition = Microphone.GetPosition(_micDevice);
        if (writePosition < 0) return;
        int available = writePosition >= _readPosition
            ? writePosition - _readPosition
            : (_micClip.samples - _readPosition) + writePosition;

        while (available >= _sourceChunkFrames)
        {
            var sourceSamples = new float[_sourceChunkFrames];
            ReadWrappedMono(sourceSamples, _readPosition);
            _readPosition = (_readPosition + _sourceChunkFrames) % _micClip.samples;
            available -= _sourceChunkFrames;

            float[] samples = Resample(sourceSamples, _targetChunkSamples);
            float conditionedGain = UpdateInputGain(samples);

            bool assistantAudioRisk = _assistantSpeaking || _assistantTurnActive;
            bool gazeMuted = !_conversationInputEnabled;
            bool muted = gazeMuted || (muteWhileAssistantSpeaks &&
                (assistantAudioRisk || Time.unscaledTime < _muteUntil));
            LogLevels(samples, muted, conditionedGain);
            byte[] pcm = FloatToPcm16(samples, muted ? 0f : conditionedGain);
            if (gazeMuted)
            {
                ResetBargeInGate();
                webSocket.SendAudio(pcm);
            }
            else if (!muteWhileAssistantSpeaks && assistantAudioRisk)
                SendBargeInGuarded(pcm, _lastRawRms * conditionedGain);
            else
            {
                ResetBargeInGate();
                webSocket.SendAudio(pcm);
            }
        }
    }

    private void ReadWrappedMono(float[] destination, int position)
    {
        int firstFrames = Mathf.Min(destination.Length, _micClip.samples - position);
        ReadMonoFrames(destination, 0, position, firstFrames);
        if (firstFrames < destination.Length)
            ReadMonoFrames(destination, firstFrames, 0, destination.Length - firstFrames);
    }

    private void ReadMonoFrames(float[] destination, int destinationOffset, int position, int frameCount)
    {
        if (frameCount <= 0) return;
        int channels = Mathf.Max(1, _micClip.channels);
        var interleaved = new float[frameCount * channels];
        _micClip.GetData(interleaved, position);
        for (int frame = 0; frame < frameCount; frame++)
        {
            float sum = 0f;
            for (int channel = 0; channel < channels; channel++)
                sum += interleaved[frame * channels + channel];
            destination[destinationOffset + frame] = sum / channels;
        }
    }

    private static float[] Resample(float[] source, int outputCount)
    {
        if (source.Length == outputCount) return source;
        var output = new float[outputCount];
        if (source.Length == 0 || outputCount == 0) return output;
        if (outputCount == 1)
        {
            output[0] = source[0];
            return output;
        }

        float scale = (source.Length - 1f) / (outputCount - 1f);
        for (int i = 0; i < outputCount; i++)
        {
            float position = i * scale;
            int left = Mathf.FloorToInt(position);
            int right = Mathf.Min(left + 1, source.Length - 1);
            output[i] = Mathf.Lerp(source[left], source[right], position - left);
        }
        return output;
    }

    private float UpdateInputGain(float[] samples)
    {
        double sum = 0;
        float peak = 0f;
        foreach (float sample in samples)
        {
            float absolute = Mathf.Abs(sample);
            peak = Mathf.Max(peak, absolute);
            sum += sample * sample;
        }
        float rms = samples.Length > 0 ? Mathf.Sqrt((float)(sum / samples.Length)) : 0f;
        _lastRawRms = rms;
        if (automaticGainControl && rms >= autoGainNoiseFloor)
        {
            float desired = Mathf.Clamp(targetRms / Mathf.Max(rms, 1e-5f), 0.5f, maxAutoGain);
            _autoGain = Mathf.Lerp(_autoGain, desired, 0.15f);
        }
        else if (!automaticGainControl)
        {
            _autoGain = 1f;
        }
        return inputGain * _autoGain;
    }

    private void SendBargeInGuarded(byte[] pcm, float conditionedRms)
    {
        if (_bargeInGateOpen)
        {
            webSocket.SendAudio(pcm);
            return;
        }

        _bargeInPreRoll.Enqueue(pcm);
        while (_bargeInPreRoll.Count > Mathf.Max(1, bargeInPreRollChunks))
            _bargeInPreRoll.Dequeue();

        if (conditionedRms >= bargeInRmsThreshold)
            _bargeInLoudChunks++;
        else
            _bargeInLoudChunks = 0;

        if (_bargeInLoudChunks >= Mathf.Max(1, bargeInConfirmChunks))
        {
            _bargeInGateOpen = true;
            while (_bargeInPreRoll.Count > 0)
                webSocket.SendAudio(_bargeInPreRoll.Dequeue());
            return;
        }

        // Preserve the realtime cadence while withholding likely speaker echo.
        webSocket.SendAudio(new byte[pcm.Length]);
    }

    private void ResetBargeInGate()
    {
        _bargeInGateOpen = false;
        _bargeInLoudChunks = 0;
        _bargeInPreRoll.Clear();
    }

    private void LogLevels(float[] samples, bool muted, float gain)
    {
        if (!logInputLevels || Time.unscaledTime - _lastLevelLog < 2f) return;
        _lastLevelLog = Time.unscaledTime;
        float peak = 0f;
        double sum = 0;
        foreach (float sample in samples)
        {
            peak = Mathf.Max(peak, Mathf.Abs(sample));
            sum += sample * sample;
        }
        float rms = samples.Length > 0 ? Mathf.Sqrt((float)(sum / samples.Length)) : 0f;
        Debug.Log($"[Mic] peak={peak:F3} rms={rms:F3} gain={gain:F2} muted={muted}");
    }

    private static byte[] FloatToPcm16(float[] samples, float gain)
    {
        var pcm = new byte[samples.Length * 2];
        for (int i = 0; i < samples.Length; i++)
        {
            short value = (short)(Mathf.Clamp(samples[i] * gain, -1f, 1f) * 32767f);
            pcm[i * 2] = (byte)(value & 0xff);
            pcm[i * 2 + 1] = (byte)((value >> 8) & 0xff);
        }
        return pcm;
    }

    void OnDestroy() => StopRecording();
}
