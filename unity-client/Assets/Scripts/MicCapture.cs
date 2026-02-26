using UnityEngine;

public class MicCapture : MonoBehaviour
{
    [Header("Audio Settings")]
    public int sampleRate = 16000;
    public int chunkSizeMs = 100;

    [Header("References")]
    public WebSocketClient webSocket;

    private AudioClip _micClip;
    private int _lastSamplePos;
    private float _sendTimer;
    private string _micDevice;
    private bool _recording;

    public bool IsRecording => _recording;

    void Start()
    {
        InitMicrophone();
    }

    public void InitMicrophone()
    {
        if (Microphone.devices.Length == 0)
        {
            Debug.LogError("[Mic] No microphone found!");
            return;
        }

        _micDevice = Microphone.devices[0];
        Debug.Log($"[Mic] Using device: {_micDevice}");
        StartRecording();
    }

    public void StartRecording()
    {
        if (_micDevice == null) return;

        _micClip = Microphone.Start(_micDevice, true, 1, sampleRate);
        _lastSamplePos = 0;
        _recording = true;
        Debug.Log("[Mic] Recording started.");
    }

    public void StopRecording()
    {
        if (_micDevice != null && Microphone.IsRecording(_micDevice))
        {
            Microphone.End(_micDevice);
            _recording = false;
            Debug.Log("[Mic] Recording stopped.");
        }
    }

    void Update()
    {
        if (!_recording || webSocket == null || !webSocket.IsConnected) return;
        if (_micClip == null || _micDevice == null) return;

        _sendTimer += Time.deltaTime;
        if (_sendTimer < chunkSizeMs / 1000f) return;
        _sendTimer = 0f;

        int pos = Microphone.GetPosition(_micDevice);
        if (pos == _lastSamplePos) return;

        int sampleCount;
        if (pos > _lastSamplePos)
            sampleCount = pos - _lastSamplePos;
        else
            sampleCount = (sampleRate - _lastSamplePos) + pos;

        if (sampleCount <= 0) return;

        float[] samples = new float[sampleCount];
        _micClip.GetData(samples, _lastSamplePos);
        _lastSamplePos = pos;

        byte[] pcm = FloatToPCM16(samples);
        webSocket.SendAudio(pcm);
    }

    private static byte[] FloatToPCM16(float[] samples)
    {
        byte[] pcm = new byte[samples.Length * 2];
        for (int i = 0; i < samples.Length; i++)
        {
            short val = (short)(Mathf.Clamp(samples[i], -1f, 1f) * 32767);
            pcm[i * 2] = (byte)(val & 0xFF);
            pcm[i * 2 + 1] = (byte)((val >> 8) & 0xFF);
        }
        return pcm;
    }

    void OnDestroy()
    {
        StopRecording();
    }
}
