using System.Collections.Concurrent;
using System.Threading;
using UnityEngine;

/// <summary>
/// Low-latency PCM ring player. The Unity audio thread pulls queued samples
/// through a streaming AudioClip, which also gives uLipSync a real AudioSource
/// signal to analyse in OnAudioFilterRead.
/// </summary>
public class AudioStreamPlayer : MonoBehaviour
{
    [Header("Audio Settings")]
    public int serverSampleRate = 24000;
    [Range(20, 500)] public int prebufferMs = 40;
    [Range(2, 30)] public int deClickFadeMs = 8;
    [Range(0.05f, 1f)] public float speakingTailSeconds = 0.25f;

    [Header("References")]
    public AudioSource audioSource;

    private readonly ConcurrentQueue<float[]> _chunks = new ConcurrentQueue<float[]>();
    private AudioClip _streamClip;
    private float[] _activeChunk;
    private int _activeOffset;
    private int _queuedSamples;
    private int _flushRequested;
    private bool _primed;
    private int _fadeInSample;
    private float _currentRms;
    private float _lastChunkTime = -100f;

    public bool IsPlaying =>
        Volatile.Read(ref _queuedSamples) > 0 ||
        Time.unscaledTime - _lastChunkTime < speakingTailSeconds;

    void Awake()
    {
        if (audioSource == null)
            audioSource = GetComponent<AudioSource>();
        if (audioSource == null)
            audioSource = gameObject.AddComponent<AudioSource>();

        audioSource.playOnAwake = false;
        audioSource.loop = true;
        audioSource.spatialBlend = 1f;

        _streamClip = AudioClip.Create(
            "ConversationalAIStream",
            serverSampleRate * 2,
            1,
            serverSampleRate,
            true,
            OnAudioRead
        );
        audioSource.clip = _streamClip;
        audioSource.Play();
    }

    public void EnqueueAudioChunk(byte[] pcmBytes)
    {
        if (pcmBytes == null || pcmBytes.Length < 2) return;

        int count = pcmBytes.Length / 2;
        var samples = new float[count];
        for (int i = 0; i < count; i++)
        {
            short value = (short)(pcmBytes[i * 2] | (pcmBytes[i * 2 + 1] << 8));
            samples[i] = value / 32768f;
        }

        _chunks.Enqueue(samples);
        Interlocked.Add(ref _queuedSamples, count);
        _lastChunkTime = Time.unscaledTime;
    }

    public float GetCurrentVolume() => Volatile.Read(ref _currentRms);

    public void StopPlayback()
    {
        Interlocked.Exchange(ref _flushRequested, 1);
        while (_chunks.TryDequeue(out _)) { }
        Interlocked.Exchange(ref _queuedSamples, 0);
        _lastChunkTime = -100f;
    }

    private void OnAudioRead(float[] output)
    {
        if (Interlocked.Exchange(ref _flushRequested, 0) != 0)
        {
            _activeChunk = null;
            _activeOffset = 0;
            _primed = false;
            _fadeInSample = 0;
        }

        int prebufferSamples = serverSampleRate * prebufferMs / 1000;
        if (!_primed && Volatile.Read(ref _queuedSamples) >= prebufferSamples)
        {
            _primed = true;
            _fadeInSample = 0;
        }

        double sum = 0;
        for (int i = 0; i < output.Length; i++)
        {
            float sample = 0f;
            if (_primed && TryReadSample(out sample))
            {
                int fadeSamples = Mathf.Max(1, serverSampleRate * deClickFadeMs / 1000);
                if (_fadeInSample < fadeSamples)
                {
                    sample *= _fadeInSample / (float)fadeSamples;
                    _fadeInSample++;
                }
                sum += sample * sample;
            }
            else if (_primed)
                _primed = false;
            output[i] = sample;
        }
        Volatile.Write(ref _currentRms, Mathf.Sqrt((float)(sum / output.Length)));
    }

    private bool TryReadSample(out float sample)
    {
        while (_activeChunk == null || _activeOffset >= _activeChunk.Length)
        {
            if (!_chunks.TryDequeue(out _activeChunk))
            {
                sample = 0f;
                return false;
            }
            _activeOffset = 0;
        }

        sample = _activeChunk[_activeOffset++];
        Interlocked.Decrement(ref _queuedSamples);
        return true;
    }
}
