using UnityEngine;

public class AudioStreamPlayer : MonoBehaviour
{
    [Header("Audio Settings")]
    public int serverSampleRate = 24000;

    [Header("References")]
    public AudioSource audioSource;

    private AudioClip _streamClip;
    private int _writePos;
    private bool _isPlaying;
    private float _silenceTimer;

    private const int CLIP_SECONDS = 30;
    private const float SILENCE_THRESHOLD = 0.001f;
    private const float SILENCE_TIMEOUT = 0.5f;

    public bool IsPlaying => _isPlaying;

    void Start()
    {
        _streamClip = AudioClip.Create(
            "ServerAudioStream",
            serverSampleRate * CLIP_SECONDS,
            1,
            serverSampleRate,
            false
        );
        audioSource.clip = _streamClip;
        audioSource.loop = true;
    }

    public void EnqueueAudioChunk(byte[] pcmBytes)
    {
        if (pcmBytes == null || pcmBytes.Length < 2) return;

        int sampleCount = pcmBytes.Length / 2;
        float[] samples = new float[sampleCount];

        for (int i = 0; i < sampleCount; i++)
        {
            short val = (short)(pcmBytes[i * 2] | (pcmBytes[i * 2 + 1] << 8));
            samples[i] = val / 32768f;
        }

        int clipLength = serverSampleRate * CLIP_SECONDS;
        _streamClip.SetData(samples, _writePos % clipLength);
        _writePos += sampleCount;

        _silenceTimer = 0f;

        if (!_isPlaying)
        {
            audioSource.Play();
            _isPlaying = true;
        }
    }

    public float GetCurrentVolume()
    {
        if (!_isPlaying || !audioSource.isPlaying) return 0f;

        float[] outputData = new float[256];
        audioSource.GetOutputData(outputData, 0);

        float sum = 0f;
        for (int i = 0; i < outputData.Length; i++)
            sum += outputData[i] * outputData[i];

        return Mathf.Sqrt(sum / outputData.Length);
    }

    public void StopPlayback()
    {
        audioSource.Stop();
        _isPlaying = false;
        _writePos = 0;
    }

    void Update()
    {
        if (!_isPlaying) return;

        // Detect when audio stream has ended (silence after playback)
        float vol = GetCurrentVolume();
        if (vol < SILENCE_THRESHOLD)
        {
            _silenceTimer += Time.deltaTime;
            if (_silenceTimer > SILENCE_TIMEOUT)
            {
                StopPlayback();
            }
        }
        else
        {
            _silenceTimer = 0f;
        }
    }
}
