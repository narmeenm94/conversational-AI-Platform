using UnityEngine;

public class ConversationManager : MonoBehaviour
{
    [Header("Server Connection")]
    [Tooltip("Use 'localhost' for desktop testing, your PC's LAN IP for Quest 3")]
    public string serverAddress = "localhost";
    public int serverPort = 8765;

    [Header("References")]
    public WebSocketClient webSocket;
    public MicCapture micCapture;
    public AudioStreamPlayer audioPlayer;
    public AvatarController avatarController;

    [Header("Status")]
    [SerializeField] private bool _isConnected;
    [SerializeField] private bool _avatarSpeaking;

    void Start()
    {
        RequestMicPermission();

        webSocket.serverUrl = $"ws://{serverAddress}:{serverPort}";

        webSocket.OnConnected += HandleConnected;
        webSocket.OnDisconnected += HandleDisconnected;
        webSocket.OnAudioReceived += HandleAudioReceived;

        webSocket.Connect();
    }

    private void RequestMicPermission()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(
                UnityEngine.Android.Permission.Microphone))
        {
            UnityEngine.Android.Permission.RequestUserPermission(
                UnityEngine.Android.Permission.Microphone);
        }
#endif
    }

    private void HandleConnected()
    {
        _isConnected = true;
        Debug.Log("[Conversation] Connected to AI server. Ready to talk.");
        avatarController.SetSpeaking(false);
    }

    private void HandleDisconnected()
    {
        _isConnected = false;
        Debug.Log("[Conversation] Disconnected from AI server.");
        avatarController.SetSpeaking(false);
        audioPlayer.StopPlayback();
    }

    private void HandleAudioReceived(byte[] audioData)
    {
        if (!_avatarSpeaking)
        {
            _avatarSpeaking = true;
            avatarController.SetSpeaking(true);
        }
        audioPlayer.EnqueueAudioChunk(audioData);
    }

    void Update()
    {
        if (_avatarSpeaking && !audioPlayer.IsPlaying)
        {
            _avatarSpeaking = false;
            avatarController.SetSpeaking(false);
        }
    }

    void OnDestroy()
    {
        if (webSocket != null)
        {
            webSocket.OnConnected -= HandleConnected;
            webSocket.OnDisconnected -= HandleDisconnected;
            webSocket.OnAudioReceived -= HandleAudioReceived;
        }
    }
}
