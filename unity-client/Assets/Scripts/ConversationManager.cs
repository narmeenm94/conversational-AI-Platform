using System;
using UnityEngine;

public class ConversationManager : MonoBehaviour
{
    [Serializable]
    private class ServerEvent
    {
        public int v;
        public string label;
        public string type;
        public string text;
        public string emotion;
        public string state;
        public float intensity;
        public float blend_seconds;
        public ServerEventData data;
    }

    [Serializable]
    private class ServerEventData
    {
        public string text;
        public string error;
        public bool final;
    }

    [Header("Server Connection")]
    [Tooltip("Use localhost in the Editor, or the PC LAN address on Quest.")]
    public string serverAddress = "localhost";
    public int serverPort = 8765;

    [Header("Conversation")]
    [Tooltip("When enabled, the microphone stays live while the avatar speaks so the user can interrupt. With laptop speakers this can also hear the avatar; disable it for echo-safe half duplex.")]
    public bool enableBargeIn = false;
    public bool logTranscripts = true;
    [Tooltip("Logs raw model tokens that may later be discarded. Keep disabled for an accurate spoken transcript.")]
    public bool logRawModelText = false;

    [Header("References")]
    public WebSocketClient webSocket;
    public MicCapture micCapture;
    public AudioStreamPlayer audioPlayer;
    public AvatarController avatarController;
    public CharacterPlatformClient characterPlatform;

    [Header("Status")]
    [SerializeField] private bool _isConnected;
    [SerializeField] private bool _avatarSpeaking;
    [SerializeField] private bool _userSpeaking;
    private bool _appliedBargeIn;
    [SerializeField] private bool _gazeEngaged = true;

    public bool IsConnected => _isConnected;
    public bool AvatarSpeaking => _avatarSpeaking;
    public bool UserSpeaking => _userSpeaking;
    public bool GazeEngaged => _gazeEngaged;

    void Start()
    {
        RequestMicPermission();
        if (characterPlatform == null)
            characterPlatform = GetComponent<CharacterPlatformClient>();
        if (characterPlatform == null)
            characterPlatform = gameObject.AddComponent<CharacterPlatformClient>();
        characterPlatform.serverAddress = serverAddress;
        characterPlatform.OnCharacterChanged += HandleCharacterChanged;
        if (characterPlatform.ActiveCharacter != null)
            HandleCharacterChanged(characterPlatform.ActiveCharacter);
        ApplyBargeInMode(force: true);
        micCapture.SetConversationInputEnabled(_gazeEngaged);
        avatarController.SetListening(_gazeEngaged);
        webSocket.serverUrl = $"ws://{serverAddress}:{serverPort}";
        webSocket.OnConnected += HandleConnected;
        webSocket.OnDisconnected += HandleDisconnected;
        webSocket.OnAudioReceived += HandleAudioReceived;
        webSocket.OnTextReceived += HandleServerEvent;
        webSocket.Connect();
    }

    private void RequestMicPermission()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(
                UnityEngine.Android.Permission.Microphone))
            UnityEngine.Android.Permission.RequestUserPermission(
                UnityEngine.Android.Permission.Microphone);
#endif
    }

    private void HandleConnected()
    {
        _isConnected = true;
        micCapture.SetAssistantTurnActive(false);
        SetAvatarSpeaking(false);
        Debug.Log("[Conversation] Connected and ready.");
    }

    private void HandleDisconnected()
    {
        _isConnected = false;
        _userSpeaking = false;
        micCapture.SetAssistantTurnActive(false);
        audioPlayer.StopPlayback();
        SetAvatarSpeaking(false);
        avatarController.SetListening(false);
        Debug.Log("[Conversation] Disconnected.");
    }

    private void HandleAudioReceived(byte[] audioData)
    {
        audioPlayer.EnqueueAudioChunk(audioData);
        SetAvatarSpeaking(true);
    }

    private void HandleServerEvent(string json)
    {
        ServerEvent message;
        try { message = JsonUtility.FromJson<ServerEvent>(json); }
        catch (Exception exception)
        {
            Debug.LogWarning($"[Conversation] Invalid server event: {exception.Message}");
            return;
        }
        if (message == null) return;

        // Pipecat emits RTVI state/transcript messages in addition to this
        // project's compact v1 protocol. Supporting both keeps the Unity
        // client compatible across Pipecat transport versions.
        if (message.label == "rtvi-ai")
        {
            HandleRtviEvent(message);
            return;
        }
        if (message.v != 1) return;

        switch (message.type)
        {
            case "user_speech_started":
                BeginUserSpeech();
                break;
            case "user_speech_stopped":
                EndUserSpeech();
                break;
            case "user_transcript":
                if (logTranscripts) Debug.Log($"[User] {message.text}");
                break;
            case "assistant_speech_started":
                SetAvatarSpeaking(true);
                break;
            case "assistant_speech_stopped":
                // Audio can still be queued. Update() ends the visual state
                // only after the actual playback tail has drained.
                break;
            case "assistant_interrupted":
                micCapture.SetAssistantTurnActive(false);
                audioPlayer.StopPlayback();
                SetAvatarSpeaking(false);
                break;
            case "assistant_expression":
                avatarController.SetEmotion(
                    message.emotion,
                    message.intensity > 0f ? message.intensity : 0.6f
                );
                break;
            case "assistant_animation":
                avatarController.SetSemanticState(
                    message.state,
                    message.blend_seconds > 0f ? message.blend_seconds : -1f
                );
                break;
            case "assistant_spoken_text":
                if (logTranscripts) Debug.Log($"[Assistant] {message.text}");
                break;
            case "assistant_response_started":
                micCapture.SetAssistantTurnActive(true);
                break;
            case "assistant_response_finished":
                // Pipecat may finish the LLM stream before queued TTS audio is
                // rendered. RTVI bot-stopped-speaking closes the guarded turn
                // only after the final audio segment is actually delivered.
                if (!_avatarSpeaking && !audioPlayer.IsPlaying)
                    avatarController.SetSemanticState("idle");
                break;
        }
    }

    private void HandleRtviEvent(ServerEvent message)
    {
        switch (message.type)
        {
            case "bot-llm-started":
                micCapture.SetAssistantTurnActive(true);
                break;
            case "user-started-speaking":
                BeginUserSpeech();
                break;
            case "user-stopped-speaking":
                EndUserSpeech();
                break;
            case "user-transcription":
                if (logTranscripts && message.data != null && message.data.final)
                    Debug.Log($"[User] {message.data.text}");
                break;
            case "bot-started-speaking":
                SetAvatarSpeaking(true);
                break;
            case "bot-stopped-speaking":
                // Keep the visual state until AudioStreamPlayer drains its tail.
                micCapture.SetAssistantTurnActive(false);
                break;
            case "bot-transcription":
                if (logRawModelText && message.data != null)
                    Debug.Log($"[Assistant raw] {message.data.text}");
                break;
            case "error":
                string error = message.data != null ? message.data.error : message.text;
                Debug.LogError($"[Conversation] Server pipeline error: {error}");
                if (!_avatarSpeaking && !_userSpeaking)
                    avatarController.SetSemanticState("idle");
                break;
        }
    }

    private void BeginUserSpeech()
    {
        _userSpeaking = true;
        avatarController.SetListening(true);
        if (enableBargeIn && _avatarSpeaking)
        {
            audioPlayer.StopPlayback();
            SetAvatarSpeaking(false);
        }
    }

    private void EndUserSpeech()
    {
        _userSpeaking = false;
        avatarController.SetListening(false);
    }

    private void SetAvatarSpeaking(bool speaking)
    {
        _avatarSpeaking = speaking;
        avatarController.SetSpeaking(speaking);
        micCapture.SetAssistantSpeaking(speaking);
        // The native Waiting motion is the avatar's attentive beat while she
        // waits for the user. Thinking and speaking intentionally share Idle
        // so those transitions never restart or snap the Avaturn body clip.
        if (!speaking && !_userSpeaking)
            avatarController.SetListening(_gazeEngaged);
    }

    private void HandleCharacterChanged(
        CharacterPlatformClient.CharacterDefinition character
    )
    {
        if (character != null)
            avatarController.ConfigureAnimationStates(character.animations);
    }

    /// <summary>Gameplay/navigation can drive the locomotion state directly.</summary>
    public void SetAvatarWalking(bool walking) => avatarController.SetWalking(walking);

    /// <summary>
    /// In the Quest scene, looking at the avatar opens the continuous mic gate.
    /// The microphone keeps recording but sends silence while gaze is elsewhere,
    /// so the server VAD sees a clean end-of-turn without push-to-talk controls.
    /// </summary>
    public void SetGazeEngaged(bool engaged)
    {
        if (_gazeEngaged == engaged) return;
        _gazeEngaged = engaged;
        micCapture.SetConversationInputEnabled(engaged);
        if (!_avatarSpeaking)
            avatarController.SetListening(engaged && !_userSpeaking);
    }

    /// <summary>Can be called by a Unity UI Toggle at runtime.</summary>
    public void SetBargeInEnabled(bool enabled)
    {
        enableBargeIn = enabled;
        ApplyBargeInMode(force: true);
    }

    public void ToggleBargeIn() => SetBargeInEnabled(!enableBargeIn);

    private void ApplyBargeInMode(bool force = false)
    {
        if (!force && _appliedBargeIn == enableBargeIn) return;
        _appliedBargeIn = enableBargeIn;
        micCapture.muteWhileAssistantSpeaks = !enableBargeIn;
        micCapture.SetAssistantSpeaking(_avatarSpeaking);
        Debug.Log(enableBargeIn
            ? "[Conversation] Barge-in ON: microphone remains live during avatar speech."
            : "[Conversation] Barge-in OFF: speaker-safe half duplex with echo tail.");
    }

    void Update()
    {
        ApplyBargeInMode();
        if (_avatarSpeaking && !audioPlayer.IsPlaying)
            SetAvatarSpeaking(false);
    }

    void OnDestroy()
    {
        if (webSocket == null) return;
        webSocket.OnConnected -= HandleConnected;
        webSocket.OnDisconnected -= HandleDisconnected;
        webSocket.OnAudioReceived -= HandleAudioReceived;
        webSocket.OnTextReceived -= HandleServerEvent;
        if (characterPlatform != null)
            characterPlatform.OnCharacterChanged -= HandleCharacterChanged;
    }
}
