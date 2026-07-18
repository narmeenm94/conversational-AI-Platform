using System;
using System.Collections;
using System.Text;
using NativeWebSocket;
using UnityEngine;

public class WebSocketClient : MonoBehaviour
{
    [Tooltip("Desktop: ws://localhost:8765 | Quest 3: ws://YOUR_PC_IP:8765")]
    public string serverUrl = "ws://localhost:8765";

    [Header("Reconnection")]
    public bool autoReconnect = true;
    [Range(0.25f, 10f)] public float reconnectDelay = 0.75f;
    [Range(2f, 30f)] public float connectTimeoutSeconds = 8f;
    [Tooltip("Small application heartbeat that keeps laptop and Quest Wi-Fi connections alive.")]
    [Range(2f, 30f)] public float heartbeatSeconds = 5f;
    [Tooltip("Maximum reconnect attempts. Use 0 to keep trying while the app is running.")]
    public int maxReconnectAttempts = 0;

    public event Action OnConnected;
    public event Action OnDisconnected;
    public event Action<byte[]> OnAudioReceived;
    public event Action<string> OnTextReceived;

    private WebSocket _ws;
    private int _reconnectAttempts;
    private bool _intentionalClose;
    private bool _connectInProgress;
    private Coroutine _reconnectCoroutine;
    private float _nextHeartbeat;
    private int _connectionGeneration;

    public bool IsConnected { get; private set; }

    public async void Connect()
    {
        if (IsConnected || _connectInProgress) return;
        _intentionalClose = false;
        _reconnectAttempts = 0;

        await CreateAndConnect();
    }

    private async System.Threading.Tasks.Task CreateAndConnect()
    {
        if (_connectInProgress || IsConnected) return;
        _connectInProgress = true;
        int generation = ++_connectionGeneration;
        var socket = new WebSocket(serverUrl);
        _ws = socket;

        socket.OnOpen += () =>
        {
            if (generation != _connectionGeneration || socket != _ws) return;
            _connectInProgress = false;
            IsConnected = true;
            _reconnectAttempts = 0;
            _nextHeartbeat = Time.unscaledTime + heartbeatSeconds;
            Debug.Log($"[WS] Connected to {serverUrl}");
            OnConnected?.Invoke();
        };

        socket.OnMessage += (bytes) =>
        {
            if (generation != _connectionGeneration || socket != _ws) return;
            // NativeWebSocket exposes both text and binary payloads as byte[].
            // Control packets are compact JSON; everything else is raw PCM16.
            if (LooksLikeJson(bytes))
                OnTextReceived?.Invoke(Encoding.UTF8.GetString(bytes));
            else
                OnAudioReceived?.Invoke(bytes);
        };

        socket.OnError += (e) =>
        {
            if (generation != _connectionGeneration || socket != _ws) return;
            Debug.LogWarning($"[WS] Connection error: {e}");
        };

        socket.OnClose += (code) =>
        {
            if (generation != _connectionGeneration || socket != _ws) return;
            _connectInProgress = false;
            IsConnected = false;
            Debug.Log($"[WS] Disconnected (code: {code})");
            OnDisconnected?.Invoke();

            if (!_intentionalClose && autoReconnect && _reconnectCoroutine == null)
                _reconnectCoroutine = StartCoroutine(TryReconnect());
        };

        Debug.Log($"[WS] Connecting to {serverUrl}...");
        await socket.Connect();
        if (generation == _connectionGeneration && socket == _ws)
            _connectInProgress = false;
    }

    private IEnumerator TryReconnect()
    {
        while ((maxReconnectAttempts <= 0 || _reconnectAttempts < maxReconnectAttempts) &&
               !IsConnected && !_intentionalClose)
        {
            _reconnectAttempts++;
            string limit = maxReconnectAttempts <= 0 ? "unlimited" : maxReconnectAttempts.ToString();
            float delay = Mathf.Min(5f, reconnectDelay * Mathf.Pow(1.5f, _reconnectAttempts - 1));
            Debug.Log($"[WS] Reconnect attempt {_reconnectAttempts}/{limit} in {delay:F1}s...");
            yield return new WaitForSecondsRealtime(delay);

            if (!IsConnected && !_intentionalClose)
            {
                _connectInProgress = false;
                var task = CreateAndConnect();
                float deadline = Time.realtimeSinceStartup + connectTimeoutSeconds;
                yield return new WaitUntil(() =>
                    IsConnected || task.IsCompleted ||
                    Time.realtimeSinceStartup >= deadline
                );
                if (IsConnected)
                {
                    _reconnectCoroutine = null;
                    yield break;
                }
                if (!task.IsCompleted && _ws != null)
                {
                    ++_connectionGeneration;
                    _ws.CancelConnection();
                    _connectInProgress = false;
                }
            }
        }

        _reconnectCoroutine = null;
        if (!IsConnected && !_intentionalClose)
            Debug.LogError("[WS] Max reconnect attempts reached.");
    }

    public void SendAudio(byte[] data)
    {
        if (IsConnected && _ws != null && _ws.State == WebSocketState.Open)
            _ws.Send(data);
    }

    private static bool LooksLikeJson(byte[] bytes)
    {
        if (bytes == null || bytes.Length < 2) return false;
        int i = 0;
        while (i < bytes.Length && (bytes[i] == 0x20 || bytes[i] == 0x0A || bytes[i] == 0x0D))
            i++;
        return i + 1 < bytes.Length && bytes[i] == (byte)'{' && bytes[i + 1] == (byte)'"';
    }

    public void SendText(string message)
    {
        if (IsConnected && _ws != null && _ws.State == WebSocketState.Open)
            _ws.SendText(message);
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        _ws?.DispatchMessageQueue();
#endif
        if (IsConnected && heartbeatSeconds > 0f && Time.unscaledTime >= _nextHeartbeat)
        {
            _nextHeartbeat = Time.unscaledTime + heartbeatSeconds;
            SendText("{\"v\":1,\"type\":\"client_ping\"}");
        }
    }

    public async void Disconnect()
    {
        _intentionalClose = true;
        ++_connectionGeneration;
        StopReconnectCoroutine();
        if (_ws != null)
        {
            if (_ws.State == WebSocketState.Open)
                await _ws.Close();
            else
                _ws.CancelConnection();
        }
    }

    async void OnDestroy()
    {
        _intentionalClose = true;
        ++_connectionGeneration;
        StopReconnectCoroutine();
        if (_ws != null)
        {
            if (_ws.State == WebSocketState.Open)
                await _ws.Close();
            else
                _ws.CancelConnection();
        }
    }

    private void StopReconnectCoroutine()
    {
        if (_reconnectCoroutine == null) return;
        StopCoroutine(_reconnectCoroutine);
        _reconnectCoroutine = null;
    }
}
