using System;
using System.Collections;
using NativeWebSocket;
using UnityEngine;

public class WebSocketClient : MonoBehaviour
{
    [Tooltip("Desktop: ws://localhost:8765 | Quest 3: ws://YOUR_PC_IP:8765")]
    public string serverUrl = "ws://localhost:8765";

    [Header("Reconnection")]
    public bool autoReconnect = true;
    public float reconnectDelay = 3f;
    public int maxReconnectAttempts = 10;

    public event Action OnConnected;
    public event Action OnDisconnected;
    public event Action<byte[]> OnAudioReceived;
    public event Action<string> OnTextReceived;

    private WebSocket _ws;
    private int _reconnectAttempts;
    private bool _intentionalClose;

    public bool IsConnected { get; private set; }

    public async void Connect()
    {
        _intentionalClose = false;
        _reconnectAttempts = 0;

        await CreateAndConnect();
    }

    private async System.Threading.Tasks.Task CreateAndConnect()
    {
        _ws = new WebSocket(serverUrl);

        _ws.OnOpen += () =>
        {
            IsConnected = true;
            _reconnectAttempts = 0;
            Debug.Log($"[WS] Connected to {serverUrl}");
            OnConnected?.Invoke();
        };

        _ws.OnMessage += (bytes) =>
        {
            OnAudioReceived?.Invoke(bytes);
        };

        _ws.OnError += (e) =>
        {
            Debug.LogError($"[WS] Error: {e}");
        };

        _ws.OnClose += (code) =>
        {
            IsConnected = false;
            Debug.Log($"[WS] Disconnected (code: {code})");
            OnDisconnected?.Invoke();

            if (!_intentionalClose && autoReconnect)
                StartCoroutine(TryReconnect());
        };

        Debug.Log($"[WS] Connecting to {serverUrl}...");
        await _ws.Connect();
    }

    private IEnumerator TryReconnect()
    {
        while (_reconnectAttempts < maxReconnectAttempts && !IsConnected && !_intentionalClose)
        {
            _reconnectAttempts++;
            Debug.Log($"[WS] Reconnect attempt {_reconnectAttempts}/{maxReconnectAttempts} in {reconnectDelay}s...");
            yield return new WaitForSeconds(reconnectDelay);

            if (!IsConnected && !_intentionalClose)
            {
                var task = CreateAndConnect();
                yield return new WaitUntil(() => task.IsCompleted);
                if (IsConnected) yield break;
            }
        }

        if (!IsConnected)
            Debug.LogError("[WS] Max reconnect attempts reached.");
    }

    public void SendAudio(byte[] data)
    {
        if (IsConnected && _ws != null && _ws.State == WebSocketState.Open)
            _ws.Send(data);
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
    }

    public async void Disconnect()
    {
        _intentionalClose = true;
        if (_ws != null)
            await _ws.Close();
    }

    async void OnDestroy()
    {
        _intentionalClose = true;
        if (_ws != null)
            await _ws.Close();
    }
}
