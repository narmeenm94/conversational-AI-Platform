using System;
using System.Threading.Tasks;

namespace NativeWebSocket
{
    public enum WebSocketState { Connecting, Open, Closing, Closed }
    public enum WebSocketCloseCode { Normal = 1000 }

    public sealed class WebSocket
    {
        public WebSocket(string url) { }
        public WebSocketState State { get; private set; } = WebSocketState.Open;
        public event Action OnOpen;
        public event Action<byte[]> OnMessage;
        public event Action<string> OnError;
        public event Action<WebSocketCloseCode> OnClose;
        public Task Connect() => Task.CompletedTask;
        public void CancelConnection() { }
        public Task Close() => Task.CompletedTask;
        public Task Send(byte[] bytes) => Task.CompletedTask;
        public Task SendText(string text) => Task.CompletedTask;
        public void DispatchMessageQueue() { }
    }
}
