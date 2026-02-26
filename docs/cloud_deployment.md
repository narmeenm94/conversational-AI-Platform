# Cloud Deployment Guide

How to deploy the AI server to a cloud GPU for remote access from anywhere.

## When to Use Cloud Deployment

- Access the AI avatar from any location (not just your local network)
- Scale to multiple concurrent users
- Don't want to keep a local PC running 24/7
- Dedicated always-on deployment

## Cloud GPU Providers

| Provider | GPU | Price | Best For |
|----------|-----|-------|----------|
| [Vast.ai](https://vast.ai) | RTX 4090 | ~$0.20-0.35/hr | Cheapest marketplace |
| [TensorDock](https://tensordock.com) | RTX 4090 | ~$0.12-0.30/hr | Marketplace pricing |
| [RunPod](https://runpod.io) | RTX 4090 | ~$0.34/hr | One-click templates |
| [Lambda](https://lambdalabs.com) | A10G/A100 | ~$0.50-1.00/hr | Enterprise reliability |

### Recommended Specs

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 4070 (12 GB) | RTX 4090 (24 GB) |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB SSD | 100 GB NVMe |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |

## Deployment Steps

### 1. Provision a GPU Server

Choose a provider and spin up an instance with:
- NVIDIA GPU (12+ GB VRAM)
- Ubuntu 22.04
- CUDA drivers pre-installed
- SSH access

### 2. Install Dependencies

```bash
ssh user@your-server-ip

# Clone project
git clone <your-repo-url>
cd conversational-ai-avatar/server

# Setup Python environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
```

### 3. Configure for Remote Access

```bash
cp .env.example .env
```

Edit `.env`:
```bash
SERVER_HOST=0.0.0.0
SERVER_PORT=8765
```

### 4. Open Firewall

```bash
sudo ufw allow 8765/tcp
sudo ufw allow 8765/udp  # If using WebRTC later
```

### 5. Start the Server

```bash
python main.py
```

For persistent operation, use a process manager:

```bash
# Using systemd
sudo tee /etc/systemd/system/ai-avatar.service > /dev/null <<EOF
[Unit]
Description=Conversational AI Avatar Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/.venv/bin/python main.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable ai-avatar
sudo systemctl start ai-avatar
sudo journalctl -u ai-avatar -f  # View logs
```

### 6. Docker Deployment (Alternative)

```bash
cd server
cp .env.example .env
# Edit .env as needed

docker compose up -d

# Pull LLM model into Ollama container
docker exec -it ollama ollama pull llama3.1:8b
```

## Connecting from Quest 3 Over the Internet

### Option A: Direct Connection (Simple)

If your server has a public IP:
1. In Unity, set `Server Address` to the server's public IP
2. Ensure port 8765 is open in the cloud provider's firewall/security group
3. Build and deploy to Quest 3

**Limitation:** Higher latency than LAN. Works for demonstration but not ideal for production.

### Option B: WebRTC (Recommended for Production)

For internet connections, WebRTC handles:
- NAT traversal (works behind firewalls)
- Packet loss recovery
- Jitter buffering
- Adaptive bitrate

To switch from WebSocket to WebRTC:

1. Use Pipecat's WebRTC transport instead of WebSocket:
   ```python
   from pipecat.transports.network.webrtc import WebRTCTransport
   transport = WebRTCTransport(...)
   ```

2. On the Unity side, use Unity's WebRTC package instead of NativeWebSocket

3. Set up a TURN server for NAT traversal:
   ```bash
   # Using coturn
   sudo apt install coturn
   # Configure /etc/turnserver.conf
   ```

Consult the [Pipecat WebRTC documentation](https://github.com/pipecat-ai/pipecat) for detailed setup.

### Option C: VPN/Tailscale (Simplest)

Use [Tailscale](https://tailscale.com) to create a private network:

1. Install Tailscale on your cloud server and Quest 3 (via sideloaded Android app)
2. Both devices join the same Tailnet
3. Use the Tailscale IP as the server address
4. No firewall or NAT configuration needed

## SSL/TLS for Secure WebSocket

For production deployments, use `wss://` (WebSocket Secure):

1. Get an SSL certificate (Let's Encrypt):
   ```bash
   sudo apt install certbot
   sudo certbot certonly --standalone -d your-domain.com
   ```

2. Use a reverse proxy (nginx):
   ```nginx
   server {
       listen 443 ssl;
       server_name your-domain.com;

       ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

       location / {
           proxy_pass http://localhost:8765;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_read_timeout 86400;
       }
   }
   ```

3. Update Unity client to use `wss://your-domain.com`

## Monitoring

### GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Server Logs
```bash
# Direct
python main.py 2>&1 | tee server.log

# systemd
sudo journalctl -u ai-avatar -f

# Docker
docker logs -f ai-server
```

### Latency Testing
```bash
python tools/benchmark_latency.py --url ws://your-server-ip:8765 -n 10
```

## Cost Optimization

- **Spot instances**: Use Vast.ai or RunPod spot pricing for ~50% savings (risk of preemption)
- **Auto-shutdown**: Stop the server when not in use; most providers bill per-minute
- **Model sizing**: Use smaller models (Whisper medium, Orpheus 1B) to fit on cheaper GPUs
- **Batch scheduling**: Run the server only during training/demo hours
