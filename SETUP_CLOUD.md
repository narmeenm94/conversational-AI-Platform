# Cloud GPU Setup Guide (RunPod)

This guide walks you through deploying the Conversational AI Avatar Server
to a RunPod cloud GPU so everything runs in the cloud and your laptop only
needs to run the Unity client.

---

## Prerequisites

- A HuggingFace account with access to [canopylabs/orpheus-tts-0.1-finetune-prod](https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod)
- A HuggingFace access token from <https://huggingface.co/settings/tokens>

---

## Step 1: Create a RunPod Account

1. Go to [runpod.io](https://www.runpod.io/) and sign up
2. Go to **Billing** and add **$10 credit** (this gives you ~25-50 hours of GPU time)

---

## Step 2: Create a GPU Pod

1. Go to **Pods** > **+ Deploy**
2. Select a GPU:
   - **RTX 3090** (24 GB) — ~$0.22/hr (cheapest, works great)
   - **RTX A5000** (24 GB) — ~$0.26/hr
   - **A40** (48 GB) — ~$0.39/hr (most comfortable, faster)
3. Under **Template**, select **RunPod Pytorch 2.4.0** (or any PyTorch template)
4. Set disk sizes:
   - **Container Disk**: 20 GB
   - **Volume Disk**: 50 GB (this persists across restarts for model storage)
5. Under **Expose TCP Ports** (in the advanced settings), add: `8765`
6. Click **Deploy**

Wait for the pod to start (usually 1-2 minutes).

---

## Step 3: Connect to Your Pod

1. Once the pod is running, click **Connect**
2. Choose **Start Web Terminal** or **SSH** (both work)

---

## Step 4: Deploy the Server

In the pod terminal, run these commands:

```bash
# Clone the project into the persistent volume
cd /workspace
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git conversational-ai
cd conversational-ai/server

# Run the deploy script (installs everything + starts the server)
chmod +x deploy.sh
./deploy.sh
```

The script will:
- Install all Python dependencies
- Install Ollama and pull the LLM model (~5 GB)
- Create a `.env` file from the cloud template
- **Pause and ask you to set your HF_TOKEN** — open `.env` in nano/vi and paste your token
- Download the TTS model (~6 GB, first run only)
- Start the server

First-time setup takes about 10-15 minutes (mostly model downloads).

---

## Step 5: Get Your Pod's Public URL

1. Go back to the RunPod dashboard
2. Click on your pod > **Connect**
3. Look for the **TCP Port Mappings** section
4. Find port `8765` — it will show something like:
   ```
   8765 -> 43.xxx.xxx.xxx:28765
   ```
5. Your WebSocket URL is: `ws://43.xxx.xxx.xxx:28765`

---

## Step 6: Test from Your Laptop

Open a PowerShell terminal on your laptop:

```powershell
pip install websockets sounddevice numpy
python tools\mic_test_client.py --url ws://43.xxx.xxx.xxx:28765
```

Replace the IP and port with your pod's actual TCP mapping. You should be
able to speak and hear the AI respond.

---

## Step 7: Connect Unity

In your Unity project, update the WebSocket URL in `WebSocketClient.cs`:

```csharp
private string serverUrl = "ws://43.xxx.xxx.xxx:28765";
```

Replace with your pod's actual TCP mapping URL.

---

## Day-to-Day Usage

### Starting the server (after pod restart)
```bash
cd /workspace/conversational-ai/server
./deploy.sh --start
```

### Stopping the server
Press `Ctrl+C` in the terminal.

### Pausing the pod (saves money)
Go to RunPod dashboard > click **Stop** on your pod. You are not charged
while stopped. Your models and data persist on the volume disk.

### Resuming the pod
Click **Start** on the pod, then SSH in and run `./deploy.sh --start`.

---

## Cost Estimate

| Usage Pattern          | GPU        | Monthly Cost |
|------------------------|------------|-------------|
| 2 hrs/day testing      | RTX 3090   | ~$13        |
| 4 hrs/day testing      | RTX 3090   | ~$26        |
| 2 hrs/day testing      | A40        | ~$23        |
| Always on (24/7)       | RTX 3090   | ~$158       |

Tip: Always **stop your pod** when you're done testing. You only pay for
compute time, not storage.

---

## Troubleshooting

### "Connection refused" from Unity/test client
- Make sure the server is running (`./deploy.sh --start`)
- Make sure you're using the correct TCP mapping URL from the RunPod dashboard
- The port shown in the dashboard (e.g., `28765`) is different from `8765`

### Models downloading every restart
- Make sure you cloned to `/workspace/` (the persistent volume)
- Set `HF_HOME=/workspace/conversational-ai/models/huggingface` in `.env`

### Out of VRAM errors
- You need at least 24 GB VRAM. Use RTX 3090, A5000, or A40.

### Ollama not responding
```bash
# Check if Ollama is running
pgrep ollama

# If not, start it
ollama serve &
sleep 3
ollama pull llama3.1:8b
```
