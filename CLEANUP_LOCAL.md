# Reclaim Local Storage

After your cloud server is confirmed working, run these commands to free
up space on your local drives. Open PowerShell as Administrator.

---

## What's Using Space

| Location                                        | Size     | What              |
|------------------------------------------------|----------|-------------------|
| `D:\Conversational AI Platform\models\`        | ~26 GB   | Ollama + HF models |
| `D:\Conversational AI Platform\server\.venv\`  | ~5 GB    | Python virtual env |
| Ollama application (C: drive)                   | ~0.5 GB  | Ollama installer   |

**Total recoverable: ~31 GB**

---

## Cleanup Commands

Run these in PowerShell (stop the server first with Ctrl+C):

```powershell
# 1. Delete all downloaded AI models (~26 GB)
Remove-Item -Recurse -Force "D:\Conversational AI Platform\models"

# 2. Delete the Python virtual environment (~5 GB)
Remove-Item -Recurse -Force "D:\Conversational AI Platform\server\.venv"

# 3. Delete HuggingFace cache on C: drive (if it exists)
Remove-Item -Recurse -Force "C:\Users\narme\.cache\huggingface" -ErrorAction SilentlyContinue

# 4. (Optional) Uninstall Ollama — open Settings > Apps > Ollama > Uninstall
```

---

## What to Keep

Keep the project source code in `D:\Conversational AI Platform\` — it's
small (a few MB) and you'll need it for Unity client scripts and if you
want to push updates to the cloud server.

---

## If You Want to Run Locally Again Later

If you get a machine with 24+ GB VRAM and want to switch back to local:

```powershell
cd "D:\Conversational AI Platform\server"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install git+https://github.com/Deathdadev/Orpheus-Speech-PyPi accelerate
# Models will re-download on first run
python main.py
```
