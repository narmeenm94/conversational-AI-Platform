# Quest 3 Setup Guide

Complete guide to opening the included Unity project and deploying it to Meta Quest 3.

## Prerequisites

- Unity 6.5 (6000.5.4f1) with Android Build Support, OpenJDK, and Android SDK/NDK tools
- Meta Quest 3, 3S, or Pro headset
- USB-C cable for sideloading (or Wi-Fi deploy via Meta Quest Developer Hub)
- [Meta Quest Developer account](https://developer.oculus.com/)
- Headset in Developer Mode

## Step 1: Open the Included Unity Project

1. Open Unity Hub
2. Click **Add** and select this repository's `unity-client` folder
3. Open it with **Unity 6.5 (6000.5.4f1)**
4. Allow Unity to restore the pinned packages and import the project

Do not create a blank project or copy the scripts manually. The repository
already contains the scenes, XR configuration, packages, avatar, animation
controller, Android manifest, WebSocket client, and uLipSync integration.

## Step 2: Switch to Android Platform

1. **File → Build Settings**
2. Select **Android**
3. Click **Switch Platform** (this takes a few minutes)

## Step 3: Player Settings

Go to **Edit → Project Settings → Player** and configure:

| Setting | Value |
|---------|-------|
| Company Name | Your company |
| Product Name | AI Avatar |
| Minimum API Level | Android 10 (API 29) |
| Scripting Backend | IL2CPP |
| Target Architectures | ARM64 only (uncheck ARMv7) |
| Install Location | Automatic |

## Step 4: Verify Included Packages

Open **Window → Package Manager** and verify that Unity restored the packages
from `unity-client/Packages/manifest.json`, including:

- **XR Interaction Toolkit**
- **OpenXR Plugin** with Meta Quest support
- **NativeWebSocket** (embedded in the repository)
- **uLipSync** (embedded in the repository)

Do not add another copy of these packages; duplicate imports can cause compile
errors or conflicting components.

For the included Avaturn head, map `A/I/U/E/O/N/-` to
`viseme_aa/viseme_I/viseme_U/viseme_E/viseme_O/viseme_nn/viseme_sil`.
Attach uLipSync to the same `AudioSource` used by `AudioStreamPlayer`, then
disable `AvatarController.driveMouthFromVolume`.

## Step 5: Configure XR

1. **Edit → Project Settings → XR Plug-in Management**
2. Android tab: check **OpenXR**
3. Under OpenXR, add **Meta Quest** feature group
4. Set Render Mode to **Multi-pass** (more compatible) or **Single Pass Instanced** (faster)

## Step 6: Verify Scripts

Unity compiles the scripts already present in `unity-client/Assets/Scripts/`.
Wait for compilation to finish and confirm that the Console has no errors
before opening a demo scene.

## Step 7: Import Avatar

Options for 3D avatars:

### Avaturn (included avatar)
1. Follow the [Avaturn Unity import guide](https://docs.avaturn.me/docs/importing/unity/).
2. Import `Avatar.glb` with glTFast and place the generated avatar in the scene.
3. Select the avatar root and run **Conversational AI → Configure Selected Avaturn Avatar**.
4. The included GLB already has a humanoid skin, ARKit expressions, blinks, jaw controls, and all Oculus visemes.

### Ready Player Me
1. Go to [readyplayer.me](https://readyplayer.me)
2. Create an avatar
3. Download as GLB
4. Import into Unity (use GLTFUtility or UniGLTF)

### VRoid Studio
1. Create avatar in [VRoid Studio](https://vroid.com/en/studio)
2. Export as VRM
3. Import using UniVRM package

### Custom FBX
- Must have blend shapes for lip sync (at least `jawOpen` or equivalent)
- Import into `Assets/Models/`

## Step 8: Verify the Included Avaturn Animations

The demo uses the compatible Avaturn animation assets and its configured
Animator Controller. Press Play and verify smooth Idle, Talking, and Waiting
state changes. Do not replace these with Mixamo clips during initial setup;
retargeting clips from a different rig can produce a T-pose or root-motion
offsets.

## Step 9: Set Up Scene Hierarchy

Create or arrange GameObjects:

```
ConversationalAvatarScene
├── XR Origin (Meta XR)
│   ├── Camera Offset → Main Camera
│   ├── Left Controller
│   └── Right Controller
├── Environment
│   ├── Floor (plane)
│   ├── Room (optional enclosure)
│   └── Directional Light
├── ConversationalAvatar [Prefab]
│   ├── AvatarModel (SkinnedMeshRenderer with blend shapes)
│   ├── Animator Controller
│   ├── AudioSource
│   ├── uLipSync component
│   └── AvatarController.cs
└── ConversationSystem [Empty GameObject]
    ├── ConversationManager.cs
    ├── WebSocketClient.cs
    ├── MicCapture.cs
    └── AudioStreamPlayer.cs
```

### Wiring Components

On the **ConversationSystem** object:
1. `ConversationManager` → drag references to WebSocket, MicCapture, AudioStreamPlayer, AvatarController
2. `MicCapture` → drag WebSocketClient reference
3. `AudioStreamPlayer` → drag the avatar's AudioSource

On the **ConversationalAvatar**:
1. `AvatarController` → assign face mesh, set blend shape index for mouth open, drag AudioStreamPlayer

## Step 10: Configure for Desktop Testing

1. Set `ConversationManager → Server Address = "localhost"`
2. Set `ConversationManager → Server Port = 8765`
3. Start the Python server (`python main.py`)
4. Press Play in Unity Editor
5. Speak into your mic — the avatar should respond!

## Step 11: Build for Quest 3

1. Find your PC's IP: run `python tools/find_my_ip.py`
2. Open Windows Firewall for conversation port 8765 and character-control port 8766:
   ```
   netsh advfirewall firewall add rule name="AI Avatar" dir=in action=allow protocol=tcp localport=8765
   netsh advfirewall firewall add rule name="AI Character Control" dir=in action=allow protocol=tcp localport=8766
   ```
3. In Unity: change `Server Address` to your PC's IP (e.g., `192.168.1.100`)
4. **File → Build Settings → Android**
5. Connect Quest 3 via USB-C
6. Click **Build and Run**
7. Put on the headset — the app auto-launches

## Quest 3 Performance Targets

| Constraint | Target |
|-----------|--------|
| Triangle count | Under 750K total scene |
| Draw calls | Under 100 |
| Textures | 2K max, use atlases |
| Shaders | URP Lit or Mobile only |
| Lighting | Baked preferred, 1 realtime directional max |
| Frame rate | 72 fps (Quest 3 native) |
| Audio | Mono, 24kHz, one AudioSource |

## Troubleshooting

**Quest 3 can't connect:**
- Ensure Quest 3 and PC are on the same Wi-Fi network (5GHz preferred)
- Verify firewall allows port 8765
- Check server shows `0.0.0.0` as host (not `127.0.0.1`)
- Try `adb shell ping YOUR_PC_IP` from a terminal

**No mic input on Quest:**
- Check `AndroidManifest.xml` has `RECORD_AUDIO` permission
- Go to Quest Settings → Privacy → check app has mic permission
- Verify the runtime permission request fires on start

**Low frame rate:**
- Reduce avatar polygon count
- Bake lighting instead of realtime
- Use simpler shaders (Mobile/Unlit)
- Profile with Meta Quest Developer Hub
