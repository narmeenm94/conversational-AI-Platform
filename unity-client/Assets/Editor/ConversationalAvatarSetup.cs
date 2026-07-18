#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class ConversationalAvatarSetup
{
    public static void LogIncludedAvatarDiagnostics()
    {
        const string avatarPath = "Assets/Models/Avatar.glb";
        foreach (Object asset in AssetDatabase.LoadAllAssetsAtPath(avatarPath))
        {
            if (asset is AnimationClip clip)
                Debug.Log($"[Avatar diagnostics] clip={clip.name}, length={clip.length:F2}s, legacy={clip.legacy}");
            if (asset is Mesh mesh)
            {
                for (int i = 0; i < mesh.blendShapeCount; i++)
                {
                    if (!mesh.GetBlendShapeName(i).StartsWith("viseme_")) continue;
                    int frame = mesh.GetBlendShapeFrameCount(i) - 1;
                    float frameWeight = frame >= 0 ? mesh.GetBlendShapeFrameWeight(i, frame) : 0f;
                    Debug.Log($"[Avatar diagnostics] mesh={mesh.name}, shape={mesh.GetBlendShapeName(i)}, frameWeight={frameWeight}");
                }
            }
        }
    }

    [MenuItem("Conversational AI/Create Demo Scene From Included Avatar")]
    public static void CreateDemoScene()
    {
        const string avatarPath = "Assets/Models/Avatar.glb";
        const string scenePath = "Assets/Scenes/ConversationalAvatarDemo.unity";
        GameObject avatarAsset = AssetDatabase.LoadAssetAtPath<GameObject>(avatarPath);
        if (avatarAsset == null)
            throw new System.InvalidOperationException(
                $"The imported avatar was not found at {avatarPath}."
            );

        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        GameObject avatar = PrefabUtility.InstantiatePrefab(avatarAsset) as GameObject;
        if (avatar == null) avatar = Object.Instantiate(avatarAsset);
        avatar.name = "ConversationalAvatar";
        avatar.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);

        Bounds bounds = GetRendererBounds(avatar);
        GameObject cameraObject = new GameObject("Main Camera");
        Camera camera = cameraObject.AddComponent<Camera>();
        cameraObject.tag = "MainCamera";
        cameraObject.transform.position = bounds.center + new Vector3(0f, 0.1f, 2.8f);
        cameraObject.transform.LookAt(bounds.center + Vector3.up * 0.05f);

        GameObject lightObject = new GameObject("Directional Light");
        Light light = lightObject.AddComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = 1.2f;
        lightObject.transform.rotation = Quaternion.Euler(45f, -30f, 0f);

        GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.position = new Vector3(bounds.center.x, bounds.min.y, bounds.center.z);
        floor.transform.localScale = Vector3.one * 0.5f;

        Selection.activeGameObject = avatar;
        ConfigureSelectedAvatar();
        EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(), scenePath);
        AssetDatabase.SaveAssets();
        Debug.Log($"[Conversational AI] Demo scene created at {scenePath}");
    }

    [MenuItem("Conversational AI/Configure Selected Avaturn Avatar")]
    public static void ConfigureSelectedAvatar()
    {
        GameObject avatar = Selection.activeGameObject;
        if (avatar == null)
        {
            EditorUtility.DisplayDialog(
                "Conversational AI",
                "Select the imported Avaturn avatar root in the Hierarchy first.",
                "OK"
            );
            return;
        }

        Undo.RegisterFullObjectHierarchyUndo(avatar, "Configure conversational avatar");
        AudioSource source = GetOrAdd<AudioSource>(avatar);
        source.playOnAwake = false;
        source.loop = true;
        source.spatialBlend = 1f;
        source.minDistance = 0.5f;
        source.maxDistance = 12f;

        AudioStreamPlayer player = GetOrAdd<AudioStreamPlayer>(avatar);
        player.audioSource = source;

        AvatarController controller = GetOrAdd<AvatarController>(avatar);
        controller.audioPlayer = player;
        controller.faceMeshes = avatar.GetComponentsInChildren<SkinnedMeshRenderer>(true);
        Animator importedAnimator = avatar.GetComponentInChildren<Animator>(true);
        controller.animator = importedAnimator != null &&
            importedAnimator.runtimeAnimatorController != null
            ? importedAnimator
            : null;

        SkinnedMeshRenderer face = FindVisemeRenderer(controller.faceMeshes);
        if (face != null)
        {
            var lipSync = GetOrAdd<uLipSync.uLipSync>(avatar);
            var lipSyncAudioSource = GetOrAdd<uLipSync.uLipSyncAudioSource>(avatar);
            lipSync.audioSourceProxy = lipSyncAudioSource;
            lipSync.profile = AssetDatabase.LoadAssetAtPath<uLipSync.Profile>(
                "Packages/com.hecomi.ulipsync/Assets/Profiles/uLipSync-Profile-Sample-Female.asset"
            );

            var blendShape = GetOrAdd<uLipSync.uLipSyncBlendShape>(avatar);
            blendShape.skinnedMeshRenderer = face;
            blendShape.blendShapes.Clear();
            blendShape.AddBlendShape("A", "viseme_aa");
            blendShape.AddBlendShape("I", "viseme_I");
            blendShape.AddBlendShape("U", "viseme_U");
            blendShape.AddBlendShape("E", "viseme_E");
            blendShape.AddBlendShape("O", "viseme_O");
            blendShape.AddBlendShape("N", "viseme_nn");
            blendShape.AddBlendShape("-", "viseme_sil");
            blendShape.maxBlendShapeValue = GetBlendShapeFrameWeight(face, "viseme_aa");
            lipSync.onLipSyncUpdate.RemoveListener(blendShape.OnLipSyncUpdate);
            lipSync.onLipSyncUpdate.AddListener(blendShape.OnLipSyncUpdate);
            controller.driveMouthFromVolume = false;
        }

        GameObject system = GameObject.Find("ConversationSystem");
        if (system == null)
        {
            system = new GameObject("ConversationSystem");
            Undo.RegisterCreatedObjectUndo(system, "Create conversation system");
        }

        WebSocketClient socket = GetOrAdd<WebSocketClient>(system);
        MicCapture mic = GetOrAdd<MicCapture>(system);
        mic.webSocket = socket;
        ConversationManager manager = GetOrAdd<ConversationManager>(system);
        manager.webSocket = socket;
        manager.micCapture = mic;
        manager.audioPlayer = player;
        manager.avatarController = controller;
        CharacterPlatformClient characterPlatform = GetOrAdd<CharacterPlatformClient>(system);
        characterPlatform.serverAddress = manager.serverAddress;
        manager.characterPlatform = characterPlatform;
        manager.enableBargeIn = false;
        mic.muteWhileAssistantSpeaks = true;
        mic.speakerEchoTailSeconds = 0.65f;

        EditorUtility.SetDirty(avatar);
        EditorUtility.SetDirty(system);
        Selection.activeGameObject = system;

        Debug.Log(face != null
            ? "[Conversational AI] Avatar, streaming audio, and uLipSync visemes configured."
            : "[Conversational AI] Conversation configured, but no renderer with viseme_aa was found."
        );
    }

    private static T GetOrAdd<T>(GameObject target) where T : Component
    {
        T component = target.GetComponent<T>();
        return component != null ? component : Undo.AddComponent<T>(target);
    }

    private static SkinnedMeshRenderer FindVisemeRenderer(
        SkinnedMeshRenderer[] renderers)
    {
        foreach (SkinnedMeshRenderer renderer in renderers)
        {
            Mesh mesh = renderer.sharedMesh;
            if (mesh == null) continue;
            for (int i = 0; i < mesh.blendShapeCount; i++)
            {
                if (mesh.GetBlendShapeName(i) == "viseme_aa") return renderer;
            }
        }
        return null;
    }

    private static Bounds GetRendererBounds(GameObject root)
    {
        Renderer[] renderers = root.GetComponentsInChildren<Renderer>(true);
        if (renderers.Length == 0) return new Bounds(Vector3.up, Vector3.one * 2f);
        Bounds bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++) bounds.Encapsulate(renderers[i].bounds);
        return bounds;
    }

    private static float GetBlendShapeFrameWeight(
        SkinnedMeshRenderer renderer,
        string blendShapeName)
    {
        if (renderer == null || renderer.sharedMesh == null) return 100f;
        int index = renderer.sharedMesh.GetBlendShapeIndex(blendShapeName);
        if (index < 0) return 100f;
        int frameCount = renderer.sharedMesh.GetBlendShapeFrameCount(index);
        if (frameCount <= 0) return 100f;
        float weight = Mathf.Abs(
            renderer.sharedMesh.GetBlendShapeFrameWeight(index, frameCount - 1)
        );
        return weight > Mathf.Epsilon ? weight : 100f;
    }
}
#endif
