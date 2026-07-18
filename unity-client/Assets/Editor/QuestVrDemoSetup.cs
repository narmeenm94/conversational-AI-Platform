using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;
using UnityEngine.XR;
using Object = UnityEngine.Object;

/// <summary>
/// Builds the standalone Quest 3 demo from the working desktop scene after the
/// official XR packages have resolved. Package APIs are accessed reflectively
/// so the project remains compilable during Package Manager installation.
/// </summary>
[InitializeOnLoad]
public static class QuestVrDemoSetup
{
    private const string SourceScene = "Assets/Scenes/ConversationalAvatarDemo.unity";
    private const string QuestScene = "Assets/Scenes/ConversationalAvatarQuest3.unity";
    private const string CigaretteAsset = "Assets/Models/cigarette_-_smoke_on.glb";
    private const string SessionKey = "ConversationalAI.QuestDemoBuilt.v1";
    private const string QuestServerAddress = "192.168.50.2";
    private static double _nextAttempt;

    static QuestVrDemoSetup() => EditorApplication.update += AutoBuildWhenReady;

    private static void AutoBuildWhenReady()
    {
        if (SessionState.GetBool(SessionKey, false))
        {
            EditorApplication.update -= AutoBuildWhenReady;
            return;
        }
        if (EditorApplication.timeSinceStartup < _nextAttempt ||
            EditorApplication.isPlayingOrWillChangePlaymode ||
            EditorApplication.isCompiling || EditorApplication.isUpdating)
            return;
        _nextAttempt = EditorApplication.timeSinceStartup + 3d;
        if (!XrPackagesReady()) return;

        try
        {
            BuildQuestDemo();
            SessionState.SetBool(SessionKey, true);
            EditorApplication.update -= AutoBuildWhenReady;
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
        }
    }

    [MenuItem("Tools/Conversational AI/Build Meta Quest 3 Demo Scene")]
    public static void BuildQuestDemo()
    {
        if (!XrPackagesReady())
            throw new InvalidOperationException(
                "XR packages are still resolving. Wait for Package Manager to finish, then run this command again."
            );

        ConfigureQuestPlayerSettings();
        ConfigureOpenXR();

        Scene active = SceneManager.GetActiveScene();
        if (active.isDirty) EditorSceneManager.SaveScene(active);
        Scene source = active.path == SourceScene
            ? active
            : EditorSceneManager.OpenScene(SourceScene, OpenSceneMode.Single);
        AttachCigarette(source);
        EditorSceneManager.SaveScene(source);

        if (!AssetDatabase.LoadAssetAtPath<SceneAsset>(QuestScene))
        {
            if (!AssetDatabase.CopyAsset(SourceScene, QuestScene))
                throw new InvalidOperationException("Could not duplicate the desktop scene.");
            AssetDatabase.Refresh();
        }

        Scene quest = EditorSceneManager.OpenScene(QuestScene, OpenSceneMode.Single);
        ConfigureQuestScene(quest);
        AttachCigarette(quest);
        EditorSceneManager.SaveScene(quest);
        ConfigureBuildScenes();
        AssetDatabase.SaveAssets();
        Debug.Log(
            "[Conversational AI] Quest 3 demo ready: OpenXR rig, gaze conversation, " +
            "state orb, native Avaturn animation flow, environment, and hand prop configured."
        );
    }

    [MenuItem("Tools/Conversational AI/Attach Cigarette To Right Hand")]
    public static void AttachCigaretteToCurrentScene()
    {
        AttachCigarette(SceneManager.GetActiveScene());
        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        EditorSceneManager.SaveOpenScenes();
    }

    private static bool XrPackagesReady() =>
        FindType("Unity.XR.CoreUtils.XROrigin", "Unity.XR.CoreUtils") != null &&
        FindType("UnityEngine.XR.OpenXR.OpenXRLoader", "Unity.XR.OpenXR") != null &&
        FindType("UnityEngine.XR.Interaction.Toolkit.XRInteractionManager", "Unity.XR.Interaction.Toolkit") != null;

    private static void ConfigureQuestScene(Scene scene)
    {
        GameObject oldCamera = FindSceneObject(scene, "Main Camera");
        if (oldCamera != null) Object.DestroyImmediate(oldCamera);

        GameObject oldRig = FindSceneObject(scene, "Quest XR Origin");
        if (oldRig != null) Object.DestroyImmediate(oldRig);

        var rig = new GameObject("Quest XR Origin");
        rig.transform.SetPositionAndRotation(new Vector3(0f, 0f, 1.8f), Quaternion.Euler(0f, 180f, 0f));
        var offset = new GameObject("Camera Offset");
        offset.transform.SetParent(rig.transform, false);
        // Floor tracking supplies the real headset height. The offset is only used
        // as a comfortable standing-height fallback when a runtime exposes Device space.
        offset.transform.localPosition = new Vector3(0f, 1.65f, 0f);
        var cameraObject = new GameObject("Main Camera");
        cameraObject.tag = "MainCamera";
        cameraObject.transform.SetParent(offset.transform, false);
        cameraObject.transform.localPosition = Vector3.zero;
        Camera camera = cameraObject.AddComponent<Camera>();
        camera.nearClipPlane = 0.05f;
        camera.farClipPlane = 50f;
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = new Color(0.018f, 0.022f, 0.03f, 1f);
        cameraObject.AddComponent<AudioListener>();
        cameraObject.AddComponent<OpenXRNodePoseDriver>().node = XRNode.CenterEye;

        Type xrOriginType = FindType("Unity.XR.CoreUtils.XROrigin", "Unity.XR.CoreUtils");
        Component xrOrigin = rig.AddComponent(xrOriginType);
        SetProperty(xrOrigin, "Camera", camera);
        SetProperty(xrOrigin, "CameraFloorOffsetObject", offset);
        SetProperty(xrOrigin, "CameraYOffset", 1.65f);
        SetEnumProperty(xrOrigin, "RequestedTrackingOriginMode", "Floor");

        CreateTrackedAnchor(offset.transform, "Left Controller", XRNode.LeftHand);
        CreateTrackedAnchor(offset.transform, "Right Controller", XRNode.RightHand);

        var interactionManager = new GameObject("XR Interaction Manager");
        interactionManager.AddComponent(FindType(
            "UnityEngine.XR.Interaction.Toolkit.XRInteractionManager",
            "Unity.XR.Interaction.Toolkit"
        ));

        ConversationManager conversation = Object.FindAnyObjectByType<ConversationManager>();
        AvatarController avatar = Object.FindAnyObjectByType<AvatarController>();
        if (conversation == null || avatar == null)
            throw new InvalidOperationException("The duplicated scene is missing its conversation system or avatar.");
        conversation.serverAddress = QuestServerAddress;
        conversation.enableBargeIn = true;

        BuildEnvironment(scene);
        Transform head = FindTransform(scene, "Head");
        BuildStatusOrb(head, conversation, camera.transform);

        GameObject gazeHost = conversation.gameObject;
        GazeConversationTarget gaze = gazeHost.GetComponent<GazeConversationTarget>();
        if (gaze == null) gaze = gazeHost.AddComponent<GazeConversationTarget>();
        gaze.viewer = camera.transform;
        gaze.target = head != null ? head : avatar.transform;
        gaze.conversation = conversation;
        gaze.indicator = Object.FindAnyObjectByType<ConversationStateOrb>();

        EditorSceneManager.MarkSceneDirty(scene);
    }

    private static void BuildEnvironment(Scene scene)
    {
        GameObject previous = FindSceneObject(scene, "Quest Demo Environment");
        if (previous != null) Object.DestroyImmediate(previous);
        GameObject originalFloor = FindSceneObject(scene, "Floor");
        if (originalFloor != null) originalFloor.SetActive(false);

        Material charcoal = GetOrCreateMaterial(
            "Assets/Materials/Quest Charcoal.mat", new Color(0.025f, 0.03f, 0.042f), 0f
        );
        Material graphite = GetOrCreateMaterial(
            "Assets/Materials/Quest Graphite.mat", new Color(0.09f, 0.105f, 0.13f), 0.15f
        );
        Material orange = GetOrCreateMaterial(
            "Assets/Materials/Metropolia Orange Glow.mat", new Color(1f, 0.22f, 0f), 2.2f
        );

        var root = new GameObject("Quest Demo Environment");
        CreatePrimitive(root.transform, PrimitiveType.Cylinder, "Stage Floor",
            new Vector3(0f, -0.08f, 0.35f), new Vector3(3.5f, 0.05f, 3.5f), charcoal);
        CreatePrimitive(root.transform, PrimitiveType.Cylinder, "Avatar Platform",
            new Vector3(0f, -0.015f, 0f), new Vector3(0.72f, 0.07f, 0.72f), graphite);
        CreatePrimitive(root.transform, PrimitiveType.Cylinder, "Orange Platform Ring",
            new Vector3(0f, -0.035f, 0f), new Vector3(0.79f, 0.025f, 0.79f), orange);
        CreatePrimitive(root.transform, PrimitiveType.Cube, "Backdrop",
            new Vector3(0f, 1.35f, -1.4f), new Vector3(4.8f, 2.7f, 0.08f), charcoal);
        CreatePrimitive(root.transform, PrimitiveType.Cube, "Left Accent",
            new Vector3(-1.45f, 1.2f, -1.28f), new Vector3(0.035f, 1.7f, 0.045f), orange);
        CreatePrimitive(root.transform, PrimitiveType.Cube, "Right Accent",
            new Vector3(1.45f, 1.2f, -1.28f), new Vector3(0.035f, 1.7f, 0.045f), orange);

        Light key = CreateLight(root.transform, "Warm Key", LightType.Spot,
            new Vector3(-1.4f, 2.6f, 1.25f), new Color(1f, 0.58f, 0.38f), 5.2f, 8f);
        key.transform.LookAt(new Vector3(0f, 1.25f, 0f));
        key.spotAngle = 48f;
        Light fill = CreateLight(root.transform, "Cool Fill", LightType.Spot,
            new Vector3(1.6f, 2.1f, 1f), new Color(0.2f, 0.55f, 1f), 3.4f, 7f);
        fill.transform.LookAt(new Vector3(0f, 1.2f, 0f));
        fill.spotAngle = 55f;

        RenderSettings.ambientMode = AmbientMode.Trilight;
        RenderSettings.ambientSkyColor = new Color(0.16f, 0.18f, 0.24f);
        RenderSettings.ambientEquatorColor = new Color(0.055f, 0.06f, 0.08f);
        RenderSettings.ambientGroundColor = new Color(0.018f, 0.02f, 0.027f);
        RenderSettings.ambientIntensity = 0.8f;
    }

    private static void BuildStatusOrb(Transform head, ConversationManager conversation, Transform viewer)
    {
        GameObject previous = GameObject.Find("Conversation Status Orb");
        if (previous != null) Object.DestroyImmediate(previous);

        var root = new GameObject("Conversation Status Orb");
        root.transform.position = head != null ? head.position + Vector3.up * 0.32f : new Vector3(0f, 2.1f, 0f);
        var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphere.name = "State Light";
        sphere.transform.SetParent(root.transform, false);
        sphere.transform.localScale = Vector3.one * 0.13f;
        Renderer renderer = sphere.GetComponent<Renderer>();
        renderer.sharedMaterial = GetOrCreateMaterial(
            "Assets/Materials/Conversation State Orb.mat", new Color(0.27f, 0.29f, 0.32f), 3.2f
        );
        Collider collider = sphere.GetComponent<Collider>();
        if (collider != null) Object.DestroyImmediate(collider);

        var glowObject = new GameObject("Orb Glow");
        glowObject.transform.SetParent(root.transform, false);
        Light glow = glowObject.AddComponent<Light>();
        glow.type = LightType.Point;
        glow.range = 1.2f;
        glow.intensity = 1.1f;
        glow.shadows = LightShadows.None;

        var labelObject = new GameObject("State Label");
        labelObject.transform.SetParent(root.transform, false);
        labelObject.transform.localPosition = new Vector3(0f, -0.2f, 0f);
        TextMesh label = labelObject.AddComponent<TextMesh>();
        label.text = "LOOK TO TALK";
        label.anchor = TextAnchor.UpperCenter;
        label.alignment = TextAlignment.Center;
        label.fontSize = 52;
        label.characterSize = 0.018f;
        label.color = new Color(0.7f, 0.72f, 0.75f);

        ConversationStateOrb indicator = root.AddComponent<ConversationStateOrb>();
        indicator.orbRenderer = renderer;
        indicator.glowLight = glow;
        indicator.statusText = label;
        indicator.followTarget = head;
        indicator.conversation = conversation;
    }

    private static void AttachCigarette(Scene scene)
    {
        Transform hand = FindTransform(scene, "RightHand");
        if (hand == null)
        {
            Debug.LogWarning("[Conversational AI] RightHand bone not found; cigarette was not attached.");
            return;
        }

        GameObject cigarette = scene.GetRootGameObjects()
            .SelectMany(root => root.GetComponentsInChildren<Transform>(true))
            .Select(item => item.gameObject)
            .FirstOrDefault(item => item.name.IndexOf("cigarette", StringComparison.OrdinalIgnoreCase) >= 0);
        bool preservePlacedPose = cigarette != null && Vector3.Distance(cigarette.transform.position, hand.position) < 0.6f;
        if (cigarette == null)
        {
            GameObject asset = AssetDatabase.LoadAssetAtPath<GameObject>(CigaretteAsset);
            if (asset == null) throw new InvalidOperationException("Cigarette GLB could not be loaded.");
            cigarette = PrefabUtility.InstantiatePrefab(asset, scene) as GameObject;
            cigarette.name = "Cigarette (Right Hand)";
        }

        cigarette.transform.SetParent(hand, preservePlacedPose);
        HandPropAttachment attachment = cigarette.GetComponent<HandPropAttachment>();
        if (attachment == null) attachment = cigarette.AddComponent<HandPropAttachment>();
        attachment.hand = hand;
        if (preservePlacedPose)
            attachment.CaptureCurrentLocalPose();
        else
        {
            attachment.localPosition = new Vector3(0.035f, -0.015f, 0.02f);
            attachment.localEulerAngles = Vector3.zero;
            attachment.localScale = Vector3.one * 0.012f;
            cigarette.transform.localPosition = attachment.localPosition;
            cigarette.transform.localRotation = Quaternion.Euler(attachment.localEulerAngles);
            cigarette.transform.localScale = attachment.localScale;
        }
        EditorUtility.SetDirty(attachment);
        EditorSceneManager.MarkSceneDirty(scene);
    }

    private static void ConfigureQuestPlayerSettings()
    {
        PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel29;
        PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
        PlayerSettings.colorSpace = ColorSpace.Linear;
        PlayerSettings.SetGraphicsAPIs(
            BuildTarget.Android,
            new[] { GraphicsDeviceType.Vulkan, GraphicsDeviceType.OpenGLES3 }
        );
    }

    private static void ConfigureOpenXR()
    {
        Type perTargetType = FindType(
            "UnityEditor.XR.Management.XRGeneralSettingsPerBuildTarget",
            "Unity.XR.Management.Editor"
        );
        object perTarget = perTargetType.GetMethod(
            "GetOrCreate", BindingFlags.NonPublic | BindingFlags.Static
        ).Invoke(null, null);
        MethodInfo hasManager = perTargetType.GetMethod("HasManagerSettingsForBuildTarget");
        if (!(bool)hasManager.Invoke(perTarget, new object[] { BuildTargetGroup.Android }))
            perTargetType.GetMethod("CreateDefaultManagerSettingsForBuildTarget")
                .Invoke(perTarget, new object[] { BuildTargetGroup.Android });
        object manager = perTargetType.GetMethod("ManagerSettingsForBuildTarget")
            .Invoke(perTarget, new object[] { BuildTargetGroup.Android });

        Type storeType = FindType(
            "UnityEditor.XR.Management.Metadata.XRPackageMetadataStore",
            "Unity.XR.Management.Editor"
        );
        MethodInfo assign = storeType.GetMethods(BindingFlags.Public | BindingFlags.Static)
            .First(method => method.Name == "AssignLoader" && method.GetParameters().Length == 3);
        bool assigned = (bool)assign.Invoke(null, new[]
        {
            manager,
            "UnityEngine.XR.OpenXR.OpenXRLoader",
            (object)BuildTargetGroup.Android,
        });
        if (!assigned)
            Debug.Log("[Conversational AI] OpenXR loader was already assigned for Android.");

        Type settingsType = FindType("UnityEngine.XR.OpenXR.OpenXRSettings", "Unity.XR.OpenXR");
        object settings = settingsType.GetMethod(
            "GetSettingsForBuildTargetGroup", BindingFlags.Public | BindingFlags.Static
        ).Invoke(null, new object[] { BuildTargetGroup.Android });
        if (settings == null)
            throw new InvalidOperationException("Android OpenXR settings were not created.");

        MethodInfo getFeatures = settings.GetType().GetMethods()
            .First(method => method.Name == "GetFeatures" &&
                !method.IsGenericMethod && method.GetParameters().Length == 0);
        IEnumerable features = (IEnumerable)getFeatures.Invoke(settings, null);
        bool metaEnabled = false;
        foreach (object feature in features)
        {
            if (feature.GetType().FullName !=
                "UnityEngine.XR.OpenXR.Features.MetaQuestSupport.MetaQuestFeature") continue;
            PropertyInfo enabled = feature.GetType().GetProperty("enabled");
            enabled.SetValue(feature, true);
            EditorUtility.SetDirty((Object)feature);
            metaEnabled = true;
        }
        if (!metaEnabled)
            throw new InvalidOperationException("Meta Quest Support feature was not found in Android OpenXR settings.");
        EditorUtility.SetDirty((Object)settings);
    }

    private static void ConfigureBuildScenes()
    {
        string[] required = { SourceScene, QuestScene };
        EditorBuildSettingsScene[] existing = EditorBuildSettings.scenes;
        EditorBuildSettings.scenes = existing
            .Concat(required.Where(path => existing.All(scene => scene.path != path))
                .Select(path => new EditorBuildSettingsScene(path, true)))
            .Select(scene => new EditorBuildSettingsScene(
                scene.path,
                scene.path == QuestScene || scene.path == SourceScene ? true : scene.enabled
            ))
            .ToArray();
    }

    private static void CreateTrackedAnchor(Transform parent, string name, XRNode node)
    {
        var anchor = new GameObject(name);
        anchor.transform.SetParent(parent, false);
        anchor.AddComponent<OpenXRNodePoseDriver>().node = node;
    }

    private static GameObject CreatePrimitive(
        Transform parent, PrimitiveType type, string name,
        Vector3 position, Vector3 scale, Material material)
    {
        GameObject item = GameObject.CreatePrimitive(type);
        item.name = name;
        item.transform.SetParent(parent, false);
        item.transform.localPosition = position;
        item.transform.localScale = scale;
        item.GetComponent<Renderer>().sharedMaterial = material;
        return item;
    }

    private static Light CreateLight(
        Transform parent, string name, LightType type, Vector3 position,
        Color color, float intensity, float range)
    {
        var item = new GameObject(name);
        item.transform.SetParent(parent, false);
        item.transform.localPosition = position;
        Light light = item.AddComponent<Light>();
        light.type = type;
        light.color = color;
        light.intensity = intensity;
        light.range = range;
        light.shadows = LightShadows.Soft;
        return light;
    }

    private static Material GetOrCreateMaterial(string path, Color color, float emission)
    {
        EnsureFolder("Assets", "Materials");
        Material material = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (material == null)
        {
            material = new Material(Shader.Find("Standard"));
            AssetDatabase.CreateAsset(material, path);
        }
        material.color = color;
        material.SetColor("_Color", color);
        if (emission > 0f)
        {
            material.EnableKeyword("_EMISSION");
            material.SetColor("_EmissionColor", color * emission);
        }
        EditorUtility.SetDirty(material);
        return material;
    }

    private static void EnsureFolder(string parent, string child)
    {
        string path = $"{parent}/{child}";
        if (!AssetDatabase.IsValidFolder(path)) AssetDatabase.CreateFolder(parent, child);
    }

    private static Transform FindTransform(Scene scene, string name) => scene.GetRootGameObjects()
        .SelectMany(root => root.GetComponentsInChildren<Transform>(true))
        .FirstOrDefault(item => item.name == name);

    private static GameObject FindSceneObject(Scene scene, string name)
    {
        Transform found = FindTransform(scene, name);
        return found != null ? found.gameObject : null;
    }

    private static Type FindType(string fullName, string assemblyName) =>
        Type.GetType($"{fullName}, {assemblyName}");

    private static void SetProperty(object instance, string name, object value)
    {
        PropertyInfo property = instance.GetType().GetProperty(name);
        if (property != null && property.CanWrite) property.SetValue(instance, value);
    }

    private static void SetEnumProperty(object instance, string name, string enumValue)
    {
        PropertyInfo property = instance.GetType().GetProperty(name);
        if (property == null || !property.CanWrite || !property.PropertyType.IsEnum) return;
        property.SetValue(instance, Enum.Parse(property.PropertyType, enumValue));
    }
}
