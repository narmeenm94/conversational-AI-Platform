using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

/// <summary>
/// Drives body state, natural blinks, fallback jaw motion, and server emotion
/// cues. For production visemes, install uLipSync on the same AudioSource and
/// disable driveMouthFromVolume.
/// </summary>
public class AvatarController : MonoBehaviour
{
    [Header("Face Rig")]
    public SkinnedMeshRenderer[] faceMeshes;
    public bool autoDiscoverFaceMeshes = true;
    public bool driveMouthFromVolume = true;
    public string fallbackMouthBlendShape = "jawOpen";
    [Range(1f, 30f)] public float mouthSensitivity = 8f;
    [Range(1f, 40f)] public float smoothing = 18f;
    [Tooltip("Time allowed for uLipSync to fill its first analysis window before reporting a real signal failure.")]
    [Range(0.4f, 3f)] public float lipSyncStartupGraceSeconds = 1.25f;
    [Tooltip("How recently uLipSync must have reported audible input before the jaw fallback takes over.")]
    [Range(0.2f, 1.5f)] public float lipSyncSignalTimeoutSeconds = 0.65f;

    [Header("Natural Reactions")]
    public bool proceduralBlink = true;
    public Vector2 blinkInterval = new Vector2(2.5f, 6f);
    [Range(0.06f, 0.3f)] public float blinkDuration = 0.13f;
    [Range(0.2f, 8f)] public float expressionHoldSeconds = 2.5f;

    [Header("Animation")]
    public Animator animator;
    public string idleState = "Idle";
    public string listeningState = "Listening";
    public string thinkingState = "Thinking";
    public string rememberingState = "Remembering";
    public string searchingState = "Searching";
    public string speakingState = "Talking";
    public string walkingState = "Walking";
    [Range(0f, 1.5f)] public float animationBlendSeconds = 0.55f;
    [Tooltip("Minimum time a non-idle body state remains active before returning to Idle.")]
    [Range(0f, 4f)] public float minimumStateHoldSeconds = 1.15f;

    [Header("References")]
    public AudioStreamPlayer audioPlayer;

    private readonly Dictionary<string, List<BlendTarget>> _targets =
        new Dictionary<string, List<BlendTarget>>();
    private readonly Dictionary<string, float> _expressionTargets =
        new Dictionary<string, float>();
    private float _mouthWeight;
    private float _nextBlink;
    private float _blinkStart = -1f;
    private float _expressionUntil;
    private global::uLipSync.uLipSync _lipSync;
    private global::uLipSync.uLipSyncAudioSource _lipSyncAudioSource;
    private global::uLipSync.uLipSyncBlendShape _lipSyncBlendShape;
    private float _lastLipSyncSignalTime = -100f;
    private float _playbackStartedAt = -100f;
    private bool _playbackWasActive;
    private bool _reportedLipSyncFallback;
    private static readonly int IsSpeakingHash = Animator.StringToHash("IsSpeaking");
    private static readonly int IsListeningHash = Animator.StringToHash("IsListening");
    private readonly HashSet<int> _reportedMissingStates = new HashSet<int>();
    private string _semanticState = "idle";
    private string _activeAnimatorStateName = "";
    private string _pendingSemanticState = "";
    private float _semanticStateEnteredAt = -100f;
    private bool _walking;
    private bool _speaking;
    private bool _listening;

    public string CurrentSemanticState => _semanticState;
    public bool IsSpeaking => _speaking;
    public bool IsListening => _listening;

    private struct BlendTarget
    {
        public SkinnedMeshRenderer renderer;
        public int index;
        public float maxWeight;
    }

    void Awake()
    {
        if (autoDiscoverFaceMeshes || faceMeshes == null || faceMeshes.Length == 0)
        {
            var all = GetComponentsInChildren<SkinnedMeshRenderer>(true);
            var usable = new List<SkinnedMeshRenderer>();
            foreach (var renderer in all)
                if (renderer.sharedMesh != null && renderer.sharedMesh.blendShapeCount > 0)
                    usable.Add(renderer);
            faceMeshes = usable.ToArray();
        }
        BuildBlendShapeCache();
        BindLipSyncAtRuntime();
        NormalizeLipSyncBlendShapeRange();
        if (animator != null)
            animator.applyRootMotion = false;
        ScheduleBlink();
    }

    private void BindLipSyncAtRuntime()
    {
        _lipSync = GetComponent<global::uLipSync.uLipSync>();
        _lipSyncBlendShape = GetComponent<global::uLipSync.uLipSyncBlendShape>();
        if (_lipSync == null || _lipSyncBlendShape == null ||
            _lipSync.profile == null || _lipSyncBlendShape.skinnedMeshRenderer == null)
        {
            driveMouthFromVolume = true;
            Debug.LogWarning(
                "[Avatar] uLipSync is incomplete; using volume-driven jaw fallback."
            );
            return;
        }

        // Route the AudioSource filter through uLipSync's official proxy. This
        // is more deterministic for a procedural streaming AudioClip than
        // relying on component callback order on the avatar root.
        _lipSyncAudioSource = GetComponent<global::uLipSync.uLipSyncAudioSource>();
        if (_lipSyncAudioSource == null)
            _lipSyncAudioSource = gameObject.AddComponent<global::uLipSync.uLipSyncAudioSource>();
        _lipSync.audioSourceProxy = _lipSyncAudioSource;

        // UnityEvent.AddListener calls made by the scene setup tool are not
        // persistent when the scene is saved. Always wire the analyser to its
        // face driver at runtime so streaming speech produces visemes.
        _lipSync.onLipSyncUpdate.RemoveListener(_lipSyncBlendShape.OnLipSyncUpdate);
        _lipSync.onLipSyncUpdate.RemoveListener(HandleLipSyncSignal);
        _lipSync.onLipSyncUpdate.AddListener(_lipSyncBlendShape.OnLipSyncUpdate);
        _lipSync.onLipSyncUpdate.AddListener(HandleLipSyncSignal);
        driveMouthFromVolume = false;
        Debug.Log("[Avatar] Runtime uLipSync listener connected.");
    }

    private void HandleLipSyncSignal(global::uLipSync.LipSyncInfo info)
    {
        if (info.rawVolume > 1e-5f)
            _lastLipSyncSignalTime = Time.unscaledTime;
    }

    private void BuildBlendShapeCache()
    {
        _targets.Clear();
        if (faceMeshes == null) return;
        foreach (var renderer in faceMeshes)
        {
            if (renderer == null || renderer.sharedMesh == null) continue;
            for (int i = 0; i < renderer.sharedMesh.blendShapeCount; i++)
            {
                string name = renderer.sharedMesh.GetBlendShapeName(i);
                if (!_targets.TryGetValue(name, out var list))
                {
                    list = new List<BlendTarget>();
                    _targets[name] = list;
                }
                int frameCount = renderer.sharedMesh.GetBlendShapeFrameCount(i);
                float maxWeight = frameCount > 0
                    ? Mathf.Abs(renderer.sharedMesh.GetBlendShapeFrameWeight(i, frameCount - 1))
                    : 100f;
                if (maxWeight < Mathf.Epsilon) maxWeight = 100f;
                list.Add(new BlendTarget {
                    renderer = renderer,
                    index = i,
                    maxWeight = maxWeight,
                });
            }
        }
    }

    public void SetSpeaking(bool value)
    {
        _speaking = value;
        SetAnimatorBool(IsSpeakingHash, "IsSpeaking", value);
        if (value)
        {
            SetAnimatorBool(IsListeningHash, "IsListening", false);
            SetSemanticState("speaking");
        }
        else if (!_walking)
            SetSemanticState(_listening ? "listening" : "idle");
    }

    public void SetListening(bool value)
    {
        _listening = value;
        SetAnimatorBool(IsListeningHash, "IsListening", value);
        if (value && !_speaking) SetSemanticState("listening");
        else if (!_walking && _semanticState == "listening") SetSemanticState("idle");
    }

    public void ConfigureAnimationStates(CharacterPlatformClient.CharacterAnimations map)
    {
        if (map == null) return;
        idleState = map.idle;
        listeningState = map.listening;
        thinkingState = map.thinking;
        rememberingState = map.remembering;
        searchingState = map.searching;
        speakingState = map.speaking;
        walkingState = map.walking;
        animationBlendSeconds = Mathf.Clamp(map.blend_seconds, 0f, 1.5f);
        _reportedMissingStates.Clear();
        SetSemanticState("idle", 0f, true);
    }

    public void SetSemanticState(string state, float blendSeconds = -1f, bool force = false)
    {
        string normalized = string.IsNullOrWhiteSpace(state)
            ? "idle"
            : state.Trim().ToLowerInvariant();
        if (!force && normalized == _semanticState) return;
        if (_walking && normalized != "walking") return;
        if (_speaking && normalized != "speaking" && normalized != "talking") return;

        // Let expressive states finish a readable beat before settling back
        // to neutral. Listening, speaking, and cognitive events still start
        // immediately so the body remains responsive to the conversation.
        if (!force && normalized == "idle" && _semanticState != "idle" &&
            Time.unscaledTime - _semanticStateEnteredAt < minimumStateHoldSeconds)
        {
            _pendingSemanticState = normalized;
            return;
        }

        ApplySemanticState(normalized, blendSeconds, force);
    }

    private void ApplySemanticState(string normalized, float blendSeconds, bool force)
    {
        string animatorState = AnimationName(normalized);
        _semanticState = normalized;
        _semanticStateEnteredAt = Time.unscaledTime;
        _pendingSemanticState = "";
        if (string.IsNullOrWhiteSpace(animatorState) || animator == null ||
            !animator.isActiveAndEnabled || animator.runtimeAnimatorController == null)
            return;

        // Several semantic states intentionally share one native Avaturn
        // clip. Do not restart that clip just because the conversation label
        // changed; this is what previously made motions look cut short.
        if (!force && animatorState == _activeAnimatorStateName)
            return;

        int shortHash = Animator.StringToHash(animatorState);
        int fullHash = Animator.StringToHash($"{animator.GetLayerName(0)}.{animatorState}");
        int hash = animator.HasState(0, fullHash) ? fullHash : shortHash;
        if (!animator.HasState(0, hash))
        {
            if (_reportedMissingStates.Add(hash))
                Debug.LogWarning($"[Avatar] Animator state '{animatorState}' is not configured; using parameter fallback.");
            return;
        }
        float blend = blendSeconds >= 0f ? blendSeconds : animationBlendSeconds;
        animator.CrossFadeInFixedTime(hash, Mathf.Clamp(blend, 0f, 1.5f), 0);
        _activeAnimatorStateName = animatorState;
    }

    public void SetWalking(bool walking)
    {
        _walking = walking;
        SetSemanticState(walking ? "walking" : "idle", -1f, true);
    }

    private string AnimationName(string state)
    {
        switch (state)
        {
            case "listening": return listeningState;
            case "thinking": return thinkingState;
            case "remembering": return rememberingState;
            case "searching": return searchingState;
            case "speaking":
            case "talking": return speakingState;
            case "walking": return walkingState;
            default: return idleState;
        }
    }

    public void SetEmotion(string emotion, float intensity = 0.65f)
    {
        _expressionTargets.Clear();
        float weight = Mathf.Clamp01(intensity) * 100f;
        switch ((emotion ?? "neutral").ToLowerInvariant())
        {
            case "happy":
                AddExpression("mouthSmileLeft", weight);
                AddExpression("mouthSmileRight", weight);
                AddExpression("cheekSquintLeft", weight * 0.45f);
                AddExpression("cheekSquintRight", weight * 0.45f);
                break;
            case "surprised":
                AddExpression("browInnerUp", weight);
                AddExpression("eyeWideLeft", weight * 0.65f);
                AddExpression("eyeWideRight", weight * 0.65f);
                break;
            case "sad":
                AddExpression("browInnerUp", weight * 0.7f);
                AddExpression("mouthFrownLeft", weight * 0.65f);
                AddExpression("mouthFrownRight", weight * 0.65f);
                break;
            case "annoyed":
            case "angry":
                AddExpression("browDownLeft", weight * 0.75f);
                AddExpression("browDownRight", weight * 0.75f);
                AddExpression("noseSneerLeft", weight * 0.35f);
                AddExpression("noseSneerRight", weight * 0.35f);
                break;
        }
        _expressionUntil = Time.unscaledTime + expressionHoldSeconds;
    }

    public void ClearEmotion()
    {
        _expressionTargets.Clear();
        _expressionUntil = 0f;
    }

    void Update()
    {
        if (!string.IsNullOrEmpty(_pendingSemanticState) &&
            Time.unscaledTime - _semanticStateEnteredAt >= minimumStateHoldSeconds)
            ApplySemanticState(_pendingSemanticState, -1f, false);
        UpdateMouth();
        UpdateExpressions();
        UpdateBlink();
    }

    private void UpdateMouth()
    {
        bool playbackActive = audioPlayer != null && audioPlayer.IsPlaying;
        float now = Time.unscaledTime;
        float volume = audioPlayer != null ? audioPlayer.GetCurrentVolume() : 0f;
        bool audibleNow = volume > 1e-5f;
        if (playbackActive && !_playbackWasActive)
        {
            _playbackStartedAt = now;
            _reportedLipSyncFallback = false;
        }
        _playbackWasActive = playbackActive;

        bool lipSyncConfigured = !driveMouthFromVolume && _lipSync != null &&
            _lipSyncBlendShape != null && _lipSyncAudioSource != null;
        bool lipSyncSignalCurrent = lipSyncConfigured &&
            now - _lastLipSyncSignalTime <= lipSyncSignalTimeoutSeconds;
        bool startupGraceExpired = now - _playbackStartedAt >= lipSyncStartupGraceSeconds;
        bool analyserHasNoSignal = lipSyncConfigured && playbackActive && audibleNow &&
            startupGraceExpired && !lipSyncSignalCurrent;
        // Keep the mouth responsive during uLipSync's first analysis window,
        // then relinquish the jaw as soon as real viseme data arrives.
        bool useFallback = driveMouthFromVolume ||
            (lipSyncConfigured && playbackActive && audibleNow && !lipSyncSignalCurrent);

        if (analyserHasNoSignal && !_reportedLipSyncFallback)
        {
            _reportedLipSyncFallback = true;
            Debug.LogWarning(
                "[Avatar] uLipSync has no audio signal; using volume-driven jaw fallback."
            );
        }
        else if (!playbackActive)
        {
            _reportedLipSyncFallback = false;
            _playbackStartedAt = -100f;
        }

        float target = 0f;
        if (useFallback && playbackActive)
            target = Mathf.Clamp01(volume * mouthSensitivity) * 100f;
        _mouthWeight = Mathf.Lerp(_mouthWeight, target, Time.deltaTime * smoothing);
        SetBlendShape(fallbackMouthBlendShape, _mouthWeight);
    }

    private void UpdateExpressions()
    {
        if (_expressionUntil > 0f && Time.unscaledTime > _expressionUntil)
            ClearEmotion();

        string[] names = {
            "mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft", "cheekSquintRight",
            "browInnerUp", "eyeWideLeft", "eyeWideRight", "mouthFrownLeft", "mouthFrownRight",
            "browDownLeft", "browDownRight", "noseSneerLeft", "noseSneerRight"
        };
        foreach (string name in names)
        {
                float normalizedTarget = _expressionTargets.TryGetValue(name, out float value)
                    ? value
                    : 0f;
                if (!_targets.TryGetValue(name, out var blendTargets)) continue;
                foreach (var blend in blendTargets)
                {
                    float current = blend.renderer.GetBlendShapeWeight(blend.index);
                    float target = Mathf.Clamp01(normalizedTarget / 100f) * blend.maxWeight;
                    blend.renderer.SetBlendShapeWeight(
                    blend.index,
                    Mathf.Lerp(current, target, Time.deltaTime * smoothing * 0.35f)
                );
            }
        }
    }

    private void UpdateBlink()
    {
        if (!proceduralBlink) return;
        if (_blinkStart < 0f && Time.unscaledTime >= _nextBlink)
            _blinkStart = Time.unscaledTime;
        if (_blinkStart < 0f) return;

        float phase = (Time.unscaledTime - _blinkStart) / blinkDuration;
        float weight = Mathf.Sin(Mathf.Clamp01(phase) * Mathf.PI) * 100f;
        SetBlendShape("eyeBlinkLeft", weight);
        SetBlendShape("eyeBlinkRight", weight);
        if (phase >= 1f)
        {
            _blinkStart = -1f;
            ScheduleBlink();
        }
    }

    private void ScheduleBlink() =>
        _nextBlink = Time.unscaledTime + Random.Range(blinkInterval.x, blinkInterval.y);

    private void AddExpression(string name, float weight) =>
        _expressionTargets[name] = weight;

    private void SetBlendShape(string name, float weight)
    {
        if (!_targets.TryGetValue(name, out var list)) return;
        foreach (var target in list)
            target.renderer.SetBlendShapeWeight(
                target.index,
                Mathf.Clamp01(weight / 100f) * target.maxWeight
            );
    }

    private void SetAnimatorBool(int hash, string name, bool value)
    {
        if (animator == null || !animator.isActiveAndEnabled ||
            animator.runtimeAnimatorController == null)
            return;
        foreach (var parameter in animator.parameters)
        {
            if (parameter.name == name && parameter.type == AnimatorControllerParameterType.Bool)
            {
                animator.SetBool(hash, value);
                return;
            }
        }
    }

    private void NormalizeLipSyncBlendShapeRange()
    {
        // Avoid a hard dependency on uLipSync so the core scripts still compile
        // when that optional package is absent. glTFast imports morph frames at
        // weight 1, while FBX commonly uses 100.
        foreach (MonoBehaviour behaviour in GetComponents<MonoBehaviour>())
        {
            if (behaviour == null ||
                behaviour.GetType().FullName != "uLipSync.uLipSyncBlendShape")
                continue;

            FieldInfo rendererField = behaviour.GetType().GetField("skinnedMeshRenderer");
            FieldInfo rangeField = behaviour.GetType().GetField("maxBlendShapeValue");
            var renderer = rendererField?.GetValue(behaviour) as SkinnedMeshRenderer;
            if (renderer == null || renderer.sharedMesh == null || rangeField == null) continue;

            int index = renderer.sharedMesh.GetBlendShapeIndex("viseme_aa");
            if (index < 0) continue;
            int frameCount = renderer.sharedMesh.GetBlendShapeFrameCount(index);
            if (frameCount <= 0) continue;
            float maxWeight = Mathf.Abs(
                renderer.sharedMesh.GetBlendShapeFrameWeight(index, frameCount - 1)
            );
            if (maxWeight < Mathf.Epsilon) continue;
            rangeField.SetValue(behaviour, maxWeight);
            Debug.Log($"[Avatar] Lip-sync morph range normalized to {maxWeight}.");
        }
    }

    void OnDestroy()
    {
        if (_lipSync != null && _lipSyncBlendShape != null)
        {
            _lipSync.onLipSyncUpdate.RemoveListener(_lipSyncBlendShape.OnLipSyncUpdate);
            _lipSync.onLipSyncUpdate.RemoveListener(HandleLipSyncSignal);
        }
    }
}
