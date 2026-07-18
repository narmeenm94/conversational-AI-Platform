using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

/// <summary>
/// Connects the imported conversational body clips to the state names exposed
/// by the character platform. State changes are driven by AvatarController, so
/// the controller deliberately contains no unconditional Animator transitions.
/// </summary>
public static class ConversationalAnimationSetup
{
    private const string ControllerPath = "Assets/Avatar.controller";
    private const string AutoSetupSessionKey =
        "ConversationalAI.AvatarAnimationsConfigured.v3";

    private static readonly Dictionary<string, string> StateClips =
        new Dictionary<string, string>
        {
            // Use only clips exported on the native Avaturn hierarchy. The
            // Mixamo FBXs require Humanoid retargeting and are intentionally
            // excluded until the animation layer is revisited.
            { "Idle", "Assets/Thinking model.glb" },
            { "Listening", "Assets/Waiting model.glb" },
            { "Talking", "Assets/Thinking model.glb" },
            { "Thinking", "Assets/Thinking model.glb" },
            { "Remembering", "Assets/Thinking model.glb" },
            { "Searching", "Assets/Waiting model.glb" },
        };

    [InitializeOnLoadMethod]
    private static void ScheduleAutomaticSetup()
    {
        if (!SessionState.GetBool(AutoSetupSessionKey, false))
            EditorApplication.delayCall += ConfigureWhenEditorIsReady;
    }

    private static void ConfigureWhenEditorIsReady()
    {
        if (EditorApplication.isCompiling || EditorApplication.isUpdating ||
            EditorApplication.isPlayingOrWillChangePlaymode)
        {
            EditorApplication.delayCall += ConfigureWhenEditorIsReady;
            return;
        }

        // Set this before reimporting: a model reimport can reload the script
        // domain, and SessionState survives that reload without creating a loop.
        SessionState.SetBool(AutoSetupSessionKey, true);
        try
        {
            ConfigureAvatarAnimations();
            ValidateController();
        }
        catch (Exception exception)
        {
            SessionState.SetBool(AutoSetupSessionKey, false);
            Debug.LogException(exception);
        }
    }

    [MenuItem("Tools/Conversational AI/Configure Avatar Animations")]
    public static void ConfigureAvatarAnimations()
    {
        foreach (string path in StateClips.Values.Distinct())
            ConfigureModelLoops(path);

        var controller = AssetDatabase.LoadAssetAtPath<AnimatorController>(ControllerPath);
        if (controller == null)
            controller = AnimatorController.CreateAnimatorControllerAtPath(ControllerPath);

        AnimatorStateMachine machine = controller.layers[0].stateMachine;
        foreach (ChildAnimatorState child in machine.states)
            foreach (AnimatorStateTransition transition in child.state.transitions.ToArray())
                child.state.RemoveTransition(transition);
        foreach (AnimatorStateTransition transition in machine.anyStateTransitions.ToArray())
            machine.RemoveAnyStateTransition(transition);

        AnimatorState idle = null;
        foreach (KeyValuePair<string, string> binding in StateClips)
        {
            AnimatorState state = FindOrAddState(machine, binding.Key);
            AnimationClip clip = FindAnimationClip(binding.Value);
            if (clip == null)
                throw new InvalidOperationException(
                    $"No animation clip found at {binding.Value} for state {binding.Key}."
                );
            state.motion = clip;
            state.writeDefaultValues = true;
            if (binding.Key == "Idle") idle = state;
        }

        machine.defaultState = idle ?? throw new InvalidOperationException("Idle state is missing.");
        EditorUtility.SetDirty(machine);
        EditorUtility.SetDirty(controller);
        AssetDatabase.SaveAssets();

        Debug.Log(
            "[Conversational AI] Avatar animations connected: Idle loops by default; " +
            "Listening, Thinking, Remembering, Searching, and Talking are selected " +
            "by live conversation events and crossfade back to Idle when complete."
        );
    }

    /// <summary>Batch-mode entry point used by the project verification command.</summary>
    public static void ConfigureAndValidateBatch()
    {
        ConfigureAvatarAnimations();
        ValidateController();
    }

    [MenuItem("Tools/Conversational AI/Validate Avatar Animations")]
    public static void ValidateController()
    {
        var controller = AssetDatabase.LoadAssetAtPath<AnimatorController>(ControllerPath);
        if (controller == null) throw new InvalidOperationException("Avatar.controller is missing.");
        AnimatorStateMachine machine = controller.layers[0].stateMachine;
        if (machine.defaultState == null || machine.defaultState.name != "Idle")
            throw new InvalidOperationException("Idle must be the default Animator state.");

        foreach (KeyValuePair<string, string> binding in StateClips)
        {
            AnimatorState state = machine.states
                .Select(child => child.state)
                .FirstOrDefault(candidate => candidate.name == binding.Key);
            if (state == null || state.motion == null)
                throw new InvalidOperationException(
                    $"Animator state {binding.Key} is missing its animation clip."
                );
            if (state.transitions.Length != 0)
                throw new InvalidOperationException(
                    $"Animator state {binding.Key} contains an automatic transition."
                );
        }
        if (machine.anyStateTransitions.Length != 0)
            throw new InvalidOperationException("Any State transitions must remain empty.");

        Debug.Log("[Conversational AI] Avatar Animator validation passed.");
    }

    private static AnimatorState FindOrAddState(AnimatorStateMachine machine, string name)
    {
        AnimatorState existing = machine.states
            .Select(child => child.state)
            .FirstOrDefault(state => state.name == name);
        return existing ?? machine.AddState(name);
    }

    private static AnimationClip FindAnimationClip(string assetPath)
    {
        return AssetDatabase.LoadAllAssetsAtPath(assetPath)
            .OfType<AnimationClip>()
            .Where(clip => !clip.name.StartsWith("__preview__", StringComparison.Ordinal))
            .OrderByDescending(clip => clip.length)
            .FirstOrDefault();
    }

    private static void ConfigureModelLoops(string assetPath)
    {
        var importer = AssetImporter.GetAtPath(assetPath) as ModelImporter;
        if (importer == null) return; // glTFast imports GLB animations as loops.

        ModelImporterClipAnimation[] clips = importer.defaultClipAnimations;
        if (clips == null || clips.Length == 0) return;
        foreach (ModelImporterClipAnimation clip in clips)
        {
            clip.loopTime = true;
            clip.loopPose = true;
            clip.lockRootRotation = true;
            clip.lockRootHeightY = true;
            clip.lockRootPositionXZ = true;
        }
        importer.clipAnimations = clips;
        importer.SaveAndReimport();
    }
}
