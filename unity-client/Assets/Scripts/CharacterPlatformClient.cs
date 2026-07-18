using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Lightweight Unity/Quest bridge to the local character authoring platform.
/// Language, RAG and speech models stay on the PC; Quest receives only the
/// active character metadata and streamed audio.
/// </summary>
public class CharacterPlatformClient : MonoBehaviour
{
    [Serializable]
    public class CharacterVoice
    {
        public string reference_audio;
        public float temperature;
        public string default_emotion;
        public float emotion_intensity;
    }

    [Serializable]
    public class CharacterAnimations
    {
        public string idle = "Idle";
        public string listening = "Listening";
        public string thinking = "Thinking";
        public string remembering = "Remembering";
        public string searching = "Searching";
        public string speaking = "Talking";
        public string walking = "Walking";
        public float blend_seconds = 0.22f;
    }

    [Serializable]
    public class CharacterDefinition
    {
        public string id;
        public string name;
        public string description;
        public string language;
        public string backstory;
        public string speaking_style;
        public string avatar_asset;
        public CharacterVoice voice;
        public CharacterAnimations animations;
    }

    [Serializable]
    private class CharacterCatalog
    {
        public string active_id;
        public CharacterDefinition[] characters;
    }

    [Serializable]
    private class PlatformStatus
    {
        public string active_character_id;
        public string active_character_name;
        public string language;
        public string tts_backend;
        public int knowledge_chunks;
    }

    [Serializable]
    public class AvatarBinding
    {
        public string characterId;
        public GameObject avatarRoot;
    }

    [Header("Platform")]
    [Tooltip("localhost in the Editor; use the PC LAN address on Quest 3.")]
    public string serverAddress = "localhost";
    public int controlPort = 8766;
    public bool pollActiveCharacter = true;
    [Range(0.5f, 10f)] public float pollIntervalSeconds = 2f;

    [Header("Optional startup selection")]
    public bool activateCharacterOnStart = false;
    public string selectedCharacterId = "alex";

    [Header("Unity avatar mappings")]
    public AvatarBinding[] avatarBindings;

    public string ActiveCharacterId { get; private set; }
    public CharacterDefinition ActiveCharacter { get; private set; }
    public CharacterDefinition[] Characters { get; private set; } = Array.Empty<CharacterDefinition>();
    public event Action<CharacterDefinition> OnCharacterChanged;
    private string _activeSignature = "";

    private string BaseUrl => $"http://{serverAddress}:{controlPort}";

    IEnumerator Start()
    {
        if (activateCharacterOnStart && !string.IsNullOrWhiteSpace(selectedCharacterId))
            yield return ActivateCharacterRoutine(selectedCharacterId);
        else
            yield return RefreshCatalogRoutine();

        while (pollActiveCharacter)
        {
            yield return new WaitForSecondsRealtime(pollIntervalSeconds);
            yield return PollStatusRoutine();
        }
    }

    public void ActivateSelectedCharacter() => ActivateCharacter(selectedCharacterId);

    public void ActivateCharacter(string characterId)
    {
        if (!string.IsNullOrWhiteSpace(characterId))
            StartCoroutine(ActivateCharacterRoutine(characterId));
    }

    public void RefreshCatalog() => StartCoroutine(RefreshCatalogRoutine());

    private IEnumerator ActivateCharacterRoutine(string characterId)
    {
        using var request = new UnityWebRequest(
            $"{BaseUrl}/api/characters/{UnityWebRequest.EscapeURL(characterId)}/activate",
            UnityWebRequest.kHttpVerbPOST
        );
        request.downloadHandler = new DownloadHandlerBuffer();
        yield return request.SendWebRequest();
        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogWarning($"[Characters] Activation failed: {request.error}");
            yield break;
        }
        CharacterDefinition definition = JsonUtility.FromJson<CharacterDefinition>(
            request.downloadHandler.text
        );
        ApplyCharacter(definition);
        yield return RefreshCatalogRoutine();
    }

    private IEnumerator RefreshCatalogRoutine()
    {
        using var request = UnityWebRequest.Get($"{BaseUrl}/api/characters");
        yield return request.SendWebRequest();
        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogWarning($"[Characters] Catalog unavailable: {request.error}");
            yield break;
        }
        CharacterCatalog catalog = JsonUtility.FromJson<CharacterCatalog>(
            request.downloadHandler.text
        );
        Characters = catalog.characters ?? Array.Empty<CharacterDefinition>();
        ApplyCharacter(FindCharacter(catalog.active_id));
    }

    private IEnumerator PollStatusRoutine()
    {
        using var request = UnityWebRequest.Get($"{BaseUrl}/api/status");
        yield return request.SendWebRequest();
        if (request.result != UnityWebRequest.Result.Success) yield break;
        PlatformStatus status = JsonUtility.FromJson<PlatformStatus>(request.downloadHandler.text);
        // Refresh even when the id is unchanged: animation names, personality,
        // and voice controls are editable live in the platform UI.
        if (status != null)
            yield return RefreshCatalogRoutine();
    }

    private CharacterDefinition FindCharacter(string characterId)
    {
        foreach (CharacterDefinition definition in Characters)
            if (definition != null && definition.id == characterId)
                return definition;
        return null;
    }

    private void ApplyCharacter(CharacterDefinition definition)
    {
        if (definition == null) return;
        string signature = CharacterSignature(definition);
        bool changed = signature != _activeSignature;
        ActiveCharacterId = definition.id;
        ActiveCharacter = definition;
        selectedCharacterId = definition.id;
        if (!changed) return;
        _activeSignature = signature;
        if (avatarBindings != null)
            foreach (AvatarBinding binding in avatarBindings)
                if (binding != null && binding.avatarRoot != null)
                    binding.avatarRoot.SetActive(binding.characterId == definition.id);
        OnCharacterChanged?.Invoke(definition);
        Debug.Log(
            $"[Characters] Active: {definition.name} ({definition.id}), " +
            $"avatar={definition.avatar_asset}"
        );
    }

    private static string CharacterSignature(CharacterDefinition definition)
    {
        CharacterAnimations animation = definition.animations;
        return string.Join("|",
            definition.id ?? "",
            definition.name ?? "",
            definition.avatar_asset ?? "",
            animation?.idle ?? "",
            animation?.listening ?? "",
            animation?.thinking ?? "",
            animation?.remembering ?? "",
            animation?.searching ?? "",
            animation?.speaking ?? "",
            animation?.walking ?? "",
            animation != null ? animation.blend_seconds.ToString("R") : ""
        );
    }
}
