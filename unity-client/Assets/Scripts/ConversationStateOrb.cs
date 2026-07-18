using UnityEngine;

/// <summary>Beautiful, diegetic conversation-state feedback above the avatar.</summary>
public class ConversationStateOrb : MonoBehaviour
{
    public Renderer orbRenderer;
    public Light glowLight;
    public TextMesh statusText;
    public Transform followTarget;
    public Vector3 followOffset = new Vector3(0f, 0.32f, 0f);
    public ConversationManager conversation;

    [Header("Metropolia Palette")]
    public Color disengagedColor = new Color(0.27f, 0.29f, 0.32f, 1f);
    public Color readyColor = new Color(1f, 0.31f, 0f, 1f);
    public Color listeningColor = new Color(0.05f, 0.78f, 0.92f, 1f);
    public Color respondingColor = new Color(1f, 0.12f, 0.03f, 1f);
    [Range(1f, 8f)] public float emissionIntensity = 3.2f;

    private Material _material;
    private bool _gazeEngaged;
    private Vector3 _baseScale;

    void Awake()
    {
        if (conversation == null) conversation = FindAnyObjectByType<ConversationManager>();
        if (orbRenderer == null) orbRenderer = GetComponentInChildren<Renderer>();
        if (glowLight == null) glowLight = GetComponentInChildren<Light>();
        if (statusText == null) statusText = GetComponentInChildren<TextMesh>();
        if (orbRenderer != null)
        {
            _material = orbRenderer.material;
            _material.EnableKeyword("_EMISSION");
        }
        _baseScale = transform.localScale;
    }

    public void SetGazeEngaged(bool engaged) => _gazeEngaged = engaged;

    void LateUpdate()
    {
        if (conversation != null) _gazeEngaged = conversation.GazeEngaged;
        if (followTarget != null)
            transform.position = Vector3.Lerp(
                transform.position,
                followTarget.position + followOffset,
                1f - Mathf.Exp(-10f * Time.unscaledDeltaTime)
            );

        Color targetColor;
        string label;
        float pulseSpeed;
        if (conversation != null && conversation.AvatarSpeaking)
        {
            targetColor = respondingColor;
            label = "RESPONDING";
            pulseSpeed = 6f;
        }
        else if (conversation != null && conversation.UserSpeaking)
        {
            targetColor = listeningColor;
            label = "LISTENING";
            pulseSpeed = 4f;
        }
        else if (_gazeEngaged)
        {
            targetColor = readyColor;
            label = "READY";
            pulseSpeed = 2.2f;
        }
        else
        {
            targetColor = disengagedColor;
            label = "LOOK TO TALK";
            pulseSpeed = 1.2f;
        }

        float pulse = 1f + Mathf.Sin(Time.unscaledTime * pulseSpeed) * 0.065f;
        transform.localScale = Vector3.Lerp(
            transform.localScale,
            _baseScale * pulse,
            1f - Mathf.Exp(-8f * Time.unscaledDeltaTime)
        );

        if (_material != null)
        {
            Color current = _material.HasProperty("_Color")
                ? _material.GetColor("_Color")
                : targetColor;
            Color smooth = Color.Lerp(
                current, targetColor, 1f - Mathf.Exp(-7f * Time.unscaledDeltaTime)
            );
            if (_material.HasProperty("_Color")) _material.SetColor("_Color", smooth);
            if (_material.HasProperty("_BaseColor")) _material.SetColor("_BaseColor", smooth);
            if (_material.HasProperty("_EmissionColor"))
                _material.SetColor("_EmissionColor", smooth * emissionIntensity);
            if (glowLight != null) glowLight.color = smooth;
        }

        if (glowLight != null)
            glowLight.intensity = 1.15f + Mathf.Sin(Time.unscaledTime * pulseSpeed) * 0.18f;
        if (statusText != null)
        {
            statusText.text = label;
            statusText.color = targetColor;
            if (Camera.main != null)
                statusText.transform.rotation = Quaternion.LookRotation(
                    statusText.transform.position - Camera.main.transform.position
                );
        }
    }

    void OnDestroy()
    {
        if (_material != null) Destroy(_material);
    }
}
