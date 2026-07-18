using UnityEngine;

/// <summary>
/// Opens the conversational microphone gate after a short, forgiving head-gaze
/// dwell. This is continuous engagement rather than push-to-talk.
/// </summary>
public class GazeConversationTarget : MonoBehaviour
{
    [Header("References")]
    public Transform viewer;
    public Transform target;
    public ConversationManager conversation;
    public ConversationStateOrb indicator;

    [Header("Gaze Engagement")]
    [Range(3f, 30f)] public float engagementAngle = 13f;
    [Range(0.05f, 1.5f)] public float dwellSeconds = 0.28f;
    [Range(0.1f, 2f)] public float disengageGraceSeconds = 0.65f;
    [Range(0.5f, 8f)] public float maximumDistance = 4f;

    private float _lookDuration;
    private float _lookAwayDuration;
    private bool _engaged;

    public bool IsEngaged => _engaged;

    void Start()
    {
        ResolveReferences();
        SetEngaged(false, true);
    }

    void Update()
    {
        ResolveReferences();
        if (viewer == null || target == null || conversation == null) return;

        Vector3 toTarget = target.position - viewer.position;
        float distance = toTarget.magnitude;
        bool looking = distance <= maximumDistance && distance > 0.01f &&
            Vector3.Angle(viewer.forward, toTarget / distance) <= engagementAngle;

        if (looking)
        {
            _lookDuration += Time.unscaledDeltaTime;
            _lookAwayDuration = 0f;
            if (!_engaged && _lookDuration >= dwellSeconds)
                SetEngaged(true);
        }
        else
        {
            _lookDuration = 0f;
            _lookAwayDuration += Time.unscaledDeltaTime;
            if (_engaged && _lookAwayDuration >= disengageGraceSeconds)
                SetEngaged(false);
        }
    }

    private void ResolveReferences()
    {
        if (viewer == null && Camera.main != null) viewer = Camera.main.transform;
        if (conversation == null) conversation = FindAnyObjectByType<ConversationManager>();
        if (indicator == null) indicator = FindAnyObjectByType<ConversationStateOrb>();
    }

    private void SetEngaged(bool engaged, bool force = false)
    {
        if (!force && _engaged == engaged) return;
        _engaged = engaged;
        conversation?.SetGazeEngaged(engaged);
        indicator?.SetGazeEngaged(engaged);
    }
}
