using UnityEngine;

public class AvatarController : MonoBehaviour
{
    [Header("Lip Sync")]
    public SkinnedMeshRenderer faceMesh;
    public int mouthOpenBlendShapeIndex;
    public float sensitivity = 5f;
    public float smoothing = 15f;

    [Header("Animation")]
    public Animator animator;

    [Header("References")]
    public AudioStreamPlayer audioPlayer;

    private float _currentMouthValue;
    private static readonly int IsSpeaking = Animator.StringToHash("IsSpeaking");

    public void SetSpeaking(bool val)
    {
        if (animator != null)
            animator.SetBool(IsSpeaking, val);
    }

    void Update()
    {
        float targetMouth = 0f;

        if (audioPlayer != null && audioPlayer.IsPlaying)
        {
            float volume = audioPlayer.GetCurrentVolume();
            targetMouth = Mathf.Clamp01(volume * sensitivity) * 100f;
        }

        _currentMouthValue = Mathf.Lerp(_currentMouthValue, targetMouth, Time.deltaTime * smoothing);

        if (faceMesh != null)
            faceMesh.SetBlendShapeWeight(mouthOpenBlendShapeIndex, _currentMouthValue);
    }
}
