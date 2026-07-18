using UnityEngine;

/// <summary>Keeps a prop rigidly attached to an animated Avaturn hand bone.</summary>
[ExecuteAlways]
public class HandPropAttachment : MonoBehaviour
{
    public Transform hand;
    public string handBoneName = "RightHand";
    public Vector3 localPosition = new Vector3(0.035f, -0.015f, 0.02f);
    public Vector3 localEulerAngles = Vector3.zero;
    public Vector3 localScale = Vector3.one * 0.012f;
    public bool lockPose = true;

    void OnEnable()
    {
        ResolveHand();
        ApplyAttachment();
    }

    void LateUpdate()
    {
        if (!lockPose) return;
        ResolveHand();
        ApplyAttachment();
    }

    public void CaptureCurrentLocalPose()
    {
        localPosition = transform.localPosition;
        localEulerAngles = transform.localEulerAngles;
        localScale = transform.localScale;
    }

    private void ResolveHand()
    {
        if (hand != null) return;
        foreach (Transform candidate in FindObjectsByType<Transform>())
            if (candidate.name == handBoneName)
            {
                hand = candidate;
                return;
            }
    }

    private void ApplyAttachment()
    {
        if (hand == null) return;
        if (transform.parent != hand) transform.SetParent(hand, false);
        transform.localPosition = localPosition;
        transform.localRotation = Quaternion.Euler(localEulerAngles);
        transform.localScale = localScale;
    }
}
