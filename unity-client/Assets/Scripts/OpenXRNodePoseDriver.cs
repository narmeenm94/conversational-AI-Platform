using UnityEngine;
using UnityEngine.XR;

/// <summary>
/// Lightweight OpenXR pose driver for the HMD and controller anchors. It uses
/// Unity's XR input abstraction and does not require button actions.
/// </summary>
public class OpenXRNodePoseDriver : MonoBehaviour
{
    public XRNode node = XRNode.CenterEye;
    public bool trackPosition = true;
    public bool trackRotation = true;

    void OnEnable() => Application.onBeforeRender += ApplyPose;
    void OnDisable() => Application.onBeforeRender -= ApplyPose;
    void Update() => ApplyPose();

    private void ApplyPose()
    {
        InputDevice device = InputDevices.GetDeviceAtXRNode(node);
        if (!device.isValid) return;
        if (trackPosition && device.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 p))
            transform.localPosition = p;
        if (trackRotation && device.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion r))
            transform.localRotation = r;
    }
}
