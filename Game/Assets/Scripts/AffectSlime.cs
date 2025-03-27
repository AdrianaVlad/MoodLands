using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AffectSlime : MonoBehaviour
{
    public InterpretWebcam cam;
    private SlimeMovement movements;
    private void Awake()
    {
        movements = GetComponent<SlimeMovement>();
    }

    private void OnEnable()
    {
        if (cam.emotion == "neutral" || cam.emotion == "sad")
        {
            movements.IsMoving = false;
            movements.IsJumping = false;
            movements.IsAttacking = false;
        }
        else if (cam.emotion == "happy" || cam.emotion == "contempt")
        {
            movements.IsMoving = true;
        }
        else if (cam.emotion == "fear" || cam.emotion == "surprise")
        {
            movements.IsJumping = true;
        }
        else
        {
            movements.IsAttacking = true;
        }
        enabled = false;
    }
}
