using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class AffectDragon : MonoBehaviour
{
    public InterpretWebcam cam;
    private DragonMovement movements;
    private void Awake()
    {
        movements = GetComponent<DragonMovement>();
    }

    private void OnEnable()
    {
        if (cam.emotion == "neutral" || cam.emotion == "sad")
        {
            movements.IsMoving = false;
            movements.IsAttacking = false;
        }
        else if (cam.emotion == "happy" || cam.emotion == "contempt")
        {
            movements.IsMoving = true;
        }
        else
        {
            movements.IsAttacking = true;
        }
        enabled = false;
    }
}
