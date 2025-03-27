using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class AffectBot : MonoBehaviour
{
    public InterpretWebcam cam;
    private BotMovement movements;
    private void Awake()
    {
        movements = GetComponent<BotMovement>();
    }

    private void OnEnable()
    {
        if (cam.emotion == "neutral" || cam.emotion == "sad")
        {
            movements.IsAttacking = false;
        }
        else if (cam.emotion == "happy" || cam.emotion == "contempt")
        {
            movements.IsMoving = true;
            movements.IsAttacking = false;
        }
        else if (cam.emotion == "fear" || cam.emotion == "surprise")
        {
            movements.IsMoving = false;
        }
        else
        { 
            movements.IsAttacking = true;
            movements.IsMoving = false;
        }
        enabled = false;
    }
}
