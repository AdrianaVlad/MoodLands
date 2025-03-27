using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PickEmotion : MonoBehaviour
{
    public bool picked;
    public InterpretWebcam interpretWebcam;
    public void SetHappy()
    {
        interpretWebcam.emotion = "happy";
        picked = true;
    }
    public void SetNeutral()
    {
        interpretWebcam.emotion = "neutral";
        picked = true;
    }

    public void SetAngry()
    {
        interpretWebcam.emotion = "angry";
        picked = true;
    }
    public void SetSad()
    {
        interpretWebcam.emotion = "sad";
        picked = true;
    }
    public void SetSurprise()
    {
        interpretWebcam.emotion = "surprise";
        picked = true;
    }
    public void SetFear()
    {
        interpretWebcam.emotion = "fear";
        picked = true;
    }
    public void SetDisgust()
    {
        interpretWebcam.emotion = "disgust";
        picked = true;
    }
    public void SetContempt()
    {
        interpretWebcam.emotion = "contempt";
        picked = true;
    }
}
