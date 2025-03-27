using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class DisplayEmotion : MonoBehaviour
{
    public InterpretWebcam cam;
    TextMeshProUGUI emotionInfo;
    public string enemyType;
    void Start()
    {
        emotionInfo = GetComponent<TextMeshProUGUI>();
    }
    void Update()
    {
        emotionInfo.text = "Emotion: " + cam.emotion + "\nAction: ";
        if (enemyType == "dragon")
        {
            if (cam.emotion == "neutral" || cam.emotion == "sad")
            {
                emotionInfo.text += "Idle";
            }
            else if (cam.emotion == "happy" || cam.emotion == "contempt")
            {
                emotionInfo.text += "Walk";
            }
            else
            {
                emotionInfo.text += "Attack";
            }
        }
        else if (enemyType == "slime")
        {
            if (cam.emotion == "neutral" || cam.emotion == "sad")
            {
                emotionInfo.text += "Idle";
            }
            else if (cam.emotion == "happy" || cam.emotion == "contempt")
            {
                emotionInfo.text += "Walk";
            }
            else if (cam.emotion == "fear" || cam.emotion == "surprise")
            {
                emotionInfo.text += "Jump";
            }
            else
            {
                emotionInfo.text += "Attack";
            }
        }
        else if(enemyType == "bot")
        {
            if (cam.emotion == "neutral" || cam.emotion == "sad")
            {
                emotionInfo.text += "StopAttack";
            }
            else if (cam.emotion == "happy" || cam.emotion == "contempt")
            {
                emotionInfo.text += "Walk";
            }
            else if (cam.emotion == "fear" || cam.emotion == "surprise")
            {
                emotionInfo.text += "StopWalk";
            }
            else
            {
                emotionInfo.text += "Attack";
            }
        }
    }
}
