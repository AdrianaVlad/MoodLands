using System.Collections;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;

public class Timer : MonoBehaviour
{
    public float totalTime = 300f; // 5 minutes
    public TextMeshProUGUI timerText;
    private float lastReadTime = 0f;
    private float readInterval = 1f;

    void Start()
    {
        UpdateTimerDisplay();
    }

    void Update()
    {
        if (Time.realtimeSinceStartup - lastReadTime >= readInterval)
        {
            lastReadTime = Time.realtimeSinceStartup;
            totalTime -= 1f;
            if (totalTime <= 0)
                ReloadScene();
            UpdateTimerDisplay();
        }
    }

    void UpdateTimerDisplay()
    {
        int minutes = Mathf.FloorToInt(totalTime / 60);
        int seconds = Mathf.FloorToInt(totalTime % 60);
        timerText.text = string.Format("{0:00}:{1:00}", minutes, seconds);
    }

    void ReloadScene()
    {
        Scene scene = SceneManager.GetActiveScene();
        SceneManager.LoadScene(scene.name);
    }
}
