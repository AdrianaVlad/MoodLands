using Assets.Scripts;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Newtonsoft.Json;
using TMPro;

public class InterpretWebcam : MonoBehaviour
{
    private string path = "D:\\Informatica\\licenta\\GameExport\\predictions.txt";
    public string gender;
    public string ageGroup;
    public int ageGroupIndex;
    public string emotion;

    private PredictionData prediction = new PredictionData();

    private string[] genderHistory  = new string[10];
    private int[] ageGroupHistory = new int[10];
    private int historyIndex = 0;
    private bool paused = false;

    private float genderAverage, ageGroupAverage, hasGlassesAverage;
    private string[] ageMap = { "0-3", "4-7", "8-13", "14-20", "21-32", "33-43", "44-59", "60+" };

    private float lastReadTime = 0f;
    private float readInterval = 1f;
    private float timeLastSeen;

    // Start is called before the first frame update
    void Start()
    {
        if (File.Exists(path))
        {
            ReadWebcam();
            for (int i = 1; i < 10; i++)
            {
                genderHistory[i] = genderHistory[0];
                ageGroupHistory[i] = ageGroupHistory[0];
            }
            UpdateValues();
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (Time.realtimeSinceStartup - lastReadTime >= readInterval && File.Exists(path))
        {
            lastReadTime = Time.realtimeSinceStartup;
            ReadWebcam();
            if (!paused)
            {
                if (PlayerPrefs.GetInt("force") == 0)
                    UpdateValues();
                else
                {
                    if (PlayerPrefs.GetString("player") == "F")
                        gender = "Female";
                    else
                        gender = "Male";
                    if (PlayerPrefs.GetString("difficulty") == "easy")
                    {
                        ageGroup = ageMap[1];
                        ageGroupIndex = 1;
                    }
                    else
                    {
                        ageGroup = ageMap[4];
                        ageGroupIndex = 4;
                    }
                        
                }
            }
        }
    }

    private void UpdateValues()
    {
        genderAverage = 0;
        ageGroupAverage = 0;
        hasGlassesAverage = 0;

        for (int i = 0; i < 10; i++)
        {
            if (genderHistory[i] == "F")
                genderAverage += 1;
            ageGroupAverage += ageGroupHistory[i];
        }

        genderAverage /= 10;
        ageGroupAverage /= 10;
        hasGlassesAverage /= 10;

        if (genderAverage >= 0.5)
            gender = "Female";
        else 
            gender = "Male";

        ageGroupIndex = (int)Math.Round(ageGroupAverage);
        ageGroup = ageMap[ageGroupIndex];

        emotion = prediction.emotion;

    }

    void ReadWebcam()
    {
        string jsonContent = File.ReadAllText(path).Trim();
        if (jsonContent == "NO WEBCAM")
        {
            PlayerPrefs.Save();
            PlayerPrefs.SetInt("force", 1);
            return;
        }
        if (jsonContent.StartsWith("{\"gender\": null"))
        {
            if(!paused)
                timeLastSeen = Time.realtimeSinceStartup;
            if (Time.realtimeSinceStartup - timeLastSeen >= 3f)
            {
                Time.timeScale = 0f;
                PlayerPrefs.SetInt("paused", 1);
            }
                
            paused = true;
            return;
        }
        else if (paused)
        {
            if(PlayerPrefs.GetInt("ability") == 0)
                Time.timeScale = 1f;
            paused = false;
            PlayerPrefs.SetInt("paused", 0);
        }
        prediction = JsonConvert.DeserializeObject<PredictionData>(jsonContent);
        genderHistory[historyIndex] = prediction.gender;
        ageGroupHistory[historyIndex] = prediction.ageIndex;
        historyIndex = (historyIndex + 1) % 10;
    }

}
