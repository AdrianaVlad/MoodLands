using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using System.IO;

public class SaveManager : MonoBehaviour
{
    private string saveFilePath = "./saveData.json";
    private void OnApplicationQuit()
    {
        SaveGame();
    }

    private void SaveGame()
    {
        SaveData saveData = new SaveData();
        saveData.lastLoadedScene = SceneManager.GetActiveScene().name;
        saveData.force = PlayerPrefs.GetInt("force", 0);
        saveData.player = PlayerPrefs.GetString("player", "");
        saveData.difficulty = PlayerPrefs.GetString("difficulty", "");

        string jsonData = JsonUtility.ToJson(saveData);
        File.WriteAllText(saveFilePath, jsonData);
    }

 
}
