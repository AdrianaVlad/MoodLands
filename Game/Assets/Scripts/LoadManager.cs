using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using System.IO;
using TMPro;
using UnityEngine.UI;

public class LoadManager : MonoBehaviour
{
    private SetDefaults defaultMenu;
    private SaveData saveData;
    public GameObject bg;
    public GameObject settingsMenu;

    private string saveFilePath = "./saveData.json";
    private void Start()
    {
        defaultMenu = settingsMenu.GetComponent<SetDefaults>();
        LoadGame();
    }
    private void LoadGame()
    {
        if (File.Exists(saveFilePath))
        {
            string jsonData = File.ReadAllText(saveFilePath);
            saveData = JsonUtility.FromJson<SaveData>(jsonData);
            PlayerPrefs.SetInt("force", saveData.force);
            PlayerPrefs.SetString("player", saveData.player);
            Debug.Log(saveData.player);
            PlayerPrefs.SetString("difficulty", saveData.difficulty);
            PlayerPrefs.Save();

            if (saveData.lastLoadedScene.StartsWith("Red"))
                bg.GetComponent<Image>().color = new Color(0.745283f, 0.4598255f, 0.5050081f, 1);
            else if (saveData.lastLoadedScene.StartsWith("Green"))
                bg.GetComponent<Image>().color = new Color(0.5050081f, 0.745283f, 0.4598255f, 1);
            else
                bg.GetComponent<Image>().color = new Color(0.4598255f, 0.5050081f, 0.745283f, 1);
        }
    }
    public void ContinueGame()
    {
        defaultMenu.SaveData();
        SceneManager.LoadScene(saveData.lastLoadedScene);
    }
    public void StartGame()
    {
        defaultMenu.SaveData();
        SceneManager.LoadScene("Red level 1");
    }
}
