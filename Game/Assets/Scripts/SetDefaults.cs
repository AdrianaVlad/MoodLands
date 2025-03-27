using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;

public class SetDefaults : MonoBehaviour
{
    public string player;
    public string difficulty;
    public bool force;
    public TextMeshProUGUI playerButton;
    public TextMeshProUGUI forceButton;
    public TextMeshProUGUI diffButton;

    private void Start()
    {
        player = "F";
        playerButton.text = "Default player:\nfeminine";
        difficulty = "normal";
        diffButton.text = "Default difficulty:\nnormal";
        force = false;
        forceButton.text = "Force default:\nno";
        if (PlayerPrefs.GetString("player") != player)
            togglePlayer();
        if (PlayerPrefs.GetString("difficulty") != difficulty)
            toggleDifficulty();
        if (PlayerPrefs.GetInt("force") == 1)
            toggleForce();
    }
    public void togglePlayer()
    { 
        if (player == "F")
        {
            player = "M";
            playerButton.text = "Default player:\nmasculine";
        }
        else
        {
            player = "F";
            playerButton.text = "Default player:\nfeminine";
        }
        
    }

    public void toggleDifficulty()
    {
        if (difficulty == "easy")
        {
            difficulty = "normal";
            diffButton.text = "Default difficulty:\nnormal";
        }
        else
        {
            difficulty = "easy";
            diffButton.text = "Default difficulty:\neasy";
        }
    }
    public void toggleForce()
    {
        force = !force;
        if (force)
            forceButton.text = "Force default:\nyes";
        else
            forceButton.text = "Force default:\nno";
    }
    public void SaveData()
    {
        PlayerPrefs.SetString("player", player);
        PlayerPrefs.SetString("difficulty", difficulty);
        PlayerPrefs.SetInt("force", force ? 1 : 0);
        PlayerPrefs.Save();
    }
    public void StartNewGame()
    {
        SaveData();
        SceneManager.LoadScene("Red Level 1");
    }
}
