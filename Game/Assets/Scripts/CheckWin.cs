using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CheckWin : MonoBehaviour
{
    public GameObject winPanel;
    void Start()
    {
        if(PlayerPrefs.GetInt("Win") == 1)
        {
            PlayerPrefs.SetInt("Win", 0);
            winPanel.SetActive(true);
        }
    }

}
