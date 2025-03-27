using Cinemachine;
using System;
using System.Collections;
using TMPro;
using UnityEngine;

public class SelectTarget : MonoBehaviour
{
    public string targetTag = "Enemy";
    public bool active = false;
    public GameObject emotionMenu;
    private PickEmotion pe;
    private AffectDragon componentToToggle;
    public CinemachineVirtualCamera virtualCamera;
    private float originalOrthoSize;
    public string componentTypeName;
    public AudioSource abilitySound;
    public GameObject emotionInfoMenu;
    public void SelectEnemy()
    {
        if (PlayerPrefs.GetInt("force") == 1)
        {
            emotionMenu.SetActive(true);
            pe = emotionMenu.GetComponent<PickEmotion>();
        }
        StartCoroutine(WaitForMouseClick());
        emotionInfoMenu.SetActive(true);
    }

    private IEnumerator WaitForMouseClick()
    {
        if (virtualCamera != null)
        {
            originalOrthoSize = virtualCamera.m_Lens.OrthographicSize;
            virtualCamera.m_Lens.OrthographicSize = 20f;
        }
        yield return new WaitForSeconds(0.1f);
        Time.timeScale = 0f;
        PlayerPrefs.SetInt("ability",1);
        while (PlayerPrefs.GetInt("force") == 1 && pe.picked == false)
        {
            yield return null;
        }
        while (!Input.GetMouseButtonDown(0))
        {
            yield return null;
        }
        if(PlayerPrefs.GetInt("paused") == 0)
            Time.timeScale = 1f;
        PlayerPrefs.SetInt("ability", 0);
        if (PlayerPrefs.GetInt("force") == 1)
        {
            emotionMenu.SetActive(false);
            pe.picked = false;
        }
        Vector2 rayOrigin = Camera.main.ScreenToWorldPoint(Input.mousePosition);
        Physics2D.queriesHitTriggers = false;
        RaycastHit2D hit = Physics2D.Raycast(rayOrigin, Vector2.zero);
        Physics2D.queriesHitTriggers = true;
        if (hit.collider != null)
        {
            GameObject clickedObject = hit.collider.gameObject;
            if (clickedObject.CompareTag(targetTag))
            {
                Type type = Type.GetType(componentTypeName);
                if (type != null)
                {
                    Component componentToToggle = clickedObject.GetComponent(type);
                    if (componentToToggle != null)
                    {
                        (componentToToggle as MonoBehaviour).enabled = true;
                        abilitySound.Play();
                    }
                }
            }
        }
        if (virtualCamera != null)
        {
            virtualCamera.m_Lens.OrthographicSize = originalOrthoSize;
        }

        active = false;
        emotionInfoMenu.SetActive(false);
    }
}
