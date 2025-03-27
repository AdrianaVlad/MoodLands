using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TriggerTarget : MonoBehaviour
{
    public GameObject target;
    public bool active;
    public float toggleCooldown = 1f;
    public float lastToggleTime = 0f;
    private AudioSource audioSource;
    private void Start()
    {
        audioSource = GetComponent<AudioSource>();
    }
    public void Toggle()
    {
        target.SetActive(!target.activeSelf);
        audioSource.Play();
    }
}
