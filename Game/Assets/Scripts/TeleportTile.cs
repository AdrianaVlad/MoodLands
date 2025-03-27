using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Tilemaps;
using static UnityEngine.EventSystems.EventTrigger;

public class TeleportTile : MonoBehaviour
{
    public GameObject UserF;
    public GameObject UserM;
    private bool inUse;
    public string InteractTag;
    public float dest_x;
    public float dest_y;
    private AudioSource audioSource;
    [SerializeField]
    public bool _isPowered;

    private Tilemap tilemap;

    public bool IsPowered
    {
        get
        {
            return _isPowered;
        }
        set
        {
            _isPowered = value;
            if (value)
                tilemap.color = Color.white;
            else
                tilemap.color = new Color(0.5f, 0.5f, 0.5f, 1);

        }
    }

    private void Start()
    {
        tilemap = GetComponent<Tilemap>();
        IsPowered = IsPowered;
        audioSource = GetComponent<AudioSource>();
    }
    private void OnCollisionEnter2D(Collision2D collision)
    {
        if (collision.gameObject.CompareTag(InteractTag) && !inUse && IsPowered)
            Teleport(collision.gameObject);
    }

    private void Teleport(GameObject collision)
    {
        audioSource.Play();
        inUse = true;
        Vector3 pos = new Vector3(dest_x, dest_y, 0);
        if (collision.CompareTag("Player"))
        {
            UserF.transform.position = pos;
            UserM.transform.position = pos;
        }
        else
            collision.transform.position = pos;
        inUse = false;
    }
}
