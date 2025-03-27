using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.Universal;

public class PowerTP : MonoBehaviour
{
    public GameObject TP;
    public float direction = 1;
    AudioSource audioSource;
    private void Start()
    {
        audioSource = GetComponent<AudioSource>();
    }
    private void Update()
    {
        Collider2D[] overlappingColliders = Physics2D.OverlapCapsuleAll(GetComponent<CapsuleCollider2D>().offset, GetComponent<CapsuleCollider2D>().size, GetComponent<CapsuleCollider2D>().direction, 0f);
        foreach (Collider2D collider in overlappingColliders)
        {
            
            if (collider.CompareTag("Enemy") && collider.GetComponent<BotMovement>().IsAttacking && collider.gameObject.transform.localScale.x * direction >= 0)
            {
                TP.GetComponent<TeleportTile>().IsPowered = true;
                enabled = false;
                audioSource.Play();
            }
        }
    }
}
