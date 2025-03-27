using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Rendering.Universal;

public class LightToggle : MonoBehaviour
{
    private Animator enemyAnimator;
    public Light2D torchLight;

    private void Start()
    {
        Collider2D[] colliders = Physics2D.OverlapBoxAll(transform.position, GetComponent<BoxCollider2D>().size, 0f);
        foreach (Collider2D collider in colliders)
        {
            if (collider.CompareTag("Enemy") && CanPerformAction(collider))
                OnTriggerAction(collider);
        }
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        if (other.CompareTag("Enemy") && CanPerformAction(other))
            OnTriggerAction(other);
    }
    private void OnTriggerAction(Collider2D enemyCollider)
    {
        enemyAnimator = enemyCollider.GetComponent<Animator>();
        if (enemyAnimator != null && enemyAnimator.GetBool("isAttacking"))
            torchLight.enabled = true;
    }

    public bool CanPerformAction(Collider2D enemy)
    {
        float enemyDirection = Mathf.Sign(enemy.transform.localScale.x);
        float directionToEnemy = Mathf.Sign(enemy.transform.position.x - transform.position.x);
        return (enemyDirection != directionToEnemy);
    }
}
