using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ManageEnemyCollisions : MonoBehaviour
{
    void Start()
    {
        Collider2D entityCollider = GetComponent<Collider2D>();
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
        foreach (GameObject enemy in enemies)
        {
            Collider2D enemyCollider = enemy.GetComponent<Collider2D>();

            if (enemyCollider != null)
            {
                Physics2D.IgnoreCollision(entityCollider, enemyCollider);
            }
        }
    }
}
