using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.Tilemaps;

public class MoveTile : MonoBehaviour
{
    public float end_x, end_y;
    public float moveSpeed = 1f;
    private bool isMoving = false;
    public string InteractTag;
    private float x, y;
    private float exitTime;
    private bool canStop;
    private bool isStopping;
    private AudioSource audioSource;
    private void Start()
    {
        x = end_x;
        y = end_y;
        exitTime = 0;
        audioSource = GetComponent<AudioSource>();
    }
    private void OnCollisionEnter2D(Collision2D collision)
    {
        if (Time.time - exitTime <= 0.3f)
            canStop = false;
        if (collision.gameObject.CompareTag(InteractTag) && !isMoving && Time.time - exitTime > 0.3f)
        {
            audioSource.Play();
            StartCoroutine(MovePlatform(collision.gameObject));
        }
            
    }
    private void OnCollisionExit2D(Collision2D collision)
    {  
        if (collision.gameObject.CompareTag(InteractTag) && isMoving && !isStopping)
        {
            exitTime = Time.time;
            canStop = true;
            StartCoroutine(StopPlatform(collision.gameObject));
        }
            
    }
    private IEnumerator StopPlatform(GameObject gameObject)
    {
        isStopping = true;
        yield return new WaitForSeconds(0.3f);
        if (canStop)
        {
            isMoving = false;
            audioSource.Stop();
        }
        isStopping = false; 
    }
    private IEnumerator MovePlatform(GameObject collision)
    {
        isMoving = true;
        Vector3 startPosition = transform.position;
        Vector3 endPosition = new Vector3(x, y, startPosition.z);
        Vector3 direction = (endPosition - startPosition).normalized;
        Vector3 pos;

        while (Vector3.Distance(transform.position, endPosition) > 0.5f && isMoving)
        {
            pos = direction * moveSpeed * Time.deltaTime;
            transform.position += pos;
            collision.transform.position += pos;
            yield return null;
        }
        isMoving = false;
        if (Vector3.Distance(transform.position, endPosition) <= 0.5f)
            if (x == end_x)
            {
                x = 0;
                y = 0;
            }
            else
            {
                x = end_x;
                y = end_y;
            }
        audioSource.Stop();
        yield return null;
    }
}