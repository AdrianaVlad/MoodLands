using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BGParallax : MonoBehaviour
{
    public Camera cam;
    public Transform player;

    private Vector2 startingPos;
    private float startingZ;

    Vector2 newPos;

    float distanceFromPlayer => transform.position.z - player.transform.position.z;
    float clippingPos => (cam.transform.position.z + (distanceFromPlayer > 0 ? cam.farClipPlane : cam.nearClipPlane));
    Vector2 camMoveSinceStart => (Vector2)cam.transform.position - startingPos;
    float parallaxFactor => - Mathf.Abs(distanceFromPlayer) / 10;

    // Start is called before the first frame update
    void Start()
    {
        startingPos = transform.position;
        startingZ = transform.position.z;
    }

    // Update is called once per frame
    void Update()
    {
        newPos = startingPos + camMoveSinceStart * parallaxFactor;
        transform.position = new Vector3(newPos.x, newPos.y, startingZ);
    }
}
