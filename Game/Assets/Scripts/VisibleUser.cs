using Cinemachine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VisibleUser : MonoBehaviour
{
    public InterpretWebcam interpretWebcam;
    public GameObject UserF;
    public GameObject UserM;
    public CinemachineVirtualCamera cameraToSwitch;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (interpretWebcam.gender == "Female")
        {
            if (UserM.activeSelf == true)
            {
                UserF.transform.position = UserM.transform.position;
                UserM.SetActive(false);
                UserF.SetActive(true);
                cameraToSwitch.Follow = UserF.transform;
            }
        }
        else if (UserF.activeSelf == true)
        {
            UserM.transform.position = UserF.transform.position;
            UserM.SetActive(true);
            UserF.SetActive(false);
            cameraToSwitch.Follow = UserM.transform;
        }

    }
}
