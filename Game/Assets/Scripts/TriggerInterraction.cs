using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TriggerInterraction : MonoBehaviour
{
    // Start is called before the first frame update
    private void OnCollisionEnter2D(Collision2D collision)
    {
        if (collision.gameObject.CompareTag("Trigger"))
        {
            TriggerTarget triggerTarget = collision.gameObject.GetComponent<TriggerTarget>();
            if (triggerTarget != null && triggerTarget.active && Time.time - triggerTarget.lastToggleTime >= triggerTarget.toggleCooldown)
            {
                triggerTarget.active = false;
                triggerTarget.lastToggleTime = Time.time;
                triggerTarget.Toggle();
            }

        }
    }

    private void OnCollisionExit2D(Collision2D collision)
    {
        if (collision.gameObject.CompareTag("Trigger"))
        {
            TriggerTarget triggerTarget = collision.gameObject.GetComponent<TriggerTarget>();
            triggerTarget.active = true;
        }
    }
}
