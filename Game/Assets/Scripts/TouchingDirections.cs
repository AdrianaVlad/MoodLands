using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TouchingDirections : MonoBehaviour
{
    Animator animator;

    [SerializeField]
    private bool _isGrounded;
    [SerializeField]
    private bool _isOnWall;
    [SerializeField]
    private bool _isOnCeiling;

    public bool IsGrounded
    {
        get
        {
            return _isGrounded;
        }
        set
        {
            _isGrounded = value;
            animator.SetBool(AnimationStrings.isGrounded, value);
        }
    }

    public bool IsOnWall
    {
        get
        {
            return _isOnWall;
        }
        set
        {
            _isOnWall = value;
            animator.SetBool(AnimationStrings.isOnWall, value);
        }
    }

    public bool IsOnCeiling
    {
        get
        {
            return _isOnWall;
        }
        set
        {
            _isOnWall = value;
            animator.SetBool(AnimationStrings.isOnCeiling, value);
        }
    }

    private void Awake()
    {
        animator = GetComponent<Animator>();
    }
    void Start()
    {
        
    }

    // Update is called once per frame
    private void OnCollisionEnter2D(Collision2D other)
    {
        if (other.gameObject.CompareTag("Wall") || other.gameObject.CompareTag("Breakable"))
        {
            IsOnWall = true;
        }
        else if (other.gameObject.CompareTag("Ground"))
        {
            IsGrounded = true;
        }
        else if (other.gameObject.CompareTag("Ceiling"))
        {
            IsOnCeiling = true;
        }
    }
    private void OnCollisionExit2D(Collision2D other)
    {
        if (other.gameObject.CompareTag("Wall") || other.gameObject.CompareTag("Breakable"))
        {
            IsOnWall = false;
        }
        else if (other.gameObject.CompareTag("Ground"))
        {
            IsGrounded = false;
        }
        else if (other.gameObject.CompareTag("Ceiling"))
        {
            IsOnCeiling = false;
        }
    }
}
