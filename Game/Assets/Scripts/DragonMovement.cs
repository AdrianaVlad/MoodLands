using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.Universal;
using static UnityEngine.Rendering.DebugUI;

public class DragonMovement : MonoBehaviour
{
    public float walkSpeed = 3f;
    Rigidbody2D rb;
    Collider2D col;
    TouchingDirections touchDirections;
    Animator animator;
    public AudioSource moveSound;
    public AudioSource attackSound;
    public enum WalkableDirection {right, left};
    [SerializeField]
    private bool _isMoving;
    [SerializeField]
    public bool _isAttacking;
    public bool IsMoving
    {
        get
        {
            return _isMoving;
        }
        set
        {
            _isMoving = value;
            animator.SetBool(AnimationStrings.isMoving, value);
            if (value == true)
            {
                walkSpeed = 3f;
                moveSound.Play();
            }
            else
            {
                walkSpeed = 0f;
                moveSound.Stop();
            }
        }
    }
    public bool IsAttacking
    {
        get
        {
            return _isAttacking;
        }
        set
        {
            _isAttacking = value;
            animator.SetBool(AnimationStrings.isAttacking, value);
            if (value) // Check for overlapping triggers when attacking
            {
                CheckForOverlappingTriggers();
                attackSound.Play();
            }
            else
                attackSound.Stop();
        }
    }

    private WalkableDirection _walkDirection;
    private Vector2 walkDirectionVector = Vector2.right;

    public WalkableDirection WalkDirection
    {
        get { return _walkDirection; }
        set
        {
            if (_walkDirection != value)
            {
                _walkDirection = value;
                gameObject.transform.localScale = new Vector2(gameObject.transform.localScale.x * (-1), gameObject.transform.localScale.y);
                if (value == WalkableDirection.right)
                {
                    walkDirectionVector = Vector2.right;
                }
                else
                {
                    walkDirectionVector = Vector2.left;
                }
            }
        }
    }

    private void Awake()
    {
        col = GetComponent<Collider2D>();
        rb = GetComponent<Rigidbody2D>();
        touchDirections = GetComponent<TouchingDirections>();
        animator = GetComponent<Animator>();
        _walkDirection = gameObject.transform.localScale.x > 0 ? WalkableDirection.right : WalkableDirection.left;
        if (_walkDirection == WalkableDirection.right)
            walkDirectionVector = Vector2.right;
        else
            walkDirectionVector = Vector2.left;
        IsMoving = !IsMoving;
        IsMoving = !IsMoving;
        animator.SetBool(AnimationStrings.isAttacking, IsAttacking);
    }

    private void FixedUpdate()
    {
        if (touchDirections.IsOnWall)
        {
            touchDirections.IsOnWall = false;
            FlipDirection();  
        }

        rb.velocity = new Vector2(walkSpeed * walkDirectionVector.x, rb.velocity.y);
    }

    private void FlipDirection()
    {
        if (WalkDirection == WalkableDirection.right)
        {
            WalkDirection = WalkableDirection.left;
        }
        else
        {
            WalkDirection = WalkableDirection.right;
        }
    }
    private void CheckForOverlappingTriggers()
    {
        Collider2D[] overlappingColliders = Physics2D.OverlapBoxAll(transform.position, transform.localScale, 0f);
        foreach (Collider2D collider in overlappingColliders)
        {
            if (collider.CompareTag("Light"))
            {
                if (collider.GetComponent<LightToggle>().CanPerformAction(col))
                     collider.GetComponent<Light2D>().enabled = true;
            }
        }
    }
}
