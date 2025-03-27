using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.Universal;

public class SlimeMovement : MonoBehaviour
{
    public float walkSpeed = 3f;
    Rigidbody2D rb;
    TouchingDirections touchDirections;
    Animator animator;
    public float heightIncrease = 1.2f;
    public CapsuleCollider2D capsuleCollider;
    public AudioSource jumpSound;
    public AudioSource moveSound;

    public enum WalkableDirection { right, left };
    [SerializeField]
    private bool _isMoving;
    [SerializeField]
    public bool _isJumping;
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
            if (value)
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
    public bool IsJumping
    {
        get
        {
            return _isJumping;
        }
        set
        {
            _isJumping = value;
            animator.SetBool(AnimationStrings.isJumping, value);
            if (IsMoving)
                walkSpeed = 3f;
            else
                walkSpeed = 0f;
            if (value)
            {
                walkSpeed *= 2;
                jumpSound.Play();
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
        rb = GetComponent<Rigidbody2D>();
        touchDirections = GetComponent<TouchingDirections>();
        animator = GetComponent<Animator>();
        _walkDirection = gameObject.transform.localScale.x > 0 ? WalkableDirection.right : WalkableDirection.left;
        if (_walkDirection == WalkableDirection.right)
            walkDirectionVector = Vector2.right;
        else
            walkDirectionVector = Vector2.left;
        IsMoving = IsMoving;
        IsJumping = IsJumping;
        IsAttacking = IsAttacking;
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
    public void OnJumpAnimationEnd()
    {
        IsJumping = false;
        jumpSound.Play();
    }
    public void IncreaseHeight()
    {
        Vector3 currentPosition = transform.position;
        currentPosition.y += heightIncrease;
        transform.position = currentPosition;
        capsuleCollider.offset += new Vector2(0f, 0.2f);
    }
    public void DecreaseHeight()
    {
        Vector3 currentPosition = transform.position;
        currentPosition.y += heightIncrease;
        transform.position = currentPosition;
        capsuleCollider.offset -= new Vector2(0f, 0.2f);
    }
}
