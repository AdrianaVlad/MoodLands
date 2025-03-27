using ClearSky;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.Universal;
using static UnityEngine.Rendering.DebugUI;

public class BotMovement : MonoBehaviour
{
    public float walkSpeed = 3f;
    Rigidbody2D rb;
    TouchingDirections touchDirections;
    Animator animator;
    private Coroutine attackCoroutine;
    public AudioSource attackAudio;
    public AudioSource moveAudio;
    public AudioSource destroyAudio;

    public enum WalkableDirection { right, left };
    [SerializeField]
    private bool _isMoving;
    [SerializeField]
    public bool _isAttacking;
    [SerializeField]
    private bool _isSleeping;
    [SerializeField]
    private bool _isWaking;
    public bool IsMoving
    {
        get
        {
            return _isMoving;
        }
        set
        {
            if (_isMoving != value)
            {
                _isMoving = value;
                animator.SetBool(AnimationStrings.isMoving, value);
                if (value)
                {
                    moveAudio.Play();
                    if (attackCoroutine != null)
                    {
                        StopCoroutine(attackCoroutine);
                        attackCoroutine = null;
                    }
                    walkSpeed = 3f;
                    if (!IsSleeping && !IsAttacking)
                    {
                        IsWaking = true;
                    }
                }
                else
                {
                    moveAudio.Stop();
                    walkSpeed = 0f;
                    if (!IsAttacking)
                    {
                        IsSleeping = true;
                    }
                }
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
            if (_isAttacking != value)
            {
                _isAttacking = value;
                animator.SetBool(AnimationStrings.isAttacking, value);
                if (value)
                {
                    if (!IsSleeping && !IsMoving)
                        IsWaking = true;
                }
                else
                {
                    if (attackCoroutine != null)
                    {
                        StopCoroutine(attackCoroutine);
                        attackCoroutine = null;
                        attackAudio.Stop();
                    }
                    if (!IsMoving)
                        IsSleeping = true;
                }
            }
        }
    }
    private bool IsWaking
    {
        get
        {
            return _isWaking;
        }
        set
        {
            _isWaking = value;
            animator.SetBool(AnimationStrings.isWaking, value);
        }
    }
    private bool IsSleeping
    {
        get
        {
            return _isSleeping;
        }
        set
        {
            _isSleeping = value;
            animator.SetBool(AnimationStrings.isSleeping, value);
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
        animator.SetBool(AnimationStrings.isMoving, IsMoving);
        _walkDirection = gameObject.transform.localScale.x > 0 ? WalkableDirection.right : WalkableDirection.left;
        if (_walkDirection == WalkableDirection.right)
            walkDirectionVector = Vector2.right;
        else
            walkDirectionVector = Vector2.left;
        if (IsMoving)
        {
            walkSpeed = 3f;
            if (!IsSleeping && !IsAttacking)
                IsWaking = true;
        }
        else
        {
            walkSpeed = 0f;
        }
        animator.SetBool(AnimationStrings.isAttacking, IsAttacking);
        if (IsAttacking)
        {
            if (attackCoroutine == null)
                attackCoroutine = StartCoroutine(AttackRoutine());
            if (!IsSleeping && !IsMoving)
                IsWaking = true;
        }
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
    public void OnWakeAnimationEnd()
    {
        IsWaking = false;
    }
    public void OnSleepAnimationEnd()
    {
        IsSleeping = false;
    }
    public void StartAttackRoutine()
    {
        if (attackCoroutine == null)
        {
            attackCoroutine = StartCoroutine(AttackRoutine());
        }
    }
    private IEnumerator AttackRoutine()
    {
        yield return new WaitForSeconds(0.8f);
        while (true)
        {
            CheckForTargets();
            yield return new WaitForSeconds(1f);
            if (!IsAttacking)
                break;
        }
    }
    private void CheckForTargets()
    {
        attackAudio.Play();
        Vector3 posStart = new Vector3(transform.position.x + 3 * walkDirectionVector.x, transform.position.y, 0);
        Vector3 posEnd = new Vector3(transform.position.x + 28 * walkDirectionVector.x, transform.position.y,0);
        Physics2D.queriesHitTriggers = false;
        RaycastHit2D hit = Physics2D.Linecast(posStart,posEnd);
        Physics2D.queriesHitTriggers = true;
        if (hit.collider != null)
        {
            if (hit.collider.CompareTag("Player"))
                hit.collider.gameObject.GetComponent<SimplePlayerController>().Hurt();
            if (hit.collider.CompareTag("Breakable"))
            {
                hit.collider.gameObject.SetActive(false);
                destroyAudio.Play();
            }
        }
    }
}
