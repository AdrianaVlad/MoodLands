using Cinemachine;
using System;
using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace ClearSky
{
    public class SimplePlayerController : MonoBehaviour
    {
        public float movePower = 10f;
        public float jumpPower = 15f;
        public int healthPoints;
        public int manaPoints;
        public SelectTarget st;
        public InterpretWebcam interpretWebcam;
        private Rigidbody2D rb;
        private Animator anim;
        private int direction = 1;
        bool isJumping = false;
        private bool alive = true;
        public CinemachineVirtualCamera virtualCamera;
        private CinemachineFramingTransposer transposer;
        public TextMeshProUGUI characterInfo;
        private float setDelayTime;
        private bool falling;
        private float startFalling;
        private float stoppedJumping;
        private bool setStoppedJumping;
        public GameObject mainUI;
        public AudioSource hpSound;
        public AudioSource mpSound;
        public AudioSource hurtSound;
        public AudioSource deathSound;
        public AudioSource jumpSound;
        public AudioSource moveSound;
        public AudioSource abilitySound;
        public AudioSource landingSound;

        // Start is called before the first frame update
        void Start()
        {
            StartCoroutine(WaitAndStart());
            rb = GetComponent<Rigidbody2D>();
            anim = GetComponent<Animator>();
            transposer = virtualCamera.GetCinemachineComponent<CinemachineFramingTransposer>();
            if (PlayerPrefs.GetInt("Tab") == 0)
                mainUI.SetActive(false);
        }
        IEnumerator WaitAndStart()
        {
            yield return new WaitForSeconds(0.1f);
            if ((PlayerPrefs.GetInt("force") == 1 && PlayerPrefs.GetString("difficulty") == "easy") ||
                 (PlayerPrefs.GetInt("force") == 0 && (interpretWebcam.ageGroupIndex <= 2 || interpretWebcam.ageGroupIndex == 7)))
            {
                healthPoints = 5;
                manaPoints = 5;
            }
            else
            {
                healthPoints = 3;
                manaPoints = 3;
            }
            
        }

        private void Update()
        {
            if (alive)
            {
                if(rb.velocity.y < -0.1f && !falling)
                {
                    falling = true;
                    startFalling = Time.time;
                }
                if(rb.velocity.y >= -0.1f && falling)
                {
                    falling = false;
                }
                Attack();
                Jump();
                Run();
                LookDown();
                Sprint();
                SetCharacterInfo();
                Restart();
                ToggleUI();
                if(transform.position.y <= -50)
                {
                    Die();
                }
            }
        }

        private void SetCharacterInfo()
        {
            characterInfo.text = "HP: " + healthPoints + ", MP: " + manaPoints;
        }

        private void LookDown()
        {
            if (Input.GetKeyDown(KeyCode.DownArrow) || Input.GetKeyDown(KeyCode.S))
            {
                transposer.m_TrackedObjectOffset.y = -4f;
            }
            if (Input.GetKeyUp(KeyCode.DownArrow) || Input.GetKeyUp(KeyCode.S))
            {
                transposer.m_TrackedObjectOffset.y = 0f;
            }
        }
        private void Sprint()
        {
            if (Input.GetKeyDown(KeyCode.LeftShift))
            {
                movePower = 15f;
            }
            if (Input.GetKeyUp(KeyCode.LeftShift))
            {
                movePower = 10f;
            }
        }


        void Run()
        {
            Vector3 moveVelocity = Vector3.zero;
            anim.SetBool("isRun", false);


            if (Input.GetAxisRaw("Horizontal") < 0)
            {
                direction = -1;
                moveVelocity = Vector3.left;

                transform.localScale = new Vector3(direction, 1, 1);
                if (!anim.GetBool("isJump"))
                {
                    anim.SetBool("isRun", true);
                    if(!moveSound.isPlaying)
                        moveSound.Play();
                }
                    

            }
            if (Input.GetAxisRaw("Horizontal") > 0)
            {
                direction = 1;
                moveVelocity = Vector3.right;

                transform.localScale = new Vector3(direction, 1, 1);
                if (!anim.GetBool("isJump"))
                {
                    anim.SetBool("isRun", true);
                    if (!moveSound.isPlaying)
                        moveSound.Play();
                }
                    

            }
            transform.position += moveVelocity * movePower * Time.deltaTime;
        }
        void Jump()
        {
            if ((Input.GetKeyDown(KeyCode.UpArrow) || Input.GetKeyDown(KeyCode.W))
            && !anim.GetBool("isJump") && Time.time - setDelayTime > 0.5f && (!falling || Time.time - startFalling <= 0.3f) && rb.velocity.y <=0.1f)
            {
                isJumping = true;
                anim.SetBool("isJump", true);
                jumpSound.Play();                
            }
            if (isJumping)
            {
                rb.velocity = Vector2.zero;

                Vector2 jumpVelocity = new Vector2(0, jumpPower);
                rb.AddForce(jumpVelocity, ForceMode2D.Impulse);
                setStoppedJumping = false;
                isJumping = false;
            }
            if (Input.GetKeyUp(KeyCode.UpArrow) || Input.GetKeyUp(KeyCode.W))
            {
                rb.velocity = new Vector2(rb.velocity.x, 0f);
            }
            if (rb.velocity.y < 0.01f && rb.velocity.y > -0.01f && anim.GetBool("isJump"))
            {
                if (!setStoppedJumping)
                {
                    stoppedJumping = Time.time;
                    setStoppedJumping = true;
                }
            }
            else
                setStoppedJumping = false;
            if (Time.time - stoppedJumping >= 0.02f && setStoppedJumping)
            {
                anim.SetBool("isJump", false);
                landingSound.Play();
            }
               
        }
        void Attack()
        {
            if ((Input.GetKeyDown(KeyCode.Z) || Input.GetKeyDown(KeyCode.E))&& manaPoints > 0)
            {
                if (st.active == false)
                {
                    st.active = true;
                    abilitySound.Play();
                    st.SelectEnemy();
                    manaPoints -= 1;
                    anim.SetTrigger("attack");
                }
            }
        }
        private void OnCollisionEnter2D(Collision2D collision)
        {
            if (collision.gameObject.CompareTag("Enemy"))
                Hurt();
            else if (collision.gameObject.CompareTag("Heal"))
            {
                if ((PlayerPrefs.GetInt("force") == 1 && PlayerPrefs.GetString("difficulty") == "easy") ||
                 (PlayerPrefs.GetInt("force") == 0 && (interpretWebcam.ageGroupIndex <= 2 || interpretWebcam.ageGroupIndex == 7)))
                    healthPoints += 2;
                else
                    healthPoints += 1;
                hpSound.Play();
                collision.gameObject.SetActive(false);
            }   
            else if (collision.gameObject.CompareTag("Mana"))
            {
                if ((PlayerPrefs.GetInt("force") == 1 && PlayerPrefs.GetString("difficulty") == "easy") ||
                 (PlayerPrefs.GetInt("force") == 0 && (interpretWebcam.ageGroupIndex <= 2 || interpretWebcam.ageGroupIndex == 7)))
                    manaPoints += 2;
                else
                    manaPoints += 1;
                mpSound.Play();
                collision.gameObject.SetActive(false);
            }
            else if (collision.gameObject.CompareTag("Ceiling"))
            {
                setDelayTime = Time.time;
            }
        }
        public void Hurt()
        {
            if (alive)
            {
                anim.SetTrigger("hurt");
                hurtSound.Play();
                if (direction == 1)
                    rb.AddForce(new Vector2(-5f, 1f), ForceMode2D.Impulse);
                else
                    rb.AddForce(new Vector2(5f, 1f), ForceMode2D.Impulse);
                healthPoints -= 1;
                if (healthPoints == 0)
                    Die();
            }
        }
        void Die()
        {
            anim.SetTrigger("die");
            alive = false;
            deathSound.Play();
            Invoke("RestartScene", 2f);
        }
        private void RestartScene()
        {
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        }
        void Restart()
        {
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                RestartScene();
            }
        }
        void ToggleUI()
        {
            if (Input.GetKeyDown(KeyCode.Tab))
            {
                mainUI.SetActive(!mainUI.activeSelf);
                PlayerPrefs.SetInt("Tab", mainUI.activeSelf ? 1 : 0);
            }
        }
    }
}