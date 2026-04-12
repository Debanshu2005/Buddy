/*
  Buddy Arduino Motor Controller
  ================================
  Protocol (Serial @ 115200 baud):
    Pi → Arduino : F | B | L | R | S | W | N  (single char + newline)
    Arduino → Pi : READY | OBSTACLE | CLEAR | WIGGLE_DONE | NOD_DONE

  Motor driver: BTS7960 (IBT-2) x2
    Right side motors : R_RPWM, R_LPWM, R_EN
    Left  side motors : L_RPWM, L_LPWM, L_LEN, L_REN

  Ultrasonic sensor: HC-SR04
    TRIG → pin 7, ECHO → pin 6
*/

// ── Right side motors (BTS7960) ───────────────────────────────────────────── //
#define R_RPWM  6    // right forward PWM
#define R_LPWM  11   // right backward PWM
#define R_EN    7    // right enable

// ── Left side motors (BTS7960) ────────────────────────────────────────────── //
#define L_LPWM  10   // left backward PWM
#define L_RPWM  9    // left forward PWM
#define L_LEN   13   // left enable
#define L_REN   12   // left enable

// ── Ultrasonic sensor ─────────────────────────────────────────────────────── //
#define TRIG_PIN  2
#define ECHO_PIN  3

// ── Config ────────────────────────────────────────────────────────────────── //
#define MOTOR_SPEED       180   // 0-255, normal drive speed
#define TURN_SPEED        160   // slightly slower for turns
#define OBSTACLE_DIST_CM  20    // stop if object closer than this
#define OBSTACLE_CHECK_MS 100   // how often to check ultrasonic (ms)

// ── State ─────────────────────────────────────────────────────────────────── //
char     currentCmd      = 'S';
bool     obstacleBlocked = false;
unsigned long lastObstacleCheck = 0;

// ── Wiggle / Nod animation state ──────────────────────────────────────────── //
bool     animating = false;


// ═══════════════════════════════════════════════════════════════════════════ //
void setup() {
  Serial.begin(115200);

  pinMode(R_RPWM, OUTPUT); pinMode(R_LPWM, OUTPUT); pinMode(R_EN, OUTPUT);
  pinMode(L_RPWM, OUTPUT); pinMode(L_LPWM, OUTPUT);
  pinMode(L_LEN,  OUTPUT); pinMode(L_REN,  OUTPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // Enable both drivers
  digitalWrite(R_EN,  HIGH);
  digitalWrite(L_LEN, HIGH);
  digitalWrite(L_REN, HIGH);

  stopMotors();
  Serial.println("READY");
}

// ═══════════════════════════════════════════════════════════════════════════ //
void loop() {
  // ── Read command from Pi ─────────────────────────────────────────────── //
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      char cmd = line.charAt(0);
      handleCommand(cmd);
    }
  }

  // ── Obstacle check (only while moving forward) ───────────────────────── //
  if (!animating && millis() - lastObstacleCheck >= OBSTACLE_CHECK_MS) {
    lastObstacleCheck = millis();
    checkObstacle();
  }
}

// ═══════════════════════════════════════════════════════════════════════════ //
void handleCommand(char cmd) {
  // While obstacle is blocked, only allow S (stop) — ignore all moves
  if (obstacleBlocked && cmd != 'S') {
    return;
  }

  currentCmd = cmd;

  switch (cmd) {
    case 'F': moveForward();  break;
    case 'B': moveBackward(); break;
    case 'L': turnLeft();     break;
    case 'R': turnRight();    break;
    case 'S': stopMotors();   break;
    case 'W': wiggle();       break;
    case 'N': nod();          break;
    default:  break;
  }
}

// ── Obstacle detection ────────────────────────────────────────────────────── //
void checkObstacle() {
  // Only care about obstacles when moving forward
  if (currentCmd != 'F') {
    if (obstacleBlocked) {
      // Was blocked but no longer moving forward — clear the flag
      obstacleBlocked = false;
      Serial.println("CLEAR");
    }
    return;
  }

  long dist = getDistanceCm();

  if (!obstacleBlocked && dist > 0 && dist < OBSTACLE_DIST_CM) {
    obstacleBlocked = true;
    currentCmd = 'S';
    stopMotors();
    Serial.println("OBSTACLE");
  } else if (obstacleBlocked && (dist <= 0 || dist >= OBSTACLE_DIST_CM)) {
    obstacleBlocked = false;
    Serial.println("CLEAR");
  }
}

long getDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);  // 30ms timeout
  if (duration == 0) return -1;  // no echo = out of range
  return duration * 0.034 / 2;
}

// ── Motor primitives ─────────────────────────────────────────────────────── //
void moveForward() {
  analogWrite(R_RPWM, MOTOR_SPEED); analogWrite(R_LPWM, 0);
  analogWrite(L_RPWM, MOTOR_SPEED); analogWrite(L_LPWM, 0);
}

void moveBackward() {
  analogWrite(R_RPWM, 0); analogWrite(R_LPWM, MOTOR_SPEED);
  analogWrite(L_RPWM, 0); analogWrite(L_LPWM, MOTOR_SPEED);
}

void turnLeft() {
  // left side backward, right side forward
  analogWrite(R_RPWM, TURN_SPEED); analogWrite(R_LPWM, 0);
  analogWrite(L_RPWM, 0);          analogWrite(L_LPWM, TURN_SPEED);
}

void turnRight() {
  // right side backward, left side forward
  analogWrite(R_RPWM, 0);          analogWrite(R_LPWM, TURN_SPEED);
  analogWrite(L_RPWM, TURN_SPEED); analogWrite(L_LPWM, 0);
}

void stopMotors() {
  analogWrite(R_RPWM, 0); analogWrite(R_LPWM, 0);
  analogWrite(L_RPWM, 0); analogWrite(L_LPWM, 0);
}

// ── Animations ───────────────────────────────────────────────────────────── //
void wiggle() {
  // Quick left-right wiggle (happy expression)
  animating = true;
  for (int i = 0; i < 3; i++) {
    turnLeft();  delay(200);
    turnRight(); delay(200);
  }
  stopMotors();
  currentCmd = 'S';
  animating  = false;
  Serial.println("WIGGLE_DONE");
}

void nod() {
  // Short forward-back nod (thinking expression)
  animating = true;
  for (int i = 0; i < 2; i++) {
    moveForward();  delay(200);
    moveBackward(); delay(200);
  }
  stopMotors();
  currentCmd = 'S';
  animating  = false;
  Serial.println("NOD_DONE");
}
