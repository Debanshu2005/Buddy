#include <Arduino.h>

// ===== MOTOR 1 =====
#define R_PWM1   6
#define L_PWM1   11
#define L_EN1    13
#define R_EN1    12

// ===== MOTOR 2 =====
#define R_EN2    7
#define L_EN2    8
#define L_PWM2   10
#define R_PWM2   9

// ===== ULTRASONIC =====
#define TRIG          3
#define ECHO_PIN      4
#define SAFE_DISTANCE 25
#define ECHO_TIMEOUT  30000

// ===== STATE =====
String runningCommand = "S";
bool obstacleBlocked = false;
unsigned long lastObstacleCheck = 0;
#define OBSTACLE_CHECK_MS 100

// ===== ANIMATIONS =====
bool wiggling = false, nodding = false, thinking = false;
int wiggleStep = 0, nodStep = 0, thinkStep = 0;
unsigned long wiggleTimer = 0, nodTimer = 0, thinkTimer = 0;

#define WIGGLE_STEP_MS 180
#define WIGGLE_STEPS   6
#define NOD_STEP_MS    200
#define NOD_STEPS      4
#define THINK_STEP_MS  400
#define THINK_STEPS    6

// ===== SETUP =====
void setup() {
  Serial.begin(115200);

  pinMode(R_PWM1, OUTPUT); pinMode(L_PWM1, OUTPUT);
  pinMode(R_EN1,  OUTPUT); pinMode(L_EN1,  OUTPUT);
  pinMode(R_PWM2, OUTPUT); pinMode(L_PWM2, OUTPUT);
  pinMode(R_EN2,  OUTPUT); pinMode(L_EN2,  OUTPUT);
  pinMode(TRIG,   OUTPUT); pinMode(ECHO_PIN, INPUT);

  digitalWrite(R_EN1, HIGH); digitalWrite(L_EN1, HIGH);
  digitalWrite(R_EN2, HIGH); digitalWrite(L_EN2, HIGH);

  stopMotors();
  Serial.println("READY");
}

// ===== DISTANCE =====
long getDistanceCM() {
  digitalWrite(TRIG, LOW);  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  long d = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT);
  return (d == 0) ? 999 : d * 0.034 / 2;
}

// ===== MOTOR PRIMITIVES =====
void stopMotors() {
  analogWrite(R_PWM1, 0); analogWrite(L_PWM1, 0);
  analogWrite(R_PWM2, 0); analogWrite(L_PWM2, 0);
}

void moveForward() {
  analogWrite(R_PWM1, 200); analogWrite(L_PWM1, 0);
  analogWrite(R_PWM2, 200); analogWrite(L_PWM2, 0);
}

void moveBackward() {
  analogWrite(R_PWM1, 0); analogWrite(L_PWM1, 200);
  analogWrite(R_PWM2, 0); analogWrite(L_PWM2, 200);
}

void moveLeft() {
  // left wheels back, right wheels forward
  analogWrite(R_PWM1, 200); analogWrite(L_PWM1, 0);
  analogWrite(R_PWM2, 0);   analogWrite(L_PWM2, 200);
}

void moveRight() {
  // right wheels back, left wheels forward
  analogWrite(R_PWM1, 0);   analogWrite(L_PWM1, 200);
  analogWrite(R_PWM2, 200); analogWrite(L_PWM2, 0);
}

void spinLeft() {
  // right side forward, left side backward = spin left in place
  analogWrite(R_PWM1, 200); analogWrite(L_PWM1, 0);
  analogWrite(R_PWM2, 200); analogWrite(L_PWM2, 0);
  analogWrite(R_PWM1, 0);   analogWrite(L_PWM1, 200);
  analogWrite(R_PWM2, 0);   analogWrite(L_PWM2, 200);
}

void spinRight() {
  // left side forward, right side backward = spin right in place
  analogWrite(R_PWM1, 0);   analogWrite(L_PWM1, 200);
  analogWrite(R_PWM2, 0);   analogWrite(L_PWM2, 200);
  analogWrite(R_PWM1, 200); analogWrite(L_PWM1, 0);
  analogWrite(R_PWM2, 200); analogWrite(L_PWM2, 0);
}

// ===== COMMAND HANDLER =====
void applyCommand(String cmd) {
  cmd.trim();

  // block movement during animations
  if ((wiggling || nodding || thinking) &&
      (cmd == "F" || cmd == "B" || cmd == "L" || cmd == "R")) return;

  // block forward when obstacle present
  if (obstacleBlocked && cmd == "F") return;

  if (cmd == runningCommand) return;
  runningCommand = cmd;

  if      (cmd == "F") moveForward();
  else if (cmd == "B") moveBackward();
  else if (cmd == "L") moveLeft();
  else if (cmd == "R") moveRight();
  else if (cmd == "W") { wiggling = true; wiggleStep = 0; wiggleTimer = millis(); }
  else if (cmd == "N") { nodding  = true; nodStep    = 0; nodTimer    = millis(); }
  else if (cmd == "T") { thinking = true; thinkStep  = 0; thinkTimer  = millis(); }
  else if (cmd == "S") stopMotors();
}

// ===== ANIMATIONS =====
void updateWiggle() {
  if (!wiggling) return;
  if (millis() - wiggleTimer < WIGGLE_STEP_MS) return;
  wiggleTimer = millis();
  if (wiggleStep >= WIGGLE_STEPS) {
    wiggling = false; stopMotors(); runningCommand = "S";
    Serial.println("WIGGLE_DONE"); return;
  }
  if (wiggleStep % 2 == 0) spinLeft(); else spinRight();
  wiggleStep++;
}

void updateNod() {
  if (!nodding) return;
  if (millis() - nodTimer < NOD_STEP_MS) return;
  nodTimer = millis();
  if (nodStep >= NOD_STEPS) {
    nodding = false; stopMotors(); runningCommand = "S";
    Serial.println("NOD_DONE"); return;
  }
  if (nodStep % 2 == 0) moveForward(); else moveBackward();
  nodStep++;
}

void updateThink() {
  if (!thinking) return;
  if (millis() - thinkTimer < THINK_STEP_MS) return;
  thinkTimer = millis();
  if (thinkStep >= THINK_STEPS) {
    thinking = false; stopMotors(); runningCommand = "S";
    Serial.println("THINK_DONE"); return;
  }
  if (thinkStep % 2 == 0) moveBackward(); else moveForward();
  thinkStep++;
}

// ===== LOOP =====
void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    applyCommand(cmd);
  }

  // ultrasonic safety — sends OBSTACLE/CLEAR to Pi
  if (millis() - lastObstacleCheck > OBSTACLE_CHECK_MS) {
    lastObstacleCheck = millis();
    long d = getDistanceCM();
    bool nowBlocked = (d < SAFE_DISTANCE);
    if (nowBlocked && !obstacleBlocked) {
      obstacleBlocked = true;
      stopMotors(); runningCommand = "S";
      Serial.println("OBSTACLE");
    } else if (!nowBlocked && obstacleBlocked) {
      obstacleBlocked = false;
      Serial.println("CLEAR");
    }
  }

  updateWiggle();
  updateNod();
  updateThink();
}
