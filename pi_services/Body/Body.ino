// ===== MOTOR PINS =====
#define R_PWM1   5
#define L_EN1   13
#define L_PWM1  11
#define R_EN1  12
#define R_EN2  7
#define L_EN2  8
#define L_PWM2 10
#define R_PWM2 9

// ===== ULTRASONIC =====
#define TRIG        3
#define ECHO_PIN        4
#define SAFE_DISTANCE   25    // cm

// ===== STATE =====
String currentCommand  = "S";
String runningCommand  = "S";   // what motors are ACTUALLY doing right now
int    obstacleCount   = 0;
bool   obstacleBlocked = false; // true while obstacle present — rejects move cmds

// ===== ANIMATIONS =====
bool          wiggling    = false;
int           wiggleStep  = 0;
unsigned long wiggleTimer = 0;
#define WIGGLE_STEP_MS 180

bool          nodding   = false;
int           nodStep   = 0;
unsigned long nodTimer  = 0;
#define NOD_STEP_MS 200

bool          thinking     = false;
int           thinkStep    = 0;
unsigned long thinkTimer   = 0;
#define THINK_STEP_MS 400
#define THINK_STEPS   6    // 3 back-and-forth cycles

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);

  pinMode(R_EN1,   OUTPUT);
  pinMode(L_EN1,   OUTPUT);
  pinMode(R_EN2,  OUTPUT);
  pinMode(L_EN2,  OUTPUT);
  pinMode(R_PWM1,  OUTPUT);
  pinMode(L_PWM1,  OUTPUT);
  pinMode(R_PWM2, OUTPUT);
  pinMode(L_PWM2, OUTPUT);
  pinMode(12, OUTPUT); // L2

  digitalWrite(R_EN1,  HIGH);
  digitalWrite(L_EN1,  HIGH);
  digitalWrite(R_EN2, HIGH);
  digitalWrite(L_EN2, HIGH);
  digitalWrite(12, HIGH); // Enable L298N motor driver

  pinMode(TRIG, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  stopMotors();
  Serial.println("READY");
}

// ===== DISTANCE =====
long getDistanceCM() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  long d = pulseIn(ECHO_PIN, HIGH, 30000);
  if (d == 0) return 999;   // timeout = open space
  return d * 0.034 / 2;
}

// ===== MOTOR PRIMITIVES =====
void stopMotors()   { analogWrite(R_PWM1,0);   analogWrite(L_PWM1,0); }
void moveForward()  { analogWrite(R_PWM1,255); analogWrite(L_PWM1,0); }
void moveBackward() { analogWrite(R_PWM1,0);   analogWrite(L_PWM1,200); }
void moveLeft()     { analogWrite(R_PWM1,255); analogWrite(L_PWM1,100); }
void moveRight()    { analogWrite(R_PWM1,100); analogWrite(L_PWM1,255); }
void spinLeft()     { analogWrite(R_PWM1,220); analogWrite(L_PWM1,0); }
void spinRight()    { analogWrite(R_PWM1,0);   analogWrite(L_PWM1,220); }

// ===== APPLY COMMAND (only acts when command changes) =====
void applyCommand(String cmd) {
  if (cmd == runningCommand) return;  // already doing this — skip
  runningCommand = cmd;
  if (cmd == "F") moveForward();
  else if (cmd == "B") moveBackward();
  else if (cmd == "L") moveLeft();
  else if (cmd == "R") moveRight();
  else if (cmd == "W") { wiggling = true; wiggleStep = 0; wiggleTimer = millis(); }
  else if (cmd == "N") { nodding  = true; nodStep    = 0; nodTimer    = millis(); }
  else if (cmd == "T") { thinking = true; thinkStep  = 0; thinkTimer  = millis(); }
  else                   stopMotors();
}

// ===== WIGGLE (happy dance) =====
void updateWiggle() {
  if (wiggling) {
    if (millis() - wiggleTimer >= WIGGLE_STEP_MS) {
      wiggleStep++;
      wiggleTimer = millis();
    }
  }
}

// ===== NOD (nodding dance) =====
void updateNod() {
  if (nodding) {
    if (millis() - nodTimer >= NOD_STEP_MS) {
      nodStep++;
      nodTimer = millis();
    }
  }
}

// ===== THINK (back and forth) =====
void updateThink() {
  if (!thinking) return;
  if (millis() - thinkTimer < THINK_STEP_MS) return;
  thinkTimer = millis();

  if (thinkStep >= THINK_STEPS) {
    thinking = false;
    thinkStep = 0;
    stopMotors();
    runningCommand = "S";
    Serial.println("THINK_DONE");
    return;
  }

  // alternate F and B each step
  if (thinkStep % 2 == 0) moveForward();
  else                     moveBackward();
  thinkStep++;
}

// ===== LOOP =====
void loop() {
  // Your code here
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    applyCommand(command);
  }
  updateWiggle();
  updateNod();
  updateThink();
  delay(10);
}
