const int dirPin1   = 1;   // Direction
const int stepPin1  = 2;   // Step
const int dirPin2   = 3;   // Direction
const int stepPin2  = 4;   // Step
const int dirPin3   = 5;   // Direction
const int stepPin3  = 6;   // Step
const int dirPin4   = 7;   // Direction
const int stepPin4  = 8;   // Step
const int dirPin5   = 9;   // Direction
const int stepPin5  = 10;   // Step
const int dirPin6   = 12;   // Direction
const int stepPin6  = 13;   // Step
const int STEPS_PER_REV = 200;
const int analogInPin = A0;
int sensorValue; //电位器的电阻数值
int Ud;
int Dd;
int Rd;
int Ld;
int Fd;
int Delaytime;
char SolveStep;
void setup() {
  Serial.begin(9600);
  pinMode(dirPin1,OUTPUT); 
  pinMode(stepPin1,OUTPUT);
  pinMode(dirPin2,OUTPUT); 
  pinMode(stepPin2,OUTPUT);
  pinMode(dirPin3,OUTPUT); 
  pinMode(stepPin3,OUTPUT);
  pinMode(dirPin4,OUTPUT); 
  pinMode(stepPin4,OUTPUT);
  pinMode(dirPin5,OUTPUT); 
  pinMode(stepPin5,OUTPUT);
  pinMode(dirPin6,OUTPUT); 
  pinMode(stepPin6,OUTPUT);
}
 
void loop() {  
   // 读取旋转电位计的数值
  sensorValue = analogRead(analogInPin);
  // 将0～1023数值范围映射到500~2000
  Delaytime= map(sensorValue, 0, 1023, 200, 2000);
  // 设置电机参数
  if (Serial.available() > 0)//判读是否串口有数据
  {
    SolveStep =char(Serial.read());
    runUsrCmd();
  }
}

void runUsrCmd(){
        switch (SolveStep){
          case 'U':
          {
            delay(Delaytime);
            digitalWrite(dirPin1, HIGH);    //正转
            for(int x=0; x<100; x++){
              digitalWrite(stepPin1, HIGH);
              delayMicroseconds(50);
              digitalWrite(stepPin1, LOW);
              delayMicroseconds(50);
            }
            Serial.println("Clockwise Face U");
            break;
          }
          case 'u':
          {         
           // digitalWrite(ENAu, 0);
            delay(Delaytime); 
            digitalWrite(dirPin1, LOW);    //反转
            for(int x=0; x<100; x++){
              digitalWrite(stepPin1, HIGH);
              delayMicroseconds(50);
              digitalWrite(stepPin1, LOW);
              delayMicroseconds(50);
            }
            Serial.println("Clockwise Face U");
           // digitalWrite(ENAu, 1);
            break;
          }
          case 'D':
          {
            delay(Delaytime);
            digitalWrite(dirPin2, HIGH);    //正转
            for(int x=0; x<100; x++){
              digitalWrite(stepPin2, HIGH);
              delayMicroseconds(50);
              digitalWrite(stepPin2, LOW);
              delayMicroseconds(50);
            }
            Serial.println("Clockwise Face D");  
            //digitalWrite(ENAd, 1);         
            break;
          }
          case 'd':
          {
            delay(Delaytime);
            digitalWrite(dirPin2, LOW);    //反转
            for(int x=0; x<100; x++){
              digitalWrite(stepPin2, HIGH);
              delayMicroseconds(50);
              digitalWrite(stepPin2, LOW);
              delayMicroseconds(50);
            }
            Serial.println("Anti-Clockwise Face D");
            //digitalWrite(ENAd, 1); 
            break;
          }
          case 'L':
          {
            delay(Delaytime);
            digitalWrite(dirPin3, HIGH);    //正转
            for(int x=0; x<100; x++){
              digitalWrite(stepPin3, HIGH);
              delayMicroseconds(50);
              digitalWrite(stepPin3, LOW);
              delayMicroseconds(50);
            }
            Serial.println("Clockwise Face L"); 
           // digitalWrite(ENAl, HIGH); 
            break;
          }
          case 'l':
          {
            delay(Delaytime);
            digitalWrite(dirPin3, LOW);    //反转
            for(int x=0; x<100; x++){
              digitalWrite(stepPin3, HIGH);
              delayMicroseconds(50);
              digitalWrite(stepPin3, LOW);
              delayMicroseconds(50);
            }
            Serial.println("Anti-Clockwise Face L");
           // digitalWrite(ENAl, HIGH); 
            break;
          }
         case 'R':
          {
            //digitalWrite(ENAr, LOW); 
            delay(Delaytime);
            digitalWrite(dirPin4, HIGH);    //正转
            for(int x=0; x<100; x++){
              digitalWrite(stepPin4, HIGH);
              delayMicroseconds(50);
              digitalWrite(stepPin4, LOW);
              delayMicroseconds(50);
            }
            Serial.println("Clockwise Face R"); 
            break;
          }
          case 'r':
          {
            //digitalWrite(ENAr, LOW);
            delay(Delaytime);
            digitalWrite(dirPin4, LOW);    //反转
            for(int x=0; x<100; x++){
              digitalWrite(stepPin4, HIGH);
              delayMicroseconds(50);
              digitalWrite(stepPin4, LOW);
              delayMicroseconds(50);
            }
            Serial.println("Anti-Clockwise Face R");
           // digitalWrite(ENAr, HIGH);
            break;
          }
          case 'F':
          {
           // digitalWrite(ENAf, LOW);
            delay(Delaytime);
            digitalWrite(dirPin5, HIGH);    //正转
            for(int x=0; x<100; x++){
              digitalWrite(stepPin5, HIGH);
              delayMicroseconds(50);
              digitalWrite(stepPin5, LOW);
              delayMicroseconds(50);
            }
            Serial.println("Clockwise Face F"); 
            break;
          }
          case 'f':
          {
            delay(Delaytime);
            digitalWrite(dirPin5, LOW);    //反转
            for(int x=0; x<100; x++){
              digitalWrite(stepPin5, HIGH);
              delayMicroseconds(50);
              digitalWrite(stepPin5, LOW);
              delayMicroseconds(50);
            }
            Serial.println("Anti-Clockwise Face F");
            break;
          }
          case 'B':
          {
            //digitalWrite(ENAb, LOW);
            delay(Delaytime);
            digitalWrite(dirPin6, HIGH);    //正转
            for(int x=0; x<3200; x++){
              digitalWrite(stepPin6, HIGH);
              delayMicroseconds(50);
              digitalWrite(stepPin6, LOW);
              delayMicroseconds(50);
            }
            Serial.println("Anti-Clockwise Face B");
            //digitalWrite(ENAb, HIGH);
            break;
          }
          case 'b':
          {
           // digitalWrite(ENAb, LOW);
            delay(Delaytime);
            digitalWrite(dirPin6, LOW);    //反转
            for(int x=0; x<3200; x++){
              digitalWrite(stepPin6, HIGH);
              delayMicroseconds(50);
              digitalWrite(stepPin6, LOW);
              delayMicroseconds(50);
            }
            Serial.println("Anti-Clockwise Face B");
           // digitalWrite(ENAb, HIGH);
            break;
          }
          default:  // 未知指令
            Serial.println(F("Unknown Command"));
        }
} 
