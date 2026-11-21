/*
  dotstar_bridge.ino  -  Arduino Micro as a USB-serial bridge to an APA102 / DotStar strip.

  The Jetson builds every frame (start frame, LED frames, end frame) and sends it
  over USB. This sketch only clocks the bytes out, MSB first, by bit-banging:

    Strip DI (data)  -> D14 (MISO pin, used here as a plain digital output)
    Strip CI (clock) -> D15 (SCK)
    Strip GND        -> Arduino GND (and the strip's 5 V supply GND)
    Strip 5V         -> 5 V supply

  The Micro's pins drive 5 V, which is what the strip needs.

  Packet from the host:   'D' 'S' <len high> <len low> <len bytes>
  Reply after the bytes:  'K'
  Bytes are clocked out as they arrive, so there is no length limit and no
  frame buffer. A packet that stalls for more than 200 ms is dropped; the next
  frame's start frame resynchronises the strip.

  No library needed. Baud rate is ignored by the Micro's native USB; the host
  uses 115200 (never 1200, which would reset the board into its bootloader).
*/

const uint8_t PIN_DATA  = 14;
const uint8_t PIN_CLOCK = 15;
const unsigned long BYTE_TIMEOUT_MS = 200;

static inline void shiftByte(uint8_t b) {
  for (uint8_t mask = 0x80; mask; mask >>= 1) {
    digitalWrite(PIN_DATA, (b & mask) ? HIGH : LOW);
    digitalWrite(PIN_CLOCK, HIGH);   // APA102 latches on the rising edge
    digitalWrite(PIN_CLOCK, LOW);
  }
}

// Read one byte, waiting up to BYTE_TIMEOUT_MS. Returns -1 on timeout.
static int readByteTimeout() {
  unsigned long start = millis();
  while (!Serial.available()) {
    if (millis() - start > BYTE_TIMEOUT_MS) return -1;
  }
  return Serial.read();
}

void setup() {
  pinMode(PIN_DATA, OUTPUT);
  pinMode(PIN_CLOCK, OUTPUT);
  digitalWrite(PIN_DATA, LOW);
  digitalWrite(PIN_CLOCK, LOW);
  Serial.begin(115200);
}

void loop() {
  static bool greeted = false;
  if (Serial && !greeted) {
    Serial.println(F("dotstar bridge ready: data D14, clock D15"));
    greeted = true;
  }
  if (!Serial) greeted = false;

  // Wait for the 'D' 'S' header, skipping anything else.
  if (!Serial.available()) return;
  if (Serial.read() != 'D') return;
  if (readByteTimeout() != 'S') return;

  int hi = readByteTimeout();
  int lo = readByteTimeout();
  if (hi < 0 || lo < 0) return;
  uint16_t len = ((uint16_t)hi << 8) | (uint16_t)lo;

  for (uint16_t i = 0; i < len; i++) {
    int b = readByteTimeout();
    if (b < 0) return;               // host stalled: drop the rest, no ack
    shiftByte((uint8_t)b);
  }
  digitalWrite(PIN_DATA, LOW);
  Serial.write('K');
}
