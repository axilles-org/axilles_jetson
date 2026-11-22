/*
  dotstar_bridge.ino  -  Arduino Micro driving the exo's APA102 / DotStar status strip.

    Strip DI (data)  -> D14 (MISO pin, used here as a plain digital output)
    Strip CI (clock) -> D15 (SCK)
    Strip GND        -> Arduino GND (and the strip's supply GND, if separate)
    Strip 5V         -> 5 V

  The Micro's pins drive 5 V, which is what the strip needs.

  The sketch plays every timed pattern itself, so a blink finishes even if the
  Jetson process that asked for it has exited:

    power-on           green blinking at BOOT_BLINK_HZ for BOOT_MS
    CALIBRATING        yellow blinking at CALIB_BLINK_HZ, until told otherwise
    READY              green, steady (calibrated, assistance off)
    ASSIST             window of `width` LEDs at the position and color sent
    STOP               red, steady for STOP_MS, then off
    ESTOP              red blinking at ESTOP_BLINK_HZ for ESTOP_MS (not cut short
                       by later requests other than another ESTOP)

  CALIBRATING, READY and ASSIST go dark if no packet arrives for WATCHDOG_MS, so
  a host that dies without a goodbye never leaves a stale pattern on.

  Packets from the host (all start with 'D'):

    'D' 'S' <len hi> <len lo> <len bytes>   raw APA102 frame, clocked out as the
                                            bytes arrive; reply 'K'. Used by
                                            led-strip/led_strip.py. Patterns stop
                                            until the next 'P' packet.
    'D' 'P' <mode> <pos> <r> <g> <b>        pattern request; no reply. Modes:
                                            0 OFF, 1 CALIBRATING, 2 READY,
                                            3 ASSIST, 4 STOP, 5 ESTOP. A one-shot
                                            mode plays once each time the
                                            requested mode changes to it. <pos> is
                                            the ASSIST window's first LED, 255 for
                                            none.
    'D' 'C' <num> <o0> <o1> <o2> <brightness> <width>
                                            strip config; no reply. o0..o2 give,
                                            for each wire byte after the
                                            brightness byte, which color it
                                            carries (0 R, 1 G, 2 B; BGR is 2 1 0).
                                            brightness is the APA102 5-bit field
                                            (0..31). Saved to EEPROM so the
                                            power-on blink uses it too.

  Strip power. With STRIP_SENSE_PIN = -1 the strip is assumed to power up with
  the Arduino, and the green blink runs at boot. If the strip has its own
  supply, wire strip 5V -> 10 kOhm -> a spare pin, plus 100 kOhm from that pin to
  GND, and set STRIP_SENSE_PIN to it: the Arduino then waits for strip power
  before the green blink (immediately if the strip is already on) and blinks
  again whenever strip power returns.

  No library needed. Baud rate is ignored by the Micro's native USB; the host
  uses 115200 (never 1200, which would reset the board into its bootloader).
*/

#include <EEPROM.h>

const uint8_t PIN_DATA  = 14;
const uint8_t PIN_CLOCK = 15;
const int8_t  STRIP_SENSE_PIN = -1;          // -1: no sense wire (see above)

const unsigned long BYTE_TIMEOUT_MS   = 200;
const unsigned long WATCHDOG_MS       = 500;
const unsigned long REFRESH_MS        = 100; // resend the frame even if unchanged
const unsigned long SENSE_DEBOUNCE_MS = 50;

// Blink rates in Hz (one on + one off per cycle, equal halves) and durations.
const float         BOOT_BLINK_HZ  = 2.0;    // 0.25 s on / 0.25 s off
const unsigned long BOOT_MS        = 1000;   // on, off, on, off: 2 blinks
const float         CALIB_BLINK_HZ = 3.0;    // ~0.17 s on / ~0.17 s off
const float         ESTOP_BLINK_HZ = 5.0;    // 0.1 s on / 0.1 s off
const unsigned long ESTOP_MS       = 10000;
const unsigned long STOP_MS        = 5000;   // steady red

const uint8_t MAX_LEDS = 64;

const uint8_t M_OFF = 0, M_CALIBRATING = 1, M_READY = 2, M_ASSIST = 3,
              M_STOP = 4, M_ESTOP = 5, M_BOOT = 6, M_RAW = 7, M_NONE = 0xFF;
const uint8_t NO_WINDOW = 0xFF;

struct Config {
  uint8_t magic;
  uint8_t num;
  uint8_t order[3];
  uint8_t bright;
  uint8_t width;
};
const uint8_t CONFIG_MAGIC = 0xA7;
// Used only until a run first sends its config (leds.py), which is then kept in
// EEPROM: 12 LEDs, GBR, half brightness, 2-LED window.
Config cfg = {CONFIG_MAGIC, 12, {1, 2, 0}, 16, 2};

uint8_t steady = M_OFF;              // OFF, CALIBRATING, READY, ASSIST or RAW
unsigned long steadySince = 0;
uint8_t oneshot = M_NONE;            // BOOT, STOP or ESTOP while playing
unsigned long oneshotStart = 0;
uint8_t lastRequested = M_NONE;
unsigned long lastPacketMs = 0;
uint8_t winPos = NO_WINDOW;
uint8_t winColor[3] = {0, 0, 0};

uint8_t want[MAX_LEDS][3];
uint8_t shown[MAX_LEDS][3];
bool shownValid = false;
unsigned long lastShowMs = 0;
bool stripPowered = true;

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

static bool readBytes(uint8_t *buf, uint8_t n) {
  for (uint8_t i = 0; i < n; i++) {
    int b = readByteTimeout();
    if (b < 0) return false;
    buf[i] = (uint8_t)b;
  }
  return true;
}

static bool configValid(const Config &c) {
  if (c.magic != CONFIG_MAGIC || c.num < 1 || c.num > MAX_LEDS) return false;
  if (c.bright > 31 || c.width < 1 || c.width > c.num) return false;
  for (uint8_t i = 0; i < 3; i++) {
    if (c.order[i] > 2) return false;
  }
  return c.order[0] != c.order[1] && c.order[0] != c.order[2] && c.order[1] != c.order[2];
}

static void setSteady(uint8_t mode, unsigned long now) {
  if (mode != steady) steadySince = now;
  steady = mode;
}

static void startOneshot(uint8_t mode, unsigned long now) {
  if (oneshot == M_ESTOP && mode != M_ESTOP) return;   // the E-stop blink always finishes
  oneshot = mode;
  oneshotStart = now;
}

// ---- packets ----------------------------------------------------------------
static void handleRaw() {
  int hi = readByteTimeout();
  int lo = readByteTimeout();
  if (hi < 0 || lo < 0) return;
  uint16_t len = ((uint16_t)hi << 8) | (uint16_t)lo;

  oneshot = M_NONE;
  setSteady(M_RAW, millis());
  lastRequested = M_NONE;
  shownValid = false;
  for (uint16_t i = 0; i < len; i++) {
    int b = readByteTimeout();
    if (b < 0) return;               // host stalled: drop the rest, no ack
    if (stripPowered) shiftByte((uint8_t)b);
  }
  digitalWrite(PIN_DATA, LOW);
  Serial.write('K');
}

static void handleState() {
  uint8_t b[5];
  if (!readBytes(b, 5)) return;
  unsigned long now = millis();
  lastPacketMs = now;
  winPos = b[1];
  winColor[0] = b[2];
  winColor[1] = b[3];
  winColor[2] = b[4];

  uint8_t mode = b[0];
  if (mode == lastRequested) return;   // repeats only refresh the watchdog and window
  lastRequested = mode;
  switch (mode) {
    case M_OFF:
    case M_CALIBRATING:
    case M_READY:
    case M_ASSIST:
      setSteady(mode, now);
      break;
    case M_STOP:
    case M_ESTOP:
      startOneshot(mode, now);
      setSteady(M_OFF, now);
      break;
    default:
      break;
  }
}

static void handleConfig() {
  uint8_t b[6];
  if (!readBytes(b, 6)) return;
  Config c = {CONFIG_MAGIC, b[0], {b[1], b[2], b[3]}, b[4], b[5]};
  if (!configValid(c)) return;
  cfg = c;
  EEPROM.put(0, cfg);                // only rewrites bytes that changed
  shownValid = false;
}

static void pollSerial() {
  while (Serial.available()) {
    if (Serial.read() != 'D') continue;
    int kind = readByteTimeout();
    if (kind == 'S') handleRaw();
    else if (kind == 'P') handleState();
    else if (kind == 'C') handleConfig();
  }
}

// ---- strip power sense ------------------------------------------------------
static void senseStrip(unsigned long now) {
  if (STRIP_SENSE_PIN < 0) return;
  static bool raw = false;
  static unsigned long since = 0;
  bool r = digitalRead(STRIP_SENSE_PIN) == HIGH;
  if (r != raw) {
    raw = r;
    since = now;
  }
  if (raw != stripPowered && now - since >= SENSE_DEBOUNCE_MS) {
    stripPowered = raw;
    shownValid = false;
    if (stripPowered) {
      startOneshot(M_BOOT, now);
    } else {
      digitalWrite(PIN_DATA, LOW);
      digitalWrite(PIN_CLOCK, LOW);
    }
  }
}

// ---- rendering --------------------------------------------------------------
static void fillWant(uint8_t r, uint8_t g, uint8_t b) {
  for (uint8_t i = 0; i < cfg.num; i++) {
    want[i][0] = r;
    want[i][1] = g;
    want[i][2] = b;
  }
}

static unsigned long oneshotLength(uint8_t mode) {
  switch (mode) {
    case M_BOOT:  return BOOT_MS;
    case M_STOP:  return STOP_MS;
    case M_ESTOP: return ESTOP_MS;
    default:      return 0;
  }
}

// On for the first half of each cycle at `hz`, `elapsed_ms` after the pattern began.
static bool blinkOn(unsigned long elapsed_ms, float hz) {
  unsigned long halves = (unsigned long)(elapsed_ms * (2.0 * hz) / 1000.0);
  return halves % 2 == 0;
}

// Fills `want` for the current pattern.
static void render(unsigned long now) {
  fillWant(0, 0, 0);

  if (oneshot != M_NONE) {
    unsigned long e = now - oneshotStart;
    if (e >= oneshotLength(oneshot)) {
      oneshot = M_NONE;
    } else {
      if (oneshot == M_BOOT) {
        if (blinkOn(e, BOOT_BLINK_HZ)) fillWant(0, 255, 0);
      } else if (oneshot == M_ESTOP) {
        if (blinkOn(e, ESTOP_BLINK_HZ)) fillWant(255, 0, 0);
      } else {                         // STOP: steady red
        fillWant(255, 0, 0);
      }
      return;
    }
  }

  if (steady == M_CALIBRATING) {
    if (blinkOn(now - steadySince, CALIB_BLINK_HZ)) fillWant(255, 255, 0);
  } else if (steady == M_READY) {
    fillWant(0, 255, 0);
  } else if (steady == M_ASSIST && winPos != NO_WINDOW) {
    for (uint8_t k = 0; k < cfg.width; k++) {
      uint16_t i = (uint16_t)winPos + k;
      if (i >= cfg.num) break;
      want[i][0] = winColor[0];
      want[i][1] = winColor[1];
      want[i][2] = winColor[2];
    }
  }
}

static void pushFrame() {
  for (uint8_t i = 0; i < 4; i++) shiftByte(0x00);              // start frame
  uint8_t header = 0xE0 | cfg.bright;
  for (uint8_t i = 0; i < cfg.num; i++) {
    shiftByte(header);
    shiftByte(want[i][cfg.order[0]]);
    shiftByte(want[i][cfg.order[1]]);
    shiftByte(want[i][cfg.order[2]]);
  }
  // 0x00 end frame: SK9822 reset plus enough clocks for the last APA102 LED.
  uint8_t endBytes = 4 + (cfg.num + 15) / 16;
  for (uint8_t i = 0; i < endBytes; i++) shiftByte(0x00);
  digitalWrite(PIN_DATA, LOW);
}

void setup() {
  pinMode(PIN_DATA, OUTPUT);
  pinMode(PIN_CLOCK, OUTPUT);
  digitalWrite(PIN_DATA, LOW);
  digitalWrite(PIN_CLOCK, LOW);

  Config stored;
  EEPROM.get(0, stored);
  if (configValid(stored)) cfg = stored;

  if (STRIP_SENSE_PIN >= 0) {
    pinMode(STRIP_SENSE_PIN, INPUT);
    stripPowered = false;            // senseStrip() starts the blink once power is seen
  } else {
    startOneshot(M_BOOT, millis());
  }
  Serial.begin(115200);
}

void loop() {
  // Checking `Serial` costs a 10 ms delay inside the USB core on the Micro, so
  // do it twice a second, not every pass: it would otherwise slow every blink.
  static bool greeted = false;
  static unsigned long lastPortCheck = 0;
  if (millis() - lastPortCheck >= 500) {
    lastPortCheck = millis();
    bool open = Serial;
    if (open && !greeted) Serial.println(F("dotstar bridge ready: data D14, clock D15"));
    greeted = open;
  }

  pollSerial();
  unsigned long now = millis();
  senseStrip(now);
  if (steady == M_RAW || !stripPowered) return;

  if ((steady == M_CALIBRATING || steady == M_READY || steady == M_ASSIST)
      && now - lastPacketMs > WATCHDOG_MS) {
    setSteady(M_OFF, now);
    lastRequested = M_NONE;          // the next packet applies again
  }

  render(now);
  uint16_t bytes = (uint16_t)cfg.num * 3;
  if (!shownValid || memcmp(want, shown, bytes) != 0 || now - lastShowMs >= REFRESH_MS) {
    pushFrame();
    memcpy(shown, want, bytes);
    shownValid = true;
    lastShowMs = now;
  }
}
