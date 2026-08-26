#include <Arduino.h>
#include <BluetoothSerial.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiServer.h>

#if __has_include("config.h")
#include "config.h"
#else
#include "config.example.h"
#endif

namespace {
constexpr uint32_t kMaxFrameBytes = 70000;
constexpr uint32_t kIoTimeoutMs = 80;

BluetoothSerial bluetooth;
WiFiServer server(TANTRA_TCP_PORT);
WiFiClient wifiClient;
HardwareSerial radio(2);

bool readExact(Stream &stream, uint8_t *target, size_t length, uint32_t timeoutMs) {
  const uint32_t started = millis();
  size_t offset = 0;
  while (offset < length && millis() - started < timeoutMs) {
    const int available = stream.available();
    if (available <= 0) {
      delay(1);
      continue;
    }
    const size_t chunk = min(length - offset, static_cast<size_t>(available));
    const size_t received = stream.readBytes(reinterpret_cast<char *>(target + offset), chunk);
    if (received == 0) return false;
    offset += received;
  }
  return offset == length;
}

bool readFrame(Stream &stream, std::unique_ptr<uint8_t[]> &payload, uint32_t &size) {
  if (stream.available() < 4) return false;
  uint8_t header[4];
  if (!readExact(stream, header, sizeof(header), kIoTimeoutMs)) return false;
  size = (static_cast<uint32_t>(header[0]) << 24) |
         (static_cast<uint32_t>(header[1]) << 16) |
         (static_cast<uint32_t>(header[2]) << 8) |
         static_cast<uint32_t>(header[3]);
  if (size == 0 || size > kMaxFrameBytes) {
    Serial.printf("Rejected invalid frame length: %u\n", size);
    return false;
  }
  payload.reset(new (std::nothrow) uint8_t[size]);
  if (!payload) {
    Serial.printf("Unable to allocate %u-byte frame\n", size);
    return false;
  }
  return readExact(stream, payload.get(), size, max(kIoTimeoutMs, size * 1000UL / TANTRA_RADIO_BAUD * 12UL));
}

bool writeFrame(Stream &stream, const uint8_t *payload, uint32_t size) {
  uint8_t header[4] = {
      static_cast<uint8_t>(size >> 24), static_cast<uint8_t>(size >> 16),
      static_cast<uint8_t>(size >> 8), static_cast<uint8_t>(size)};
  return stream.write(header, sizeof(header)) == sizeof(header) &&
         stream.write(payload, size) == size;
}

void forward(Stream &source, bool toBluetooth, bool toWifi, bool toRadio) {
  std::unique_ptr<uint8_t[]> payload;
  uint32_t size = 0;
  if (!readFrame(source, payload, size)) return;
  if (toBluetooth && bluetooth.hasClient()) writeFrame(bluetooth, payload.get(), size);
  if (toWifi && wifiClient && wifiClient.connected()) writeFrame(wifiClient, payload.get(), size);
#if TANTRA_RADIO_ENABLED
  if (toRadio) writeFrame(radio, payload.get(), size);
#else
  (void)toRadio;
#endif
  Serial.printf("Forwarded %u bytes\n", size);
}
}  // namespace

void setup() {
  Serial.begin(115200);
  delay(200);
  if (!bluetooth.begin(TANTRA_DEVICE_NAME)) {
    Serial.println("Bluetooth initialization failed");
  }
  WiFi.mode(WIFI_AP);
  if (!WiFi.softAP(TANTRA_WIFI_SSID, TANTRA_WIFI_PASSWORD)) {
    Serial.println("Wi-Fi AP initialization failed");
  }
  server.begin();
  server.setNoDelay(true);
#if TANTRA_RADIO_ENABLED
  radio.begin(TANTRA_RADIO_BAUD, SERIAL_8N1, TANTRA_RADIO_RX_PIN, TANTRA_RADIO_TX_PIN);
#endif
  Serial.printf("TANTRA bridge ready: BT=%s WiFi=%s IP=%s TCP=%d\n", TANTRA_DEVICE_NAME,
                TANTRA_WIFI_SSID, WiFi.softAPIP().toString().c_str(), TANTRA_TCP_PORT);
}

void loop() {
  if (!wifiClient || !wifiClient.connected()) {
    WiFiClient candidate = server.available();
    if (candidate) {
      if (wifiClient) wifiClient.stop();
      wifiClient = candidate;
      wifiClient.setNoDelay(true);
      Serial.println("Wi-Fi peer connected");
    }
  }

  if (bluetooth.available() >= 4) forward(bluetooth, false, true, true);
  if (wifiClient && wifiClient.connected() && wifiClient.available() >= 4) {
    forward(wifiClient, true, false, true);
  }
#if TANTRA_RADIO_ENABLED
  if (radio.available() >= 4) forward(radio, true, true, false);
#endif
  delay(1);
}
