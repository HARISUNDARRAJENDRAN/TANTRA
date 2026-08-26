#pragma once

#define TANTRA_DEVICE_NAME "TANTRA-Bridge"
#define TANTRA_WIFI_SSID "TANTRA-Bridge"
// Use a unique password of at least 12 characters in include/config.h.
#define TANTRA_WIFI_PASSWORD "replace-this-password"
#define TANTRA_TCP_PORT 47821

// Optional UART radio/modem. Set to 0 to disable.
#define TANTRA_RADIO_ENABLED 1
#define TANTRA_RADIO_RX_PIN 16
#define TANTRA_RADIO_TX_PIN 17
#define TANTRA_RADIO_BAUD 9600
