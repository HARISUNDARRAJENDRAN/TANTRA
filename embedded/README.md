# ESP32 TANTRA bridge

This optional firmware relays length-prefixed AksharaLink frames between:

- Android Bluetooth Classic SPP;
- an ESP32 local Wi-Fi access point/TCP socket; and
- a UART-connected low-rate radio or modem.

It does not run ASR/TTS and does not inspect or store transcripts. The bridge is useful when each phone connects to a nearby radio node while the inter-node hop is a constrained serial modem, LoRa packet modem, or another custom link.

## Build

```bash
cd embedded/esp32-tantra-bridge
cp include/config.example.h include/config.h
# Edit the AP password and radio pins/baud.
pio run
pio run --target upload
```

The default UART rate is 9,600 bit/s. TANTRA's token payload normally occupies only tens of bytes per committed clause, although framing, retransmission, and encryption overhead must be included in the real link budget.

## Security

Change the AP password. Production field use must add application-layer AES-GCM/peer authentication; Bluetooth pairing and an AP password alone are not sufficient against a determined nearby attacker. The bridge forwards encrypted payloads without needing the session key.
