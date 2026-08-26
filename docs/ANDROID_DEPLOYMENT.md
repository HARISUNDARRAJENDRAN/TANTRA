# Android deployment

## Requirements

- Android Studio with Android SDK 35
- JDK 17
- Android 8.0/API 26 or newer
- Two physical phones for Bluetooth/Wi-Fi end-to-end tests
- One validated `.tantra-pack` model pack per device tier

## Build

```bash
cd android
./gradlew assembleDebug
```

The debug APK is written to `app/build/outputs/apk/debug/app-debug.apk`.

## Device setup

1. Install the APK on both phones.
2. Grant microphone, nearby-device/Bluetooth, and notification permissions.
3. Import the same compatible `.tantra-pack` on both phones.
4. On Wi-Fi, put both phones on the same hotspot or local network. Start **Host** on one phone and enter its LAN address on the other.
5. On Bluetooth, pair the phones first, start the RFCOMM host, and connect using the paired device address.
6. Select the same language and use PTT. Continuous mode removes the button gate but still applies acoustic echo cancellation and turn arbitration.

## Competition measurement

Run a minimum of 100 utterances per language on low- and mid-range phones. Capture:

- microphone end to stable text commit;
- stable commit to packet receive;
- packet receive to first synthesized sample;
- end-of-sentence to remote first audio;
- WER/CER by language and noise condition;
- ASR/TTS real-time factor;
- app RAM, APK/model-pack size, idle CPU and battery drain;
- packet bytes and retransmissions.

Do not report target values as measured values. Store raw JSON/CSV under `benchmarks/device-runs/` and identify the exact APK, model-pack hash, phone, Android version, and transport.
