# CONV2D CPU Runner v1

Enthalten:

- gehärtete `main_conv2d_intel.cpp` und `main_conv2d_amd.cpp`
- `quickcheck_CPU_*_CONV2D.sh`
- `02_run_CPU_*_CONV2D_only.sh`

## Installation

Im Repository-Root:

```bash
cd ~/projects/energy
unzip -o CONV2D_CPU_RUNNERS_v1.zip
chmod +x new/INTEL/scripts/02_run_CPU_Intel_CONV2D_only.sh \
         new/INTEL/scripts/quickcheck_CPU_Intel_CONV2D.sh \
         new/AMD/scripts/02_run_CPU_AMD_CONV2D_only.sh \
         new/AMD/scripts/quickcheck_CPU_AMD_CONV2D.sh
```

Das ZIP enthält Pfade ab `new/` und überschreibt nur die sechs enthaltenen Dateien.

## Intel-Quickcheck

```bash
cd ~/projects/energy/new/INTEL
sudo bash scripts/01_enable_CPU_Intel.sh
bash scripts/quickcheck_CPU_Intel_CONV2D.sh
```

Der Quickcheck stellt die CPU-Einstellungen anschließend wieder her und fährt nicht herunter.

## Intel-Kampagne

Nach einem separaten Quickcheck muss der Messzustand erneut aktiviert werden:

```bash
cd ~/projects/energy/new/INTEL
sudo bash scripts/01_enable_CPU_Intel.sh
bash scripts/02_run_CPU_Intel_CONV2D_only.sh
```

Der `02`-Runner führt selbst nochmals den Quickcheck aus, misst danach fünf Sessions und fährt nach vollständig erfolgreicher Validierung herunter.

## AMD-Quickcheck

```bash
cd ~/projects/energy/new/AMD
sudo bash scripts/01_enable_CPU_AMD.sh
bash scripts/quickcheck_CPU_AMD_CONV2D.sh
```

## AMD-Kampagne

```bash
cd ~/projects/energy/new/AMD
sudo bash scripts/01_enable_CPU_AMD.sh
bash scripts/02_run_CPU_AMD_CONV2D_only.sh
```

Der AMD-Runner hält die `sudo`-Berechtigung bis nach dem Restore aktiv. Er stellt Governors, EPP und `perf_event_paranoid` zuerst wieder her und ruft nur danach `systemctl poweroff` auf.

## Testlauf ohne Herunterfahren

```bash
POWER_OFF_AT_END=0 bash scripts/02_run_CPU_AMD_CONV2D_only.sh
```

oder entsprechend für Intel.

## Messmatrix

Quickcheck je CPU:

- Shapes `1..6`
- Threads Intel `{1,20}` / AMD `{1,64}`
- zwei Wiederholungen
- 24 validierte CSV-Zeilen
- oneDNN-Verbose
- exklusiver Anti-Collapse-Test für Shape 1

Offizielle Kampagne:

- fünf Sessions
- zehn Wiederholungen
- Intel: 6 Shapes × 7 Threadzahlen × 10 = 420 Zeilen je Session
- AMD: 6 Shapes × 9 Threadzahlen × 10 = 540 Zeilen je Session
- 60 Sekunden Pause zwischen Sessions

## Sicherheitslogik

Automatischer Poweroff erfolgt ausschließlich, wenn:

1. Quickcheck-Matrix bestanden ist,
2. Anti-Collapse bestanden ist,
3. alle fünf Sessions vollständig validiert wurden,
4. Restore erfolgreich war,
5. `POWER_OFF_AT_END=1` gilt.

Bei Build-, Mess-, Checksum-, CSV-, Restore- oder Validierungsfehlern wird nicht automatisch heruntergefahren.
