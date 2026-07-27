# RTX 5060 Ti — GPU AXPY runners

Enthalten:

- `scripts/quickcheck_GPU_5060ti_AXPY.sh`
- `scripts/02_run_GPU_5060ti_AXPY_only.sh`

Voraussetzung:

- CUDA-Source:
  `~/projects/energy/new/5060ti/scripts/AXPY/main_axpy.cu`
- erwarteter Source-Hash:
  `c4e3099929f736b3fe101a10b870f7e677f0b6c28e13be4577aeb1a376c18f93`
- vorhandene Geräteskripte:
  `scripts/01_enable_GPU_5060ti.sh`
  `scripts/03_disable_GPU_5060ti.sh`

## Installation

```bash
cd ~/projects/energy/new/5060ti
unzip -o ~/Downloads/GPU_AXPY_5060ti_RUNNERS_FINAL.zip
chmod +x scripts/quickcheck_GPU_5060ti_AXPY.sh
chmod +x scripts/02_run_GPU_5060ti_AXPY_only.sh
```

## Quickcheck

```bash
cd ~/projects/energy/new/5060ti
sudo bash scripts/01_enable_GPU_5060ti.sh
bash scripts/quickcheck_GPU_5060ti_AXPY.sh
```

Der Quickcheck:

- kompiliert `main_axpy.cu`
- erwartet `RTX 5060 Ti`
- misst `N={1M, 64M, 256M}`
- nutzt 2 Wiederholungen
- aktiviert das Anti-Collapse-Gate auf `kernel_time_s`
- validiert die 45-spaltige CSV
- fährt den Rechner nicht herunter
- stellt die GPU-Einstellungen nicht automatisch wieder her

## Offizielle Kampagne

Erst nach bestandenem Quickcheck:

```bash
cd ~/projects/energy/new/5060ti
sudo bash scripts/01_enable_GPU_5060ti.sh

POWER_OFF_AT_END=0 \
bash scripts/02_run_GPU_5060ti_AXPY_only.sh
```

Die offizielle Kampagne:

- startet keinen Quickcheck
- misst 5 Sessions
- misst 10 Wiederholungen je Konfiguration
- nutzt 9 Größen von 1M bis 256M
- pausiert 60 Sekunden zwischen Sessions
- validiert jede Session
- stellt die GPU-Einstellungen bei Ende, Fehler oder Ctrl+C wieder her
- fährt nur mit `POWER_OFF_AT_END=1` nach erfolgreichem Abschluss herunter

## Prüfsummen

```text
da90c4578a2471a9baf345529dbccf901cb0ca864718b333b97403ccec502761  scripts/quickcheck_GPU_5060ti_AXPY.sh
6f3c0478aad1dcf321daa456788d3343cd605d787d8d45d552fe1286f4291883  scripts/02_run_GPU_5060ti_AXPY_only.sh
```
