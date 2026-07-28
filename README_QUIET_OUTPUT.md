# CONV2D CPU quiet-output overlay v4

Diese Version reduziert ausschließlich die Konsolen- und Logausgabe des normalen
Conv2D-Laufs. Mathematische Semantik, Shapes, CSV-Schema, RAPL-Grenze,
oneDNN-C-Execute-Pfad, Scratchpad, Checksum-Gate und Shutdown-Verhalten bleiben
unverändert.

## Normale Source-Ausgabe

```text
CONV2D | <CPU> | platform=AMD|INTEL | implementation=onednn_convolution_auto | session=... | reps=... | configs=... | DRAM-RAPL=...
[CALIBRATION] shape=... threads=... batches=...
[CONV2D] shape=... threads=... rep=... batches=... e2e_time_s=... kernel_time_s=... device_energy_j=... avg_power_w=... runtime_status=... checksum=OK
```

Ausführliche `[CONFIG]`, `[ONEDNN]`, `[ENV]` und `[CHECKSUM]`-Diagnostik wird nur
noch mit `CONV2D_DIAGNOSTICS=1` erzeugt. Der Runner setzt dies ausschließlich für
den separaten, in eine Datei umgeleiteten oneDNN-Preflight.

## Installation

```bash
cd ~/projects/energy
unzip -o ~/Downloads/CONV2D_CPU_QUIET_OUTPUT_v4.zip

chmod +x \
  new/AMD/scripts/02_run_CPU_AMD_CONV2D_only.sh \
  new/AMD/scripts/quickcheck_CPU_AMD_CONV2D.sh \
  new/INTEL/scripts/02_run_CPU_Intel_CONV2D_only.sh \
  new/INTEL/scripts/quickcheck_CPU_Intel_CONV2D.sh
```

Danach auf AMD:

```bash
cd ~/projects/energy/new/AMD/scripts
sudo ./01_enable_CPU_AMD.sh
./02_run_CPU_AMD_CONV2D_only.sh
```

Nur `01_enable` mit `sudo` starten.
