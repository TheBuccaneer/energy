# STREAM Triad GPU Benchmark - RTX 5050 Dokumentation

## Überblick

Diese Dokumentation beschreibt die Implementierung eines **STREAM Triad Memory Bandwidth Benchmarks** für NVIDIA GPUs mit:
- **NVML Energy Measurement** (TotalEnergyConsumption API)
- **Größen-Sweep** (7 Größen für FP32, 6 für FP64)
- **50 Runs pro Größe** für statistische Robustheit
- **CSV-Schema kompatibel** mit reduction Benchmarks
- **Kernel-only Timing** via CUDA Events

---

## Was ist STREAM Triad?

### Die Operation

```c
a[i] = b[i] + q * c[i]    // q = scalar (z.B. 3.0)
```

### Speicherzugriffe pro Iteration

- **2 Reads:** `b[i]`, `c[i]` aus Global Memory
- **1 Write:** `a[i]` nach Global Memory
- **Total: 12 Bytes (FP32)** oder **24 Bytes (FP64)**

### Ziel

**DRAM-Bandbreite** der GPU messen, **nicht** Cache-Bandbreite.

**STREAM-Konvention:**
```
Bandwidth [GB/s] = (Bytes pro Iter × N × Passes) / Time / 1e9
```

---

## Kritische STREAM-Regeln für GPU

### 1. Array-Größe muss L2-Cache überschreiten

**Problem:** Moderne GPUs haben große L2-Caches:
- RTX 3090: 6 MB L2
- RTX 4090: 72 MB L2  
- RTX 5050: ~20-40 MB L2 (geschätzt)

**Regel:** Arrays sollten **deutlich größer** als L2 sein.

**Unsere Size-Liste:**
```cpp
// FP32 (4 bytes per element)
2^20 = 1M   →   4 MB pro Array  →  12 MB total  ⚠️  Cache!
2^22 = 4M   →  16 MB pro Array  →  48 MB total  ⚠️  Teilweise Cache
2^24 = 16M  →  64 MB pro Array  → 192 MB total  ✓  DRAM-Plateau
2^26 = 64M  → 256 MB pro Array  → 768 MB total  ✓  Sauberes DRAM
2^27 = 128M → 512 MB pro Array  → 1.5 GB total  ✓  Sauberes DRAM
2^28 = 256M →   1 GB pro Array  → 3.0 GB total  ✓  Sauberes DRAM
2^29 = 512M →   2 GB pro Array  → 6.0 GB total  ✓  Max für 8GB GPU
```

**Erwartete Bandbreiten (RTX 5050, geschätzt ~256-bit Bus, GDDR6):**
- N = 2^20: **~1200 GB/s** (L2-Cache!)
- N = 2^22: **~400-800 GB/s** (Mix)
- N = 2^24-2^29: **~200-350 GB/s** (DRAM-Plateau)

### 2. Warum kleine Größen so unrealistisch sind

**Bei N=2^20 (1M Elemente):**
```
Total Memory: 3 × 1M × 4 bytes = 12 MB
```

Wenn L2-Cache = 30 MB → **alle Arrays passen komplett in L2!**

**Ergebnis:** Du misst L2-Cache-Bandbreite (~1000-2000 GB/s), **nicht** DRAM.

**Für saubere DRAM-Messung:** Verwende N ≥ 2^26 (64M).

---

## Implementierungs-Details

### Grid-Stride Kernel

```cuda
__global__ void triad_kernel(real* __restrict__ a, 
                            const real* __restrict__ b, 
                            const real* __restrict__ c, 
                            real q, 
                            size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < N; i += stride) {
        a[i] = b[i] + q * c[i];
    }
}
```

**Grid-Stride Loop:** Erlaubt beliebige N-Werte mit fester Grid-Größe.

### Launch Configuration

```cpp
int grid_nom = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
int grid_size = min(grid_nom, sm_count * 32);
const int BLOCK_SIZE = 256;
```

**SM-basiertes Limit:** Grid wird auf `SM × 32` begrenzt, um Overhead zu vermeiden.

**Beispiel RTX 5050 (angenommen 20 SMs):**
```
grid_size = min(grid_nom, 20 × 32) = min(grid_nom, 640)
```

---

## Zeitmessung - Nur Kernel!

### CUDA Events um Iterationsschleife

```cpp
cudaEventRecord(start_event, stream);

for (size_t p = 0; p < passes; p++) {
    triad_kernel<<<grid, block, 0, stream>>>(...);
}

cudaEventRecord(stop_event, stream);
cudaEventSynchronize(stop_event);

float ms;
cudaEventElapsedTime(&ms, start_event, stop_event);
double time_s = ms / 1000.0;
```

**Wichtig:**
- **Keine H2D/D2H Transfers** im Messfenster
- Events umschließen **nur Kernel-Loop**
- `memsetAsync` für Array-Reset **vor** Start-Event

### Kalibrierung mit Warm-up

```cpp
// 1. Untimed Warm-up (1 Pass)
triad_kernel<<<...>>>();
cudaStreamSynchronize(stream);

// 2. Timed Calibration (1 Pass)
cudaEventRecord(start);
triad_kernel<<<...>>>();
cudaEventRecord(stop);
cudaEventSynchronize(stop);

// 3. Berechne Passes für ~1.0s
float t_pass = ...;
passes = ceil((TARGET_S / t_pass) * SAFETY_FACTOR);
```

**Safety Factor 1.02:** Stellt sicher dass Zeit ≥ 1.0s (aktuell zu niedrig, besser 1.10).

---

## NVML Energy Measurement

### TotalEnergyConsumption API

```cpp
// Vor Kernel
unsigned long long energy_start_mj = 0;
nvmlDeviceGetTotalEnergyConsumption(device, &energy_start_mj);

cudaEventRecord(start);
// ... Kernel Loop ...
cudaEventSynchronize(stop);

// Nach Kernel
unsigned long long energy_stop_mj = 0;
nvmlDeviceGetTotalEnergyConsumption(device, &energy_stop_mj);

// Delta in Joule
double energy_j = (energy_stop_mj - energy_start_mj) / 1000.0;
```

**Wichtig:**
- Werte sind in **Millijoule (mJ)**
- **Kumulativ seit Driver-Reload** (nicht pro Messung!)
- **Immer Differenzen bilden:** `E_stop - E_start`
- Overflow möglich bei sehr langen Uptimes

### Power berechnen

```cpp
double avg_power_w = energy_j / time_s;
```

**Beispiel RTX 5050:**
- Energy: 85 J
- Time: 0.78 s
- Power: **109 W** (typisch unter Last)

---

## Größen-Sweep Logik

### OOM-Safety Check

```cpp
bool checkVRAMAvailable(size_t N) {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    
    size_t required = 3 * N * sizeof(real);
    size_t safe_threshold = free_bytes * 0.85;  // 85% Limit
    
    return required <= safe_threshold;
}
```

**85% Schwelle:** Vermeidet OOM durch:
- CUDA-Context Overhead
- Display-Memory
- Fragmentierung

**Bei Skip:** CSV-Zeile mit `notes="skip_oom"` wird geschrieben.

### Loop-Struktur

```cpp
for (size_t N : SIZES) {
    // 1. VRAM-Check
    if (!checkVRAMAvailable(N)) {
        writeSkippedRow(...);
        continue;
    }
    
    // 2. Allokiere A, B, C
    cudaMalloc(...);
    
    // 3. Kalibriere (mit Warm-up)
    passes = calibrateWithWarmup(...);
    
    // 4. REPEATS Messungen
    for (int run = 0; run < REPEATS; run++) {
        // Reset + Messen + CSV
    }
    
    // 5. Cleanup
    cudaFree(...);
}
```

**Pro Größe:** Genau 50 CSV-Zeilen (= REPEATS).

---

## CSV-Schema

### Header (identisch mit reduction)

```csv
timestamp,host,gpu_name,matrix_size,mode,batches,seconds_target,
seconds_gpu,seconds_wall,energy_j,avg_power_w,below_target,workload,
impl,dtype,N,passes_kernel,passes_e2e,seconds_kernel,energy_kernel_j,
avg_power_w_kernel,avg_power_w_e2e,bytes_total,bw_gb_s,time_mode,
energy_mode,includes_transfer,device_name,driver_version,
pcie_gen_current,pcie_width_current,pcie_rx_kbs,pcie_tx_kbs,
clocks_sm_mhz,clocks_mem_mhz,temp_c,throttle_reasons,notes
```

### Beispiel-Zeilen

```csv
2025-10-13T12:34:56,rock-ms7a95,NVIDIA GeForce RTX 5050,0,kernel,0,1.00,0.7816,0.7816,91.0,116.0,1,stream_triad,cuda,fp32,1048576,75081,75081,0.7816,91.000,116.0,116.0,945972000000,1208.78,kernel,kernel,0,NVIDIA GeForce RTX 5050,580.65.06,0,0,0,0,2925,9801,53,0,sweep;repeats=50;passes=75081;grid=640;block=256;dtype=fp32
```

### Wichtige Felder

| Feld | Beispiel | Bedeutung |
|------|----------|-----------|
| `N` | `1048576` | Vektor-Größe (Elemente) |
| `passes_kernel` | `75081` | Anzahl Kernel-Calls |
| `seconds_kernel` | `0.7816` | Kernel-Zeit (s) |
| `energy_kernel_j` | `91.0` | Kernel-Energie (J) |
| `avg_power_w_kernel` | `116.0` | Durchschnitts-Power (W) |
| `bytes_total` | `945972000000` | 12 × N × passes |
| `bw_gb_s` | `1208.78` | Bandbreite (GB/s) |
| `clocks_sm_mhz` | `2925` | GPU Core Clock |
| `clocks_mem_mhz` | `9801` | Memory Clock (effektiv) |
| `temp_c` | `53` | GPU Temperatur |
| `throttle_reasons` | `0` | Throttle-Flags (0 = kein Throttle) |

### Notes-Format

```
sweep;repeats=50;passes=75081;grid=640;block=256;dtype=fp32
```

Bei Fehlern:
```
sweep;repeats=50;...;no_energy     # NVML nicht verfügbar
skip_oom                            # Size übersprungen
```

---

## Compilation & Usage

### FP32 Version

```bash
nvcc -O3 -std=c++17 -lnvidia-ml -o stream_triad_sweep main.cu
./stream_triad_sweep
```

### FP64 Version

```bash
nvcc -O3 -std=c++17 -DUSE_DOUBLE -lnvidia-ml -o stream_triad_sweep_fp64 main.cu
./stream_triad_sweep_fp64
```

**Wichtig:** Das `-lnvidia-ml` Flag ist **zwingend** für NVML!

### Ausgabe

```
========================================
STREAM Triad - Size Sweep + NVML Energy
========================================
Device:         NVIDIA GeForce RTX 5050
Compute Cap:    9.0
Driver:         580.65.06
SMs:            20
Total Memory:   8.00 GB
Data type:      fp32 (4 bytes)
Bytes/iter:     12 (STREAM)
Target runtime: 1.0s
Repeats/size:   50
Sizes:          7
Output:         data/raw/stream_triad_sweep.csv
========================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Size 1/7: N = 1048576 (2^20)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Calibrating (with warm-up)... 75081 passes (grid=640)
Measuring 50 runs...
  [ 1/50] 0.7816s E=91.0J P=116W | BW=1208.78 GB/s
  [10/50] 0.7894s E=90.8J P=115W | BW=1196.75 GB/s
  ...
  [50/50] 0.7923s E=90.7J P=114W | BW=1192.38 GB/s
✓ Complete
```

---

## Erwartete Ergebnisse - RTX 5050

### Theoretische Spezifikationen (geschätzt)

| Parameter | Wert |
|-----------|------|
| Memory Bus | 128-bit |
| Memory Type | GDDR6 |
| Memory Clock | ~9500 MHz (effektiv ~19 GT/s) |
| **Theoretische BW** | **~304 GB/s** |

```
BW = (Bus Width / 8) × Memory Clock × 2 (DDR)
   = (128 / 8) × 9500 MHz × 2
   = 16 bytes × 9500 × 2
   = 304,000 MB/s = 304 GB/s
```

### STREAM Achievable (80-95% von Theorie)

**Erwartung:**
```
STREAM BW ≈ 80-95% × 304 GB/s = 243-289 GB/s
```

### Tatsächliche Messungen nach Größe

| N (2^x) | Array Size | Total Mem | Erwartete BW | Was gemessen wird |
|---------|------------|-----------|--------------|-------------------|
| 2^20 (1M) | 4 MB | 12 MB | **~1200 GB/s** ⚠️ | **L2-Cache!** |
| 2^22 (4M) | 16 MB | 48 MB | ~400-800 GB/s | Mix (L2 + DRAM) |
| 2^24 (16M) | 64 MB | 192 MB | ~250-300 GB/s | DRAM-Beginn |
| 2^26 (64M) | 256 MB | 768 MB | **~250-280 GB/s** ✓ | **Sauberes DRAM** |
| 2^27 (128M) | 512 MB | 1.5 GB | **~250-280 GB/s** ✓ | **Plateau** |
| 2^28 (256M) | 1 GB | 3 GB | **~250-280 GB/s** ✓ | **Plateau** |
| 2^29 (512M) | 2 GB | 6 GB | **~250-280 GB/s** ✓ | **Plateau** |

### Power & Energy

**Typische Werte unter Last:**
- **Idle:** ~10-15 W
- **STREAM Load:** ~100-130 W
- **Peak (Boost):** ~150 W (kurzzeitig)

**Pro Run (1.0s):**
- Energy: 80-100 J
- Power: 80-130 W (variiert mit Boost)

---

## Interpretation der Ergebnisse

### 1. Cache vs. DRAM erkennen

**Problem bei deinen aktuellen Messungen:**
```
N = 2^20: 1217 GB/s  ← Viel zu hoch!
N = 2^22:  290 GB/s  ← Immer noch etwas hoch
```

**Lösung:** Fokus auf **N ≥ 2^26** für valide DRAM-Messungen.

### 2. Plateau-Region identifizieren

**Erwartetes Verhalten:**
```
BW (GB/s)
  1200 |     ●              (2^20, L2-Cache)
       |
   800 |
       |
   400 |        ●           (2^22, Mix)
       |
   280 |           ●━●━●━●  (2^24-2^29, DRAM-Plateau)
       |
     0 +━━━━━━━━━━━━━━━━━━━→ N
       20   22   24   26   28   (log2 scale)
```

**DRAM-Plateau** bei N ≥ 2^26 ist der valide STREAM-Wert.

### 3. Varianz über Runs

**Gut:** CV < 5% (Coefficient of Variation)
```
CV = (Std Dev / Mean) × 100%
```

**Bei hoher Varianz (>10%) prüfen:**
- Throttling? (`throttle_reasons != 0`)
- Thermal Issues? (Temp > 80°C)
- Background Load? (andere GPU-Prozesse)

### 4. Zeit vs. Target

**Aktuelles Problem:**
```
Target: 1.00s
Actual: 0.77s  ← 23% zu kurz!
```

**Fix:** `SAFETY_FACTOR` erhöhen:
```cpp
static constexpr double SAFETY_FACTOR = 1.10;  // statt 1.02
```

---

## Troubleshooting

### Problem: Bandbreite viel zu hoch (>1000 GB/s)

**Ursache:** Arrays passen in L2-Cache.

**Lösung:**
- Ignoriere N < 2^24
- Berichte nur N ≥ 2^26 als DRAM-BW
- Erwähne in Paper: "Small sizes measure cache hierarchy"

### Problem: NVML Error "Not Supported"

**Ursache:** Alte Treiber oder Laptop-GPU.

**Check:**
```bash
nvidia-smi -q | grep "Total Energy"
```

**Lösung:**
- Update auf Driver ≥ 525
- Bei Laptop: Energie-Messung oft nicht verfügbar
- Code schreibt dann `notes=";no_energy"`

### Problem: Zeit < 1.0s trotz Kalibrierung

**Ursache:** `SAFETY_FACTOR` zu niedrig (1.02).

**Fix in Code:**
```cpp
static constexpr double SAFETY_FACTOR = 1.10;  // +10% Buffer
```

### Problem: OOM bei 2^29 (FP32)

**Ursache:** Display + Context belegen VRAM.

**Check:**
```bash
nvidia-smi
```

**Lösung:**
- Code skippt automatisch bei <85% Free
- Für Tests: Xorg beenden oder headless booten

### Problem: Hohe Varianz zwischen Runs

**Checken:**
```bash
# GPU-Frequenz locked?
nvidia-smi -q -d CLOCK

# Thermal?
nvidia-smi dmon -c 1

# Governor?
cat /sys/class/drm/card*/device/power_dpm_force_performance_level
```

**Stabilisierung:**
```bash
# Persistence Mode (Root)
sudo nvidia-smi -pm 1

# Lock Clocks (optional)
sudo nvidia-smi -lgc 2700  # Lock GPU clock

# Max Power
sudo nvidia-smi -pl 150    # Max Power Limit (W)
```

---

## Wichtige Metriken für Paper

### Pro Size berichten

**Für jedes N ≥ 2^26:**

| Metrik | Formel | Einheit |
|--------|--------|---------|
| Median BW | `median(bw_gb_s)` | GB/s |
| Median Energy | `median(energy_j)` | J |
| Median Power | `median(power_w)` | W |
| EDP | `median(energy_j × time_s)` | J·s |
| CV (Time) | `std(time_s) / mean(time_s)` | % |
| CV (Energy) | `std(energy_j) / mean(energy_j)` | % |

### Aggregierte Statistik

**DRAM-BW (N ≥ 2^26):**
```
Mean BW = mean(bw über alle Runs bei N ≥ 2^26)
% of Peak = (Mean BW / Theoretical BW) × 100%
```

**Erwartung für RTX 5050:**
```
Mean BW ≈ 260 GB/s
% of Peak ≈ 85% (von 304 GB/s)
```

---

## Configuration Tuning

### Wenn Zeit-Target ständig verfehlt wird

**Aktuell:**
```cpp
static constexpr double SAFETY_FACTOR = 1.02;  // +2%
```

**Besser für ≥1.0s:**
```cpp
static constexpr double SAFETY_FACTOR = 1.15;  // +15%
```

### Wenn zu lange dauert (Testing)

**Für schnelle Tests:**
```cpp
static constexpr int REPEATS = 10;  // statt 50
```

**Für finale Messungen:**
```cpp
static constexpr int REPEATS = 50;  // wie im Projektplan
```

### Wenn VRAM knapp (6GB GPU)

**FP64 Size-Liste anpassen:**
```cpp
#ifdef USE_DOUBLE
static constexpr size_t SIZES[] = {
    1ULL << 20, 1ULL << 22, 1ULL << 24,
    1ULL << 26, 1ULL << 27  // Stopp bei 2^27 statt 2^28
};
#endif
```

---

## Vergleich mit anderen Benchmarks

### BabelStream

**Unterschiede:**
- BabelStream: Allgemeiner, mehrere Operationen (Copy, Mul, Add, Triad)
- Unser Code: Nur Triad, aber mit Energy + Größen-Sweep

**Vorteile unseres Codes:**
- NVML Energy direkt integriert
- CSV-kompatibel mit reduction
- Größen-Sweep automatisch
- 50 Repeats für Power-Analyse

### NVIDIA STREAM (nvidia-smi)

**Unterschiede:**
- `nvidia-smi`: Sampling-basiert, weniger präzise
- Unser Code: CUDA Events + NVML TotalEnergy (präziser)

---

## Best Practices für Messungen

### 1. System vorbereiten

```bash
# GPU in Persistence Mode
sudo nvidia-smi -pm 1

# Background minimieren
# - Schließe Browser (WebGL)
# - Stoppe Compositor (wenn möglich)

# Thermal: Warte bis Idle-Temp erreicht
watch -n 1 nvidia-smi
```

### 2. Mehrere Sessions messen

**Empfehlung:** 3-5 komplette Durchläufe an verschiedenen Tagen/Zeiten.

**Warum?**
- Driver-Resets → TotalEnergy Counter reset
- Thermal Drift über Tageszeit
- Background-Load variiert

### 3. Metadaten loggen

**Vor jeder Session:**
```bash
# System-Info
nvidia-smi -q > session_info.txt
uname -a >> session_info.txt
cat /proc/cpuinfo | grep "model name" | head -1 >> session_info.txt

# Timestamp
date -Iseconds >> session_info.txt
```

---

## Datei-Struktur

```
project/
├── main.cu                              # Benchmark-Code
├── data/
│   └── raw/
│       ├── stream_triad_sweep.csv       # FP32 Ergebnisse
│       └── stream_triad_sweep_fp64.csv  # FP64 Ergebnisse (optional)
├── docs/
│   └── STREAM_GPU_5050.md               # Diese Doku
└── analysis/
    └── plot_stream.py                   # Visualisierung
```

---

## Beispiel-Analyse-Code (Python)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/raw/stream_triad_sweep.csv')

# Filter kernel mode only
df_kernel = df[df['mode'] == 'kernel']

# Group by N, compute median
summary = df_kernel.groupby('N').agg({
    'bw_gb_s': 'median',
    'energy_kernel_j': 'median',
    'avg_power_w_kernel': 'median',
    'seconds_kernel': 'median'
}).reset_index()

# Plot BW vs N
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(summary['N'], summary['bw_gb_s'], 'o-', linewidth=2)
ax.axhline(304, color='red', linestyle='--', label='Theoretical Peak (304 GB/s)')
ax.axhline(304 * 0.85, color='orange', linestyle='--', label='85% of Peak')
ax.set_xscale('log', base=2)
ax.set_xlabel('N (elements)', fontsize=12)
ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
ax.set_title('STREAM Triad - RTX 5050', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('stream_bw_vs_n.png', dpi=300)
plt.show()
```

---

## Zusammenfassung - Kernpunkte

### ✅ Was funktioniert

- NVML Energy Measurement (TotalEnergy API)
- Größen-Sweep mit OOM-Protection
- 50 Runs pro Size für statistische Robustheit
- CSV-Schema kompatibel mit reduction
- Kernel-only Timing (keine PCIe)

### ⚠️ Was zu beachten ist

- **Kleine Sizes (N < 2^24) messen Cache, nicht DRAM!**
- DRAM-Plateau erst bei N ≥ 2^26 sichtbar
- Safety Factor 1.02 zu niedrig → Zeit oft < 1.0s
- NVML TotalEnergy kumulativ → immer Differenzen!

### 📊 Erwartete RTX 5050 Ergebnisse

- **Theoretische BW:** ~304 GB/s
- **STREAM Achievable:** ~250-280 GB/s (82-92%)
- **Power unter Last:** ~100-130 W
- **Energy pro Run (1s):** ~80-100 J

### 🎯 Für Paper verwenden

**Berichte nur N ≥ 2^26:**
- Median BW: ~270 GB/s
- % of Peak: ~89%
- Median Power: ~115 W
- CV: <3% (sehr stabil)

**Erwähne kleine Sizes als:**
"Cache hierarchy characterization (N < 2^24 measures L2 bandwidth)"

---

## Referenzen

1. **STREAM Benchmark:** https://www.cs.virginia.edu/stream/
2. **NVML API:** https://docs.nvidia.com/deploy/nvml-api/
3. **CUDA Events:** https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
4. **BabelStream:** https://github.com/UoB-HPC/BabelStream
5. **Grid-Stride Loops:** https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

---

**Dokumentation Version:** 1.0  
**Datum:** Oktober 2025  
**Autor:** Benchmark Implementierung für Projektplan v4  
**GPU:** NVIDIA GeForce RTX 5050  
**Driver:** 580.65.06
