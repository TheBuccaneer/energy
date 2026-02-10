#!/bin/bash

# CPU Stabilisierungs-Script für AMD Ryzen/Threadripper
# Fixiert CPU-Frequenz und wärmt auf für konsistente Energiemessungen

echo "=== CPU Stabilization (AMD) ==="

# Dynamische Frequenz-Erkennung für AMD
# AMD CPUs haben oft kein "@X.XGHz" im Model-String, daher direkt cpuinfo_max_freq nutzen
BASE_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/base_frequency 2>/dev/null)

if [ -z "$BASE_FREQ" ] || [ "$BASE_FREQ" -eq 0 ]; then
    # Fallback: Nutze cpuinfo_max_freq (nicht min!)
    BASE_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null)
fi

# Wenn immer noch leer/0, versuche aus lscpu zu parsen
if [ -z "$BASE_FREQ" ] || [ "$BASE_FREQ" -eq 0 ]; then
    BASE_FREQ=$(lscpu | grep "Model name" | grep -oP '@\s*\K[0-9.]+' | awk '{print int($1*1000000)}')
fi

# Letzter Fallback: Für Threadripper 3970X bekannte Base-Freq
if [ -z "$BASE_FREQ" ] || [ "$BASE_FREQ" -eq 0 ]; then
    echo "Warning: Could not detect base frequency, using 3.7 GHz (Threadripper default)"
    BASE_FREQ=3700000
fi

echo "Detected base frequency: $(($BASE_FREQ/1000)) MHz"

# Verfügbare Governors prüfen
AVAILABLE_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors 2>/dev/null)
if [[ "$AVAILABLE_GOV" == *"performance"* ]]; then
    echo "Setting CPU governor to performance mode..."
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        echo performance | sudo tee $cpu/cpufreq/scaling_governor >/dev/null 2>&1
    done
else
    echo "Performance governor not available, using current: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
fi

# Fixiere CPU-Frequenz auf erkannte Basis-Frequenz
echo "Locking CPU frequency to $(($BASE_FREQ/1000)) MHz..."
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    if [ -d "$cpu/cpufreq" ]; then
        echo $BASE_FREQ | sudo tee $cpu/cpufreq/scaling_min_freq >/dev/null 2>&1
        echo $BASE_FREQ | sudo tee $cpu/cpufreq/scaling_max_freq >/dev/null 2>&1
    fi
done

# AMD Precision Boost / CPB deaktivieren für konsistente Frequenz
if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null
    echo "Precision Boost / CPB disabled"
elif [ -f /sys/devices/system/cpu/cpu0/cpufreq/boost ]; then
    # Alternative path on some AMD systems
    for cpu in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/boost; do
        [ -f "$cpu" ] && echo 0 | sudo tee "$cpu" >/dev/null 2>&1
    done
    echo "Precision Boost disabled (per-core)"
else
    echo "Warning: Could not disable Precision Boost (not found)"
fi

# Status zeigen
echo ""
echo "Current CPU settings:"
echo -n "Governor: "
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo -n "Current freq: "
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
echo -n "Min freq: "
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
echo -n "Max freq: "
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq

# Check boost status
if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    echo -n "Boost: "
    BOOST_STATUS=$(cat /sys/devices/system/cpu/cpufreq/boost)
    if [ "$BOOST_STATUS" -eq 0 ]; then
        echo "disabled"
    else
        echo "enabled"
    fi
fi

echo ""

# CPU Warmup - erkenne Anzahl der CPUs
NUM_CPUS=$(nproc)
echo "Warming up CPU ($NUM_CPUS threads) for 60 seconds..."

stress-ng --cpu $NUM_CPUS --timeout 60s --metrics-brief 2>/dev/null || {
    # Fallback: Inline C-Programm für CPU-Last (wenn stress-ng fehlt)
    echo "stress-ng not found, using inline C warmup..."

    WARMUP_C=$(mktemp --suffix=.c)
    WARMUP_BIN=$(mktemp)

    cat > "$WARMUP_C" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define MATRIX_SIZE 512
#define DURATION_SEC 60

void matrix_multiply(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

void* worker(void* arg) {
    int tid = *(int*)arg;
    size_t n = MATRIX_SIZE;
    size_t size = n * n * sizeof(float);

    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C = (float*)malloc(size);

    // Initialize with simple values
    for (size_t i = 0; i < n*n; i++) {
        A[i] = (float)(i % 100) / 100.0f;
        B[i] = (float)((i + tid) % 100) / 100.0f;
    }

    time_t start = time(NULL);
    time_t end = start + DURATION_SEC;

    while (time(NULL) < end) {
        matrix_multiply(A, B, C, n);
    }

    free(A);
    free(B);
    free(C);

    return NULL;
}

int main(int argc, char** argv) {
    int num_threads = (argc > 1) ? atoi(argv[1]) : 1;

    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    int* tids = (int*)malloc(num_threads * sizeof(int));

    printf("Starting %d warmup threads...\n", num_threads);

    for (int i = 0; i < num_threads; i++) {
        tids[i] = i;
        pthread_create(&threads[i], NULL, worker, &tids[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(tids);

    printf("CPU warmup complete!\n");
    return 0;
}
EOF

    # Compile and run
    if gcc -O3 -pthread -o "$WARMUP_BIN" "$WARMUP_C" 2>/dev/null; then
        "$WARMUP_BIN" "$NUM_CPUS"
        rm -f "$WARMUP_BIN" "$WARMUP_C"
    else
        # Ultimate fallback: pure bash CPU burn
        echo "gcc not available, using bash fallback..."

        for ((i=0; i<NUM_CPUS; i++)); do
            (
                end=$((SECONDS + 60))
                while [ $SECONDS -lt $end ]; do
                    # CPU-intensive operations
                    echo "scale=1000; 4*a(1)" | bc -l > /dev/null 2>&1
                done
            ) &
        done
        wait

        echo "CPU warmup complete!"
        rm -f "$WARMUP_C"
    fi
}

echo ""
echo "CPU stabilized and ready for measurements!"
echo ""
echo "Current CPU frequency:"
grep MHz /proc/cpuinfo | head -n 5
