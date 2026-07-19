AMD CPU measurement tree for the Ryzen Threadripper 3970X

Directory layout
----------------
  scripts/common/benchmark_common.hpp
  scripts/<WORKLOAD>/CPU/AMD/main_*.cpp
  scripts/00_default_text_optional.sh
  scripts/01_enable_CPU_AMD.sh
  scripts/02_run_CPU_AMD_5min_autoshutdown.sh
  scripts/03_disable_CPU_AMD.sh
  scripts/04_default_graphic.sh
  scripts/check_CPU_AMD.sh
  scripts/check_energy_backend.cpp
  runs2/<WORKLOAD>/CPU/AMD/

No script writes to runs/ or CPU/INTEL.

Measurement grid
----------------
Threads: 1, 2, 4, 8, 10, 16, 20, 32, 64
The direct Intel/AMD comparison uses the shared points through 20 threads.
32 threads measures all physical Threadripper cores; 64 threads is the SMT extension.

Default campaign:
  10 repetitions per configuration
  5 complete passes
  5-minute pause after compilation before the first workload
  5-minute pause between complete workloads
  5-minute pause between complete passes
  22,950 total data rows

There are no pauses between the ten repetitions of one configuration. Before a
new configuration is calibrated, the program prints a progress line so long
GEMM calibration is no longer silent.

Energy backend
--------------
device_energy_j: AMD package energy via perf power/energy-pkg/
total_energy_j: same as device_energy_j
dram_energy_j: -1.000000 because no separate AMD DRAM event is available

Dependencies
------------
  sudo apt update
  sudo apt install build-essential libopenblas-dev libdnnl-dev linux-tools-common python3

Mandatory audit before the long run
-----------------------------------
From the graphical session:
  cd /home/rock/projects/energy/new2
  scripts/check_CPU_AMD.sh

The check compiles and links all six workloads, verifies the 64-thread grid,
temporarily enables/restores AMD measurement permissions when necessary, and
tests the exact C++ perf_event_open energy backend used by the benchmarks.
Do not start the long run unless this check ends with:
  Environment and package check passed.

Official run
------------
1. Optional text mode (closes the GUI):
     scripts/00_default_text_optional.sh
   Then log in on a text console.

2. Prepare CPU and energy permissions:
     cd /home/rock/projects/energy/new2
     sudo scripts/01_enable_CPU_AMD.sh

3. Start the complete campaign without sudo:
     scripts/02_run_CPU_AMD_5min_autoshutdown.sh

The runner compiles all workloads again, tests the exact energy backend, waits
five minutes after compilation, writes only below runs2, audits every finished
CSV automatically, saves a campaign log and metadata, restores CPU/perf settings
on exit, and powers off only after every workload and every CSV audit succeeds.

Quick runner test without shutdown or pauses
--------------------------------------------
This invokes the normal randomized workload code and may enter a large first
configuration; use Ctrl+C after confirming startup and energy output:

  sudo scripts/01_enable_CPU_AMD.sh
  env REPS=1 SESSIONS=1 INITIAL_PAUSE_SECONDS=0 WORKLOAD_PAUSE_SECONDS=0 \
      SESSION_PAUSE_SECONDS=0 POWER_OFF_AT_END=0 \
      scripts/02_run_CPU_AMD_5min_autoshutdown.sh

Ctrl+C restores settings and does not shut the machine down.
