# Installation and execution

Extract this bundle into the repository root (`~/projects/energy`), preserving paths. It adds only analysis files below `new/`.

```bash
cd ~/projects/energy
unzip -o CONV2D_ANALYSIS_BUNDLE.zip
python3 -m pip install pandas numpy matplotlib tabulate
chmod +x new/run_conv2d_analysis_all.sh
find new -path '*/analyse/CONV2D/run_all.sh' -exec chmod +x {} +
chmod +x 'new/ALL AUDIT/CONV2D/analyse/run_all.sh'
cd new
./run_conv2d_analysis_all.sh
```

By default, the newest complete five-session campaign is selected independently for each platform. Optional explicit selection:

```bash
AMD_CAMPAIGN=YYYYMMDD_HHMMSS \
INTEL_CAMPAIGN=YYYYMMDD_HHMMSS \
GPU3090_CAMPAIGN=YYYYMMDD_HHMMSS \
GPU5060TI_CAMPAIGN=YYYYMMDD_HHMMSS \
./run_conv2d_analysis_all.sh
```

Individual results: `<PLATFORM>/results/CONV2D/`.
Combined results: `ALL AUDIT/CONV2D/results/`.
