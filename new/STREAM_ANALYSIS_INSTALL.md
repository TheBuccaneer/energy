# Installation and execution

Unpack from the project root:

```bash
cd ~/projects/energy/new
unzip -o ~/Downloads/STREAM_analysis_all_platforms_v1.zip
chmod +x run_stream_analysis_all.sh \
  AMD/analyse/STREAM/run_all.sh \
  INTEL/analyse/STREAM/run_all.sh \
  3090/analyse/STREAM/run_all.sh \
  5060ti/analyse/STREAM/run_all.sh \
  "ALL AUDIT/STREAM/analyse/run_all.sh"
```

Run after all four complete five-session campaigns exist:

```bash
source ~/projects/energy/.venv/bin/activate
cd ~/projects/energy/new
./run_stream_analysis_all.sh
```

The pipeline refuses partial campaigns and does not mix campaign timestamps within a platform.
