# Install and run the REDUCTION analysis pipeline

From the project root (`~/projects/energy/new`):

```bash
unzip -o ~/Downloads/REDUCTION_analysis_all_platforms_v1.3.zip
chmod +x run_reduction_analysis_all.sh \
  AMD/analyse/REDUCTION/run_all.sh \
  INTEL/analyse/REDUCTION/run_all.sh \
  3090/analyse/REDUCTION/run_all.sh \
  5060ti/analyse/REDUCTION/run_all.sh \
  "ALL AUDIT/REDUCTION/analyse/run_all.sh"
```

Install Python dependencies in the active environment:

```bash
python3 -m pip install -r "ALL AUDIT/REDUCTION/analyse/requirements.txt"
```

After all four complete five-session campaigns exist:

```bash
cd ~/projects/energy/new
./run_reduction_analysis_all.sh
```

To analyze a specific timestamp, pass it to each platform script individually with `--campaign` before running the combined audit.

Quickcheck CSVs are ignored by the campaign filename pattern.
