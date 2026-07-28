# Install v3 from repository root

```bash
cd ~/projects/energy
unzip -o /path/to/CONV2D_CPU_RUNNERS_v3_REPO_OVERLAY.zip
chmod +x \
  new/AMD/scripts/02_run_CPU_AMD_CONV2D_only.sh \
  new/AMD/scripts/quickcheck_CPU_AMD_CONV2D.sh \
  new/INTEL/scripts/02_run_CPU_Intel_CONV2D_only.sh \
  new/INTEL/scripts/quickcheck_CPU_Intel_CONV2D.sh
```

The runner must be started as the normal user. Only `01_enable` is started with sudo.
