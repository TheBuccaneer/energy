# Validation

- AMD runner: `bash -n` PASS
- Intel runner: `bash -n` PASS
- AMD quickcheck wrapper: `bash -n` PASS
- Intel quickcheck wrapper: `bash -n` PASS
- Compact normal-log validator: synthetic PASS
- Quiet anti-collapse validator: synthetic PASS
- Intel/AMD source bracket balance: PASS
- Normal successful checksum diagnostics: suppressed
- Failed checksum diagnostics: retained
- oneDNN detailed diagnostics: retained only under `CONV2D_DIAGNOSTICS=1`
- Official and normal quickcheck measurements explicitly unset:
  - `ONEDNN_VERBOSE`
  - `DNNL_VERBOSE`
  - `CONV2D_DIAGNOSTICS`
- Separate verbose preflight remains redirected to its own log file.

A real compilation was not repeated in this container because the oneDNN
development headers are not installed here. The supplied pre-patch sources were
already compiled successfully on the user's AMD system; source changes in v4 are
limited to output gating and formatting.
