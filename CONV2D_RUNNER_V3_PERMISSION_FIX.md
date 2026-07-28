# Permission fix in v3

A quickcheck started with `sudo` can create `runs/CONV2D` or the local build directory as root. A later normal-user `02` runner then cannot create its manifest.

v3 prevents recurrence:

- Quickcheck and `02` abort immediately when launched as root.
- `01_enable` is still run with sudo.
- The normal-user `02` obtains a sudo timestamp internally.
- Before compilation and manifest creation it creates/repairs only:
  - `scripts/CONV2D/.build`
  - `runs/CONV2D`
- Both directories are recursively returned to the invoking user's UID/GID.
- Restore and poweroff behavior is unchanged.
