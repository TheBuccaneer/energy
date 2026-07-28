# Validation Report

## Statische Prüfungen

- `bash -n` für alle vier Shellskripte: PASS
- keine Template-Platzhalter: PASS
- Quickcheck-Wrapper referenzieren die korrekten Runner: PASS
- Cleanup-Funktionen sind vor der EXIT-Trap-Registrierung definiert: PASS
- GNU-`env`-Optionen stehen vor allen `NAME=VALUE`-Zuweisungen: PASS
- Quickcheck setzt `POWER_OFF_AT_END=0`: PASS
- offizieller Runner verwendet standardmäßig `POWER_OFF_AT_END=1`: PASS

## Simulierter Ablauf

Mit einem Fake-CONV2D-Binary wurden echte Runner- und Validatorpfade ausgeführt.

- AMD Quickcheck-only: PASS
- Intel Quickcheck-only: PASS
- AMD vollständige Kampagne mit fünf Sessions: PASS
- AMD Restore vor Poweroff: PASS
- AMD `systemctl poweroff` nur nach Erfolg: PASS
- absichtlich fehlgeschlagener AMD-Build: Restore PASS, kein Poweroff PASS
- 24-Zeilen-Quickcheckvalidierung: PASS
- 540-Zeilen-AMD-Sessionvalidierung: PASS
- exklusive Anti-Collapse-Ausführung ohne CSV: PASS

## Nicht hier verifiziert

- realer Build gegen die auf den beiden Messrechnern installierte oneDNN-Version
- echte RAPL-/AMD-perf-Energie
- tatsächliche oneDNN-Kernel und Scratchpadgrößen
- reale Checksumfehlerverteilungen
- reale Laufzeitkalibrierung

Diese Punkte sind Aufgabe des Hardware-Quickchecks.
