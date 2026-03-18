# AGENTS.md

## Cursor Cloud specific instructions

This is a single-file Python library (`ParticleRigidityCalculationTools.py`) for converting between particle kinetic energy (MeV) and rigidity (GV). No external services, databases, or Docker containers are needed.

### Lint, Test, Build

- **Lint:** `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics` (critical errors only; CI also runs a relaxed pass with `--exit-zero --max-complexity=10 --max-line-length=127`)
- **Test:** `pytest` (7 tests in `test_ParticleRigidityCalculationTools.py`)
- **Install (editable):** `pip install .` (uses hatchling build backend; installs numpy and pandas)

### Notes

- The library is installed as a user package (`~/.local/`). Ensure `~/.local/bin` is on `PATH` for `pytest` and `flake8` CLI access.
- Python 3.8–3.12 are supported per CI matrix (`.github/workflows/python-package.yml`).
