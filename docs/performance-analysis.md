# Performance analysis and optimization notes

## Scope

This repository is small and dominated by numeric conversion routines in `ParticleRigidityCalculationTools.py`.
The optimization work focused on preserving public API and default behavior while improving throughput for large vector inputs.

## Main bottlenecks identified

1. **Per-element `decimal.Decimal` conversion** in hot paths (`Series.apply(dec.Decimal)`).
2. **Object-dtype math** (`np.sqrt` over `Decimal` objects), which disables fast vectorized NumPy code paths.
3. **Repeated construction of immutable lookup data** (atomic mass table created inside `getAtomicMass` on each call).
4. **Extra Python-layer overhead** from repeated `.apply(float)` patterns.

## Changes implemented

1. Replaced core numeric paths with **NumPy/Pandas float64 vectorized operations**.
2. Kept output shapes and types compatible:
   - scalar/list/array/Series input support still works through `allowForNonSeriesInputArgs`.
   - conversion functions still return `pd.Series`.
   - spectrum conversion functions still return `pd.DataFrame`.
3. Added stable constants (`MeVtoJoules`, `JoulesToMeV`, `cSquared`) to avoid recomputation.
4. Moved atomic mass lookup values to module-level `AtomicMassTable`.
5. Added `benchmarks/benchmark_conversions.py` for reproducible timing runs.

## Timing record

Measured on this cloud environment using sample size `N = 200000`.

### Baseline (before optimization)

- `convertParticleEnergyToRigidity`: **0.711054 s**
- `convertParticleRigidityToEnergy`: **0.646278 s**
- `convertParticleEnergySpecToRigiditySpec`: **1.634429 s**
- `convertParticleRigiditySpecToEnergySpec`: **1.644606 s**

### After optimization

- `convertParticleEnergyToRigidity`: **0.003961 s**
- `convertParticleRigidityToEnergy`: **0.000900 s**
- `convertParticleEnergySpecToRigiditySpec`: **0.005586 s**
- `convertParticleRigiditySpecToEnergySpec`: **0.004511 s**

### Speedup factors

- `convertParticleEnergyToRigidity`: **~179.5x faster**
- `convertParticleRigidityToEnergy`: **~718.1x faster**
- `convertParticleEnergySpecToRigiditySpec`: **~292.6x faster**
- `convertParticleRigiditySpecToEnergySpec`: **~364.6x faster**

## Correctness validation

- Existing tests pass after optimization:
  - `7 passed`

## Additional speed-up ideas (without changing default behavior)

1. Introduce optional `out` parameters for power users to reuse arrays and reduce allocations.
2. Add optional Numba acceleration for very large repeated workloads.
3. Add optional strict physical-range validation mode (off by default) for earlier failure on invalid data.
4. Add dedicated benchmark CI job to prevent performance regressions.

## Artifacts and recording note

This repository is library-only and was optimized/validated in a headless CLI environment.
Timing records are included above and reproducible with `benchmarks/benchmark_conversions.py`.
Machine-readable benchmark output is saved at `docs/benchmark-results.json`.
No GUI/video target is part of this project, so video/screenshot capture was not applicable for this optimization pass.
