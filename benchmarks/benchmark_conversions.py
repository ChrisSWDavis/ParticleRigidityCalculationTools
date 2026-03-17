#!/usr/bin/env python3
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the benchmark imports the local working tree module.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import ParticleRigidityCalculationTools as prct


def run_one_pass(sample_size: int) -> dict:
    energies = np.linspace(1.0, 10000.0, sample_size)
    flux = np.linspace(0.1, 10.0, sample_size)

    start = time.perf_counter()
    rigidities = prct.convertParticleEnergyToRigidity(
        energies,
        particleMassAU=1.0,
        particleChargeAU=1.0,
    )
    energy_to_rigidity = time.perf_counter() - start

    start = time.perf_counter()
    prct.convertParticleRigidityToEnergy(
        rigidities,
        particleMassAU=1.0,
        particleChargeAU=1.0,
    )
    rigidity_to_energy = time.perf_counter() - start

    start = time.perf_counter()
    rspec = prct.convertParticleEnergySpecToRigiditySpec(
        energies,
        flux,
        particleMassAU=1.0,
        particleChargeAU=1.0,
    )
    energy_spec_to_rigidity_spec = time.perf_counter() - start

    start = time.perf_counter()
    prct.convertParticleRigiditySpecToEnergySpec(
        rspec["Rigidity"],
        rspec["Rigidity distribution values"],
        particleMassAU=1.0,
        particleChargeAU=1.0,
    )
    rigidity_spec_to_energy_spec = time.perf_counter() - start

    return {
        "convertParticleEnergyToRigidity": energy_to_rigidity,
        "convertParticleRigidityToEnergy": rigidity_to_energy,
        "convertParticleEnergySpecToRigiditySpec": energy_spec_to_rigidity_spec,
        "convertParticleRigiditySpecToEnergySpec": rigidity_spec_to_energy_spec,
    }


def run_benchmark(sample_size: int, repeats: int, warmup: int) -> dict:
    for _ in range(warmup):
        run_one_pass(sample_size)

    measurements = [run_one_pass(sample_size) for _ in range(repeats)]
    metric_names = measurements[0].keys()

    summary = {}
    for metric in metric_names:
        values = [row[metric] for row in measurements]
        summary[metric] = {
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
        }

    return {
        "sample_size": sample_size,
        "repeats": repeats,
        "warmup": warmup,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark core conversion routines.")
    parser.add_argument("--sample-size", type=int, default=200000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    results = run_benchmark(args.sample_size, args.repeats, args.warmup)
    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(rendered + "\n")


if __name__ == "__main__":
    main()
