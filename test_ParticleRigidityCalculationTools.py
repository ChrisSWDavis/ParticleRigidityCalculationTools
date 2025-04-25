"""
Test suite for Particle Rigidity Calculation Tools.

This module contains tests for the functionality provided by the 
ParticleRigidityCalculationTools package. It verifies that various
conversions between particle energy and rigidity are performed correctly
and that the spectral transformations preserve the underlying particle distributions.

The tests cover:
1. Basic atomic mass lookup functionality
2. Energy to rigidity conversions
3. Rigidity to energy conversions
4. Spectral conversions in both directions
5. Validation of roundtrip conversions (energy → rigidity → energy)
6. Testing with different particle types (protons, alpha particles)
"""

import numpy as np
import ParticleRigidityCalculationTools as PRCT

def test_getAtomicMass():
    """
    Test the atomic mass lookup functionality.

    Verifies that the getAtomicMass function returns the correct
    atomic mass for a given atomic number (magnesium in this case).
    """
    assert PRCT.getAtomicMass(12) == 24.3

def test_rigidityConversion():
    """
    Test conversion from kinetic energy to magnetic rigidity.
    
    Verifies that the convertParticleEnergyToRigidity function correctly
    calculates rigidity values for a set of proton energies. The expected
    values have been pre-calculated and validated against established
    particle physics references.
    """
    particleKineticEnergyInMeV = np.array([250.0, 578.5, 1056.8, 5123.9])

    outputValues = PRCT.convertParticleEnergyToRigidity(particleKineticEnergyInMeV, particleMassAU=1.0, particleChargeAU=1.0)

    roundedOutputValues = np.round(outputValues, 2)

    assert list(roundedOutputValues) == [0.73, 1.19, 1.76, 5.99]

def oneDPround(inputVal):
    """
    Helper function to round a value to one decimal place.
    
    Parameters
    ----------
    inputVal : float
        Value to be rounded.
        
    Returns
    -------
    float
        Rounded value to one decimal place.
    """
    return round(inputVal, 1)

def test_rigidityAndEnergyConversion():
    """
    Test roundtrip conversion: energy to rigidity and back to energy.
    
    This test verifies that converting from energy to rigidity and then
    back to energy returns the original values, confirming that the
    conversions are invertible and mathematically consistent.
    """
    particleKineticEnergyInMeV = np.array([250.0, 578.5, 1056.8, 5123.9])
    
    # Convert energy to rigidity
    outputRigidityValues = PRCT.convertParticleEnergyToRigidity(particleKineticEnergyInMeV, particleMassAU=1.0, particleChargeAU=1.0)

    # Convert rigidity back to energy
    outputEnergyValues = PRCT.convertParticleRigidityToEnergy(outputRigidityValues, particleMassAU=1.0, particleChargeAU=1.0)

    rounded_values = np.round(outputEnergyValues, 1)
    expected_values = np.round(particleKineticEnergyInMeV, 1)
    
    assert list(rounded_values) == list(expected_values)

#######################################################

def test_EnergySpecToRigiditySpec():
    """
    Test conversion of a differential energy spectrum to a rigidity spectrum.
    
    Verifies that the spectral conversion function correctly transforms
    an energy-based particle spectrum to a rigidity-based spectrum, with
    appropriately scaled flux values to preserve the particle distribution.
    """
    energyValuesInMeV = np.array([1000, 2000, 3000, 4000, 5000])
    energyDistributionValues = np.array([1, 0.5, 0.2, 0.1, 0.01])

    rigiditySpec = PRCT.convertParticleEnergySpecToRigiditySpec(energyValuesInMeV, energyDistributionValues, particleMassAU=1, particleChargeAU=1)

    # Verify specific expected values
    assert round(rigiditySpec["Rigidity"][2], 2) == 3.82
    assert round(rigiditySpec["Rigidity distribution values"][2], 2) == 194.24

def test_RigiditySpecToEnergySpec():
    """
    Test conversion of a differential rigidity spectrum to an energy spectrum.
    
    Verifies that the spectral conversion function correctly transforms
    a rigidity-based particle spectrum to an energy-based spectrum.
    
    Note: This test only checks that the function runs without errors;
    specific value testing is done in other test functions.
    """
    rigidityValuesInGV = np.linspace(0.1, 10.0, 19)  # Avoid starting with 0 to prevent division by zero
    rigidityDistributionValues = np.linspace(0.0, 10.0, 19)

    energySpec = PRCT.convertParticleRigiditySpecToEnergySpec(rigidityValuesInGV, rigidityDistributionValues, particleMassAU=1, particleChargeAU=1)
    
    # No specific assertions in the original test, just checking it runs

def test_BothSpectralConversions():
    """
    Test roundtrip spectral conversion for protons.
    
    This test verifies that converting a spectrum from energy to rigidity and
    then back to energy preserves both the energy values and the flux values,
    confirming that the spectral conversions are invertible and consistent.
    """
    energyValuesInMeV = np.array([1000, 2000, 3000, 4000, 5000])
    energyDistributionValues = np.array([1, 0.5, 0.2, 0.1, 0.01])
    
    # Convert from energy spectrum to rigidity spectrum
    rigiditySpec = PRCT.convertParticleEnergySpecToRigiditySpec(energyValuesInMeV, energyDistributionValues, particleMassAU=1, particleChargeAU=1)

    # Convert back from rigidity spectrum to energy spectrum
    energySpec = PRCT.convertParticleRigiditySpecToEnergySpec(
        rigiditySpec["Rigidity"], 
        rigiditySpec["Rigidity distribution values"],
        particleMassAU=1, 
        particleChargeAU=1
    )

    # Verify that both energy values and flux values are preserved
    assert np.allclose(np.round(energySpec["Energy"], 2), np.round(energyValuesInMeV, 2))
    assert np.allclose(np.round(energySpec["Energy distribution values"], 2), np.round(energyDistributionValues, 2))

def test_BothSpectralConversionsAlpha():
    """
    Test roundtrip spectral conversion for alpha particles.
    
    Similar to test_BothSpectralConversions, but tests the conversion for
    alpha particles (mass=4, charge=2) instead of protons.
    This ensures that the code works correctly for different particle types.
    """
    energyValuesInMeV = np.array([1000, 2000, 3000, 4000, 5000])
    energyDistributionValues = np.array([1, 0.5, 0.2, 0.1, 0.01])
    
    # Convert from energy spectrum to rigidity spectrum
    rigiditySpec = PRCT.convertParticleEnergySpecToRigiditySpec(energyValuesInMeV, energyDistributionValues, particleMassAU=4, particleChargeAU=2)

    # Convert back from rigidity spectrum to energy spectrum
    energySpec = PRCT.convertParticleRigiditySpecToEnergySpec(
        rigiditySpec["Rigidity"],
        rigiditySpec["Rigidity distribution values"],
        particleMassAU=4,
        particleChargeAU=2
    )

    # Verify that both energy values and flux values are preserved
    assert np.allclose(np.round(energySpec["Energy"], 2), np.round(energyValuesInMeV, 2))
    assert np.allclose(np.round(energySpec["Energy distribution values"], 2), np.round(energyDistributionValues, 2))