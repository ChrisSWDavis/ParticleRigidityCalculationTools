
import numpy as np
import pandas as pd
import ParticleRigidityCalculationTools as PRCT

def test_getAtomicMass():

    assert PRCT.getAtomicMass(12) == 24.3

def test_rigidityConversion():

    particleKineticEnergyInMeV = [250.0, 578.5, 1056.8, 5123.9]

    outputValues = PRCT.convertParticleEnergyToRigidity(particleKineticEnergyInMeV, particleMassAU = 1.0, particleChargeAU = 1.0)

    roundedOutputValues = outputValues.apply(lambda x:round(x,2))

    assert list(roundedOutputValues) == [0.73, 1.19,1.76, 5.99]

def oneDPround(inputVal):

    return round(inputVal,1)

def test_rigidityAndEnergyConversion():

    particleKineticEnergyInMeV = [250.0, 578.5, 1056.8, 5123.9]

    outputRigidityValues = PRCT.convertParticleEnergyToRigidity(particleKineticEnergyInMeV, particleMassAU = 1.0, particleChargeAU = 1.0)

    outputEnergyValues = PRCT.convertParticleRigidityToEnergy(outputRigidityValues, particleMassAU = 1.0, particleChargeAU = 1.0)

    assert list(outputEnergyValues.apply(oneDPround)) == particleKineticEnergyInMeV


def test_smallKineticEnergy_roundTrip_rigidityEnergy_stable():
    """
    E_kin << m_p c^2 (eV–keV scale in MeV): stable formulas should round-trip
    energy → rigidity → energy without catastrophic loss (float64).
    """
    # 1e-6 MeV = 1 eV; 1e-3 MeV = 1 keV; up to 100 keV
    particleKineticEnergyInMeV = np.array(
        [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1],
        dtype=np.float64,
    )
    rigidity_gv = PRCT.convertParticleEnergyToRigidity(
        particleKineticEnergyInMeV,
        particleMassAU=1.0,
        particleChargeAU=1.0,
    )
    energy_back_mev = PRCT.convertParticleRigidityToEnergy(
        rigidity_gv,
        particleMassAU=1.0,
        particleChargeAU=1.0,
    )
    np.testing.assert_allclose(
        energy_back_mev.to_numpy(),
        particleKineticEnergyInMeV,
        rtol=1e-12,
        atol=1e-18,
    )
    assert np.all(rigidity_gv.to_numpy() >= 0.0)
    assert np.all(np.diff(rigidity_gv.to_numpy()) > 0.0)


#######################################################

def test_EnergySpecToRigiditySpec():

    energyValuesInMeV = [1000, 2000, 3000, 4000, 5000]
    energyDistributionValues = [1, 0.5, 0.2, 0.1, 0.01]

    rigiditySpec = PRCT.convertParticleEnergySpecToRigiditySpec(energyValuesInMeV,energyDistributionValues,particleMassAU = 1,particleChargeAU = 1)

    assert rigiditySpec["Rigidity"].round(2).iloc[2] == 3.82
    assert rigiditySpec["Rigidity distribution values"].round(2).iloc[2] == 194.24

def test_RigiditySpecToEnergySpec():

    rigidityValuesInGV = np.linspace(0.0,10.0,19)
    rigidityDistributionValues = np.linspace(0.0,10.0,19)

    energySpec = PRCT.convertParticleRigiditySpecToEnergySpec(rigidityValuesInGV,rigidityDistributionValues,particleMassAU = 1,particleChargeAU = 1)

def test_BothSpectralConversions():

    energyValuesInMeV = [1000, 2000, 3000, 4000, 5000]
    energyDistributionValues = [1, 0.5, 0.2, 0.1, 0.01]

    rigiditySpec = PRCT.convertParticleEnergySpecToRigiditySpec(energyValuesInMeV,energyDistributionValues,particleMassAU = 1,particleChargeAU = 1)

    energySpec = PRCT.convertParticleRigiditySpecToEnergySpec(rigiditySpec["Rigidity"],rigiditySpec["Rigidity distribution values"],particleMassAU = 1,particleChargeAU = 1)

    assert energySpec["Energy"].round(2).array == pd.Series(energyValuesInMeV).apply(float).round(2).array
    assert energySpec["Energy distribution values"].round(2).array == pd.Series(energyDistributionValues).apply(float).round(2).array


def test_BothSpectralConversionsAlpha():

    energyValuesInMeV = [1000, 2000, 3000, 4000, 5000]
    energyDistributionValues = [1, 0.5, 0.2, 0.1, 0.01]

    rigiditySpec = PRCT.convertParticleEnergySpecToRigiditySpec(energyValuesInMeV,energyDistributionValues,particleMassAU = 4,particleChargeAU = 2)

    energySpec = PRCT.convertParticleRigiditySpecToEnergySpec(rigiditySpec["Rigidity"],rigiditySpec["Rigidity distribution values"],particleMassAU = 4,particleChargeAU = 2)

    assert energySpec["Energy"].round(2).array == pd.Series(energyValuesInMeV).apply(float).round(2).array
    assert energySpec["Energy distribution values"].round(2).array == pd.Series(energyDistributionValues).apply(float).round(2).array