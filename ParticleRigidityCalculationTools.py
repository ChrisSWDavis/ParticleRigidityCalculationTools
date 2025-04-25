import numpy as np
from numba import njit, jit, vectorize, prange

onekftinkm = 0.3048

protonRestMass = 1.67262192e-27  # kg
chargeOfElectron = 1.60217663e-19  # C
c = 299792458.0  # m/s

def getAtomicMass(atomicNumber):
    A = [1.0, 4.0, 6.9, 9.0, 10.8, 12.0, 14.0, 16.0, 19.0, 20.2,
       23.0, 24.3, 27.0, 28.1, 31.0, 32.1, 35.4, 39.9, 39.1, 40.1,
       44.9, 47.9, 50.9, 52.0, 54.9, 55.8, 58.9, 58.7, 63.5, 65.4,
       69.7, 72.6, 74.9, 79.0, 79.9, 83.8, 85.5, 87.6, 88.9, 91.2,
       92.9, 95.9, 97.0, 101.0, 102.9, 106.4, 107.9, 112.4, 114.8, 118.7,
       121.8, 127.6, 126.9, 131.3, 132.9, 137.3, 138.9, 140.1, 140.9, 144.2,
       145.0, 150.4, 152.0, 157.3, 158.3, 162.5, 164.9, 167.3, 168.9, 173.0,
       175.0, 178.5, 180.9, 183.9, 186.2, 190.2, 192.2, 195.1, 197.0, 200.6,
       204.4, 207.2, 209.0, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.0,
       231.0, 238.0]

    if atomicNumber < 0:
        atomicMass = 1.0  # handle case for electron
    elif atomicNumber <= 92:
        atomicMass = A[atomicNumber - 1]  # look up table for other elements.
    else:
        atomicMass = 0.0  # just in case

    return atomicMass

@njit
def determineParticleAttributes_numba(particleMassAU, particleChargeAU, protonRestMass, chargeOfElectron, c):
    m0 = particleMassAU * protonRestMass  # kg
    particleCharge = particleChargeAU * chargeOfElectron  # C
    particleRestEnergy = m0 * (c**2)
    return particleCharge, particleRestEnergy

def determineParticleAttributes(particleMassAU, particleChargeAU):
    return determineParticleAttributes_numba(particleMassAU, particleChargeAU, protonRestMass, chargeOfElectron, c)

@njit
def convertParticleEnergyToRigidity_numba(particleKineticEnergyInMeV, particleCharge, particleRestEnergy, chargeOfElectron, c):
    # Convert kinetic energy from MeV to Joules
    particleKineticEnergyInJoules = particleKineticEnergyInMeV * chargeOfElectron * 1e6
    
    # Calculate total relativistic energy
    totalParticleEnergy = particleKineticEnergyInJoules + particleRestEnergy
    
    # Calculate momentum using relativistic relation: (pc)² = E² - (mc²)²
    pc = np.sqrt((totalParticleEnergy**2) - (particleRestEnergy**2))
    
    # Calculate rigidity R = pc/q (in GV)
    rigidityInGV = (pc / particleCharge) * 1e-9
    
    return rigidityInGV

def convertParticleEnergyToRigidity(particleKineticEnergyInMeV, particleMassAU=1, particleChargeAU=1):
    """
    Convert particle kinetic energy to magnetic rigidity.
    
    Parameters
    ----------
    particleKineticEnergyInMeV : array_like or scalar
        Particle kinetic energy in MeV.
        
    particleMassAU : float, optional
        Particle mass in atomic mass units (proton mass = 1).
        Default is 1 (proton).
        
    particleChargeAU : float, optional
        Particle charge in elementary charge units.
        Default is 1 (singly charged).
        
    Returns
    -------
    numpy.ndarray
        Magnetic rigidity in GV (gigavolts).
    """
    # Convert input to numpy array if it's a scalar
    if isinstance(particleKineticEnergyInMeV, (int, float)):
        particleKineticEnergyInMeV = np.array([particleKineticEnergyInMeV])
    else:
        particleKineticEnergyInMeV = np.array(particleKineticEnergyInMeV)
    
    particleCharge, particleRestEnergy = determineParticleAttributes(particleMassAU, particleChargeAU)
    return convertParticleEnergyToRigidity_numba(particleKineticEnergyInMeV, particleCharge, particleRestEnergy, chargeOfElectron, c)

@njit
def convertParticleRigidityToEnergy_numba(particleRigidityInGV, particleCharge, particleRestEnergy, chargeOfElectron, c):
    # Calculate momentum from rigidity: pc = q·R (in Joules)
    pc = particleRigidityInGV * particleCharge * 1e9
    
    # Calculate total energy using relativistic relation: E² = (pc)² + (mc²)²
    totalParticleEnergy = np.sqrt((pc**2) + (particleRestEnergy**2))
    
    # Calculate kinetic energy: KE = E - mc²
    particleKEinJoules = totalParticleEnergy - particleRestEnergy
    
    # Convert from Joules to MeV
    KEinMeV = particleKEinJoules / (chargeOfElectron * 1e6)
    
    return KEinMeV

def convertParticleRigidityToEnergy(particleRigidityInGV, particleMassAU=1, particleChargeAU=1):
    """
    Convert magnetic rigidity to particle kinetic energy.
    
    Parameters
    ----------
    particleRigidityInGV : array_like or scalar
        Magnetic rigidity in GV (gigavolts).
        
    particleMassAU : float, optional
        Particle mass in atomic mass units (proton mass = 1).
        Default is 1 (proton).
        
    particleChargeAU : float, optional
        Particle charge in elementary charge units.
        Default is 1 (singly charged).
        
    Returns
    -------
    numpy.ndarray
        Particle kinetic energy in MeV.
    """
    # Convert input to numpy array if it's a scalar
    if isinstance(particleRigidityInGV, (int, float)):
        particleRigidityInGV = np.array([particleRigidityInGV])
    else:
        particleRigidityInGV = np.array(particleRigidityInGV)
    
    particleCharge, particleRestEnergy = determineParticleAttributes(particleMassAU, particleChargeAU)
    return convertParticleRigidityToEnergy_numba(particleRigidityInGV, particleCharge, particleRestEnergy, chargeOfElectron, c)

@njit
def calculate_dKEoverdR_numba(particleKineticEnergyInMeV, particleChargeInCoulombs, particleRestEnergy, chargeOfElectron):
    particleKineticEnergyInJoules = particleKineticEnergyInMeV * chargeOfElectron * 1e6
    
    totalParticleEnergy = particleKineticEnergyInJoules + particleRestEnergy
    pc = np.sqrt((totalParticleEnergy**2) - (particleRestEnergy**2))  # units of pc are in joules
    
    fullFactor = pc / totalParticleEnergy
    
    dKEInMeV_drigidityInGV = fullFactor * particleChargeInCoulombs * 1e9 / (chargeOfElectron * 1e6)
    return dKEInMeV_drigidityInGV  # output units are in milli electron charges

def calculate_dKEoverdR(particleKineticEnergyInMeV, particleChargeInCoulombs, particleRestEnergy):
    return calculate_dKEoverdR_numba(particleKineticEnergyInMeV, particleChargeInCoulombs, particleRestEnergy, chargeOfElectron)

@njit
def convertParticleEnergySpecToRigiditySpec_numba(
    particleKineticEnergyInMeV, 
    fluxInEnergyMeVform, 
    particleCharge, 
    particleRestEnergy,
    chargeOfElectron, 
    c
):
    # Calculate the derivative dE/dR
    dKEInMeV_drigidityInGV = calculate_dKEoverdR_numba(
        particleKineticEnergyInMeV, 
        particleCharge, 
        particleRestEnergy,
        chargeOfElectron
    )
    
    # Convert energy to rigidity
    outputRigidities = convertParticleEnergyToRigidity_numba(
        particleKineticEnergyInMeV, 
        particleCharge, 
        particleRestEnergy,
        chargeOfElectron,
        c
    )
    
    # Transform the flux: J(R) = J(E) * (dE/dR)
    outputRigiditySpectrum = dKEInMeV_drigidityInGV * fluxInEnergyMeVform
    
    return outputRigidities, outputRigiditySpectrum

def convertParticleEnergySpecToRigiditySpec(particleKineticEnergyInMeV, fluxInEnergyMeVform, particleMassAU=1, particleChargeAU=1):
    """
    Convert a differential energy spectrum to a differential rigidity spectrum.
    
    Parameters
    ----------
    particleKineticEnergyInMeV : array_like or scalar
        Energy points of the spectrum in MeV.
        
    fluxInEnergyMeVform : array_like or scalar
        Differential flux values at each energy point (particles/MeV).
        
    particleMassAU : float, optional
        Particle mass in atomic mass units (proton mass = 1).
        Default is 1 (proton).
        
    particleChargeAU : float, optional
        Particle charge in elementary charge units.
        Default is 1 (singly charged).
        
    Returns
    -------
    dict
        Dictionary with keys:
        - "Rigidity": Rigidity points in GV
        - "Rigidity distribution values": Differential flux values (particles/GV)
    """
    # Convert inputs to numpy arrays if they're scalars
    if isinstance(particleKineticEnergyInMeV, (int, float)):
        particleKineticEnergyInMeV = np.array([particleKineticEnergyInMeV])
    else:
        particleKineticEnergyInMeV = np.array(particleKineticEnergyInMeV)
        
    if isinstance(fluxInEnergyMeVform, (int, float)):
        fluxInEnergyMeVform = np.array([fluxInEnergyMeVform])
    else:
        fluxInEnergyMeVform = np.array(fluxInEnergyMeVform)
    
    particleCharge, particleRestEnergy = determineParticleAttributes(particleMassAU, particleChargeAU)
    
    outputRigidities, outputRigiditySpectrum = convertParticleEnergySpecToRigiditySpec_numba(
        particleKineticEnergyInMeV, 
        fluxInEnergyMeVform, 
        particleCharge, 
        particleRestEnergy,
        chargeOfElectron,
        c
    )
    
    return {
        "Rigidity": outputRigidities, 
        "Rigidity distribution values": outputRigiditySpectrum
    }

@njit
def convertParticleRigiditySpecToEnergySpec_numba(
    particleRigidityInGV, 
    fluxInRigidityGVform, 
    particleCharge, 
    particleRestEnergy,
    chargeOfElectron,
    c
):
    # Convert rigidity to energy
    particleKineticEnergyInMeV = convertParticleRigidityToEnergy_numba(
        particleRigidityInGV, 
        particleCharge, 
        particleRestEnergy,
        chargeOfElectron,
        c
    )
    
    # Calculate the derivative dE/dR
    dKEInMeV_drigidityInGV = calculate_dKEoverdR_numba(
        particleKineticEnergyInMeV, 
        particleCharge, 
        particleRestEnergy,
        chargeOfElectron
    )
    
    # Set output energies
    outputEnergies = particleKineticEnergyInMeV
    
    # Transform the flux: J(E) = J(R) / (dE/dR)
    # Handle division by zero by creating a mask
    outputEnergySpectrum = np.zeros_like(fluxInRigidityGVform)
    for i in range(len(dKEInMeV_drigidityInGV)):
        if dKEInMeV_drigidityInGV[i] != 0:
            outputEnergySpectrum[i] = fluxInRigidityGVform[i] / dKEInMeV_drigidityInGV[i]
    
    return outputEnergies, outputEnergySpectrum

def convertParticleRigiditySpecToEnergySpec(particleRigidityInGV, fluxInRigidityGVform, particleMassAU=1, particleChargeAU=1):
    """
    Convert a differential rigidity spectrum to a differential energy spectrum.
    
    Parameters
    ----------
    particleRigidityInGV : array_like or scalar
        Rigidity points of the spectrum in GV.
        
    fluxInRigidityGVform : array_like or scalar
        Differential flux values at each rigidity point (particles/GV).
        
    particleMassAU : float, optional
        Particle mass in atomic mass units (proton mass = 1).
        Default is 1 (proton).
        
    particleChargeAU : float, optional
        Particle charge in elementary charge units.
        Default is 1 (singly charged).
        
    Returns
    -------
    dict
        Dictionary with keys:
        - "Energy": Energy points in MeV
        - "Energy distribution values": Differential flux values (particles/MeV)
    """
    # Convert inputs to numpy arrays if they're scalars
    if isinstance(particleRigidityInGV, (int, float)):
        particleRigidityInGV = np.array([particleRigidityInGV])
    else:
        particleRigidityInGV = np.array(particleRigidityInGV)
        
    if isinstance(fluxInRigidityGVform, (int, float)):
        fluxInRigidityGVform = np.array([fluxInRigidityGVform])
    else:
        fluxInRigidityGVform = np.array(fluxInRigidityGVform)
    
    particleCharge, particleRestEnergy = determineParticleAttributes(particleMassAU, particleChargeAU)
    
    outputEnergies, outputEnergySpectrum = convertParticleRigiditySpecToEnergySpec_numba(
        particleRigidityInGV, 
        fluxInRigidityGVform, 
        particleCharge, 
        particleRestEnergy,
        chargeOfElectron,
        c
    )
    
    return {
        "Energy": outputEnergies, 
        "Energy distribution values": outputEnergySpectrum
    }
