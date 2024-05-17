import numpy as np

# Code written by Samuel Aktar Laskar (200103097)
# Scroll to bottom



def printM(arr):
    with np.printoptions(precision=2, suppress=True, formatter={'float': '{:0.2e}'.format}):
        print(arr)


class Lamina:
    def __init__(self, E1: float, E2: float, G12: float, nu12: float, theta: float, thickness: float,
                 thermalParameters: np.array(3) = np.array([0, 0, 0]),
                 hygroParameters: np.array(3) = np.array([0, 0, 0]),
                 uSigmaT1 = 1000, uSigmaC1 = 1000, uSigmaT2=1000, uSigmaC2 = 1000,  uTou12 = 1000
                 ):
        self.uSigmaT1 = uSigmaT1
        self.uSigmaC1 = uSigmaC1
        self.uSigmaT2 = uSigmaT2
        self.uSigmaC2 = uSigmaC2
        self.uTou12 = uTou12
        self.matR = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 2]
        ])
        self.matQ = None
        self.matT: np.array([[]]) = None
        self.matQBar = None
        self.strainAlongAnalysisAxes: np.array(3) = np.zeros(3)
        # self.strainAlongMaterialAxes: np.array(3) = np.zeros(3)
        self.sigmaValues: np.array(3) = np.zeros(3)
        self.residualStressAlongMaterialAxis = np.zeros(3)
        self.residualStressAlongAnalysisAxis = np.zeros(3)
        self.theta = theta
        self.freeThermalStrains = np.zeros(3)
        self.residualThermalStrains = np.zeros(3)
        self.computeTransformationMatrix()

        thermalParameters[2] /= 2
        hygroParameters[2] /= 2
        self.thermalParameters = np.linalg.inv(self.matT) @ thermalParameters
        self.thermalParameters[2] *= 2
        self.hygroParameters = np.linalg.inv(self.matT) @ hygroParameters
        self.hygroParameters[2] *= 2


        self.E1 = E1 * 10 ** 9
        self.E2 = E2 * 10 ** 9
        self.G12 = G12 * 10 ** 9
        self.nu12 = nu12
        self.nu21 = nu12 * E2 / E1
        self.computeStiffnessMatrix()
        self.computeReducedStiffnessMatrix()
        self.thickness = thickness * 0.001


    def computeStiffnessMatrix(self):
        deno = 1.0 - self.nu21 * self.nu21
        self.matQ = np.array([[self.E1 / deno, self.nu12 * self.E2 / deno, 0.0],
                              [self.nu12 * self.E2 / deno, self.E2 / deno, 0.0],
                              [0.0, 0.0, self.G12]])

    def computeTransformationMatrix(self):
        rad = np.deg2rad(self.theta)
        c = np.cos(rad)
        s = np.sin(rad)
        self.matT = np.array([
            [c * c, s * s, 2 * s * c],
            [s * s, c * c, -2 * s * c],
            [-s * c, s * c, c * c - s * s]
        ])

    def computeReducedStiffnessMatrix(self):
        self.matQBar = np.linalg.inv(self.matT) @ self.matQ @ self.matR @ self.matT @ np.linalg.inv(self.matR)
        # print("Printing stiffness matrix")
        # printM(self.matQ)
        # print("Printing reduced stiffness matrix")
        # printM(self.matQBar)
        # print("\n")


class Laminate:
    def __init__(self, laminaStack: list[Lamina],
                 deltaT: float,
                 deltaC: float,
                 appliedForce: np.array(3) = np.array([100, 0, 0]),
                 appliedMoment: np.array(3) = np.array([0, 0, 0]),
                 ):
        self.tempMidSurfaceStrainAndCurvature = None
        self.midSurfaceCurvature = None
        self.midSurfaceStrain = None
        self.appliedMoment = np.zeros(3)
        self.appliedForce = np.array([100000, 0, 0])
        self.totalForce = np.array([0, 0, 0])
        self.totalMoment = np.array([0, 0, 0])
        self.hygroMoments = None
        self.thermalMoments = None
        self.hygroForces = None
        self.thermalForces = None
        self.deltaC = deltaC
        self.deltaT = deltaT
        self.matABBD = None
        self.matD = None
        self.matB = None
        self.matA = None
        self.laminaStack = laminaStack
        self.N = len(laminaStack)
        self.zCenter = 0
        for i in range(self.N):
            self.zCenter += self.laminaStack[i].thickness
        self.zCenter /= 2
        self.computeABDMatrix()
        self.computeThermalStresses()
        self.midSurfaceStrainAndCurvature = np.array(6)
        self.computeMidSurfaceStrainsAndCurvature()
        self.computeResidualStrainAndStresses()
        self.computeSigmaValues()

        for i in range(self.N):
            targetIndex = 0
            bSRL = 0
            bSRT = 0
            bSRS = 0

            
            # print("Printing SR")
            for j in range(self.N):
                lamina = laminaStack[j]
                sigma1 = lamina.sigmaValues[0]*0.000001
                sigma2 = lamina.sigmaValues[1]*0.000001
                tou12 = lamina.sigmaValues[2] * 0.000001
                SRL = abs(sigma1)/(lamina.uSigmaT1 if sigma1 > 0 else lamina.uSigmaC1)
                SRT = abs(sigma2)/(lamina.uSigmaT2 if sigma2 > 0 else lamina.uSigmaC2)
                SRS = abs(tou12)/(lamina.uTou12)
                # print(f" {SRL} {SRT} {SRS}")
                mn = max(bSRL, bSRT, bSRS)
                mnn = max(SRL, SRT, SRS)
                bSRL = max(bSRL, SRL)
                bSRT = max(bSRT, SRT)
                bSRS = max(bSRS, SRS)
                if mnn > mn:
                    targetIndex = j

            print(f"The modified ABBD matrix")
            printM(self.matABBD)
            print(f"Lamina {targetIndex + 1} fails")

            sigma = 0
            residual = 0
            ultimate = 0
            maxi = max(bSRL, bSRT, bSRS)

            failingLamina = laminaStack[targetIndex]
            if bSRL == maxi:
                sigma = laminaStack[targetIndex].sigmaValues[0]
                residual = laminaStack[targetIndex].residualStressAlongMaterialAxis[0]
                ultimate = (failingLamina.uSigmaT1 if sigma > 0 else failingLamina.uSigmaC1)
            elif bSRT == maxi:
                sigma = laminaStack[targetIndex].sigmaValues[1]
                residual = laminaStack[targetIndex].residualStressAlongMaterialAxis[1]
                ultimate = (failingLamina.uSigmaT2 if sigma > 0 else failingLamina.uSigmaC2)
            else:
                sigma = laminaStack[targetIndex].sigmaValues[2]
                residual = laminaStack[targetIndex].residualStressAlongMaterialAxis[2]
                ultimate = (failingLamina.uTou12 if sigma > 0 else failingLamina.uTou12)
            # print(f"{ultimate} {residual} {sigma}")
            Nx = self.appliedForce[0] * (ultimate*10**6 - residual)/sigma
            print(f"The Nx value is {Nx*0.001} N/mm \n")
            self.totalForce = np.array([0, 0, 0])
            self.totalMoment = np.array([0, 0, 0])
            self.laminaStack[targetIndex].matQ = np.zeros((3,3))
            self.laminaStack[targetIndex].matQBar = np.zeros((3,3))
            self.computeABDMatrix()
            self.computeThermalStresses()
            self.midSurfaceStrainAndCurvature = np.zeros(6)
            self.computeMidSurfaceStrainsAndCurvature()
            self.computeResidualStrainAndStresses()
            self.computeSigmaValues()

            


    def computeABDMatrix(self):
        self.matA = np.zeros((3, 3))
        self.matB = np.zeros((3, 3))
        self.matD = np.zeros((3, 3))
        z1 = -self.zCenter
        for i in range(self.N):
            lamina: Lamina = self.laminaStack[i]
            z0 = z1
            z1 += lamina.thickness
            self.matA += lamina.thickness * lamina.matQBar
            self.matB += ((z1 * z1 - z0 * z0) / 2) * lamina.matQBar
            self.matD += ((z1 ** 3 - z0 ** 3) / 3) * lamina.matQBar
        self.matABBD = np.block([
            [self.matA, self.matB],
            [self.matB, self.matD]
        ])
        # print("Printing ABBD matrix")
        # printM(self.matABBD)

    def computeThermalStresses(self):
        self.thermalForces = np.zeros(3)
        self.hygroForces = np.zeros(3)
        self.thermalMoments = np.zeros(3)
        self.hygroMoments = np.zeros(3)

        z1 = -self.zCenter
        for i in range(self.N):
            lamina = self.laminaStack[i]
            z0 = z1
            z1 += lamina.thickness
            self.thermalForces += lamina.thickness * (lamina.matQBar @ lamina.thermalParameters)
            self.hygroForces += lamina.thickness * (lamina.matQBar @ lamina.hygroParameters)
            self.thermalMoments += (z1 ** 2 - z0 ** 2) * (lamina.matQBar @ lamina.thermalParameters)
            self.hygroMoments += (z1 ** 2 - z0 ** 2) * (lamina.matQBar @ lamina.hygroParameters)
        self.thermalForces = self.deltaT * self.thermalForces
        self.hygroForces = self.deltaC * self.hygroForces
        self.thermalMoments = self.deltaT / 2 * self.thermalMoments
        self.hygroMoments = self.deltaC / 2 * self.hygroMoments

        # self.totalForce = self.appliedForce
        # self.totalMoment = self.appliedMoment
        # print('Hygro Forces')
        # printM(self.hygroForces)
        # print('Thermal Forces')
        # printM(self.thermalForces)
        # print('Thermal Moments')
        # printM(self.thermalMoments)

    def computeMidSurfaceStrainsAndCurvature(self):
        self.midSurfaceStrainAndCurvature = (np.linalg.inv(self.matABBD) if np.linalg.det(self.matABBD) != 0 else np.zeros((6,6))) @ np.block([self.appliedForce, self.appliedMoment])
        self.tempMidSurfaceStrainAndCurvature = (np.linalg.inv(self.matABBD) if np.linalg.det(self.matABBD) != 0 else np.zeros((6,6))) @ np.block([self.thermalForces, self.thermalMoments])

        self.midSurfaceStrain = np.array(self.midSurfaceStrainAndCurvature[0:3])
        self.midSurfaceCurvature = np.array(self.midSurfaceStrainAndCurvature[3:])
        # print("Mid Surface strains and curvatures")
        # print(self.midSurfaceStrainAndCurvature)
        # print("For temp ")
        # print(self.tempMidSurfaceStrainAndCurvature)

    def computeResidualStrainAndStresses(self):
        z = -self.zCenter
        for i in range(self.N):
            lamina = self.laminaStack[i]
            z += lamina.thickness/2
            lamina.strainAlongAnalysisAxes = self.midSurfaceStrain + z*self.midSurfaceCurvature
            lamina.tempStrainAlongAnalysisAxes = self.tempMidSurfaceStrainAndCurvature[0:3] + z*self.tempMidSurfaceStrainAndCurvature[3:]

            lamina.freeThermalStrains = self.deltaT * lamina.thermalParameters
            lamina.freeHygroStrains = self.deltaC * lamina.hygroParameters
            # print(f"residual thermal strains {lamina.freeThermalStrains}")
            lamina.residualThermalStrains = lamina.tempStrainAlongAnalysisAxes - lamina.freeThermalStrains - lamina.freeHygroStrains
            # print(f"residual strains {lamina.residualThermalStrains}")
            lamina.residualStressAlongAnalysisAxis = lamina.matQBar @ lamina.residualThermalStrains
            lamina.residualStressAlongMaterialAxis = lamina.matT @ lamina.residualStressAlongAnalysisAxis
            # print(f"residual along material {lamina.residualStressAlongMaterialAxis}")
            z += lamina.thickness/2
            # print('Stress Along material axis')
            # printM(lamina.stressAlongMaterialAxis)

    def computeSigmaValues(self):
        for i in range(self.N):
            lamina = self.laminaStack[i]

            lamina.strainAlongAnalysisAxes[2] /= 2
            lamina.strainAlongMaterialAxes = lamina.matT @ lamina.strainAlongAnalysisAxes
            lamina.strainAlongMaterialAxes[2] *= 2
            lamina.strainAlongAnalysisAxes[2] *= 2

            lamina.sigmaValues = lamina.matQ @ lamina.strainAlongMaterialAxes





# gpa, gpa, gpa, , degree, mm, 
# applied force N/m

# A sample lamina with temperature change, taken from sir's slide.
Laminate([
    Lamina(38.6, 8.27, 4.14, 0.28, 0, 0.125,[8.6*10**-6,22.1*10**-6,0], [0,0,0], 1062, 610, 31, 118, 72 ),
    Lamina(38.6, 8.27, 4.14, 0.28, 45, 0.125, [8.6*10**-6,22.1*10**-6,0], [0,0,0], 1062, 610, 31, 118, 72 ),
    Lamina(38.6, 8.27, 4.14, 0.28, -45, 0.125, [8.6*10**-6,22.1*10**-6,0], [0,0,0], 1062, 610, 31, 118, 72 ),
    Lamina(38.6, 8.27, 4.14, 0.28, 90, 0.125, [8.6*10**-6,22.1*10**-6,0], [0,0,0], 1062, 610, 31, 118, 72 ),
    Lamina(38.6, 8.27, 4.14, 0.28, 90, 0.125, [8.6*10**-6,22.1*10**-6,0], [0,0,0], 1062, 610, 31, 118, 72 ),
    Lamina(38.6, 8.27, 4.14, 0.28, -45, 0.125, [8.6*10**-6,22.1*10**-6,0], [0,0,0], 1062, 610, 31, 118, 72 ),
    Lamina(38.6, 8.27, 4.14, 0.28, 45, 0.125, [8.6*10**-6,22.1*10**-6,0], [0,0,0], 1062, 610, 31, 118, 72 ),
    Lamina(38.6, 8.27, 4.14, 0.28, 0, 0.125, [8.6*10**-6,22.1*10**-6,0], [0,0,0], 1062, 610, 31, 118, 72 ),
],
         50,
         0,
         [100000,0,0]
         )
