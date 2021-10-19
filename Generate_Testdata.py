from cbadc.digital_estimator import DigitalEstimator
from HardCB import *
import cbadc
import numpy as np


N = 4   # Analog states
M = N   # Digital states
samples_num = 24000     # Length of generated test data
FIR_size = 400          # Number of coefficients generated
adc = HardCB(M)

f_clk = 250e6           # ADC sampling frequency
fu = 20e6               # integrator Unity gain frequency
fc = 5e6                # Filter cut-off
kappa = -1.0
OSR = 12                # Oversampling ratio

# Input signal
amplitude = 0.8
fs = 500e3

T = 1.0/f_clk
beta = 1/(2*T)

#beta = 2*np.pi*fu
#T = 1.0/(2*beta)

wp = 2*np.pi*fc
rho = -wp**2/(4*beta)

betaVec = beta * np.ones(N)
rhoVec = np.array([0 if i==0 else rho for i in range(N)])
gammaVec = kappa * beta * np.eye(N)

# Instantiate a leapfrog analog system.
analog_system = cbadc.analog_system.LeapFrog(betaVec, rhoVec, gammaVec)
# print the analog system such that we can very it being correctly initalized.
print(analog_system)

# Initialize the digital control.
digital_control = cbadc.digital_control.DigitalControl(T, M)
# print the digital control to verify proper initialization.
print(digital_control)

# Create band-limiting filter
gpass = 0.1
gstop = 80
filter = cbadc.analog_system.IIRDesign(wp=wp, ws=2*wp, gpass=gpass, gstop=gstop, ftype="ellip")

# Create new analog system with filter
analog_system_prefiltered = cbadc.analog_system.chain([filter, analog_system])
print(analog_system_prefiltered)

G_at_omega = np.linalg.norm(analog_system.transfer_function_matrix(np.array([wp/2])))
eta2 = G_at_omega**2

##### Generate IIR coefficients #####

# Initialize estimator
digital_estimator = cbadc.digital_estimator.ParallelEstimator(analog_system, digital_control, eta2, samples_num)
print(digital_estimator)

# Write to csv files
adc.WriteCSVFile("data/Ab", digital_estimator.Ab)
adc.WriteCSVFile("data/Af", digital_estimator.Af)
adc.WriteCSVFile("data/Bb", digital_estimator.Bb)
adc.WriteCSVFile("data/Bf", digital_estimator.Bf)
adc.WriteCSVFile("data/WT", digital_estimator.WT)

# Generate IIR coefficients
#adc.ReadOfflineFiles('data')
#adc.CalculateIIRCoefficients()
#adc.WriteCSVFile('data/Lf', adc.Lf) 
#adc.WriteCSVFile('data/Lb', adc.Lb)
#adc.WriteCSVFile('data/Ff', adc.Ff)
#adc.WriteCSVFile('data/Fb', adc.Fb)
#adc.WriteCSVFile('data/Wf', adc.Wf)
#adc.WriteCSVFile('data/Wb', adc.Wb)

adc.WriteCSVFile('data/Lf', digital_estimator.forward_a) 
adc.WriteCSVFile('data/Lb', digital_estimator.backward_a)
adc.WriteCSVFile('data/Ff', digital_estimator.forward_b)
adc.WriteCSVFile('data/Fb', digital_estimator.backward_b)
adc.WriteCSVFile('data/Wf', np.reshape(digital_estimator.forward_w, M))
adc.WriteCSVFile('data/Wb', np.reshape(digital_estimator.backward_w, M))

##### Generate FIR coefficients #####

# Prepare FIR parameters
L1 = FIR_size
L2 = FIR_size

# Instantiate FIR filter without Downsampling
FIR_estimator_ref = cbadc.digital_estimator.FIRFilter(analog_system, digital_control, eta2, L1, L2)
print(FIR_estimator_ref)

# Write Coefficients to files
h1 = FIR_estimator_ref.h[0][0:FIR_size]
h2 = FIR_estimator_ref.h[0][FIR_size:]

hb = h2
hf = np.zeros((FIR_size, M))
for i in range(0, FIR_size):
    hf[i] = h1[FIR_size-i-1]
adc.WriteCSVFile("data/FIR1_hb", hb)
adc.WriteCSVFile("data/FIR1_hf", hf)

# Write verilog header
adc.ReadIIRCoefficients('data')
adc.ReadFIRCoefficients('data', 1)
adc.WriteVerilogIIRCoefficients('data/Coefficients', 20)
adc.WriteVerilogFIRCoefficients('data/Coefficients_FIR1', 1)

#G_at_omega = np.linalg.norm(analog_system_new.transfer_function_matrix(np.array([wp/2])))
#eta2 = G_at_omega**2

# Instantiate FIR filter with downsampling
#FIR_estimator_ds = cbadc.digital_estimator.FIRFilter(analog_system_new, digital_control, eta2, L1, L2, downsample=OSR)
#print(FIR_estimator_ds)

# Write coefficients to file
#h1 = FIR_estimator_ref.h[0][0:FIR_size]
#h2 = FIR_estimator_ref.h[0][FIR_size:]
#hb = h2
#hf = np.zeros((FIR_size, M))
#for i in range(0, FIR_size):
#    hf[i] = h1[FIR_size-i-1]
#adc.WriteCSVFile("data/FIR" + str(OSR) + "_hf", hf)
#adc.WriteCSVFile("data/FIR" + str(OSR) + "_hb", hb)

##### Generate Stimuli #####

# Instantiate the analog signal
analog_signal = cbadc.analog_signal.Sinusodial(amplitude, fs)
# print to ensure correct parametrization.
print(analog_signal)

end_time = T * samples_num  # Simulation end

# Instantiate the simulator.
simulator = cbadc.simulator.StateSpaceSimulator(analog_system, digital_control, [
                            analog_signal], t_stop=end_time)
print(simulator)

tVectors = np.zeros((N, samples_num), int)
x = 0
for s in simulator:
    y = 0
    for num in s:
        if (num == True):
            tVectors[y][x] = 1
        elif (num == False):
            tVectors[y][x] = -1
        y += 1
    x += 1

# Write stimuli files
adc.WriteCSVFile("data/clean_signals", tVectors)
adc.ReadStimuliFile("data/clean_signals")
adc.WriteCSVFile('data/hardware_signals', adc.GetHardwareStimuli())
adc.WriteVerilogStimuli('data/verilog_signals')


