from HardCB import *
import cbadc
import numpy as np

N = 3   # Analog states
M = N   # Digital states
samples_num = 32768     # Length of generated test data
FIR_size = 256

f_clk = 240e6   # ADC sampling frequency
OSR = 12        # Oversampling ratio
T = 1.0/f_clk
beta = 1.0/(2 * T)
adc = HardCB()
rho = - 1e-2
kappa = - 1.0
eta2 = 1e7
end_time = T * samples_num  # Simulation end

betaVec = beta * np.ones(N)
rhoVec = betaVec * rho
kappaVec = kappa * beta * np.eye(N)

# Set up input signal
amplitude = 0.6
fs = 500e3


# Instantiate a chain-of-integrators analog system.
analog_system = cbadc.analog_system.ChainOfIntegrators(betaVec, rhoVec, kappaVec)
# print the analog system such that we can very it being correctly initalized.
print(analog_system)

# Initialize the digital control.
digital_control = cbadc.digital_control.DigitalControl(T, M)
# print the digital control to verify proper initialization.
print(digital_control)

# Initialize estimator
digital_estimator = cbadc.digital_estimator.DigitalEstimator(analog_system, digital_control, eta2, samples_num)
print(digital_estimator)

# Write to csv files
adc.WriteCSVFile("data/Ab", digital_estimator.Ab)
adc.WriteCSVFile("data/Af", digital_estimator.Af)
adc.WriteCSVFile("data/Bb", digital_estimator.Bb)
adc.WriteCSVFile("data/Bf", digital_estimator.Bf)
adc.WriteCSVFile("data/WT", digital_estimator.WT)

# Instantiate the analog signal
analog_signal = cbadc.analog_signal.Sinusodial(amplitude, fs)
# print to ensure correct parametrization.
print(analog_signal)

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
adc.WriteCSVFile("data/clean_signals2", tVectors)
adc.ReadStimuliFile("data/clean_signals2")
adc.WriteCSVFile('data/hardware_signals', adc.GetHardwareStimuli())
adc.WriteVerilogStimuli('data/verilog_signals')

# Prepare FIR parameters
omega_3dB = 2 * np.pi /(T * OSR)
G_at_omega = np.linalg.norm(analog_system.transfer_function_matrix(np.array([omega_3dB/2])))
eta2 = G_at_omega**2
L1 = FIR_size
L2 = FIR_size

# Instantiate FIR filter without Downsampling
FIR_estimator_ref = cbadc.digital_estimator.FIRFilter(analog_system, digital_control, eta2, L1, L2)
print(FIR_estimator_ref)

# Write Coefficients to files
#h1 = FIR_estimator_ref.h[0][0:FIR_size-1]
#h2 = FIR_estimator_ref.h[0][FIR_size:-1]
#print("h1 = ")
#print(h1)
#print("h2 = ")
#print(h2)
#adc.WriteCSVFile("data/FIR1_h1", h1)
#adc.WriteCSVFile("data/FIR1_h2", h2)

hb = np.zeros((FIR_size, M))
hf = np.zeros((FIR_size, M))
tempb = FIR_estimator_ref.Bb
tempf = FIR_estimator_ref.Bf
for i in range(0, FIR_size):
    hb[i] = np.dot(FIR_estimator_ref.WT, tempb)
    hf[i] = np.dot(FIR_estimator_ref.WT, -tempf)
    tempb = np.dot(FIR_estimator_ref.Ab, tempb)
    tempf = np.dot(FIR_estimator_ref.Af, tempf)
adc.WriteCSVFile("data/FIR1_hb", hb)
adc.WriteCSVFile("data/FIR1_hf", hf)

# Create anti-alias filter
wp = omega_3dB / 2.0
ws = omega_3dB
gpass = 0.1
gstop = 80

filter = cbadc.analog_system.IIRDesign(wp, ws, gpass, gstop, ftype="ellip")

# Create new analog system with filter
analog_system_new = cbadc.analog_system.chain([filter, analog_system])
print(analog_system_new)

# Instantiate FIR filter with downsampling
FIR_estimator_ds = cbadc.digital_estimator.FIRFilter(analog_system_new, digital_control, eta2, L1, L2, downsample=OSR)
print(FIR_estimator_ds)

# Write coefficients to file
#h1 = FIR_estimator_ds.h[0][0:FIR_size-1]
#h2 = FIR_estimator_ds.h[0][FIR_size:-1]
h2 = np.zeros((FIR_size, M))
h1 = np.zeros((FIR_size, M))
temp2 = FIR_estimator_ds.Bb
temp1 = FIR_estimator_ds.Bf
for i in range(0, FIR_size):
    h2[i] = np.dot(FIR_estimator_ds.WT, temp2)
    h1[i] = np.dot(FIR_estimator_ds.WT, -temp1)
    temp2 = np.dot(FIR_estimator_ds.Ab, temp2)
    temp1 = np.dot(FIR_estimator_ds.Af, temp1)
adc.WriteCSVFile("data/FIR" + str(OSR) + "_hf", h1)
adc.WriteCSVFile("data/FIR" + str(OSR) + "_hb", h2)
