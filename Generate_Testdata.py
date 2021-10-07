import cbadc
from HardCB import HardCB
import numpy as np

N = 3   # Analog states
M = N   # Digital states
samples_num = 32768     # Length of generated test data

f_clk = 240e6   # ADC sampling frequency
OSR = 12        # Oversampling ratio
T = 1.0/f_clk
beta = 1.0/(2 * T)
adc = HardCB()
rho = - 1e-2
kappa = - 1.0
end_time = T * samples_num  # Simulation end

betaVec = beta * np.ones(N)
rhoVec = betaVec * rho
kappaVec = kappa * beta * np.eye(N)

# Set up input signal
amplitude = 0.5
frequency = 5e6


# Instantiate a chain-of-integrators analog system.
analog_system = cbadc.analog_system.ChainOfIntegrators(betaVec, rhoVec, kappaVec)
# print the analog system such that we can very it being correctly initalized.
print(analog_system)

# Initialize the digital control.
digital_control = cbadc.digital_control.DigitalControl(T, M)
# print the digital control to verify proper initialization.
print(digital_control)

# Instantiate the analog signal
analog_signal = cbadc.analog_signal.Sinusodial(amplitude, frequency, phase, offset)
# print to ensure correct parametrization.
print(analog_signal)

# Instantiate the simulator.
simulator = cbadc.simulator.StateSpaceSimulator(analog_system, digital_control, [
                            analog_signal], t_stop=end_time)
print(simulator)

tVectors = np.zeros((N, seq_len))
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

adc.WriteCSVFile("data/clean_signals2", tVectors)