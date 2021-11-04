from cbadc.digital_estimator import DigitalEstimator
from HardCB import *
import cbadc
import numpy as np

# 0 == write files
# 1 == reference FIR
# 2 == reference IIR
# 3 == downsampled FIR with prefilter
# 4 == downsampled FIR no prefilter
plot_reference = 0

N = 4   # Analog states
M = N   # Digital states
samples_num = 24000     # Length of generated stmuli
FIR_size = 400          # Number of coefficients generated
adc = HardCB(M)

adc.DirectoryCheck('data')

f_clk = 240e6           # ADC sampling frequency
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
print(analog_system)

# Initialize the digital control.
digital_control = cbadc.digital_control.DigitalControl(T, M)
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

# Initialize estimator
digital_estimator = cbadc.digital_estimator.ParallelEstimator(analog_system, digital_control, eta2, samples_num)
print(digital_estimator)

# Prepare FIR parameters
L1 = FIR_size
L2 = FIR_size

# Instantiate FIR filter without Downsampling
FIR_estimator_ref = cbadc.digital_estimator.FIRFilter(analog_system, digital_control, eta2, L1, L2)
print(FIR_estimator_ref)

G_at_omega = np.linalg.norm(analog_system_prefiltered.transfer_function_matrix(np.array([wp/2])))
eta2 = G_at_omega**2

# Instantiate FIR filter with downsampling
FIR_estimator_ds = cbadc.digital_estimator.FIRFilter(analog_system_prefiltered, digital_control, eta2, L1, L2, downsample=OSR)
print(FIR_estimator_ds)

end_time = T * samples_num  # Simulation end

# Instantiate the analog signal
analog_signal = cbadc.analog_signal.Sinusodial(amplitude, fs)

# Instantiate the simulator.
simulator = cbadc.simulator.StateSpaceSimulator(analog_system, digital_control, [
                            analog_signal], t_stop=end_time)

# print to ensure correct parametrization.
print(analog_signal)
print(simulator)

# Write files
if plot_reference == 0:
    # Write offline coefficients to csv files
    adc.WriteCSVFile("data/Ab", digital_estimator.Ab)
    adc.WriteCSVFile("data/Af", digital_estimator.Af)
    adc.WriteCSVFile("data/Bb", digital_estimator.Bb)
    adc.WriteCSVFile("data/Bf", digital_estimator.Bf)
    adc.WriteCSVFile("data/WT", digital_estimator.WT)

    # Write IIR coefficients
    adc.WriteCSVFile('data/Lf', digital_estimator.forward_a) 
    adc.WriteCSVFile('data/Lb', digital_estimator.backward_a)
    adc.WriteCSVFile('data/Ff', digital_estimator.forward_b)
    adc.WriteCSVFile('data/Fb', digital_estimator.backward_b)
    adc.WriteCSVFile('data/Wf', np.reshape(digital_estimator.forward_w, M))
    adc.WriteCSVFile('data/Wb', np.reshape(digital_estimator.backward_w, M))

    # Write FIR Coefficients without prefilter
    h1 = FIR_estimator_ref.h[0][0:FIR_size]
    h2 = FIR_estimator_ref.h[0][FIR_size:]

    hb_ref = h2
    hf_ref = np.zeros((FIR_size, M))
    for i in range(0, FIR_size):
        hf_ref[i] = h1[FIR_size-i-1]
    adc.WriteCSVFile("data/FIR_hb", hb_ref)
    adc.WriteCSVFile("data/FIR_hf", hf_ref)

    # Write FIR Coefficients with prefilter
    h1 = FIR_estimator_ds.h[0][0:FIR_size]
    h2 = FIR_estimator_ds.h[0][FIR_size:]
    hb_ds = h2
    hf_ds = np.zeros((FIR_size, M))
    for i in range(0, FIR_size):
        hf_ds[i] = h1[FIR_size-i-1]
    adc.WriteCSVFile("data/FIR_hf_prefilt", hf_ds)
    adc.WriteCSVFile("data/FIR_hb_prefilt", hb_ds)

    # Write verilog headers
    adc.ReadIIRCoefficients('data')
    adc.ReadFIRCoefficients('data', 'none')
    adc.WriteVerilogCoefficients('data/Coefficients', 20)
    adc.WriteVerilogCoefficients_Fixedpoint('data/Coefficients_Fixedpoint', 20, 48)

    adc.ReadFIRCoefficients('data', 'pre')
    adc.WriteVerilogFIRCoefficients('data/Coefficients_FIR_prefilt')

    # Format stimuli
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

if plot_reference != 0:
    adc.SetPlotDirectory('cbadc_plot')

# Plot results from FIR estimator without downsampling
if plot_reference == 1:
    # Plot reference 
    FIR_estimator_ref(simulator)
    u_ref = []
    for result in FIR_estimator_ref:
        u_ref.append(result)
    u_ref = np.array(u_ref)
    u_ref = u_ref.flatten()

    adc.PlotFigure(u_ref[1920:-1919], 1000, "FIR Estimator Reference " + str(FIR_size) + " coefficients", "cbadc_FIR" + str(FIR_size))

# Plot results from parallel estimator without downsampling
if plot_reference == 2:
    IIR_estimator_ref = cbadc.digital_estimator.ParallelEstimator(analog_system, digital_control, eta2, FIR_size, FIR_size)
    IIR_estimator_ref(simulator)
    u_ref = []
    for result in IIR_estimator_ref:
        u_ref.append(result)
    u_ref = np.array(u_ref)
    u_ref = u_ref.flatten()

    slice_ends = 1920 - int(FIR_size/2)
    adc.PlotFigure(u_ref[slice_ends:-slice_ends], 1000, "IIR Estimator Reference " + str(FIR_size) + " batch size", "cbadc_batch" + str(FIR_size))

# Plot results from FIR estimator with downsampling and prefiltering
if plot_reference == 3:
    FIR_estimator_ds(simulator)
    u_ds = []
    for result in FIR_estimator_ds:
        u_ds.append(result)
    u_ds = np.array(u_ds)
    u_ds = u_ds.flatten()

    adc.PlotFigure(u_ds[int(1920/OSR):-int(1920/OSR)+1], int(960/OSR), "FIR Estimator Reference " + str(FIR_size) + " coefficients, OSR=" + str(OSR), "cbadc_FIR" + str(FIR_size) + "_OSR" + str(OSR))

# Plot results from FIR estimator with downsampling, without prefiltering
if plot_reference == 4:
    FIR_estimator_ds = cbadc.digital_estimator.FIRFilter(analog_system, digital_control, eta2, L1, L2, downsample=OSR)
    FIR_estimator_ds(simulator)
    u_ds = []
    for result in FIR_estimator_ds:
        u_ds.append(result)
    u_ds = np.array(u_ds)
    u_ds = u_ds.flatten()

    adc.PlotFigure(u_ds[int(1920/OSR):-int(1920/OSR)+1], int(960/OSR), "FIR Estimator unfiltered Reference " + str(FIR_size) + " coefficients, OSR=" + str(OSR), "cbadc_FIR" + str(FIR_size) + "_OSRUF" + str(OSR))

# Plot results with postfiltering...

