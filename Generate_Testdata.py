from cbadc.digital_estimator import DigitalEstimator
from HardCB import *
import cbadc
import scipy
import numpy as np

# 0 == write files
# 1 == reference FIR
# 2 == reference IIR
# 3 == downsampled FIR with prefilter
# 4 == downsampled FIR with postfilter
# 5 == downsampled FIR no filter
plot_reference = 0

N = 4   # Analog states
M = N   # Digital states
samples_num = 24000     # Length of generated stmuli
FIR_size = 300          # Number of coefficients generated
adc = HardCB(N,M)

adc.DirectoryCheck('data')

f_clk = 960e6           # ADC sampling frequency
fu = 20e6               # integrator Unity gain frequency
fc = 5e6                # Filter cut-off
kappa = -1.0
OSR = 6                # Oversampling ratio

# Input signal
amplitude = 0.95
fs = 1.95e6
shape = 'sine'    # sine / ramp / dc

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

G_at_omega = np.linalg.norm(analog_system.transfer_function_matrix(np.array([wp])))
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

# Instantiate FIR filter
FIR_estimator_pre = cbadc.digital_estimator.FIRFilter(analog_system_prefiltered, digital_control, eta2, L1, L2, downsample=OSR)
print(FIR_estimator_pre)

postFilter = scipy.signal.firwin(1 << 10, 1.0/OSR)
FIR_estimator_post = cbadc.digital_estimator.FIRFilter(analog_system, digital_control, eta2, L1, L2, downsample=OSR)
FIR_estimator_post.convolve(postFilter)
print(FIR_estimator_post)

end_time = T * samples_num  # Simulation end

# Instantiate the analog signal
if shape == 'ramp':
    analog_signal = cbadc.analog_signal.Ramp(amplitude, 1/fs)
elif shape == 'dc':
    analog_signal = cbadc.analog_signal.ConstantSignal(amplitude)
else:
    analog_signal = cbadc.analog_signal.Sinusodial(amplitude, fs)

# Instantiate the simulator.
simulator = cbadc.simulator.StateSpaceSimulator(analog_system, digital_control, [
                            analog_signal], t_stop=end_time)

# print to ensure correct parametrization.
#print(analog_signal)
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
    h1 = FIR_estimator_pre.h[0][0:FIR_size]
    h2 = FIR_estimator_pre.h[0][FIR_size:]
    hb_ds = h2
    hf_ds = np.zeros((FIR_size, M))
    for i in range(0, FIR_size):
        hf_ds[i] = h1[FIR_size-i-1]
    adc.WriteCSVFile("data/FIR_hf_prefilt", hf_ds)
    adc.WriteCSVFile("data/FIR_hb_prefilt", hb_ds)

    # Write FIR Coefficients with postfilter
    h1 = FIR_estimator_post.h[0][0:FIR_size]
    h2 = FIR_estimator_post.h[0][FIR_size:]
    hb_ds = h2
    hf_ds = np.zeros((FIR_size, M))
    for i in range(0, FIR_size):
        hf_ds[i] = h1[FIR_size-i-1]
    adc.WriteCSVFile("data/FIR_hf_postfilt", hf_ds)
    adc.WriteCSVFile("data/FIR_hb_postfilt", hb_ds)


    # Write verilog headers
    adc.ReadIIRCoefficients('data')
    adc.ReadFIRCoefficients('data', 'none')
    adc.WriteVerilogCoefficients(f'data/Coefficients_{N}N{M}M_F{int(round(f_clk/1e6))}', 20)
    adc.WriteVerilogCoefficients_Fixedpoint(f'data/Coefficients_Fxp_{N}N{M}M_F{int(round(f_clk/1e6))}', 20, 48)

    adc.ReadFIRCoefficients('data', 'pre')
    adc.WriteVerilogFIRCoefficients(f'data/Coefficients_FIR_prefilt_{N}N{M}M_F{int(round(f_clk/1e6))}')
    adc.WriteVerilogCoefficients_Fixedpoint(f'data/Coefficients_FxPre_{N}N{M}M_F{int(round(f_clk/1e6))}', 20, 48)

    adc.ReadFIRCoefficients('data', 'post')
    adc.WriteVerilogFIRCoefficients(f'data/Coefficients_FIR_postfilt_{N}N{M}M_F{int(round(f_clk/1e6))}')
    adc.WriteVerilogCoefficients_Fixedpoint(f'data/Coefficients_FxPost_{N}N{M}M_F{int(round(f_clk/1e6))}', 20, 48)

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
    adc.WriteCSVFile(f"data/clean_signals_{N}N{M}M_F{int(round(f_clk/1e6))}", tVectors)
    adc.ReadStimuliFile(f"data/clean_signals_{N}N{M}M_F{int(round(f_clk/1e6))}")
    adc.WriteCSVFile(f'data/hardware_signals_{N}N{M}M_F{int(round(f_clk/1e6))}', adc.GetHardwareStimuli())
    adc.WriteVerilogStimuli(f'data/verilog_signals_{N}N{M}M_F{int(round(f_clk/1e6))}')

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
    FIR_estimator_pre(simulator)
    u_ds = []
    for result in FIR_estimator_pre:
        u_ds.append(result)
    u_ds = np.array(u_ds)
    u_ds = u_ds.flatten()

    adc.PlotFigure(u_ds[int(1920/OSR):-int(1920/OSR)+1], int(960/OSR), "FIR Estimator prefilter Reference " + str(FIR_size) + " coefficients, OSR=" + str(OSR), "cbadc_FIRPre" + str(FIR_size) + "_OSR" + str(OSR))

# Plot results from FIR estimator with downsampling and postfiltering
if plot_reference == 4:
    FIR_estimator_post(simulator)
    u_ds = []
    for result in FIR_estimator_post:
        u_ds.append(result)
    u_ds = np.array(u_ds)
    u_ds = u_ds.flatten()

    adc.PlotFigure(u_ds[int(1920/OSR):-int(1920/OSR)+1], int(960/OSR), "FIR Estimator postfilter Reference " + str(FIR_size) + " coefficients, OSR=" + str(OSR), "cbadc_FIRPost" + str(FIR_size) + "_OSR" + str(OSR))

# Plot results from FIR estimator with downsampling, without extra filtering
if plot_reference == 5:
    FIR_estimator_ds = cbadc.digital_estimator.FIRFilter(analog_system, digital_control, eta2, L1, L2, downsample=OSR)
    FIR_estimator_ds(simulator)
    u_ds = []
    for result in FIR_estimator_ds:
        u_ds.append(result)
    u_ds = np.array(u_ds)
    u_ds = u_ds.flatten()

    adc.PlotFigure(u_ds[int(1920/OSR):-int(1920/OSR)+1], int(960/OSR), "FIR Estimator unfiltered Reference " + str(FIR_size) + " coefficients, OSR=" + str(OSR), "cbadc_FIR" + str(FIR_size) + "_OSR" + str(OSR))


