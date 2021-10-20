import numpy as np
from HardCB import HardCB
from matplotlib import pyplot as plt

OSR = 12
compareUnfiltered = 1
top = 750
bottom = 100
step = 25

adc = HardCB(4)
adc.f_clk = 250e6
adc.SetFloatBitWidth(32)
adc.SetPlotDirectory('test_plot')
adc.ReadOfflineFiles('data')
adc.ReadFIRCoefficients('data', 'pre')
adc.ReadStimuliFile('data/clean_signals')

golden = adc.GoldenBatch()
adc.PlotFigure(golden[1920:-1920], 960, "Golden Batch architecture", 'GoldBatch')


SNR_FIR = []
adc.osr = OSR
# Simulate
x = np.arange(bottom, top+1, step)

for langth in x:
	print("Testing FIR with " + str(langth) + " coefficients...")
	results = adc.FIR(langth, OSR)
	SNR_FIR.append(adc.PlotFigure(results[int(1920/OSR):-int(1920/OSR)], int(round(960 / OSR)), "FIR architecture - length = " + str(langth) + ", OSR = " + str(OSR), "FIR_" + str(langth) + "_OSR" + str(OSR)))

if compareUnfiltered:
	adc.ReadFIRCoefficients('data', 'none')
	SNR_UNFILT = []
	for langth in x:
		print("Testing unfiltered FIR with " + str(langth) + " coefficients...")
		results = adc.FIR(langth, OSR)
		SNR_UNFILT.append(adc.PlotFigure(results[int(1920/OSR):-int(1920/OSR)], int(round(960 / OSR)), "FIR architecture - length = " + str(langth) + ", OSR = " + str(OSR) + ", Unfiltered", "FIR_" + str(langth) + "_OSRUF" + str(OSR)))
	
	# Display every 4th tick
	xdisp = []
	for i in range(0, x.size):
		if (i % 4 == 0):
			xdisp.append(str(x[i]))
		else:
			xdisp.append("")
	# Plot figure
	peak = np.amax(SNR_FIR)
	where = np.where(SNR_FIR == peak)
	plt.figure(figsize=(10,8))
	plt.plot(x, SNR_FIR)
	plt.plot(x, SNR_UNFILT)
	plt.legend(["Prefiltered", "Unfiltered"], loc ="lower right")
	plt.title("SNR for FIR architecture - 32bit, OSR=" + str(OSR))
	plt.ylabel("SNR [dB]")
	plt.xlabel("FIR length")
	plt.minorticks_off()
	plt.xticks(x, xdisp, rotation=45)
	plt.grid(True)
	plt.savefig("test_plot/" + "SNRFIR_32bit_UNFILT_O" + str(OSR))

else:
	adc.PlotSNR(x, SNR_FIR, "SNR for FIR architecture - 32bit, OSR=" + str(OSR), "FIR length", "SNRFIR_32bit_O" + str(OSR))

