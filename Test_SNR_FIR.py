import numpy as np
from HardCB import HardCB

OverRate = 1
top = 240
bottom = 80
step = 20

adc = HardCB(4)
adc.f_clk = 250e6
adc.SetFloatBitWidth(32)
adc.SetPlotDirectory('test_plot')
adc.ReadOfflineFiles('data')
adc.ReadFIRCoefficients('data', OverRate)
adc.ReadStimuliFile('data/clean_signals')

golden = adc.GoldenBatch()
adc.PlotFigure(golden[2000:-2000], int(round(1000 / OverRate)), "Golden Batch architecture", 'GoldBatch')


SNR_FIR = []
adc.osr = OverRate
# Simulate
x = np.arange(bottom, top+1, step)
for langth in x:
	print("Testing FIR with length " + str(langth) + "...")
	results = adc.FIR(langth, OverRate)
	SNR_FIR.append(adc.PlotFigure(results[int(2000/OverRate):int(-2000/OverRate)], int(round(1000 / OverRate)), "FIR architecture - length = " + str(langth), "FIR_" + str(langth)))

adc.PlotSNR(x, SNR_FIR, "SNR for FIR architecture - 32bit", "FIR length", "SNRFIR_32bit")
