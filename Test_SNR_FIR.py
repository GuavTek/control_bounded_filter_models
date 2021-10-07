import numpy as np
from HardCB import HardCB

OverRate = 1
top = 100
step = 20

adc = HardCB()
adc.SetSystemOrder(3)
adc.SetFloatBitWidth(32)
adc.SetPlotDirectory('test_plot')
adc.ReadOfflineFiles('data')
adc.ReadFIRCoefficients('data', 1)
adc.ReadStimuliFile('data/clean_signals')

golden = adc.GoldenBatch()
adc.PlotFigure(golden, int(round(1536 / OverRate)), "Golden Batch architecture", 'GoldBatch')


SNR_FIR = []
# Simulate
x = np.arange(step, top+1, step)
for langth in x:
	print("Testing FIR with length " + str(langth) + "...")
	results = adc.FIR(langth, OverRate)
	SNR_FIR.append(adc.PlotFigure(results, int(round(1536 / OverRate)), "FIR architecture - length = " + str(langth), "FIR_" + str(langth)))

adc.PlotSNR(x, SNR_FIR, "SNR for FIR architecture - 32bit", "FIR length", "SNRFIR_32bit")
