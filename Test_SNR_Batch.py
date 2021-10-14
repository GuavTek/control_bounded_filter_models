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
adc.ReadIIRCoefficients('data')
adc.ReadStimuliFile('data/clean_signals')

golden = adc.GoldenBatch()
adc.PlotFigure(golden[2000:-2000], int(round(1000 / OverRate)), "Golden Batch architecture", 'GoldBatch')

golden = adc.GoldenParallel()
adc.PlotFigure(golden[2000:-2000], int(round(1000 / OverRate)), "Golden Parallel architecture", 'GoldParallelBatch')


SNR_Batch = []
# Simulate
x = np.arange(bottom, top+1, step)
for langth in x:
	print("Testing batch with parameter " + str(langth) + "...")
	results = adc.BatchIIR(langth, OverRate)
	SNR_Batch.append(adc.PlotFigure(results[2000:-2000], int(round(1000 / OverRate)), "Batch architecture - batch size = " + str(langth), "Batch_" + str(langth)))
	
adc.PlotSNR(x, SNR_Batch, "SNR for Batch architecture - 32bit", "Batch Size", "SNRBatch_32bit")
