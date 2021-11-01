import numpy as np
from HardCB import HardCB

OverRate = 1
top = 1000
bottom = 1000
step = 25

adc = HardCB(4)
adc.f_clk = 240e6
adc.SetFixedBitWidth(8, 16)
adc.SetPlotDirectory('fixed_plot')
adc.ReadOfflineFiles('data')
adc.ReadIIRCoefficients('data')
adc.ReadStimuliFile('data/clean_signals')

#golden = adc.GoldenBatch()
#adc.PlotFigure(golden[1920:-1920], int(round(960 / OverRate)), "Golden Batch architecture", 'GoldBatch')

#golden = adc.GoldenParallel()
#adc.PlotFigure(golden[1920:-1920], int(round(960 / OverRate)), "Golden Parallel architecture", 'GoldParallelBatch')


SNR_Batch = []
# Simulate
x = np.arange(bottom, top+1, step)
for langth in x:
	print("Testing batch with parameter " + str(langth) + "...")
	results = adc.BatchIIRFixed(langth, OverRate)
	SNR_Batch.append(adc.PlotFigure(results[int(1920/OverRate):-int(1920/OverRate)], int(round(960 / OverRate)), "Batch architecture - batch size = " + str(langth), "Batch_" + str(langth)))
	
adc.PlotSNR(x, SNR_Batch, "SNR for Batch architecture - 32bit", "Batch Size", "SNRBatch_32bit")
