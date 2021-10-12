import numpy as np
from HardCB import HardCB

OverRate = 1
top = 240
step = 20

adc = HardCB()
adc.f_clk = 240e6
adc.SetSystemOrder(3)
adc.SetFloatBitWidth(32)
adc.SetPlotDirectory('test_plot')
adc.ReadOfflineFiles('data')
adc.ReadIIRCoefficients('data')
adc.ReadStimuliFile('data/clean_signals2')

#golden = adc.GoldenBatch()
#adc.PlotFigure(golden, int(round(1536 / OverRate)), "Golden Batch architecture", 'GoldBatch')


SNR_Batch = []
# Simulate
x = np.arange(step, top+1, step)
for langth in x:
	print("Testing batch with parameter " + str(langth) + "...")
	results = adc.BatchIIR(langth, OverRate)
	SNR_Batch.append(adc.PlotFigure(results[1920:-1920], int(round(960 / OverRate)), "Batch architecture - batch size = " + str(langth), "Batch_" + str(langth)))
	
adc.PlotSNR(x, SNR_Batch, "SNR for Batch architecture - 32bit", "Batch Size", "SNRBatch_32bit")
