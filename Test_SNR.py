import numpy as np
from HardCB import HardCB

OverRate = 1
top = 250
step = 25

adc = HardCB()
adc.SetSystemOrder(3)
adc.SetFloatBitWidth(32)
adc.SetPlotDirectory('test_plot')
adc.ReadOfflineMatrixFile('data/offline_matrices')
adc.ReadParallelCoefficients('data')
adc.ReadStimuliFile('data/clean_signals')

golden = adc.GoldenBatch()
adc.PlotFigure(golden, int(round(1536 / OverRate)), "Golden Batch architecture", 'GoldBatch')


SNR_Batch = []
SNR_FIIR = []
# Simulate
x = np.arange(step, top+1, step)
for langth in x:
	print("Testing batch with parameter " + str(langth) + "...")
	results = adc.BatchIIR(langth, OverRate)
	SNR_Batch.append(adc.PlotFigure(results, int(round(1536 / OverRate)), "Batch architecture - batch size = " + str(langth), "Batch_" + str(langth)))
	print("Testing FIIR with lookahead " + str(langth) + "...")
	results = adc.FIIR(langth)
	SNR_FIIR.append(adc.PlotFigure(results, int(round(1536 / OverRate)), "FIIR architecture - lookahead size = " + str(langth), "FIIR_" + str(langth)))

adc.PlotSNR(x, SNR_Batch, "SNR for Batch architecture - 32bit", "Batch Size", "SNRBatch_32bit")
adc.PlotSNR(x, SNR_FIIR, "SNR for FIIR architecture - 32bit", "Lookahead Size", "SNRFIIR_32bit")
