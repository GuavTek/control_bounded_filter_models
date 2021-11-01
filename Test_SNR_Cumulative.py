import numpy as np
from HardCB import HardCB

OverRate = 1
top = 500
bottom = 500
step = 25

n_int = 8
n_frac = 23

adc = HardCB(4)
adc.f_clk = 240e6
adc.SetFixedBitWidth(n_int, n_frac)
adc.SetPlotDirectory('fixed_plot')
adc.ReadOfflineFiles('data')
adc.ReadIIRCoefficients('data')
adc.ReadStimuliFile('data/clean_signals')

#golden = adc.GoldenBatch()
#adc.PlotFigure(golden[1920:-1920], int(round(960 / OverRate)), "Golden Batch architecture", 'GoldBatch')

#golden = adc.GoldenParallel()
#adc.PlotFigure(golden[1920:-1920], int(round(960 / OverRate)), "Golden Parallel architecture", 'GoldParallelBatch')


SNR_Cumulative = []
# Simulate
x = np.arange(bottom, top+1, step)
for langth in x:
	print("Testing Cumulative with parameter " + str(langth) + "...")
	results = adc.CumulativeIIR(langth)
	SNR_Cumulative.append(adc.PlotFigure(results[int(1920/OverRate):-int(1920/OverRate)], int(round(960 / OverRate)), "Cumulative architecture - Lookahead length = " + str(langth), "Cumul_" + str(langth)))
	
adc.PlotSNR(x, SNR_Cumulative, "SNR for cumulative architecture - " + str(n_int+1) + "." + str(n_frac) + "bit", "Lookahead length", "SNRCumul_" + str(n_int+1) + "p" + str(n_frac) + "bit")
