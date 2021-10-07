import numpy as np
from HardCB import HardCB

OverRate = 1
top = 240
step = 20

adc = HardCB()
adc.SetSystemOrder(3)
adc.SetFloatBitWidth(32)
adc.SetPlotDirectory('test_plot')
adc.ReadOfflineMatrixFile('data/offline_matrices')
adc.ReadIIRCoefficients('data')
adc.ReadStimuliFile('data/clean_signals')

results = adc.BatchIIR(220, OverRate)
