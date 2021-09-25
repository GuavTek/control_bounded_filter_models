from HardCB import HardCB
import numpy as np
from matplotlib import pyplot as plt

adc = HardCB()
adc.SetPlotDirectory('test_plot')

res = adc.ReadResultFile('Results', 0)

adc.PlotFigure(res, 4000, "Waveform from Verilog simulation", "VerilogPlot")
#u_bat = arc.TestBatch(256)
#arc.PlotPSD(u_bat, "", 1)
#SNR = arc.PlotPSD(res, "PSD for stratus", 1)
#plt.figtext(0.13, 0.42, "SNR = " + ('%.2f' % SNR) + "dB")
#plt.legend(["Python simulation", "Verilog simulation"])
#plt.savefig(("StratusPlot_32bit_Veri"))
#plt.show()
#plt.close()
