import Test_Architecture as arc
import numpy as np
from matplotlib import pyplot as plt

arc.SetBitWidth(32)
arc.SetTestParameters(32768, 3)

res = arc.ReadResults('Results.csv', 0)

plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
arc.PlotWave(res, 4000, "Waveform from Verilog simulation")

plt.subplot(2,1,2)
u_bat = arc.TestBatch(256)
arc.PlotPSD(u_bat, "", 1)
SNR = arc.PlotPSD(res, "PSD for stratus", 1)
plt.figtext(0.13, 0.42, "SNR = " + ('%.2f' % SNR) + "dB")
plt.legend(["Python simulation", "Verilog simulation"])
plt.savefig(("StratusPlot_32bit_Veri"))
plt.show()
plt.close()
