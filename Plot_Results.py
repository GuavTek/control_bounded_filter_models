import Test_Architecture as arc
from matplotlib import pyplot as plt

arc.SetBitWidth(32)
arc.SetTestParameters(16384, 3)
arc.SetPlotParameters(512, 16)


res = arc.ReadResults('Results.csv', -14)

plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
arc.PlotWave(res, 1536, "Waveform from stratus simulation")

plt.subplot(2,1,2)
SNR = arc.PlotPSD(res, "PSD for stratus")
plt.figtext(0.13, 0.42, "SNR = " + ('%.2f' % SNR) + "dB")
plt.savefig(("StratusPlot_20bit"))
plt.close()