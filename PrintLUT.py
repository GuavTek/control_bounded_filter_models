from HardCB import HardCB

adc = HardCB()
adc.SetSystemOrder(3)
adc.SetFloatBitWidth(32)
adc.SetPlotDirectory('test_plot')
adc.ReadParallelCoefficients('data')
adc.PrintLUTValues()