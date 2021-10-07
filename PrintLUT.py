from HardCB import HardCB

adc = HardCB()
adc.SetSystemOrder(3)
adc.SetFloatBitWidth(32)
adc.SetPlotDirectory('test_plot')
adc.ReadIIRCoefficients('data')
adc.PrintLUTValues()