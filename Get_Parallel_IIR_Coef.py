from HardCB import HardCB

adc = HardCB()
adc.SetSystemOrder(3)
adc.ReadOfflineFiles('data')
adc.ReadLegacyStimuliFile('data/control_signals')
adc.CalculateIIRCoefficients()

WriteFiles = 1
WriteHead = 1

#Write CSV files
if (WriteFiles):
	adc.WriteCSVFile('data/Lf', adc.Lf)
	adc.WriteCSVFile('data/Lb', adc.Lb)
	adc.WriteCSVFile('data/Ff', adc.Ff)
	adc.WriteCSVFile('data/Fb', adc.Fb)
	adc.WriteCSVFile('data/Wf', adc.Wf)
	adc.WriteCSVFile('data/Wb', adc.Wb)
	adc.WriteCSVFile('data/hardware_signals', adc.GetHardwareStimuli())
	adc.WriteVerilogStimuli('data/verilog_signals')
	adc.WriteCSVFile('data/clean_signals', adc.S)

if (WriteHead):
	adc.WriteCPPCoefficients('data/Coefficients', 256)
	adc.WriteVerilogCoefficients('data/Coefficients', 20)










