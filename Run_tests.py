import Test_Architecture as arc

arc.SetBitWidth(32)
arc.SetTestParameters(16384, 3)
arc.SetPlotParameters(512, 16)

arc.DirectoryCheck()

arc.RunGolden(0)
arc.RunGolden(1)

arc.RunBatchTest(32767)
arc.RunSNRBatch(plotTop, plotStep)
arc.RunSNRFIIR(plotTop, plotStep)