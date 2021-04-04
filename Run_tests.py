import Test_Architecture as arc

arc.SetBitWidth(32)
arc.SetTestParameters(32768, 3)

arc.DirectoryCheck()

arc.RunGolden(0)
arc.RunGolden(1)

arc.RunBatchTest(32767)
arc.RunSNRBatch(1000, 50)
arc.RunSNRFIIR(1000, 50)