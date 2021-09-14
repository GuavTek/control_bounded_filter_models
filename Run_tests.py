import Test_Architecture as arc

arc.SetBitWidth(32)
arc.SetTestParameters(32768, 3)

arc.DirectoryCheck()

#arc.RunGolden(0)
#arc.RunGolden(1)

arc.RunBatchTest(32640, 1)
arc.RunSNRBatch(1000, 48, 1)

arc.RunBatchTest(32640, 2)
arc.RunSNRBatch(1000, 48, 2)

arc.RunBatchTest(32640, 4)
arc.RunSNRBatch(1000, 48, 4)

arc.RunBatchTest(32640, 8)
arc.RunSNRBatch(1000, 48, 8)


arc.SetTestParameters(32760, 3)
arc.RunBatchTest(32640, 12)
arc.RunSNRBatch(1000, 48, 12)
