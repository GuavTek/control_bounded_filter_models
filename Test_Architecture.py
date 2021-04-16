import csv
import numpy as np
import matplotlib as mpl
import os
from matplotlib import pyplot as plt
from scipy import signal

typeLabel = '32bit'
floatType = np.float32
complexType = np.complex64
# For 16bit set complexType = complex

# Options for SNR graph
plotTop = 512
plotStep = 16

signalLength = 32768
N = 3
f_clk = 504e6
fs = 5e6

Af = []
Ab = []
Bf = []
Bb = []
W = []
S = []

def SetBitWidth(width):
	global typeLabel
	global floatType
	global complexType
	if width == 16:
		typeLabel = '16bit'
		floatType = np.float16
		complexType = complex
	elif width == 32:
		typeLabel = '32bit'
		floatType = np.float32
		complexType = np.complex64
	elif width == 64:
		typeLabel = '64bit'
		floatType = np.float64
		complexType = np.complex128

def SetPlotParameters(top, step):
	global plotTop
	global plotStep
	plotTop = top
	plotStep = step

def SetTestParameters(stimuli_length, system_order):
	global signalLength
	global N
	signalLength = stimuli_length
	N = system_order

def Complex32(var: complex) -> complex:
	var.real = np.float16(var.real)
	var.imag = np.float16(var.imag)
	return var

def ReadFile(fileName):
	csvfile = open(fileName, newline='')
	r = csv.reader(csvfile, delimiter=',')
	temp = []
	for line in r:
		tempList = []
		for num in line:
			num = num.replace("[", "")
			num = num.replace("]", "")
			tempList.append(complex(num))
		temp.append(tempList)
	temp = np.array(temp, complexType)
	if complexType == complex: temp = Complex32(temp)
	csvfile.close()
	return temp

def ReadResults(fileName, exp):
	csvfile = open(fileName, newline='')
	r = csv.reader(csvfile, delimiter=',')
	temp = []
	for line in r:
		for num in line:
			num = num.replace("[", "")
			num = num.replace("]", "")
			temp.append(float(num)/2**exp)
	while (len(temp) < signalLength):
		temp.append(0.0)
	temp = np.array(temp)
	csvfile.close()
	return temp

#### Load offline matrices ####
with open('data/offline_matrices.csv', newline='') as csvfile:
	test = csv.reader(csvfile, delimiter=';')
	targetVar = "0"
	for line in test:
		if(line == []):
			continue
		try:
			tempFloat = []
			for num in line:
				tempFloat.append(float(num))
			if(targetVar == 'Af'):
				Af.append(tempFloat)
			elif(targetVar == 'Ab'):
				Ab.append(tempFloat)
			elif(targetVar == 'Bf'):
				Bf.append(tempFloat)
			elif(targetVar == 'Bb'):
				Bb.append(tempFloat)
			elif(targetVar == 'W'):
				W.append(tempFloat)
		except:
			#Save which variable we are reading
			targetVar = line[0]
	#Convert lists to arrays
	Af = np.array(Af)
	Ab = np.array(Ab)
	Bf = np.array(Bf)
	Bb = np.array(Bb)
	W = np.array(W)
	csvfile.close()

#### Load control signals ####
with open('data/clean_signals.csv', newline='') as csvfile:
	r = csv.reader(csvfile, delimiter=',')
	for line in r:
		tempList = []
		for num in line:
			num = num.replace('[', '')
			num = num.replace(']', '')
			tempList.append(int(num))
		S.append(tempList)
	S = np.array(S)
	csvfile.close()

#### Load parallel coefficients ####
Fb = ReadFile('data/Fb.csv')
Ff = ReadFile('data/Ff.csv')
Lb = ReadFile('data/Lb.csv')
Lf = ReadFile('data/Lf.csv')
Wf = ReadFile('data/Wf.csv')
Wb = ReadFile('data/Wb.csv')

# The offline batch estimator algorithm we measure against
def GoldenBatch():
	M = np.zeros((N, signalLength + 1))
	#Compute forward recursion
	for k1 in range(1, signalLength):
		M[:, k1] = np.dot(Af, M[:, k1 - 1]) + np.dot(Bf, S[:, k1 - 1])
	#Compute backward recursion
	for k2 in range(1, signalLength):
		kk2 = signalLength - k2
		tempM = np.dot(Ab, M[:, kk2]) + np.dot(Bb, S[:, kk2 - 1])
		u_gold[kk2 - 1] = np.dot(W.T, (tempM - M[:, kk2 - 1]))
		M[:, kk2 - 1] = tempM

# Offline Parallel batch estimator
def GoldenParallel():
	Mf = np.zeros((N, signalLength), complex)
	Mb = np.zeros((N, signalLength), complex)
	for i in range(1, signalLength):
		for m in range(0, N):
			Mf[m, i] = Lf[m]*Mf[m, i-1] + Ff[m,0]*S[0, i-1] + Ff[m,1]*S[1, i-1] + Ff[m,2]*S[2, i-1]
			Mb[m, signalLength-i-1] = Lb[m]*Mb[m, signalLength-i] + Fb[m,0]*S[0, signalLength-i-1] + Fb[m,1]*S[1, signalLength-i-1] + Fb[m,2]*S[2, signalLength-i-1]
	for i in range(0, signalLength):
		tempSum = 0
		for m in range(0, N):
			tempSum = tempSum + (Wf[m]*Mf[m, i]).real + (Wb[m]*Mb[m, i]).real
		u_gold[i] = tempSum

# FIIR online implementation
# Baking all factors into LUTs so backward recursion is just summing
def TestFIIR(lookahead):
	SBuff = np.concatenate(([[0],[0],[0]], S[:, 0:lookahead]), axis=1)
	Mf = np.zeros(N, complexType)
	Lbw = np.zeros((N, lookahead), complexType)
	# Pre-calculate constants
	for i in range(0, N):
		for j in range(0, lookahead):
			Lbw[i][j] = (Wb[i] * Lb[i]**j)

	#k is the current clock cycle
	for k in range(0, signalLength):
		# Shift sample buffer
		for j in range(0, lookahead):
			SBuff[:, j] = SBuff[:, j+1]
		# Insert latest sample
		try:
			SBuff[:, lookahead] = S[:, k+lookahead]
		except:
			SBuff[:, lookahead] = [0,0,0]
		# Compute
		for i in range(0, N):
			#Calculate backward recursion
			Mb = floatType(0)
			for a in range(0, lookahead):
				Mb = Mb + (Lbw[i][a] * np.dot(Fb[i,:], SBuff[:, a+1])).real
			#Calculate forward recursion and product
			Mf[i] = (Lf[i]*Mf[i] + np.dot(Ff[i,:],SBuff[:, 0]))
			if complexType == complex: Mf = Complex32(Mf)
			u_par[k] = u_par[k] + Mb + (Wf[i] * Mf[i]).real

# Online batch architecture
def TestBatch(SampleSize):
	tempRes = np.zeros(signalLength, floatType)
	Reg_Loading = np.zeros((N,SampleSize), floatType)
	Reg_Lookahead = np.zeros((N,SampleSize), floatType)
	M_LH = np.zeros(N, complexType)
	Reg_PreComp = np.zeros((N,SampleSize), floatType)
	Reg_Compute = np.zeros((N,SampleSize), floatType)
	Mb = np.zeros(N, complexType)
	Mf = np.zeros(N, complexType)
	Reg_PartResultB = np.zeros((N,SampleSize), floatType)
	Reg_PartResultF = np.zeros((N,SampleSize), floatType)
	Reg_ResultB = np.zeros((N,SampleSize), floatType)
	Reg_ResultF = np.zeros((N,SampleSize), floatType)
	M_DF = np.zeros(N, floatType)
	M_DB = np.zeros(N, floatType)
	# Batch cycle
	for k in range(0, signalLength + 4*SampleSize, SampleSize):
		# Clock cycle i, everything in loop happens in parallel
		for i in range(0, SampleSize):
			j = SampleSize - i - 1
			# Sample loading
			try:
				Reg_Loading[:,i] = S[:,k+i]
			except:
				Reg_Loading[:,i] = np.array([0, 0, 0])
			# Calculation
			for n in range(0, N):
				# Lookahead stage
				M_LH[n] = (Lb[n] * M_LH[n] + np.dot(Fb[n,:], Reg_Lookahead[:,j]))
				if complexType == complex: M_LH = Complex32(M_LH) 
				# Forward recursion
				Mf[n] = (Lf[n] * Mf[n] + np.dot(Ff[n,:], Reg_Compute[:,i]))
				if complexType == complex: Mf = Complex32(Mf)
				Reg_PartResultF[n,i] = M_DF[n]
				M_DF[n] = (Wf[n] * Mf[n]).real
				# Backward Recursion
				Mb[n] = (Lb[n] * Mb[n] + np.dot(Fb[n,:], Reg_Compute[:,j]))
				if complexType == complex: Mb = Complex32(Mb)
				Reg_PartResultB[n,j] = (Wb[n] * Mb[n]).real
				# Output stage
				k_delayed = k - 4*SampleSize
				if ((k_delayed >= 0) and (k_delayed+i < signalLength)):
					tempRes[k_delayed+i] = tempRes[k_delayed+i] + Reg_ResultB[n, i] + Reg_ResultF[n, i]
		# Propagate registers
		Reg_ResultF = np.array(Reg_PartResultF)
		Reg_ResultB = np.array(Reg_PartResultB)
		Mb = np.array(M_LH)
		M_LH = np.zeros(N, complexType)
		Reg_Compute = np.array(Reg_PreComp)
		Reg_PreComp = np.array(Reg_Lookahead)
		Reg_Lookahead = np.array(Reg_Loading)
	return tempRes

# Experimental architecture with reversed sample order for backward recursion
# Becomes unstable when a rounding error occurs
# Which is very common for float numbers
def TestReverse(testLength):
	M1 = np.complex64(0)			#Algo from batch code
	M2 = np.complex128(0)			#Algo being tested
	Lb_inv = (np.array([1/Lb[0], 1/Lb[1], 1/Lb[2]]))		#Inverted reverse lambda coeff
	for i in range(0, testLength):
		M1 = 0
		for a in range(0,Lookahead):
			if (i-a < 0):
				break
			M1 = Lb[0] * M1 + np.dot(Fb[0,:], S[:,i-a])
		M2 = (Lb_inv[0] * M2) + (Lb[0]**(Lookahead-1) * np.dot(Fb[0,:], S[:,i]))
		if (i-Lookahead >= 0):
			M2 = M2 - (Lb_inv[0] * np.dot(Fb[0,:], S[:,i-Lookahead]))
		print("i = " + str(i))
		print(M1)
		print(M2)
		if ((abs(M1 - M2) > 0.0000001) and (i > Lookahead)):
			break

# Single pole IIR, fc = 100MHz
def IIR_100(arr):
	temp = arr
	m = np.float16(0)
	a0 = np.float16(0.712537755)
	b1 = np.float16(0.28746224)
	for i in range(0, len(arr)):
		temp[i] = a0 * arr[i] + b1 * m
		m = temp[i]
	return temp

def IIR_50(arr):
	temp = arr
	m = np.float16(0)
	a0 = np.float16(0.46384494)
	b1 = np.float16(0.53615505)
	for i in range(0, len(arr)):
		temp[i] = a0 * arr[i] + b1 * m
		m = temp[i]
	return temp

def IIR_20(arr):
	temp = arr
	m = np.float16(0)
	a0 = np.float16(0.22067939)
	b1 = np.float16(0.77932061)
	for i in range(0, len(arr)):
		temp[i] = a0 * arr[i] + b1 * m
		m = temp[i]
	return temp

# Biquad filter, fc = 10MHz
def BIQ_10(arr):
	temp = np.zeros(len(arr), np.float32)
	m0 = np.float32(0)
	m1 = np.float32(0)
	m2 = np.float32(0)
	a0 = np.float32(0.0035667865634507064)
	a1 = np.float32(0.007133573126901413)
	a2 = np.float32(0.0035667865634507064)
	b1 = np.float32(1.824094670479931)
	b2 = np.float32(-0.8383618167337338)
	for i in range(0, len(arr)):
		m0 = arr[i]
		m0 = m0 + b1 * m1
		m0 = m0 + b2 * m2
		temp[i] = a0 * m0
		temp[i] = temp[i] + a1 * m1
		temp[i] = temp[i] + a2 * m2
		m2 = m1
		m1 = m0
	return temp

def BIQ_50(arr):
	temp = np.zeros(len(arr), np.float32)
	m0 = np.float32(0)
	m1 = np.float32(0)
	m2 = np.float32(0)
	a0 = np.float32(0.06655775035418053)
	a1 = np.float32(0.13311550070836106)
	a2 = np.float32(0.06655775035418053)
	b1 = np.float32(1.1494245244651997)
	b2 = np.float32(-0.41565552588192206)
	for i in range(0, len(arr)):
		m0 = arr[i]
		m0 = m0 + b1 * m1
		m0 = m0 + b2 * m2
		temp[i] = a0 * m0
		temp[i] = temp[i] + a1 * m1
		temp[i] = temp[i] + a2 * m2
		m2 = m1
		m1 = m0
	return temp

def BIQ_20(arr):
	temp = np.zeros(len(arr), np.float32)
	m0 = np.float32(0)
	m1 = np.float32(0)
	m2 = np.float32(0)
	a0 = np.float32(0.013164365887905859)
	a1 = np.float32(0.026328731775811718)
	a2 = np.float32(0.013164365887905859)
	b1 = np.float32(1.6502158334595687)
	b2 = np.float32(-0.7028732970111922)
	for i in range(0, len(arr)):
		m0 = arr[i]
		m0 = m0 + b1 * m1
		m0 = m0 + b2 * m2
		temp[i] = a0 * m0
		temp[i] = temp[i] + a1 * m1
		temp[i] = temp[i] + a2 * m2
		m2 = m1
		m1 = m0
	return temp

# Find the largest absolute and relative error
def GetPeakError(arr):
	e = 0
	f = 0
	for i in range(100, signalLength-100):
		e = max(e, abs(arr[i] - u_gold[i]))
		f = max(f, abs((arr[i]-u_gold[i])/u_gold[i]))
	return e, f

# Plot a section of the wave
def PlotWave(arr, length, tit):
	k = np.arange(0, length)
	plt.title(tit)
	plt.xlabel("time k")
	plt.ylabel("result u")
	plt.plot(k, arr[:length])

# Plot the PSD and return SNR
def PlotPSD(arr, tit, sig_leak=0):
	T = 1.0 / f_clk
	
	arr_f, freq = plt.psd(arr[4096:-4096], NFFT=signalLength-8192, Fs=f_clk)
	plt.xscale('log')
	plt.grid(True)

	#Find signal position
	sigpos = max(range(len(arr_f)), key=lambda i: abs(arr_f[i]))

	#Normalize
	#arr_f = arr_f/arr_f[sigpos]

	#Calculate signal power
	Ps = 0
	for i in range(sigpos-sig_leak, sigpos+sig_leak+1):
		Ps = Ps + arr_f[i]
		arr_f[i] = 0
	
	#Calculate noise power
	Pn = sum(arr_f)

	SNR = 10*np.log10(Ps/Pn)

	return SNR

# Run online batch test and generate figure
def RunBatchTest(buff):
	u_bat = TestBatch(buff)
	#u_bat = BIQ_10(u_bat)
	plt.figure(figsize=(10, 8))
	plt.subplot(2,1,1)
	PlotWave(u_bat, 1536, "Waveform with buffer size " + str(buff))

	plt.subplot(2,1,2)
	SNR = PlotPSD(u_bat, "PSD for buffer size " + str(buff), 1)
	plt.figtext(0.13, 0.42, "SNR = " + ('%.2f' % SNR) + "dB")
	plt.savefig(("plots_" + typeLabel + "/BatchPlot_" + str(buff)))
	plt.close()
	return SNR

# Run online FIIR test and generate figure
def RunFIIRTest(lookahead):
	global u_par
	u_par = np.zeros(signalLength, floatType)
	TestFIIR(lookahead)
	#u_par = BIQ_10(u_par)
	plt.figure(figsize=(10, 8))
	plt.subplot(2,1,1)
	PlotWave(u_par, 1536, "Waveform with lookahead length: " + str(lookahead))

	plt.subplot(2,1,2)
	SNR = PlotPSD(u_par, "PSD for lookahead length: " + str(lookahead), 1)
	plt.figtext(0.13, 0.42, "SNR = " + ('%.2f' % SNR) + "dB")
	plt.savefig(("plots_" + typeLabel + "/FIIRPlot_" + str(lookahead)))
	plt.close()
	return SNR

def RunGolden(mode):
	global u_gold
	u_gold = np.zeros(signalLength)

	# Run test
	if mode == 1:
		GoldenBatch()
		# Plot figure
		plt.figure(figsize=(10,8))
		plt.subplot(2,1,1)
		PlotWave(u_gold, 1536, "Regular offline algorithm")
		plt.subplot(2,1,2)
		SNR = PlotPSD(u_gold, "PSD", 1)
		plt.figtext(0.13, 0.42, "SNR = " + ('%.2f' % SNR) + "dB")
		plt.savefig("plots_64bit/GoldenRegular")
	else:
		GoldenParallel()
		plt.figure(figsize=(10,8))
		plt.subplot(2,1,1)
		PlotWave(u_gold, 1536, "Offline Parallel algorithm")
		plt.subplot(2,1,2)
		SNR = PlotPSD(u_gold, "PSD", 1)
		plt.figtext(0.13, 0.42, "SNR = " + ('%.2f' % SNR) + "dB")
		plt.savefig("plots_" + typeLabel + "/GoldenParallel")

def RunSNRBatch(top, step):
	SNR_Batch = []
	# Simulate
	x = np.arange(step, top+1, step)
	for langth in x:
		print("Testing batch with parameter " + str(langth) + "...")
		SNR_Batch.append(RunBatchTest(langth))
	# Display every 4th tick
	xdisp = []
	for i in range(0, x.size):
		if (i % 4 == 0):
			xdisp.append(str(x[i]))
		else:
			xdisp.append("")
	# Plot figure
	peak = np.amax(SNR_Batch)
	where = np.where(SNR_Batch == peak)
	plt.figure(figsize=(10,8))
	plt.plot(x, SNR_Batch)
	plt.title("SNR for batch architecture - " + typeLabel)
	plt.xlabel("Buffer size")
	plt.ylabel("SNR - dB")
	plt.minorticks_off()
	plt.xticks(x, xdisp, rotation=45)
	plt.figtext(0.13, 0.85, "Peak is " + ('%.2f' % peak) + "dB with buffer size = " + str(int(x[where])))
	plt.grid(True)
	plt.savefig("plots_" + typeLabel + "/SNR_Batch_" + typeLabel)

def RunSNRFIIR(top, step):
	SNR_FIIR = []
	# Run simulations
	x = np.arange(step, top+1, step)
	for langth in x:
		print("Testing FIIR with lookahead " + str(langth) + "...")
		SNR_FIIR.append(RunFIIRTest(langth))
	# Display every 4th tick
	xdisp = []
	for i in range(0, x.size):
		if (i % 4 == 0):
			xdisp.append(str(x[i]))
		else:
			xdisp.append("")
	# Plot figure
	peak = np.amax(SNR_FIIR)
	where = np.where(SNR_FIIR == peak)
	plt.figure(figsize=(10,8))
	plt.plot(x, SNR_FIIR)
	plt.title("SNR for FIIR achitecture - " + typeLabel)
	plt.xlabel("Lookahead length")
	plt.ylabel("SNR - dB")
	plt.minorticks_off()
	plt.xticks(x, xdisp, rotation=45)
	plt.figtext(0.13, 0.85, "Peak is " + ('%.2f' % peak) + "dB with lookahead length = " + str(int(x[where])))
	plt.grid(True)
	plt.savefig("plots_" + typeLabel + "/SNR_FIIR_" + typeLabel)

def DirectoryCheck():
	path = os.path.realpath(__file__)
	pathLabel = "plots_" + typeLabel
	path64 = "plots_64bit"
	if not os.path.isdir(pathLabel):
		print("making directory: " + pathLabel)
		os.mkdir(pathLabel)
	if not os.path.isdir(path64):
		print("making directory: " + path64)
		os.mkdir(path64)

