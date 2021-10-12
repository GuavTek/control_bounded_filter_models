import os
import csv
import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
import fixedpoint as fp

class ComplexFixed(complex):
	def __init__(self, initVal, **qFormat) -> None:
		super().__init__()
		self.real = fp.FixedPoint(initVal.real, **qFormat)
		self.imag = fp.FixedPoint(initVal.imag, **qFormat)


class HardCB:
	def __init__(self) -> None:
		self.S = []
		self.plotFolder = 'plots_32bit'
		self.f_clk = 500e6
		self.osr = 1
		self.S_Length = 0
		self.floatType = np.float32
		self.complexType = np.complex64
		self.qformat = {'signed': True, 'm': 64, 'n': 64, 'overflow': 'wrap', 'rounding': 'down', 'overflow_alert': 'warning'}
		self.N = 3
		pass

	def SetSystemOrder(self, orderN):
		self.N = orderN

	def SetPlotDirectory(self, folder):
		self.plotFolder = folder
		self.DirectoryCheck(folder)

	def DirectoryCheck(self, folder):
		path = os.path.realpath(__file__)
		pathLabel = folder
		if not os.path.isdir(pathLabel):
			print("making directory: " + pathLabel)
			os.mkdir(pathLabel)

	def ReadCoeffFile(self, fileName):
		csvfile = open(fileName + '.csv', newline='')
		r = csv.reader(csvfile, delimiter=',')
		temp = []
		for line in r:
			tempList = []
			for num in line:
				num = num.replace("[", "")
				num = num.replace("]", "")
				tempList.append(complex(num))
			temp.append(tempList)
		temp = np.array(temp, complex)
		csvfile.close()
		return temp

	def ReadLegacyOfflineFile(self, fileName):
		csvfile = open(fileName + '.csv', newline='')
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
					self.Af.append(tempFloat)
				elif(targetVar == 'Ab'):
					self.Ab.append(tempFloat)
				elif(targetVar == 'Bf'):
					self.Bf.append(tempFloat)
				elif(targetVar == 'Bb'):
					self.Bb.append(tempFloat)
				elif(targetVar == 'W'):
					self.W.append(tempFloat)
			except:
				#Save which variable we are reading
				targetVar = line[0]
		#Convert lists to arrays
		self.Af = np.array(self.Af)
		self.Ab = np.array(self.Ab)
		self.Bf = np.array(self.Bf)
		self.Bb = np.array(self.Bb)
		self.W = np.array(self.W)
		self.W = np.resize(self.W, self.N)
		csvfile.close()

	def ReadOfflineFiles(self, folder):
		self.Af = self.ReadCoeffFile(folder + '/Af')
		self.Ab = self.ReadCoeffFile(folder + '/Ab')
		self.Bf = self.ReadCoeffFile(folder + '/Bf')
		self.Bb = self.ReadCoeffFile(folder + '/Bb')
		self.W = self.ReadCoeffFile(folder + '/WT')
		self.W = np.resize(self.W, self.N)

	def ReadStimuliFile(self, fileName):
		csvfile = open(fileName + '.csv', newline='')
		r = csv.reader(csvfile, delimiter=',')
		lineLength = 0
		for line in r:
			tempList = []
			lineLength = np.size(line)
			for num in line:
				num = num.replace('[', '')
				num = num.replace(']', '')
				tempList.append(int(num))
			self.S.append(tempList)
		self.S_Length = lineLength
		self.S = np.array(self.S)
		print("Read " + str(lineLength) + " samples")
		csvfile.close()

	def ReadResultFile(self, fileName, exp):
		csvfile = open(fileName + '.csv', newline='')
		r = csv.reader(csvfile, delimiter=',')
		temp = []
		for line in r:
			for num in line:
				num = num.replace("[", "")
				num = num.replace("]", "")
				try:
					temp.append(float(num)/2**exp)
				except:
					print("!!! Ignored from result: " + num)
		#	while (len(temp) < self.S_Length):
		#		temp.append(0.0)
		temp = np.array(temp)
		print("Read " + str(temp.size) + " samples")
		csvfile.close()
		return temp
	
	#### Load parallel coefficients ####
	def ReadIIRCoefficients(self, folder):
		self.Fb = self.ReadCoeffFile(folder + '/Fb')
		self.Ff = self.ReadCoeffFile(folder + '/Ff')
		self.Lb = self.ReadCoeffFile(folder + '/Lb')
		self.Lf = self.ReadCoeffFile(folder + '/Lf')
		self.Wf = self.ReadCoeffFile(folder + '/Wf')
		self.Wb = self.ReadCoeffFile(folder + '/Wb')

	def ReadFIRCoefficients(self, folder, OSR):
		try:
			self.hf = self.ReadCoeffFile(folder + '/FIR' + str(OSR) + '_hf')
			self.hb = self.ReadCoeffFile(folder + '/FIR' + str(OSR) + '_hb')
			self.hf = np.array(self.hf, self.floatType)
			self.hb = np.array(self.hb, self.floatType)
		except:
			print("Error! Could not find files for given OSR")
			raise SystemExit

	def Complex32(self, var: complex) -> complex:
		var.real = np.float16(var.real)
		var.imag = np.float16(var.imag)
		return var
		
	def SetFixedBitWidth(self, intBits, fracBits):
		self.qformat = {'signed': True, 'm': intBits, 'n': fracBits, 'overflow': 'wrap', 'rounding': 'down', 'overflow_alert': 'warning'}

	def SetFloatBitWidth(self, width):
		if width == 16:
			self.floatType = np.float16
			self.complexType = complex
		elif width == 32:
			self.floatType = np.float32
			self.complexType = np.complex64
		elif width == 64:
			self.floatType = np.float64
			self.complexType = np.complex128

	# Find the largest absolute and relative error
	def GetPeakError(self, arr, ref):
		e = 0
		f = 0
		for i in range(100, self.S_Length-100):
			e = max(e, abs(arr[i] - ref[i]))
			f = max(f, abs((arr[i]-ref[i])/ref[i]))
		return e, f

	# The offline batch estimator algorithm we measure against
	def GoldenBatch(self):
		M = np.zeros((self.N, self.S_Length + 1))
		result = np.zeros(self.S_Length, float)
		#Compute forward recursion
		for k1 in range(1, self.S_Length):
			M[:, k1] = np.dot(self.Af, M[:, k1 - 1]) + np.dot(self.Bf, self.S[:, k1 - 1])
		#Compute backward recursion
		for k2 in range(1, self.S_Length):
			kk2 = self.S_Length - k2
			tempM = np.dot(self.Ab, M[:, kk2]) + np.dot(self.Bb, self.S[:, kk2 - 1])
			result[kk2 - 1] = np.dot(self.W.T, (tempM - M[:, kk2 - 1]))
			M[:, kk2 - 1] = tempM
		return result

	# Offline Parallel batch estimator
	def GoldenParallel(self):
		Mf = np.zeros((self.N, self.S_Length), complex)
		Mb = np.zeros((self.N, self.S_Length), complex)
		result = np.zeros(self.S_Length, float)
		for i in range(1, self.S_Length):
			for m in range(0, self.N):
				Mf[m, i] = self.Lf[m]*Mf[m, i-1] + self.Ff[m,0]*self.S[0, i-1] + self.Ff[m,1]*self.S[1, i-1] + self.Ff[m,2]*self.S[2, i-1]
				Mb[m, self.S_Length-i-1] = self.Lb[m]*Mb[m, self.S_Length-i] + self.Fb[m,0]*self.S[0, self.S_Length-i-1] + self.Fb[m,1]*self.S[1, self.S_Length-i-1] + self.Fb[m,2]*self.S[2, self.S_Length-i-1]
		for i in range(0, self.S_Length):
			tempSum = 0
			for m in range(0, self.N):
				tempSum = tempSum + (self.Wf[m]*Mf[m, i]).real + (self.Wb[m]*Mb[m, i]).real
			result[i] = tempSum

	# FIR IIR hybrid online implementation
	# Baking all factors into LUTs so backward recursion is just summing
	def FIIR(self, lookahead):
		SBuff = np.concatenate(([[0],[0],[0]], self.S[:, 0:lookahead]), axis=1)
		Mf = np.zeros(self.N, self.complexType)
		Lbw = np.zeros((self.N, lookahead), self.complexType)
		result = np.zeros(self.S_Length, float)
		# Pre-calculate constants
		for i in range(0, self.N):
			for j in range(0, lookahead):
				Lbw[i][j] = (self.Wb[i] * self.Lb[i]**j)

		#k is the current clock cycle
		for k in range(0, self.S_Length):
			# Shift sample buffer
			for j in range(0, lookahead):
				SBuff[:, j] = SBuff[:, j+1]
			# Insert latest sample
			try:
				SBuff[:, lookahead] = self.S[:, k+lookahead]
			except:
				SBuff[:, lookahead] = [0,0,0]
			# Compute
			for i in range(0, self.N):
				#Calculate backward recursion
				Mb = self.floatType(0)
				for a in range(0, lookahead):
					Mb = Mb + (Lbw[i][a] * np.dot(self.Fb[i,:], SBuff[:, a+1])).real
				#Calculate forward recursion and product
				Mf[i] = (self.Lf[i]*Mf[i] + np.dot(self.Ff[i,:],SBuff[:, 0]))
				if self.complexType == complex: Mf = self.Complex32(Mf)
				result[k] = result[k] + Mb + (self.Wf[i] * Mf[i]).real
		return result

	# Online batch architecture
	def BatchIIR(self, SampleSize, DownSampFactor = 1):
		DownSize = int(round(SampleSize/DownSampFactor))
		result = np.zeros(int(round(self.S_Length/DownSampFactor)), self.floatType)
		Reg_Loading = np.zeros((self.N,SampleSize), self.floatType)
		Reg_Lookahead = np.zeros((self.N,SampleSize), self.floatType)
		M_LH = np.zeros(self.N, self.complexType)
		Reg_PreComp = np.zeros((self.N,SampleSize), self.floatType)
		Reg_Compute = np.zeros((self.N,SampleSize), self.floatType)
		Mb = np.zeros(self.N, self.complexType)
		Mf = np.zeros(self.N, self.complexType)
		Reg_PartResultB = np.zeros((self.N,DownSize), self.floatType)
		Reg_PartResultF = np.zeros((self.N,DownSize), self.floatType)
		Reg_ResultB = np.zeros((self.N,DownSize), self.floatType)
		Reg_ResultF = np.zeros((self.N,DownSize), self.floatType)
		M_DF = np.zeros(self.N, self.floatType)
		M_DB = np.zeros(self.N, self.floatType)
		# Batch cycle
		for k in range(0, self.S_Length + 4*SampleSize, SampleSize):
			# Clock cycle i, everything in loop happens in parallel
			for i in range(0, SampleSize):
				j = SampleSize - i - 1
				# Sample loading
				try:
					Reg_Loading[:,i] = self.S[:,k+i]
				except:
					Reg_Loading[:,i] = np.array([0, 0, 0])
				# Downsample clock
				if (i % DownSampFactor != 0):
					continue
				id = int(round(i / DownSampFactor))
				jd = int(np.floor(j / DownSampFactor))
				# Calculation
				for n in range(0, self.N):
					# Lookahead stage
					temp = 0
					for a in range(0,DownSampFactor):
						temp += np.dot(self.Fb[n,:], Reg_Lookahead[:,j-a]) * (self.Lb[n] ** (DownSampFactor - 1 - a))
					M_LH[n] = ((self.Lb[n] ** DownSampFactor) * M_LH[n] + temp)
					if self.complexType == complex: M_LH = self.Complex32(M_LH) 
					# Forward recursion
					temp = 0
					for a in range(0,DownSampFactor):
						temp += np.dot(self.Ff[n,:], Reg_Compute[:,i+a]) * (self.Lf[n] ** (DownSampFactor - 1 - a))
					Mf[n] = ((self.Lf[n] ** DownSampFactor) * Mf[n] + temp)
					if self.complexType == complex: Mf = self.Complex32(Mf)
					Reg_PartResultF[n,id] = M_DF[n]
					M_DF[n] = (self.Wf[n] * Mf[n]).real
					# Backward Recursion
					temp = 0
					for a in range(0,DownSampFactor):
						temp += np.dot(self.Fb[n,:], Reg_Compute[:,j-a]) * (self.Lb[n] ** (DownSampFactor - 1 - a))
					Mb[n] = ((self.Lb[n] ** DownSampFactor) * Mb[n] + temp)
					if self.complexType == complex: Mb = self.Complex32(Mb)
					Reg_PartResultB[n,jd] = (self.Wb[n] * Mb[n]).real
					# Output stage
					k_delayed = int(round((k - 4*SampleSize) / DownSampFactor))
					if ((k_delayed >= 0) and (k_delayed+id < int(round(self.S_Length/DownSampFactor)))):
						result[k_delayed+id] = result[k_delayed+id] + Reg_ResultB[n, id] + Reg_ResultF[n, id]
			# Propagate registers
			Reg_ResultF = np.array(Reg_PartResultF)
			Reg_ResultB = np.array(Reg_PartResultB)
			Mb = np.array(M_LH)
			M_LH = np.zeros(self.N, self.complexType)
			Reg_Compute = np.array(Reg_PreComp)
			Reg_PreComp = np.array(Reg_Lookahead)
			Reg_Lookahead = np.array(Reg_Loading)
		return result

	# Online batch architecture with fixed point numbers
	def BatchIIRFixed(self, SampleSize):
		result = np.zeros(self.S_Length, fp.FixedPoint(0, **self.qformat))
		Reg_Loading = np.zeros((self.N,SampleSize), fp.FixedPoint(0, **self.qformat))
		Reg_Lookahead = np.zeros((self.N,SampleSize), fp.FixedPoint(0, **self.qformat))
		M_LH = np.zeros(self.N, self.complexType)
		Reg_PreComp = np.zeros((self.N,SampleSize), fp.FixedPoint(0, **self.qformat))
		Reg_Compute = np.zeros((self.N,SampleSize), fp.FixedPoint(0, **self.qformat))
		Mb = np.zeros(self.N, self.complexType)
		Mf = np.zeros(self.N, self.complexType)
		Reg_PartResultB = np.zeros((self.N,SampleSize), fp.FixedPoint(0, **self.qformat))
		Reg_PartResultF = np.zeros((self.N,SampleSize), fp.FixedPoint(0, **self.qformat))
		Reg_ResultB = np.zeros((self.N,SampleSize), fp.FixedPoint(0, **self.qformat))
		Reg_ResultF = np.zeros((self.N,SampleSize), fp.FixedPoint(0, **self.qformat))
		M_DF = np.zeros(self.N, fp.FixedPoint(0, **self.qformat))
		M_DB = np.zeros(self.N, fp.FixedPoint(0, **self.qformat))
		# Batch cycle
		for k in range(0, self.S_Length + 4*SampleSize, SampleSize):
			# Clock cycle i, everything in loop happens in parallel
			for i in range(0, SampleSize):
				j = SampleSize - i - 1
				# Sample loading
				try:
					Reg_Loading[:,i] = self.S[:,k+i]
				except:
					Reg_Loading[:,i] = np.array([0, 0, 0])
				# Calculation
				for n in range(0, self.N):
					# Lookahead stage
					M_LH[n] = (self.Lb[n] * M_LH[n] + np.dot(self.Fb[n,:], Reg_Lookahead[:,j]))
					if self.complexType == complex: M_LH = self.Complex32(M_LH) 
					# Forward recursion
					Mf[n] = (self.Lf[n] * Mf[n] + np.dot(self.Ff[n,:], Reg_Compute[:,i]))
					if self.complexType == complex: Mf = self.Complex32(Mf)
					Reg_PartResultF[n,i] = M_DF[n]
					M_DF[n] = (self.Wf[n] * Mf[n]).real
					# Backward Recursion
					Mb[n] = (self.Lb[n] * Mb[n] + np.dot(self.Fb[n,:], Reg_Compute[:,j]))
					if self.complexType == complex: Mb = self.Complex32(Mb)
					Reg_PartResultB[n,j] = (self.Wb[n] * Mb[n]).real
					# Output stage
					k_delayed = k - 4*SampleSize
					if ((k_delayed >= 0) and (k_delayed+i < self.S_Length)):
						result[k_delayed+i] = result[k_delayed+i] + Reg_ResultB[n, i] + Reg_ResultF[n, i]
			# Propagate registers
			Reg_ResultF = np.array(Reg_PartResultF)
			Reg_ResultB = np.array(Reg_PartResultB)
			Mb = np.array(M_LH)
			M_LH = np.zeros(self.N, self.complexType)
			Reg_Compute = np.array(Reg_PreComp)
			Reg_PreComp = np.array(Reg_Lookahead)
			Reg_Lookahead = np.array(Reg_Loading)
		return result

	# Experimental architecture with reversed sample order for backward recursion
	# Becomes unstable when a rounding error occurs
	def CumulativeIIR(self, lookahead):
		s_buff = np.zeros((self.N, lookahead))
		Mb = np.zeros(self.N, ComplexFixed(0, **self.qformat))
		Mf = np.zeros(self.N, ComplexFixed(0, **self.qformat))
		result = np.zeros(self.S_Length, fp.FixedPoint(0, **self.qformat))
		Lb_inv = (np.array([1/self.Lb[0], 1/self.Lb[1], 1/self.Lb[2]]))		#Inverted reverse lambda coeff
		for k in range(0, self.S_Length):
			# Sample shift register
			s_buff[:, 1:lookahead-1] = s_buff[:, 0:lookahead-2]
			try:
				s_buff[:,0] = self.S[:,k]
			except:
				s_buff[:,0] = np.zeros(self.N)
			
			for n in range(0, self.N):
				Mb[n] = (self.Lb[n]**(lookahead-1) * np.dot(self.Fb[n,:], s_buff[:,0])) + (Lb_inv[n] * Mb[n])
				if(lookahead > k):
					continue
				Mb[n] = Mb[n] - (Lb_inv[n] * np.dot(self.Fb[n,:], s_buff[:,lookahead-1]))
				Mf[n] = (self.Lf[n] * Mf[n] + np.dot(self.Ff[n,:], s_buff[:,lookahead-1]))
				result[k-lookahead] = result[k-lookahead] + (self.Wf[n] * Mf[n]).real + (self.Wb[n] * Mb[n]).real
		return result

	def FIR(self, length, OSR = 1):
		samples = np.zeros([self.N, length*2])
		result = np.zeros(int(round(self.S_Length/OSR)), self.floatType)
		for k in range(0, int(round(self.S_Length/OSR))):
			samples[:, 1:2*length-1] = samples[:, 0:2*length-2]
			try:
				samples[:, 0] = self.S[:, k*OSR]
			except:
				print("Filling empty sample at time " + str(k))
				samples[:, 0] = np.zeros(3)
			# Lookahead
			for i in range(0, length):
				result[k] += np.dot(self.hb[i, :], samples[:, length-i-1])
			# Lookback
			for i in range(0, length):
				result[k] += np.dot(self.hf[i, :], samples[:, length+i])
		return result

	# Plot a section of the wave
	def PlotWave(self, arr, length, tit):
		k = np.arange(0, length)
		plt.title(tit)
		plt.xlabel("time k")
		plt.ylabel("result u")
		plt.plot(k, arr[:length])

	# Plot the PSD and return SNR
	def PlotPSD(self, arr, freq, sig_leak=1):
		T = 1.0 / freq
		arrLength = arr.size
		endSlice = int(round(arrLength / 16))
		remSlice = arrLength - 2 * endSlice
		print("array length: " + str(arrLength))
		print("end slice: " + str(endSlice))
		print("rem slice: " + str(remSlice))

		arr_f, freq = plt.psd(arr, NFFT=arrLength, Fs=freq)
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

	# Makes a figure of wave arr, and returns SNR
	def PlotFigure(self, arr, plotLength, label, fileName):
		plt.figure(figsize=(10, 8))
		plt.subplot(2,1,1)
		self.PlotWave(arr, plotLength, label)

		plt.subplot(2,1,2)
		SNR = self.PlotPSD(arr, self.f_clk / self.osr, 1)
		plt.figtext(0.13, 0.42, "SNR = " + ('%.2f' % SNR) + "dB")
		plt.savefig((self.plotFolder + "/" + fileName))
		plt.close()
		return SNR

	def PlotSNR(self, ticks, SNRs, tit, paramX, fileName):
		# Display every 4th tick
		xdisp = []
		for i in range(0, ticks.size):
			if (i % 4 == 0):
				xdisp.append(str(ticks[i]))
			else:
				xdisp.append("")
		# Plot figure
		peak = np.amax(SNRs)
		where = np.where(SNRs == peak)
		plt.figure(figsize=(10,8))
		plt.plot(ticks, SNRs)
		plt.title(tit)
		plt.ylabel("SNR [dB]")
		plt.xlabel(paramX)
		plt.minorticks_off()
		plt.xticks(ticks, xdisp, rotation=45)
		plt.figtext(0.13, 0.85, "Peak is " + ('%.2f' % peak) + "dB with " + paramX + " = " + str(int(ticks[where])))
		plt.grid(True)
		plt.savefig(self.plotFolder + "/" + fileName)

	def WriteCSVFile (self, fileName, data):
		# Convert arrays to list
		try:
			listData = data.tolist()
		except:
			listData = data
		CSVfile = open(fileName + '.csv', 'w', newline='')
		fileWriter = csv.writer(CSVfile, delimiter=';')
		fileWriter.writerows(map(lambda x: [x], listData))
		CSVfile.close()

	#Convert test-signals to hardware levels
	def GetHardwareStimuli(self):
		hardSig = []
		for line in self.S:
			temp = []
			temp = [0 if item == -1 else item for item in line]
			hardSig.append(temp)
		return hardSig

	# Write a file which is friendly to the verilog testbench
	def WriteVerilogStimuli(self, fileName):
		hardSig = self.GetHardwareStimuli()
		f = open(fileName + '.csv', 'w', newline='')
		for i in range(0, self.S_Length):
			binString = ''
			try:
				test = hardSig[0][i]
			except:
				break
			for j in range(0, self.N):
				binString = str(hardSig[j][i]) + binString
			binString = binString + ",\r"
			f.write(binString)
		f.close()

	def CalculateIIRCoefficients(self):
		#Get eigenvectors and eigenvalues
		self.Lf, Qf = la.eig(self.Af)
		self.Lb, Qb = la.eig(self.Ab)

		#Inverted eigenvectors
		Qif = np.linalg.inv(Qf)
		Qib = np.linalg.inv(Qb)

		#Vectorized functions
		self.Ff = np.dot(Qif, self.Bf)
		self.Fb = np.dot(Qib, self.Bb)

		self.Wf = np.zeros(3)
		self.Wb = np.zeros(3)

		#Final coefficients
		self.Wf = -np.dot(Qf.T, self.W)
		self.Wb = np.dot(Qb.T, self.W)

	def ReadLegacyStimuliFile(self, fileName):
		#### Load Test-data ####
		csvfile = open(fileName + '.csv', newline='')
		test = csv.reader(csvfile, delimiter=';')
		self.S = []
		for line in test:
			if(line == []):
				continue
			temp = []
			for num in line:
				x = num.split(',')
				for y in x:
					if (y == '') or (y == '0'):
						continue
					temp.append(int(y))
			self.S.append(temp)
		self.S_Length = np.size(temp)
		self.S = np.array(self.S)
		csvfile.close()

	def PrintLUTValues(self):
		s = np.zeros(self.N)
		for n in range(0, self.N):
			for i in range(0, 2**self.N):
				# Make input vector
				ii = i
				for j in range(0, self.N):
					k = self.N - j - 1
					if ii >= 2**k:
						s[k] = 1
						ii -= 2**k
					else:
						s[k] = -1
				print("Input vector = " + str(s))
				print("LUT Fb #" + str(i) + " n=" + str(n) + " \t" + str(np.dot(self.Fb[n,:], s)))
				print("LUT Ff #" + str(i) + " n=" + str(n) + " \t" + str(np.dot(self.Ff[n,:], s)))

	def WriteVerilog1D (self, f, name, data):
		f.write("\tlocalparam real " + name + "[0:%d] = {" %(np.size(data)-1))
		for i in range(0, np.size(data)):
			if (i > 0):
				f.write(", ")
			f.write(str(data[i]))
		f.write("};\n\r")

	def WriteVerilog2D (self, f, name, data):
		f.write("\tlocalparam real " + name + "r[0:%d][0:%d] = {\n" %((self.N-1),(self.N-1)))
		for i in range(0, self.N):
			f.write("\t\t{")
			for j in range(0, self.N):
				if (j > 0):
					f.write(", ")
				f.write(str(data[i][j].real))
			if (i < self.N-1):
				f.write("},\n")
			else:
				f.write("}")
		f.write("};\n\r")
		f.write("\tlocalparam real " + name + "i[0:%d][0:%d] = {\n" %((self.N-1),(self.N-1)))
		for i in range(0, self.N):
			f.write("\t\t{")
			for j in range(0, self.N):
				if (j > 0):
					f.write(", ")
				f.write(str(data[i][j].imag))
			if (i < self.N-1):
				f.write("},\n")
			else:
				f.write("}")
		f.write("};\n\r")

	def WriteVerilog2DExtended (self, f, name, data, expData, exponent):
		# Prepare extended data
		tempData = np.zeros((self.N, self.N*exponent), np.complex128)
		for i in range(0, exponent):
			for j in range(0,self.N):
				for k in range(0,self.N):
					tempData[j][i*self.N + k] = data[j][k] * expData[j]**i
		f.write("localparam real " + name + "r[0:%d][0:%d] = '{\n" %((self.N-1),(self.N*exponent-1)))
		for i in range(0, self.N):
			f.write("\t'{")
			for j in range(0, self.N*exponent):
				if (j > 0):
					f.write(", ")
				f.write(str(tempData[i][j].real))
			if (i < self.N-1):
				f.write("},\n")
			else:
				f.write("}")
		f.write("};\n\r")
		f.write("localparam real " + name + "i[0:%d][0:%d] = '{\n" %((self.N-1),(self.N*exponent-1)))
		for i in range(0, self.N):
			f.write("\t'{")
			for j in range(0, self.N*exponent):
				if (j > 0):
					f.write(", ")
				f.write(str(tempData[i][j].imag))
			if (i < self.N-1):
				f.write("},\n")
			else:
				f.write("}")
		f.write("};\n\r")

	def WriteVerilogCoefficients(self, fileName, exponent):
		f = open(fileName + '.sv', 'w')
		f.write("package Coefficients;\r\n")
		self.WriteVerilog1D(f, "Lfr", self.Lf.real)
		self.WriteVerilog1D(f, "Lfi", self.Lf.imag)
		self.WriteVerilog1D(f, "Lbr", self.Lb.real)
		self.WriteVerilog1D(f, "Lbi", self.Lb.imag)
		self.WriteVerilog1D(f, "Wfr", self.Wf.real)
		self.WriteVerilog1D(f, "Wfi", self.Wf.imag)
		self.WriteVerilog1D(f, "Wbr", self.Wb.real)
		self.WriteVerilog1D(f, "Wbi", self.Wb.imag)
		self.WriteVerilog2DExtended(f, "Ff", self.Ff, self.Lf, exponent)
		self.WriteVerilog2DExtended(f, "Fb", self.Fb, self.Lb, exponent)
		f.write("\rendpackage")
		f.write("\n\r")
		f.close()

	def WriteCPPCoefficients (self, fileName, length):
		f = open(fileName + ".h", 'w')
		f.write('#include "FloatType.h"\n\n')
		self.WriteCPP1D(f, "Lf", self.Lf)
		self.WriteCPP1D(f, "Lb", self.Lb)
		self.WriteCPP1D(f, "Wf", self.Wf)
		self.WriteCPP1D(f, "Wb", self.Wb)
		self.WriteCPP2D(f, "Ff", self.Ff)
		self.WriteCPP2D(f, "Fb", self.Fb)
		Lbw = np.zeros((self.N, length), complex)
		# Pre-calculate constants
		for i in range(0, self.N):
			for j in range(0, length):
				Lbw[i][j] = (self.Wb[i] * self.Lb[i]**j)
		
		f.write("const floatType Lbwr[%d][%d] = {\n" %(self.N,length))
		for i in range(0, self.N):
			f.write("\t{")
			for j in range(0, length):
				if (j > 0):
					f.write(", ")
				f.write(str(Lbw[i][j].real))
			if (i < self.N-1):
				f.write("},\n")
			else:
				f.write("}")
		f.write("};\n\r")
		f.write("const floatType Lbwi[%d][%d] = {\n" %(self.N,length))
		for i in range(0, self.N):
			f.write("\t{")
			for j in range(0, length):
				if (j > 0):
					f.write(", ")
				f.write(str(Lbw[i][j].imag))
			if (i < self.N-1):
				f.write("},\n")
			else:
				f.write("}")
		f.write("};\n\r")
		f.close()

	def WriteCPP1D (self, f, name, data):
		f.write("const floatType " + name + "r[%d] = {" %self.N)
		for i in range(0, self.N):
			if (i > 0):
				f.write(", ")
			f.write(str(data[i].real))
		f.write("};\n\r")
		f.write("const floatType " + name + "i[%d] = {" %self.N)
		for i in range(0, self.N):
			if (i > 0):
				f.write(", ")
			f.write(str(data[i].imag))
		f.write("};\n\r")

	def WriteCPP2D (self, f, name, data):
		f.write("const floatType " + name + "r[%d][%d] = {\n" %(self.N,self.N))
		for i in range(0, self.N):
			f.write("\t{")
			for j in range(0, self.N):
				if (j > 0):
					f.write(", ")
				f.write(str(data[i][j].real))
			if (i < self.N-1):
				f.write("},\n")
			else:
				f.write("}")
		f.write("};\n\r")
		f.write("const floatType " + name + "i[%d][%d] = {\n" %(self.N,self.N))
		for i in range(0, self.N):
			f.write("\t{")
			for j in range(0, self.N):
				if (j > 0):
					f.write(", ")
				f.write(str(data[i][j].imag))
			if (i < self.N-1):
				f.write("},\n")
			else:
				f.write("}")
		f.write("};\n\r")
