import os
import csv
import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
import fxpmath as fp


class HardCB:
	def __init__(self, M) -> None:
		self.S = []
		self.plotFolder = 'plots_32bit'
		self.f_clk = 500e6
		self.osr = 1
		self.S_Length = 0
		self.floatType = np.float32
		self.complexType = np.complex64
		self.fxpFormat = 'fxp-s32/16-complex'
		self.N = M
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

	# Filters are 'pre' or 'post'
	def ReadFIRCoefficients(self, folder, filt='none'):
		try:
			if (filt == 0) or (filt == 'none'):
				self.hf = self.ReadCoeffFile(folder + '/FIR_hf')
				self.hb = self.ReadCoeffFile(folder + '/FIR_hb')
			if filt == 'pre':
				self.hf = self.ReadCoeffFile(folder + '/FIR_hf_prefilt')
				self.hb = self.ReadCoeffFile(folder + '/FIR_hb_prefilt')
			if filt == 'post':
				self.hf = self.ReadCoeffFile(folder + '/FIR_hf_postfilt')
				self.hb = self.ReadCoeffFile(folder + '/FIR_hb_postfilt')
			self.hf = np.array(self.hf, self.floatType)
			self.hb = np.array(self.hb, self.floatType)
		except:
			print("Error! Could not find files")
			raise SystemExit

	def Complex32(self, var: complex) -> complex:
		var.real = np.float16(var.real)
		var.imag = np.float16(var.imag)
		return var
		
	def SetFixedBitWidth(self, intBits, fracBits):
		self.qformat = f'fxp-s{intBits+fracBits}/{fracBits}-complex'

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

	# The offline estimator algorithm we measure against
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

	# Offline Parallel estimator
	def GoldenParallel(self):
		Mf = np.zeros((self.N, self.S_Length), complex)
		Mb = np.zeros((self.N, self.S_Length), complex)
		result = np.zeros(self.S_Length, float)
		for i in range(1, self.S_Length):
			for m in range(0, self.N):
				Mf[m, i] = self.Lf[m]*Mf[m, i-1] + np.dot(self.Ff[m,:], self.S[:, i-1])
				Mb[m, self.S_Length-i-1] = self.Lb[m]*Mb[m, self.S_Length-i] + np.dot(self.Fb[m,:], self.S[:, self.S_Length-i-1])
		for i in range(0, self.S_Length):
			tempSum = 0
			for m in range(0, self.N):
				tempSum = tempSum + (self.Wf[m]*Mf[m, i]).real + (self.Wb[m]*Mb[m, i]).real
			result[i] = tempSum
		return result

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
	def BatchIIR(self, SampleSize, OSR = 1):
		DownSize = int(round(SampleSize/OSR))	# length of downsampled memories
		result = np.zeros(int(round(self.S_Length/OSR)), self.floatType)

		# Recursion registers
		M_LH = np.zeros(self.N, self.complexType)		# Lookahead register
		M_B = np.zeros(self.N, self.complexType)		# Backward recursion
		M_F = np.zeros(self.N, self.complexType)		# Forward recursion
		S_DF = np.zeros((self.N,OSR))					# Delayed sample for forward recursion

		# Sample memories
		Reg_Loading = np.zeros((self.N,SampleSize))		# Input cycle
		Reg_Lookahead = np.zeros((self.N,SampleSize))	# Lookahead cycle
		Reg_PreComp = np.zeros((self.N,SampleSize))		# Wait cycle
		Reg_Compute = np.zeros((self.N,SampleSize))		# Compute cycle

		# Part-result storage
		Reg_PartResultB = np.zeros(DownSize, self.floatType)
		Reg_PartResultF = np.zeros(DownSize, self.floatType)
		Reg_ResultB = np.zeros(DownSize, self.floatType)
		Reg_ResultF = np.zeros(DownSize, self.floatType)

		# Batch cycle
		for k in range(0, self.S_Length + 4*SampleSize, SampleSize):
			# Clock cycle i, everything in loop happens in parallel
			for i in range(0, SampleSize):
				j = SampleSize - i - 1
				# Sample loading
				try:
					Reg_Loading[:,i] = self.S[:,k+i]
				except:
					Reg_Loading[:,i] = np.zeros(self.N)
				# Downsample clock
				if (i % OSR != 0):
					continue
				id = int(round(i / OSR))
				jd = int(np.floor(j / OSR))
				# Calculation
				for n in range(0, self.N):
					# Lookahead stage
					temp = 0
					for a in range(0,OSR):
						temp += np.dot(self.Fb[n,:], Reg_Lookahead[:,j-a]) * (self.Lb[n] ** (OSR - 1 - a))
					M_LH[n] = ((self.Lb[n] ** OSR) * M_LH[n] + temp)
					if self.complexType == complex: M_LH = self.Complex32(M_LH) 

					# Forward recursion
					temp = 0
					for a in range(0,OSR):
						temp += np.dot(self.Ff[n,:], S_DF[:,a]) * (self.Lf[n] ** (OSR - 1 - a))
					M_F[n] = ((self.Lf[n] ** OSR) * M_F[n] + temp)
					if self.complexType == complex: M_F = self.Complex32(M_F)
					Reg_PartResultF[id] += (self.Wf[n] * M_F[n]).real

					# Backward Recursion
					temp = 0
					for a in range(0,OSR):
						temp += np.dot(self.Fb[n,:], Reg_Compute[:,j-a]) * (self.Lb[n] ** (OSR - 1 - a))
					M_B[n] = ((self.Lb[n] ** OSR) * M_B[n] + temp)
					if self.complexType == complex: M_B = self.Complex32(M_B)
					Reg_PartResultB[jd] += (self.Wb[n] * M_B[n]).real
				
				# Save forward samples
				for a in range(0,OSR):
					S_DF[:,a] = Reg_Compute[:,i+a]

				# Output stage
				k_delayed = int(round((k - 4*SampleSize) / OSR))
				if ((k_delayed >= 0) and (k_delayed+id < int(round(self.S_Length/OSR)))):
					result[k_delayed+id] = Reg_ResultB[id] + Reg_ResultF[id]
			# Propagate registers (not how it's done in hardware)
			Reg_ResultF = np.array(Reg_PartResultF)
			Reg_ResultB = np.array(Reg_PartResultB)
			Reg_PartResultB = np.zeros(DownSize, self.floatType)
			Reg_PartResultF = np.zeros(DownSize, self.floatType)
			M_B = np.array(M_LH)
			M_LH = np.zeros(self.N, self.complexType)
			Reg_Compute = np.array(Reg_PreComp)
			Reg_PreComp = np.array(Reg_Lookahead)
			Reg_Lookahead = np.array(Reg_Loading)
		return result

	# Online batch architecture with fixed point numbers
	# Work in progress
	def BatchIIRFixed(self, SampleSize, OSR = 1):
		DownSize = int(round(SampleSize/OSR))	# length of downsampled memories
		result = fp.Fxp(np.zeros(int(round(self.S_Length/OSR)), self.floatType), dtype=self.fxpFormat)

		# Constants
		tempLf = np.zeros((self.N, OSR + 1), dtype=complex)
		tempLb = np.zeros((self.N, OSR + 1), dtype=complex)
		for n in range(0, self.N):
			for i in range(0, OSR+1):
				tempLf[n,i] = self.Lf[n] ** (i)
				tempLb[n,i] = self.Lb[n] ** (i)
		Lb = fp.Fxp(tempLb, dtype=self.fxpFormat)
		Lf = fp.Fxp(tempLf, dtype=self.fxpFormat)
		Wb = fp.Fxp(self.Wb, dtype=self.fxpFormat)
		Wf = fp.Fxp(self.Wf, dtype=self.fxpFormat)
		Ff = fp.Fxp(self.Ff, dtype=self.fxpFormat)
		Fb = fp.Fxp(self.Fb, dtype=self.fxpFormat)

		# Recursion registers
		M_LH = fp.Fxp(np.zeros(self.N, self.complexType), dtype=self.fxpFormat)		# Lookahead register
		M_B = fp.Fxp(np.zeros(self.N, self.complexType), dtype=self.fxpFormat)		# Backward recursion
		M_F = fp.Fxp(np.zeros(self.N, self.complexType), dtype=self.fxpFormat)		# Forward recursion
		S_DF = fp.Fxp(np.zeros((self.N,OSR)), dtype=self.fxpFormat)					# Delayed sample for forward recursion

		# Sample memories
		Reg_Loading = np.zeros((self.N,SampleSize))		# Input cycle
		Reg_Lookahead = np.zeros((self.N,SampleSize))	# Lookahead cycle
		Reg_PreComp = np.zeros((self.N,SampleSize))		# Wait cycle
		Reg_Compute = np.zeros((self.N,SampleSize))		# Compute cycle

		# Part-result storage
		Reg_PartResultB = fp.Fxp(np.zeros(DownSize, self.floatType), dtype=self.fxpFormat)
		Reg_PartResultF = fp.Fxp(np.zeros(DownSize, self.floatType), dtype=self.fxpFormat)
		Reg_ResultB = fp.Fxp(np.zeros(DownSize, self.floatType), dtype=self.fxpFormat)
		Reg_ResultF = fp.Fxp(np.zeros(DownSize, self.floatType), dtype=self.fxpFormat)

		# Batch cycle
		for k in range(0, self.S_Length + 4*SampleSize, SampleSize):
			print('k = ' + str(k))
			# Clock cycle i, everything in loop happens in parallel
			for i in range(0, SampleSize):
				j = SampleSize - i - 1
				# Sample loading
				try:
					Reg_Loading[:,i] = self.S[:,k+i]
				except:
					Reg_Loading[:,i] = np.zeros(self.N)
				# Downsample clock
				if (i % OSR != 0):
					continue
				id = int(round(i / OSR))
				jd = int(np.floor(j / OSR))
				# Calculation
				for n in range(0, self.N):
					# Lookahead stage
					temp = fp.Fxp(0.0+0.0j, dtype=self.fxpFormat)
					for a in range(0,OSR):
						tempDot = fp.Fxp(0.0+0.0j, dtype=self.fxpFormat)
						for b in range(0, self.N):
							tempDot += Fb[n,b] * Reg_Lookahead[b,j-a]
						temp += tempDot * Lb[n,OSR-1-a]
					M_LH[n] = Lb[n, OSR] * M_LH[n] + temp

					# Forward recursion
					temp = fp.Fxp(0.0+0.0j, dtype=self.fxpFormat)
					for a in range(0,OSR):
						tempDot = fp.Fxp(0.0+0.0j, dtype=self.fxpFormat)
						for b in range(0, self.N):
							tempDot += Ff[n,b] * S_DF[b,a]
						temp += tempDot * Lf[n,OSR-1-a]
					M_F[n] = Lf[n, OSR] * M_F[n] + temp
					Reg_PartResultF[id] += (Wf[n] * M_F[n]).real

					# Backward Recursion
					temp = fp.Fxp(0.0+0.0j, dtype=self.fxpFormat)
					for a in range(0,OSR):
						tempDot = fp.Fxp(0.0+0.0j, dtype=self.fxpFormat)
						for b in range(0, self.N):
							tempDot += Fb[n,b] * Reg_Compute[b,j-a]
						temp += tempDot * Lb[n,OSR-1-a]
					M_B[n] = Lb[n, OSR] * M_B[n] + temp
					Reg_PartResultB[jd] += (Wb[n] * M_B[n]).real
				
				# Save forward samples
				for a in range(0,OSR):
					S_DF[:,a] = Reg_Compute[:,i+a]

				# Output stage
				k_delayed = int(round((k - 4*SampleSize) / OSR))
				if ((k_delayed >= 0) and (k_delayed+id < int(round(self.S_Length/OSR)))):
					result[k_delayed+id] = Reg_ResultB[id] + Reg_ResultF[id]
			# Propagate registers (not how it's done in hardware)
			Reg_ResultF = Reg_PartResultF
			Reg_ResultB = Reg_PartResultB # fp.Fxp(Reg_PartResultB, dtype=self.fxpFormat)
			M_B = M_LH ##fp.Fxp(M_LH, dtype=self.fxpFormat)
			M_LH = fp.Fxp(np.zeros(self.N, self.complexType), dtype=self.fxpFormat)  
			Reg_Compute = np.array(Reg_PreComp)
			Reg_PreComp = np.array(Reg_Lookahead)
			Reg_Lookahead = np.array(Reg_Loading)
		return result

	# Experimental architecture with reversed sample order for backward recursion
	# Becomes unstable when a rounding error occurs
	def CumulativeIIR(self, lookahead):
		# Constants
		Lb = fp.Fxp(self.Lb, dtype=self.fxpFormat)
		sumLb = fp.Fxp(self.Lb**(lookahead-1), dtype=self.fxpFormat)
		subLb = fp.Fxp(self.Lb, dtype=self.fxpFormat)
		Lf = fp.Fxp(self.Lf, dtype=self.fxpFormat)
		Wb = fp.Fxp(self.Wb, dtype=self.fxpFormat)
		Wf = fp.Fxp(self.Wf, dtype=self.fxpFormat)
		Ff = fp.Fxp(self.Ff, dtype=self.fxpFormat)
		Fb = fp.Fxp(self.Fb, dtype=self.fxpFormat)
		tempLb_inv = np.zeros(self.N, dtype=complex)
		tempSubFb = np.zeros((self.N, self.N), dtype=complex)
		for i in range(0,self.N):
			tempLb_inv[i] = 1/self.Lb[i]
			tempSubFb[i,:] = tempLb_inv[i] * self.Fb[i,:]
		subFb = fp.Fxp(tempSubFb, dtype=self.fxpFormat) 
		Lb_inv = fp.Fxp(tempLb_inv, dtype=self.fxpFormat)		#Inverted reverse lambda coeff

		# Samples
		s_buff = np.zeros((self.N, lookahead))

		# Internal storage
		Mb = fp.Fxp(np.zeros(self.N), dtype=self.fxpFormat)
		Mf = fp.Fxp(np.zeros(self.N), dtype=self.fxpFormat)

		result = fp.Fxp(np.zeros(self.S_Length), dtype=self.fxpFormat)
		
		# Timestep k
		for k in range(0, self.S_Length):
			if k % 100 == 0:
				print('k = ' + str(k))
			# Sample shift register
			s_buff[:, 1:] = s_buff[:, :-1]
			try:
				s_buff[:,0] = self.S[:,k]
			except:
				print('Writing blank sample at time ' + str(k))
				s_buff[:,0] = np.zeros(self.N)
			
			for n in range(0, self.N):
				tempDot = fp.Fxp(0.0+0.0j, dtype=self.fxpFormat)
				for b in range(0, self.N):
					tempDot += Fb[n,b] * s_buff[b,0]
				Mb[n] = (sumLb[n] * tempDot) + (Lb_inv[n] * Mb[n])
				if(lookahead > k):
					continue

				tempDot = fp.Fxp(0.0+0.0j, dtype=self.fxpFormat)
				for b in range(0, self.N):
					tempDot += subFb[n,b] * s_buff[b,lookahead-1]
				Mb[n] -= (Lb_inv[n] * tempDot)

				tempDot = fp.Fxp(0.0+0.0j, dtype=self.fxpFormat)
				for b in range(0, self.N):
					tempDot += Ff[n,b] * s_buff[b,lookahead-1]
				Mf[n] = (Lf[n] * Mf[n] + tempDot)
				result[k-lookahead] += (Wf[n] * Mf[n]).real + (Wb[n] * Mb[n]).real
		return result

	def FIR(self, length, OSR = 1):
		samples = np.zeros([self.N, length*2])
		result = np.zeros(int(round(self.S_Length/OSR)), self.floatType)
		for k in range(0, int(self.S_Length)):
			samples[:, 1:] = samples[:, 0:-1]
			try:
				samples[:, 0] = self.S[:, k]
			except:
				print("Filling empty sample at time " + str(k))
				samples[:, 0] = np.zeros(self.N)
			# Skip calculations when oversampling
			if k % OSR != 0:
				continue
			k_ds = int(k / OSR)
			# Lookahead
			for i in range(0, length):
				result[k_ds] += np.dot(self.hb[i, :], samples[:, length-i-1])
			# Lookback
			for i in range(0, length):
				result[k_ds] += np.dot(self.hf[i, :], samples[:, length+i])
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
		print("Plotting array with length: " + str(arrLength))

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

	def WriteVerilog1D_Fixedpoint (self, f, name, data, bias):
		f.write("\tlocalparam logic signed[63:0] " + name + "[0:%d] = {" %(np.size(data)-1))
		for i in range(0, np.size(data)):
			if (i > 0):
				f.write(", ")
			f.write(str(int(round(data[i] * 2.0**bias))))
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
		f.write("\tlocalparam real " + name + "r[0:%d][0:%d] = '{\n" %((self.N-1),(self.N*exponent-1)))
		for i in range(0, self.N):
			f.write("\t\t'{")
			for j in range(0, self.N*exponent):
				if (j > 0):
					f.write(", ")
				f.write(str(tempData[i][j].real))
			if (i < self.N-1):
				f.write("},\n")
			else:
				f.write("}")
		f.write("};\n\r")
		f.write("\tlocalparam real " + name + "i[0:%d][0:%d] = '{\n" %((self.N-1),(self.N*exponent-1)))
		for i in range(0, self.N):
			f.write("\t\t'{")
			for j in range(0, self.N*exponent):
				if (j > 0):
					f.write(", ")
				f.write(str(tempData[i][j].imag))
			if (i < self.N-1):
				f.write("},\n")
			else:
				f.write("}")
		f.write("};\n\r")

	def WriteVerilog2DExtended_Fixedpoint (self, f, name, data, expData, exponent, bias):
		# Prepare extended data
		tempData = np.zeros((self.N, self.N*exponent), np.complex128)
		for i in range(0, exponent):
			for j in range(0,self.N):
				for k in range(0,self.N):
					tempData[j][i*self.N + k] = data[j][k] * expData[j]**i
		f.write("\tlocalparam logic signed[63:0] " + name + "r[0:%d][0:%d] = '{\n" %((self.N-1),(self.N*exponent-1)))
		for i in range(0, self.N):
			f.write("\t\t'{")
			for j in range(0, self.N*exponent):
				if (j > 0):
					f.write(", ")
				f.write(str(int(round( tempData[i][j].real * 2.0**bias ))))
			if (i < self.N-1):
				f.write("},\n")
			else:
				f.write("}")
		f.write("};\n\r")
		f.write("\tlocalparam logic signed[63:0] " + name + "i[0:%d][0:%d] = '{\n" %((self.N-1),(self.N*exponent-1)))
		for i in range(0, self.N):
			f.write("\t\t'{")
			for j in range(0, self.N*exponent):
				if (j > 0):
					f.write(", ")
				f.write(str(int(round( tempData[i][j].imag * 2.0**bias ))))
			if (i < self.N-1):
				f.write("},\n")
			else:
				f.write("}")
		f.write("};\n\r")

	def WriteVerilogCoefficients(self, fileName, exponent):
		f = open(fileName + '.sv', 'w')
		f.write("`ifndef COEFFICIENTS_SV_\n`define COEFFICIENTS_SV_\r\n")
		f.write("package Coefficients;\r\n")
		f.write("\tlocalparam N = " + str(self.N) + ";\n")
		self.WriteVerilog1D(f, "Lfr", self.Lf.real.flatten())
		self.WriteVerilog1D(f, "Lfi", self.Lf.imag.flatten())
		self.WriteVerilog1D(f, "Lbr", self.Lb.real.flatten())
		self.WriteVerilog1D(f, "Lbi", self.Lb.imag.flatten())
		self.WriteVerilog1D(f, "Wfr", self.Wf.real.flatten())
		self.WriteVerilog1D(f, "Wfi", self.Wf.imag.flatten())
		self.WriteVerilog1D(f, "Wbr", self.Wb.real.flatten())
		self.WriteVerilog1D(f, "Wbi", self.Wb.imag.flatten())
		self.WriteVerilog2DExtended(f, "Ff", self.Ff, self.Lf, exponent)
		self.WriteVerilog2DExtended(f, "Fb", self.Fb, self.Lb, exponent)
		self.WriteVerilog1D(f, "hf", self.hf.flatten())
		self.WriteVerilog1D(f, "hb", self.hb.flatten())
		f.write("\rendpackage\n`endif")
		f.write("\n\r")
		f.close()

	def WriteVerilogCoefficients_Fixedpoint(self, fileName, exponent, bias):
		f = open(fileName + '.sv', 'w')
		f.write("`ifndef COEFFICIENTS_SV_\n`define COEFFICIENTS_SV_\r\n")
		f.write("package Coefficients;\r\n")
		f.write("\tlocalparam N = " + str(self.N) + ";\n")
		f.write("\tlocalparam COEFF_BIAS = " + str(bias) + ";\n")
		self.WriteVerilog1D_Fixedpoint(f, "Lfr", self.Lf.real.flatten(), bias)
		self.WriteVerilog1D_Fixedpoint(f, "Lfi", self.Lf.imag.flatten(), bias)
		self.WriteVerilog1D_Fixedpoint(f, "Lbr", self.Lb.real.flatten(), bias)
		self.WriteVerilog1D_Fixedpoint(f, "Lbi", self.Lb.imag.flatten(), bias)
		self.WriteVerilog1D_Fixedpoint(f, "Wfr", self.Wf.real.flatten(), bias)
		self.WriteVerilog1D_Fixedpoint(f, "Wfi", self.Wf.imag.flatten(), bias)
		self.WriteVerilog1D_Fixedpoint(f, "Wbr", self.Wb.real.flatten(), bias)
		self.WriteVerilog1D_Fixedpoint(f, "Wbi", self.Wb.imag.flatten(), bias)
		self.WriteVerilog2DExtended_Fixedpoint(f, "Ff", self.Ff, self.Lf, exponent, bias)
		self.WriteVerilog2DExtended_Fixedpoint(f, "Fb", self.Fb, self.Lb, exponent, bias)
		self.WriteVerilog1D_Fixedpoint(f, "hf", self.hf.flatten(), bias)
		self.WriteVerilog1D_Fixedpoint(f, "hb", self.hb.flatten(), bias)
		f.write("\rendpackage\n`endif")
		f.write("\n\r")
		f.close()

	def WriteVerilogFIRCoefficients(self, fileName):
		f = open(fileName + '.sv', 'w')
		f.write("`ifndef COEFFICIENTS_SV_\n`define COEFFICIENTS_SV_\r\n")
		f.write("package Coefficients;\r\n")
		f.write("\tlocalparam N = " + str(self.N) + ";\n")
		self.WriteVerilog1D(f, "hf", self.hf.flatten())
		self.WriteVerilog1D(f, "hb", self.hb.flatten())
		f.write("\rendpackage\n`endif")
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
