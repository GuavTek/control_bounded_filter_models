import csv
import numpy as np
import scipy.linalg as la

WriteFiles = 0
WriteHead = 1
SignalLength = 32768
N = 3

Af = []
Ab = []
Bf = []
Bb = []
W = []

def WriteFile (name, data):
	CSVfile = open('data/' + name + '.csv', 'w', newline='')
	fileWriter = csv.writer(CSVfile, delimiter=';')
	fileWriter.writerows(map(lambda x: [x], data))
	CSVfile.close()

# Write a file which is friendly to the verilog testbench
def WriteHarderSig(name, data):
	f = open('data/' + name + '.csv', 'w', newline='')
	for i in range(0, SignalLength):
		binString = ''
		try:
			test = data[0][i]
		except:
			break
		for j in range(0, N):
			binString = str(data[j][i]) + binString
		binString = binString + ",\r"
		f.write(binString)
	f.close()

def WriteVerilog():
	global f
	f = open("data/Coefficients.sv", 'w')
	f.write("package Coefficients\r\n")
	WriteVerilog1D("Lf", Lf)
	WriteVerilog1D("Lb", Lb)
	WriteVerilog1D("Wf", Wf)
	WriteVerilog1D("Wb", Wb)
	WriteVerilog2DExtended("Ff", Ff, Lf, 20)
	WriteVerilog2DExtended("Fb", Fb, Lb, 20)
	f.write("\rendpackage")
	f.write("\n\r")
	f.close()

def WriteVerilog1D (name, data):
	f.write("\tlocalparam real " + name + "r[%d:0] = {" %(N-1))
	for i in range(0, N):
		if (i > 0):
			f.write(", ")
		f.write(str(data[i].real))
	f.write("};\n\r")
	f.write("\tlocalparam real " + name + "i[%d:0] = {" %(N-1))
	for i in range(0, N):
		if (i > 0):
			f.write(", ")
		f.write(str(data[i].imag))
	f.write("};\n\r")

def WriteVerilog2D (name, data):
	f.write("\tlocalparam real " + name + "r[%d:0][%d:0] = {\n" %((N-1),(N-1)))
	for i in range(0, N):
		f.write("\t\t{")
		for j in range(0, N):
			if (j > 0):
				f.write(", ")
			f.write(str(data[i][j].real))
		if (i < N-1):
			f.write("},\n")
		else:
			f.write("}")
	f.write("};\n\r")
	f.write("\tlocalparam real " + name + "i[%d:0][%d:0] = {\n" %((N-1),(N-1)))
	for i in range(0, N):
		f.write("\t\t{")
		for j in range(0, N):
			if (j > 0):
				f.write(", ")
			f.write(str(data[i][j].imag))
		if (i < N-1):
			f.write("},\n")
		else:
			f.write("}")
	f.write("};\n\r")

def WriteVerilog2DExtended (name, data, expData, exponent):
	# Prepare extended data
	tempData = np.zeros((N, N*exponent), np.complex128)
	for i in range(0, exponent):
		for j in range(0,N):
			for k in range(0,N):
				tempData[j][i*N + k] = data[j][k] * expData[j]**i
	f.write("const real " + name + "r[%d:0][%d:0] = '{\n" %((N-1),(N*exponent-1)))
	for i in range(0, N):
		f.write("\t'{")
		for j in range(0, N*exponent):
			if (j > 0):
				f.write(", ")
			f.write(str(tempData[i][j].real))
		if (i < N-1):
			f.write("},\n")
		else:
			f.write("}")
	f.write("};\n\r")
	f.write("const real " + name + "i[%d:0][%d:0] = '{\n" %((N-1),(N*exponent-1)))
	for i in range(0, N):
		f.write("\t'{")
		for j in range(0, N*exponent):
			if (j > 0):
				f.write(", ")
			f.write(str(tempData[i][j].imag))
		if (i < N-1):
			f.write("},\n")
		else:
			f.write("}")
	f.write("};\n\r")


def WriteHeader (length):
	global f
	f = open("data/Coefficients.h", 'w')
	f.write('#include "FloatType.h"\n\n')
	Write1D("Lf", Lf)
	Write1D("Lb", Lb)
	Write1D("Wf", Wf)
	Write1D("Wb", Wb)
	Write2D("Ff", Ff)
	Write2D("Fb", Fb)
	Lbw = np.zeros((N, length), complex)
	# Pre-calculate constants
	for i in range(0, N):
		for j in range(0, length):
			Lbw[i][j] = (Wb[i] * Lb[i]**j)
	
	f.write("const floatType Lbwr[%d][%d] = {\n" %(N,length))
	for i in range(0, N):
		f.write("\t{")
		for j in range(0, length):
			if (j > 0):
				f.write(", ")
			f.write(str(Lbw[i][j].real))
		if (i < N-1):
			f.write("},\n")
		else:
			f.write("}")
	f.write("};\n\r")
	f.write("const floatType Lbwi[%d][%d] = {\n" %(N,length))
	for i in range(0, N):
		f.write("\t{")
		for j in range(0, length):
			if (j > 0):
				f.write(", ")
			f.write(str(Lbw[i][j].imag))
		if (i < N-1):
			f.write("},\n")
		else:
			f.write("}")
	f.write("};\n\r")

	f.close()

def Write1D (name, data):
	f.write("const floatType " + name + "r[%d] = {" %N)
	for i in range(0, N):
		if (i > 0):
			f.write(", ")
		f.write(str(data[i].real))
	f.write("};\n\r")
	f.write("const floatType " + name + "i[%d] = {" %N)
	for i in range(0, N):
		if (i > 0):
			f.write(", ")
		f.write(str(data[i].imag))
	f.write("};\n\r")

def Write2D (name, data):
	f.write("const floatType " + name + "r[%d][%d] = {\n" %(N,N))
	for i in range(0, N):
		f.write("\t{")
		for j in range(0, N):
			if (j > 0):
				f.write(", ")
			f.write(str(data[i][j].real))
		if (i < N-1):
			f.write("},\n")
		else:
			f.write("}")
	f.write("};\n\r")
	f.write("const floatType " + name + "i[%d][%d] = {\n" %(N,N))
	for i in range(0, N):
		f.write("\t{")
		for j in range(0, N):
			if (j > 0):
				f.write(", ")
			f.write(str(data[i][j].imag))
		if (i < N-1):
			f.write("},\n")
		else:
			f.write("}")
	f.write("};\n\r")

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
			#print(tempFloat)
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
			#print("Target is: " + str(line[0]))
	csvfile.close()

sig = []
#### Load Test-data ####
with open('data/control_signals.csv', newline='') as csvfile:
	test = csv.reader(csvfile, delimiter=';')
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
		sig.append(temp)
	csvfile.close()

#Convert test-signals to hardware levels
hardSig = []
for line in sig:
	temp = []
	temp = [0 if item == -1 else item for item in line]
	hardSig.append(temp)

#Convert lists to arrays
Af = np.array(Af)
Ab = np.array(Ab)
Bf = np.array(Bf)
Bb = np.array(Bb)
#Make W a 1D array
W = np.array(W)
W = W.reshape(3)

#Get eigenvectors and eigenvalues
Lf, Qf = la.eig(Af)
Lb, Qb = la.eig(Ab)

#Inverted eigenvectors
Qif = np.linalg.inv(Qf)
Qib = np.linalg.inv(Qb)

#Vectorized functions
Ff = np.dot(Qif, Bf)
Fb = np.dot(Qib, Bb)

Wf = np.zeros(3)
Wb = np.zeros(3)

#Final coefficients
Wf = -np.dot(Qf.T, W)
Wb = np.dot(Qb.T, W)

#Convert to lists
Lf = Lf.tolist()
Lb = Lb.tolist()
Ff = Ff.tolist()
Fb = Fb.tolist()
Wf = Wf.tolist()
Wb = Wb.tolist()


#Write CSV files
if (WriteFiles):
	WriteFile('Lf', Lf)
	WriteFile('Lb', Lb)
	WriteFile('Ff', Ff)
	WriteFile('Fb', Fb)
	WriteFile('Wf', Wf)
	WriteFile('Wb', Wb)
	WriteFile('hardware_signals', hardSig)
	WriteHarderSig('verilog_signals', hardSig)
	WriteFile('clean_signals', sig)

if (WriteHead):
	WriteHeader(256)
	WriteVerilog()
