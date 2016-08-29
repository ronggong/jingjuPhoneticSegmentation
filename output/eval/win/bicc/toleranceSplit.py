import numpy as np

fn = "win_bicc_dmfcc_mfccBands_1-5_"
fh = open(fn + '.csv')

t_02,t_04,t_06,t_08,t_10,t_20 = [],[],[],[],[],[]
for line in fh.readlines():
	lineArray 		= line.split(',')
	lineArray[-1] 	= lineArray[-1][:-2]
	lineArray 		= [float(l) for l in lineArray]
	if len(lineArray) <= 4:
		continue
	if float(lineArray[0]) == 0.02:
		t_02.append(lineArray)

	if float(lineArray[0]) == 0.04:
		t_04.append(lineArray)

	if float(lineArray[0]) == 0.06:
		t_06.append(lineArray)

	if float(lineArray[0]) == 0.08:
		t_08.append(lineArray)

	if float(lineArray[0]) == 0.10:
		t_10.append(lineArray)

	if float(lineArray[0]) == 0.20:
		t_20.append(lineArray)

np.savetxt(fn+str(0.02)+'.csv',t_02,fmt='%.3f',delimiter = ',')
np.savetxt(fn+str(0.04)+'.csv',t_04,fmt='%.3f',delimiter = ',')
np.savetxt(fn+str(0.06)+'.csv',t_06,fmt='%.3f',delimiter = ',')
np.savetxt(fn+str(0.08)+'.csv',t_08,fmt='%.3f',delimiter = ',')
np.savetxt(fn+str(0.10)+'.csv',t_10,fmt='%.3f',delimiter = ',')
np.savetxt(fn+str(0.20)+'.csv',t_20,fmt='%.3f',delimiter = ',')



