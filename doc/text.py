import os
import numpy as np
#
dirname=os.path.split(os.path.realpath(__file__))[0]
filename='text.txt'
#
class ldpara_text():
	def __init__(self):
		f=np.loadtxt(open(dirname+'/'+filename, encoding='utf8'),dtype=str,delimiter='\t')
		f=f[f[:,0]=='ldpara']
		for i in f:
			self.__setattr__(str(i[1]),str(i[2]))
