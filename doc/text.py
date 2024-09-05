import os
import numpy as np
import warnings as wn
wn.filterwarnings('ignore')
#
dirname=os.path.split(os.path.realpath(__file__))[0]
filename='text.txt'
#
class output_text():
	def __init__(self,prog):
		f=np.loadtxt(open(dirname+'/'+filename, encoding='utf8'),dtype=str,delimiter='\t')
		f=f[f[:,0]==prog]
		for i in f:
			self.__setattr__(str(i[1]),str(i[2]))
