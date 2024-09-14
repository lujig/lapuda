import os
import numpy as np
import warnings as wn
wn.filterwarnings('ignore')
#
language={	# Chinese or English
'output'   :  'Chinese',
'figure'   :  'Chinese',
'interface':  'Chinese'
}
#
dirname=os.path.split(os.path.realpath(__file__))[0]
filechoose={'Chinese':'text.txt','English':'text_en.txt'}
filename=filechoose[language['output']]
plotname=filechoose[language['figure']]
tkname=filechoose[language['interface']]
#
class output_text():
	def __init__(self,prog):
		f=np.loadtxt(open(dirname+'/'+filename, encoding='utf8'),dtype=str,delimiter='\t')
		f0=f[f[:,0]==prog]
		for i in f0:
			self.__setattr__(str(i[1]),str(i[2]))
		if plotname==filename:
			f1=f[f[:,0]==prog+'_plot']
		else:
			f=np.loadtxt(open(dirname+'/'+plotname, encoding='utf8'),dtype=str,delimiter='\t')
			f1=f[f[:,0]==prog+'_plot']
		for i in f1:
			self.__setattr__(str(i[1]),str(i[2]))
		if prog=='ldtimi':
			if tkname==plotname:
				f1=f[f[:,0]=='ldtimi_tk']
			else:
				f=np.loadtxt(open(dirname+'/'+tkname, encoding='utf8'),dtype=str,delimiter='\t')
				f1=f[f[:,0]=='ldtimi_tk']
			for i in f1:
				self.__setattr__(str(i[1]),str(i[2]))
			
