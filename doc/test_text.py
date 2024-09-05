import sys,os
import numpy as np
import warnings as wn
wn.filterwarnings('ignore')
import subprocess as sp
#
dirname=os.path.split(os.path.realpath(__file__))[0]
filename='text.txt'
f=np.loadtxt(open(dirname+'/'+filename, encoding='utf8'),dtype=str,delimiter='\t')
fn=np.unique(f[:,0])
#
for i in fn:
	fi=f[f[:,0]==i][:,1:]
	for fik,fic in fi:
		t=sp.getoutput('grep '+fik+' ../'+i+'.py').split('\n')
		if t=='':print(i,fik,fic)
		for ti in t:
			idx=ti.find(fik)
			if ti[idx+len(fik)].isalpha():
				t.remove(ti)
		for ti in t:
			tc=fic.count('%s')
			if fic.count('{0}'):
				tc+=1
			if fic.count('{1}'):
				tc+=1
			if fic.count('{2}'):
				tc+=1
			if fic.count('{3}'):
				tc+=1
			if tc==0: continue
			ti0=ti[ti.find(fik):]
			if ti0.count('text.'): continue
			idx0=ti0.find('(')
			tmp=ti0[(idx0+1):]
			if tmp.find('(')>=0:
				li=1
				idx1=idx0+1
				while li:
					it0,it1=tmp.find('('),tmp.find(')')
					if it0==-1 or it0>it1:
						tmp=tmp[(it1+1):]
						idx1+=it1+1
						li-=1
					else:
						tmp=tmp[(it0+1):]
						idx1+=it0+1
						li+=1
				if len(ti0[idx0:idx1].split(','))!=tc:
					print(ti0,fic)
			elif len(tmp[:tmp.find(')')].split(','))!=tc:
				print(ti0,fic)
			
			

