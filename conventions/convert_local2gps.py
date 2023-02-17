import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
#
a0,a1=np.loadtxt('fast-hmaser-gps-clock-diff.txt').T
def screen(a0,a1):
	lena=len(a0)
	d0=a0[1:]-a0[:-1]
	d1=a1[1:]-a1[:-1]
	dd=d1/d0
	jj=np.abs(dd)>5e-7
	#
	#
	j0=np.arange(lena-1,dtype=np.int32)[jj]
	j1=np.array(list(set(np.append(j0,j0+1))))
	setj1=set(j1)
	j2=[]
	for s in j1:
		if s<500:
			m1=np.arange(s,s+1000,dtype=np.int32)
		elif (lena-s)<510:
			m1=np.arange(s-1000,s,dtype=np.int32)
		else:
			m1=np.arange(s-500,s+500,dtype=np.int32)
		a1m=a1[m1]
		j2.extend(m1[np.abs(a1m-np.sort(a1m)[250:-250].mean())>5e-6])
	j2=np.sort(list(set(j2)))
	j3=np.ones(lena,dtype=np.bool)
	j3[j2]=False
	return a0[j3],a1[j3]
b0,b1=screen(a0,a1)
#plt.figure(1)
#plt.clf()
#plt.plot(a0,a1)
#plt.plot(b0,b1)
#plt.ylim(-0.5e-5,4.5e-5)
#
dt0=10
dt1=86400
c0=np.arange(b0[0],b0[-1]+dt0,dt0,dtype=np.int64)
c1=np.interp(c0,b0,b1)
ld=int(len(c0)//(dt1/dt0))
lc=int(ld*(dt1/dt0))
d0=c0[int(dt1/dt0/2):lc:int(dt1/dt0)]
d1=c1[:lc].reshape(ld,-1).mean(1)
a=open('local2gps.txt','w')
for i in range(ld):
	a.write(str(d0[i])+'     '+'{:.12f}'.format(d1[i])+'\n')
a.close()













