#!/usr/bin/env python
import numpy as np
import time_eph as te
import os,sys,time,ld
import scipy.signal as ss
import argparse as ap
import matplotlib.pyplot as plt
import scipy.optimize as so
plt.rcParams['font.family']='Serif'
#
version='JigLu_20220317'
#
parser=ap.ArgumentParser(prog='update_tdiff',description='Updating the time difference between local clock and gps.',epilog='Ver '+version)
parser.add_argument('-v','--version', action='version', version=version)
parser.add_argument('--verbose', action="store_true",default=False,help="print detailed information")
parser.add_argument("-w","--rewrite",action="store_true",default=False,help="update the whole clock difference file.")
parser.add_argument('-s','--show',action="store_true",default=False,help='show the handled clock difference.')
args=(parser.parse_args())
#
dtj0=np.array([1537588300,1537623000,1538314761,1541724000,1541916000,1548391000,1550469000,1554826000])
dt0j=np.array([1538314761])
ldt0j=len(dt0j)
def poly(x,*p):
	start=0
	y=np.zeros_like(x,dtype=np.float64)
	y0=0
	for i in np.arange(lseg):
		coeff=p[start:polyn[i]]
		start=polyn[i]
		xj=(x>=dtj[i])&(x<=dtj[i+1])
		px=x[xj]
		if int(dtj[i]) in dt0j:
			y0+=p[polyn[-1]+np.where(dt0j==int(dtj[i]))[0][0]]
		y[xj]=np.polyval(coeff,px-dtj[i])*(px-dtj[i])+y0
		y0=np.polyval(coeff,dtj[i+1]-dtj[i])*(dtj[i+1]-dtj[i])+y0
	return y+p[-1]
#
dirname=os.path.split(os.path.realpath(__file__))[0]
outfile=dirname+'/conventions/local2gps.txt'
if args.rewrite:
	d0=open(outfile,'w+')
	d0.close()
	endunix=0
else:
	d0=np.loadtxt(outfile)
	end0=d0[-1,0]
	if end0>dtj0[-1]: endunix=dtj0[-1]
	else: endunix=end0
	end=te.time(endunix,0,scale='unix',unit=1).unix2local().mjd[0]
	endyr,endmo,endday,_,_=te.mjd2datetime(end)
#
str0='FAST_CLOCKDIFF-'
l0=len(str0)
deltat=50
flist=os.listdir(dirname+'/clock')
t0=[]
dt0=[]
for i in flist:
	if i[:l0]!=str0: continue
	date=i[l0:-4]
	if not args.rewrite:
		if int(date[:4])<endyr: continue
		if len(date)==6:
			if int(date[:4])==endyr and int(date[-2:])<endmo: continue
	tmp=np.genfromtxt(dirname+'/clock/'+i,skip_header=21)
	if not args.rewrite:
		if tmp[-1,0]-endunix<=50: continue
	t0.extend(np.int64(tmp[:,0]))
	dt0.extend(np.float64(tmp[:,1]))
#
if len(t0)<20:	
	print('All present clock data have been included in the clock-difference file.')
	end=True
else:
	dt0=np.array(dt0)[np.argsort(t0)]
	t0=np.sort(t0)
	jt0=(t0>(endunix-deltat))&(dt0>0)
	t1=t0[jt0]
	dt1=dt0[jt0]
	end=False
if end: pass
elif len(t1)<20:
	print('All present clock data have been included in the clock-difference file.')
	end=True
else:
	lt=len(t1)
	if endunix==0: sttunix=t1[0]
	else: sttunix=endunix
	t1t=np.arange(sttunix,t1[-1],10)
	dt1t=np.interp(t1t,t1,dt1)
	ltt=len(t1t)
	t1a=t1t[:int(ltt//150*150)].reshape(-1,150).mean(1)
	dt1a=dt1t[:int(ltt//150*150)].reshape(-1,150)
	jj=(np.max(dt1a,1)-np.min(dt1a,1))<1e-5
	#dt1b=np.polyval(np.polyfit(t1a[jj],dt1a[jj].mean(1),3),t1)
	dt1b=np.interp(t1,t1a[jj],dt1a[jj].mean(1))
	jt1=np.abs(dt1b-dt1)<1e-5
	t2=t1[jt1]
	dt2=dt1[jt1]
if end: pass
elif len(t2)<20:
	print('All present clock data have been included in the clock-difference file.')
	end=True
else:
	if endunix==0: sttunix=t2[0]
	else: sttunix=endunix
	t3=np.arange(sttunix,t2[-1],deltat)
if end: pass
elif len(t3)<20:
	print('All present clock data have been included in the clock-difference file.')
	end=True
else:
	dt3=np.interp(t3,t2,dt2)
	dtj0=np.append(sttunix-1,np.append(dtj0[(dtj0>sttunix)&(dtj0<=t3[-1])],t3[-1]+1))
	dtj=[]
	for i in dtj0:
		if len(dtj)==0:
			value=i
			dtj.append(value)
		else:
			if i-value>5e6:
				lv=np.linspace(value,i,int(np.ceil((i-value)/5e6+1)))[1:]
				dtj.extend(lv)
				value=i
			else:
				value=i
				dtj.append(value)
	lseg=len(dtj)-1
	polyn=(np.ones(lseg,dtype=np.int8)*2).cumsum()
	a=so.curve_fit(poly,t3,dt3,p0=np.zeros(polyn[-1]+ldt0j+1))
	dt3a=poly(t3,*a[0])
	#b,a=ss.butter(15,0.15,'lowpass')
	#dt3a=ss.filtfilt(b,a,dt3)
	if args.rewrite: stt=sttunix
	else: stt=sttunix-(sttunix-end0)%deltat
	t4=np.arange(stt,t3[-1],deltat)
	dt4=poly(t4,*a[0])
	f=open(outfile,'r+')
	if not args.rewrite:
		f0=int(f.read(10))
		nstt=int((stt-f0)/deltat)
		f.seek(nstt*30)
		f.truncate()
	for i in range(len(t4)):
		f.write(str(int(t4[i]))+'     '+'{:.12f}'.format(dt4[i])+'\n')
	f.close()
#
if args.show:
	unix,dt=np.loadtxt(outfile).T
	local=te.time(unix,np.zeros_like(unix),scale='unix',unit=1).unix2local().mjd
	fig=plt.figure(1)
	ax=fig.add_axes([0.13,0.14,0.82,0.81])
	ax.plot(local,dt*1e6)
	ax.set_xlabel('MJD',fontsize=20)
	ax.set_ylabel('Clock Diff. ($\mathrm{\mu}$s)',fontsize=20)
	plt.show()
