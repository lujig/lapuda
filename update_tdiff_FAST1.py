#!/usr/bin/env python
import numpy as np
import time_eph as te
import os,sys,time,ld
import scipy.signal as ss
import argparse as ap
import matplotlib.pyplot as plt
import scipy.optimize as so
import adfunc as af
import fnmatch as fm
plt.rcParams['font.family']='Serif'
#
version='JigLu_20240411'
#
parser=ap.ArgumentParser(prog='update_tdiff',description='Updating the time difference between local clock and gps.',epilog='Ver '+version)
parser.add_argument('-v','--version', action='version', version=version)
parser.add_argument('--verbose', action="store_true",default=False,help="print detailed information")
parser.add_argument("-w","--rewrite",action="store_true",default=False,help="update the whole clock difference file.")
parser.add_argument('-s','--show',nargs='?',default=False,const=True,help='show the handled clock difference.')
args=(parser.parse_args())
#
dirname=os.path.split(os.path.realpath(__file__))[0]
outfile=dirname+'/materials/clock/FAST_new.txt'
flist=os.listdir(dirname+'/materials/clock/FAST_tmp/')
t0,dt0=[],[]
for i in flist.copy():
	if fm.fnmatch(i,'[0-9][0-9]q[0-9].txt'):
		t0tmp,dt0tmp=np.loadtxt(dirname+'/materials/clock/FAST_tmp/'+i).T
		t0.extend(list(t0tmp))
		dt0.extend(list(dt0tmp))
order=np.argsort(t0)
t0=te.time(np.array(t0)[order],np.zeros_like(t0),scale='FAST',unit=86400).local2unix().mjd
dt0=np.array(dt0)[order]
#
def calpoly(lseg0,dtj0,p0,t0s,t0e):
	dy0=af.poly(t0s,lseg0,dtj0,p0)
	sttunix=t0s
	deltat=400
	t1=np.arange(t0s,t0e+deltat,deltat)
	dt1=np.interp(t1,t0,-dt0/1e9)
	t1a=t1[:int(len(t1)//216*216)].reshape(-1,216).mean(1)
	dt1a=dt1[:int(len(dt1)//216*216)].reshape(-1,216).mean(1)
	dtja=np.array([sttunix-1,t1[-1]+1])
	dtj=[]
	deltat=2e6
	for i in dtja:
		if len(dtj)==0:
			value=i
			dtj.append(value)
		else:
			if i-value>deltat:
				lv=np.linspace(value,i,int(np.ceil((i-value)/deltat+1)))[1:]
				dtj.extend(lv)
				value=i
			else:
				value=i
				dtj.append(value)
	lseg=len(dtj)-1
	polyn=(np.ones(lseg,dtype=np.int16)*2).cumsum()
	#
	def poly(x,*p):	# polynomial fit for the clock correction
		start=0
		y=np.zeros_like(x,dtype=np.float64)
		y0=0
		for i in np.arange(lseg):
			coeff=p[start:polyn[i]]
			start=polyn[i]
			xj=(x>=dtj[i])&(x<=dtj[i+1])
			px=x[xj]
			y[xj]=np.polyval(coeff,px-dtj[i])*(px-dtj[i])+y0
			y0=np.polyval(coeff,dtj[i+1]-dtj[i])*(dtj[i+1]-dtj[i])+y0
		return y+p[-1]
	#
	a=so.curve_fit(poly,t1a,dt1a-dy0,p0=np.ones(polyn[-1]))
	dtj2=np.append(dtj0[:-1],dtj)
	lseg2=lseg+lseg0
	p2=np.concatenate((p0[:(lseg0*2)],a[0],p0[(lseg0*2):]))
	return lseg2,dtj2,p2
#
if args.rewrite:
	f=open(dirname+'/materials/clock/FAST_poly.txt')
else:
	f=open(outfile)
cont=f.readlines()
f.close()
lseg=int(cont[0])
dtj=np.float64(cont[1].split())
p=np.float64(cont[2].split())
if args.rewrite:
	p[-1]+=936.9195996478957e-9
	start=t0[0]
else:
	if dtj[-1]-t0[0]<2e7: raise
	start=dtj[-1]-2e7
#
if args.rewrite or dtj[-1]-t0[-1]>0:
	ldtj=(dtj<start).sum()
	dtj=dtj[:(ldtj+1)]
	p=np.append(p[:(ldtj*2)],p[(lseg*2):])
	lseg=ldtj
	nseg=np.ceil((t0[-1]-start)/1e7)
	deltat=(t0[-1]-start)/nseg
	for i in np.arange(nseg):
		t1s=start+deltat*i
		t1e=t1s+deltat
		lseg,dtj,p=calpoly(lseg,dtj,p,t1s,t1e)
	#
	f=open(outfile,'w')
	f.write(str(lseg)+'\n')
	for i in dtj: f.write(str(i)+' ')
	f.write('\n')
	for i in p: f.write(str(i)+' ')
	f.close()
#
else:
	print('The difference file between FAST clock and standard is already the newest.')
#
if args.show:
	fig=plt.figure(1)
	ax=fig.add_axes([0.13,0.14,0.82,0.81])
	if args.show==True:
		t1=np.linspace(dtj[0],dtj[-1],100000)
		dt2=af.poly(t1,lseg,dtj,p)
		ax.plot(te.time(t1,np.zeros_like(t1),scale='unix',unit=1).unix2local('FAST').mjd,dt2*1e6)
		ax.set_ylabel('Clock Diff. ($\mathrm{\mu}$s)',fontsize=20)
	else:
		dt2=af.poly(t0,lseg,dtj,p)
		ax.plot(te.time(t0,np.zeros_like(t0),scale='unix',unit=1).unix2local('FAST').mjd,dt2*1e9+dt0)
		ax.set_ylabel('Clock Diff. (ns)',fontsize=20)
	ax.set_xlabel('MJD',fontsize=20)
	plt.show()
