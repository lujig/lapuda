#!/usr/bin/env python
import numpy as np
import time_eph as te
import os,sys,time,ld
import scipy.signal as ss
import argparse as ap
import matplotlib.pyplot as plt
import scipy.optimize as so
import adfunc as af
plt.rcParams['font.family']='Serif'
#
version='JigLu_20240411'
#
parser=ap.ArgumentParser(prog='update_tdiff',description='Updating the time difference between local clock and gps.',epilog='Ver '+version)
parser.add_argument('-v','--version', action='version', version=version)
parser.add_argument('--verbose', action="store_true",default=False,help="print detailed information")
parser.add_argument("-w","--rewrite",action="store_true",default=False,help="update the whole clock difference file.")
parser.add_argument('-s','--show',action="store_true",default=False,help='show the handled clock difference.')
args=(parser.parse_args())
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
dirname=os.path.split(os.path.realpath(__file__))[0]
f=open(dirname+'/materials/clock/FAST_poly.txt')
cont=f.readlines()
f.close()
lseg=int(cont[0])
dtj=np.float64(cont[1].split())
p=np.float64(cont[2].split())
#
readfile=dirname+'/materials/clock/FAST_tmp/dt.txt'
outfile=dirname+'/materials/clock/FAST_new.txt'
#
t0,dt0=np.loadtxt(readfile).T
t0=te.time(t0,np.zeros_like(t0),scale='FAST',unit=86400).local2unix().mjd
print(t0[0],af.poly(t0[0],lseg,dtj,p)*1e9)
sttunix=t0[0]
deltat=400
t1=np.arange(sttunix,t0[-1]+deltat,deltat)
dt1=np.interp(t1,t0,-dt0/1e9)
t1a=t1[:int(len(t1)//216*216)].reshape(-1,216).mean(1)
dt1a=dt1[:int(len(dt1)//216*216)].reshape(-1,216).mean(1)
dtj0=np.array([sttunix-1,t1[-1]+1])
dtj=[]
deltat=2e6
for i in dtj0:
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
polyn=(np.ones(lseg,dtype=np.int8)*2).cumsum()
a=so.curve_fit(poly,t1a,dt1a,p0=np.zeros(polyn[-1]+1))
stt=sttunix
dt2=poly(t1,*a[0])
f=open(outfile,'w')
f.write(str(lseg)+'\n')
for i in dtj: f.write(str(i)+' ')
f.write('\n')
for i in a[0]: f.write(str(i)+' ')
f.close()
#print((dt0+poly(t0,*a[0])).std())
print(t0[0],poly(t0[0],*a[0])*1e9)
#
if args.show:
	fig=plt.figure(1)
	ax=fig.add_axes([0.13,0.14,0.82,0.81])
	ax.plot(te.time(t1,np.zeros_like(t1),scale='unix',unit=1).unix2local('FAST').mjd,dt2*1e6)
	ax.set_xlabel('MJD',fontsize=20)
	ax.set_ylabel('Clock Diff. ($\mathrm{\mu}$s)',fontsize=20)
	plt.show()
