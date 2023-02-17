#!/usr/bin/env python
import numpy as np
import numpy.fft as fft
import numpy.polynomial.chebyshev as nc
import scipy.optimize as so
import scipy.signal as ss
import argparse as ap
import os,ld
import subprocess as sp
import time
#
version='JigLu_20220711'
#
parser=ap.ArgumentParser(prog='ldaz',description='Analyzing the interfered channels in frequency domain of LD file.',epilog='Ver '+version)
parser.add_argument('-v','--version', action='version', version=version)
parser.add_argument('--verbose', action="store_true",default=False,help="print detailed information")
parser.add_argument("filename",nargs='+',help="name of file or filelist")
parser.add_argument('-j','--crit',dest='crit',default=0.1,type=np.float64,help="the criterion to screen the interfered channels on calibration parameters")
parser.add_argument("-p","--polynum",dest="polynum",default=7,type=int,help="numbers of Chebyshev polynomial coefficients on calibration parameters")
parser.add_argument("-f","--channel",dest='chanrange',default="1,650",help="channel range without interference in form start_chan,end_chan")
parser.add_argument('-m',action='store_true',default=False,dest='modify',help='modify the zap parameter of the LD file')
parser.add_argument('-c',action='store_true',default=False,dest='correct',help='correct the data of the LD file')
args=(parser.parse_args())
command=['ldaz.py']
#
if args.verbose:
	sys.stdout.write('Analyzing the arguments...\n')
filelist=args.filename
filenum=len(filelist)
#
def ld_check(fname,notfirst=True):
	fnametmp='___'+fname
	if fnametmp[-3:]!='.ld':
		parser.error('File '+fname+' is not LD format.')
	if not os.path.isfile(fname):
		parser.error('Ld files name '+fname+' is invalid.')
	try:
		f=ld.ld(filelist[i])
		finfo=f.read_info()
	except:
		parser.error('Ld files '+fname+' is invalid.')
	global nchan
	if not ((finfo['mode']=='single')|(finfo['mode']=='subint')):
		parser.error('Ld files '+fname+' is not pulsar data file.')
	if not notfirst:
		nchan=np.int16(finfo['nchan'])
	else:
		if nchan!=np.int16(finfo['nchan']):
			parser.error("Ld files have different channel numbers.")
#
for i in np.arange(filenum):
	ld_check(filelist[i],notfirst=i)
#
command.append('-f '+args.chanrange)
chanstart,chanend=np.int16(args.chanrange.split(','))
if chanstart>chanend:
	parser.error("Starting channel number larger than ending channel number.")
elif chanstart<0 or chanend>nchan:
	parser.error("Input channel range is overrange.")
#
command.append('-j '+str(args.crit))
command.append('-p '+str(args.polynum))
if args.modify:
	command.append('-m')
if args.correct:
	command.append('-c')
command=' '.join(command)
#
def func(x0,y0,c0):
	c=[0.0015304279396914935,-5.142979752233712]
	def lineres(p):
		k,b=p
		y1=(k*x0+b-y0)%(2*np.pi)
		y1[y1>np.pi]-=np.pi*2
		return y1
	#
	a=so.leastsq(lineres,x0=c,full_output=1)
	k,b=a[0]
	y1=k*x0+b
	y0[(y0-y1)>np.pi]-=2*np.pi
	y0[(y0-y1)<-np.pi]+=2*np.pi
	jj=np.arange(len(x0))[np.abs(y0-y1)<c0]
	return k,b,jj
#
def calzap(aa,bb,cr,ci):
	aa=aa/aa[chanstart:chanend].mean()
	bb=bb/bb[chanstart:chanend].mean()
	b,a=ss.butter(10,0.1,'lowpass')
	aa0,bb0=ss.filtfilt(b,a,aa),ss.filtfilt(b,a,bb)
	da=aa-aa0
	da/=da.std()
	db=bb-bb0
	db/=db.std()
	jj=np.zeros_like(da,dtype=np.bool)
	x0=np.arange(nchan)
	j1a=np.zeros(nchan,dtype=np.bool)
	j1t=np.ones(nchan,dtype=np.bool)
	loop=0
	while not np.all(j1t==j1a):
		j1t=j1a.copy()
		j0a=np.abs(da)<=3
		while not np.all(j0a|j1a):
			j1a=np.abs(da)>3
			da/=da[j0a].std()
			j0a=np.abs(da)<=3
			loop+=1
			if loop>1000: break
		if loop>1000: return np.zeros(nchan,dtype=np.bool)
		xtmp=x0[j0a]
		aa1=np.interp(x0,xtmp,aa[j0a])
		aa0=ss.filtfilt(b,a,aa1)
		da=aa1-aa0
		da/=da.std()
		j1a=j1a|j1t
	j0a=np.logical_not(j1a)
	x2=x0[j0a]
	aa2=aa[j0a]
	y0=nc.chebval(x2,nc.chebfit(x2,aa2,args.polynum))
	dy=aa2-y0
	j0a=np.abs(dy)<=0.1
	j2a=np.zeros(len(dy),dtype=np.bool)
	while not np.all(j0a|j2a):
		j2a=np.abs(dy)>0.1
		xtmp=x2[j0a]
		y0=nc.chebval(x2,nc.chebfit(xtmp,aa2[j0a],args.polynum))
		dy=aa2-y0
		j0a=np.abs(dy)<=0.1
	j01a=np.logical_not(j1a)
	j02a=np.logical_not(j2a)
	#
	j1b=np.zeros(nchan,dtype=np.bool)
	j1t=np.ones(nchan,dtype=np.bool)
	while not np.all(j1t==j1b):
		j1t=j1b.copy()
		j0b=np.abs(db)<=3
		while not np.all(j0b|j1b):
			j1b=np.abs(db)>3
			db/=db[j0b].std()
			j0b=np.abs(db)<=3
		xtmp=x0[j0b]
		bb1=np.interp(x0,xtmp,bb[j0b])
		bb0=ss.filtfilt(b,a,bb1)
		db=bb1-bb0
		db/=db.std()
		j1b=j1b|j1t
	j0b=np.logical_not(j1b)
	x2=x0[j0b]
	bb2=bb[j0b]
	y0=nc.chebval(x2,nc.chebfit(x2,bb2,args.polynum))
	dy=bb2-y0
	j0b=np.abs(dy)<=args.crit
	j2b=np.zeros(len(dy),dtype=np.bool)
	while not np.all(j0b|j2b):
		j2b=np.abs(dy)>args.crit
		xtmp=x2[j0b]
		y0=nc.chebval(x2,nc.chebfit(xtmp,bb2[j0b],args.polynum))
		dy=bb2-y0
		j0b=np.abs(dy)<=args.crit
	j01b=np.logical_not(j1b)
	j02b=np.logical_not(j2b)
	#
	y0=np.arctan2(ci,cr)
	k,b0,jj=func(x0,y0,args.crit*4)
	k,b0,jj1=func(x0[jj],y0[jj],args.crit*2)
	jj=jj[jj1]
	k,b0,jj1=func(x0[jj],y0[jj],args.crit)
	jj=jj[jj1]
	y1=k*x0+b0
	#
	tmp=np.zeros(nchan,dtype=np.bool)
	tmp[jj]=True
	tmp1a=np.arange(nchan,dtype=np.int32)
	tmp1a=tmp1a[j01a][j02a]
	j0a=tmp1a.copy()
	tmp2a=np.zeros(nchan,dtype=np.bool)
	tmp2a[j0a]=True
	tmp1b=np.arange(nchan,dtype=np.int32)
	tmp1b=tmp1b[j01b][j02b]
	j0b=tmp1b.copy()
	tmp2b=np.zeros(nchan,dtype=np.bool)
	tmp2b[j0b]=True
	jj=tmp&tmp2a&tmp2b	
	return jj
#
for i in filelist:
	dfile=ld.ld(i)
	d=dfile.period_scrunch()[:,:,0]
	info=dfile.read_info()
	nchan,nperiod,nbin0,npol=dfile.read_shape()
	cal=info['cal'][-4:].reshape(4,-1)
	aa,bb,cr,ci=cal
	jc=calzap(aa,bb,cr,ci)&(d.mean(1)>0)
	if jc.sum()==0:
		print(i)
		continue
	d=d[jc]
	f=fft.rfft(d,axis=1)
	ff=np.abs(f*f.conj())
	nbin=np.int16(ff.shape[1]/2)
	r2=ff[:,nbin:].sum(1)
	rs=np.argsort(r2)
	f0=f[rs].cumsum(0)
	ff=np.abs(f0*f0.conj())
	a=ff[:,1:nbin].sum(1)/ff[:,nbin:].sum(1)
	j1=r2<=r2[rs][(np.argmax(a))]
	tmp=np.ones(nchan,dtype=np.bool)
	tmp[np.arange(nchan,dtype=np.int32)[jc][j1]]=False
	zchan0=np.arange(nchan,dtype=np.int32)[tmp]
	if 'zchan' in info.keys():
		zchan1=info['zchan']
	else:
		zchan1=np.int32([])
	zchan=list(set(zchan0).union(zchan1))
	if args.modify or args.correct:
		info['zchan']=list(zchan)
		if 'history' in info.keys():
			info['history'].append(command)
			info['file_time'].append(time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime()))
		else:
			info['history']=[command]
			info['file_time']=[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())]
		if args.correct:
			for k in zchan:
				dfile.write_chan(np.zeros([nperiod,nbin0,npol]),k)
		dfile.write_info(info)
	else:
		np.savetxt(i[:-3]+'.txt',zchan,fmt='%i')
