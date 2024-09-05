#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import time_eph as te
import psr_read as pr
import psr_model as pm
import os,sys
import scipy.optimize as so
import warnings as wn
import argparse as ap
import ld
wn.filterwarnings('ignore')
dirname=os.path.split(os.path.realpath(__file__))[0]
sys.path.append(dirname+'/doc')
import text
#
text=text.output_text('ldtim')
version='JigLu_20220633'
parser=ap.ArgumentParser(prog='ldtim',description=text.help,epilog='Ver '+version,add_help=False,formatter_class=lambda prog: ap.RawTextHelpFormatter(prog, max_help_position=50))
parser.add_argument('-h', '--help', action='help', default=ap.SUPPRESS,help=text.help_h)
parser.add_argument('-v','--version',action='version',version=version,help=text.help_v)
parser.add_argument("filename",help=text.help_filename)
parser.add_argument('-p','--par',dest='par',help=text.help_p)
parser.add_argument('--fit',dest='fit',default='f0',help=text.help_fit)
parser.add_argument('-i','--illu',default='prepost',dest='illu',help=text.help_i)
parser.add_argument('--time',action='store_true',default=False,dest='time',help=text.help_time)
parser.add_argument('-s','--save',default='',dest='save',nargs='*',help=text.help_s)
parser.add_argument('-x',dest='x',default='mjd',help=text.help_x)
parser.add_argument('-m','--merge',default=0,nargs='?',type=np.float64,dest='merge',help=text.help_m)
parser.add_argument('-e','--err',dest='err',type=np.float64,default=1,help=text.help_e)
parser.add_argument('-c','--crit',dest='crit',type=np.float64,default=1e6,help=text.help_c)
parser.add_argument('-z','--zero',dest='zero',default=0,type=np.float64,help=text.help_z)
parser.add_argument('-d','--date',dest='date',default="",help=text.help_d)
args=(parser.parse_args())
#
if not os.path.isfile(args.filename):
	parser.error(text.error_ntfn)
d=ld.ld(args.filename)
data=d.read_chan(0)[:,:,0]
info=d.read_info()
if args.par:
	psr=pr.psr(args.par,parfile=True)
else:
	psr=pr.psr(info['pulsar_info']['psr_name'])
date,sec,dterr=data[:,0],data[:,1],data[:,4]
if args.date:
	date0=args.date.split(',')
	try:
		if len(date0)==1:
			date0=[0,np.float64(date0[0])]
		elif len(date0)>=2:
			date0=np.float64(date0)
		if len(date0)==2:
			jj=(date>date0[0])&(date<date0[1])
		elif len(date0)>2:
			jj=np.zeros_like(date,dtype=bool)
			for i in np.arange(len(date0)):
				jj=jj^(date>date0[i])
	except:
		parser.error(text.error_ndr)
else:
	jj=np.ones_like(date,dtype=bool)
if args.merge==0:
	jj=jj&(dterr<args.err)
#
if type(args.save) is list:
	nsave=len(args.save)
	if nsave>2:
		parser.error(text.error_nsfn)
	if nsave==0:
		name='psr'
		name0=name
		tmp=1
		while os.path.isfile(name0+'.par') or os.path.isfile(name0+'.resi'):
			name0=name+'_'+str(tmp)
			tmp+=1
		parsave=name0+'.par'
		resisave=name0+'.resi'
	else:
		if nsave==1:
			parsave=args.save[0]+'.par'
			resisave=args.save[0]+'.resi'
		elif nsave==2:
			parsave=args.save[0]
			resisave=args.save[1]
		if os.path.isfile(parsave):
			parser.error(text.error_pfe)
		if os.path.isfile(resisave):
			parser.error(text.error_rfe)
#
date,sec,toae,dt,dterr,freq_start,freq_end,dm,dmerr,period=data[jj].T
freq=(freq_start+freq_end)/2
nt=len(date)
time=te.times(te.time(date,sec,scale=info['telescope_info']['telename']))
psrt=pm.psr_timing(psr,time,np.inf)
phase=psrt.phase
dt=phase.offset
dterr=toae/period
dt=(dt-args.zero)%1
#
def merge(date,sec,dt,dterr,freq,dm,period):	# merge adjacent ToAs
	ttmp=time.local.mjd
	jj=np.zeros(nt,dtype=np.int32)
	j0=1
	t0=ttmp[0]
	if type(args.merge)!=np.float64:
		merge_time=0.5
	else:
		merge_time=args.merge
	for i in np.arange(nt):
		if np.abs(ttmp[i]-t0)<merge_time:
			jj[i]=j0
		else:
			t0=ttmp[i]
			j0+=1
			jj[i]=j0
	#
	dt2=[]
	dt2err=[]
	date2=[]
	sec2=[]
	dm2=[]
	dmerr2=[]
	freq2=[]
	period2=[]
	for i in np.arange(jj[-1])+1:
		setj=jj==i
		t0=dt[setj][:-1]
		ta=dterr[setj][:-1]
		errtmp=np.sqrt(1/(1/ta**2).sum())+t0.std()
		if errtmp<args.err:
			date2.append(date[setj].mean())
			sec2.append(sec[setj].mean())
			dm2.append(dm[setj].mean())
			dmerr2.append(dmerr[setj].mean())
			dt2.append((t0/ta**2).sum()/(1/ta**2).sum())
			dt2err.append(np.sqrt(1/(1/ta**2).sum()))
			freq2.append(freq[setj].mean())
			period2.append(period[setj].mean())
	date2=np.array(date2)
	sec2=np.array(sec2)
	dt2=np.array(dt2)
	dt2err=np.array(dt2err)
	freq2=np.array(freq2)
	dm2=np.array(dm2)
	dmerr2=np.array(dmerr2)
	period2=np.array(period2)
	return date2,sec2,dt2,dt2err,freq2,dm2,dmerr2,period2
#
if args.merge!=0:
	date,sec,dt,dterr,freq,dm,dmerr,period=merge(date,sec,dt,dterr,freq,dm,period)
time=te.times(te.time(date,sec,scale=info['telescope_info']['telename']))
#
paras=args.fit.split(',')
#
def psrfit(psr,paras,time,dt,toae,freq):	# fit the ToAs with pulsar parameters
	psrt=pm.psr_timing(psr,time,np.inf)
	lpara=len(paras)
	x0=np.zeros(lpara+1)
	for i in np.arange(lpara):
		tmp=psr.__getattribute__(paras[i])
		if type(tmp)==te.time:
			x0[i]=tmp.mjd[0]
		else:
			x0[i]=tmp
	#
	def fit(para):
		psr1=psr.copy()
		for i in np.arange(lpara):
			psr1.modify(paras[i],para[i])
		psrt1=pm.psr_timing(psr1,time,np.inf)
		dphase=psrt1.phase.minus(psrt.phase)
		#print(para,dphase.mjd.mean())
		return dphase.phase+para[-1]
	#
	def dfunc(para):
		psr1=psr.copy()
		for i in np.arange(lpara):
			psr1.modify(paras[i],para[i])
		psrt1=pm.psr_timing(psr1,time,np.inf)
		tmp=psrt1.phase_der_para(paras).T
		#print(para,tmp)
		return np.concatenate((tmp,np.ones([time.size,1])),axis=1)/toae.reshape(-1,1)
	#
	def resi(para):
		return (fit(para)+dt)/toae
	#
	a=so.leastsq(resi,x0=x0,full_output=True,Dfun=dfunc)
	popt,pcov=a[0],(np.diag(a[1])*(resi(a[0])**2).sum()/(len(dt)-len(x0)))**0.5
	psr1=psr.copy()
	for i in np.arange(lpara):
		psr1.modify(paras[i],popt[i])
	psrt2=pm.psr_timing(psr1,time,np.inf)
	#print(fit(popt),dt,x0)
	return popt,np.sqrt(pcov),fit(popt)+dt,resi(popt),psr1,psrt2
#
psrp,psrpe,res,rese,psr1,psrt=psrfit(psr,paras,time,dt,dterr,freq)
if args.time:
	j1=np.abs(res)<args.crit
	period=period.mean()*1e6
	res=res*period
	dt=(dt-dt.mean())*period
	dterr*=period
	yunit=' ($\mu$s)'
else:
	yunit=''
	j1=np.abs(res)<args.crit
phasestd=np.sqrt((rese[j1]**2).sum()/(1/dterr[j1]**2).sum())
if args.time:
	print(tuple(psrp),'\n',tuple(psrpe),phasestd/1e6)
else:
	print(tuple(psrp),'\n',tuple(psrpe),phasestd*psr.p0)
#
if type(args.save) is list:
	psr1.writepar(parsave)
	if args.time: dt_tmp,dterr_tmp=res,dterr
	else: dt_tmp,dterr_tmp=(res-res.mean())*psr.p0,dterr*psr.p0
	np.savetxt(resisave,np.array([time.local.date,time.local.second,dt_tmp,dterr_tmp]).T)
	print(text.info_save % (parsave, resisave))
#
fig=plt.figure(3)
plt.clf()
x1,x2=0.18,0.95
y1,y2,y3=0.16,0.3,0.95
if args.x=='orbit':
	xaxis=psrt.orbits[j1]%1
	xlim=0,1
	xlabel='Orbital Phase'
elif args.x=='lst':
	lst=psrt.time.lst[j1]
	lst[(lst-psr.raj/np.pi/2)>0.5]-=1
	lst[(lst-psr.raj/np.pi/2)<-0.5]+=1
	xaxis=lst*24
	xmax,xmin=np.max(xaxis),np.min(xaxis)
	xlim=xmin*1.05-xmax*0.05,-xmin*0.05+xmax*1.05
	xlabel='Sidereal Time (h)'
elif args.x=='mjd':
	xaxis=time.local.mjd[j1]
	xlim=xaxis[0]*1.05-xaxis[-1]*0.05,-xaxis[0]*0.05+xaxis[-1]*1.05
	xlabel='MJD (d)'
elif args.x=='year':
	xaxis=te.mjd2datetime(time.local.mjd)[4][j1]
	xlim=0,366
	xlabel='Day in a Year (d)'
#
if args.illu=='post':
	ax1=fig.add_axes((x1,y1,x2-x1,y3-y1))
	ax1.errorbar(xaxis,res[j1],dterr[j1],fmt='.')
	ax1.set_xlim(*xlim)
	ax1.set_xlabel(xlabel,fontsize=25,family='serif')
	ax1.set_ylabel('Fit Resi.'+yunit,fontsize=25,family='serif')
elif args.illu=='dm':
	ax1=fig.add_axes((x1,y1,x2-x1,y3-y1))
	ax1.errorbar(xaxis,dm[j1],yerr=dmerr[j1],fmt='.')
	ax1.set_xlim(*xlim)
	ax1.set_xlabel(xlabel,fontsize=25,family='serif')
	ax1.set_ylabel('DM',fontsize=25,family='serif')
elif args.illu=='prepost':
	ax1=fig.add_axes((x1,y2,x2-x1,y3-y2))
	ax2=fig.add_axes((x1,y1,x2-x1,y2-y1))
	ax1.errorbar(xaxis,dt[j1],dterr[j1],fmt='.')
	ax2.errorbar(xaxis,res[j1],dterr[j1],fmt='.')
	ax1.set_xlim(*xlim)
	ax2.set_xlim(*xlim)
	ax2.set_xlabel(xlabel,fontsize=25,family='serif')
	ax1.set_ylabel('Pulse Phase Resi.'+yunit,fontsize=25,family='serif')
	ax2.set_ylabel('Fit Resi.'+yunit,fontsize=25,family='serif')
	ax1.set_xticks([])
plt.show()
#
