#!/usr/bin/env python
import numpy as np
import numpy.polynomial.chebyshev as nc
import argparse as ap
import numpy.fft as fft
import os,ld,time,sys
import scipy.optimize as so
import time_eph as te
import psr_read as pr
import psr_model as pm
import warnings as wn
import adfunc as af
wn.filterwarnings('ignore')
#
version='JigLu_20201202'
parser=ap.ArgumentParser(prog='ldtoa',description='Get the relative pulse rotating phase and DM of the ld file.',epilog='Ver '+version)
parser.add_argument('-v','--version',action='version',version=version)
parser.add_argument("filename",nargs='+',help="input ld file or files")
parser.add_argument('-p',dest='template',required=True,help="pulse profile template ld file")
parser.add_argument('--freq_align',action='store_true',default=False,dest='freq_align',help="use same frequency band to obtain the ToA")
parser.add_argument('-T','--tscrunch',action='store_true',default=False,dest='tscrunch',help='time scrunch to one subint to obtain result')
parser.add_argument('--fr','--frequency_range',default=0,dest='freqrange',help='calculate in the frequency range (FREQ0,FREQ1)')
parser.add_argument('--sr','--subint_range',default=0,dest='subint_range',help='calculate in the subint range (SUBINT0,SUBINT1)')
parser.add_argument('--br','--phase_range',default=0,dest='phaserange',help='calculate with the data in phase range (PHASE0,PHASE1) of the template')
parser.add_argument('-z',"--zap",dest="zap_file",default=0,help="file recording zap channels")
parser.add_argument('-Z',action='store_true',default=False,dest="zap_template",help="zap same channels for the template file")
parser.add_argument('-o',"--output",dest="output",default="",help="output file name")
parser.add_argument('-d',action='store_true',default=False,dest='dm_corr',help='the progress will not correcting the DM deviation before calculating rotating phase')
parser.add_argument('-l',"--linear_number",dest="lnumber",type=np.int8,default=0,help="the number of frequency-domain points for linear fitting")
parser.add_argument('-n',action='store_true',default=False,dest='norm',help='normalized the data at each channel before cal')
parser.add_argument('--freqtem',action='store_true',default=False,dest='freqtoa',help='use 2d freq domain template to obtain ToA')
parser.add_argument('-a',default='pgs',dest='algorithm',help='shift algorithm (default: pgs)')
args=(parser.parse_args())
command=['ldtoa.py']
dirname=os.path.abspath('.')
#
if not os.path.isfile(args.template):
	parser.error('A valid ld file name is required.')
#
d0=ld.ld(args.template)
info0=d0.read_info()
if info0['data_info']['mode']=='template':
	psrname=info0['pulsar_info']['psr_name']
else:
	psrname=pr.psr(info0['pulsar_info']['psr_par']).name
#
nchan0=info0['data_info']['nchan']
nsub0=info0['data_info']['nsub']
nbin0=info0['data_info']['nbin']
#
if args.zap_file:
	command.append('-z')
	if not os.path.isfile(args.zap_file):
		parser.error('The zap channel file is invalid.')
	zchan=np.loadtxt(args.zap_file,dtype=np.int32)
	if np.min(zchan)<0:
		parser.error('The zapped channel number is overrange.')
	if args.zap_template:
		command.append('-Z')
		if np.max(zchan)>=nchan0:
			parser.error('The zapped channel number is overrange.')
		zchan=np.array(list(set(np.where(info0['data_info']['chan_weight']==0)[0]).union(zchan)))
else:
	zchan=np.int32([])
#
if args.subint_range:
	command.append('-s '+args.subint_range)
	sub_start,sub_end=np.int32(args.subint_range.split(','))
	if sub_end>0 and sub_start>sub_end:
		parser.error("Starting subint number larger than ending subint number.")
	elif sub_start<0:
		parser.error("Input subint is overrange.")
#
if args.tscrunch:
	command.append('-T')
#
if args.freq_align:
	command.append('-f')
#
if args.dm_corr: 
	command.append('-d ')
	if args.freqtoa: parser.error('The freq-domain template need not to correct the dispersion measure.')
#
if args.freqtoa:
	command.append('-F ')
#
freq_start0,freq_end0=info0['data_info']['freq_start'],info0['data_info']['freq_end']
freq0=(freq_start0+freq_end0)/2.0
channel_width0=(freq_end0-freq_start0)/nchan0
#
if args.freqrange:
	command.append('--fr '+args.freqrange)
	freq_s,freq_e=np.float64(args.freqrange.split(','))
	if freq_s>freq_e:
		parser.error("Starting frequency larger than ending frequency.")
else:
	freq_s,freq_e=freq_start0,freq_end0
#
if args.phaserange:
	command.append('--pr '+args.freqrange)
	phase_s,phase_e=np.float64(args.freqrange.split(','))
	if phase_s>1 or phase_s<0 or phase_e>1 or phase_e<0:
		parser.error("The given phase range is invalid.")
	if phase_e>phase_s: phase_s+=1
	zbins0=np.sort(np.arange(nbin0).reshape(1,-1).repeat(2,axis=1).reshape(-1)[int(nbin0*phase_e):int(nbin0*phase_s)])
	if zbins0.size==nbin0:
		parser.error("The given phase range is too small.")
else:
	phase_s,phase_e=0,1
	zbins0=np.array([])
#
filelist=args.filename
filenum=len(filelist)
nsub_new=[]
telename=''
def ld_check(fname,filetype='Ld file'): # check the consistensy of LD files
	global freq_s,freq_e,tmptele,telename
	if not os.path.isfile(fname):
		parser.error(filetype+' name '+fname+' '+'is invalid.')
	try:
		f=ld.ld(filelist[i])
		finfo=f.read_info()
	except:
		parser.error(filetype+' '+fname+' is invalid.')
	tmpname=pr.psr(finfo['pulsar_info']['psr_par'],warning=False).name
	if not telename: telename=finfo['telescope_info']['telename']
	tmptele=finfo['telescope_info']['telename']
	if psrname!=tmpname:
		parser.error('The pulsar recorded in '+fname+' is different from the template.')
	if telename!=tmptele:
		parser.error('The data should come from the same telescope.')
	#
	nchan=finfo['data_info']['nchan']
	nperiod=finfo['data_info']['nsub']
	#
	if args.zap_file:
		if args.zap_template:
			if nchan0!=nchan: parser.error('The channel numbers of data and template are different, and the zap file should not be same.')
		if np.max(zchan)>=nchan or np.min(zchan)<0: parser.error('The zapped channel number is overrange.')
	#
	if args.subint_range:
		if sub_end<0: sub_end_tmp=nperiod+sub_end
		else: sub_end_tmp=sub_end
		if sub_start>sub_end_tmp: parser.error("Starting subint number larger than ending subint number.")
		elif sub_end_tmp>nperiod or sub_end_tmp<0: parser.error("Input subint is overrange.")
	#
	if args.tscrunch: nsub_new.append(1)
	elif args.subint_range: nsub_new.append(sub_end_tmp-sub_start)
	else: nsub_new.append(nperiod)
	#
	freq_start,freq_end=finfo['data_info']['freq_start'],finfo['data_info']['freq_end']
	if freq_start0>=freq_end or freq_end0<=freq_start: parser.error('The template has different frequency band from the data.')
	freq=(freq_start+freq_end)/2.0
	channel_width=(freq_end-freq_start)/nchan
	if args.zap_file and args.zap_template:
		if max(np.abs(freq_end-freq_end0),np.abs(freq_start-freq_start0))>min(channel_width,channel_width0):
			parser.error('The frequency ranges of data and template are different, and the zap file should not be same.')
	if args.freq_align:
		if freq_start0>freq_end or freq_end<freq_start0: parser.error("Frequency bands of data and template have no overlap.")
		if args.freqrange: 
			if freq_s<freq_start or freq_e>freq_end: parser.error("Input frequency is overrange.")
		else:
			freq_s=max(freq_s,freq_start)
			freq_e=min(freq_e,freq_end)
	elif args.freqrange:
		if freq_s<freq_start or freq_e>freq_end: parser.error("Input frequency is overrange.")
	if args.freqtoa:
		if args.freq_align:
			chanstart,chanend=np.int16(np.round((np.array([freq_s,freq_e])-freq)/channel_width+0.5*nchan))
		else:
			chanstart,chanend=0,nchan
		nchan_new=chanend-chanstart
		if nchan_new!=nchan0: parser.error("The freq-domain ToA cannot be obtained for data with different frequency parameters from that of the template.")				
#
for i in np.arange(filenum):
	ld_check(filelist[i])
#
nsub_new=np.array(nsub_new)
if args.freq_align:
	chanstart0,chanend0=np.int16(np.round((np.array([freq_s,freq_e])-freq0)/channel_width0+0.5*nchan0))
else:
	chanstart0,chanend0=0,nchan0
#
name=args.output
if not name:
	output='screen'
else:
	name_tmp='    '+name
	if name_tmp[-3:]=='.ld':
		output='ld'
		name=name[:-3]
	elif name_tmp[-4:] in ['.txt','.tim']:
		if os.path.isfile(name):
			parser.error('The name of output file already existed. Please provide a new name.')
		output=name_tmp[-3:]
	else: output='ld'
	if output=='ld':
		if os.path.isfile(name+'.ld'):
			parser.error('The name of output file already existed. Please provide a new name.')
#
def shift(y,x):
	ffts=y*np.exp(x*1j)
	fftr=fft.irfft(ffts)
	return fftr
#
def dmcor(data,freq,rchan,period,dm0,output=1):	# correct the DM for multi-frequency data
	data=data[rchan]
	freq=freq[rchan]
	fftdata=fft.rfft(data,axis=1)
	tmp=np.shape(fftdata)[-1]
	const=(1/freq**2*pm.dm_const/period*np.pi*2.0).repeat(tmp).reshape(-1,tmp)*np.arange(tmp)
	dm_zone=np.max([0.1,np.float64(dm0)/100])
	dm_zone=np.min([0.5,dm_zone])
	ddm,ddmerr=af.dmdet(fftdata,const,0,dm_zone,9,prec=1e-4)
	if output==1:
		data=shift(fftdata,const*ddm)
		return data,ddm,ddmerr
	else:
		return ddm,ddmerr
#
if args.zap_file and args.zap_template:
	rchan=np.array(list(set(range(chanstart0,chanend0))-set(list(zchan))))-chanstart0
else:
	rchan=np.arange(chanend0-chanstart0)
frequse=np.arange(freq_start0,freq_end0,channel_width0)[chanstart0:chanend0][rchan]
#
if info0['data_info']['mode']!='template':	# processing the template
	data0=d0.period_scrunch()[chanstart0:chanend0,:,0]*np.asarray(info0['data_info']['chan_weight'])[chanstart0:chanend0].reshape(-1,1)
	if not args.freqtoa:
		if not args.dm_corr and chanend0-chanstart0>1:
			psr=pm.psr_timing(pr.psr(info0['pulsar_info']['psr_par'],warning=False),te.times(te.time(info0['data_info']['stt_time'],info0['data_info']['length']/2,scale=info['telescope_info']['telename'])),(freq_start0+freq_end0)/2)
			freq_real0=np.linspace(freq_start0,freq_end0,nchan0+1)[:-1]*psr.vchange.mean()
			if 'additional_info' in info0.keys():
				if 'best_dm' in info0['additional_info'].keys():
					tmp=True
					ddm0=info0['additional_info']['best_dm'][0]-info0['data_info']['dm']
					fftdata0=fft.rfft(data0,axis=1)
					tmp=np.shape(fftdata0)[-1]
					const=(1/freq_real0**2*pm.dm_const/info0['data_info']['period']*np.pi*2.0).repeat(tmp).reshape(-1,tmp)*np.arange(tmp)
					data0=shift(fftdata0,const*ddm0)
			if not tmp:
				data0[rchan],ddm0,ddm0err=dmcor(data0,freq_real0,rchan,info0['data_info']['period'],info0['data_info']['dm'])
	data_tmp=data0[rchan]
	if args.norm:
		command.append('-n')
		data_tmp-=data_tmp.mean(1).reshape(-1,1)
		data_tmp/=data_tmp.std(1).reshape(-1,1)
	if args.freqtoa:
		tpdata0=data_tmp
		tmp=tpdata0.sum(0)
		center0=np.arctan2((tmp*np.sin(np.pi*2/nbin0*np.arange(nbin0))).sum(),(tmp*np.cos(np.pi*2/nbin0*np.arange(nbin0))).sum())
		if zbins0.size<nbin0/10: tmp=tmp-af.baseline(tmp)
		else: tmp=tmp-tmp[zbins0].mean()
		if zbins0.size>0: tpdata0[:,zbins0]=tpdata0[:,zbins0].mean(1).reshape(-1,1)
	else:
		tpdata0=data_tmp.sum(0)
		center0=np.arctan2((tpdata0*np.sin(np.pi*2/nbin0*np.arange(nbin0))).sum(),(tpdata0*np.cos(np.pi*2/nbin0*np.arange(nbin0))).sum())
		if zbins0.size<nbin0/10: tmp=tpdata0-af.baseline(tpdata0)
		else: tmp=tpdata0-tpdata0[zbins0].mean()
		if zbins0.size>0: tpdata0[zbins0]=tpdata0[zbins0].mean()
else:
	if not args.freqtoa:
		data0=d0.chan_scrunch()[:,:,0]
		if nsub0==1:
			tpdata0=data0[0]
			center0=np.arctan2((tpdata0*np.sin(np.pi*2/nbin0*np.arange(nbin0))).sum(),(tpdata0*np.cos(np.pi*2/nbin0*np.arange(nbin0))).sum())
			if zbins0.size<nbin0/10: tmp=tpdata0-af.baseline(tpdata0)
			else: tmp=tpdata0-tpdata0[zbins0].mean()
			if zbins0.size>0: tpdata0[zbins0]=tpdata0[zbins0].mean()
		else:
			if d0.read_shape()[0]>1:
				parser.error("The multi-component freq-domain template cannot be adopted for non-freq-domain ToA determination.")
			tpdata0=data0
			tmp=tpdata0[0]
			center0=np.arctan2((tmp*np.sin(np.pi*2/nbin0*np.arange(nbin0))).sum(),(tmp*np.cos(np.pi*2/nbin0*np.arange(nbin0))).sum())
			if zbins0.size<nbin0/10: tmp=tmp-af.baseline(tmp)
			else: tmp=tmp-tmp[zbins0].mean()
			if zbins0.size>0: tpdata0[:,zbins0]=tpdata0[:,zbins0].mean(1).reshape(-1,1)
	else:
		data0=d0.read_data()[:,:,:,0]
		if len(data0)==1: parser.error("The freq-domain ToA cannot be obtained for template with only one frequency channel.")
		if nsub0==1:
			tpdata0=data0[:,0]*np.asarray(info0['data_info']['chan_weight']).reshape(-1,1)
			tmp=tpdata0.sum(0)
			center0=np.arctan2((tmp*np.sin(np.pi*2/nbin0*np.arange(nbin0))).sum(),(tmp*np.cos(np.pi*2/nbin0*np.arange(nbin0))).sum())
			if zbins0.size<nbin0/10: tmp=tmp-af.baseline(tmp)
			else: tmp=tmp-tmp[zbins0].mean()
			if zbins0.size>0: tpdata0[:,zbins0]=tpdata0[:,zbins0].mean(1).reshape(-1,1)
		else:
			tpdata0=data0*np.asarray(info0['data_info']['chan_weight']).reshape(-1,1,1)
			tmp=tpdata0[:,0].sum(0)
			center0=np.arctan2((tmp*np.sin(np.pi*2/nbin0*np.arange(nbin0))).sum(),(tmp*np.cos(np.pi*2/nbin0*np.arange(nbin0))).sum())
			if zbins0.size<nbin0/10: tmp=tmp-af.baseline(tmp)
			else: tmp=tmp-tmp[zbins0].mean()
			if zbins0.size>0: tpdata0[:,:,zbins0]=tpdata0[:,:,zbins0].mean(2).reshape(nchan0,-1,1)
#
ew=tmp.sum()/tmp.max()
#
if args.lnumber:
	command.append('-l '+str(args.lnumber))
	lnumber=args.lnumber
	if lnumber>100 or lnumber<=2:
		parser.error('The number of frequency-domain points for linear fitting is invalid.')
else:
	lnumber=int(round(nbin0/ew/2))
	lnumber=max(3,lnumber)
#
algorithms=['pgs','corr']
if args.algorithm in algorithms:
	command.append('-m '+args.algorithm)
else:
	parser.error('The timing method cannot be recognized.')
#
command=' '.join(command)
#
def lin(x,k):
	return k*x
#
def poa(tpdata0,tpdata):	# phase gradient method
	nb=int(min(nbin0,nbin)//2+1)
	tpdata-=tpdata.mean()
	tpdata/=tpdata.max()
	tpdata0-=tpdata0.mean()
	tpdata0/=tpdata0.max()
	f0=fft.rfft(tpdata0)[:nb]
	d0=fft.irfft(f0)
	f=fft.rfft(tpdata)[:nb]
	d=fft.irfft(f)
	tmpnum=np.argmax(fft.irfft(f0*f.conj()))
	d0=np.append(d0[tmpnum:],d0[:tmpnum])
	f0=fft.rfft(d0)
	df=f0/f
	#errbinnum=np.min([int(nb/6),20])
	#sr0=np.std(f0.real[-errbinnum:])
	#si0=np.std(f0.imag[-errbinnum:])
	#sr=np.std(f.real[-errbinnum:])
	#si=np.std(f.imag[-errbinnum:])
	#err=np.sqrt((sr**2+si**2)/np.abs(f)**2+(sr0**2+si0**2)/np.abs(f0)**2)
	#ang=np.angle(df)
	#fitnum=lnumber
	#popt,pcov=so.curve_fit(lin,np.arange(1,fitnum),ang[1:fitnum],p0=[0.0],sigma=err[1:fitnum])
	#dt=popt[0]/(2*np.pi)-tmpnum/((nb-1)*2)
	#dterr=pcov[0,0]**0.5/(2*np.pi)
	fitnum=lnumber
	#err=np.sqrt((sr**2+si**2)/np.abs(f)**2+(sr0**2+si0**2)/np.abs(f0)**2)[1:fitnum]/np.arange(1,fitnum)
	err=1/np.abs(f0)[1:fitnum]/np.arange(1,fitnum)
	ang=np.angle(df[1:fitnum])/np.arange(1,fitnum)
	ang0=(ang/err**2).sum()/(1/err**2).sum()
	dt=ang0/(2*np.pi)-tmpnum/((nb-1)*2)
	dterr=np.sqrt(((ang-ang0)**2/err**2).sum()/(1/err**2).sum())/(2*np.pi)
	return [dt,dterr]
#
def coa(tpdata0,tpdata):	# sinc interpolation correlation method
	nb=int(min(nbin0,nbin)//2+1)
	tpdata-=tpdata.mean()
	tpdata/=tpdata.max()
	tpdata0-=tpdata0.mean()
	tpdata0/=tpdata0.max()
	f0=fft.rfft(tpdata0)[:nb]
	d0=fft.irfft(f0)
	f=fft.rfft(tpdata)[:nb]
	d=fft.irfft(f)
	tmpnum=np.argmax(fft.irfft(f0*f.conj()))
	d0=np.append(d0[tmpnum:],d0[:tmpnum])
	f0=fft.rfft(d0)
	theta=np.linspace(-1,1,400)/nb
	corr=(d0*fft.irfft(f*np.exp(1j*np.arange(nb)*theta.reshape(-1,1)*(2*np.pi)))).sum(1)
	polyfunc=np.polyfit(theta,corr,9)
	fitvalue=np.polyval(polyfunc,theta)
	roots=np.roots(np.polyder(polyfunc))
	roots=np.real(roots[np.isreal(roots)])
	thetamax=roots[np.argmin(np.abs(roots-theta[np.argmax(corr)]))]
	error=np.std(corr-fitvalue)
	errfunc=np.append(polyfunc[:-1],polyfunc[-1]-np.polyval(polyfunc,thetamax)+error)
	roots=np.roots(errfunc)
	roots=np.real(roots[np.isreal(roots)])
	err=np.mean(np.abs(np.array([roots[np.argmin(np.abs(roots-0.5/nb))],roots[np.argmin(np.abs(roots+0.5/nb))]])-thetamax))
	dt=thetamax-tmpnum/((nb-1)*2)
	dterr=err
	return [dt,dterr]
#
def foa(tpdata0,tpdata):	# leastsq method
	nb=int(min(nbin0,nbin)//2+1)
	tpdata-=tpdata.mean()
	tpdata/=tpdata.max()
	tpdata0-=tpdata0.mean()
	tpdata0/=tpdata0.max()
	f0=fft.rfft(tpdata0)[:nb]
	d0=fft.irfft(f0)
	f=fft.rfft(tpdata)[:nb]
	d=fft.irfft(f)
	tmpnum=np.argmax(fft.irfft(f0*f.conj()))
	d0=np.append(d0[tmpnum:],d0[:tmpnum])
	ftmp=1j*np.arange(nb)*(2*np.pi)
	def fit(x,t,k):
		return k*fft.irfft(fft.rfft(x)*np.exp(t*ftmp))
	popt,pcov=so.curve_fit(fit,d,d0,p0=(0.0,1.0))
	dt=popt[0]-tmpnum/((nb-1)*2)
	dterr=np.sqrt(pcov[0,0])
	return [dt,dterr]
#
def ftoa(tpdata0,tpdata,comp,kvalue):	# leastsq method for 2 components template
	nb=int(min(nbin0,nbin)//2+1)
	tpdata-=tpdata.mean()
	tpdata/=tpdata.max()
	tpdata0-=tpdata0.mean()
	comp-=comp.mean()
	comp/=tpdata0.max()
	tpdata0/=tpdata0.max()
	fcomp=fft.rfft(comp)[:nb]
	dcomp=fft.irfft(fcomp)
	f0=fft.rfft(tpdata0)[:nb]
	d0=fft.irfft(f0)
	f=fft.rfft(tpdata)[:nb]
	d=fft.irfft(f)
	tmpnum=np.argmax(fft.irfft((f0+fcomp)*f.conj()))
	d0=np.append(d0[tmpnum:],d0[:tmpnum])
	dcomp=np.append(dcomp[tmpnum:],dcomp[:tmpnum])
	ftmp=1j*np.arange(nb)*(2*np.pi)
	def fit(x,t,k,k1):
		return k*fft.irfft(fft.rfft(x)*np.exp(t*ftmp))-k1*dcomp
	popt,pcov=so.curve_fit(fit,d,d0,p0=(0.0,1.0,0.0))
	dt=popt[0]-tmpnum/((nb-1)*2)
	dterr=np.sqrt(pcov[0,0])
	return [dt,dterr]
#
result=np.zeros([nsub_new.sum(),10])
cumsub=np.append(0,np.cumsum(nsub_new)[:-1])
discard=[]
reserve=[]
if args.algorithm=='pgs':
	tfunc=poa
elif args.algorithm=='corr':
	tfunc=coa
elif args.algorithm=='fit':
	tfunc=foa
#
def dmdt(freq,dm,c):
	return 1/freq**2*pm.dm_const/period0*dm+c
#
def toafunc(tpdata0,tpdata,freq=0,comp=[],kv=0):	# choose ToA calculation method
	if not args.freqtoa:
		return tfunc(tpdata0,tpdata)
	else:
		jj=(tpdata0.mean(1)!=0)&(tpdata.mean(1)!=0)
		tp0,tp=tpdata0[jj],tpdata[jj]
		ff=freq[jj]
		kv=kv[jj]
		comp=comp[jj]
		nchan=jj.sum()
		dp,dperr=np.zeros([2,nchan])
		for i in np.arange(nchan):
			if len(comp)==0:
				dp[i],dperr[i]=tfunc(tp0[i],tp[i])
			else:
				dp_tmp,dperr_tmp=np.zeros([2,nk])
				kvalue=np.linspace(*kv[i],nk)
				for ind in np.arange(nk):
					dp_tmp[ind],dperr_tmp[ind]=ftoa(tp0[i],tp[i],comp[i],kvalue[ind])
				ind=np.argmin(dperr_tmp)
				dp[i],dperr[i]=dp_tmp[ind],dperr_tmp[ind]
		dp0=dp%1
		dp0[dp0>0.5]-=1
		if nchan>1:
			res=so.curve_fit(dmdt,ff,dp0,p0=(0.0,0.0),sigma=dperr)
			ddm,ddmerr=res[0][0],np.sqrt(res[1][0][0])
			dp1=((dp-dmdt(freq[jj],ddm,0))/dperr**2).sum()/(1/dperr**2).sum()
			dp1err=np.sqrt(1/(1/dperr**2).sum())
			return dp1,dp1err,ddm,ddmerr
		else:
			return dp[0],dperr[0],0,1e100
#
for k in np.arange(filenum):
	d=ld.ld(filelist[k])
	info=d.read_info()
	psr0=pr.psr(info['pulsar_info']['psr_par'],warning=False)
	period0=info['data_info']['period']
	#
	nchan=info['data_info']['nchan']
	nbin=info['data_info']['nbin']
	nperiod=info['data_info']['nsub']
	npol=info['data_info']['npol']
	#
	zchan_tmp=np.array(list(set(np.where(info['data_info']['chan_weight']==0)[0]).union(zchan)))
	#
	if args.subint_range:
		if sub_end<0:
			sub_e=nperiod+sub_end
		else:
			sub_e=sub_end
		sub_s=sub_start
	else:
		sub_s,sub_e=0,nperiod
	#
	freq_start,freq_end=info['data_info']['freq_start'],info['data_info']['freq_end']
	freq=(freq_start+freq_end)/2.0
	channel_width=(freq_end-freq_start)/nchan
	if args.freq_align:
		chanstart,chanend=np.int16(np.round((np.array([freq_s,freq_e])-freq)/channel_width+0.5*nchan))
	else:
		chanstart,chanend=0,nchan
	#
	nchan_new=chanend-chanstart
	if nchan==1:
		rchan=[0]
	else:
		rchan=np.array(list(set(range(chanstart,chanend))-set(list(zchan_tmp))))-chanstart
	if not args.tscrunch:
		phase0=info['additional_info']['phase0']
		sub_nperiod=info['data_info']['sub_nperiod']*np.ones(nperiod,dtype=np.float64)
		sub_nperiod[-1]=info['data_info']['sub_nperiod_last']
		middle=sub_nperiod.cumsum()-sub_nperiod/2
		time0=info['data_info']['length']*(nc.chebpts1(20)+1)/2
		psr=pm.psr_timing(psr0,te.times(te.time(info['data_info']['stt_date']*np.ones(20,dtype=np.float64),info['data_info']['stt_sec']+time0,scale=info['telescope_info']['telename'])),np.inf)
		phase1=psr.phase
		chebc=nc.chebfit(time0,phase1.integer-phase1.integer[0]+phase1.offset,12)
		chebd=nc.chebder(chebc)
		time0=np.linspace(0,info['data_info']['length'],nperiod)
		psr=pm.psr_timing(psr0,te.times(te.time(info['data_info']['stt_date']*np.ones(nperiod,dtype=np.float64),info['data_info']['stt_sec']+time0,scale=info['telescope_info']['telename'])),np.inf)
		phase1=psr.phase
		phase_start=phase1.integer[0]
		middle_time=np.interp(middle,phase1.integer-phase0+phase1.offset,time0)[sub_s:sub_e]
		psr1=pm.psr_timing(psr0,te.times(te.time(info['data_info']['stt_date']*np.ones(nsub_new[k],dtype=np.float64),info['data_info']['stt_sec']+middle_time,scale=info['telescope_info']['telename'])),np.inf)
		middle_phase=psr1.phase
		data1=d.period_scrunch(sub_s,sub_e)[chanstart:chanend,:,0]*np.asarray(info['data_info']['chan_weight'])[chanstart:chanend].reshape(-1,1)
		freq_real=(np.linspace(freq_start,freq_end,nchan+1)[:-1]+channel_width/2)*psr.vchange.mean()
		if not args.freqtoa:
			if not args.dm_corr:
				tmp=False
				if 'additional_info' in info.keys():
					if 'best_dm' in info['additional_info'].keys():
						ddm=info['additional_info']['best_dm'][0]-info['data_info']['dm']
						ddmerr=info['additional_info']['best_dm'][1]
						tmp=True
				if not tmp:
					ddm,ddmerr=dmcor(data1,freq_real,rchan,period0,info['data_info']['dm'],output=0)
				#print('The relative DM from the template file to data file is '+str(ddm-ddm0))
				dm_new=ddm+info['data_info']['dm']
			else:
				dm_new=info['data_info']['dm']
				ddmerr=0
		if info0['data_info']['mode']=='template' and nsub0>1:
			krange=info0['additional_info']['krange']
			if not args.freqtoa:
				kvalue=np.float64(krange.split(','))
			else:
				kvalue=np.float64(list(map(lambda x: x.split(','),krange)))
		for s in np.arange(nsub_new[k]):
			data=d.read_period(s+sub_s)[chanstart:chanend,:,0]*np.asarray(info['data_info']['chan_weight'])[chanstart:chanend]
			if np.any(np.isnan(data)) or np.any(np.isinf(data)) or np.all(data==0):
				discard.append([filelist[k],s+sub_s])
				continue
			else: reserve.append([os.path.abspath(filelist[k]),int(s+sub_s)])
			if not args.freqtoa:
				if not args.dm_corr:
					fftdata=fft.rfft(data,axis=1)
					tmp=np.shape(fftdata)[-1]
					const=(1/freq_real**2*pm.dm_const/period0*np.pi*2.0).repeat(tmp).reshape(-1,tmp)*np.arange(tmp)
					data=shift(fftdata,const*ddm)
			data_tmp=data[rchan]
			if args.norm:
				data_tmp-=data_tmp.mean(1).reshape(-1,1)
				data_tmp/=data_tmp.std(1).reshape(-1,1)
			if not args.freqtoa:
				tpdata=data_tmp.mean(0)
				center=np.arctan2((tpdata*np.sin(np.pi*2/nbin*np.arange(nbin))).sum(),(tpdata*np.cos(np.pi*2/nbin*np.arange(nbin))).sum())
			else:
				tpdata=data_tmp
				center=np.arctan2((tpdata.mean(0)*np.sin(np.pi*2/nbin*np.arange(nbin))).sum(),(tpdata.mean(0)*np.cos(np.pi*2/nbin*np.arange(nbin))).sum())
			if zbins0.size>0:
				phase_s_tmp,phase_e_tmp=phase_s+center-center0,phase_e+center-center0
				if phase_s_tmp>2:
					phase_s_tmp-=1
					phase_e_tmp-=1
				zbins=np.sort(np.arange(nbin).reshape(1,-1).repeat(2,axis=1).reshape(-1)[int(nbin*phase_e_tmp):int(nbin*phase_s_tmp)])
				if not args.freqtoa: tpdata[:,zbins]=tpdata[:,zbins].mean(1).reshape(-1,1)
				else: tpdata[zbins]=tpdata[zbins].mean()
			middle_int=middle_phase.integer[s]-phase_start
			middle_offs=middle_phase.offset[s]
			if info0['data_info']['mode']=='template' and nsub0>1:
				nk=20
				dp_tmp,dpe_tmp,ddm_tmp,ddmerr_tmp=np.zeros([4,nk])
				if args.freqtoa:
					dp,dpe,dm_new,ddmerr=toafunc(tpdata0[:,0],tpdata,freq_real,tpdata0[:,1],kvalue)
				else:
					kvalue=np.linspace(*kvalue,nk)
					for ind in np.arange(nk):
						dp_tmp[ind],dpe_tmp[ind]=ftoa(tpdata0[0],tpdata,tpdata0[1],kvalue[ind])
					ind=np.argmin(dpe_tmp)
					dp,dpe=dp_tmp[ind],dpe_tmp[ind]
			else:
				if args.freqtoa:
					dp,dpe,ddm,ddmerr=toafunc(tpdata0,tpdata,freq_real)
					dm_new=ddm+info['data_info']['dm']
				else: dp,dpe=toafunc(tpdata0,tpdata)
			dp0=dp+np.round(middle_offs-dp)
			roots=nc.chebroots(chebc-([dp0+middle_int]+[0]*12))
			roots=np.real(roots[np.isreal(roots)])
			root=roots[np.argmin(np.abs(roots-middle_int*period0))]
			toa=te.time(info['data_info']['stt_date'],root+info['data_info']['stt_sec'],scale=info['telescope_info']['telename'])
			period=1/nc.chebval(root,chebd)
			toae=dpe*period
			if output=='screen':
				print(toa.date[0],toa.second[0],dp,dpe,freq_start,freq_end,dm_new,period)
			else:
				result[cumsub[k]+s-len(discard)]=[toa.date[0],toa.second[0],toae,dp,dpe,freq_start,freq_end,dm_new,ddmerr,period]
	else:
		phase0=info['additional_info']['phase0']
		sub_nperiod=info['data_info']['sub_nperiod']*np.ones(nperiod,dtype=np.float64)
		sub_nperiod[-1]=info['data_info']['sub_nperiod_last']
		sub_nperiod_cumsum=sub_nperiod.cumsum()
		if sub_s==0: middle=sub_nperiod_cumsum[sub_e-1]/2
		else: middle=(sub_nperiod_cumsum[sub_s-1]+sub_nperiod_cumsum[sub_e-1])/2
		time0=np.linspace(0,info['data_info']['length'],12)
		phase1=pm.psr_timing(psr0,te.times(te.time(info['data_info']['stt_date']*np.ones(12,dtype=np.float64),info['data_info']['stt_sec']+time0,scale=info['telescope_info']['telename'])),np.inf).phase
		chebc=nc.chebfit(phase1.integer-phase0+phase1.offset,time0,7)
		chebd=nc.chebder(chebc)
		middle_time=nc.chebval(middle,chebc)
		psr1=pm.psr_timing(psr0,te.times(te.time(info['data_info']['stt_date'],info['data_info']['stt_sec']+middle_time,scale=info['telescope_info']['telename'])),np.inf)
		data=d.period_scrunch(sub_s,sub_e)[chanstart:chanend,:,0]*np.asarray(info['data_info']['chan_weight'])[chanstart:chanend].reshape(-1,1)
		if np.any(np.isnan(data)) or np.any(np.isinf(data)) or np.all(data==0):
			discard.append(filelist[k])
			continue
		else: reserve.append([os.path.abspath(filelist[k]),int(sub_s),int(sub_e)])
		freq_real=(np.linspace(freq_start,freq_end,nchan+1)[:-1]+channel_width/2)*psr.vchange.mean()
		if not args.freqtoa:
			if not args.dm_corr:
				tmp=False
				if 'additional_info' in info.keys():
					if 'best_dm' in info['additional_info'].keys():
						tmp=True
						ddm=info['additional_info']['best_dm'][0]-info['data_info']['dm']
						ddmerr=info['additional_info']['best_dm'][1]
						fftdata=fft.rfft(data,axis=1)
						tmp=np.shape(fftdata)[-1]
						const=(1/freq_real**2*pm.dm_const/period0*np.pi*2.0).repeat(tmp).reshape(-1,tmp)*np.arange(tmp)
						data=shift(fftdata,const*ddm)[rchan]
				if not tmp:
					data,ddm,ddmerr=dmcor(data,freq_real,rchan,period0,info['data_info']['dm'])
				dm_new=ddm+info['data_info']['dm']
			else:
				dm_new=info['data_info']['dm']
				ddmerr=0
		if args.norm:
			data-=data.mean(1).reshape(-1,1)
			data/=data.std(1).reshape(-1,1)
		if not args.freqtoa:
			tpdata=data.mean(0)
			center=np.arctan2((tpdata*np.sin(np.pi*2/nbin*np.arange(nbin))).sum(),(tpdata*np.cos(np.pi*2/nbin*np.arange(nbin))).sum())
		else:
			tpdata=data
			center=np.arctan2((tpdata.mean(0)*np.sin(np.pi*2/nbin*np.arange(nbin))).sum(),(tpdata.mean(0)*np.cos(np.pi*2/nbin*np.arange(nbin))).sum())
		if zbins0.size>0:
			phase_s_tmp,phase_e_tmp=phase_s+center-center0,phase_e+center-center0
			if phase_s_tmp>2:
				phase_s_tmp-=1
				phase_e_tmp-=1
			zbins=np.sort(np.arange(nbin).reshape(1,-1).repeat(2,axis=1).reshape(-1)[int(nbin*phase_e_tmp):int(nbin*phase_s_tmp)])
			if not args.freqtoa: tpdata[:,zbins]=tpdata[:,zbins].mean(1).reshape(-1,1)
			else: tpdata[zbins]=tpdata[zbins].mean()
		if nsub0>1:
			nk=20
			dp_tmp,dpe_tmp,ddm_tmp,ddmerr_tmp=np.zeros([4,nk])
			if args.freqtoa:
				dp,dpe,dm_new,ddmerr=toafunc(tpdata0[:,0],tpdata,freq_real,tpdata0[:,1],kvalue)
			else:
				kvalue=np.linspace(*kvalue,nk)
				for ind in np.arange(nk):
					dp_tmp[ind],dpe_tmp[ind]=toafunc(tpdata0[0]+kvalue[ind]*tpdata0[1],tpdata)
				ind=np.argmin(dpe_tmp)
				dp,dpe=dp_tmp[ind],dpe_tmp[ind]
		else:
			if args.freqtoa:
				dp,dpe,ddm,ddmerr=toafunc(tpdata0,tpdata)
				dm_new=ddm+info['data_info']['dm']
			else: dp,dpe=toafunc(tpdata0,tpdata)
		middle_int,middle_offs=np.divmod(middle,1)
		dp0=dp+np.round(middle_offs-dp)
		root=nc.chebval(dp0+middle_int,chebc)
		toa=te.time(info['data_info']['stt_date'],root+info['data_info']['stt_sec'],scale=info['telescope_info']['telename'])
		period=nc.chebval(root,chebd)
		toae=dpe*period
		if output=='screen':
			print(toa.date[0],toa.second[0],toae,freq_start,freq_end,dm_new,period)
		else:
			result[cumsub[k]-len(discard)]=[toa.date[0],toa.second[0],toae,dp,dpe,freq_start,freq_end,dm_new,ddmerr,period]
#
if len(discard)>0:
	if args.tscrunch:
		print('The file'+(len(discard)>1)*'s'+' '+', '.join(discard)+' are discarded in calculating the ToA.')
	else:
		print('The following subint'+(len(discard)>1)*'s'+' are discarded in calculating the ToA: '+' '.join(list(map(lambda x: 'the '+str(x[1])+'th subint of the file '+x[0],discard)))+'.')
	result=result[:(-len(discard))]
#
if output=='ld':
	d1=ld.ld(name+'.ld')
	d1.write_shape([1,nsub_new.sum()-len(discard),10,1])
	d1.write_chan(result,0)
	toainfo={'data_info':{'mode':'ToA'},'original_data_info':{'filenames':reserve},'pulsar_info':{'psr_name':psrname},'history_info':{'history':[command],'file_time':[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())]},'telescope_info':{'telename':telename},'toa_info':{'method':args.algorithm}}
	d1.write_info(toainfo)
elif output=='txt':
	fout=open(name,'w')
	fout.write(psrname+' ToA\n')
	fout.write(telename+'\n')
	fout.write('{:28s} {:10s} {:10s} {:10s} {:13s} {:18s}'.format('ToA','ToAe','FreqStart','FreqEnd','DM','Period')+'\n')
	for i in result:
		fout.write('{:28s} {:10s} {:10.2f} {:10.2f} {:13f} {:18f}'.format(str(int(i[0]))+str(i[1]/86400)[1:],"%.3e"%(i[2]/86400),i[5],i[6],i[7],i[9])+'\n')
	fout.close()
elif output=='tim':
	fout=open(name,'w')
	fout.write('FORMAT 1\n')
	nind=max(4,np.ceil(np.log10(len(result)+1)))
	for i in np.arange(len(result)):
		ftmp='ToA_'+psrname+'_'+str(i).zfill(nind)+'.dat'
		fout.write('{:26s} {:10.6f} {:28s} {:4f} {:8s}'.format(ftmp,(result[i,5]+result[i,6])/2,str(int(result[i,0]))+str(result[i,1]/86400)[1:],result[i,2]*1e6,info['telescope_info']['telename'])+'\n')
	fout.close()
#

