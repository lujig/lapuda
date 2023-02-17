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
parser.add_argument('-t',dest='template',required=True,help="template ld file")
parser.add_argument('-f',action='store_true',default=False,dest='freq_align',help="use same frequency band to obtain the ToA")
parser.add_argument('-T','--tscrunch',action='store_true',default=False,dest='tscrunch',help='time scrunch to one subint to obtain result')
parser.add_argument('-r','--frequency_range',default=0,dest='freqrange',help='calculate in the frequency range (FREQ0,FREQ1)')
parser.add_argument('-s','--subint_range',default=0,dest='subint_range',help='calculate in the subint range (SUBINT0,SUBINT1)')
parser.add_argument('-z',"--zap",dest="zap_file",default=0,help="file recording zap channels")
parser.add_argument('-Z',action='store_true',default=False,dest="zap_template",help="zap same channels for the template file")
parser.add_argument('-o',"--output",dest="output",default="",help="outputfile name")
parser.add_argument('-d',action='store_true',default=False,dest='dm_corr',help='the progress will not correcting the DM deviation before calculating rotating phase')
parser.add_argument('-l',"--linear_number",dest="lnumber",type=np.int8,default=0,help="the number of frequency-domain points for linear fitting")
parser.add_argument('-n',action='store_true',default=False,dest='norm',help='normalized the data at each channel before cal')
parser.add_argument('-F',action='store_true',default=False,dest='freqtoa',help='use 2d freq domain template to obtain ToA')
parser.add_argument('-a',default='pgs',dest='algorithm',help='shift algorithm (default: pgs)')
args=(parser.parse_args())
command=['ldtoa.py']
#
if not os.path.isfile(args.template):
	parser.error('A valid ld file name is required.')
#
d0=ld.ld(args.template)
info0=d0.read_info()
if info0['mode']=='template':
	psrname=info0['psr_name']
else:
	psrname=pr.psr(info0['psr_par']).name
#
if 'compressed' in info0.keys():
	nchan0=info0['nchan_new']
	nsub0=info0['nsub_new']
	nbin0=info0['nbin_new']
else:
	nchan0=info0['nchan']
	nsub0=info0['nsub']
	nbin0=info0['nbin']
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
		if 'zchan' in info0.keys():
			info0['zchan']=str(list(set(map(int,info0['zchan'])).union(zchan)))[1:-1]
		else:
			info0['zchan']=zchan
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
freq_start0,freq_end0=info0['freq_start'],info0['freq_end']
freq0=(freq_start0+freq_end0)/2.0
channel_width0=(freq_end0-freq_start0)/nchan0
#
if args.freqrange:
	command.append('-r '+args.freqrange)
	freq_s,freq_e=np.float64(args.freqrange.split(','))
	if freq_s>freq_e:
		parser.error("Starting frequency larger than ending frequency.")
else:
	freq_s,freq_e=freq_start0,freq_end0
#
filelist=args.filename
filenum=len(filelist)
nsub_new=[]
def ld_check(fname,filetype='Ld file'):
	global freq_s,freq_e
	if not os.path.isfile(fname):
		parser.error(filetype+' name '+fname+' '+'is invalid.')
	try:
		f=ld.ld(filelist[i])
		finfo=f.read_info()
	except:
		parser.error(filetype+' '+fname+' is invalid.')
	sys.stdout=open(os.devnull,'w')
	tmpname=pr.psr(finfo['psr_par']).name
	sys.stdout=sys.__stdout__
	if psrname!=tmpname:
		parser.error('The pulsar recorded in '+fname+' is different from the template.')
	#
	if 'compressed' in finfo.keys():
		nchan=finfo['nchan_new']
		nperiod=finfo['nsub_new']
	else:
		nchan=finfo['nchan']
		nperiod=finfo['nsub']
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
	freq_start,freq_end=finfo['freq_start'],finfo['freq_end']
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
	elif name_tmp[-4:]=='.txt': 
		if os.path.isfile(name):
			parser.error('The name of output file already existed. Please provide a new name.')
		output='txt'
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
def dmcor(data,freq,rchan,period,dm0,output=1):
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
if info0['mode']!='template':
	data0=d0.period_scrunch()[chanstart0:chanend0,:,0]
	if not args.freqtoa:
		if not args.dm_corr and chanend0-chanstart0>1:
			sys.stdout=open(os.devnull,'w')
			psr=pm.psr_timing(pr.psr(info0['psr_par']),te.times(te.time(info0['stt_time']+info0['length']/86400,0)),(freq_start0+freq_end0)/2)
			sys.stdout=sys.__stdout__
			freq_real0=np.linspace(freq_start0,freq_end0,nchan0+1)[:-1]*psr.vchange.mean()
			if 'best_dm' in info0.keys():
				ddm0=info0['best_dm'][0]-info0['dm']
				fftdata0=fft.rfft(data0,axis=1)
				tmp=np.shape(fftdata0)[-1]
				const=(1/freq_real0**2*pm.dm_const/info0['period']*np.pi*2.0).repeat(tmp).reshape(-1,tmp)*np.arange(tmp)
				data0=shift(fftdata0,const*ddm0)
			else:
				data0[rchan],ddm0,ddm0err=dmcor(data0,freq_real0,rchan,info0['period'],info0['dm'])
	data_tmp=data0[rchan]
	if args.norm:
		command.append('-n')
		data_tmp-=data_tmp.mean(1).reshape(-1,1)
		data_tmp/=data_tmp.std(1).reshape(-1,1)
	if args.freqtoa:
		tpdata0=data_tmp
		tmp=tpdata0.sum(0)-af.baseline(tpdata0.sum(0))
	else:
		tpdata0=data_tmp.sum(0)
		tmp=tpdata0-af.baseline(tpdata0)
else:
	if not args.freqtoa:
		data0=d0.chan_scrunch()[:,:,0]
		if nsub0==1:
			tpdata0=data0[0]
			tmp=tpdata0-af.baseline(tpdata0)
		else:
			if d0.read_shape()[0]>1:
				parser.error("The multi-component freq-domain template cannot be adopted for non-freq-domain ToA determination.")
			tpdata0=data0
			tmp=tpdata0[0]-af.baseline(tpdata0[0])
	else:
		data0=d0.read_data()[:,:,:,0]
		if len(data0)==1: parser.error("The freq-domain ToA cannot be obtained for template with only one frequency channel.")
		if nsub0==1:
			tpdata0=data0[:,0]
			tmp=tpdata0.sum(0)-af.baseline(tpdata0.sum(0))
		else:
			tpdata0=data0
			tmp=tpdata0[:,0].sum(0)-af.baseline(tpdata0[:,0].sum(0))
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
def poa(tpdata0,tpdata):
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
	errbinnum=np.min([int(nb/6),20])
	sr0=np.std(f0.real[-errbinnum:])
	si0=np.std(f0.imag[-errbinnum:])
	sr=np.std(f.real[-errbinnum:])
	si=np.std(f.imag[-errbinnum:])
	df=f0/f
	err=np.sqrt((sr**2+si**2)/np.abs(f)**2+(sr0**2+si0**2)/np.abs(f0)**2)
	ang=np.angle(df)
	fitnum=lnumber
	popt,pcov=so.curve_fit(lin,np.arange(1,fitnum),ang[1:fitnum],p0=[0.0],sigma=err[1:fitnum])
	dt=popt[0]/(2*np.pi)-tmpnum/((nb-1)*2)
	dterr=pcov[0,0]**0.5/(2*np.pi)
	return [dt,dterr]
#
def coa(tpdata0,tpdata):
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
	errbinnum=np.min([int(nb/6),20])
	sr0=np.std(f0.real[-errbinnum:])
	si0=np.std(f0.imag[-errbinnum:])
	sr=np.std(f.real[-errbinnum:])
	si=np.std(f.imag[-errbinnum:])
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
result=np.zeros([nsub_new.sum(),10])
cumsub=np.append(0,np.cumsum(nsub_new)[:-1])
discard=[]
if args.algorithm=='pgs':
	tfunc=poa
elif args.algorithm=='corr':
	tfunc=coa
#
def dmdt(freq,dm,c):
	return 1/freq**2*pm.dm_const/period0*dm+c
#
def toafunc(tpdata0,tpdata,freq=0,comp=[],kv=0):
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
					dp_tmp[ind],dperr_tmp[ind]=tfunc(tp0[i]+kvalue[ind]*comp[i],tp[i])
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
	sys.stdout=open(os.devnull,'w')
	psr0=pr.psr(info['psr_par'])
	sys.stdout=sys.__stdout__
	period0=info['period']
	#
	if 'compressed' in info.keys():
		nchan=info['nchan_new']
		nbin=info['nbin_new']
		nperiod=info['nsub_new']
		npol=info['npol_new']
	else:
		nchan=info['nchan']
		nbin=info['nbin']
		nperiod=info['nsub']
		npol=info['npol']
	#
	if args.zap_file:
		if 'zchan' in info.keys():
			zchan_tmp=np.array(list(set(map(int,info['zchan'])).union(zchan)))
	elif 'zchan' in info.keys():
		zchan_tmp=np.int32(info['zchan'])
	else:
		zchan_tmp=np.int32([])
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
	freq_start,freq_end=info['freq_start'],info['freq_end']
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
		phase0=info['phase0']
		sub_nperiod=info['sub_nperiod']*np.ones(nperiod,dtype=np.float64)
		sub_nperiod[-1]=info['sub_nperiod_last']
		middle=sub_nperiod.cumsum()-sub_nperiod/2
		time0=info['length']*(nc.chebpts1(20)+1)/2
		psr=pm.psr_timing(psr0,te.times(te.time(info['stt_date']*np.ones(20,dtype=np.float64),info['stt_sec']+time0)),(freq_start+freq_end)/2)
		phase1=psr.phase
		chebc=nc.chebfit(time0,phase1.integer-phase1.integer[0]+phase1.offset,12)
		chebd=nc.chebder(chebc)
		time0=np.linspace(0,info['length'],nperiod)
		psr=pm.psr_timing(psr0,te.times(te.time(info['stt_date']*np.ones(nperiod,dtype=np.float64),info['stt_sec']+time0)),(freq_start+freq_end)/2)
		phase1=psr.phase
		phase_start=phase1.integer[0]
		middle_time=np.interp(middle,phase1.integer-phase0+phase1.offset,time0)[sub_s:sub_e]
		psr1=pm.psr_timing(psr0,te.times(te.time(info['stt_date']*np.ones(nsub_new[k],dtype=np.float64),info['stt_sec']+middle_time)),(freq_start+freq_end)/2)
		middle_phase=psr1.phase
		data1=d.period_scrunch(sub_s,sub_e)[chanstart:chanend,:,0]
		freq_real=(np.linspace(freq_start,freq_end,nchan+1)[:-1]+channel_width/2)*psr.vchange.mean()
		if not args.freqtoa:
			if not args.dm_corr:
				if 'best_dm' in info.keys():
					ddm=info['best_dm'][0]-info['dm']
					ddmerr=info['best_dm'][1]
				else:
					ddm,ddmerr=dmcor(data1,freq_real,rchan,period0,info['dm'],output=0)
				#print('The relative DM from the template file to data file is '+str(ddm-ddm0))
				dm_new=ddm+info['dm']
			else:
				dm_new=info['dm']
				ddmerr=0
		if info0['mode']=='template' and nsub0>1:
			krange=info0['krange']
			if not args.freqtoa:
				kvalue=np.float64(krange.split(','))
			else:
				kvalue=np.float64(list(map(lambda x: x.split(','),krange)))
		for s in np.arange(nsub_new[k]):
			data=d.read_period(s+sub_s)[chanstart:chanend,:,0]
			if np.any(np.isnan(data)) or np.any(np.isinf(data)) or np.all(data==0):
				discard.append([filelist[k],s])
				continue
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
			else:
				tpdata=data_tmp
			middle_int=middle_phase.integer[s]-phase_start
			middle_offs=middle_phase.offset[s]
			if info0['mode']=='template' and nsub0>1:
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
					dp,dpe,ddm,ddmerr=toafunc(tpdata0,tpdata,freq_real)
					dm_new=ddm+info['dm']
				else: dp,dpe=toafunc(tpdata0,tpdata)
			dp0=dp+np.round(middle_offs-dp)
			roots=nc.chebroots(chebc-([dp0+middle_int]+[0]*12))
			roots=np.real(roots[np.isreal(roots)])
			root=roots[np.argmin(np.abs(roots-middle_int*period0))]
			toa=te.time(info['stt_date'],root+info['stt_sec'])
			period=1/nc.chebval(root,chebd)
			toae=dpe*period
			if output=='screen':
				print(toa.date[0],toa.second[0],dp,dpe,freq_start,freq_end,dm_new,period)
			else:
				result[cumsub[k]+s-len(discard)]=[toa.date[0],toa.second[0],toae,dp,dpe,freq_start,freq_end,dm_new,ddmerr,period]
	else:
		phase0=info['phase0']
		sub_nperiod=info['sub_nperiod']*np.ones(nperiod,dtype=np.float64)
		sub_nperiod[-1]=info['sub_nperiod_last']
		sub_nperiod_cumsum=sub_nperiod.cumsum()
		if sub_s==0: middle=sub_nperiod_cumsum[sub_e-1]/2
		else: middle=(sub_nperiod_cumsum[sub_s-1]+sub_nperiod_cumsum[sub_e-1])/2
		time0=np.linspace(0,info['length'],12)
		phase1=pm.psr_timing(psr0,te.times(te.time(info['stt_date']*np.ones(12,dtype=np.float64),info['stt_sec']+time0)),(freq_start+freq_end)/2).phase
		chebc=nc.chebfit(phase1.integer-phase0+phase1.offset,time0,7)
		chebd=nc.chebder(chebc)
		middle_time=nc.chebval(middle,chebc)
		psr1=pm.psr_timing(psr0,te.times(te.time(info['stt_date'],info['stt_sec']+middle_time)),(freq_start+freq_end)/2)
		data=d.period_scrunch(sub_s,sub_e)[chanstart:chanend,:,0]
		if np.any(np.isnan(data)) or np.any(np.isinf(data)) or np.all(data==0):
			discard.append(filelist[k])
			continue
		freq_real=(np.linspace(freq_start,freq_end,nchan+1)[:-1]+channel_width/2)*psr.vchange.mean()
		if not args.freqtoa:
			if not args.dm_corr:
				if 'best_dm' in info.keys():
					ddm=info['best_dm'][0]-info['dm']
					ddmerr=info['best_dm'][1]
					fftdata=fft.rfft(data,axis=1)
					tmp=np.shape(fftdata)[-1]
					const=(1/freq_real**2*pm.dm_const/period0*np.pi*2.0).repeat(tmp).reshape(-1,tmp)*np.arange(tmp)
					data=shift(fftdata,const*ddm)[rchan]
				else:
					data,ddm,ddmerr=dmcor(data,freq_real,rchan,period0,info['dm'])
				dm_new=ddm+info['dm']
			else:
				dm_new=info['dm']
				ddmerr=0
		if args.norm:
			data-=data.mean(1).reshape(-1,1)
			data/=data.std(1).reshape(-1,1)
		if not args.freqtoa:
			tpdata=data.mean(0)
		else:
			tpdata=data
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
				dm_new=ddm+info['dm']
			else: dp,dpe=toafunc(tpdata0,tpdata)
		middle_int,middle_offs=np.divmod(middle,1)
		dp0=dp+np.round(middle_offs-dp)
		root=nc.chebval(dp0+middle_int,chebc)
		toa=te.time(info['stt_date'],root+info['stt_sec'])
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
		print('The subint'+(len(discard)>1)*'s'+' '+', '.join(list(map(lambda x: 'the '+str(x[1])+'th subint of the file '+x[0],discard)))+' are discarded in calculating the ToA.')
	result=result[:(-len(discard))]
#
if output=='ld':
	d1=ld.ld(name+'.ld')
	d1.write_shape([1,nsub_new.sum()-len(discard),10,1])
	d1.write_chan(result,0)
	toainfo={'psr_name':psrname,'history':[command],'file_time':[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())],'mode':'ToA','method':args.algorithm}
	d1.write_info(toainfo)
elif output=='txt':
	fout=open(name,'w')
	fout.write(psrname+' ToA\n')
	for i in result:
		fout.write('{:28s} {:10s} {:10.2f} {:10.2f} {:13f} {:18f}'.format(str(int(i[0]))+str(i[1]/86400)[1:],"%.3e"%(i[2]/86400),i[5],i[6],i[7],i[9])+'\n')
	fout.close()
#

