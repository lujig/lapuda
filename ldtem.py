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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
wn.filterwarnings('ignore')
#
version='JigLu_20201202'
parser=ap.ArgumentParser(prog='ldtem',description='Generate the profile template with multi-profiles.',epilog='Ver '+version)
parser.add_argument('-v','--version',action='version',version=version)
parser.add_argument("filename",nargs='+',help="input ld file or files")
parser.add_argument('-T','--tscrunch',action='store_true',default=False,dest='tscrunch',help='time scrunch to one subint for each file')
parser.add_argument('-r','--frequency_range',default=0,dest='freqrange',help='calculate in the frequency range (FREQ0,FREQ1)')
parser.add_argument('-s','--subint_range',default=0,dest='subint_range',help='calculate in the subint range (SUBINT0,SUBINT1)')
parser.add_argument('-z',"--zap",dest="zap_file",default=0,help="file recording zap channels")
parser.add_argument('-o',"--output",dest="output",default="template",help="outputfile name")
parser.add_argument('-d',action='store_true',default=False,dest='dm_corr',help='the progress will not correcting the DM deviation before calculating rotating phase')
parser.add_argument('-l',"--linear_number",dest="lnumber",type=np.int8,default=0,help="the number of frequency-domain points for linear fitting")
parser.add_argument('-b',"--nbin",dest="nbin",type=np.int16,default=256,help="the bin number of output profile")
parser.add_argument('-a',action='store_true',default=False,dest='auto',help='do not discard the low-rms data')
parser.add_argument('-m','--red',action='store_true',default=False,dest='red',help='use only red component of the profile as template')
parser.add_argument('-c','--component',action='store_true',default=False,dest='component',help='the template has multi-components')
parser.add_argument('-f','--freqtem',action='store_true',default=False,dest='freqtem',help='generate 2-D freqdomain template')
args=(parser.parse_args())
command=['ldtem.py']
#
if args.zap_file:
	command.append('-z')
	if not os.path.isfile(args.zap_file):
		parser.error('The zap channel file is invalid.')
	zchan=np.loadtxt(args.zap_file,dtype=np.int32)
	if np.min(zchan)<0:
		parser.error('The zapped channel number is overrange.')
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
if args.freqrange:
	command.append('-r '+args.freqrange)
	freq_s,freq_e=np.float64(args.freqrange.split(','))
	if freq_s>freq_e:
		parser.error("Starting frequency larger than ending frequency.")
else:
	freq_s,freq_e=1,1e6
	freq_s_tmp,freq_e_tmp=freq_s,freq_e
#
name=args.output
if not name:
	output='screen'
else:
	name_tmp='    '+name
	if name_tmp[-3:]=='.ld':
		output='ld'
		name=name[:-3]
	if os.path.isfile(name+'.ld'):
		parser.error('The name of output file already existed. Please provide a new name.')
#
filelist=args.filename
filenum=len(filelist)
nsub_new=[]
psrname=''
dm=0
def ld_check(fname,filetype='Ld file',notfirst=True):
	global freq_s,freq_e,psrname,dm,freq_s_tmp,freq_e_tmp,nchan0
	if not os.path.isfile(fname):
		parser.error(filetype+' name '+fname+' '+'is invalid.')
	try:
		f=ld.ld(filelist[i])
		finfo=f.read_info()
	except:
		parser.error(filetype+' '+fname+' is invalid.')
	if notfirst:
		tmpname=pr.psr(finfo['psr_par']).name
		if psrname!=tmpname:
			parser.error('The pulsar recorded in '+fname+' is different from the template.')
	else:
		psr=pr.psr(finfo['psr_par'])
		psrname=psr.name
		dm=psr.dm
	#
	if 'compressed' in finfo.keys():
		nchan=finfo['nchan_new']
		nperiod=finfo['nsub_new']
	else:
		nchan=finfo['nchan']
		nperiod=finfo['nsub']
	#
	if args.zap_file:
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
	freq=(freq_start+freq_end)/2.0
	channel_width=(freq_end-freq_start)/nchan
	if notfirst:
		if args.freqtem and not args.freqrange:
			if freq_start!=freq_s_tmp or freq_end!=freq_e_tmp: parser.error("The frequency ranges of the input data are different.")
		freq_s_tmp=min(freq_start,freq_s_tmp)
		freq_e_tmp=max(freq_end,freq_e_tmp)
	else:
		freq_s_tmp=freq_start
		freq_e_tmp=freq_end
	if args.freqrange:
		if freq_s<freq_start or freq_e>freq_end: parser.error("Input frequency is overrange.")
	if args.freqtem:
		chanstart,chanend=np.int16(np.round((np.array([max(freq_s,freq_start),min(freq_e,freq_end)])-freq)/channel_width+0.5*nchan))
		nchan_new=chanend-chanstart
		if notfirst:
			if nchan_new!=nchan0: parser.error("The freq-domain template cannot be constructed for data with different frequency parameters.")				
		else:
			nchan0=nchan_new
#
sys.stdout=open(os.devnull,'w')
for i in np.arange(filenum):
	ld_check(filelist[i],notfirst=i)
sys.stdout=sys.__stdout__
if not args.freqrange:
	freq_s=freq_s_tmp
	freq_e=freq_e_tmp
#
nsub_new=np.array(nsub_new)
#
name=args.output
name_tmp='    '+name
if name_tmp[-3:]=='.ld':
	name=name[:-3]
if os.path.isfile(name+'.ld'):
	parser.error('The name of output file already existed. Please provide a new name.')
#
def shift(y,x):
	fftdata=fft.rfft(y,axis=1)
	tmp=int(args.nbin/2+1)
	ns,nb=fftdata.shape
	if nb>tmp: fftdata=fftdata[:,:tmp]
	elif nb<tmp: fftdata=np.concatenate((fftdata,np.zeros([ns,tmp-nb])),axis=1)
	if x is not int(0):
		fftdata=fftdata*np.exp(x.repeat(tmp).reshape(-1,tmp)*np.arange(tmp)*1j)
	fftr=fft.irfft(fftdata)
	return fftr
#
if args.dm_corr: 
	command.append('-d ')
dm_zone=np.max([0.1,dm/100])
dm_zone=np.min([0.5,dm_zone])
#
def dmcor(data,freq,rchan,period,output=1):
	data=data[rchan]
	freq=freq[rchan]
	fftdata=fft.rfft(data,axis=1)
	tmp=np.shape(fftdata)[-1]
	const=(1/freq**2*pm.dm_const/period*np.pi*2.0).repeat(tmp).reshape(-1,tmp)*np.arange(tmp)
	ddm,ddmerr=af.dmdet(fftdata,const,0,dm_zone,9,prec=1e-4)
	if output==1:
		const=(1/freq**2*pm.dm_const/period*np.pi*2.0)*ddm
		data=shift(data[rchan],const)
		return data,ddm,ddmerr
	else:
		return ddm,ddmerr
#
if args.lnumber:
	command.append('-l '+str(args.lnumber))
#
if args.auto:
	command.append('-a ')
#
if args.red:
	command.append('-m ')
	if args.component:
		parser.error('The multi-component mode doesnot support red component template.')
#
if args.component:
	command.append('-c')
#
if args.freqtem:
	command.append('-f')
#
def lin(x,k):
	return k*x
#
command=' '.join(command)
#
if args.freqtem:
	result0=np.zeros([nchan0,nsub_new.sum(),args.nbin])
else:
	result0=np.zeros([1,nsub_new.sum(),args.nbin])
cumsub=np.append(0,np.cumsum(nsub_new)[:-1])
krange=[]
discard=0
for k in np.arange(filenum):
	d=ld.ld(filelist[k])
	info=d.read_info()
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
			zchan_tmp=np.array(list(set(info['zchan']).union(zchan)))
	elif 'zchan' in info.keys():
		zchan_tmp=info['zchan']
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
	chanstart,chanend=np.int16(np.round((np.array([max(freq_s,freq_start),min(freq_e,freq_end)])-freq)/channel_width+0.5*nchan))
	#
	nchan_new=chanend-chanstart
	if nchan==1:
		rchan=[0]
	else:
		rchan=np.array(list(set(range(chanstart,chanend))-set(list(zchan_tmp))))-chanstart
	if not args.freqtem:
		if not args.dm_corr:
			data1=d.period_scrunch(sub_s,sub_e)[chanstart:chanend,:,0]
			freq_real=np.linspace(freq_start,freq_end,nchan+1)[:-1]
			data1-=data1.mean(1).reshape(-1,1)
			data1/=data1.max(1).reshape(-1,1)
			if np.any(np.isnan(data1)) or np.any(np.isinf(data1)):
				discard+=1
				continue
			if 'best_dm' in info.keys():
				ddm=info['best_dm'][0]-info['dm']
			else:
				ddm,ddmerr=dmcor(data1,freq_real,rchan,info['period'],output=0)
			const=(1/freq_real**2*pm.dm_const/info['period']*np.pi*2.0)*ddm
		else:
			data1=d.period_scrunch(sub_s,sub_e)[chanstart:chanend,:,0]
			if np.any(np.isnan(data1)) or np.any(np.isinf(data1)):
				discard+=1
				continue
			const=0
	if not args.tscrunch:
		for s in np.arange(nsub_new[k]):
			data=d.read_period(s+sub_s)[chanstart:chanend,:,0]
			if np.any(np.isnan(data)) or np.any(np.isinf(data)):
				discard+=1
				continue
			if not args.freqtem:
				data=shift(data,const)[rchan].mean(0)
				if nbin!=args.nbin:
					fdata=fft.rfft(data)
					if len(data)>args.nbin: fdata=fdata[:int(args.nbin/2+1)]
					else: fdata=np.append(fdata,np.zeros(int(args.nbin/2+1)-len(fdata)))
					result0[0,cumsub[k]+s-discard]=fft.irfft(fdata)
				else:
					result0[0,cumsub[k]+s-discard]=data
			else:
				data=d.read_period(s+sub_s)[chanstart:chanend,:,0]
				if nbin!=args.nbin:
					fdata=fft.rfft(data,axis=1)
					if np.shape(data)[1]>args.nbin: fdata=fdata[:,:int(args.nbin/2+1)]
					else: fdata=np.concatenate((fdata,np.zeros([nchan_new,int(args.nbin/2+1)-len(fdata)])),axis=1)
					result0[:,cumsub[k]+s-discard]=fft.irfft(fdata,axis=1)
				else:
					result0[:,cumsub[k]+s-discard]=data
	else:
		if not args.freqtem:
			data=shift(data1,const)[rchan].mean(0)
			if nbin!=args.nbin:
				fdata=fft.rfft(data)
				if len(data)>args.nbin: fdata=fdata[:int(args.nbin/2+1)]
				else: fdata=np.append(fdata,np.zeros(int(args.nbin/2+1)-len(fdata)))
				result0[0,cumsub[k]-discard]=fft.irfft(fdata)
			else:
				result0[0,cumsub[k]-discard]=data
		else:
			data=d.period_scrunch(sub_s,sub_e)[chanstart:chanend,:,0]
			if np.any(np.isnan(data)) or np.any(np.isinf(data)):
				discard+=1
				continue
			if nbin!=args.nbin:
				fdata=fft.rfft(data,axis=1)
				if np.shape(data)[1]>args.nbin: fdata=fdata[:,:int(args.nbin/2+1)]
				else: fdata=np.concatenate((fdata,np.zeros([nchan_new,int(args.nbin/2+1)-len(fdata)])),axis=1)
				result0[:,cumsub[k]-discard]=fft.irfft(fdata,axis=1)
			else:
				result0[:,cumsub[k]-discard]=data
#
if discard>0:
	result0=result0[:,:(-discard)]
nchan_res=result0.shape[0]
if args.component:
	prof=np.zeros([nchan_res,2,args.nbin])
else:
	prof=np.zeros([nchan_res,args.nbin])
#

for s in np.arange(nchan_res):
	result=result0[s]
	if np.all(result.mean(1)==0): continue
	result-=result.mean(1).reshape(-1,1)
	result/=np.sqrt((result**2).sum(1)).reshape(-1,1)
	jj=(result.sum(1)**2)>=0
	result=result[jj]
	fres=fft.rfft(result,axis=1)
	if not args.auto:
		fabs=np.abs(fres)
		fabss=fabs.sum(0)
		ntmp=int(np.round(fabss.sum()/fabss.max()))
		ntmp=max(ntmp,20)
		tmp=fabs[:,1:ntmp]
		l=(tmp-tmp.mean(0))/tmp.std(0)
		jj=np.abs(l).max(1)<2
		result=result[jj]
		fres=fres[jj]
		tmp=result.max(1)-result.min(1)
		l=(tmp-tmp.mean(0))/tmp.std(0)
	#
	fabs=np.abs(fres)
	if args.lnumber:
		command.append('-l '+str(args.lnumber))
		lnumber=args.lnumber
		if lnumber>100 or lnumber<=2:
			parser.error('The number of frequency-domain points for linear fitting is invalid.')
	else:
		fabss=fabs.sum(0)
		lnumber=int(np.round(fabss.sum()/fabss.max()))
		lnumber=max(4,lnumber)
	#
	knumber=lnumber*2
	nsub,nbin=result.shape
	dt=np.zeros(nsub)
	angerr=np.zeros([nsub,knumber-1])
	nb=int(nbin/2+1)
	errbinnum=np.min([int(nb/6),20])
	for i in np.arange(nsub):
		f=fres[-1]
		tmpnum=np.argmax(fft.irfft(fres[i]*f.conj()))
		d0=np.append(result[i,tmpnum:],result[i,:tmpnum])
		f0=fft.rfft(d0)
		df=f0/f
		ang=np.angle(df)
		sr0=np.std(f0.real[-errbinnum:])
		si0=np.std(f0.imag[-errbinnum:])
		sr=np.std(f.real[-errbinnum:])
		si=np.std(f.imag[-errbinnum:])
		err=np.sqrt((sr**2+si**2)/fabs[-1]**2+(sr0**2+si0**2)/fabs[i]**2)
		popt,pcov=so.curve_fit(lin,np.arange(1,lnumber),ang[1:lnumber],p0=[0.0],sigma=err[1:lnumber])
		dt[i]=popt[0]/(2*np.pi)-tmpnum/((nb-1)*2)
		angerr[i]=err[1:knumber]
	#
	result[:-1]=shift(result[:-1],-dt[:-1]*2*np.pi)
	#
	if args.red:
		fres=fft.rfft(result,axis=1)
		fang=np.angle(fres)[:,1:knumber]
		fang0=fang[0]*1
		fang-=fang0
		fang[fang>np.pi]-=np.pi*2
		fang[fang<-np.pi]+=np.pi*2
		sum0=fang.mean(0).sum()
		def angle(para):
			ang0=para[:(knumber-2)]
			ang0=np.append(ang0,sum0-ang0.sum())
			k=para[(knumber-2):]
			angc=ang0+np.arange(1,knumber)*k.reshape(-1,1)-fang
			return (angc/angerr).reshape(-1)
		#
		p0=np.zeros(knumber-2+nsub)
		res=so.leastsq(angle,x0=p0,full_output=True)
		popt=res[0]
		if not args.auto:
			k=popt[(knumber-2):]*1
			k-=k.mean()
			k/=k.std()
			jj=np.abs(k)<1
		else:
			jj=np.ones(nsub,dtype=np.bool)
		ang1=popt[:(knumber-2)]
		ang1=np.append(ang1,sum0-ang1.sum())
		weight=1/(np.var(fres[:,-errbinnum:].real,1)+np.var(fres[:,-errbinnum:].imag,1))[jj]
		abs0=(np.abs(fres[jj])*weight.reshape(-1,1)).sum(0)/weight.sum()
		prof0=abs0[1:knumber]*np.exp(1j*(ang1+fang0))
		prof[s]=fft.irfft(np.concatenate(([0],prof0,np.zeros(nb-knumber))))
	else:
		fres=fft.rfft(result,axis=1)
		weight=1/(np.var(fres[:,-errbinnum:].real,1)+np.var(fres[:,-errbinnum:].imag,1))
		prof0=(result*weight.reshape(-1,1)).sum(0)/weight.sum()
		f=fft.rfft(prof0)
		for i in np.arange(nsub):
			tmpnum=np.argmax(fft.irfft(fres[i]*f.conj()))
			d0=np.append(result[i,tmpnum:],result[i,:tmpnum])
			f0=fft.rfft(d0)
			df=f0/f
			ang=np.angle(df)
			sr0=np.std(f0.real[-errbinnum:])
			si0=np.std(f0.imag[-errbinnum:])
			sr=np.std(f.real[-errbinnum:])
			si=np.std(f.imag[-errbinnum:])
			err=np.sqrt((sr**2+si**2)/fabs[-1]**2+(sr0**2+si0**2)/fabs[i]**2)
			popt,pcov=so.curve_fit(lin,np.arange(1,lnumber),ang[1:lnumber],p0=[0.0],sigma=err[1:lnumber])
			dt[i]=popt[0]/(2*np.pi)-tmpnum/((nb-1)*2)
			angerr[i]=err[1:knumber]
		#
		result[:-1]=shift(result[:-1],-dt[:-1]*2*np.pi)
		if args.component:
			baseline,pos=af.baseline(result.mean(0),pos=True)
			result-=result[:,pos:(pos+int(nbin/10))].mean(1).reshape(-1,1)
			result/=result.max(1).reshape(-1,1)
			nc=2
			pca=PCA(n_components=nc)
			coeff=pca.fit_transform(result)[:,0]
			x1=pca.components_[0]
			#mean=((res1.reshape(-1,1)@x1.reshape(1,-1)).T+pca.mean_.reshape(-1,1)).mean(1)
			mean=pca.mean_
			prof[s]=np.array([mean,x1])
			kmax,kmin=np.max(coeff),np.min(coeff)
			krange.append(str(kmin*1.3-kmax*0.3)+','+str(kmax*1.3-kmin*0.3))
		else:
			prof[s]=(result*weight.reshape(-1,1)).sum(0)/weight.sum()
#
if args.freqtem:
	fprof=fft.rfft(prof,axis=-1)
	if args.component: 
		f=fprof[0,0]
		fabs=np.abs(fprof[:,0])		
	else:
		f=fprof[0]
		fabs=np.abs(fprof)
	sr=np.std(f.real[-errbinnum:])
	si=np.std(f.imag[-errbinnum:])
	dt=np.zeros(nchan_res-1)
	for i in np.arange(1,nchan_res):
		tmpnum=np.argmax(fft.irfft(fprof[i]*f.conj()))
		if args.component: d0=np.append(prof[i,0,tmpnum:],prof[i,0,:tmpnum])
		else: d0=np.append(prof[i,tmpnum:],prof[i,:tmpnum])
		f0=fft.rfft(d0)
		df=f0/f
		ang=np.angle(df)
		sr0=np.std(f0.real[-errbinnum:])
		si0=np.std(f0.imag[-errbinnum:])
		err=np.sqrt((sr**2+si**2)/fabs[0]**2+(sr0**2+si0**2)/fabs[i]**2)
		popt,pcov=so.curve_fit(lin,np.arange(1,lnumber),ang[1:lnumber],p0=[0.0],sigma=err[1:lnumber])
		dt[i-1]=popt[0]/(2*np.pi)-tmpnum/((nb-1)*2)
	#
	if args.component:
		prof[1:,0]=shift(prof[1:,0],-dt*2*np.pi)
		prof[1:,1]=shift(prof[1:,1],-dt*2*np.pi)
	else: prof[1:]=shift(prof[1:],-dt*2*np.pi)

if np.sum(prof**2)==0: parser.error('Unexpected error. The produced profile is zero in every phase bin.')
info={'mode':'template','nchan_new':int(nchan_res), 'nbin_new':int(args.nbin), 'npol_new':1, 'file_time':[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())], 'pol_type':'I','compressed':True,'history':[command],'psr_name':psrname,'freq_start':freq_s,'freq_end':freq_e,'length':1}
do=ld.ld(name+'.ld')

if args.component:
	do.write_shape([nchan_res,2,args.nbin,1])
	info['krange']=krange
	info['nsub_new']=2
else:
	do.write_shape([nchan_res,1,args.nbin,1])
	info['nsub_new']=1
for i in np.arange(nchan_res):
	do.write_chan(prof[i],i)
do.write_info(info)
#
