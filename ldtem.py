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
from sklearn.decomposition import PCA
wn.filterwarnings('ignore')
#
version='JigLu_20201202'
parser=ap.ArgumentParser(prog='ldtem',description='Generate the profile template with multi-profiles.',epilog='Ver '+version)
parser.add_argument('-v','--version',action='version',version=version)
parser.add_argument("filename",nargs='+',help="input ld file or files")
parser.add_argument('-T','--tscrunch',action='store_true',default=False,dest='tscrunch',help='time scrunch to one subint for each file')
parser.add_argument('--fr','--frequency_range',default=0,dest='freqrange',help='calculate in the frequency range (FREQ0,FREQ1)')
parser.add_argument('--sr','--subint_range',default=0,dest='subint_range',help='calculate in the subint range (SUBINT0,SUBINT1)')
parser.add_argument('-z',"--zap",dest="zap_file",default=0,help="file recording zap channels")
parser.add_argument('-o',"--output",dest="output",default="template",help="outputfile name")
parser.add_argument('-d',action='store_true',default=False,dest='dm_corr',help='the progress will not correcting the DM deviation before calculating rotating phase')
parser.add_argument('-l',"--linear_number",dest="lnumber",type=np.int8,default=0,help="the number of frequency-domain points for linear fitting")
parser.add_argument('-b',"--nbin",dest="nbin",type=np.int16,default=256,help="the bin number of output profile")
parser.add_argument('-a',action='store_true',default=False,dest='auto',help='do not discard the low-rms data')
parser.add_argument('-m','--red',action='store_true',default=False,dest='red',help='use only red component of the profile as template')
parser.add_argument('-c','--component',action='store_true',default=False,dest='component',help='the template has multi-components')
parser.add_argument('--freqtem',action='store_true',default=False,dest='freqtem',help='generate 2-D freqdomain template')
parser.add_argument('-p','--peak_fit',action='store_true',default=False,dest='peakfit',help='fit the template with several peaks')
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
	command.append('--sr '+args.subint_range)
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
	command.append('--fr '+args.freqrange)
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
	global freq_s,freq_e,psrname,dm,freq_s_tmp,freq_e_tmp,nchan0,psr
	if not os.path.isfile(fname):
		parser.error(filetype+' name '+fname+' '+'is invalid.')
	try:
		f=ld.ld(filelist[i])
		finfo=f.read_info()
	except:
		parser.error(filetype+' '+fname+' is invalid.')
	if notfirst:
		tmpname=pr.psr(finfo['pulsar_info']['psr_par'],warning=False).name
		if psrname!=tmpname:
			parser.error('The pulsar recorded in '+fname+' is different from the template.')
	else:
		psr=pr.psr(finfo['pulsar_info']['psr_par'],warning=False)
		psrname=psr.name
		dm=psr.dm
	#
	nchan=finfo['data_info']['nchan']
	nperiod=finfo['data_info']['nsub']
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
	freq_start,freq_end=finfo['data_info']['freq_start'],finfo['data_info']['freq_end']
	freq=(freq_start+freq_end)/2.0
	channel_width=(freq_end-freq_start)/nchan
	if args.freqtem and not args.freqrange:
		if freq_start!=freq_s_tmp or freq_end!=freq_e_tmp: parser.error("The frequency ranges of the input data are different.")
	freq_s_tmp=min(freq_start,freq_s_tmp)
	freq_e_tmp=max(freq_end,freq_e_tmp)
	if args.freqrange:
		if freq_s<freq_start or freq_e>freq_end: parser.error("Input frequency is overrange.")
	if args.freqtem:
		chanstart,chanend=np.int16(np.round((np.array([max(freq_s,freq_start),min(freq_e,freq_end)])-freq_start)/channel_width))
		nchan_new=chanend-chanstart
		if notfirst:
			if nchan_new!=nchan0: parser.error("The freq-domain template cannot be constructed for data with different frequency parameters.")				
		else:
			nchan0=nchan_new
#
for i in np.arange(filenum):
	ld_check(filelist[i],notfirst=i)
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
def dmcor(data,freq,rchan,period,output=1):	# correct the DM before extracting the template
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
	command.append('-a')
#
if args.red:
	command.append('-m')
	if args.component:
		parser.error('The multi-component mode doesnot support red component template.')
#
if args.peakfit:
	command.append('-p')
#
if args.component:
	if args.peakfit: parser.error('The multi-component mode doesnot support peak-fit.')
	command.append('-c')
#
if args.freqtem:
	if args.peakfit: parser.error('The multi-frequency mode doesnot support peak-fit.')
	command.append('--freqtem')
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
for k in np.arange(filenum):	# read all data in files
	d=ld.ld(filelist[k])
	info=d.read_info()
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
	freq_start0,freq_end0=info['data_info']['freq_start'],info['data_info']['freq_end']
	freq=(freq_start0+freq_end0)/2.0
	channel_width=(freq_end0-freq_start0)/nchan
	chanstart,chanend=np.int16(np.round((np.array([max(freq_s,freq_start0),min(freq_e,freq_end0)])-freq_start0)/channel_width))
	freq_start,freq_end=np.array([chanstart,chanend])*channel_width+freq_start0
	#
	nchan_new=chanend-chanstart
	if nchan==1:
		rchan=[0]
	else:
		rchan=np.array(list(set(range(chanstart,chanend))-set(list(zchan_tmp))))-chanstart
	if not args.freqtem:
		if not args.dm_corr:
			data1=d.period_scrunch(sub_s,sub_e)[chanstart:chanend,:,0]*np.asarray(info['data_info']['chan_weight'])[chanstart:chanend].reshape(-1,1)
			freq_real=np.arange(freq_start,freq_end,channel_width)+channel_width*0.5
			data1-=data1.mean(1).reshape(-1,1)
			data1/=data1.max(1).reshape(-1,1)
			if np.any(np.isnan(data1)) or np.any(np.isinf(data1)):
				discard+=1
				continue
			if 'additional_info' in info.keys():
				if 'best_dm' in info['additional_info'].keys(): ddm=info['additional_info']['best_dm'][0]-info['data_info']['dm']
				else: ddm,ddmerr=dmcor(data1,freq_real,rchan,info['data_info']['period'],output=0)
			else:
				ddm,ddmerr=dmcor(data1,freq_real,rchan,info['data_info']['period'],output=0)
			const=(1/freq_real**2*pm.dm_const/info['data_info']['period']*np.pi*2.0)*ddm
		else:
			data1=d.period_scrunch(sub_s,sub_e)[chanstart:chanend,:,0]*np.asarray(info['data_info']['chan_weight'])[chanstart:chanend].reshape(-1,1)
			if np.any(np.isnan(data1)) or np.any(np.isinf(data1)):
				discard+=1
				continue
			const=0
	if not args.tscrunch:
		for s in np.arange(nsub_new[k]):
			data=d.read_period(s+sub_s)[chanstart:chanend,:,0]*np.asarray(info['data_info']['chan_weight'])[chanstart:chanend].reshape(-1,1)
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
			data=d.period_scrunch(sub_s,sub_e)[chanstart:chanend,:,0]*np.asarray(info['data_info']['chan_weight'])[chanstart:chanend].reshape(-1,1)
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
	import matplotlib.pyplot as plt
	#print(result.std(1))
	result-=result.mean(1).reshape(-1,1)
	jj=((result**2).sum(1))>0
	result=result[jj]
	result/=np.sqrt((result**2).sum(1)).reshape(-1,1)
	fres=fft.rfft(result,axis=1)
	if not args.auto and len(result)>=5:	# remove profiles with strong noise
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
		lnumber=int(np.round(fabss.sum()/fabss.max()*2))
		lnumber=max(4,lnumber)
	#
	knumber=lnumber*2
	nsub,nbin=result.shape
	dt=np.zeros(nsub)
	angerr=np.zeros([nsub,knumber-1])
	nb=int(nbin/2+1)
	errbinnum=np.min([int(nb/6),20])
	for i in np.arange(nsub):	# pre-align all the profiles
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
	if args.red:	# align the pulse phase and reserve the low frequency component
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
			jj=np.ones(nsub,dtype=bool)
		ang1=popt[:(knumber-2)]
		ang1=np.append(ang1,sum0-ang1.sum())
		weight=1/(np.var(fres[:,-errbinnum:].real,1)+np.var(fres[:,-errbinnum:].imag,1))[jj]
		abs0=(np.abs(fres[jj])*weight.reshape(-1,1)).sum(0)/weight.sum()
		prof0=abs0[1:knumber]*np.exp(1j*(ang1+fang0))
		prof[s]=fft.irfft(np.concatenate(([0],prof0,np.zeros(nb-knumber))))
	else:	# align with the pre-aligh profile
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
			pca=PCA(n_components=nc)	# PCA to get the initial value for fitting
			def resi(x):
				knk,cin,dk=x[:nc*nsub],x[nc*sub:nc*nsub+nc*nbin],x[nc*nsub+nc*nbin:]
				kk=knk.reshape(nsub,nc)
				cc=cin.reshape(nc,nbin)
				pp=np.fft.irfft(np.fft.rfft(kk@cc,axis=1)*np.exp(2j*np.pi*np.arange(nb)*dk.reshape(-1,1)),axis=1)
				return (result-pp).reshape(-1)
			c0=np.concatenate((pca.mean_,pca.components_[:nc-1]),axis=0)
			k0=np.array([[1]+[0]*(nc-1)]*nsub).reshape(-1)
			d0=np.zeros(nsub)
			p0=np.concatenate((k0,c0,d0),axis=0)
			res=so.leastsq(resi,x0=p0,full_output=True)	# fit to obtain different components
			knk,cin,dk=res[0][:nc*nsub],res[0][nc*sub:nc*nsub+nc*nbin],res[0][nc*nsub+nc*nbin:]
			kk=knk.reshape(nsub,nc)
			cc=cin.reshape(nc,nbin)
			#coeff=pca.fit_transform(result)[:,0]
			#x1=pca.components_[0]
			#mean=pca.mean_
			#prof[s]=np.array([mean,x1])
			coeff=kk[:,1]/kk[:,0]
			prof=cc[:2]
			kmax,kmin=np.max(coeff),np.min(coeff)
			krange.append(str(kmin*1.3-kmax*0.3)+','+str(kmax*1.3-kmin*0.3))
		else:
			prof[s]=(result*weight.reshape(-1,1)).sum(0)/weight.sum()
#
if args.freqtem:	# align the multi-frequency template
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
#
if args.peakfit:
	def peak(x,x0,sigma,a):
		x1=(x-x0)%1
		x1[x1>0.5]-=1
		# return a/np.cosh(x1*(2/np.pi)**0.5/sigma)**2
		return a*np.exp((np.cos((x-x0)*2*np.pi)-1)/(sigma*4*np.pi)**2)	# not a standard von-Mises peak, the width is doubled.
	#
	def mpeak(x,*para):
		m=int((len(para)-1)/3)
		y=np.zeros(len(x))
		for i in np.arange(m):
			y+=peak(x,*para[i*3:(i+1)*3])
		return y+para[-1]
	#
	def xcal(x):
		xlim=ax2.get_xlim()
		return (x-ax2.bbox.extents[0])/ax2.bbox.bounds[2]*(xlim[1]-xlim[0])+xlim[0]
	#
	def ycal(y):
		ylim=ax2.get_ylim()
		return (fig.bbox.extents[3]-y-ax2.bbox.extents[1])/ax2.bbox.bounds[3]*(ylim[1]-ylim[0])+ylim[0]
	#
	import tkinter as tk
	from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
	from matplotlib.figure import Figure
	import matplotlib.lines as ln
	import matplotlib.pyplot as plt
	import matplotlib as mpl
	mpl.use('TkAgg')
	plt.rcParams['font.family']='Serif'
	root=tk.Tk()
	root.title('Peak Fit')
	root.geometry('800x600+100+100')
	#
	fig=Figure(figsize=(40,30),dpi=80)
	fig.clf()
	x0,x1=0.13,0.95
	y0,y1,y2=0.11,0.35,0.96
	ax1=fig.add_axes([x0,y0,x1-x0,y1-y0])
	ax2=fig.add_axes([x0,y1,x1-x0,y2-y1])
	l1 = ln.Line2D([0.5,0.5],[0,1],color='k',transform=fig.transFigure,figure=fig)
	l2 = ln.Line2D([0,1],[0,0],color='k',transform=fig.transFigure,figure=fig)
	l3 = ln.Line2D([-1,-1],[0,1],color='y',transform=fig.transFigure,figure=fig)
	l4 = ln.Line2D([-1,-1],[0,1],color='y',transform=fig.transFigure,figure=fig)
	fig.lines.extend([l1,l2,l3,l4])
	#
	phase=np.arange(0,1,1/args.nbin)
	fitmark='x0'
	xymark='x'
	paras=[]
	prof1=prof[0].copy()
	bbin0=int((af.baseline(prof1,pos=True)[1]+int(args.nbin/20))%args.nbin)
	prof1=np.append(prof1[bbin0:],prof1[:bbin0])
	tmppara=np.zeros(3)
	h0=af.baseline(prof1)
	prof1-=h0
	prof1/=prof1.max()
	prof2=prof1.copy()
	h0=0
	savemark=False
	def update_fig():
		global fprof
		ax1.cla()
		ax2.cla()
		ax1.plot(phase,prof1,'b-')
		ax2.plot(phase,prof2-h0,'k-')
		fprof=np.zeros_like(phase)
		for para in paras:
			ftmp=peak(phase,*para)
			ax2.plot(phase,ftmp,'y:')
			fprof+=ftmp
		ax2.plot(phase,fprof,'r--')
		ax2.plot(phase,prof1,'b-')
		ax1.set_xlabel('Pulse Phase',fontsize=30)
		ax2.set_ylabel('Intensity (a.u.)',fontsize=30)
		ax1.set_ylabel('resi.',fontsize=30)
	#
	def keymotion(event):	# press a key
		global fitmark,xymark,paras,h0,prof1,tmppara,savemark
		a=event.keysym
		if a=='a':
			if fitmark=='x0':
				try:
					height=np.max(prof1)
					b0=np.argmax(prof1)
					peakprof=np.zeros(args.nbin)
					for i in np.arange(args.nbin-1):
						if prof1[(i+b0)%args.nbin]<0: break
						else: peakprof[b0+i]=prof1[b0+i]
					for i in np.arange(1,args.nbin-1):
						if prof1[(b0-i)%args.nbin]<0: break
						else: peakprof[b0-i]=prof1[b0-i]
					width=peakprof.sum()/height/args.nbin
					x0=phase[b0]
					tmppara=[x0,width,height]
					popt,pcov=so.curve_fit(mpeak,phase,prof2,p0=np.concatenate((np.reshape(paras,-1),tmppara,[h0])))
					paras=popt[:-1].reshape(-1,3)
					h0=popt[-1]
					prof1=prof2-mpeak(phase,*popt)
				except:
					sys.stdout.write("Failed to add a new peak automatically, please add it manually.\n")
				update_fig()
				tmppara=np.zeros(3)
				canvas.draw()
			if np.size(paras)*2>args.nbin:
				fitmark='stop'
				sys.stdout.write("The peak number is too large, and the fitting has stopped.\n\n")
		if a=='f':
			if fitmark=='fit':
				try: 
					popt,pcov=so.curve_fit(mpeak,phase,prof2,p0=np.concatenate((np.reshape(paras,-1),tmppara,[h0])))
					paras=popt[:-1].reshape(-1,3)
					h0=popt[-1]
					prof1=prof2-mpeak(phase,*popt)
				except:
					sys.stdout.write("The fiting for the peak "+str(list(tmppara))+" failed.\n")
				update_fig()
				fitmark='x0'
				xymark='x'
				tmppara=np.zeros(3)
				x=event.x/fig.bbox.extents[2]
				l1.set_xdata([x,x])
				canvas.draw()
			if np.size(paras)*2>args.nbin:
				fitmark='stop'
				sys.stdout.write("The peak number is too large, and the fitting has stopped.\n\n")
		if a=='q':
			root.destroy()
			savemark=True
		if a=='b':
			root.destroy()
		if a=='r':
			if fitmark!='stop':
				fitmark='x0'
				xymark='x'
				tmppara=np.zeros(3)
				x=event.x/fig.bbox.extents[2]
				l1.set_xdata([x,x])
				l3.set_xdata([1,1])
				l4.set_xdata([1,1])
				canvas.draw()
				update_fig()
		if a=='h':
			sys.stdout.write("\nldtem interactive commands\n\n")
			sys.stdout.write("Black solid curve       :   processed pulse profile\n")
			sys.stdout.write("Yellow dotted curve     :   fitted peak\n")
			sys.stdout.write("Red dashed curve        :   fitting curve for pulse profile\n")
			sys.stdout.write("Green dash-dotted curve :   a new peak to be fitted\n")
			sys.stdout.write("Blue solid curve        :   fitting residuals\n\n")
			sys.stdout.write("Mouse:\n")
			sys.stdout.write("  Left-click to select the centre of a peak\n")
			sys.stdout.write("    then left-click again to select a width of a peak.\n")
			sys.stdout.write("    then left-click again to select a height of a peak.\n")
			sys.stdout.write("Keyboard:\n")
			sys.stdout.write("  h  Show this help\n")
			sys.stdout.write("  f  fit the profile with appending the green dash-dotted peak\n")
			sys.stdout.write("  r  Reset a peak\n")
			sys.stdout.write("  q  Exit peak fitting and save the template.\n")
			sys.stdout.write("  b  Exit peak fitting without saving.\n\n")
	#
	def leftclick(event):
		global fitmark,xymark,tmppara
		if fitmark=='x0':
			if event.y<(fig.bbox.extents[3]-ax2.bbox.extents[1]) and event.y>(fig.bbox.extents[3]-ax2.bbox.extents[3]) and event.x>ax2.bbox.extents[0] and event.x<ax2.bbox.extents[2]: 
				tmppara[0]=xcal(event.x)
				fitmark='sigma'
				xymark='x'
				x=event.x/fig.bbox.extents[2]
				l3.set_xdata([x,x])
				canvas.draw()
		elif fitmark=='sigma':
			if event.y<(fig.bbox.extents[3]-ax2.bbox.extents[1]) and event.y>(fig.bbox.extents[3]-ax2.bbox.extents[3]) and event.x>ax2.bbox.extents[0] and event.x<ax2.bbox.extents[2]: 
				tmppara[1]=np.abs(xcal(event.x)-tmppara[0])
				fitmark='a'
				xymark='y'
				x=event.x/fig.bbox.extents[2]
				l4.set_xdata([x,x])
				l1.set_xdata([1,1])
				l2.set_ydata([0.5,0.5])
				canvas.draw()
		elif fitmark=='a':
			if event.y<(fig.bbox.extents[3]-ax2.bbox.extents[1]) and event.y>(fig.bbox.extents[3]-ax2.bbox.extents[3]) and event.x>ax2.bbox.extents[0] and event.x<ax2.bbox.extents[2]: 
				tmppara[2]=ycal(event.y)
				fitmark='fit'
				xymark='0'
				ax2.plot(phase,peak(phase,*tmppara),'g-.')
				l3.set_xdata([1,1])
				l4.set_xdata([1,1])
				canvas.draw()
	#
	def move(event):
		if xymark=='x':
			if event.y<(fig.bbox.extents[3]-ax2.bbox.extents[1]) and event.y>(fig.bbox.extents[3]-ax2.bbox.extents[3]): 
				x=event.x/fig.bbox.extents[2]
				l1.set_xdata([x,x])
				l2.set_ydata([1,1])
				canvas.draw()
		elif xymark=='y':
			if event.y<(fig.bbox.extents[3]-ax2.bbox.extents[1]) and event.y>(fig.bbox.extents[3]-ax2.bbox.extents[3]): 
				y=(fig.bbox.extents[3]-event.y)/fig.bbox.extents[3]
				l2.set_ydata([y,y])
				l1.set_xdata([1,1])
				canvas.draw()
		else:
				l2.set_ydata([1,1])
				l1.set_xdata([1,1])
				canvas.draw()			
	#
	update_fig()
	#
	canvas=FigureCanvasTkAgg(fig,master=root)
	canvas.get_tk_widget().grid()  
	canvas.get_tk_widget().pack(fill='both')
	root.bind('<KeyPress>',keymotion)
	root.bind('<ButtonPress-1>',leftclick)
	root.bind('<Motion>',move)
	canvas.draw()
	root.mainloop()
#
info={'data_info':{'mode':'template','nchan':int(nchan_res),'chan_weight':list(np.ones(nchan_res)),'nbin':int(args.nbin),'npol':1,'pol_type':'I','compressed':True,'freq_start':freq_s,'freq_end':freq_e,'length':1,'period':psr.p0,'sublen':psr.p0,'sub_nperiod_last':1},'pulsar_info':{'psr_name':psrname},'history_info':{'file_time':[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())],'history':[command]}}
#
if args.peakfit:
	if not savemark:sys.exit()
	if np.sum(np.array(paras)**2)==0:
		print('The produced template is zero on every phase bin, and it will not be saved.')
		sys.exit()
	info['template_info']={'peak_paras':paras.tolist()}
	prof=np.array([[fprof]])
else:
	if np.sum(prof**2)==0: parser.error('Unexpected error. The produced profile is zero on every phase bin.')
	prof=prof.reshape(nchan_res,-1,args.nbin)
#
do=ld.ld(name+'.ld')
if args.component:
	do.write_shape([nchan_res,2,args.nbin,1])
	info['additional_info']={'krange':krange}
	info['data_info']['nsub']=2
else:
	do.write_shape([nchan_res,1,args.nbin,1])
	info['data_info']['nsub']=1
#
prof-=prof[:,0,:].min(1).reshape(-1,1,1)
prof/=prof[:,0,:].max(1).reshape(-1,1,1)
for i in np.arange(nchan_res):
	do.write_chan(prof[i],i)
#
do.write_info(info)
#
