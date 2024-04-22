#!/usr/bin/env python
import numpy as np
import numpy.random as nr
import numpy.polynomial.chebyshev as nc
import matplotlib as mpl
import os,sys,time,ld,json
import scipy.stats as ss
import scipy.interpolate as si
import time_eph as te
import psr_read as pr
import psr_model as pm
import adfunc as af
import astropy.io.fits as ps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext as st
import tkinter.messagebox as tm
import tkinter.filedialog as tf
import shutil as sh
mpl.use('TkAgg')  
#
directory=os.path.dirname(os.path.abspath(__file__))
noiselist=['None','Narrow','Wide','Complex']
poltypelist=['AABBCRCI','AABB','IQUV']
callist=['None','Random']
compshapelist=['vonMises','CosN','Lorentz','Sech','Sech2']
singlelist=['Driftless','Coherent','Diffused']
proftypelist=['One Peak','MP+IP','Random']
ppatypelist=['RVM','RVM+90','Arbitrary']
profpoltypelist=['Strong Lin.','Weak Lin.','Weak Pol.']
obsdistlist=['Random','Uniform']
afiles=[noiselist,callist,proftypelist,obsdistlist]
#
def vonmises(x,a,x0,sigma):
	return a*np.exp((np.cos((x-x0)*2*np.pi)-1)/(sigma*2*np.pi)**2)
#
def cosn(x,a,x0,sigma):
	x1=(x-x0)%1
	x1[x1>0.5]-=1
	return a*np.cos(x1*np.pi)**((np.pi/10/sigma)**2)
#
def lorentz(x,a,x0,b):
	x1=(x-x0)%1
	x1[x1>0.5]-=1
	return a/((x1*(np.pi/2)**0.5/b)**2+1)
#	
def sech2(x,a,x0,sigma):
	x1=(x-x0)%1
	x1[x1>0.5]-=1
	return a/np.cosh(x1*(2/np.pi)**0.5/sigma)**2
#
def sech(x,a,x0,sigma):
	x1=(x-x0)%1
	x1[x1>0.5]-=1
	return a/np.cosh(x1*(np.pi/2)**0.5/sigma)
#
def gen_noise(ntype,nchan,gain,nfac0=[]):
	noise=np.zeros(nchan)
	nfac=[]
	if ntype=='None':pass
	elif ntype in ['Narrow','Wide','Complex']:
		if ntype in ['Narrow','Complex']:
			if not nfac0:
				if nchan>10: nn=nr.randint(max(5,int(nchan/200)),max(100,int(nchan/50)))
				else: nn=nr.randint(nchan)
				pn=nr.randint(0,nchan,nn)
			else:
				nn,pn=nfac0[:2]
			noise[pn]+=(np.abs(nr.randn(nn))+0.1)*gain[pn]
			nfac.extend([nn,pn])
		if ntype in ['Wide','Complex']:
			if not nfac0:
				if nchan>10:
					nnoise=nr.randint(1,max(10,int(nchan/800)))
					wnoise=nr.rand(nnoise)*max(2,min(nchan/5,30))+2
				else:
					nnoise=1
					wnoise=nr.rand()*2+2
				pnoise=nr.rand(nnoise)*nchan
			else:
				nnoise,pnoise,wnoise=nfac0[-3:]
			chan=np.arange(nchan)
			noisew=np.zeros(nchan)
			for i in np.arange(nnoise):
				noisew+=np.exp(-(chan-pnoise[i])**2/2/wnoise[i]**2)*(np.abs(nr.randn())+0.1)
			noise+=noisew*gain
			nfac.extend([nnoise,pnoise,wnoise])
	elif os.path.isfile(ntype):
		if zchan:
			tmp0=zchan[0]
			tmp=[[tmp0]]
			for i in zchan[1:]:
				if i-tmp0==1: tmp[-1].append(i)
				else: tmp.append([i])
				tmp0=i
			noise_narrow=[]
			noise_wide=[]
			for i in tmp:
				if len(i)==1: noise_narrow.append(i[0])
				else: noise_wide.append([np.mean(i),(i[-1]-i[0])/6])
			noisew=np.zeros(nchan)
			chan=np.arange(nchan)
			noise[noise_narrow]+=(np.abs(nr.randn(len(noise_narrow)))+0.1)*gain[noise_narrow]
			for p,w in noise_wide:
				noisew+=np.exp(-(chan-p)**2/2/w**2)*(np.abs(nr.randn())+0.1)
			noise+=noisew*gain
			nfac=[]
	else:
		raise
	return noise,nfac
#
def gen_cal(ctype,freqs,gain,cfac=[]):
	if ctype=='None': return np.ones([4,len(freqs)]),[]
	elif ctype=='Random':
		if cfac:
			a1,a2,dl=cfac
		else:
			a1=nr.rand()*0.6+0.7
			a2=1/a1
			dl=nr.rand()*2-1
		lam=te.sl/freqs/1e6
		dphi=lam/dl*np.pi*2
		return np.asarray([a1*gain,a2*gain,np.sqrt(a1*a2)*np.cos(dphi),np.sqrt(a1*a2)*np.sin(dphi)]),[a1,a2,dl]
	elif os.path.isfile(ctype):
		return ld.ld(ctype).read_period(0).reshape(-1,4).T,[]
	else: raise
#
def rvm(phi,alpha,beta,psi0,pa0):
	p=(phi-psi0)*np.pi*2
	a=alpha/180.0*np.pi
	b=beta/180.0*np.pi
	c=a+b
	tpa=np.sin(a)*np.sin(p)/(np.sin(c)*np.cos(a)-np.cos(c)*np.sin(a)*np.cos(p))
	pa=np.arctan(tpa)
	pa=pa+pa0/180.0*np.pi
	return pa*180/np.pi
#
def gen_profile(proftype,ncomp,compshape,poltype,ppatype,nbin,pfac=[]):
	ii,qq,uu,vv=np.zeros([4,nbin])
	bins=np.arange(nbin)
	func={'vonMises':vonmises,'CosN':cosn,'Lorentz':lorentz,'Sech':sech,'Sech2':sech2}[compshape]
	if pfac:
		pcomp,hcomp,wcomp,lincoeff,circoeff,vsign,ppap0=pfac
		ncomp=len(pcomp)
	else:
		if proftype=='One Peak':
			pcomp=np.asarray(nr.randn(ncomp)/30+0.5).reshape(-1)
			ppap0=0.5
		elif proftype=='MP+IP':
			if ncomp<2: raise
			ncomp1=nr.randint(1,ncomp)
			ncomp2=ncomp-ncomp1
			pcomp=np.append(nr.randn(ncomp1)/30+0.25,nr.randn(ncomp2)/30+0.75)
			ppap0=0.25
		elif proftype=='Random':
			pcomp=nr.rand(ncomp)
			ppap0=nr.rand()
		elif os.path.isfile(proftype):
			fd=ld.ld(proftype)
			f=fd.chan_scrunch()[0]
			if fd.read_para('poltype')=='IQUV' and f.shape[-1]==4:
				baseline,bin0=af.baseline(f[:,0],pos=True)
				f-=f.reshape(1,-1,4).repeat(2,axis=0).reshape(-1,4)[bin0:(bin0+int(len(f)/10))].mean(0)
				ppa=1/2*np.arctan2(f[:,2],f[:,1])
				ll=np.sqrt(f[:,1]**2+f[:,2]**2)
				return [f[:,0],f[:,1],f[:,2],f[:,3],ll,ppa],[]
			elif fd.read_para('poltype')[:4]=='AABB' or f.shape[-1]==2:
				prof=f[:,:2].mean(1)
			elif f.shape[-1]==1: prof=f.mean(1)
			theta=np.arange(len(prof))/len(prof)*np.pi*2
			ppap0=np.arctan2((prof*np.sin(theta)).mean(),(prof*np.cos(theta)).mean())/np.pi/2
		#
		if poltype=='Strong Lin.':
			lincoeff=nr.rand()*0.5+0.5
			circoeff=np.sqrt(1-lincoeff**2)*(nr.rand()*0.5+0.5)
		elif poltype=='Weak Lin.':
			circoeff=nr.rand()*0.1+0.9
			lincoeff=np.sqrt(1-circoeff**2)*(nr.rand()*0.5+0.5)
		elif poltype=='Weak Pol.':
			lincoeff=nr.rand()*0.1+0.05
			circoeff=nr.rand()*0.1+0.05
		#
		vsign=[-1,1][nr.randint(2)]
		hcomp,wcomp=nr.rand(ncomp)+0.1,(nr.rand(ncomp)/40+1/100)
	#
	if ppatype=='RVM':
		ppa=rvm(bins/nbin,nr.rand()*180,(nr.randn()*30+90)%180-90,ppap0,nr.rand()*180)
		pj=[]
	elif ppatype=='RVM+90':
		ppa=rvm(bins/nbin,nr.rand()*180,(nr.randn()*30+90)%180-90,ppap0,nr.rand()*180)
		pj=[ppap0+nr.randn()*0.01,(ppap0+nr.rand()*0.4+0.3)%1]
		pj=[min(*pj),max(*pj)]
	elif ppatype=='Arbitrary':
		if nr.rand()<0.5:
			pj=[ppap0+nr.randn()*0.01,(ppap0+nr.rand()*0.4+0.3)%1]
			pj=[min(*pj),max(*pj)]
		else: pj=[]
		nsin=10
		ppa=(np.cos(((bins.reshape(-1,1)/nbin)@np.arange(nsin).reshape(1,-1)+nr.rand(1,nsin))*2*np.pi)*nr.randn(nsin)).sum(1)
		ppa*=(nr.rand()*80+100)/(ppa.max()-ppa.min())
		if nr.rand()>0.667: ppa+=bins/nbin*180
		elif nr.rand()<0.5: ppa-=bins/nbin*180
	#
	if proftype in proftypelist:
		vvtmp=np.zeros([ncomp,nbin])
		for i in range(ncomp):
			tmp=func(bins/nbin,hcomp[i],pcomp[i],wcomp[i])
			vvtmp[i]=tmp*circoeff*np.sin((bins/nbin-pcomp[i]+nr.randn()/30)*2*np.pi*vsign)
			ii+=tmp
		nvv=2
		vv=(vvtmp**nvv*np.sign(vvtmp)).sum(0)
		vv=np.abs(vv)**(1/nvv)*np.sign(vv)
	else:
		fprof=np.fft.rfft(profile)
		if nbin>len(profile):
			fprof0=np.zeros([int(nbin0/2)+1],dtype=np.complex128)
			fprof0[:,:len(fprof)]=fprof
			ii=np.fft.irfft(fprof0,axis=1)
		else:
			ii=np.fft.irfft(fprof[:int(nbin/2+1)])
		vv=ii*circoeff*np.sin((bins/nbin-ppap0+nr.randn()/30)*2*np.pi*vsign)
	#
	nsin=3
	tmp=(np.cos(((bins.reshape(-1,1)/nbin)@np.arange(nsin).reshape(1,-1)+nr.rand(1,nsin))*2*np.pi)*nr.randn(nsin)).sum(1)
	ll=(1-(tmp-tmp.min())/(tmp.max()-tmp.min())*nr.rand()*0.2)*lincoeff*ii
	if pj:
		jj=(bins>pj[0]*nbin)&(bins<pj[1]*nbin)
		ppa[jj]+=90
		factor=np.cos((bins/nbin-np.mean(pj))*2*np.pi)-np.cos(np.mean(pj)*2*np.pi)
		fmax,fmin=factor.max(),np.abs(factor.min())
		if fmax>fmin: 
			jj=factor>0
			factor[jj]=factor[jj]/fmax*(2-factor[jj]/fmax)*fmin
		else:
			jj=factor<0
			factor[jj]=factor[jj]/fmin*(2+factor[jj]/fmin)*fmax
		factor=np.abs(factor)/min(fmax,fmin)
		ll*=factor
		vv*=1+np.sqrt(1-factor**2)*lincoeff*circoeff
	ppa=ppa%180
	qq=ll*np.cos(ppa/180*np.pi)
	uu=ll*np.sin(ppa/180*np.pi)
	imax=ii.max()
	ii/=imax
	qq/=imax
	uu/=imax
	vv/=imax
	ll/=imax
	return [ii,qq,uu,vv,ll,ppa],[pcomp,hcomp,wcomp,lincoeff,circoeff,vsign,ppap0]
#
def gen_single(prof,drtype,nperiod,nrate,pfac,sfac=False):
	nbin=len(prof)
	result=np.zeros([nperiod,nbin])
	if sfac:
		wsub,dsub,hsub,factor=pfac
	else:
		wsub=(nr.rand()+0.3)*pfac[2].min()/2
		dsub=(nr.rand()*6+6)*wsub
		hsub=dsub/np.sqrt(2*np.pi)/wsub
		factor=nr.rand()*4+2
	#
	if drtype=='Driftless':
		nsub=int(nperiod/dsub)+1
		psub=dsub*np.arange(nsub)+nr.randn(nsub)*dsub/3
		wsub0=(nr.rand(nsub)*1.5+0.25)*wsub
	elif drtype=='Coherent':
		nsub=int(nperiod/dsub)+1
		psub=dsub*np.arange(nsub)+(nr.rand(nsub)-0.5)*dsub/10
		wsub0=(nr.rand(nsub)*0.4+0.8)*wsub
	elif drtype=='Diffused':
		nsub=int((nperiod+2)/dsub)+1
		psub0=dsub*np.arange(nsub)
		psub=psub0+(nr.rand(nsub)-0.5)*dsub/10+si.interp1d(np.arange(-1,nperiod+10,5),nr.rand(int(nperiod/5)+3)-0.5)(psub0)*0.3
		wsub0=(nr.rand(nsub)*0.4+0.8)*wsub
	#
	for i in np.arange(nsub):
		isub=int(psub[i]//1)
		if isub<=0: isub=1
		if isub>=(nperiod-1): isub=nperiod-2
		result[isub-1:isub+2]+=(np.exp(-(np.arange(nbin*3)/nbin+isub-1-psub[i])**2/2/wsub0[i]**2)*hsub*ss.chi2.rvs(factor)/factor).reshape(3,-1)
	nj=nr.rand(nperiod)<nrate
	result[nj]=0
	return result,[wsub,dsub,hsub,factor]
#
class psr_para:
	def __init__(self,name,p0,dm,rm,flux,specind,nrate):
		self.name=name
		self.p0=p0
		self.dm=dm
		self.rm=rm
		self.flux=flux
		self.specind=specind
		self.nrate=nrate
#
class cal_para:
	def __init__(self,period,duty,start,duration,ppa,temperature):
		self.period=period
		self.duty=duty
		self.start=start
		self.duration=duration
		self.ppa=ppa
		self.temperature=temperature
#
class ldpsr_para:
	def __init__(self,nbin,nchan,sub_nperiod,poltype,cal):
		self.nbin=nbin
		self.nchan=nchan
		self.sub_nperiod=sub_nperiod
		self.poltype=poltype
		self.cal=cal
#
class fitsdata_para:
	def __init__(self,tsamp,nsblk,nsub,nchan,poltype,cal):
		self.tsamp=tsamp
		self.nsblk=nsblk
		self.nsub=nsub
		self.nchan=nchan
		self.poltype=poltype
		self.cal=cal
#
class obs_para:
	def __init__(self,freq,bw,be,mjd,length,noise):
		self.freq=freq
		self.bw=bw
		self.be=be
		self.mjd=mjd
		self.length=length
		self.noise=noise
#
class tele_para:
	def __init__(self,name,aperture,temperature,efficiency):
		self.name=name
		self.aperture=aperture
		self.temperature=temperature
		self.efficiency=efficiency
#
class prof_para:
	def __init__(self,proftype,ncomp,compshape,single,poltype,ppatype):
		self.proftype=proftype
		self.ncomp=ncomp
		self.compshape=compshape
		self.single=single
		self.poltype=poltype
		self.ppatype=ppatype
#
class psrfits_psr:
	def __init__(self,psr,fitsdata,obs,tele,prof):
		self.psr=psr
		self.data=fitsdata
		self.obs=obs
		self.tele=tele
		self.prof=prof
		self.keys={'psr':[('name',str),('p0',float),('dm',float),('rm',float),('flux',float),('specind',float),('nrate',float)],
			'obs':[('freq',float),('bw',float),('be',float),('mjd',float),('length',float),('noise',str)],
			'tele':[('name',str),('aperture',float),('temperature',float),('efficiency',float)],
			'prof':[('proftype',str),('ncomp',int),('compshape',str),('single',str),('poltype',str),('ppatype',str)],
			'data':[('tsamp',float),('nsblk',int),('nsub',int),('nchan',int),('poltype',str),('cal',str)]}
	#
	def update(self,update=True):
		values=dict()
		for i in self.keys.keys():
			values[i]=dict()
			for k,f in self.keys[i]:
				values[i][k]=f(self.__getattribute__(i).__getattribute__(k).get())
		if update: 
			self.values=values
			self.mode='input'
		else: return values
	#
	def check(self):
		def warning(str):
			tm.showwarning('Warning!',str)
			return True
		#
		acclen=self.values['data']['tsamp']*self.values['obs']['bw']/self.values['data']['nchan']
		if acclen<1: return warning('The time and frequency resolution is too high!')
		#
		try: af.reco(self.values['tele']['name'])
		except: return warning('Unknown Telescope!')
		#
		try:
			psr=pr.psr(self.values['psr']['name'])
			mode='eph'
		except:
			try:
				psr=pr.psr(self.values['psr']['name'],parfile=True)
				mode='eph'
			except:
				mode='pdm'
		if mode=='eph':
			time=te.times(te.time(self.values['obs']['mjd'],0,scale=self.values['tele']['name']))
			tran,l=np.array(time.obs(psr,5)).reshape(-1)			
			if tran is None:
				if l==0: return warning('The pulsar cannot be observed with the specified telescope!')
			else:
				if self.values['obs']['length']>l*86400: return warning('The pulsar cannot be monitored for a such long integration time with the specified telescope!')
				elif tran>l/2 or (tran+l/2)*86400<self.values['obs']['length']:
					return warning('The pulsar cannot be observed for the given observation time! A proper start time could be between '+str(self.values['obs']['mjd']+tran-l/2)+' and '+str(self.values['obs']['mjd']+tran+l/2)+'.')
		#
		if self.values['obs']['noise'] not in noiselist and os.path.isfile(self.values['obs']['noise']):
			try: zchan=np.loadtxt(self.values['obs']['noise'],dtype=int)
			except: return warning('The noise file cannot be recognized!')
		#
		if self.values['data']['cal'] not in callist and os.path.isfile(self.values['data']['cal']):
			try:
				d=ld.ld(self.values['data']['cal'])
				if d.read_para('mode')!='cal': warning('LD file is not caliration file!')
				elif d.read_para('cal_mode')!='seg': warning('Calibration file can only contain 1 segment!')
				elif d.read_shape()[1]!=1: warning('Calibration file can only contain 1 segment!')
			except: return warning('The calibration file cannot be recognized!')
		#
		if self.values['prof']['proftype'] not in proftypelist and os.path.isfile(self.values['prof']['proftype']):
			try:
				d=ld.ld(self.values['prof']['proftype'])
				if d.read_para('mode') not in ['single','subint','template']: warning('LD file does not contain a pulse profile!')
				elif d.read_shape()[1]!=1: warning('LD file does contain more than one pulse profiles!')
			except: return warning('The profile file cannot be recognized!')
		#
		if self.values['psr']['nrate']>=1 or self.values['psr']['nrate']<0: warning('The nulling rate is illegal!')
	#
	def genpsr(self,fac=0):
		self.freq_start=self.values['obs']['freq']-self.values['obs']['bw']/2
		self.freq_end=self.values['obs']['freq']+self.values['obs']['bw']/2
		self.freqs=np.arange(self.freq_start,self.freq_end,self.values['obs']['bw']/self.values['data']['nchan'])
		self.gain=np.ones(self.values['data']['nchan'])
		if self.values['obs']['be']>0:
			j0=(self.freqs-self.freq_start)<self.values['obs']['be']
			self.gain[j0]=np.sin((self.freqs[j0]-self.freq_start)/self.values['obs']['be']*np.pi/2)**2
			j1=(self.freqs-self.freq_end)>(-self.values['obs']['be'])
			self.gain[j1]=np.sin((self.freqs[j1]-self.freq_end)/self.values['obs']['be']*np.pi/2)**2
		self.nsingle=100
		nbin=4096
		self.phase=np.arange(nbin)/nbin
		self.sefd=self.values['tele']['temperature']/(self.values['tele']['aperture']**2*2.84e-4*self.values['tele']['efficiency'])
		a=self.values['psr']['specind']
		if a!=1: self.speccoeff=self.values['psr']['flux']*self.values['obs']['bw']*(1-a)/(self.freq_end**(1-a)-self.freq_start**(1-a))
		else: self.speccoeff=self.values['psr']['flux']*self.values['obs']['bw']/np.log(self.freq_end/self.freq_start)
		self.spec=self.freqs**(-a)*self.speccoeff
		if fac:
			self.noise,self.nfac=gen_noise(self.values['obs']['noise'],self.values['data']['nchan'],self.gain,nfac0=self.nfac)
			self.profile,self.pfac=gen_profile(self.values['prof']['proftype'],self.values['prof']['ncomp'],self.values['prof']['compshape'],self.values['prof']['poltype'],self.values['prof']['ppatype'],nbin,pfac=self.pfac)
			prof0=np.fft.irfft(np.fft.rfft(self.profile[0])[:65])
			self.tdomain,self.sfac=gen_single(prof0,self.values['prof']['single'],self.nsingle,self.values['psr']['nrate'],self.sfac,sfac=True)
			self.tdomain*=prof0
			self.cal,self.calpara=gen_cal(self.values['data']['cal'],self.freqs,self.gain,cfac=self.calpara)
		else:
			self.noise,self.nfac=gen_noise(self.values['obs']['noise'],self.values['data']['nchan'],self.gain)
			self.profile,self.pfac=gen_profile(self.values['prof']['proftype'],self.values['prof']['ncomp'],self.values['prof']['compshape'],self.values['prof']['poltype'],self.values['prof']['ppatype'],nbin)
			prof0=np.fft.irfft(np.fft.rfft(self.profile[0])[:65])
			self.tdomain,self.sfac=gen_single(prof0,self.values['prof']['single'],self.nsingle,self.values['psr']['nrate'],self.pfac)
			self.tdomain*=prof0
			self.cal,self.calpara=gen_cal(self.values['data']['cal'],self.freqs,self.gain)
		self.fdomain=np.fft.irfft(np.fft.rfft(self.spec)[:65]).reshape(-1,1)@prof0.reshape(1,-1)
	#
	def loadpsr(self,dic):
		self.nfac=dic.pop('noise factors')
		self.sfac=dic.pop('single factors')
		self.pfac=dic.pop('profile factors')
		self.calpara=dic.pop('calibration factors')
		self.values=dic
		if self.refresh(): return True
		self.mode='load'
		self.genpsr(fac=1)
	#
	def refresh(self):
		for i in self.keys.keys():
			for k,f in self.keys[i]:
				try:
					value=f(self.values[i][k])
				except:
					tm.showwarning('Warning','The parameter '+i+':'+k+' is invalid!')
					return True
				widget=self.__getattribute__(i).__getattribute__(k)
				if widget.winfo_class()=='TCombobox':
					values=widget['value']
					if list(values)[:-1] not in afiles and self.values[i][k] not in values:
						tm.showwarning('Warning','The selected value '+i+':'+k+' is invalid!')
						return True
		for i in self.keys.keys():
			for k,f in self.keys[i]:
				widget=self.__getattribute__(i).__getattribute__(k)
				if widget.winfo_class()=='Entry':
					widget.delete(0,'end')
					widget.insert(0,str(self.values[i][k]))
				elif widget.winfo_class()=='TCombobox':
					values=widget['value']
					if self.values[i][k] in values and self.values[i][k]!='From file':
						widget.current(values.index(self.values[i][k]))
					elif list(values)[:-1] in afiles:
						widget['value']=list(values[:-1])+[self.values[i][k]]
						widget.current(len(values)-1)
	#
	def gendata(self,prog=0,label=0):
		values=self.update(False)
		if values!=self.values: 
			if update_fits_psr(): return
		#
		try:
			psr=pr.psr(self.values['psr']['name'])
			mode='eph'
		except:
			try:
				psr=pr.psr(self.values['psr']['name'],parfile=True)
				mode='eph'
			except:
				mode='pdm'
		#
		tsamp=self.values['data']['tsamp']*1e-6
		nbin0=2**int(np.log2(self.values['psr']['p0']/tsamp)+1)
		profile=np.array(self.profile[:4])
		if nbin0>4096:
			fprof=np.fft.rfft(profile,axis=1)
			fprof0=np.zeros([4,int(nbin0/2)+1],dtype=np.complex128)
			fprof0[:,:(fprof.shape[1])]=fprof
			profile=np.fft.irfft(fprof0,axis=1)
		else:
			profile=profile.reshape(4,nbin0,-1).mean(2)
		#
		profile/=profile[0].mean()
		nsblk=self.values['data']['nsblk']
		sublen=tsamp*nsblk
		nsub=int(self.values['obs']['length']/sublen)
		nfile=int(np.ceil(nsub/self.values['data']['nsub']))
		#
		if self.values['data']['poltype'] in ['AABBCRCI','IQUV']: npol=4
		elif self.values['data']['poltype']=='AABB': npol=2
		#
		fsize=(self.values['data']['nchan']*npol*(nsblk+8)+16)*nsub/2**30
		if fsize>1:
			answer=tm.askokcancel('Warning','The total file size is larger than '+str(int(fsize))+'G, please determine whether to save them.')
			if not answer: return	
		#
		while True:
			filename=tf.asksaveasfilename(filetypes=[('FITS file','.fits')])
			if not filename: return
			if len(filename)>5:
				if filename[-5:]=='.fits': filename=filename[:-5]
			if nfile==1:
				if os.path.isfile(filename+'.fits'):
					answer=tm.askyesnocancel('Warning','File exists!\n\nReplace? \'No\' for rename the file')
					if answer: pass
					elif answer is None: return
					else: continue
				filenames=[filename+'.fits']
				break
			mark=False
			for i in (np.arange(nfile)+1):
				if os.path.isfile(filename+'_'+str(i).zfill(4)+'.fits'):
					mark=True
					break
			if mark:
				answer=tm.askyesnocancel('Warning','File exists!\n\nReplace? \'No\' for rename the file')
				if answer: pass
				elif answer is None: return
				else: continue
			filenames=list(map(lambda x:filename+'_'+str(x).zfill(4)+'.fits',np.arange(nfile)+1))
			break
		#
		stt_imjd=int(self.values['obs']['mjd'])
		stt_smjd=int(self.values['obs']['mjd']%1*86400)
		stt_offs=self.values['obs']['mjd']%1*86400%1
		#
		header=ps.Header()
		header.append(('SIMPLE',True,'file does conform to FITS standard'))
		header.append(('NAXIS',0,'number of data axes'))
		header.append(('EXTEND',True,'FITS dataset may contain extensions'))
		header.append(('OBS_MODE','SEARCH','(PSR, CAL, SEARCH)'))
		header.append(('TELESCOP',self.values['tele']['name'],'Telescope name'))
		header.append(('OBSFREQ',self.values['obs']['freq'],'[MHz] Centre frequency for observation'))
		header.append(('OBSBW',self.values['obs']['bw'],'[MHz] Bandwidth for observation'))
		header.append(('OBSNCHAN',self.values['data']['nchan'],'Number of frequency channels (original)'))
		header.append(('SRC_NAME',self.values['psr']['name'],'Source or scan ID'))
		header.append(('TRK_MODE','TRACK','Track mode (TRACK, SCANGC, SCANLAT)'))
		header.append(('STT_IMJD',stt_imjd,'Start MJD (UTC days) (J - long integer)'))
		header.append(('STT_SMJD',stt_smjd,'[s] Start time (sec past UTC 00h) (J)'))
		header.append(('STT_OFFS',stt_offs,'[s] Start time offset (D)'))
		primary=ps.PrimaryHDU(header=header)
		subint_header=ps.Header()
		subint_header.append(('XTENSION','BINTABLE','***** Subintegration data  *****'))
		subint_header.append(('EXTNAME','SUBINT','name of this binary table extension'))
		subint_header.append(('NAXIS',2,'2-dimensional binary table'))
		subint_header.append(('NAXIS1',int(self.values['data']['nchan']*npol*nsblk+4*self.values['data']['nchan']*npol*2+2*8),'width of table in bytes'))
		subint_header.append(('NAXIS2',self.values['data']['nsub'],'Number of rows in table (NSUBINT)'))
		subint_header.append(('NPOL',npol,'Nr of polarisations'))
		subint_header.append(('NCHAN',self.values['data']['nchan'],'Number of channels/sub-bands in this file'))
		subint_header.append(('CHAN_BW',self.values['obs']['bw']/self.values['data']['nchan'],'[MHz] Channel/sub-band width'))
		subint_header.append(('TBIN',tsamp,'[s] Time per bin or sample'))
		subint_header.append(('NSBLK',nsblk,'Samples/row (SEARCH mode, else 1)'))
		subint_header.append(('POL_TYPE',self.values['data']['poltype'],'Polarisation identifier (e.g., AABBCRCI, AA+BB)'))
		subint_header.append(('TFIELDS',5,'Number of fields per row'))
		subint_header.append(('INT_TYPE','TIME','Time axis (TIME, BINPHSPERI, BINLNGASC, etc)'))
		subint_header.append(('INT_UNIT','SEC','Unit of time axis (SEC, PHS (0-1), DEG)'))
		#
		if mode=='eph':
			dm=psr.dm
			period=psr.p0
			ncoeff=max(int(self.values['obs']['length']/100),12)
			chebx_test0=nc.chebpts1(ncoeff)
			chebx_test=np.concatenate(([-1],chebx_test0,[1]),axis=0)
			second_test=(chebx_test+1)/2*nsblk*nsub*tsamp
			time_test=te.time(stt_imjd*np.ones_like(second_test),(stt_smjd+stt_offs)+second_test,scale=self.values['tele']['name'])
			times_test=te.times(time_test)
			timing_test_end=pm.psr_timing(psr,times_test,self.freq_end)
			timing_test_start=pm.psr_timing(psr,times_test,self.freq_start)
			phase_end=timing_test_end.phase.integer[-1]+1
			phase_start=timing_test_start.phase.integer[0]
			phase=timing_test_end.phase.integer-phase_start+timing_test_end.phase.offset
			nperiod=int(np.ceil(phase_end)-np.floor(phase_start))
			cheb_end=nc.chebfit(chebx_test,timing_test_end.phase.integer-phase_start+timing_test_end.phase.offset,ncoeff-1)
			roots=nc.chebroots(cheb_end)
			roots=np.real(roots[np.isreal(roots)])
			root=roots[np.argmin(np.abs(roots))]
			stt_time_test=te.time(stt_imjd,(root+1)/2*nsblk*nsub*tsamp,scale=self.values['tele']['name'])
			stt_sec=stt_time_test.second[0]
			stt_date=stt_time_test.date[0]
			ncoeff_freq=10
			phase_tmp=np.zeros([ncoeff_freq,ncoeff+2])
			disp_tmp=np.zeros(ncoeff_freq)
			cheby=nc.chebpts1(ncoeff_freq)
			freqy=(cheby+1)/2*self.values['obs']['bw']+self.freq_start
			for i in np.arange(ncoeff_freq):
				timing_test=pm.psr_timing(psr,times_test,freqy[i])
				disp_tmp[i]=((timing_test.tdis1+timing_test.tdis2)/period).mean()
				phase_tmp[i]=timing_test.phase.integer-phase_start+timing_test.phase.offset+disp_tmp[i]
			coeff_freq=np.polyfit(1/freqy,disp_tmp,4)
			coeff=nc.chebfit(chebx_test,nc.chebfit(cheby,phase_tmp,1).T,ncoeff-1)
			phase_start=int(nc.chebval2d(-1,-1,coeff)-np.polyval(coeff_freq,1/self.freqs[0]))
		else:
			dm=self.values['psr']['dm']
			period=self.values['psr']['p0']
			dmdelay=(1/self.freq_start**2-1/self.freq_end**2)*pm.dm_const*dm
			nperiod=int((self.values['obs']['length']+dmdelay)//period+2)
			phase_start=int(-dm/self.freqs[0]**2*pm.dm_const/period)
		#
		single,sfac=gen_single(profile[0],self.values['prof']['single'],nperiod,self.values['psr']['nrate'],self.sfac,sfac=True)
		single=single*(profile*self.values['psr']['flux']*1e-3).reshape(4,1,-1)
		acclen=tsamp*(self.freq_end-self.freq_start)*1e6
		#
		if self.values['psr']['rm']!=0:
			lam=te.sl/self.freqs/1e6
			theta=self.values['psr']['rm']*lam**2*2
			cth,sth=np.sin(theta),np.cos(theta)
		#
		a12,a22,cdphi,sdphi=self.cal
		#
		for ifile in np.arange(nfile):
			if ifile==nfile-1:
				fnsub=nsub-(nfile-1)*self.values['data']['nsub']
			else:
				fnsub=self.values['data']['nsub']
			for isub in np.arange(fnsub):
				ksub=int(isub+ifile*self.values['data']['nsub'])
				if prog:
					prog['value']=ksub/nsub*100
					label.set(str(ksub)+'/'+str(nsub)+' sub-integrations ('+str(round(ksub/nsub*100,1))+'%)')
					prog.update()
				tline=ksub*nsblk+np.arange(nsblk)
				fline=self.freqs
				if mode=='eph':
					phase=nc.chebval2d(*np.meshgrid(tline/(nsblk*nsub)*2-1,fline/self.values['obs']['bw']*2-1),coeff)-np.polyval(coeff_freq,1/fline).reshape(-1,1)
				else:
					phase=(tline.reshape(1,-1)*tsamp-dm/fline.reshape(-1,1)**2*pm.dm_const)/period
				#
				phase-=phase_start
				pstart,pend=int(phase.min()),int(phase.max())+1
				if pend<=0: continue
				elif pstart<0: pstart=0
				phase0=np.arange(pstart,pend,1/nbin0)
				data=np.zeros([nsblk,4,self.values['data']['nchan']])
				for i in np.arange(self.values['data']['nchan']):
					d0=single[:4,pstart:pend].reshape(4,-1)*(self.spec[i]*self.gain[i])
					for k in np.arange(4):
						data[:,k,i]=np.interp(phase[i],phase0,d0[k])
				#
				if self.values['psr']['rm']!=0:
					data[:,1],data[:,2]=data[:,1]*cth-data[:,2]*sth,data[:,2]*cth+data[:,1]*sth
				#
				if self.values['data']['poltype'][0:4]=='AABB':
					data[:,0],data[:,1]=(data[:,0]+data[:,1])/2,(data[:,0]-data[:,1])/2
				data=data[:,:npol]
				#
				baseline=np.zeros([nsblk,npol,self.values['data']['nchan']])
				if self.values['data']['poltype'][0:4]=='AABB':
					baseline[:,:2]=ss.chi2.rvs(2*acclen,size=2*self.values['data']['nchan']*nsblk).reshape(nsblk,2,-1)/2/acclen
				else:
					baseline[:,0]=ss.chi2.rvs(4*acclen,size=self.values['data']['nchan']*nsblk).reshape(nsblk,-1)/4/acclen
					baseline[:,1]=nr.randn(self.values['data']['nchan']*nsblk).reshape(nsblk,-1)/np.sqrt(acclen*2)+1
				if self.values['data']['poltype']=='AABBCRCI':
					baseline[:,2:]=nr.randn(2*self.values['data']['nchan']*nsblk).reshape(nsblk,2,-1)/np.sqrt(acclen)+1
				elif self.values['data']['poltype']=='IQUV':
					baseline[:,2:]=nr.randn(2*self.values['data']['nchan']*nsblk).reshape(nsblk,2,-1)/np.sqrt(acclen)*2+1
				#
				noise=self.noise.reshape(1,-1).repeat(npol,axis=0)
				if self.values['data']['cal']!='None':
					if self.values['data']['poltype'] in ['AABBCRCI','IQUV']:
						data[:,2],data[:,3]=(cdphi*data[:,2]-sdphi*data[:,3],sdphi*data[:,2]+cdphi*data[:,3])*np.sqrt(a12*a22)
					if self.values['data']['poltype']=='IQUV':
						data[:,0],data[:,1]=a12*data[:,0]+a22*data[:,1],a12*data[:,0]-a22*data[:,1]
						noise[0]*=(a12+a22)
						noise[1]*=(a12-a22)
					else:
						data[:,0]*=a12
						data[:,1]*=a22
						noise[0]*=a12
						noise[1]*=a22
				#
				data+=baseline*(noise+self.gain)*self.sefd
				dat_offs_array=data.min(0)
				data-=dat_offs_array
				dat_scl_array=data.max(0)
				dsj=(dat_scl_array!=0)
				data[:,dsj]=np.int16(np.round((data[:,dsj]/dat_scl_array[dsj]*256-0.5)))
				data[:,dat_scl_array==0]=0
				data[data<0]=0
				data[data>255]=255
				data=np.uint8(data)
				dat_offs_array+=dat_scl_array*0.5
				if isub==0:
					tsubint=ps.Column(name='TSUBINT',format='1D',unit='s',array=np.asarray([nsblk*tsamp]))
					offs_sub=ps.Column(name='OFFS_SUB',format='1D',unit='s',array=np.asarray([tline.mean()*tsamp]))
					dat_offs=ps.Column(name='DAT_OFFS',format=str(self.values['data']['nchan']*npol)+'E',array=dat_offs_array)
					dat_scl=ps.Column(name='DAT_SCL',format=str(self.values['data']['nchan']*npol)+'E',array=dat_scl_array)
					dim='(1,'+str(self.values['data']['nchan'])+','+str(npol)+','+str(nsblk)+')'
					data=ps.Column(name='DATA',format=str(self.values['data']['nchan']*npol*nsblk)+'B',dim=dim,array=data.reshape(1,-1))
					cols=ps.ColDefs([tsubint,offs_sub,dat_offs,dat_scl,data])
					subint=ps.BinTableHDU.from_columns(cols,header=subint_header,nrows=fnsub)
					subint.header.comments['TTYPE1']='Length of subintegration'
					subint.header.comments['TFORM1']='Double'
					subint.header.comments['TUNIT1']='Units of field'
					subint.header.comments['TTYPE2']='Offset from Start of subint centre'
					subint.header.comments['TFORM2']='Double'
					subint.header.comments['TUNIT2']='Units of field'
					subint.header.comments['TTYPE3']='Data offset for each channel'
					subint.header.comments['TFORM3']='NCHAN*NPOL floats'
					subint.header.comments['TTYPE4']='Data scale factor for each channel'
					subint.header.comments['TFORM4']='NCHAN*NPOL floats'
					subint.header.comments['TTYPE5']='Subint data table'
					subint.header.comments['TFORM5']='NBIN*NCHAN*NPOL*NSBLK int, byte(B) or bit(X)'
					subint.header.comments['TDIM5']='Dimensions (NBITS or NBIN,NCHAN,NPOL,NSBLK)'
					fitsfile=ps.HDUList([primary,subint])
					fitsfile.writeto(filenames[ifile],overwrite=True)
				else:
					f=ps.open(filenames[ifile],memmap=True,mode='update')
					fd=f['subint'].data[isub]
					fd.setfield('tsubint',np.asarray([nsblk*tsamp]))
					fd.setfield('offs_sub',np.asarray([tline.mean()*tsamp]))
					fd.setfield('dat_offs',dat_offs_array.reshape(-1))
					fd.setfield('dat_scl',dat_scl_array.reshape(-1))
					fd.setfield('data',data.reshape(nsblk,npol,self.values['data']['nchan'],1))
					f.flush()
					f.close()

				
#
def select(widget):
	value=widget['value']
	if widget.get()==value[-1]:
		fname=tf.askopenfilename()
		if fname:
			widget['value']=list(value[:-1])+[fname]
			widget.current(len(value)-1)
		else:
			widget.current(0)
#
class ToolTip(object):
	def __init__(self, widget):
		self.widget = widget
		self.tipwindow = None
		self.id = None
		self.x = self.y = 0
	#
	def showtip(self, text,x0):
		"Display text in tooltip window"
		self.text = text
		if self.tipwindow or not self.text:
			return
		x = self.widget.winfo_rootx()+x0
		y = self.widget.winfo_rooty()
		self.tipwindow = tw = tk.Toplevel(self.widget)
		tw.wm_overrideredirect(1)
		tw.wm_geometry("+%d+%d" % (x, y))
		label = tk.Label(tw, text=self.text,background="#ffffe0", relief=tk.SOLID, borderwidth=1,font=("serif","12","normal"))
		label.pack(ipadx=1)
	#
	def hidetip(self):
		tw = self.tipwindow
		self.tipwindow = None
		if tw:
			tw.destroy()
#
def addpara(frame,text,tp,n,content='',align='x',func=lambda x:0,htext='aaa'):
	label=tk.Label(frame,text=text,bg='white',width=13,font=font1)
	label.grid(row=n,column=0)
	if tp in [str,float,int]:
		def isfloat():
			try: 
				float(widget.get())
				return True
			except: 
				tm.showwarning('','Invalid input!')
				widget.delete(0,'end')
				widget.insert(0,content)
				return False
		#
		def isint():
			try: 
				int(widget.get())
				return True
			except: 
				tm.showwarning('','Invalid input!')
				widget.delete(0,'end')
				widget.insert(0,content)
				return False
		#
		if tp==str:
			widget=tk.Entry(frame,width=10,font=font1)
		elif tp==float:
			widget=tk.Entry(frame,width=10,font=font1,validate='focus',validatecommand=isfloat)
		elif tp==int:
			widget=tk.Entry(frame,width=10,font=font1,validate='focusout',validatecommand=isint)
		widget.delete(0, "end")
		widget.insert(0,content)
		widget.grid(row=n,column=1)
	elif tp is list:
		widget=ttk.Combobox(frame,width=10,font=font1)
		if content not in afiles: widget['value']=content
		else: widget['value']=content+['From file']
		if len(widget['value'])>=1: widget.current(0)
		widget.grid(row=n,column=1)
		widget.bind("<<ComboboxSelected>>",lambda x:func(widget))
	elif tp=='entryorfile':
		widget=ttk.Combobox(frame,width=10,font=font1)
		widget['value']=['From file']
		widget.current(0)
		widget.set(content)
		widget.grid(row=n,column=1)
		widget.bind("<<ComboboxSelected>>",lambda x:func(widget))
	#
	toolTip = ToolTip(widget)
	def enter(event):
		toolTip.showtip(htext,250)
	def leave(event):
		toolTip.hidetip()
	widget.bind('<Enter>', enter)
	widget.bind('<Leave>', leave)
	toolTip = ToolTip(label)
	def enter1(event):
		toolTip.showtip(htext,120)
	def leave1(event):
		toolTip.hidetip()
	label.bind('<Enter>', enter1)
	label.bind('<Leave>', leave1)
	return widget
#
def genfits_psr():
	gbttn1_1['state']='disabled'
	gbttn1_2['state']='disabled'
	gbttn1_3['state']='disabled'
	gbttn1_4['state']='disabled'
	prog1=ttk.Progressbar(pbox1,length=pbox1.bbox()[2])
	prog1.grid(row=3,column=0,columnspan=2,pady=5)
	prog1['maximum']=100
	prog1['value']=0
	prog1.update()
	labeltxt1=tk.StringVar()
	label1=tk.Label(pbox1,textvariable=labeltxt1,bg='white',font=('serif',12))
	label1.grid(row=4,column=0,columnspan=2)
	labeltxt1.set('Preparing Data ...')
	psrfits1.gendata(prog1,labeltxt1)
	prog1.destroy()
	label1.destroy()
	gbttn1_1['state']='normal'
	gbttn1_2['state']='normal'
	gbttn1_3['state']='normal'
	gbttn1_4['state']='normal'
#
stk=tk.Tk()
stk.geometry("1000x600+100+100")
stk.configure(bg='white')
stk.resizable(False,False)
stk.title('Pulsar Data Simulation')
#
font1=('serif',17)
#
style=ttk.Style()
style.theme_create("note",parent="alt",settings={"TNotebook":{"configure":{"tabmargins":[0,0,0,0],"background":"white"}},"TNotebook.Tab":{"configure":{"padding":[5,1],"background":"#CCCCCC"},
            "map":{"background":[("selected","White")],"expand":[("selected",[0,1,0,0])]}}})
style.theme_use('note')
style.configure('.',font=font1)
tabs=ttk.Notebook(stk)
tabs.place(relx=0.01,rely=0.02,relwidth=0.98,relheight=0.96)
#
tab1=tk.Frame(bg='white')
tabs.add(tab1,text=' PSRFITS (PSR) ')
tab2=tk.Frame(bg='white')
tabs.add(tab2,text=' PSRFITS (CAL) ')
tab3=tk.Frame(bg='white')
tabs.add(tab3,text='  LD (PSR)  ')
tab4=tk.Frame(bg='white')
tabs.add(tab4,text='   LD (ToA)   ')
#
# PSRFITS (PSR)
pbox1=tk.Frame(tab1,bg='white')
pbox1.grid(row=0,column=0)
pbox1.pack(side='left',fill='y',pady=20)
pbox1_1=tk.Frame(pbox1,bg='white',height=100)
pbox1_1.grid(row=0,column=0,columnspan=2)
tabs1=ttk.Notebook(pbox1_1)
tabs1.place(relx=0.02,rely=0.01,relwidth=0.96,relheight=0.85)
tabs1.pack(fill='x')
tabs1_1=tk.Frame(bg='white')
tabs1.add(tabs1_1,text='PSR')
tabs1_2=tk.Frame(bg='white')
tabs1.add(tabs1_2,text='Tele')
tabs1_3=tk.Frame(bg='white')
tabs1.add(tabs1_3,text='Obs')
tabs1_4=tk.Frame(bg='white')
tabs1.add(tabs1_4,text='Data')
tabs1_5=tk.Frame(bg='white')
tabs1.add(tabs1_5,text='Prof')
#
fbox1=tk.Frame(tab1,bg='white')
fbox1.pack(side='right',fill='y')
#
psrname1=addpara(tabs1_1,'Pulsar Name:',str,0,'J0000+0000',htext='the name of the pulsar')
psrp01=addpara(tabs1_1,'Period (s):',float,1,'1.0',htext='the pulsar rotating period')
psrdm1=addpara(tabs1_1,'DM:',float,2,0.0,htext='Dispersion Measure')
psrrm1=addpara(tabs1_1,'RM:',float,3,'0.0',htext='Rotation Measure')
psrflux1=addpara(tabs1_1,'Flux (mJy):',float,4,'1.0',htext='the mean radio flux of the pulsar in the selected frequency band')
psrspecind1=addpara(tabs1_1,'Spec. index:',float,5,'0.0',htext='spectra index \N{GREEK SMALL LETTER ALPHA}')
psrnrate1=addpara(tabs1_1,'Null rate:',float,6,'0.0',htext='pulsar nulling rate')
psr1=psr_para(psrname1,psrp01,psrdm1,psrrm1,psrflux1,psrspecind1,psrnrate1)
#
telename1=addpara(tabs1_2,'Tele. Name:',str,0,'FAST',htext='the name of the telescope')
teleaper1=addpara(tabs1_2,'Eff. Aper. (m):',float,1,'300.0',htext='the effective aperture of the telescope')
telesyst1=addpara(tabs1_2,'Sys. Tem. (K):',float,2,'20.0',htext='the system temperature of the telescope')
teleeff1=addpara(tabs1_2,'Efficiency:',float,3,'0.63',htext='the observational efficiency of the telescope')
tele1=tele_para(telename1,teleaper1,telesyst1,teleeff1)
#
obsfreq1=addpara(tabs1_3,'Freq. (MHz):',float,0,'1250.0',htext='the central observational frequency')
obsbw1=addpara(tabs1_3,'BW (MHz):',float,1,'500.0',htext='Band Width')
obsbe1=addpara(tabs1_3,'Edge (MHz):',float,2,'50.0',htext='the width of the band edge')
obsmjd1=addpara(tabs1_3,'MJD:',float,3,'60000.0',htext='the observational date')
obslength1=addpara(tabs1_3,'Length (s):',float,4,'1',htext='the observational duration')
obsnoise1=addpara(tabs1_3,'Noise:',list,5,noiselist,func=select,htext='the frequency domain RFI (Radio Frequency Interference)')
obs1=obs_para(obsfreq1,obsbw1,obsbe1,obsmjd1,obslength1,obsnoise1)
#
datatsamp1=addpara(tabs1_4,'Tsamp (\N{GREEK SMALL LETTER MU}s):',float,0,'49.152',htext='time resolution')
datansblk1=addpara(tabs1_4,'Nsblk:',int,1,'1024',htext='the number of the spectra in one sub-integration')
datansub1=addpara(tabs1_4,'Nsubint:',int,2,'128',htext='the maximum number of sub-integrations in one FITS file')
datanchan1=addpara(tabs1_4,'Nchan:',int,3,'4096',htext='the number of frequency channels in one spectrum')
datapol1=addpara(tabs1_4,'Pol. Type:',list,4,poltypelist,htext='the polarization type in the simulated data')
datacal1=addpara(tabs1_4,'Gain/Phase:',list,5,callist,func=select,htext='the calibration type in the simulated data')
data1=fitsdata_para(datatsamp1,datansblk1,datansub1,datanchan1,datapol1,datacal1)
#
proftype1=addpara(tabs1_5,'Type:',list,0,proftypelist,func=select,htext='the type of the mean pulse profile')
profncomp1=addpara(tabs1_5,'N Comp.:',int,1,'2',htext='the number of the components in the mean pulse profile')
profcompshape1=addpara(tabs1_5,'Comp. shape:',list,2,compshapelist,htext='the shape of the component')
profsingle1=addpara(tabs1_5,'Single pulse:',list,3,singlelist,htext='the drift property of the single pulse')
profpoltype1=addpara(tabs1_5,'Pol. type:',list,4,profpoltypelist,htext='the polarization level of the pulsar')
ppatype1=addpara(tabs1_5,'PPA type:',list,5,ppatypelist,htext='the property of the polarization position angle (PPA) curve')
prof1=prof_para(proftype1,profncomp1,profcompshape1,profsingle1,profpoltype1,ppatype1)
#
psrfits1=psrfits_psr(psr1,data1,obs1,tele1,prof1)
#
fig1=Figure(figsize=(6.6,5.4), dpi=100)
x0,x1,x2,x3=0.1,0.49,0.59,0.98
y0,y1,y2,y3=0.1,0.49,0.58,0.98
ax1=fig1.add_axes([x0,y0,x1-x0,y1-y0])
ax20=fig1.add_axes([x0,y2,x1-x0,(y3-y2)/2])
ax21=fig1.add_axes([x0,y2+(y3-y2)/2,x1-x0,(y3-y2)/2])
ax3=fig1.add_axes([x2,y0,x3-x2,y1-y0])
ax4=fig1.add_axes([x2,y2,x3-x2,y3-y2])
ax1.set_xlabel('Frequency (MHz)',family='serif',fontsize=12)
ax20.set_xlabel('Pulse Phase',family='serif',fontsize=12)
ax21.set_xticks([])
ax3.set_xlabel('Pulse Phase',family='serif',fontsize=12)
ax4.set_xlabel('Pulse Phase',family='serif',fontsize=12)
ax1.set_ylabel('SEFD (Jy)',family='serif',fontsize=12)
ax20.set_ylabel('Flux',family='serif',fontsize=12)
ax21.set_ylabel('PPA ($\degree$)',family='serif',fontsize=12)
ax3.set_ylabel('Frequency (MHz)',family='serif',fontsize=12)
ax4.set_ylabel('Pulse Index',family='serif',fontsize=12)
canvas1=FigureCanvasTkAgg(fig1,master=fbox1)
canvas1.get_tk_widget().grid(row=0,column=1)  
#
def update_fits_psr():
	psrfits1.update()
	if psrfits1.check(): return True
	psrfits1.genpsr()
	plot_fits_psr()
#
def plot_fits_psr():
	ax1.cla()
	ax20.cla()
	ax21.cla()
	ax3.cla()
	ax4.cla()
	fac=psrfits1.profile[0].max()
	ax1.plot(psrfits1.freqs,psrfits1.gain+psrfits1.noise,'k-')
	ax20.plot(psrfits1.phase,psrfits1.profile[0]/fac,'k-')
	ax20.plot(psrfits1.phase,psrfits1.profile[4]/fac,'b-')
	ax20.plot(psrfits1.phase,psrfits1.profile[3]/fac,'r-')
	ax21.plot(psrfits1.phase,psrfits1.profile[5]-90,'y--')
	ax3.imshow(psrfits1.fdomain,origin='lower',aspect='auto',extent=(0,1,psrfits1.freq_start,psrfits1.freq_end))
	ax4.imshow(psrfits1.tdomain,origin='lower',aspect='auto',extent=(0,1,0,psrfits1.nsingle))
	ax1.set_xlabel('Frequency (MHz)',family='serif',fontsize=12)
	ax20.set_xlabel('Pulse Phase',family='serif',fontsize=12)
	ax21.set_xticks([])
	ax3.set_xlabel('Pulse Phase',family='serif',fontsize=12)
	ax4.set_xlabel('Pulse Phase',family='serif',fontsize=12)
	ax1.set_ylabel('SEFD (Jy)',family='serif',fontsize=12)
	ax20.set_ylabel('Flux',family='serif',fontsize=12)
	ax21.set_ylabel('PPA ($\degree$)',family='serif',fontsize=12)
	ax3.set_ylabel('Frequency (MHz)',family='serif',fontsize=12)
	ax4.set_ylabel('Pulse Index',family='serif',fontsize=12)
	canvas1.draw()
#
def tolist(a):
	b=[]
	for i in range(len(a)):
		if type(a[i])==np.ndarray: b.append(a[i].tolist())
		else: b.append(a[i])
	return b
#
def savep_fits_psr():
	dic=psrfits1.values
	dic['noise factors']=tolist(psrfits1.nfac)
	dic['single factors']=tolist(psrfits1.sfac)
	dic['profile factors']=tolist(psrfits1.pfac)
	dic['calibration factors']=tolist(psrfits1.calpara)
	f=tf.asksaveasfile(mode='w')
	if f:
		f.write(json.dumps(dic,indent=2))
		f.close()
#
def loadp_fits_psr():
	f=tf.askopenfile()
	if f: dic=json.load(f)
	else: return
	if psrfits1.loadpsr(dic): return
	if psrfits1.check(): return
	plot_fits_psr()
#
gbttn1_1=tk.Button(pbox1,text='Save Para.',command=lambda:savep_fits_psr(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn1_1.grid(row=1,column=0,pady=10)
gbttn1_2=tk.Button(pbox1,text='Load Para.',command=lambda:loadp_fits_psr(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn1_2.grid(row=2,column=0)
gbttn1_3=tk.Button(pbox1,text='Apply',command=lambda:update_fits_psr(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn1_3.grid(row=1,column=1,pady=10)
gbttn1_4=tk.Button(pbox1,text='Generate',command=lambda:genfits_psr(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn1_4.grid(row=2,column=1)
#
#
#
def gen_calprof(duty,ppa,start,period,nbin):
	prof=np.zeros([4,nbin])
	prof0=np.zeros(nbin*2)
	sphase=start/period%1*nbin
	sbin=int(np.ceil(sphase))
	resi0=sbin-sphase
	ephase=sphase+duty/100*nbin
	ebin=int(np.floor(ephase))
	resi1=ephase-ebin
	prof0[sbin:ebin]=1
	prof0[sbin-1]=resi0
	prof0[ebin+1]=resi1
	prof[0]=prof0[:nbin]+prof0[nbin:]
	ppa=ppa/90*np.pi
	prof[1]=prof[0]*np.cos(ppa)
	prof[2]=prof[0]*np.sin(ppa)
	return prof
#
class psrfits_cal:
	def __init__(self,cal,tele,obs,data):
		self.cal=cal
		self.data=data
		self.obs=obs
		self.tele=tele
		self.keys={'cal':[('period',float),('duty',float),('start',float),('duration',float),('ppa',float),('temperature',float)],
			'obs':[('freq',float),('bw',float),('be',float),('mjd',float),('length',float),('noise',str)],
			'tele':[('name',str),('aperture',float),('temperature',float),('efficiency',float)],
			'data':[('tsamp',float),('nsblk',int),('nsub',int),('nchan',int),('poltype',str),('cal',str)]}
	#
	def update(self,update=True):
		values=dict()
		for i in self.keys.keys():
			values[i]=dict()
			for k,f in self.keys[i]:
				values[i][k]=f(self.__getattribute__(i).__getattribute__(k).get())
		if update: 
			self.values=values
			self.mode='input'
		else: return values
	#
	def check(self):
		def warning(str):
			tm.showwarning('Warning!',str)
			return True
		#
		acclen=self.values['data']['tsamp']*self.values['obs']['bw']/self.values['data']['nchan']
		if acclen<1: return warning('The time and frequency resolution is too high!')
		#
		try: af.reco(self.values['tele']['name'])
		except: return warning('Unknown Telescope!')
		#
		if self.values['obs']['noise'] not in noiselist and os.path.isfile(self.values['obs']['noise']):
			try: zchan=np.loadtxt(self.values['obs']['noise'],dtype=int)
			except: return warning('The noise file cannot be recognized!')
		#
		if self.values['data']['cal'] not in callist and os.path.isfile(self.values['data']['cal']):
			try:
				d=ld.ld(self.values['data']['cal'])
				if d.read_para('mode')!='cal': warning('LD file is not caliration file!')
				elif d.read_para('cal_mode')!='seg': warning('Calibration file can only contain 1 segment!')
				elif d.read_shape()[1]!=1: warning('Calibration file can only contain 1 segment!')
			except: return warning('The calibration file cannot be recognized!')
		#
		if self.values['cal']['duty']>100: return warning('The duty cycle of the calibration signal cannot surpass 100%!')
	#
	def gencal(self,fac=0):
		self.freq_start=self.values['obs']['freq']-self.values['obs']['bw']/2
		self.freq_end=self.values['obs']['freq']+self.values['obs']['bw']/2
		self.freqs=np.arange(self.freq_start,self.freq_end,self.values['obs']['bw']/self.values['data']['nchan'])
		self.gain=np.ones(self.values['data']['nchan'])
		if self.values['obs']['be']>0:
			j0=(self.freqs-self.freq_start)<self.values['obs']['be']
			self.gain[j0]=np.sin((self.freqs[j0]-self.freq_start)/self.values['obs']['be']*np.pi/2)**2
			j1=(self.freqs-self.freq_end)>(-self.values['obs']['be'])
			self.gain[j1]=np.sin((self.freqs[j1]-self.freq_end)/self.values['obs']['be']*np.pi/2)**2
		self.nsingle=100
		nbin=4096
		self.phase=np.arange(nbin)/nbin
		self.sefd=self.values['tele']['temperature']/(self.values['tele']['aperture']**2*2.84e-4*self.values['tele']['efficiency'])
		self.profile=gen_calprof(self.values['cal']['duty'],self.values['cal']['ppa'],self.values['cal']['start'],self.values['cal']['period'],nbin)
		prof0=np.fft.irfft(np.fft.rfft(self.profile[0])[:65])
		if fac:
			self.noise,self.nfac=gen_noise(self.values['obs']['noise'],self.values['data']['nchan'],self.gain,nfac0=self.nfac)
			self.calib,self.calpara=gen_cal(self.values['data']['cal'],self.freqs,self.gain,cfac=self.calpara)
		else:
			self.noise,self.nfac=gen_noise(self.values['obs']['noise'],self.values['data']['nchan'],self.gain)
			self.calib,self.calpara=gen_cal(self.values['data']['cal'],self.freqs,self.gain)
		self.fdomain=np.ones(128).reshape(-1,1)@prof0.reshape(1,-1)
	#
	def loadcal(self,dic):
		self.nfac=dic.pop('noise factors')
		self.calpara=dic.pop('calibration factors')
		self.values=dic
		if self.refresh(): return True
		self.mode='load'
		self.gencal(fac=1)
	#
	def refresh(self):
		for i in self.keys.keys():
			for k,f in self.keys[i]:
				try:
					value=f(self.values[i][k])
				except:
					tm.showwarning('Warning','The parameter '+i+':'+k+' is invalid!')
					return True
				widget=self.__getattribute__(i).__getattribute__(k)
				if widget.winfo_class()=='TCombobox':
					values=widget['value']
					if list(values)[:-1] not in afiles and self.values[i][k] not in values:
						tm.showwarning('Warning','The selected value '+i+':'+k+' is invalid!')
						return True
		for i in self.keys.keys():
			for k,f in self.keys[i]:
				widget=self.__getattribute__(i).__getattribute__(k)
				if widget.winfo_class()=='Entry':
					widget.delete(0,'end')
					widget.insert(0,str(self.values[i][k]))
				elif widget.winfo_class()=='TCombobox':
					values=widget['value']
					if self.values[i][k] in values and self.values[i][k]!='From file':
						widget.current(values.index(self.values[i][k]))
					elif list(values)[:-1] in afiles:
						widget['value']=list(values[:-1])+[self.values[i][k]]
						widget.current(len(values)-1)
	#
	def gendata(self,prog=0,label=0):
		values=self.update(False)
		if values!=self.values: 
			if update_fits_cal(): return
		#
		tsamp=self.values['data']['tsamp']*1e-6
		nbin0=2**int(np.log2(self.values['cal']['period']/tsamp)+1)
		#
		period=self.values['cal']['period']
		duty=self.values['cal']['duty']
		duration=self.values['cal']['duration']
		profile=self.profile
		if nbin0>4096:
			fprof=np.fft.rfft(profile,axis=1)
			fprof0=np.zeros([4,int(nbin0/2)+1],dtype=np.complex128)
			fprof0[:,:(fprof.shape[1])]=fprof
			profile=np.fft.irfft(fprof0,axis=1)
		else:
			profile=profile.reshape(4,nbin0,-1).mean(2)
		#
		profile/=profile[0].mean()
		nsblk=self.values['data']['nsblk']
		sublen=tsamp*nsblk
		nsub=int(self.values['obs']['length']/sublen)
		nfile=int(np.ceil(nsub/self.values['data']['nsub']))
		#
		if self.values['data']['poltype'] in ['AABBCRCI','IQUV']: npol=4
		elif self.values['data']['poltype']=='AABB': npol=2
		#
		fsize=(self.values['data']['nchan']*npol*(nsblk+8)+16)*nsub/2**30
		if fsize>1:
			answer=tm.askokcancel('Warning','The total file size is larger than '+str(int(fsize))+'G, please determine whether to save them.')
			if not answer: return	
		#
		while True:
			filename=tf.asksaveasfilename(filetypes=[('FITS file','.fits')])
			if not filename: return
			if len(filename)>5:
				if filename[-5:]=='.fits': filename=filename[:-5]
			if nfile==1:
				if os.path.isfile(filename+'.fits'):
					answer=tm.askyesnocancel('Warning','File exists!\n\nReplace? \'No\' for rename the file')
					if answer: pass
					elif answer is None: return
					else: continue
				filenames=[filename+'.fits']
				break
			mark=False
			for i in (np.arange(nfile)+1):
				if os.path.isfile(filename+'_'+str(i).zfill(4)+'.fits'):
					mark=True
					break
			if mark:
				answer=tm.askyesnocancel('Warning','File exists!\n\nReplace? \'No\' for rename the file')
				if answer: pass
				elif answer is None: return
				else: continue
			filenames=list(map(lambda x:filename+'_'+str(x).zfill(4)+'.fits',np.arange(nfile)+1))
			break
		#
		stt_imjd=int(self.values['obs']['mjd'])
		stt_smjd=int(self.values['obs']['mjd']%1*86400)
		stt_offs=self.values['obs']['mjd']%1*86400%1
		#
		header=ps.Header()
		header.append(('SIMPLE',True,'file does conform to FITS standard'))
		header.append(('NAXIS',0,'number of data axes'))
		header.append(('EXTEND',True,'FITS dataset may contain extensions'))
		header.append(('OBS_MODE','CAL','(PSR, CAL, SEARCH)'))
		header.append(('CAL_MODE','SYNC','Cal mode (OFF, SYNC, EXT1, EXT2)'))
		header.append(('CAL_FREQ',1/period,'[Hz] Cal modulation frequency (E)'))
		header.append(('CAL_DCYC',duty/100,'Cal duty cycle (E)'))
		header.append(('CAL_PHS',self.values['cal']['start'],'Cal phase (wrt start time) (E)'))
		header.append(('TELESCOP',self.values['tele']['name'],'Telescope name'))
		header.append(('OBSFREQ',self.values['obs']['freq'],'[MHz] Centre frequency for observation'))
		header.append(('OBSBW',self.values['obs']['bw'],'[MHz] Bandwidth for observation'))
		header.append(('OBSNCHAN',self.values['data']['nchan'],'Number of frequency channels (original)'))
		header.append(('STT_IMJD',stt_imjd,'Start MJD (UTC days) (J - long integer)'))
		header.append(('STT_SMJD',stt_smjd,'[s] Start time (sec past UTC 00h) (J)'))
		header.append(('STT_OFFS',stt_offs,'[s] Start time offset (D)'))
		primary=ps.PrimaryHDU(header=header)
		subint_header=ps.Header()
		subint_header.append(('XTENSION','BINTABLE','***** Subintegration data  *****'))
		subint_header.append(('EXTNAME','SUBINT','name of this binary table extension'))
		subint_header.append(('NAXIS',2,'2-dimensional binary table'))
		subint_header.append(('NAXIS1',int(self.values['data']['nchan']*npol*nsblk+4*self.values['data']['nchan']*npol*2+2*8),'width of table in bytes'))
		subint_header.append(('NAXIS2',self.values['data']['nsub'],'Number of rows in table (NSUBINT)'))
		subint_header.append(('NPOL',npol,'Nr of polarisations'))
		subint_header.append(('NCHAN',self.values['data']['nchan'],'Number of channels/sub-bands in this file'))
		subint_header.append(('CHAN_BW',self.values['obs']['bw']/self.values['data']['nchan'],'[MHz] Channel/sub-band width'))
		subint_header.append(('TBIN',tsamp,'[s] Time per bin or sample'))
		subint_header.append(('NSBLK',nsblk,'Samples/row (SEARCH mode, else 1)'))
		subint_header.append(('POL_TYPE',self.values['data']['poltype'],'Polarisation identifier (e.g., AABBCRCI, AA+BB)'))
		subint_header.append(('TFIELDS',5,'Number of fields per row'))
		subint_header.append(('INT_TYPE','TIME','Time axis (TIME, BINPHSPERI, BINLNGASC, etc)'))
		subint_header.append(('INT_UNIT','SEC','Unit of time axis (SEC, PHS (0-1), DEG)'))
		#
		nperiod=int(self.values['obs']['length']//period+2)
		#
		single=np.zeros(nperiod*nbin0)
		single[int(start/period*nbin0):int((start+duration)/period*nbin0)]=1
		single[int((start+duration)/period*nbin0+1)]=(start+duration)/period*nbin0%1
		single=single*(profile*self.values['cal']['temperature']/(self.values['tele']['aperture']**2*2.84e-4*self.values['tele']['efficiency'])).reshape(4,1,-1)
		acclen=tsamp*(self.freq_end-self.freq_start)*1e6
		#
		a12,a22,cdphi,sdphi=self.calib
		#
		for ifile in np.arange(nfile):
			if ifile==nfile-1:
				fnsub=nsub-(nfile-1)*self.values['data']['nsub']
			else:
				fnsub=self.values['data']['nsub']
			for isub in np.arange(fnsub):
				ksub=int(isub+ifile*self.values['data']['nsub'])
				if prog:
					prog['value']=ksub/nsub*100
					label.set(str(ksub)+'/'+str(nsub)+' sub-integrations ('+str(round(ksub/nsub*100,1))+'%)')
					prog.update()
				tline=ksub*nsblk+np.arange(nsblk)
				fline=self.freqs
				phase=(tline.reshape(1,-1)*tsamp-dm/fline.reshape(-1,1)**2*pm.dm_const)/period
				#
				phase-=phase_start
				pstart,pend=int(phase.min()),int(phase.max())+1
				if pend<=0: continue
				elif pstart<0: pstart=0
				phase0=np.arange(pstart,pend,1/nbin0)
				data=np.zeros([nsblk,4,self.values['data']['nchan']])
				for i in np.arange(self.values['data']['nchan']):
					d0=single[:4,pstart:pend].reshape(4,-1)*self.gain[i]
					for k in np.arange(4):
						data[:,k,i]=np.interp(phase[i],phase0,d0[k])
				#
				if self.values['data']['poltype'][0:4]=='AABB':
					data[:,0],data[:,1]=(data[:,0]+data[:,1])/2,(data[:,0]-data[:,1])/2
				data=data[:,:npol]
				#
				baseline=np.zeros([nsblk,npol,self.values['data']['nchan']])
				if self.values['data']['poltype'][0:4]=='AABB':
					baseline[:,:2]=ss.chi2.rvs(2*acclen,size=2*self.values['data']['nchan']*nsblk).reshape(nsblk,2,-1)/2/acclen
				else:
					baseline[:,0]=ss.chi2.rvs(4*acclen,size=self.values['data']['nchan']*nsblk).reshape(nsblk,-1)/4/acclen
					baseline[:,1]=nr.randn(self.values['data']['nchan']*nsblk).reshape(nsblk,-1)/np.sqrt(acclen*2)+1
				if self.values['data']['poltype']=='AABBCRCI':
					baseline[:,2:]=nr.randn(2*self.values['data']['nchan']*nsblk).reshape(nsblk,2,-1)/np.sqrt(acclen)+1
				elif self.values['data']['poltype']=='IQUV':
					baseline[:,2:]=nr.randn(2*self.values['data']['nchan']*nsblk).reshape(nsblk,2,-1)/np.sqrt(acclen)*2+1
				#
				noise=self.noise.reshape(1,-1).repeat(npol,axis=0)
				if self.values['data']['cal']!='None':
					if self.values['data']['poltype'] in ['AABBCRCI','IQUV']:
						data[:,2],data[:,3]=(cdphi*data[:,2]-sdphi*data[:,3],sdphi*data[:,2]+cdphi*data[:,3])*np.sqrt(a12*a22)
					if self.values['data']['poltype']=='IQUV':
						data[:,0],data[:,1]=a12*data[:,0]+a22*data[:,1],a12*data[:,0]-a22*data[:,1]
						noise[0]*=(a12+a22)
						noise[1]*=(a12-a22)
					else:
						data[:,0]*=a12
						data[:,1]*=a22
						noise[0]*=a12
						noise[1]*=a22
				#
				data+=baseline*(noise+self.gain)*self.sefd
				dat_offs_array=data.min(0)
				data-=dat_offs_array
				dat_scl_array=data.max(0)
				dsj=(dat_scl_array!=0)
				data[:,dsj]=np.int16(np.round((data[:,dsj]/dat_scl_array[dsj]*256-0.5)))
				data[:,dat_scl_array==0]=0
				data[data<0]=0
				data[data>255]=255
				data=np.uint8(data)
				dat_offs_array+=dat_scl_array*0.5
				if isub==0:
					tsubint=ps.Column(name='TSUBINT',format='1D',unit='s',array=np.asarray([nsblk*tsamp]))
					offs_sub=ps.Column(name='OFFS_SUB',format='1D',unit='s',array=np.asarray([tline.mean()*tsamp]))
					dat_offs=ps.Column(name='DAT_OFFS',format=str(self.values['data']['nchan']*npol)+'E',array=dat_offs_array)
					dat_scl=ps.Column(name='DAT_SCL',format=str(self.values['data']['nchan']*npol)+'E',array=dat_scl_array)
					dim='(1,'+str(self.values['data']['nchan'])+','+str(npol)+','+str(nsblk)+')'
					data=ps.Column(name='DATA',format=str(self.values['data']['nchan']*npol*nsblk)+'B',dim=dim,array=data.reshape(1,-1))
					cols=ps.ColDefs([tsubint,offs_sub,dat_offs,dat_scl,data])
					subint=ps.BinTableHDU.from_columns(cols,header=subint_header,nrows=fnsub)
					subint.header.comments['TTYPE1']='Length of subintegration'
					subint.header.comments['TFORM1']='Double'
					subint.header.comments['TUNIT1']='Units of field'
					subint.header.comments['TTYPE2']='Offset from Start of subint centre'
					subint.header.comments['TFORM2']='Double'
					subint.header.comments['TUNIT2']='Units of field'
					subint.header.comments['TTYPE3']='Data offset for each channel'
					subint.header.comments['TFORM3']='NCHAN*NPOL floats'
					subint.header.comments['TTYPE4']='Data scale factor for each channel'
					subint.header.comments['TFORM4']='NCHAN*NPOL floats'
					subint.header.comments['TTYPE5']='Subint data table'
					subint.header.comments['TFORM5']='NBIN*NCHAN*NPOL*NSBLK int, byte(B) or bit(X)'
					subint.header.comments['TDIM5']='Dimensions (NBITS or NBIN,NCHAN,NPOL,NSBLK)'
					fitsfile=ps.HDUList([primary,subint])
					fitsfile.writeto(filenames[ifile],overwrite=True)
				else:
					f=ps.open(filenames[ifile],memmap=True,mode='update')
					fd=f['subint'].data[isub]
					fd.setfield('tsubint',np.asarray([nsblk*tsamp]))
					fd.setfield('offs_sub',np.asarray([tline.mean()*tsamp]))
					fd.setfield('dat_offs',dat_offs_array.reshape(-1))
					fd.setfield('dat_scl',dat_scl_array.reshape(-1))
					fd.setfield('data',data.reshape(nsblk,npol,self.values['data']['nchan'],1))
					f.flush()
					f.close()

				
#
#PSRFITS (CAL)
pbox2=tk.Frame(tab2,bg='white')
pbox2.grid(row=0,column=0)
pbox2.pack(side='left',fill='y',pady=20)
pbox2_1=tk.Frame(pbox2,bg='white',height=100)
pbox2_1.grid(row=0,column=0,columnspan=2)
tabs2=ttk.Notebook(pbox2_1)
tabs2.place(relx=0.02,rely=0.01,relwidth=0.96,relheight=0.85)
tabs2.pack(fill='x')
tabs2_1=tk.Frame(bg='white')
tabs2.add(tabs2_1,text='Cal')
tabs2_2=tk.Frame(bg='white')
tabs2.add(tabs2_2,text='Tele')
tabs2_3=tk.Frame(bg='white')
tabs2.add(tabs2_3,text='Obs')
tabs2_4=tk.Frame(bg='white')
tabs2.add(tabs2_4,text='Data')
#
calperiod2=addpara(tabs2_1,'Period (s):',float,0,'1.0',htext='the modulation period of the injected calibration signal')
calduty2=addpara(tabs2_1,'Duty Cyc. (%):',float,1,'50.0',htext='duty cycle for the injected calibration signal')
calstart2=addpara(tabs2_1,'Start (s):',float,2,'0.0',htext='time of the leading edge of the injected calibration signal')
calduration2=addpara(tabs2_1,'Duration (s):',float,3,'-1.0',htext='duration of the calibration signal, -1 for lasting to the end of the data')
calppa2=addpara(tabs2_1,'PPA ('+chr(176)+'):',float,4,'45.0',htext='Polarization Angle of the calibration signal')
caltemperature2=addpara(tabs2_1,'Temper. (K):',float,5,'10.0',htext='equivalent temperature of the calibration signal')
cal2=cal_para(calperiod2,calduty2,calstart2,calduration2,calppa2,caltemperature2)
#
telename2=addpara(tabs2_2,'Tele. Name:',str,0,'FAST',htext='the name of the telescope')
teleaper2=addpara(tabs2_2,'Eff. Aper. (m):',float,1,'300.0',htext='the effective aperture of the telescope')
telesyst2=addpara(tabs2_2,'Sys. Tem. (K):',float,2,'20.0',htext='the system temperature of the telescope')
teleeff2=addpara(tabs2_2,'Efficiency:',float,3,'0.63',htext='the observational efficiency of the telescope')
tele2=tele_para(telename2,teleaper2,telesyst2,teleeff2)
#
obsfreq2=addpara(tabs2_3,'Freq. (MHz):',float,0,'1250.0',htext='the central observational frequency')
obsbw2=addpara(tabs2_3,'BW (MHz):',float,1,'500.0',htext='Band Width')
obsbe2=addpara(tabs2_3,'Edge (MHz):',float,2,'50.0',htext='the width of the band edge')
obsmjd2=addpara(tabs2_3,'MJD:',float,3,'60000.0',htext='the observational date')
obslength2=addpara(tabs2_3,'Length (s):',float,4,'1',htext='the observational duration')
obsnoise2=addpara(tabs2_3,'Noise:',list,5,noiselist,func=select,htext='the frequency domain RFI (Radio Frequency Interference)')
obs2=obs_para(obsfreq2,obsbw2,obsbe2,obsmjd2,obslength2,obsnoise2)
#
datatsamp2=addpara(tabs2_4,'Tsamp (\N{GREEK SMALL LETTER MU}s):',float,0,'49.152',htext='time resolution')
datansblk2=addpara(tabs2_4,'Nsblk:',int,1,'1024',htext='the number of the spectra in one sub-integration')
datansub2=addpara(tabs2_4,'Nsubint:',int,2,'128',htext='the maximum number of sub-integrations in one FITS file')
datanchan2=addpara(tabs2_4,'Nchan:',int,3,'4096',htext='the number of frequency channels in one spectrum')
datapol2=addpara(tabs2_4,'Pol. Type:',list,4,poltypelist,htext='the polarization type in the simulated data')
datacal2=addpara(tabs2_4,'Gain/Phase:',list,5,callist,func=select,htext='the calibration type in the simulated data')
data2=fitsdata_para(datatsamp2,datansblk2,datansub2,datanchan2,datapol2,datacal2)
#
psrfits2=psrfits_cal(cal2,tele2,obs2,data2)
#
fbox2=tk.Frame(tab2,bg='white')
fbox2.pack(side='right',fill='y')
fig2=Figure(figsize=(6.6,5.4), dpi=100)
x0,x1,x2,x3=0.1,0.49,0.59,0.98
y0,y1,y2,y3=0.1,0.49,0.58,0.98
ax5=fig2.add_axes([x0,y0,x1-x0,y1-y0])
ax6=fig2.add_axes([x0,y2,x1-x0,y3-y2])
ax7=fig2.add_axes([x2,y0,x3-x2,y1-y0])
ax8=fig2.add_axes([x2,y2,x3-x2,y3-y2])
ax5.set_xlabel('Frequency (MHz)',family='serif',fontsize=12)
ax6.set_xlabel('Pulse Phase',family='serif',fontsize=12)
ax7.set_xlabel('Pulse Phase',family='serif',fontsize=12)
ax8.set_xlabel('Frequency (MHz)',family='serif',fontsize=12)
ax5.set_ylabel('SEFD (Jy)',family='serif',fontsize=12)
ax6.set_ylabel('Flux (arbi.)',family='serif',fontsize=12)
ax7.set_ylabel('Frequency (MHz)',family='serif',fontsize=12)
ax8.set_ylabel('Cal. Para.',family='serif',fontsize=12)
canvas2=FigureCanvasTkAgg(fig2,master=fbox2)
canvas2.get_tk_widget().grid(row=0,column=1)
#
def update_fits_cal():
	psrfits2.update()
	if psrfits2.check(): return True
	psrfits2.gencal()
	plot_fits_cal()
#
def genfits_cal():
	gbttn2_1['state']='disabled'
	gbttn2_2['state']='disabled'
	gbttn2_3['state']='disabled'
	gbttn2_4['state']='disabled'
	prog2=ttk.Progressbar(pbox2,length=pbox2.bbox()[2])
	prog2.grid(row=3,column=0,columnspan=2,pady=5)
	prog2['maximum']=100
	prog2['value']=0
	prog2.update()
	labeltxt2=tk.StringVar()
	label2=tk.Label(pbox2,textvariable=labeltxt2,bg='white',font=('serif',12))
	label2.grid(row=4,column=0,columnspan=2)
	labeltxt2.set('Preparing Data ...')
	psrfits2.gendata(prog2,labeltxt2)
	prog2.destroy()
	label2.destroy()
	gbttn2_1['state']='normal'
	gbttn2_2['state']='normal'
	gbttn2_3['state']='normal'
	gbttn2_4['state']='normal'
#
def plot_fits_cal():
	ax5.cla()
	ax6.cla()
	ax7.cla()
	ax8.cla()
	fac=psrfits2.profile[0].max()
	ax5.plot(psrfits2.freqs,psrfits2.gain+psrfits2.noise,'k-')
	ax6.plot(psrfits2.phase,psrfits2.profile[0]/fac,'k-')
	ax6.plot(psrfits2.phase,psrfits2.profile[1]/fac,'b-')
	ax6.plot(psrfits2.phase,psrfits2.profile[2]/fac,'r-')
	ax7.imshow(psrfits2.fdomain,origin='lower',aspect='auto',extent=(0,1,psrfits2.freq_start,psrfits2.freq_end))
	ax8.plot(psrfits2.freqs,psrfits2.calib.T)
	ax5.set_xlabel('Frequency (MHz)',family='serif',fontsize=12)
	ax6.set_xlabel('Pulse Phase',family='serif',fontsize=12)
	ax7.set_xlabel('Pulse Phase',family='serif',fontsize=12)
	ax8.set_xlabel('Frequency (MHz)',family='serif',fontsize=12)
	ax5.set_ylabel('SEFD (Jy)',family='serif',fontsize=12)
	ax6.set_ylabel('Flux (arbi.)',family='serif',fontsize=12)
	ax7.set_ylabel('Frequency (MHz)',family='serif',fontsize=12)
	ax8.set_ylabel('Cal. Para.',family='serif',fontsize=12)
	canvas2.draw()
#
def savep_fits_cal():
	dic=psrfits2.values
	dic['noise factors']=tolist(psrfits2.nfac)
	dic['calibration factors']=tolist(psrfits2.calpara)
	f=tf.asksaveasfile(mode='w')
	if f:
		f.write(json.dumps(dic,indent=2))
		f.close()
#
def loadp_fits_cal():
	f=tf.askopenfile()
	if f: dic=json.load(f)
	else: return
	if psrfits2.loadpsr(dic): return
	if psrfits2.check(): return
	plot_fits_cal()
#
gbttn2_1=tk.Button(pbox2,text='Save Para.',command=lambda:savep_fits_cal(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn2_1.grid(row=1,column=0,pady=10)
gbttn2_2=tk.Button(pbox2,text='Load Para.',command=lambda:loadp_fits_cal(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn2_2.grid(row=2,column=0)
gbttn2_3=tk.Button(pbox2,text='Apply',command=lambda:update_fits_cal(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn2_3.grid(row=1,column=1,pady=10)
gbttn2_4=tk.Button(pbox2,text='Generate',command=lambda:genfits_cal(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn2_4.grid(row=2,column=1)
#
class ld_psr:
	def __init__(self,psr,fitsdata,obs,tele,prof):
		self.psr=psr
		self.data=fitsdata
		self.obs=obs
		self.tele=tele
		self.prof=prof
		self.keys={'psr':[('name',str),('p0',float),('dm',float),('rm',float),('flux',float),('specind',float),('nrate',float)],
			'obs':[('freq',float),('bw',float),('be',float),('mjd',float),('length',float),('noise',str)],
			'tele':[('name',str),('aperture',float),('temperature',float),('efficiency',float)],
			'prof':[('proftype',str),('ncomp',int),('compshape',str),('single',str),('poltype',str),('ppatype',str)],
			'data':[('nbin',int),('nchan',int),('sub_nperiod',int),('poltype',str),('cal',str)]}
	#
	def update(self,update=True):
		values=dict()
		for i in self.keys.keys():
			values[i]=dict()
			for k,f in self.keys[i]:
				values[i][k]=f(self.__getattribute__(i).__getattribute__(k).get())
		if update: 
			self.values=values
			self.mode='input'
		else: return values
	#
	def check(self):
		def warning(str):
			tm.showwarning('Warning!',str)
			return True
		#
		try: af.reco(self.values['tele']['name'])
		except: return warning('Unknown Telescope!')
		try:
			psr=pr.psr(self.values['psr']['name'])
			mode='eph'
		except:
			try:
				psr=pr.psr(self.values['psr']['name'],parfile=True)
				mode='eph'
			except:
				mode='pdm'
		if mode=='eph':
			time=te.times(te.time(self.values['obs']['mjd'],0,scale=self.values['tele']['name']))
			tran,l=np.array(time.obs(psr,5)).reshape(-1)			
			if tran is None:
				if l==0: return warning('The pulsar cannot be observed with the specified telescope!')
			else:
				if self.values['obs']['length']>l*86400: return warning('The pulsar cannot be monitored for a such long integration time with the specified telescope!')
				elif tran>l/2 or (tran+l/2)*86400<self.values['obs']['length']:
					return warning('The pulsar cannot be observed for the given observation time! A proper start time could be between '+str(self.values['obs']['mjd']+tran-l/2)+' and '+str(self.values['obs']['mjd']+tran+l/2)+'.')
		#
		if self.values['obs']['noise'] not in noiselist and os.path.isfile(self.values['obs']['noise']):
			try: zchan=np.loadtxt(self.values['obs']['noise'],dtype=int)
			except: return warning('The noise file cannot be recognized!')
		#
		if self.values['data']['cal'] not in callist and os.path.isfile(self.values['data']['cal']):
			try:
				d=ld.ld(self.values['data']['cal'])
				if d.read_para('mode')!='cal': warning('LD file is not caliration file!')
				elif d.read_para('cal_mode')!='seg': warning('Calibration file can only contain 1 segment!')
				elif d.read_shape()[1]!=1: warning('Calibration file can only contain 1 segment!')
			except: return warning('The calibration file cannot be recognized!')
		#
		if self.values['prof']['proftype'] not in proftypelist and os.path.isfile(self.values['prof']['proftype']):
			try:
				d=ld.ld(self.values['prof']['proftype'])
				if d.read_para('mode') not in ['single','subint','template']: warning('LD file does not contain a pulse profile!')
				elif d.read_shape()[1]!=1: warning('LD file does contain more than one pulse profiles!')
			except: return warning('The profile file cannot be recognized!')
		#
		if self.values['psr']['nrate']>=1 or self.values['psr']['nrate']<0: warning('The nulling rate is illegal!')
	#
	def genpsr(self,fac=0):
		self.freq_start=self.values['obs']['freq']-self.values['obs']['bw']/2
		self.freq_end=self.values['obs']['freq']+self.values['obs']['bw']/2
		self.freqs=np.arange(self.freq_start,self.freq_end,self.values['obs']['bw']/self.values['data']['nchan'])
		self.gain=np.ones(self.values['data']['nchan'])
		if self.values['obs']['be']>0:
			j0=(self.freqs-self.freq_start)<self.values['obs']['be']
			self.gain[j0]=np.sin((self.freqs[j0]-self.freq_start)/self.values['obs']['be']*np.pi/2)**2
			j1=(self.freqs-self.freq_end)>(-self.values['obs']['be'])
			self.gain[j1]=np.sin((self.freqs[j1]-self.freq_end)/self.values['obs']['be']*np.pi/2)**2
		self.nsingle=100
		nbin=self.values['data']['nbin']
		self.phase=np.arange(nbin)/nbin
		self.sefd=self.values['tele']['temperature']/(self.values['tele']['aperture']**2*2.84e-4*self.values['tele']['efficiency'])
		a=self.values['psr']['specind']
		if a!=1: self.speccoeff=self.values['psr']['flux']*self.values['obs']['bw']*(1-a)/(self.freq_end**(1-a)-self.freq_start**(1-a))
		else: self.speccoeff=self.values['psr']['flux']*self.values['obs']['bw']/np.log(self.freq_end/self.freq_start)
		self.spec=self.freqs**(-a)*self.speccoeff
		if fac:
			self.noise,self.nfac=gen_noise(self.values['obs']['noise'],self.values['data']['nchan'],self.gain,nfac0=self.nfac)
			self.profile,self.pfac=gen_profile(self.values['prof']['proftype'],self.values['prof']['ncomp'],self.values['prof']['compshape'],self.values['prof']['poltype'],self.values['prof']['ppatype'],nbin,pfac=self.pfac)
			prof0=np.fft.irfft(np.fft.rfft(self.profile[0])[:65])
			self.tdomain,self.sfac=gen_single(prof0,self.values['prof']['single'],self.nsingle,self.values['psr']['nrate'],self.sfac,sfac=True)
			self.tdomain*=prof0
			self.cal,self.calpara=gen_cal(self.values['data']['cal'],self.freqs,self.gain,cfac=self.calpara)
		else:
			self.noise,self.nfac=gen_noise(self.values['obs']['noise'],self.values['data']['nchan'],self.gain)
			self.profile,self.pfac=gen_profile(self.values['prof']['proftype'],self.values['prof']['ncomp'],self.values['prof']['compshape'],self.values['prof']['poltype'],self.values['prof']['ppatype'],nbin)
			prof0=np.fft.irfft(np.fft.rfft(self.profile[0])[:65])
			self.tdomain,self.sfac=gen_single(prof0,self.values['prof']['single'],self.nsingle,self.values['psr']['nrate'],self.pfac)
			self.tdomain*=prof0
			self.cal,self.calpara=gen_cal(self.values['data']['cal'],self.freqs,self.gain)
		self.fdomain=np.fft.irfft(np.fft.rfft(self.spec)[:65]).reshape(-1,1)@prof0.reshape(1,-1)
	#
	def loadpsr(self,dic):
		self.nfac=dic.pop('noise factors')
		self.sfac=dic.pop('single factors')
		self.pfac=dic.pop('profile factors')
		self.calpara=dic.pop('calibration factors')
		self.values=dic
		if self.refresh(): return True
		self.mode='load'
		self.genpsr(fac=1)
	#
	def refresh(self):
		for i in self.keys.keys():
			for k,f in self.keys[i]:
				try:
					value=f(self.values[i][k])
				except:
					tm.showwarning('Warning','The parameter '+i+':'+k+' is invalid!')
					return True
				widget=self.__getattribute__(i).__getattribute__(k)
				if widget.winfo_class()=='TCombobox':
					values=widget['value']
					if list(values)[:-1] not in afiles and self.values[i][k] not in values:
						tm.showwarning('Warning','The selected value '+i+':'+k+' is invalid!')
						return True
		for i in self.keys.keys():
			for k,f in self.keys[i]:
				widget=self.__getattribute__(i).__getattribute__(k)
				if widget.winfo_class()=='Entry':
					widget.delete(0,'end')
					widget.insert(0,str(self.values[i][k]))
				elif widget.winfo_class()=='TCombobox':
					values=widget['value']
					if self.values[i][k] in values and self.values[i][k]!='From file':
						widget.current(values.index(self.values[i][k]))
					elif list(values)[:-1] in afiles:
						widget['value']=list(values[:-1])+[self.values[i][k]]
						widget.current(len(values)-1)
	#
	def gendata(self,prog=0,label=0):
		values=self.update(False)
		if values!=self.values: 
			if update_fits_psr(): return
		#
		info=dict()
		try:
			psr=pr.psr(self.values['psr']['name'])
			mode='eph'
			info['psr_name']=psr.name
			info['psr_par']=psr.__str__()
		except:
			try:
				psr=pr.psr(self.values['psr']['name'],parfile=True)
				mode='eph'
				info['psr_name']=psr.name
				info['psr_par']=psr.__str__()
			except:
				mode='pdm'
				info['psr_name']=self.values['psr']['name']
		#
		profile=np.array(self.profile[:4])
		profile/=profile[0].mean()
		#
		if self.values['data']['poltype'] in ['AABBCRCI','IQUV']: npol=4
		elif self.values['data']['poltype']=='AABB': npol=2
		#
		stt_date=int(self.values['obs']['mjd'])
		stt_sec=self.values['obs']['mjd']*86400
		#
		if mode=='eph':
			dm=psr.dm
			period=psr.p0
			ncoeff=max(int(self.values['obs']['length']/100),12)
			chebx_test0=nc.chebpts1(ncoeff)
			chebx_test=np.concatenate(([-1],chebx_test0,[1]),axis=0)
			second_test=(chebx_test+1)/2*self.values['obs']['length']
			time_test=te.time(stt_date*np.ones_like(second_test),stt_sec+second_test,scale=self.values['tele']['name'])
			times_test=te.times(time_test)
			timing_test_end=pm.psr_timing(psr,times_test,self.freq_end)
			timing_test_start=pm.psr_timing(psr,times_test,self.freq_start)
			phase_start=timing_test_end.phase.integer[0]+1
			phase_end=timing_test_start.phase.integer[-1]
			info['phase0']=int(phase_start)
			phase=timing_test_end.phase.integer-phase_start+timing_test_end.phase.offset
			nperiod=int(np.ceil(phase_end)-np.floor(phase_start))
			cheb_end=nc.chebfit(chebx_test,timing_test_end.phase.integer-phase_start+timing_test_end.phase.offset,ncoeff-1)
			roots=nc.chebroots(cheb_end)
			roots=np.real(roots[np.isreal(roots)])
			root=roots[np.argmin(np.abs(roots))]
			stt_time_test=te.time(stt_date,(root+1)/2*nsblk*nsub*tsamp,scale=self.values['tele']['name'])
			stt_sec=stt_time_test.second[0]
			stt_date=stt_time_test.date[0]
			ncoeff_freq=10
			phase_tmp=np.zeros([ncoeff_freq,ncoeff+2])
			disp_tmp=np.zeros(ncoeff_freq)
			cheby=nc.chebpts1(ncoeff_freq)
			freqy=(cheby+1)/2*self.values['obs']['bw']+self.freq_start
			for i in np.arange(ncoeff_freq):
				timing_test=pm.psr_timing(psr,times_test,freqy[i])
				disp_tmp[i]=((timing_test.tdis1+timing_test.tdis2)/period).mean()
				phase_tmp[i]=timing_test.phase.integer-phase_start+timing_test.phase.offset+disp_tmp[i]
			coeff_freq=np.polyfit(1/freqy,disp_tmp,4)
			coeff=nc.chebfit(chebx_test,nc.chebfit(cheby,phase_tmp,1).T,ncoeff-1)
			info['predictor']=coeff.tolist()
			info['predictor_freq']=coeff_freq.tolist()
		else:
			dm=0
			period=self.values['psr']['p0']
			phase=self.values['obs']['length']/period
			info['phase0']=0
			nperiod=int(np.ceil(np.max(phase)))
		#
		acclen=period/self.values['data']['nbin']*self.values['obs']['bw']/self.values['data']['nchan']*1e6
		if acclen<1: 
			tm.showwarning('Warning!','The time and frequency resolution is too high!')
			return
		#
		if nperiod==0:
			tm.showwarning('Warning!','The integration time is shorter than one period!')
		fsize=self.values['data']['nchan']*npol*self.values['data']['nbin']*nperiod*8/2**30
		if fsize>1:
			answer=tm.askokcancel('Warning','The total file size is larger than '+str(int(fsize))+'G, please determine whether to save them.')
			if not answer: return	
		#
		while True:
			filename=tf.asksaveasfilename(filetypes=[('LD file','.ld')])
			if not filename: return
			if len(filename)>3:
				if filename[-3:]=='.ld': filename=filename[:-3]
			filename=filename+'.ld'
			break
		#
		info['telename']=self.values['tele']['name']
		info['freq_start_origin']=self.freq_start
		info['freq_end_origin']=self.freq_end
		info['freq_start']=self.freq_start
		info['freq_end']=self.freq_end
		info['nchan_origin']=int(self.values['data']['nchan'])
		info['nchan']=int(self.values['data']['nchan'])
		info['chan_weight']=np.ones(int(self.values['data']['nchan'])).tolist()
		info['stt_time_origin']=stt_date+stt_sec/86400.0
		info['npol']=int(npol)
		info['pol_type']=self.values['data']['poltype']
		info['dm']=dm
		info['stt_sec']=stt_sec
		info['stt_date']=int(stt_date)
		info['stt_time']=stt_date+stt_sec/86400.0
		#
		info['nperiod']=int(nperiod)
		info['period']=period
		info['nbin']=int(self.values['data']['nbin'])
		info['length']=period*nperiod
		info['history']=['gen.py']
		if self.values['data']['sub_nperiod']==1:
			info['mode']='single'
			info['nsub']=int(nperiod)
		else:
			info['mode']='subint'
			info['sub_nperiod']=int(self.values['data']['sub_nperiod'])
			info['sublen']=period*info['sub_nperiod']
			nsub=int(np.ceil(nperiod/info['sub_nperiod']))
			info['nsub']=nsub
			info['sub_nperiod_last']=int(nperiod-(nsub-1)*info['sub_nperiod'])
			d0=np.zeros([nsub,int(self.values['data']['nbin']),npol])
		#		
		single,sfac=gen_single(profile[0],self.values['prof']['single'],nperiod+2,self.values['psr']['nrate'],self.sfac,sfac=True)
		single=(single*(profile*self.values['psr']['flux']*1e-3).reshape(4,1,-1)).reshape(4,-1).T
		#
		if self.values['psr']['rm']!=0:
			lam=te.sl/self.freqs/1e6
			theta=self.values['psr']['rm']*lam**2*2
			cth,sth=np.sin(theta),np.cos(theta)
		#
		if self.values['data']['cal']!='None':
			info['cal_mode']='single'
			info['cal']=self.cal.tolist()
		a12,a22,cdphi,sdphi=self.cal
		#
		d=ld.ld(filename)
		info['file_time']=time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())
		d.write_shape([int(self.values['data']['nchan']),nperiod,int(self.values['data']['nbin']),npol])
		if self.values['psr']['dm']!=0:
			dphi=np.exp(np.arange(int(nperiod*self.values['data']['nbin']/2+1))*2j*np.pi*self.values['psr']['dm']*pm.dm_const/self.freqs[i]**2/nperiod/period)
		#
		for i in np.arange(int(self.values['data']['nchan'])):
			if prog:
				prog['value']=i/self.values['data']['nchan']*100
				label.set(str(i)+'/'+str(int(self.values['data']['nchan']))+' channels ('+str(round(i/self.values['data']['nchan']*100,1))+'%)')
				prog.update()
			#
			data=single*(self.spec[i]*self.gain[i])
			#
			if self.values['psr']['dm']!=0:
				data=np.irfft(np.fft.rfft(data,axis=0)*dphi,axis=0)
			data=data[self.values['data']['nbin']:-self.values['data']['nbin']]
			#
			if self.values['psr']['rm']!=0:
				data[:,1],data[:,2]=data[:,1]*cth[i]-data[:,2]*sth[i],data[:,2]*cth[i]+data[:,1]*sth[i]
			#
			if self.values['data']['poltype'][0:4]=='AABB':
				data[:,0],data[:,1]=(data[:,0]+data[:,1])/2,(data[:,0]-data[:,1])/2
			data=data[:,:npol]
			#
			baseline=np.zeros([int(nperiod*self.values['data']['nbin']),npol])
			if self.values['data']['poltype'][0:4]=='AABB':
				baseline[:,:2]=ss.chi2.rvs(2*acclen,size=2*self.values['data']['nbin']*nperiod).reshape(-1,2)/2/acclen
			else:
				baseline[:,0]=ss.chi2.rvs(4*acclen,size=self.values['data']['nbin']*nperiod)/4/acclen
				baseline[:,1]=nr.randn(self.values['data']['nbin']*nperiod)/np.sqrt(acclen*2)+1
			if self.values['data']['poltype']=='AABBCRCI':
				baseline[:,2:]=nr.randn(2*self.values['data']['nbin']*nperiod).reshape(-1,2)/np.sqrt(acclen)+1
			elif self.values['data']['poltype']=='IQUV':
				baseline[:,2:]=nr.randn(2*self.values['data']['nbin']*nperiod).reshape(-1,2)/np.sqrt(acclen)*2+1
			#
			noise=self.noise.reshape(1,-1).repeat(npol,axis=0)
			#
			data+=baseline*(self.noise[i]+self.gain[i])*self.sefd
			#
			if self.values['data']['sub_nperiod']==1:
				d.write_chan(data,i)
			else:
				d0[:-1]=data[:(nsub-1)*info['sub_nperiod']*int(self.values['data']['nbin'])].reshape(nsub-1,self.values['data']['sub_nperiod'],int(self.values['data']['nbin']),npol).sum(1)
				d0[-1]=data[(nsub-1)*info['sub_nperiod']*int(self.values['data']['nbin']):].reshape(-1,int(self.values['data']['nbin']),npol).sum(0)
				d.write_chan(d0,i)
		#
		d.write_info(info)
#
# LD (PSR)
pbox3=tk.Frame(tab3,bg='white')
pbox3.grid(row=0,column=0)
pbox3.pack(side='left',fill='y',pady=20)
pbox3_1=tk.Frame(pbox3,bg='white',height=100)
pbox3_1.grid(row=0,column=0,columnspan=2)
tabs3=ttk.Notebook(pbox3_1)
tabs3.place(relx=0.02,rely=0.01,relwidth=0.96,relheight=0.85)
tabs3.pack(fill='x')
tabs3_1=tk.Frame(bg='white')
tabs3.add(tabs3_1,text='PSR')
tabs3_2=tk.Frame(bg='white')
tabs3.add(tabs3_2,text='Tele')
tabs3_3=tk.Frame(bg='white')
tabs3.add(tabs3_3,text='Obs')
tabs3_4=tk.Frame(bg='white')
tabs3.add(tabs3_4,text='Data')
tabs3_5=tk.Frame(bg='white')
tabs3.add(tabs3_5,text='Prof')
#
fbox3=tk.Frame(tab3,bg='white')
fbox3.pack(side='right',fill='y')
#
psrname3=addpara(tabs3_1,'Pulsar Name:',str,0,'J0000+0000',htext='the name of the pulsar')
psrp03=addpara(tabs3_1,'Period (s):',float,1,'1.0',htext='the pulsar rotating period')
psrdm3=addpara(tabs3_1,'\N{GREEK SMALL LETTER DELTA} DM:',float,2,0.0,htext='Dispersion Measure deviation from the factual value')
psrrm3=addpara(tabs3_1,'RM:',float,3,'0.0',htext='Rotation Measure')
psrflux3=addpara(tabs3_1,'Flux (mJy):',float,4,'1.0',htext='the mean radio flux of the pulsar in the selected frequency band')
psrspecind3=addpara(tabs3_1,'Spec. index:',float,5,'0.0',htext='spectra index \N{GREEK SMALL LETTER ALPHA}')
psrnrate3=addpara(tabs3_1,'Null rate:',float,6,'0.0',htext='pulsar nulling rate')
psr3=psr_para(psrname3,psrp03,psrdm3,psrrm3,psrflux3,psrspecind3,psrnrate3)
#
telename3=addpara(tabs3_2,'Tele. Name:',str,0,'FAST',htext='the name of the telescope')
teleaper3=addpara(tabs3_2,'Eff. Aper. (m):',float,1,'300.0',htext='the effective aperture of the telescope')
telesyst3=addpara(tabs3_2,'Sys. Tem. (K):',float,2,'20.0',htext='the system temperature of the telescope')
teleeff3=addpara(tabs3_2,'Efficiency:',float,3,'0.63',htext='the observational efficiency of the telescope')
tele3=tele_para(telename3,teleaper3,telesyst3,teleeff3)
#
obsfreq3=addpara(tabs3_3,'Freq. (MHz):',float,0,'1250.0',htext='the central observational frequency')
obsbw3=addpara(tabs3_3,'BW (MHz):',float,1,'500.0',htext='Band Width')
obsbe3=addpara(tabs3_3,'Edge (MHz):',float,2,'50.0',htext='the width of the band edge')
obsmjd3=addpara(tabs3_3,'MJD:',float,3,'60000.0',htext='the observational date')
obslength3=addpara(tabs3_3,'Length (s):',float,4,'1',htext='the observational duration')
obsnoise3=addpara(tabs3_3,'Noise:',list,5,noiselist,func=select,htext='the frequency domain RFI (Radio Frequency Interference)')
obs3=obs_para(obsfreq3,obsbw3,obsbe3,obsmjd3,obslength3,obsnoise3)
#
datanbin3=addpara(tabs3_4,'Nbin:',int,1,'1024',htext='the number of the bin in one period')
datanchan3=addpara(tabs3_4,'Nchan:',int,2,'4096',htext='the number of frequency channels in one spectrum')
sub_nperiod3=addpara(tabs3_4,'Period/sub:',int,3,'1',htext='the number of pulse period in one sub-integration')
datapol3=addpara(tabs3_4,'Pol. Type:',list,4,poltypelist[::-1],htext='the polarization type in the simulated data')
datacal3=addpara(tabs3_4,'Gain/Phase:',list,5,callist,func=select,htext='the calibration type in the simulated data')
data3=ldpsr_para(datanbin3,datanchan3,sub_nperiod3,datapol3,datacal3)
#
proftype3=addpara(tabs3_5,'Type:',list,0,proftypelist,func=select,htext='the type of the mean pulse profile')
profncomp3=addpara(tabs3_5,'N Comp.:',int,1,'2',htext='the number of the components in the mean pulse profile')
profcompshape3=addpara(tabs3_5,'Comp. shape:',list,2,compshapelist,htext='the shape of the component')
profsingle3=addpara(tabs3_5,'Single pulse:',list,3,singlelist,htext='the drift property of the single pulse')
profpoltype3=addpara(tabs3_5,'Pol. type:',list,4,profpoltypelist,htext='the polarization level of the pulsar')
ppatype3=addpara(tabs3_5,'PPA type:',list,5,ppatypelist,htext='the property of the polarization position angle (PPA) curve')
prof3=prof_para(proftype3,profncomp3,profcompshape3,profsingle3,profpoltype3,ppatype3)
#
ldpsr3=ld_psr(psr3,data3,obs3,tele3,prof3)
#
fig3=Figure(figsize=(6.6,5.4), dpi=100)
x0,x1,x2,x3=0.1,0.49,0.59,0.98
y0,y1,y2,y3=0.1,0.49,0.58,0.98
ax9=fig3.add_axes([x0,y0,x1-x0,y1-y0])
ax100=fig3.add_axes([x0,y2,x1-x0,(y3-y2)/2])
ax101=fig3.add_axes([x0,y2+(y3-y2)/2,x1-x0,(y3-y2)/2])
ax11=fig3.add_axes([x2,y0,x3-x2,y1-y0])
ax12=fig3.add_axes([x2,y2,x3-x2,y3-y2])
ax9.set_xlabel('Frequency (MHz)',family='serif',fontsize=12)
ax100.set_xlabel('Pulse Phase',family='serif',fontsize=12)
ax101.set_xticks([])
ax11.set_xlabel('Pulse Phase',family='serif',fontsize=12)
ax12.set_xlabel('Pulse Phase',family='serif',fontsize=12)
ax9.set_ylabel('SEFD (Jy)',family='serif',fontsize=12)
ax100.set_ylabel('Flux',family='serif',fontsize=12)
ax101.set_ylabel('PPA ($\degree$)',family='serif',fontsize=12)
ax11.set_ylabel('Frequency (MHz)',family='serif',fontsize=12)
ax12.set_ylabel('Pulse Index',family='serif',fontsize=12)
canvas3=FigureCanvasTkAgg(fig3,master=fbox3)
canvas3.get_tk_widget().grid(row=0,column=1)  
#
def update_ld_psr():
	ldpsr3.update()
	if ldpsr3.check(): return True
	ldpsr3.genpsr()
	plot_ld_psr()
#
def genld_psr():
	gbttn3_1['state']='disabled'
	gbttn3_2['state']='disabled'
	gbttn3_3['state']='disabled'
	gbttn3_4['state']='disabled'
	prog3=ttk.Progressbar(pbox3,length=pbox3.bbox()[2])
	prog3.grid(row=3,column=0,columnspan=2,pady=5)
	prog3['maximum']=100
	prog3['value']=0
	prog3.update()
	labeltxt3=tk.StringVar()
	label3=tk.Label(pbox3,textvariable=labeltxt3,bg='white',font=('serif',12))
	label3.grid(row=4,column=0,columnspan=2)
	labeltxt3.set('Preparing Data ...')
	ldpsr3.gendata(prog3,labeltxt3)
	prog3.destroy()
	label3.destroy()
	gbttn3_1['state']='normal'
	gbttn3_2['state']='normal'
	gbttn3_3['state']='normal'
	gbttn3_4['state']='normal'
#
def plot_ld_psr():
	ax9.cla()
	ax100.cla()
	ax101.cla()
	ax11.cla()
	ax12.cla()
	fac=ldpsr3.profile[0].max()
	ax9.plot(ldpsr3.freqs,ldpsr3.gain+ldpsr3.noise,'k-')
	ax100.plot(ldpsr3.phase,ldpsr3.profile[0]/fac,'k-')
	ax100.plot(ldpsr3.phase,ldpsr3.profile[4]/fac,'b-')
	ax100.plot(ldpsr3.phase,ldpsr3.profile[3]/fac,'r-')
	ax101.plot(ldpsr3.phase,ldpsr3.profile[5]-90,'y--')
	ax11.imshow(ldpsr3.fdomain,origin='lower',aspect='auto',extent=(0,1,ldpsr3.freq_start,ldpsr3.freq_end))
	ax12.imshow(ldpsr3.tdomain,origin='lower',aspect='auto',extent=(0,1,0,ldpsr3.nsingle))
	ax9.set_xlabel('Frequency (MHz)',family='serif',fontsize=12)
	ax100.set_xlabel('Pulse Phase',family='serif',fontsize=12)
	ax101.set_xticks([])
	ax11.set_xlabel('Pulse Phase',family='serif',fontsize=12)
	ax12.set_xlabel('Pulse Phase',family='serif',fontsize=12)
	ax9.set_ylabel('SEFD (Jy)',family='serif',fontsize=12)
	ax100.set_ylabel('Flux',family='serif',fontsize=12)
	ax101.set_ylabel('PPA ($\degree$)',family='serif',fontsize=12)
	ax11.set_ylabel('Frequency (MHz)',family='serif',fontsize=12)
	ax12.set_ylabel('Pulse Index',family='serif',fontsize=12)
	canvas3.draw()
#
def savep_ld_psr():
	dic=ldpsr3.values
	dic['noise factors']=tolist(ldpsr3.nfac)
	dic['single factors']=tolist(ldpsr3.sfac)
	dic['profile factors']=tolist(ldpsr3.pfac)
	dic['calibration factors']=tolist(ldpsr3.calpara)
	f=tf.asksaveasfile(mode='w')
	if f:
		f.write(json.dumps(dic,indent=2))
		f.close()
#
def loadp_ld_psr():
	f=tf.askopenfile()
	if f: dic=json.load(f)
	else: return
	if ldpsr3.loadpsr(dic): return
	if ldpsr3.check(): return
	plot_ld_psr()
#
gbttn3_1=tk.Button(pbox3,text='Save Para.',command=lambda:savep_ld_psr(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn3_1.grid(row=1,column=0,pady=10)
gbttn3_2=tk.Button(pbox3,text='Load Para.',command=lambda:loadp_ld_psr(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn3_2.grid(row=2,column=0)
gbttn3_3=tk.Button(pbox3,text='Apply',command=lambda:update_ld_psr(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn3_3.grid(row=1,column=1,pady=10)
gbttn3_4=tk.Button(pbox3,text='Generate',command=lambda:genld_psr(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn3_4.grid(row=2,column=1)
#
#
class psrtoa_para:
	def __init__(self,par,flux,specind,redi,redp):
		self.par=par
		self.flux=flux
		self.specind=specind
		self.redind=redi
		self.redpow=redp
#
class obstoa_para:
	def __init__(self,dist,sttmjd,endmjd,nobs,length,nsub,freq,bw):
		self.dist=dist
		self.sttmjd=sttmjd
		self.endmjd=endmjd
		self.nobs=nobs
		self.length=length
		self.nsub=nsub
		self.freq=freq
		self.bw=bw
#
class ld_toa:
	def __init__(self,psr,tele,obs):
		self.psr=psr
		self.obs=obs
		self.tele=tele
		self.keys={'psr':[('par',str),('flux',float),('specind',float),('redind',float),('redpow',float)],
			'obs':[('dist',str),('sttmjd',float),('endmjd',float),('nobs',int),('length',float),('nsub',int),('freq',float),('bw',float)],
			'tele':[('name',str),('aperture',float),('temperature',float),('efficiency',float)],
			}
	#
	def update(self,update=True):
		values=dict()
		for i in self.keys.keys():
			values[i]=dict()
			for k,f in self.keys[i]:
				values[i][k]=f(self.__getattribute__(i).__getattribute__(k).get())
		if update: 
			self.values=values
			self.mode='input'
		else: return values
	#
	def check(self):
		def warning(str):
			tm.showwarning('Warning!',str)
			return True
		#
		try: af.reco(self.values['tele']['name'])
		except: return warning('Unknown Telescope!')
		#
		try: psr=pr.psr(self.values['psr']['par'])
		except:
			try: psr=pr.psr(self.values['psr']['par'],parfile=True)
			except: return warning('Unrecognized pulsar or parfile!')
		#
		if self.values['obs']['dist']==self.obs.dist['value'][-1]:
			if not os.path.isfile(self.values['obs']['dist']): return warning('The observation allocating file is not existed.')
			try:
				obsa=np.loadtxt(self.values['obs']['dist'])[:,np.arange(5)]
			except: warning('The observation allocating file has wrong format.') 
			mjd,length,nsub,freq,bw=obsa.T
			time=te.times(te.time(mjd[0],0,scale=self.values['tele']['name']))
			tran,l=np.array(time.obs(psr,5)).reshape(-1)
			if tran is None:
				if l==0: return warning('The pulsar cannot be observed with the specified telescope!')
				elif (np.diff(mjd)<length[:-1]).sum()>0: return warning('The allocated observation time has epoch of overlap.')
			else:
				time=te.times(te.time(mjd,0,scale=self.values['tele']['name']))
				tran,l=np.array(time.obs(psr,5)).reshape(-1)
				if (length>l*86400).sum()>0: return warning('The pulsar cannot be monitored for longer than'+str(np.round(l*24,3))+'hour!')
				elif (tran>l/2).sum()>0 or ((tran+l/2)*86400<length).sum()>0:
					return warning('The pulsar is not observable for all allocated time epochs.')
		else:
			if self.values['obs']['sttmjd']>self.values['obs']['endmjd']: return warning('The start MJD is later than the end MJD.')
			if self.values['obs']['length']/self.values['obs']['nsub']<psr.p0: return warning('The integration length must be larger than the pulsar period.')
			time=te.times(te.time(self.values['obs']['sttmjd'],0,scale=self.values['tele']['name']))
			tran,l=np.array(time.obs(psr,5)).reshape(-1)
			if tran is None:
				if l==0: return warning('The pulsar cannot be observed with the specified telescope!')
				elif self.values['obs']['endmjd']-self.values['obs']['sttmjd']<self.values['obs']['length']*self.values['obs']['nobs']:
					return warning('The pulsar cannot be observed for such many times!')
			else:
				if l*86400<self.values['obs']['length']: return warning('The observation length is larger than the observable duration each day.')
				tmp=1/1.002737909
				etime=np.arange(self.values['obs']['sttmjd']+(tran+l/2)%tmp,self.values['obs']['endmjd']+self.values['obs']['length']/86400+tmp,tmp)
				stime=etime-l
				jj=(stime<self.values['obs']['endmjd'])&(etime>self.values['obs']['sttmjd']+self.values['obs']['length']/86400)
				stime=stime[jj]
				etime=etime[jj]
				stime[stime<self.values['obs']['sttmjd']]=self.values['obs']['sttmjd']
				etime[etime>(self.values['obs']['endmjd']+self.values['obs']['length']/86400)]=self.values['obs']['endmjd']+self.values['obs']['length']/86400
				if ((etime-stime)*86400//self.values['obs']['length']).sum()<self.values['obs']['nobs']:
					return warning('The pulsar cannot be observed for such many times!')
	#
	def gentoa(self):
		try:
			psr=pr.psr(self.values['psr']['par'])
		except:
			psr=pr.psr(self.values['psr']['par'],parfile=True)
		#
		if self.values['obs']['dist']==self.obs.dist['value'][-1]:
			obsa=np.loadtxt(self.values['obs']['dist'])[:,np.arange(5)]
			mjd,self.length,self.nsub,self.freq,self.bw=obsa.T
			self.nsub=np.int64(self.nsub)
			self.nobs=len(mjd)
			self.mjd=mjd+length/86400/2
		else:
			self.nobs=self.values['obs']['nobs']
			self.length,self.freq,self.bw,self.nsub=np.array([[self.values['obs']['length']],[self.values['obs']['freq']],[self.values['obs']['bw']],[self.values['obs']['nsub']]])*np.ones(self.nobs).reshape(1,-1)
			self.nsub=np.int64(self.nsub)
			sttmjd,endmjd,length=self.values['obs']['sttmjd'],self.values['obs']['endmjd'],self.values['obs']['length']/86400
			time0=te.times(te.time(sttmjd,0,scale=self.values['tele']['name']))
			tran,l=np.array(time0.obs(psr,5)).reshape(-1)
			if tran:
				tmp=1/1.002737909
				etime=np.arange(sttmjd+(tran+l/2)%tmp,endmjd+length+tmp,tmp)
				stime=etime-l
				jj=(stime<endmjd)&(etime>sttmjd+length)
				stime=stime[jj]
				etime=etime[jj]
				stime[stime<sttmjd]=sttmjd
				etime[etime>(endmjd+length)]=endmjd+length
				npday=np.int64((etime-stime)//length)
				if self.values['obs']['dist']=='Random':
					if jj.sum()>=self.nobs: ind=(npday.cumsum()-npday[0])[np.sort(nr.randint(0,jj.sum()-self.nobs,self.nobs))+np.arange(self.nobs)]
					else: ind=np.sort(nr.randint(0,int(npday.sum()-self.nobs+1),size=self.nobs))+np.arange(self.nobs)
				elif self.values['obs']['dist']=='Uniform':
					ind=np.int64(np.round(np.linspace(0,npday.sum()-1,self.nobs)))
				k,imjd,nmjd=0,[],[]
				npdaycum=npday.cumsum()
				for i in ind:
					while i>npdaycum[k]: k=k+1
					if imjd: 
						if imjd[-1]!=k:
							imjd.append(k)
							nmjd.append(0)
					else:
						imjd.append(k)
						nmjd.append(0)
					nmjd[-1]+=1
				self.mjd=np.concatenate(list(map(lambda x:nr.rand(nmjd[x])*(etime[imjd[x]]-stime[imjd[x]]-length*nmjd[x])+length*np.arange(nmjd[x])+stime[imjd[x]],np.arange(len(imjd)))))+length/2
			else:
				if self.values['obs']['dist']=='Random':
					self.mjd=nr.rand(self.nobs)*(endmjd-sttmjd-length*self.nobs)+length*np.arange(self.nobs)+sttmjd+length/2
				elif self.values['obs']['dist']=='Uniform':
					self.mjd=np.linspace(sttmjd,endmjd-length,self.nobs)+length/2
		time=te.times(te.time(self.mjd,self.length/2,scale=self.values['tele']['name']))
		self.psrt=pm.psr_timing(psr,time,self.freq)
		self.dt_intr=self.psrt.dt_intr
		if psr.binary: 
			self.dt_bin=self.psrt.torb
			self.orbits=self.psrt.orbits%1
		else:
			self.dt_bin=np.zeros_like(self.mjd)
			self.orbits=np.linspace(0,1,self.mjd.size)
		self.dt_dm1=self.psrt.tdis1+self.psrt.tdis2
		self.dt_ssb=self.psrt.dt_ssb+self.dt_dm1
		dt=self.mjd.max()-self.mjd.min()
		nsin=min(100,int(dt/np.diff(self.mjd).min()))
		asin=np.arange(1,nsin+1)**(-self.values['psr']['redind']/2)
		dt_redn=((np.cos(((self.mjd.reshape(-1,1)/dt/(nr.rand()+1))@np.arange(1,nsin+1).reshape(1,-1)+nr.rand(1,nsin))*2*np.pi)*asin).sum(1))
		self.dt_redn=dt_redn/dt_redn.std()*self.values['psr']['redpow']/1e6
	#
	def loadtoa(self,dic):
		self.values=dic
		if self.refresh(): return True
		self.mode='load'
		self.gentoa()
	#
	def refresh(self):
		for i in self.keys.keys():
			for k,f in self.keys[i]:
				try:
					value=f(self.values[i][k])
				except:
					tm.showwarning('Warning','The parameter '+i+':'+k+' is invalid!')
					return True
				widget=self.__getattribute__(i).__getattribute__(k)
				if widget.winfo_class()=='TCombobox':
					values=widget['value']
					if list(values)[:-1] not in afiles and self.values[i][k] not in values:
						tm.showwarning('Warning','The selected value '+i+':'+k+' is invalid!')
						return True
		for i in self.keys.keys():
			for k,f in self.keys[i]:
				widget=self.__getattribute__(i).__getattribute__(k)
				if widget.winfo_class()=='Entry':
					widget.delete(0,'end')
					widget.insert(0,str(self.values[i][k]))
				elif widget.winfo_class()=='TCombobox':
					values=widget['value']
					if self.values[i][k] in values and self.values[i][k]!='From file':
						widget.current(values.index(self.values[i][k]))
					elif list(values)[:-1] in afiles:
						widget['value']=list(values[:-1])+[self.values[i][k]]
						widget.current(len(values)-1)
	#
	def gendata(self,label=0,labeltxt=0):
		values=self.update(False)
		if values!=self.values: 
			if update_fits_psr(): return
		#
		while True:
			filename=tf.asksaveasfilename(filetypes=[('LD file','.ld'),('TIM file','.tim'),('TXT','.txt')])
			if not filename: return
			if len(filename)>3:
				if filename[-3:]=='.ld':
					ftype='ld'
					filename=filename[:-3]
				if len(filename)>4:
					if filename[-4:] in ['.tim','.txt']:
						ftype=filename[-3:]
						filename=filename[:-4]
			else: ftype='ld'
			filename=filename+'.'+ftype
			break
		#
		if label:
			labeltxt.set('Generating Data ...')
			label.update()
		imjd,smjd,toae,dpe,freq_start,freq_end,dm,period=[],[],[],[],[],[],[],[]
		snrc=self.values['psr']['flux']*1e-3*(self.values['tele']['aperture']**2*2.84e-4*self.values['tele']['efficiency'])/self.values['tele']['temperature']
		for i in np.arange(self.nobs):
			if self.nsub[i]==1:
				dtime0=nc.chebpts1(13)
				time0=te.times(te.time(self.psrt.time.local.date[i]*np.ones(13,dtype=np.int64),self.psrt.time.local.second[i]+dtime0*60,scale=self.values['tele']['name']))
				phase=pm.psr_timing(self.psrt.psr,time0,self.freq[i]).phase
				phase0=phase.integer-phase.integer[6]+phase.offset
				chebd=nc.chebder(nc.chebfit(dtime0,phase0,12))
				dtime=nc.chebval(0,nc.chebfit(phase0,dtime0,7),0)*60
				p0=[60/nc.chebval(dtime/60,chebd)]
			else:
				nt=int(max(6,min(50,self.length[i]/20)))*2
				dtime0=nc.chebpts1(nt+1)
				time0=te.times(te.time(self.psrt.time.local.date[i]*np.ones(nt+1,dtype=np.int64),self.psrt.time.local.second[i]+dtime0*self.length[i]/2,scale=self.values['tele']['name']))
				phase=pm.psr_timing(self.psrt.psr,time0,self.freq[i]).phase
				phase0=phase.integer-phase.integer[int(nt/2)]+phase.offset
				chebc0=nc.chebfit(dtime0,phase0,nt)
				phase1=nc.chebval(np.arange(-1,1,2/self.nsub[i])+1/self.nsub[i],chebc0)
				dtime=nc.chebval(np.int64(np.round(phase1)),nc.chebfit(phase0,dtime0,int(nt/2)))*self.length[i]/2
				p0=(self.length[i]/2)/nc.chebval(dtime/(self.length[i]/2),nc.chebder(chebc0))
			snr=snrc*np.sqrt(self.bw[i]*self.length[i]/self.nsub[i]*1e6)*(self.freq[i]/1000)**(-self.values['psr']['specind'])
			dpe.extend([0.1/snr]*self.nsub[i])
			wn=0.1/snr*np.mean(p0)
			time1=te.time(self.psrt.time.local.date[i]*np.ones(self.nsub[i]),self.psrt.time.local.second[i]+dtime+self.dt_redn[i]+wn*nr.randn(self.nsub[i]))
			imjd.extend(time1.date)
			smjd.extend(time1.second)
			toae.extend([wn]*self.nsub[i])
			period.extend(p0)
			freq_start.extend([self.freq[i]-self.bw[i]/2]*self.nsub[i])
			freq_end.extend([self.freq[i]+self.bw[i]/2]*self.nsub[i])
			dm.extend([self.psrt.psr.dm]*self.nsub[i])
		dp,dmerr=np.zeros([2,len(dm)])
		#
		if ftype=='ld':
			result=np.array([imjd,smjd,toae,dp,dpe,freq_start,freq_end,dm,dmerr,period]).T
			d1=ld.ld(filename)
			d1.write_shape([1,len(dm),10,1])
			d1.write_chan(result,0)
			info={'psr_name':self.psrt.psr.name, 'history':['gen.py'], 'file_time':[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())], 'mode':'ToA', 'method':'simulated', 'telename':self.values['tele']['name']}
			d1.write_info(info)
		elif ftype=='txt':
			fout=open(filename,'w')
			fout.write(self.psrt.psr.name+' ToA\n')
			fout.write(self.values['tele']['name']+'\n')
			for i in np.arange(len(dm)):
				fout.write('{:28s} {:10s} {:10.2f} {:10.2f} {:13f} {:18f}'.format(str(imjd[i])+str(smjd[i]/86400)[1:],"%.3e"%(toae[i]/86400),freq_start[i],freq_end[i],dm[i],period[i])+'\n')
			fout.close()
		elif ftype=='tim':
			fout=open(filename,'w')
			fout.write('FORMAT 1\n')
			nind=max(4,np.ceil(np.log10(len(dm)+1)))
			for i in np.arange(len(dm)):
				ftmp='gen_'+self.psrt.psr.name+'_'+str(i).zfill(nind)+'.dat'
				fout.write('{:26s} {:10.6f} {:28s} {:4f} {:8s}'.format(ftmp,(freq_start[i]+freq_end[i])/2,str(imjd[i])+str(smjd[i]/86400)[1:],toae[i]*1e6,self.values['tele']['name'])+'\n')
			fout.close()

# LD ToA
pbox4=tk.Frame(tab4,bg='white')
pbox4.grid(row=0,column=0)
pbox4.pack(side='left',fill='y',pady=20)
pbox4_1=tk.Frame(pbox4,bg='white',height=100)
pbox4_1.grid(row=0,column=0,columnspan=2)
tabs4=ttk.Notebook(pbox4_1)
tabs4.place(relx=0.02,rely=0.01,relwidth=0.96,relheight=0.85)
tabs4.pack(fill='x')
tabs4_1=tk.Frame(bg='white')
tabs4.add(tabs4_1,text='PSR')
tabs4_2=tk.Frame(bg='white')
tabs4.add(tabs4_2,text='Tele')
tabs4_3=tk.Frame(bg='white')
tabs4.add(tabs4_3,text='Obs')
tabs4_4=tk.Frame(bg='white')
tabs4.add(tabs4_4,text='GW')
#
psrpar4=addpara(tabs4_1,'Name/Parfile:','entryorfile',0,'B2016+28',func=select,htext='the parameter file path or the name of the pulsar')
psrflux4=addpara(tabs4_1,'Flux (mJy):',float,3,'1.0',htext='the mean radio flux of the pulsar at 1GHz')
psrspecind4=addpara(tabs4_1,'Spec. index:',float,4,'0.0',htext='spectra index \N{GREEK SMALL LETTER ALPHA}')
psrredind4=addpara(tabs4_1,'Ind. of RN:',float,5,'4.0',htext='spectra index of red noise')
psrredp4=addpara(tabs4_1,'RN std. (\N{GREEK SMALL LETTER MU}s):',float,6,'0.0',htext='std of red noise')
psr4=psrtoa_para(psrpar4,psrflux4,psrspecind4,psrredind4,psrredp4)
#
telename4=addpara(tabs4_2,'Tele. Name:',str,0,'FAST',htext='the name of the telescope')
teleaper4=addpara(tabs4_2,'Eff. Aper. (m):',float,1,'300.0',htext='the effective aperture of the telescope')
telesyst4=addpara(tabs4_2,'Sys. Tem. (K):',float,2,'20.0',htext='the system temperature of the telescope')
teleeff4=addpara(tabs4_2,'Efficiency:',float,3,'0.63',htext='the observational efficiency of the telescope')
tele4=tele_para(telename4,teleaper4,telesyst4,teleeff4)
#
obsdist4=addpara(tabs4_3,'Obs. Dist.:',list,0,obsdistlist,func=select,htext='the distribution of the observation')
obssttmjd4=addpara(tabs4_3,'Start MJD:',float,1,'59000.0',htext='the start MJD date of the ToAs')
obsendmjd4=addpara(tabs4_3,'End MJD:',float,2,'60000.0',htext='the end MJD date of the ToAs')
obsnobs4=addpara(tabs4_3,'Nobs:',int,3,'50',htext='the ToA numbers for each observation')
obslength4=addpara(tabs4_3,'Length (s):',float,4,'1000',htext='the observational duration for each observation')
obsnsub4=addpara(tabs4_3,'Nsub:',int,5,'1',htext='the ToA numbers for each observation')
obsfreq4=addpara(tabs4_3,'Freq. (MHz):',float,6,'1250.0',htext='the central observational frequency')
obsbw4=addpara(tabs4_3,'BW (MHz):',float,7,'400.0',htext='Band Width')
obs4=obstoa_para(obsdist4,obssttmjd4,obsendmjd4,obsnobs4,obslength4,obsnsub4,obsfreq4,obsbw4)
#
ldtoa4=ld_toa(psr4,tele4,obs4)
#
fbox4=tk.Frame(tab4,bg='white')
fbox4.pack(side='right',fill='y')
fig4=Figure(figsize=(6.6,5.4), dpi=100)
x0,x1,x2,x3=0.12,0.49,0.61,0.98
y0,y1,y2,y3=0.1,0.49,0.58,0.98
axt1=fig4.add_axes([x0,y0,x1-x0,y1-y0])
axt2=fig4.add_axes([x0,y2,x1-x0,y3-y2])
axt3=fig4.add_axes([x2,y0,x3-x2,y1-y0])
axt4=fig4.add_axes([x2,y2,x3-x2,y3-y2])
axt1.set_xlabel('MJD',family='serif',fontsize=12)
axt2.set_xlabel('MJD',family='serif',fontsize=12)
axt3.set_xlabel('Orbital Phase',family='serif',fontsize=12)
axt4.set_xlabel('MJD',family='serif',fontsize=12)
axt1.set_ylabel('Dt Pdot & DM (s)',family='serif',fontsize=12)
axt2.set_ylabel('Dt Red Noise ($\mathrm{\mu}$s)',family='serif',fontsize=12)
axt3.set_ylabel('Dt Binary (s)',family='serif',fontsize=12)
axt4.set_ylabel('Dt SSB (s)',family='serif',fontsize=12)
canvas4=FigureCanvasTkAgg(fig4,master=fbox4)
canvas4.get_tk_widget().grid(row=0,column=1)
#
def update_ld_toa():
	ldtoa4.update()
	if ldtoa4.check(): return True
	ldtoa4.gentoa()
	plot_ld_toa()
#
def genld_toa():
	gbttn4_1['state']='disabled'
	gbttn4_2['state']='disabled'
	gbttn4_3['state']='disabled'
	gbttn4_4['state']='disabled'
	labeltxt4=tk.StringVar()
	label4=tk.Label(pbox4,textvariable=labeltxt4,bg='white',font=('serif',12))
	label4.grid(row=4,column=0,columnspan=2)
	labeltxt4.set('Preparing Data ...')
	ldtoa4.gendata(label4,labeltxt4)
	label4.destroy()
	gbttn4_1['state']='normal'
	gbttn4_2['state']='normal'
	gbttn4_3['state']='normal'
	gbttn4_4['state']='normal'
#
def plot_ld_toa():
	axt1.cla()
	axt2.cla()
	axt3.cla()
	axt4.cla()
	axt1.plot(ldtoa4.mjd,ldtoa4.dt_intr+ldtoa4.dt_dm1,'k.')
	axt2.plot(ldtoa4.mjd,ldtoa4.dt_redn*1e6,'r.')
	axt3.plot(ldtoa4.orbits,ldtoa4.dt_bin,'g.')
	axt4.plot(ldtoa4.mjd,ldtoa4.dt_ssb,'b.')
	axt1.set_xlabel('MJD',family='serif',fontsize=12)
	axt2.set_xlabel('MJD',family='serif',fontsize=12)
	axt3.set_xlabel('Orbital Phase',family='serif',fontsize=12)
	axt4.set_xlabel('MJD',family='serif',fontsize=12)
	axt1.set_ylabel('Dt Pdot & DM (s)',family='serif',fontsize=12)
	axt2.set_ylabel('Dt Red Noise ($\mathrm{\mu}$s)',family='serif',fontsize=12)
	axt3.set_ylabel('Dt Binary (s)',family='serif',fontsize=12)
	axt4.set_ylabel('Dt SSB (s)',family='serif',fontsize=12)
	canvas4.draw()
#
def savep_ld_toa():
	dic=ldtoa4.values
	f=tf.asksaveasfile(mode='w')
	if f:
		f.write(json.dumps(dic,indent=2))
		f.close()
#
def loadp_ld_toa():
	f=tf.askopenfile()
	if f: dic=json.load(f)
	else: return
	if ldtoa4.loadtoa(dic): return
	if ldtoa4.check(): return
	plot_ld_toa()
#
gbttn4_1=tk.Button(pbox4,text='Save Para.',command=lambda:savep_ld_toa(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn4_1.grid(row=1,column=0,pady=10)
gbttn4_2=tk.Button(pbox4,text='Load Para.',command=lambda:loadp_ld_toa(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn4_2.grid(row=2,column=0)
gbttn4_3=tk.Button(pbox4,text='Apply',command=lambda:update_ld_toa(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn4_3.grid(row=1,column=1,pady=10)
gbttn4_4=tk.Button(pbox4,text='Generate',command=lambda:genld_toa(),bg='white',activebackground='#E5E35B',font=font1,width=10)
gbttn4_4.grid(row=2,column=1)


stk.mainloop()
