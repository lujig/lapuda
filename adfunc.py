import numpy as np
import numpy.fft as fft
import scipy.optimize as so
import os
import sys
#
def reco(x):	# recognize the telescope name from its aliases
	dirname=os.path.split(os.path.realpath(__file__))[0]
	with open(dirname+'/materials/aliase.txt') as f:
		names=f.readlines()
	for i in names:
		if x in i.split('    '):
			return i.split()[0]
	raise
#
def poly(x,lseg,dtj,p,y0=0):	# polynomial fit for the clock correction
	dt0j=np.array([1538314761])
	start=0
	y=np.zeros_like(x,dtype=np.float64)
	polyn=np.arange(1,lseg+1,dtype=np.int16)*2
	for i in np.arange(lseg):
		xj=(x>=dtj[i])&(x<=dtj[i+1])
		px=x[xj]
		coeff=p[start:polyn[i]]
		start=polyn[i]
		if int(dtj[i]) in dt0j:
			y0+=p[polyn[-1]+np.where(dt0j==int(dtj[i]))[0][0]]
		y[xj]=np.polyval(coeff,px-dtj[i])*(px-dtj[i])+y0
		y0=np.polyval(coeff,dtj[i+1]-dtj[i])*(dtj[i+1]-dtj[i])+y0
	return y+p[-1]
#
def cal_time(psr,phase,freq=np.inf,telescope='FAST',ttest=0):
	import psr_read as pr
	import time_eph as te
	import psr_model as pm
	if ttest==0:
		ttest=psr.pepoch.mjd+phase.phase*(psr.p0+1/2*phase.phase*(psr.p0*psr.p1+1/3*phase.phase*(psr.p2*psr.p0**2+psr.p1**2*psr.p0+1/4*phase.phase*(psr.p3*psr.p0**3+4*psr.p2*psr.p1*psr.p0**2+psr.p1**3*psr.p0))))/86400
	dt=1
	ttestd=np.int64(ttest)
	ttests=(ttest-ttestd)*86400
	ti=te.time(ttestd,ttests,scale=telescope)
	p0=pm.psr_timing(psr,te.times(ti),freq)
	while np.all(np.abs(dt)>p0.period_intrinsic.mean()*1e-7):
		p0=pm.psr_timing(psr,te.times(ti),freq)
		dt=phase.minus(p0.phase).phase*p0.period_intrinsic
		ti=ti.add(dt)
	return ti
#
def parakey():
	dirname=os.path.split(os.path.realpath(__file__))[0]
	sys.path.append(dirname+'/doc')
	from functools import reduce
	import class_doc as cd
	li=cd.ld_file_info
	ldic=dict(zip(reduce(lambda x,y:x+y,list(map(lambda x: list(x.keys()),li.values()))),reduce(lambda x,y:x+y,list(map(lambda x:[x]*len(li[x]),li.keys())))))
	return li,ldic
#
def dic2json(dic):
	dk=dic.keys()
	dj={}
	li,ldic=parakey()
	for i in dk:
		if i=='zchan':
			djk='data_info'
			if djk not in dj.keys(): dj[djk]={}
			va=np.ones(int(dic['nchan']))
			va[np.int32(dic[i].split(','))]=0
			if 'nchan_new' in dic.keys():
				va=va.reshape(int(dic['nchan_new']),-1).sum(1)
				va[va>0]=1/va[va>0]
				dj[djk]['chan_weight']=va.tolist()
			else:
				dj[djk]['chan_weight']=va.tolist()
		elif i in ['nchan_new', 'nsub_new', 'nbin_new', 'npol_new']:
			itmp=i[:-4]
			djk=ldic[itmp]
			if djk not in dj.keys(): dj[djk]={}
			dj[djk][itmp]=int(dic[i])
		elif i=='cal':
			djk='calibration_info'
			if djk not in dj.keys(): dj[djk]={}
			dj[djk][i]=np.float64(dic[i]).reshape(-1,4,int(dic['nchan'])).tolist()
		elif i=='predictor':
			djk='folding_info'
			if djk not in dj.keys(): dj[djk]={}
			dj[djk][i]=list(map(lambda x:np.float64(x[1:-1].split(',')).tolist(),dic[i]))
		else:
			djk=ldic[i]
			if djk not in dj.keys(): dj[djk]={}
			ti=li[djk][i]
			if type(ti) is not list:
				dj[djk][i]=ti(dic[i])
			elif len(ti[1])==1:
				if type(dic[i]) is not list:
					dj[djk][i]=[ti[2](dic[i])]
				else:
					dj[djk][i]=list(map(ti[2],dic[i]))
			elif len(ti[1])>1:
				if type(dic[i]) is not list:
					dj[djk][i]=[ti[2](dic[i])]
				else:
					dj[djk][i]=np.array(dic[i],dtype=ti[2]).tolist()
			else:
				raise
	if 'data_info' in dj.keys():
		if 'chan_weight' not in dj['data_info'].keys():
			dj['data_info']['chan_weight']=np.ones(int(dj['data_info']['nchan'])).tolist()
	return dj
#
def json2dic(js):
	dic={}
	list(map(lambda x:dic.update(x),js.values()))
	return dic
#
def shift(y,x):	# assistant function for dmdet(); return the shifted counterpart of the Fourier-domain array
	ffts=y*np.exp(x*1j)
	fftr=fft.irfft(ffts)
	return fftr
#
def dmdet(fftdata,dmconst,dm0,dmw,polynum,prec=0):	# determine the best DM with the specified precision
	length=100
	dm=np.linspace(dm0-dmw,dm0+dmw,length)
	value=np.zeros(length)
	for i in np.arange(length):
		disp=dm[i]*dmconst
		#value[i]=np.max(shift(fftdata,disp).sum(0))
		value[i]=(shift(fftdata,disp).sum(0)**2).sum()	# using the Minkowski inequality, and assuming that the pulse profiles have uniform shape along frequency
	polyfunc=np.polyfit(dm,value,polynum)
	fitvalue=np.polyval(polyfunc,dm)
	roots=np.roots(np.polyder(polyfunc))
	roots=np.real(roots[np.isreal(roots)])
	if len(roots):
		dmmax=roots[np.argmin(np.abs(roots-dm[np.argmax(value)]))]
		if dmmax<dm[-1] and dmmax>dm[0] and np.polyval(np.polyder(polyfunc,2),dmmax)<0:
			res=value-fitvalue
			error=np.std(res)
			errfunc=np.append(polyfunc[:-1],polyfunc[-1]-np.polyval(polyfunc,dmmax)+error)
			roots=np.roots(errfunc)
			roots=np.real(roots[np.isreal(roots)])
			dmerr=np.mean(np.abs(np.array([roots[np.argmin(np.abs(roots-dmmax+dmw/10))],roots[np.argmin(np.abs(roots-dmmax-dmw/10))]])-dmmax))
			error1=np.std(res[1:]-res[:-1])
			if prec>0:
				if dmerr<prec or error1>error: return dmmax,dmerr
				else: return dmdet(fftdata,dmconst,dmmax,dmerr*20,polynum,prec=prec)
			else: return dmmax,dmerr,dm,value,fitvalue
	return 0,0
#
def baseline0(data):	# determine the baseline of the data (old version)
	nbin=data.size
	bins,mn=10,nbin/10
	stat,val=np.histogram(data[:,0],bins)
	while stat.max()>mn and 2*bins<mn:
		bins*=2
		stat,val=np.histogram(data,bins)
	val=(val[1:]+val[:-1])/2.0
	argmaxstat=np.argmax(stat)
	if argmaxstat==0:
		base=data[(data>(val[1]*-0.5+val[0]*1.5))&(data<(val[1]*0.5+val[0]*0.5))].mean(0)
	elif argmaxstat==1:
		poly=np.polyfit(val[:3],stat[:3],2)
		#base=-poly[1]/poly[0]/2.0
		base=data[(data[:,0]>(-poly[1]/poly[0]/2.0-(val[1]-val[0])*0.5))&(data[:,0]<(-poly[1]/poly[0]/2.0+(val[1]-val[0])*0.5))].mean(0)
	else:
		poly=np.polyfit(val[(argmaxstat-2):(argmaxstat+3)],stat[(argmaxstat-2):(argmaxstat+3)],2)
		#base=-poly[1]/poly[0]/2.0
		base=data[(data[:,0]>(-poly[1]/poly[0]/2.0-(val[1]-val[0])*0.5))&(data[:,0]<(-poly[1]/poly[0]/2.0+(val[1]-val[0])*0.5))].mean(0)
	return base
#
def baseline(data,base_nbin=0,pos=False):	# determine the baseline of the data
	nbin=data.size
	base_nbin=int(base_nbin)
	if base_nbin<=0:
		base_nbin=int(nbin/10)
	tmp=np.append(np.zeros(base_nbin),np.ones(nbin-base_nbin))
	tmp0=fft.irfft(fft.rfft(data)*fft.rfft(tmp).conj())
	bin0=np.argmax(tmp0)
	base=np.append(data,data)[bin0:(bin0+base_nbin)].mean()
	if pos:
		return base,bin0
	else:
		return base
#
def radipos(data,crit=10,base0=False,base_nbin=0):	# determine the radiation position of the data
	base,pos=baseline(data,pos=True,base_nbin=base_nbin)
	nbin=data.size
	base_nbin=int(nbin/10)
	noise=data[pos:(pos+base_nbin)].std()
	data-=base
	bin0=np.arange(nbin)[data>(crit*noise)]
	if base0:
		return bin0,pos
	else:
		return bin0
#
