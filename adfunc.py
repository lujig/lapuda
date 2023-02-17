import numpy as np
import numpy.fft as fft
import scipy.optimize as so
#
def shift(y,x):
	ffts=y*np.exp(x*1j)
	fftr=fft.irfft(ffts)
	return fftr
#
def dmdet(fftdata,dmconst,dm0,dmw,polynum,prec=0):
	length=100
	dm=np.linspace(dm0-dmw,dm0+dmw,length)
	value=np.zeros(length)
	for i in np.arange(length):
		disp=dm[i]*dmconst
		value[i]=np.max(shift(fftdata,disp).sum(0))
		value[i]=(shift(fftdata,disp).sum(0)**2).sum()
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
def baseline0(data):
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
def baseline(data,base_nbin=0,pos=False):
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
def radipos(data,crit=10,base=False,base_nbin=0):
	base,pos=baseline(data,pos=True,base_nbin=base_nbin)
	nbin=data.size
	base_nbin=int(nbin/10)
	noise=data[pos:(pos+base_nbin)].std()
	data-=base
	bin0=np.arange(nbin)[data>(crit*noise)]
	if base:
		return bin0,pos
	else:
		return bin0
#

