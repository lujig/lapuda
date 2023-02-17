import numpy as np
import numpy.polynomial.chebyshev as nc
import struct as st
import datetime as dt
import os
import copy as cp
#
ephname='DE436.1950.2050'
sl=299792458.0
km1=1.55051979176e-8
lc=1.48082686742e-8
iftek=1.000000015505197917599998747524139082638872
lg=6.969290133091514e-10
tdb0=-65.564518e-6
mjd0=43144.0003725
au_dist=149597870691
pi_mm=3.14159265358979323846264338327950288
pi=np.float64(pi_mm)
dirname=os.path.split(os.path.realpath(__file__))[0]
#
def readeph(et,ephname=ephname):
	# 0 mercury; 1 venus; 2 earth; 3 mars; 4 jupiter; 5 saturn; 6 uranus; 7 neptune; 8 pluto; 9 moon; 10 sun; 11 barycenter; 12 earth-moon-center
	f=open(dirname+'/conventions/'+ephname,'rb')
	title=f.read(84)
	ephemeris_version=int(title[26:29])
	f.seek(2652,0)
	ipt=np.zeros([45],dtype=np.uint32)
	sz=5*8+41*4
	if ephemeris_version<430:
		ephem_start,ephem_end,ephem_step,ncon,au,emrat,*ipt[:40]=st.unpack('>3d1L2d40L',f.read(sz))
		ipt[36:39]=ipt[37:40]
		ipt[39:45]=st.unpack('>6L',f.read(24))
	else:
		ephem_start,ephem_end,ephem_step,ncon,au,emrat,*ipt[:40]=st.unpack('<3d1L2d40L',f.read(sz))
		ipt[36:39]=ipt[37:40]
		f.seek((ncon-400)*6,1)
		ipt[39:45]=st.unpack('<6L',f.read(24))
	#
	au=au*1000
	ipt=ipt.reshape(15,3)
	dimension=np.array([3,3,3,3,3,3,3,3,3,3,3,2,3,3,1])
	kernel_size=4+(2*ipt[:,1]*ipt[:,2]*dimension).sum()
	recsize=kernel_size*4
	ncoeff=np.uint32(kernel_size/2)
	f.seek(84*3,0)
	if ephemeris_version<430:
		tmp=np.reshape(list(f.read(ncon*6).decode()),(-1,6))
		f.seek(recsize,0)
		cons=np.array(st.unpack('>'+str(ncon)+'d',f.read(ncon*8)))
		consname=list(map(lambda x:''.join(x).strip(),tmp))
	else:
		tmp=np.reshape(list(f.read(400*6).decode()),(-1,6))
		f.seek(84 * 3 + 400 * 6 + 5 * 8 + 41 * 4,0)
		tmp2=np.reshape(list(f.read((ncon-400)*6).decode()),(-1,6))
		consname=list(map(lambda x:''.join(x).strip(),tmp))+list(map(lambda x:''.join(x).strip(),tmp2))
		f.seek(recsize,0)
		cons=np.array(st.unpack('<'+str(ncon)+'d',f.read(ncon*8)))
	cons=dict(zip(consname,cons))
	date=et.date
	second=et.second
	pv=np.zeros([et.size,13,9])
	nut=np.zeros([et.size,6])
	lib=np.zeros([et.size,9])
	block_loc=(date+2400000.5-ephem_start+second/86400)/ephem_step
	nr=np.uint32(block_loc)
	t0=block_loc-nr
	j_t0=t0==0
	if j_t0.sum():
		t0[j_t0]=1
		nr[j_t0]-=1
	set_nr=list(set(nr))
	set_len=len(set_nr)
	for n in range(set_len):
		j_nr=nr==set_nr[n]
		f.seek((set_nr[n]+2)*recsize,0)
		if ephemeris_version<430:
			a=f.read(ncoeff*8)
			coef=np.array(st.unpack('>'+str(ncoeff)+'d',a))
		else:
			coef=np.array(st.unpack('<'+str(ncoeff)+'d',f.read(ncoeff*8)))
		for i in range(15):
			if i==12 or i==13: continue
			if i<10:
				iptr=ipt[i]
				ncm=3
			elif i==14:
				iptr=ipt[10]
				ncm=3
			else:
				iptr=ipt[i+1]
				ncm=dimension[i+1]
			coeff_start=iptr[0]-1
			n_cheb=iptr[1]
			n_intervals=iptr[2]
			dna=n_intervals
			temp=dna*t0[j_nr]
			interval_num=np.uint32(temp)
			tc=temp%1*2-1
			j_tc=t0[j_nr]==1
			if j_tc.sum():
				interval_num[j_tc]-=1
				tc[j_tc]=1.0
			set_interval=list(set(interval_num))
			set_interval_len=len(set_interval)
			for s in range(set_interval_len):
				j_interval=interval_num==set_interval[s]
				j_sum=np.arange(et.size)[j_nr][j_interval]
				vfac=dna*2/ephem_step/86400
				posvel=np.zeros([j_interval.sum(),ncm*3])
				for k in np.arange(ncm):
					start_coeff_k=coeff_start+(k+set_interval[s]*ncm)*n_cheb
					coef0=coef[start_coeff_k:(start_coeff_k+n_cheb)]
					posvel[:,k]=nc.chebval(tc[j_interval],coef0)
					posvel[:,k+ncm]=nc.chebval(tc[j_interval],nc.chebder(coef0))*vfac
					posvel[:,k+2*ncm]=nc.chebval(tc[j_interval],nc.chebder(coef0,2))*vfac**2
				if i<10:
					if i==2:
						pv[j_sum,12]=posvel
					else:
						pv[j_sum,i]=posvel
				elif i==14:
					pv[j_sum,10]=posvel
				elif i==10:
					nut[j_sum]=posvel
				elif i==11:
					lib[j_sum]=posvel
	f.close()
	pv[:,2],pv[:,9]=pv[:,12]-pv[:,9]/(1+emrat),pv[:,12]+pv[:,9]
	pos0,vel0,acc0=(pv*1000/sl).reshape(-1,13,3,3).transpose(2,1,3,0) #*iftek
	pos=np.array(list(map(lambda x:vector(x[0],x[1],x[2],center='bary',scale='tdb',coord='equ',unit=sl,type0='pos'),pos0)))
	vel=np.array(list(map(lambda x:vector(x[0],x[1],x[2],center='bary',scale='tdb',coord='equ',unit=sl,type0='vel'),vel0)))
	acc=np.array(list(map(lambda x:vector(x[0],x[1],x[2],center='bary',scale='tdb',coord='equ',unit=sl,type0='acc'),acc0)))
	return pos,vel,acc,nut,cons
#
def rotz(psi,mat):
	sp=np.sin(psi)
	cp=np.cos(psi)
	rot=np.array([[cp,sp,np.zeros_like(cp)],[-sp,cp,np.zeros_like(cp)],[np.zeros_like(cp),np.zeros_like(cp),np.ones_like(cp)]])
	if np.size(rot.shape)==3:
		rot=rot.transpose(2,0,1)
	return rot@mat
#
def rotx(phi,mat):
	sp=np.sin(phi)
	cp=np.cos(phi)
	rot=np.array([[np.ones_like(cp),np.zeros_like(cp),np.zeros_like(cp)],[np.zeros_like(cp),cp,sp],[np.zeros_like(cp),-sp,cp]])
	if np.size(rot.shape)==3:
		rot=rot.transpose(2,0,1)
	return rot@mat
#
def roty(theta,mat):
	st=np.sin(theta)
	ct=np.cos(theta)
	rot=np.array([[ct,np.zeros_like(ct),-st],[np.zeros_like(ct),np.ones_like(ct),np.zeros_like(ct)],[st,np.zeros_like(ct),ct]])
	if np.size(rot.shape)==3:
		rot=rot.transpose(2,0,1)
	return rot@mat
#
def multiply(a,b):
	if a.shape[-1]==b.shape[-1]==3:
		ax,ay,az=a[...,0],a[...,1],a[...,2]
		bx,by,bz=b[...,0],b[...,1],b[...,2]
		return np.array([ay*bz-az*by,az*bx-ax*bz,ax*by-ay*bx]).T
#
def normalize(a):
	return a/(a**2).sum(-1).reshape([*a.shape[:-1],1])
#
def lmst(mjd,olong):# sidereal time and the derivative, here mjd is the ut1_mjd
	a = 24110.54841
	b = 8640184.812866
	c = 0.093104
	d = -6.2e-6
	nmjdu1=np.int32(mjd)
	fmjdu1=mjd-nmjdu1
	tu0=((nmjdu1-51545)+0.5)/3.6525e4
	dtu=fmjdu1/3.6525e4
	tu=tu0+dtu
	gmst0=(a + tu0*(b+tu0*(c+tu0*d)))/86400.0
	seconds_per_jc = 86400.0*36525.0
	bprime = 1.0 + b/seconds_per_jc
	cprime = 2.0 * c/seconds_per_jc
	dprime = 3.0 * d/seconds_per_jc
	sdd = bprime+tu*(cprime+tu*dprime)
	gst = gmst0 + dtu*(seconds_per_jc + b + c*(tu+tu0) + d*(tu**2+tu*tu0+tu0**2))/86400
	xlst = gst - olong/360.0
	xlst = xlst%1.0
	return xlst,sdd
#
def get_precessionMatrix(et0,nut): #et0 is the ut1_mjd
	par_zeta=np.array([2306.2181, 0.30188, 0.017998])
	par_z=np.array([2306.2181, 1.09468, 0.018203])
	par_theta=np.array([2004.3109, -0.42665, -0.041833])
	seconds_per_rad = 3600.0*180.0/pi
	oblq=23.4458333333333333
	eps=oblq*pi/180.0
	t=(et0-51544.5)/36525.0
	zeta=t*(par_zeta[0]+t*(par_zeta[1]+t*par_zeta[2]))/seconds_per_rad
	z=t*(par_z[0]+t*(par_z[1]+t*par_z[2]))/seconds_per_rad
	theta=t*(par_theta[0]+t*(par_theta[1]+t*par_theta[2]))/seconds_per_rad
	czeta=np.cos(zeta)
	szeta=np.sin(zeta)
	cz=np.cos(z)
	sz=np.sin(z)
	ctheta=np.cos(theta)
	stheta=np.sin(theta)
	prc=np.array([[czeta*ctheta*cz - szeta*sz,-szeta*ctheta*cz - czeta*sz,-stheta*cz],[czeta*ctheta*sz + szeta*cz,-szeta*ctheta*sz + czeta*cz,-stheta*sz],[czeta*stheta,-szeta*stheta,ctheta]])
	ceps=np.cos(eps)
	seps=np.sin(eps)
	nut0=np.ones([3,3,len(nut)])
	nut0[0,1]=-nut[:,0]*ceps
	nut0[1,0]=-nut0[0,1]
	nut0[0,2]=-nut[:,0]*seps
	nut0[2,0]=-nut0[0,2]
	nut0[1,2]=-nut[:,1]
	nut0[2,1]=-nut0[1,2]
	prn=(nut0.transpose(2,0,1)@(prc.transpose(2,0,1))).transpose(0,2,1)
	return prn
#
def datetime2mjd(datetime):
	ndate=int(np.size(datetime)/6)
	yr,mo,dat,hr,mi,se=np.array(datetime).reshape(ndate,6).T
	dat0=hr*3600+mi*60+se
	yr,mo,dat=np.int32(yr),np.int32(mo),np.int32(dat)
	modat=np.array([[31,28,31,30,31,30,31,31,30,31,30,31]]*ndate)
	modat[((yr%4==0) & (yr%100!=0)) | (yr%400==0),1]=29
	modat=np.array(list(map(lambda x:x[0][:(x[1]-1)].sum(),zip(modat,mo))))
	yrdat=np.ones([ndate,100])*365
	yrdat[:,[i-1950 for i in range(1950,2050) if (((i%4==0)&(i%100!=0))|(i%400==0))]]=366
	yrdat=np.array(list(map(lambda x:x[0][:(x[1]-1950)].sum(),zip(yrdat,yr))))
	return time(33281+yrdat+modat+dat,dat0)
#
def mjd2datetime(mjd):
	mjd=np.array(mjd).reshape(-1)
	ndate=np.array(mjd).size
	day,sec=np.divmod(mjd,1)
	day-=33281
	yrdat=np.ones([ndate,100])*365
	yrdat[:,[i-1950 for i in range(1950,2050) if (((i%4==0)&(i%100!=0))|(i%400==0))]]=366
	yrdat=yrdat.cumsum(1)
	yrnum=((day.reshape(-1,1)-yrdat)>0).sum(1)
	yr=1950+yrnum
	iday=day-yrdat[np.arange(ndate),yrnum-1]
	modat0=np.array([[31,28,31,30,31,30,31,31,30,31,30,31]]*ndate)
	modat0[((yr%4==0) & (yr%100!=0)) | (yr%400==0),1]=29
	modat=modat0.cumsum(1)
	monum=((iday.reshape(-1,1)-modat)>0).sum(1)
	mo=1+monum
	day=np.int8(iday-modat[np.arange(ndate),monum]+modat0[np.arange(ndate),monum])
	return yr,mo,day,sec,iday
#
class vector:
	def __init__(self,x,y,z,center='geo',scale='si',coord='equ',unit=1.0,type0='pos'):
		'''
		center=geo, bary
		scale=si, tdb, itrs, grs80
		coord=equ, ecl
		unit=1.0, sl, au_dist
		type0=pos, vel, acc
		'''
		if np.array(x).size>1:
			x,y,z=np.array(x),np.array(y),np.array(z)
			if not x.size==y.size==z.size:
				raise
		self.x=x
		self.y=y
		self.z=z
		self.center=center
		self.scale=scale
		self.coord=coord
		self.unit=unit
		self.type=type0
		self.size=np.array(x).size
	#
	def __eq__(self,other):
		if type(other) is not vector:
			return False
		return (self.x==other.x)&(self.y==other.y)&(self.z==other.z)&(self.center==other.center)&(self.scale==other.scale)&(self.coord==other.coord)&(self.unit==other.unit)&(self.type==other.type)
	#
	def __str__(self):
		if self.size>6:
			x=self.x.copy()
			xstr='[ '+str(x[0])+', '+str(x[1])+', '+str(x[2])+', ..., '+str(x[-3])+', '+str(x[-2])+', '+str(x[-1])+' ]'
			x=self.y.copy()
			ystr='[ '+str(x[0])+', '+str(x[1])+', '+str(x[2])+', ..., '+str(x[-3])+', '+str(x[-2])+', '+str(x[-1])+' ]'
			x=self.z.copy()
			zstr='[ '+str(x[0])+', '+str(x[1])+', '+str(x[2])+', ..., '+str(x[-3])+', '+str(x[-2])+', '+str(x[-1])+' ]'
		else:
			xstr=str(self.x)
			ystr=str(self.y)
			zstr=str(self.z)
		return 'x='+xstr+',\ny='+ystr+',\nz='+zstr+',\ncenter='+self.center+', scale='+self.scale+', coord='+self.coord+', unit='+str(self.unit)+', type='+self.type+', size='+str(self.size)
	#
	def __repr__(self):
		return self.__str__()
	#
	def copy(self):
		return cp.deepcopy(self)
	#
	def equ2ecl(self):
		if ((self.scale=='si')|(self.scale=='tdb'))&(self.coord=='equ'):
			arcsec2rad=pi/648000.0
			obliq=84381.4059*arcsec2rad
			ce=np.cos(obliq)
			se=np.sin(obliq)
			self.y,self.z=ce*self.y+se*self.z,ce*self.z-se*self.y
			self.coord='ecl'
	#
	def ecl2equ(self):
		if ((self.scale=='si')|(self.scale=='tdb'))&(self.coord=='ecl'):
			arcsec2rad=pi/648000.0
			obliq=84381.4059*arcsec2rad
			ce=np.cos(obliq)
			se=np.sin(obliq)
			self.y,self.z=ce*self.y-se*self.z,ce*self.z+se*self.y
			self.coord='equ'
	#
	def si2tdb(self):
		if self.scale=='si':
			if self.type=='pos':
				self.x*=(1/iftek)
				self.y*=(1/iftek)
				self.z*=(1/iftek)
			elif self.type=='acc':
				self.x*=iftek
				self.y*=iftek
				self.z*=iftek
			self.scale='tdb'
	#
	def tdb2si(self):
		if self.scale=='tdb':
			if self.type=='pos':
				self.x*=iftek
				self.y*=iftek
				self.z*=iftek
			elif self.type=='acc':
				self.x*=(1/iftek)
				self.y*=(1/iftek)
				self.z*=(1/iftek)
			self.scale='si'
	#
	def itrs2grs80(self):
		if self.scale=='itrs' and self.type=='pos':
			x,y,z=self.x,self.y,self.z
			grs80_a=6378137.0
			p=(x**2+y**2)/grs80_a**2
			esq=1.0/298.257222101*(2-1.0/298.257222101)
			q=(1-esq)/(grs80_a**2)*z**2
			r=(p+q-esq**2)/6.0
			s=esq**2*p*q/(4*r**3)
			t=(1+s+np.sqrt(s*(2+s)))**(1./3)
			u=r*(1+t+1/t)
			v=np.sqrt(u**2+esq**2*q)
			w=esq*(u+v-q)/(2.0*v)
			k=np.sqrt(u+v+w**2)-w
			dd=k*np.sqrt(x**2+y**2)/(k+esq)
			height=(k+esq-1)/k*np.sqrt(dd**2+z**2)
			lat=2*np.arctan2(z,dd+np.sqrt(dd**2+z**2))
			if y>0:
				lon=0.5*pi-2.0*np.arctan2(x,np.sqrt(x**2+y**2)+y)
			else:
				lon=-0.5*pi+2.0*np.arctan2(x,np.sqrt(x**2+y**2)-y)
			self.x,self.y,self.z=lon,lat,height
			self.scale='grs80'
	#
	def grs802itrs(self):
		if self.scale=='grs80' and self.type=='pos':
			lon,lat,height=self.x,self.y,self.z
			esq=1.0/298.257222101*(2-1.0/298.257222101)
			grs80_a=6378137.0
			nn=grs80_a/np.sqrt(1-esq*np.sin(lat)**2)
			self.x=(nn+height)*np.cos(lat)*np.cos(lon)
			self.y=(nn+height)*np.cos(lat)*np.sin(lon)
			self.z=(nn*(1-esq)+height)*np.sin(lat)
			self.scale='itrs'
	#
	def change_unit(self,unit):
		ratio=self.unit/unit
		self.x*=ratio
		self.y*=ratio
		self.z*=ratio
		self.unit=unit
	#
	def dot(self,vec):
		if self.coord==vec.coord and ((self.scale in ['tdb','si']) and (vec.scale in ['tdb','si'])):
			self.change_unit(sl)
			vec.change_unit(sl)
			self.tdb2si()
			vec.tdb2si()
			return self.x*vec.x+self.y*vec.y+self.z*vec.z
	#
	def add(self,vec):
		if self.coord==vec.coord and self.type==vec.type and ((self.scale in ['tdb','si']) and (vec.scale in ['tdb','si'])):
			if self.center==vec.center:
				center=self.center
			else:
				center='bary'
			if not self.unit==vec.unit:
				self.change_unit(sl)
				vec.change_unit(sl)			
			self.tdb2si()
			vec.tdb2si()
			return vector(self.x+vec.x,self.y+vec.y,self.z+vec.z,center=center,scale=self.scale,coord=self.coord,unit=self.unit,type0=self.type)
	#
	def minus(self,vec):
		if self.coord==vec.coord and self.type==vec.type and ((self.scale in ['tdb','si']) and (vec.scale in ['tdb','si'])):
			if self.center==vec.center:
				center=self.center
			else:
				center='bary'
			if not self.unit==vec.unit:
				self.change_unit(sl)
				vec.change_unit(sl)			
			self.tdb2si()
			vec.tdb2si()
			return vector(self.x-vec.x,self.y-vec.y,self.z-vec.z,center=center,scale=self.scale,coord=self.coord,unit=self.unit,type0=self.type)
	#
	def multi(self,factor,type0=0):
		if not type0:
			type0=self.type
		return vector(self.x*factor,self.y*factor,self.z*factor,center=self.center,scale=self.scale,coord=self.coord,unit=self.unit,type0=type0)
	#
	def length(self):
		if self.scale in ['tdb','si']:
			self.change_unit(sl)
			self.tdb2si()
			return np.sqrt(self.x**2+self.y**2+self.z**2)
	#
	def angle(self,vec):
		if self.coord==vec.coord:
			if (((self.scale=='si')|(self.scale=='tdb')) and ((vec.scale=='si')|(vec.scale=='tdb'))) or (self.scale==vec.scale):
				l1=self.length()
				l2=vec.length()
				return np.arccos(self.dot(vec)/(l1*l2))
	#
	def xyz(self):
		return np.array([self.x,self.y,self.z]).T
#
class time:
	def __init__(self,date,second,scale='local',unit=86400):
		date=np.array(date).reshape(-1)
		second=np.array(second).reshape(-1)
		size=np.array(date).size
		if type(date[0])==str:
			date0,second0=np.zeros_like(date,dtype=np.int64),np.zeros_like(date,dtype=np.float64)
			for i in range(size):
				tmp=date[i].split('\.')
				data0[i],second0[i]=np.int64(tmp[0]),np.float64('0.'+tmp[1])
			date,second=date0,second0
		datemain,dateresi=np.divmod(date,1)
		secondmain,secondresi=np.divmod(second+dateresi*unit,unit)
		if size==np.array(second).size:
			self.date=np.reshape(np.int64(datemain)+np.int64(secondmain),-1)
			self.second=np.reshape(np.float64(secondresi),-1)
			self.mjd=date+second/unit
			self.scale=scale
			self.unit=unit
			self.size=size
	#
	def __eq__(self,other):
		if type(other) is not time:
			return False
		return (self.date==other.date)&(self.second==other.second)&(self.scale==other.scale)&(self.unit==other.unit)
	#
	def __str__(self):
		if self.size>6:
			x=self.mjd
			tstr='[ '+str(x[0])+', '+str(x[1])+', '+str(x[2])+', ..., '+str(x[-3])+', '+str(x[-2])+', '+str(x[-1])+' ]'
		else:
			tstr=str(self.mjd)
		return 'mjdtime='+tstr+',\n scale='+self.scale+', unit='+str(self.unit)+', size='+str(self.size)
	#
	def __repr__(self):
		return self.__str__()
	#
	def copy(self):
		return cp.deepcopy(self)
	#
	def minus(self,time1):
		if self.unit==time1.unit:
			if self.scale==time1.scale:
				return time(self.date-time1.date,self.second-time1.second,scale='delta_t',unit=self.unit)
			elif (self.scale in ['tdb','tcb']) and (time1.scale in ['tcb','tdb']):
				return self.tcb().minus(time1.tcb())
	#
	def add(self,dt,scale=0):
		if not scale:
			scale=self.scale
		dt=np.array(dt)
		if dt.size==1:
			dt=dt.reshape(-1)[0]
			if type(dt)==time:
				if self.unit==dt.unit and dt.scale=='delta_t':
					return time(self.date+dt.date,self.second+dt.second,scale=scale,unit=self.unit)
			else:
				return time(self.date,self.second+dt,scale=scale,unit=self.unit)
		elif self.size==dt.size:
			return time(self.date,self.second+dt,scale=scale,unit=self.unit)
	#
	def update(self):
		datemain,dateresi=np.divmod(self.date,1)
		secondmain,secondresi=np.divmod(self.second+dateresi*unit,unit)
		self.date=np.int64(datemain+secondmain)
		self.second=np.float64(secondresi)
	#
	def local2unix(self):
		if self.scale=='local':
			unit=1
			date=np.int64((self.date-40587)*86400+self.second)
			second=self.second%unit
			return time(date,second,'unix',unit)			
	#
	def unix2local(self):
		if self.scale=='unix':
			unit=86400
			date,second=np.divmod(self.date,86400)
			date+=40587
			second=np.float64(second)+self.second
			return time(date,second,'local',unit)			
	#
	def local2utc(self):
		if self.scale=='local':
			f=open(dirname+'/conventions/'+'local2gps.txt')
			t0=np.int64(f.read(10))
			t1=np.int64(f.read(30)[20:])
			flen=f.seek(0,2)/30
			localunix=self.local2unix()
			main,resi=np.divmod(localunix.date-t0,t1-t0)
			nr0=main
			resi=resi+localunix.second
			set_nr=list(set(nr0))
			set_len=len(set_nr)
			interplist=[]
			gpssec=np.zeros_like(resi,dtype=np.float64)
			for i in range(set_len):
				nri=set_nr[i]
				if nri>(flen-2): nri=flen-2
				f.seek(nri*30)
				interplist.extend(f.read(60).split())
			t,dt=np.float64(interplist).reshape(-1,2).T
			order=np.argsort(t)
			gpssec=np.interp(localunix.mjd,t[order],dt[order])
			f.close()
			f=np.loadtxt(dirname+'/conventions/'+'gps2utc.txt')
			t,dt=f[:,0:2].T
			utcsec=np.interp(gpssec/86400+self.mjd,t,dt)*1e-9
			return utcsec+gpssec
	#
	def utc2tai(self):
		if self.scale=='utc':
			f=np.loadtxt(dirname+'/conventions/'+'leap.txt')
			leap_time=np.array(list(map(lambda x:datetime2mjd([x[0],x[1],x[2],0,0,0]).mjd,f))).reshape(-1)
			leap_time0=leap_time.repeat(self.size).reshape(-1,self.size)
			leap_sec=f[:,3].cumsum()
			leap_time=leap_sec[(self.date>leap_time0).sum(0)-1]
			return -leap_time+10
	#
	def tai2utc(self):
		if self.scale=='tai':
			f=np.loadtxt(dirname+'/conventions/'+'leap.txt')
			leap_mjd=np.array(list(map(lambda x:datetime2mjd([x[0],x[1],x[2],0,0,0]).mjd,f))).reshape(-1)
			leap_sec=f[:,3].cumsum()
			leap_tai=(leap_mjd*86400-leap_sec-10).repeat(self.size).reshape(-1,self.size)
			leap_time=leap_sec[((self.mjd*86400)>leap_tai).sum(0)-1]
			return leap_time-10
	#
	def tai2ut1(self):
		if self.scale=='tai':
			mjd0,taimjd0,deltat=np.loadtxt(dirname+'/conventions/'+'tai2ut1.txt').T
			jj=(taimjd0>(self.mjd.min()-1))&(taimjd0<self.mjd.max()+1)
			if len(taimjd0[jj])>0:
				taimjd0,deltat=taimjd0[jj],deltat[jj]
			deltat=np.interp(self.mjd,taimjd0,deltat)
			return deltat
	#
	def utc2tt(self):
		if self.scale=='utc':
			mjd0,deltat=np.loadtxt(dirname+'/conventions/'+'tai2tt.txt')[:,[0,2]].T
			ttmjd=self.utc2tai()+32.184+np.interp(self.mjd,mjd0,deltat)*1e-6
			return ttmjd
	#
	def tai2tt(self):
		if self.scale=='tai':
			utc=self.mjd # using tai instead of utc
			mjd0,deltat=np.loadtxt(dirname+'/conventions/'+'tai2tt.txt')[:,[0,2]].T
			tt_tai=32.184+np.interp(utc,mjd0,deltat)*1e-6
			return tt_tai
	#
	def tt2tai(self):
		if self.scale=='tt':
			mjd0,deltat=np.loadtxt(dirname+'/conventions/'+'tai2tt.txt')[:,[0,2]].T
			tt_tai=32.184+np.interp(self.mjd,mjd0,deltat)*1e-6
			return -tt_tai
	#
	def tdb2tcb(self):
		if self.scale=='tdb':
			return (self.date-mjd0)*km1*self.unit+self.second*km1-tdb0*iftek
	#
	def tcb2tdb(self):
		if self.scale=='tcb':
			km2=np.float64(1/iftek-1)
			return (self.date-mjd0)*self.unit*km2+self.second*km2+tdb0
	#
	def utc(self):
		if self.scale=='local':
			return self.add(self.local2utc(),scale='utc')
		elif self.scale=='tai':
			return self.add(self.tai2utc(),scale='utc')
		elif self.scale=='utc': return self.copy()
	#
	def tai(self):
		if self.scale=='utc':
			return self.add(self.utc2tai(),scale='tai')
		elif self.scale=='tt':
			return self.add(self.tt2tai(),scale='tai')
		elif self.scale=='tai': return self.copy()
	#
	def ut1(self):
		if self.scale=='tai':
			return self.add(self.tai2ut1(),scale='ut1')
		elif self.scale=='utc':
			return self.tai().ut1()
		elif self.scale=='ut1': return self.copy()
	#
	def tt(self):
		if self.scale=='tai':
			return self.add(self.tai2tt(),scale='tt')
		elif self.scale=='utc':
			return self.add(self.utc2tt(),scale='tt')
		elif self.scale=='tt': return self.copy()
	#
	def tcb(self):
		if self.scale=='tdb':
			return self.add(self.tdb2tcb(),scale='tcb')
		elif self.scale=='tcb': return self.copy()
	#
	def tdb(self):
		if self.scale=='tcb':
			return self.add(self.tcb2tdb(),scale='tdb')
		elif self.scale=='tdb': return self.copy()
	#
#
class phase():
	def __init__(self,integer,offset,scale='phase'):
		integer=np.array(integer).reshape(-1)
		offset=np.array(offset).reshape(-1)
		size=np.array(integer).size
		if type(integer[0])==str:
			integer0,offset0=np.zeros_like(integer,dtype=np.int64),np.zeros_like(offset,dtype=np.float64)
			for i in range(size):
				tmp=integer[i].split('\.')
				integer0[i],offset0[i]=np.int64(tmp[0]),np.float64('0.'+tmp[1])
			integer,offset=integer0,offset0
		integer_main,integer_resi=np.divmod(integer,1)
		offset_main,offset_resi=np.divmod(offset+integer_resi,1)
		if size==np.array(offset).size:
			self.integer=np.reshape(np.int64(integer_main+offset_main),-1)
			self.offset=np.reshape(np.float64(offset_resi),-1)
			self.scale=scale
			self.phase=self.integer+self.offset
			self.size=size

	#
	def __str__(self):
		if self.size>6:
			x=self.phase
			tstr='[ '+str(x[0])+', '+str(x[1])+', '+str(x[2])+', ..., '+str(x[-3])+', '+str(x[-2])+', '+str(x[-1])+' ]'
		else:
			tstr=str(self.phase)
		return 'phase='+tstr+' size='+str(self.size)
	#
	def __repr__(self):
		return self.__str__()
	#
	def __eq__(self,other):
		if type(other) is not phase:
			return False
		return (self.integer==other.integer)&(self.offset==other.offset)
	#
	def copy(self):
		return cp.deepcopy(self)
	#
	def minus(self,phase1):
			return phase(self.integer-phase1.integer,self.offset-phase1.offset,scale='delta_phase')
	#
	def add(self,dt):
		dt=np.array(dt)
		if dt.size==1:
			dt=dt.reshape(-1)[0]
			if type(dt)==phase:
				if dt.scale=='delta_phase':
					return phase(self.integer+dt.integer,self.offset+dt.offset)
			else:
				return phase(self.integer,self.offset+dt)
		elif self.size==dt.size:
			return phase(self.integer,self.offset+dt)
	#
	def update(self):
		integer_main,integer_resi=np.divmod(self.integer,1)
		offset_main,offset_resi=np.divmod(self.offset+integer_resi,1)
		self.integer=np.int64(integer_main+offset_main)
		self.offset=np.float64(offset_resi)
#
class times:
	def __init__(self,time0,ephem='DE436',ephver=5):
		if time0.scale=='local':
			self.local=time0.copy()
			self.unix=time0.local2unix()
			self.utc=time0.utc()
			self.tai=self.utc.tai()
			self.ut1=self.tai.ut1()
			self.tt=self.utc.tt()
		elif time0.scale=='utc':
			self.utc=time0.copy()
			self.tai=time0.tai()
			self.ut1=self.tai.ut1()
			self.tt=self.utc.tt()
		elif time0.scale=='tai':
			self.tai=time0.copy()
			self.utc=time0.utc()
			self.ut1=self.tai.ut1()
			self.tt=self.utc.tt()
		elif time0.scale=='tt':
			self.tt=time0.copy()
			self.tai=self.tt.tai()
			self.utc=self.tai.utc()
			self.ut1=self.tai.ut1()
		elif time0.scale=='tcb':
			self.tcb=time0.copy()
			self.tt=self.tcb2tt()
			self.tai=self.tt.tai()
			self.utc=self.tai.utc()
			self.ut1=self.tai.ut1()
		elif time0.scale=='tdb':
			self.tdb=time0.copy()
			self.tt=self.tdb2tt()
			self.tai=self.tt.tai()
			self.utc=self.tai.utc()
			self.ut1=self.tai.ut1()
		else:
			pass
		self.size=time0.size
		self.ephver=ephver
		self.ephem=ephem
		self.ephem_compute(ephname=ephem+'.1950.2050')
		self.tt2tdb()
		self.ephem_compute(ephname=ephem+'.1950.2050')
	#
	def __eq__(self,other):
		if type(other) is not time:
			return False
		return (self.ephem==other.ephem)&(self.ephver==other.ephver)&(self.utc==other.utc)
	#
	def copy(self):
		return cp.deepcopy(self)
	#
	def tcb2tt(self):
		t1=self.tcb.copy()
		t1.scale='tt'
		tmp=times(t1)
		dt=tmp.tt.minus(tmp.tcb)
		dt1=self.tcb.minus(tmp.tcb)
		dts=((dt1.mjd*dt1.unit)**2).mean()
		while dts>1e-21:
			tmp=times(self.tcb.add(dt,scale='tt'))
			dt=tmp.tt.minus(tmp.tcb)
			dt1=self.tcb.minus(tmp.tcb)
			dts=((dt1.mjd*dt1.unit)**2).mean()
		return tmp.tt
	#
	def tdb2tt(self):
		t1=self.tdb.copy()
		t1.scale='tt'
		tmp=times(t1)
		dt=tmp.tt.minus(tmp.tdb)
		dt1=self.tdb.minus(tmp.tdb)
		dts=((dt1.mjd*dt1.unit)**2).mean()
		while dts>1e-21:
			tmp=times(self.tdb.add(dt,scale='tt'))
			dt=tmp.tt.minus(tmp.tdb)
			dt1=self.tdb.minus(tmp.tdb)
			dts=((dt1.mjd*dt1.unit)**2).mean()
		return tmp.tt
	#
	def deltat_fb(self):
		f=open('./conventions/'+'TDB.1950.2050','rb')
		b=st.unpack('>2d2i5d',f.read(64))
		tdbd1,tdbd2,tdbdt,tdbncf=b[:4]
		block_loc=(self.tt.date+2400000.5-tdbd1+self.tt.second/86400)/tdbdt
		nr=np.uint32(block_loc)
		t0=block_loc-nr
		set_nr=list(set(nr))
		set_len=len(set_nr)
		tttdb=np.zeros_like(t0,dtype=np.float64)
		for n in range(set_len):
			j_nr=(nr==set_nr[n])
			f.seek(64+(set_nr[n])*tdbncf*8,0)
			coef=st.unpack('>'+str(tdbncf)+'d',f.read(tdbncf*8))
			tc=t0[j_nr]*2-1
			tttdb[j_nr]=nc.chebval(tc,coef)
		f.close()
		return tttdb-tdb0
	#
	def deltat_if(self):
		f=open(dirname+'/conventions/'+'TIMEEPH_short.te405','rb')
		f.seek(264)
		startjd,endjd,stepjd,ncon=st.unpack('>3d1L',f.read(28))
		ipt=np.reshape(st.unpack('>6L',f.read(24)),(2,3))
		ncoeff=ipt[1,0]-1+3*ipt[1,1]*ipt[1,2]
		reclen=8*ncoeff
		f.seek(reclen)
		ephver=int(np.floor(st.unpack('>1d',f.read(8))))
		if hasattr(self,'tdb'):
			time0=self.tdb
		else:
			time0=self.tt
		jd0=time0.date+2400000.5
		jd1=time0.second/86400
		irec=np.int32(np.floor((jd0-startjd)/stepjd))+2
		t0=(jd0-(startjd+stepjd*(irec-2))+jd1)/stepjd
		t1=stepjd
		pv=np.zeros([time0.size,2])
		set_nr=list(set(irec))
		set_len=len(set_nr)
		for i in range(set_len):
			nr=set_nr[i]
			j_nr=irec==nr
			t2=t0[j_nr]
			len_nr=len(t2)
			posvel=np.zeros([len_nr,2])
			f.seek(reclen*nr)
			buf=list(st.unpack('>'+str(ncoeff)+'d',f.read(reclen)))
			ncf,na=ipt[0,1:]
			vfac=na*2/t1
			l,tmp=np.divmod(na*t2,1)
			set_l=list(set(l))
			len_l=len(set_l)
			for k in range(len_l):
				lr=int(set_l[k])
				j_l=l==lr
				tc=2.0*tmp[j_l]-1.0
				coef=buf[(ipt[0,0]-1+ncf*lr):(ipt[0,0]+6+ncf*lr)]
				posvel[j_l]=np.array([nc.chebval(tc,coef),nc.chebval(tc,nc.chebder(coef))*vfac]).T
			pv[j_nr]=posvel
		return -pv.T
	#
	def ve_if(self):
		f=open(dirname+'/conventions/'+'TIMEEPH_short.te405','rb')
		f.seek(264)
		startjd,endjd,stepjd,ncon=st.unpack('>3d1L',f.read(28))
		ipt=np.reshape(st.unpack('>6L',f.read(24)),(2,3))
		ncoeff=ipt[1,0]-1+3*ipt[1,1]*ipt[1,2]
		reclen=8*ncoeff
		f.seek(reclen)
		ephver=int(np.floor(st.unpack('>1d',f.read(8))))
		if hasattr(self,'tdb'):
			time0=self.tdb
		else:
			time0=self.tt
		jd0=time0.date+2400000.5
		jd1=time0.second/86400
		irec=np.int32(np.floor((jd0+jd1-startjd)/stepjd))+2
		t0=(jd0-(startjd+stepjd*(irec-2))+jd1)/stepjd
		t1=stepjd
		pv1=np.zeros([time0.size,2,3])
		set_nr=list(set(irec))
		set_len=len(set_nr)
		for i in range(set_len):
			nr=set_nr[i]
			j_nr=irec==nr
			t2=t0[j_nr]
			len_nr=len(t2)
			posvel1=np.zeros([len_nr,2,3])
			f.seek(reclen*nr)
			buf=list(st.unpack('>'+str(ncoeff)+'d',f.read(reclen)))
			ncf1,na1=ipt[1,1:]
			vfac=na1*2/t1
			l1,tmp=np.divmod(na1*t2,1)
			set_l1=list(set(l1))
			len_l1=len(set_l1)
			for k in range(len_l1):
				lr1=int(set_l1[k])
				j_l1=l1==lr1
				tc1=2.0*tmp[j_l1]-1.0
				coef1=np.reshape(buf[(ipt[1,0]-1+3*ncf1*lr1):(ipt[1,0]+20+3*ncf1*lr1)],(3,7))
				posvel1[j_l1]=np.array([nc.chebval(tc1,coef1.T),nc.chebval(tc1,nc.chebder(coef1.T))*vfac]).transpose(2,0,1)
			pv1[j_nr]=posvel1
		vel=vector(pv1[:,0,0],pv1[:,0,1],pv1[:,0,2],center='bary',scale='tdb',coord='equ',unit=sl,type0='vel')
		acc=vector(pv1[:,1,0],pv1[:,1,1],pv1[:,1,2],center='bary',scale='tdb',coord='equ',unit=sl,type0='acc')
		vel.equ2ecl()
		acc.equ2ecl()
		self.earthvel=vel
		self.earthacc=acc.multi(1/86400)
	#
	def tt2tdb(self):
		if self.ephver==2: 
			deltat=self.deltat_fb()
			self.sitecalc_old()
		else: 
			deltat=-self.deltat_if()[0]*86400
			self.sitecalc()
		sitepos=self.sitepos
		sitevel=self.sitevel
		#
		obsterm=self.earthvel.dot(self.sitepos)/(1-lc)
		correction=obsterm+tdb0+deltat/(1-lc)
		correction1=(correction-tdb0)*iftek+km1*(self.tt.mjd-mjd0)*86400
		self.tdb=time(self.tt.date,self.tt.second+correction,scale='tdb')
		self.tcb=time(self.tt.date,self.tt.second+correction1,scale='tcb')
		deltat_dot=-self.deltat_if()[1]
		self.ve_if()
		obstermdot=(self.earthacc.dot(self.sitepos)+self.earthvel.dot(self.sitevel))/(1-lc)
		self.einsteinrate=iftek*(1+obstermdot+deltat_dot/(1-lc))
	#
	def ephem_compute(self,ephname=ephname):
		if hasattr(self,'tdb'):
			time0=self.tdb
		else:
			time0=self.tt
		self.pos,self.vel,self.acc,self.nut,self.cons=readeph(time0,ephname=ephname)
		for i in range(13):
			self.pos[i].equ2ecl()
			self.vel[i].equ2ecl()
			self.acc[i].equ2ecl()
		self.earthpos=self.pos[2]
		self.earthvel=self.vel[2]
		self.earthacc=self.acc[2]
	#
	def sitecalc_old(self): # tempo old method
		lat,lon,height=25.65295181388,106.856666872,1110.029
		lon=lon*np.pi/180
		lat=lat*np.pi/180
		self.site_grs80=vector(lon,lat,height,center='geo',scale='grs80',coord='equ',unit=1.0,type0='pos')
		site_itrs=self.site_grs80.copy()
		site_itrs.grs802itrs()
		if not hasattr(self,'nut'):
			self.ephem_compute()
		x,y,z=site_itrs.x,site_itrs.y,site_itrs.z
		nut=self.nut
		erad=np.sqrt(x**2+y**2+z**2)
		hlt=np.arcsin(z/erad)
		alng=np.arctan2(-y,x)
		hrd=erad/(sl*499.004786)  #height (AU)
		coord0=hrd*np.cos(hlt)*499.004786
		coord1=coord0*np.tan(hlt)
		coord2=alng
		toblq=(self.utc.date+2400000.5-2451545.0+self.utc.second/self.utc.unit)/36525.0
		oblq=(((1.813e-3*toblq-5.9e-4)*toblq-4.6815e1)*toblq +84381.448)/3600.0
		pc=np.cos(oblq*np.pi/180+nut[:,1])*nut[:,0]
		tsid,sdd=lmst(self.ut1.mjd,0.0)
		tsid*=(2*np.pi)
		ph=tsid+pc-coord2
		self.lst=ph*12/np.pi
		eeq=np.array([coord0*np.cos(ph),coord0*np.sin(ph),coord1*np.ones(self.size)]).T.reshape(-1,3,1)
		prn=get_precessionMatrix(self.ut1.mjd,nut)
		sitex,sitey,sitez=(prn@eeq).reshape(-1,3).T
		self.sitepos=vector(sitex,sitey,sitez,center='geo',scale='si',coord='equ',unit=sl,type0='pos')
		sitera = np.arctan2(self.sitepos.y,self.sitepos.x)
		self.sitepos.equ2ecl()
		self.sitepos.change_unit(sl)
		speed=2.0*np.pi*coord0/(86400.0/1.00273)
		self.sitevel=vector(-np.sin(sitera)*speed,np.cos(sitera)*speed,np.zeros(self.size),center='geo',scale='si',coord='equ',unit=sl,type0='vel')
		self.sitevel.equ2ecl()
		self.sitevel.change_unit(sl)
		sitepos=self.sitepos.copy()
		sitepos.ecl2equ()
		sitepos.change_unit(1)
		self.zenith=sitepos.multi(height/sitepos.length())
	#
	def sitecalc(self): #IAU 2000B tempo2 method
		lat,lon,height=25.65295181388,106.856666872,1110.029
		lon=lon*np.pi/180
		lat=lat*np.pi/180
		self.site_grs80=vector(lon,lat,height,center='geo',scale='grs80',coord='equ',unit=1.0,type0='pos')
		site_itrs=self.site_grs80.copy()
		site_itrs.grs802itrs()
		x,y,z=site_itrs.x,site_itrs.y,site_itrs.z
		zenith_x=height*np.cos(lon)*np.cos(lat)
		zenith_y=height*np.sin(lon)*np.cos(lat)
		zenith_z=height*np.sin(lat)
		sprime=0.0
		utc0,xp0,yp0=np.loadtxt(dirname+'/conventions/'+'eopc.txt')[:,3:6].T
		tai0,dut10=np.loadtxt(dirname+'/conventions/'+'tai2ut1.txt')[:,1:3].T
		tai0=np.round(tai0%1*86400)
		dut1dot0=(dut10[1:]-dut10[:-1])/86400
		xp=np.interp(self.utc.mjd,utc0,xp0)*(np.pi/(180*60*60))
		yp=np.interp(self.utc.mjd,utc0,yp0)*(np.pi/(180*60*60))
		dut1dot=np.interp(self.utc.mjd,utc0[:-1],dut1dot0)
		ut1_jd=time(self.ut1.date+2400000.5,self.ut1.second,scale='ut1jd')
		ut1_jd1=ut1_jd.date
		ut1_jd2=ut1_jd.second/86400
		tt_jd=time(self.tt.date+2400000.5,self.tt.second,scale='ttjd')
		tt_jd1=tt_jd.date
		tt_jd2=tt_jd.second/86400
		trs0=np.array([x,y,z])
		polarmotion=np.diag(np.ones(3)).repeat(self.size).reshape(3,3,-1).transpose(2,0,1)
		polarmotion=rotx(-yp,roty(-xp,rotz(sprime,polarmotion)))
		pole_itrs=polarmotion[:,:,2]
		eradot = 2.0*np.pi*1.00273781191135448*(1.0+dut1dot)/86400.0
		omega_itrs=pole_itrs*eradot.reshape(-1,1)
		trs1=multiply(omega_itrs,trs0)
		dj00=2451545.0
		djc=36525.0
		turnas=1296000.0
		das2r=4.848136811095359935899141e-6
		t=(tt_jd1-dj00+tt_jd2)/djc
		el=(485868.249036 + (1717915923.2178) * t)%turnas*das2r
		elp=(1287104.79305 + (129596581.0481) * t)%turnas*das2r
		f=(335779.526232 + (1739527262.8478) * t)%turnas*das2r
		d=(1072260.70369 + (1602961601.2090) * t)%turnas*das2r
		om=(450160.398036 + (-6962890.5431) * t)%turnas*das2r
		arg=(nutarray[:,:5]@np.array([el,elp,f,d,om]))%(2*np.pi)
		sarg=np.sin(arg)
		carg=np.cos(arg)
		dp=((nutarray[:,5].reshape(-1,1)+nutarray[:,6].reshape(-1,1)@t.reshape(1,-1))*sarg+nutarray[:,7].reshape(-1,1)*carg).sum(0)
		de=((nutarray[:,8].reshape(-1,1)+nutarray[:,9].reshape(-1,1)@t.reshape(1,-1))*carg+nutarray[:,10].reshape(-1,1)*sarg).sum(0)
		dpsi=dp*das2r/1e7-0.135*das2r/1e3 # nutation resulted from moon, sun and planets
		deps=de*das2r/1e7+0.388*das2r/1e3
		dpsipr=-0.29965*das2r*t # IAU 2000 precession corrections
		depspr=-0.02524*das2r*t
		epsa=das2r*(84381.448 +(-46.8150 +(-0.00059+0.001813 * t) * t) * t)+depspr
		eps0=84381.448*das2r
		dpsibi=-0.041775*das2r  # frame bias
		depsbi=-0.0068192*das2r
		dra0=-0.0146*das2r  # ICRS RA of the J2000 equinox
		psia77 = (5038.7784 + (-1.07259 + (-0.001147) * t) * t) * t * das2r # precision angle in Lieske et al. 1977
		oma77  =       eps0 + ((0.05127 + (-0.007726) * t) * t) * t * das2r
		chia   = (  10.5526 + (-2.38064 + (-0.001125) * t) * t) * t * das2r
		psia=psia77+dpsipr 
		oma=oma77+depspr
		rbw=np.diag(np.ones(3)).repeat(self.size).reshape(3,3,-1).transpose(2,0,1)
		rb=rotx(-depsbi,roty(dpsibi*np.sin(eps0),rotz(dra0,rbw))) #bias matrix
		rp=rotz(chia,rotx(-oma,rotz(-psia,rotx(eps0,rbw))))  #precision matrix
		rbp=rp@rb
		rn=rotx(-(epsa+deps),rotz(-dpsi,rotx(epsa,rbw))) #nutation matrix
		rbpn=rn@rbp
		x,y=rbpn[:,2,:2].T
		fa=np.zeros([8,self.size])
		fa[0]=(485868.249036  + t * ( 1717915923.2178 + t * ( 31.8792 + t * ( 0.051635 + t * (-0.00024470 ) ) ) ))%turnas*das2r #mean anomaly of the Moon
		fa[1]=(1287104.793048 + t * ( 129596581.0481 + t * ( -0.5532 + t * ( 0.000136 + t * (-0.00001149 ) ) ) ))%turnas*das2r #mean anomaly of the Sun
		fa[2]=(335779.526232 + t * ( 1739527262.8478 + t * (-12.7512 + t * ( -0.001037 + t * (0.00000417 ) ) ) ))%turnas*das2r #mean argument of the latitude of the Moon
		fa[3]=(1072260.703692 + t * ( 1602961601.2090 + t * ( -6.3706 + t * ( 0.006593 + t * (-0.00003169 ) ) ) ))%turnas*das2r #mean elongation of the Moon from the Sun
		fa[4]=(450160.398036 + t * ( -6962890.5431 + t * ( 7.4722 + t * ( 0.007702 + t * ( -0.00005939 ) ) ) ))%turnas*das2r #mean longitude of the Moon's ascending node
		fa[5]=(3.176146697 + 1021.3285546211 * t)%(2*np.pi) #mean longitude of Venus
		fa[6]=(1.753470314 + 628.3075849991 * t)%(2*np.pi) #mean longitude of Earth
		fa[7]=(0.024381750 + 0.00000538691 * t) * t #general accumulated precession in longitude
		a=np.array(list(map(lambda x:(fa*np.reshape(x[0],(-1,1))).sum(0),cios0)))
		w0=ciosp[0]+np.array(list(map(lambda x:np.sin(x[1])*x[0][1]+np.cos(x[1])*x[0][2],zip(cios0,a)))).sum(0)
		a=np.array(list(map(lambda x:(fa*np.reshape(x[0],(-1,1))).sum(0),cios1)))
		w1=ciosp[1]+np.array(list(map(lambda x:np.sin(x[1])*x[0][1]+np.cos(x[1])*x[0][2],zip(cios1,a)))).sum(0)
		a=np.array(list(map(lambda x:(fa*np.reshape(x[0],(-1,1))).sum(0),cios2)))
		w2=ciosp[2]+np.array(list(map(lambda x:np.sin(x[1])*x[0][1]+np.cos(x[1])*x[0][2],zip(cios2,a)))).sum(0)
		a=np.array(list(map(lambda x:(fa*np.reshape(x[0],(-1,1))).sum(0),cios3)))
		w3=ciosp[3]+np.array(list(map(lambda x:np.sin(x[1])*x[0][1]+np.cos(x[1])*x[0][2],zip(cios3,a)))).sum(0)
		a=np.array(list(map(lambda x:(fa*np.reshape(x[0],(-1,1))).sum(0),cios4)))
		w4=ciosp[4]+np.array(list(map(lambda x:np.sin(x[1])*x[0][1]+np.cos(x[1])*x[0][2],zip(cios4,a)))).sum(0)
		w5=ciosp[5]
		s = (w0 + (w1 + (w2 + (w3 + (w4 + w5 * t) * t) * t) * t) * t) * das2r-x*y/2.0
		r2=x*x+y*y
		e=np.where(r2>0,np.arctan2(y,x),0)
		d=np.arctan(np.sqrt(r2/(1-r2)))
		rc2i=rotz(-(e+s),roty(d,rotz(e,rbw))) # rotation matrix of transform celestial system to intermediate system
		#
		# the calculation of era is just 1st order in the sofa code, it is lower than the 3rd order of lmst in tempo2? or the lmst is not the era?
		era=2*np.pi*((ut1_jd2+0.7790572732640 + 0.00273781191135448 * (ut1_jd1-dj00+ut1_jd2))%1) # earth rotation angle, Greenwich sidereal time
		self.lst=(era+lon)/(2*np.pi)%1
		t2c=polarmotion@rotz(era,rc2i)
		crs0=t2c.transpose(0,2,1)@(trs0.reshape(3,1))
		crs1=t2c.transpose(0,2,1)@(trs1.reshape(-1,3,1))
		zenith_crs=t2c.transpose(0,2,1)@np.array([[zenith_x],[zenith_y],[zenith_z]])
		sitex,sitey,sitez=crs0.reshape(self.size,3).T
		self.sitepos=vector(sitex,sitey,sitez,center='geo',scale='si',coord='equ',unit=1,type0='pos')
		self.sitepos.equ2ecl()
		self.sitepos.change_unit(sl)
		sitevx,sitevy,sitevz=crs1.reshape(self.size,3).T
		self.sitevel=vector(sitevx,sitevy,sitevz,center='geo',scale='si',coord='equ',unit=1,type0='vel')
		self.sitevel.equ2ecl()
		self.sitevel.change_unit(sl)
		zenith_x_crs,zenith_y_crs,zenith_z_crs=zenith_crs.reshape(self.size,3).T
		self.zenith=vector(zenith_x_crs,zenith_y_crs,zenith_z_crs,center='geo',scale='si',coord='equ',unit=1,type0='pos')
		
		


nutarray=np.array([[ 0, 0, 0, 0,1, -172064161.0, -174666.0, 33386.0, 92052331.0, 9086.0, 15377.0],
[ 0, 0, 2,-2,2, -13170906.0, -1675.0, -13696.0, 5730336.0, -3015.0, -4587.0],
[ 0, 0, 2, 0,2,-2276413.0,-234.0, 2796.0, 978459.0,-485.0,1374.0],
[ 0, 0, 0, 0,2,2074554.0,  207.0, -698.0,-897492.0, 470.0,-291.0],
[ 0, 1, 0, 0,0,1475877.0,-3633.0,11817.0, 73871.0,-184.0,-1924.0],
[ 0, 1, 2,-2,2,-516821.0, 1226.0, -524.0, 224386.0,-677.0,-174.0],
[ 1, 0, 0, 0,0, 711159.0,   73.0, -872.0,  -6750.0,   0.0, 358.0],
[ 0, 0, 2, 0,1,-387298.0, -367.0,  380.0, 200728.0,  18.0, 318.0],
[ 1, 0, 2, 0,2,-301461.0,  -36.0,  816.0, 129025.0, -63.0, 367.0],
[ 0,-1, 2,-2,2, 215829.0, -494.0,  111.0, -95929.0, 299.0, 132.0],
[ 0, 0, 2,-2,1, 128227.0,  137.0,  181.0, -68982.0,  -9.0,  39.0],
[-1, 0, 2, 0,2, 123457.0,   11.0,   19.0, -53311.0,  32.0,  -4.0],
[-1, 0, 0, 2,0, 156994.0,   10.0, -168.0,  -1235.0,   0.0,  82.0],
[ 1, 0, 0, 0,1,  63110.0,   63.0,   27.0, -33228.0,   0.0,  -9.0],
[-1, 0, 0, 0,1, -57976.0,  -63.0, -189.0,  31429.0,   0.0, -75.0],
[-1, 0, 2, 2,2, -59641.0,  -11.0,  149.0,  25543.0, -11.0,  66.0],
[ 1, 0, 2, 0,1, -51613.0,  -42.0,  129.0,  26366.0,   0.0,  78.0],
[-2, 0, 2, 0,1,  45893.0,   50.0,   31.0, -24236.0, -10.0,  20.0],
[ 0, 0, 0, 2,0,  63384.0,   11.0, -150.0,  -1220.0,   0.0,  29.0],
[ 0, 0, 2, 2,2, -38571.0,   -1.0,  158.0,  16452.0, -11.0,  68.0],
[ 0,-2, 2,-2,2,  32481.0,    0.0,    0.0, -13870.0,   0.0,   0.0],
[-2, 0, 0, 2,0, -47722.0,    0.0,  -18.0,    477.0,   0.0, -25.0],
[ 2, 0, 2, 0,2, -31046.0,   -1.0,  131.0,  13238.0, -11.0,  59.0],
[ 1, 0, 2,-2,2,  28593.0,    0.0,   -1.0, -12338.0,  10.0,  -3.0],
[-1, 0, 2, 0,1,  20441.0,   21.0,   10.0, -10758.0,   0.0,  -3.0],
[ 2, 0, 0, 0,0,  29243.0,    0.0,  -74.0,   -609.0,   0.0,  13.0],
[ 0, 0, 2, 0,0,  25887.0,    0.0,  -66.0,   -550.0,   0.0,  11.0],
[ 0, 1, 0, 0,1, -14053.0,  -25.0,   79.0,   8551.0,  -2.0, -45.0],
[-1, 0, 0, 2,1,  15164.0,   10.0,   11.0,  -8001.0,   0.0,  -1.0],
[ 0, 2, 2,-2,2, -15794.0,   72.0,  -16.0,   6850.0, -42.0,  -5.0],
[ 0, 0,-2, 2,0,  21783.0,    0.0,   13.0,   -167.0,   0.0,  13.0],
[ 1, 0, 0,-2,1, -12873.0,  -10.0,  -37.0,   6953.0,   0.0, -14.0],
[ 0,-1, 0, 0,1, -12654.0,   11.0,   63.0,   6415.0,   0.0,  26.0],
[-1, 0, 2, 2,1, -10204.0,    0.0,   25.0,   5222.0,   0.0,  15.0],
[ 0, 2, 0, 0,0,  16707.0,  -85.0,  -10.0,    168.0,  -1.0,  10.0],
[ 1, 0, 2, 2,2,  -7691.0,    0.0,   44.0,   3268.0,   0.0,  19.0],
[-2, 0, 2, 0,0, -11024.0,    0.0,  -14.0,    104.0,   0.0,   2.0],
[ 0, 1, 2, 0,2,   7566.0,  -21.0,  -11.0,  -3250.0,   0.0,  -5.0],
[ 0, 0, 2, 2,1,  -6637.0,  -11.0,   25.0,   3353.0,   0.0,  14.0],
[ 0,-1, 2, 0,2,  -7141.0,   21.0,    8.0,   3070.0,   0.0,   4.0],
[ 0, 0, 0, 2,1,  -6302.0,  -11.0,    2.0,   3272.0,   0.0,   4.0],
[ 1, 0, 2,-2,1,   5800.0,   10.0,    2.0,  -3045.0,   0.0,  -1.0],
[ 2, 0, 2,-2,2,   6443.0,    0.0,   -7.0,  -2768.0,   0.0,  -4.0],
[-2, 0, 0, 2,1,  -5774.0,  -11.0,  -15.0,   3041.0,   0.0,  -5.0],
[ 2, 0, 2, 0,1,  -5350.0,    0.0,   21.0,   2695.0,   0.0,  12.0],
[ 0,-1, 2,-2,1,  -4752.0,  -11.0,   -3.0,   2719.0,   0.0,  -3.0],
[ 0, 0, 0,-2,1,  -4940.0,  -11.0,  -21.0,   2720.0,   0.0,  -9.0],
[-1,-1, 0, 2,0,   7350.0,    0.0,   -8.0,    -51.0,   0.0,   4.0],
[ 2, 0, 0,-2,1,   4065.0,    0.0,    6.0,  -2206.0,   0.0,   1.0],
[ 1, 0, 0, 2,0,   6579.0,    0.0,  -24.0,   -199.0,   0.0,   2.0],
[ 0, 1, 2,-2,1,   3579.0,    0.0,    5.0,  -1900.0,   0.0,   1.0],
[ 1,-1, 0, 0,0,   4725.0,    0.0,   -6.0,    -41.0,   0.0,   3.0],
[-2, 0, 2, 0,2,  -3075.0,    0.0,   -2.0,   1313.0,   0.0,  -1.0],
[ 3, 0, 2, 0,2,  -2904.0,    0.0,   15.0,   1233.0,   0.0,   7.0],
[ 0,-1, 0, 2,0,   4348.0,    0.0,  -10.0,    -81.0,   0.0,   2.0],
[ 1,-1, 2, 0,2,  -2878.0,    0.0,    8.0,   1232.0,   0.0,   4.0],
[ 0, 0, 0, 1,0,  -4230.0,    0.0,    5.0,    -20.0,   0.0,  -2.0],
[-1,-1, 2, 2,2,  -2819.0,    0.0,    7.0,   1207.0,   0.0,   3.0],
[-1, 0, 2, 0,0,  -4056.0,    0.0,    5.0,     40.0,   0.0,  -2.0],
[ 0,-1, 2, 2,2,  -2647.0,    0.0,   11.0,   1129.0,   0.0,   5.0],
[-2, 0, 0, 0,1,  -2294.0,    0.0,  -10.0,   1266.0,   0.0,  -4.0],
[ 1, 1, 2, 0,2,   2481.0,    0.0,   -7.0,  -1062.0,   0.0,  -3.0],
[ 2, 0, 0, 0,1,   2179.0,    0.0,   -2.0,  -1129.0,   0.0,  -2.0],
[-1, 1, 0, 1,0,   3276.0,    0.0,    1.0,     -9.0,   0.0,   0.0],
[ 1, 1, 0, 0,0,  -3389.0,    0.0,    5.0,     35.0,   0.0,  -2.0],
[ 1, 0, 2, 0,0,   3339.0,    0.0,  -13.0,   -107.0,   0.0,   1.0],
[-1, 0, 2,-2,1,  -1987.0,    0.0,   -6.0,   1073.0,   0.0,  -2.0],
[ 1, 0, 0, 0,2,  -1981.0,    0.0,    0.0,    854.0,   0.0,   0.0],
[-1, 0, 0, 1,0,   4026.0,    0.0, -353.0,   -553.0,   0.0,-139.0],
[ 0, 0, 2, 1,2,   1660.0,    0.0,   -5.0,   -710.0,   0.0,  -2.0],
[-1, 0, 2, 4,2,  -1521.0,    0.0,    9.0,    647.0,   0.0,   4.0],
[-1, 1, 0, 1,1,   1314.0,    0.0,    0.0,   -700.0,   0.0,   0.0],
[ 0,-2, 2,-2,1,  -1283.0,    0.0,    0.0,    672.0,   0.0,   0.0],
[ 1, 0, 2, 2,1,  -1331.0,    0.0,    8.0,    663.0,   0.0,   4.0],
[-2, 0, 2, 2,2,   1383.0,    0.0,   -2.0,   -594.0,   0.0,  -2.0],
[-1, 0, 0, 0,2,   1405.0,    0.0,    4.0,   -610.0,   0.0,   2.0],
[ 1, 1, 2,-2,2,   1290.0,    0.0,    0.0,   -556.0,   0.0,   0.0]])
#
cios0=[[[ 0,  0,  0,  0,  1,  0,  0,  0], -2640.73e-6,   0.39e-6 ],
[[ 0,  0,  0,  0,  2,  0,  0,  0],   -63.53e-6,   0.02e-6 ],
[[ 0,  0,  2, -2,  3,  0,  0,  0],   -11.75e-6,  -0.01e-6 ],
[[ 0,  0,  2, -2,  1,  0,  0,  0],   -11.21e-6,  -0.01e-6 ],
[[ 0,  0,  2, -2,  2,  0,  0,  0],     4.57e-6,   0.00e-6 ],
[[ 0,  0,  2,  0,  3,  0,  0,  0],    -2.02e-6,   0.00e-6 ],
[[ 0,  0,  2,  0,  1,  0,  0,  0],    -1.98e-6,   0.00e-6 ],
[[ 0,  0,  0,  0,  3,  0,  0,  0],     1.72e-6,   0.00e-6 ],
[[ 0,  1,  0,  0,  1,  0,  0,  0],     1.41e-6,   0.01e-6 ],
[[ 0,  1,  0,  0, -1,  0,  0,  0],     1.26e-6,   0.01e-6 ],
[[ 1,  0,  0,  0, -1,  0,  0,  0],     0.63e-6,   0.00e-6 ],
[[ 1,  0,  0,  0,  1,  0,  0,  0],     0.63e-6,   0.00e-6 ],
[[ 0,  1,  2, -2,  3,  0,  0,  0],    -0.46e-6,   0.00e-6 ],
[[ 0,  1,  2, -2,  1,  0,  0,  0],    -0.45e-6,   0.00e-6 ],
[[ 0,  0,  4, -4,  4,  0,  0,  0],    -0.36e-6,   0.00e-6 ],
[[ 0,  0,  1, -1,  1, -8, 12,  0],     0.24e-6,   0.12e-6 ],
[[ 0,  0,  2,  0,  0,  0,  0,  0],    -0.32e-6,   0.00e-6 ],
[[ 0,  0,  2,  0,  2,  0,  0,  0],    -0.28e-6,   0.00e-6 ],
[[ 1,  0,  2,  0,  3,  0,  0,  0],    -0.27e-6,   0.00e-6 ],
[[ 1,  0,  2,  0,  1,  0,  0,  0],    -0.26e-6,   0.00e-6 ],
[[ 0,  0,  2, -2,  0,  0,  0,  0],     0.21e-6,   0.00e-6 ],
[[ 0,  1, -2,  2, -3,  0,  0,  0],    -0.19e-6,   0.00e-6 ],
[[ 0,  1, -2,  2, -1,  0,  0,  0],    -0.18e-6,   0.00e-6 ],
[[ 0,  0,  0,  0,  0,  8,-13, -1],     0.10e-6,  -0.05e-6 ],
[[ 0,  0,  0,  2,  0,  0,  0,  0],    -0.15e-6,   0.00e-6 ],
[[ 2,  0, -2,  0, -1,  0,  0,  0],     0.14e-6,   0.00e-6 ],
[[ 0,  1,  2, -2,  2,  0,  0,  0],     0.14e-6,   0.00e-6 ],
[[ 1,  0,  0, -2,  1,  0,  0,  0],    -0.14e-6,   0.00e-6 ],
[[ 1,  0,  0, -2, -1,  0,  0,  0],    -0.14e-6,   0.00e-6 ],
[[ 0,  0,  4, -2,  4,  0,  0,  0],    -0.13e-6,   0.00e-6 ],
[[ 0,  0,  2, -2,  4,  0,  0,  0],     0.11e-6,   0.00e-6 ],
[[ 1,  0, -2,  0, -3,  0,  0,  0],    -0.11e-6,   0.00e-6 ],
[[ 1,  0, -2,  0, -1,  0,  0,  0],    -0.11e-6,   0.00e-6 ]]
#
cios1=[[[ 0,  0,  0,  0,  2,  0,  0,  0],    -0.07e-6,   3.57e-6 ],
[[ 0,  0,  0,  0,  1,  0,  0,  0],     1.71e-6,  -0.03e-6 ],
[[ 0,  0,  2, -2,  3,  0,  0,  0],     0.00e-6,   0.48e-6 ]]
#
cios2=[[[ 0,  0,  0,  0,  1,  0,  0,  0],   743.53e-6,  -0.17e-6 ],
[[ 0,  0,  2, -2,  2,  0,  0,  0],    56.91e-6,   0.06e-6 ],
[[ 0,  0,  2,  0,  2,  0,  0,  0],     9.84e-6,  -0.01e-6 ],
[[ 0,  0,  0,  0,  2,  0,  0,  0],    -8.85e-6,   0.01e-6 ],
[[ 0,  1,  0,  0,  0,  0,  0,  0],    -6.38e-6,  -0.05e-6 ],
[[ 1,  0,  0,  0,  0,  0,  0,  0],    -3.07e-6,   0.00e-6 ],
[[ 0,  1,  2, -2,  2,  0,  0,  0],     2.23e-6,   0.00e-6 ],
[[ 0,  0,  2,  0,  1,  0,  0,  0],     1.67e-6,   0.00e-6 ],
[[ 1,  0,  2,  0,  2,  0,  0,  0],     1.30e-6,   0.00e-6 ],
[[ 0,  1, -2,  2, -2,  0,  0,  0],     0.93e-6,   0.00e-6 ],
[[ 1,  0,  0, -2,  0,  0,  0,  0],     0.68e-6,   0.00e-6 ],
[[ 0,  0,  2, -2,  1,  0,  0,  0],    -0.55e-6,   0.00e-6 ],
[[ 1,  0, -2,  0, -2,  0,  0,  0],     0.53e-6,   0.00e-6 ],
[[ 0,  0,  0,  2,  0,  0,  0,  0],    -0.27e-6,   0.00e-6 ],
[[ 1,  0,  0,  0,  1,  0,  0,  0],    -0.27e-6,   0.00e-6 ],
[[ 1,  0, -2, -2, -2,  0,  0,  0],    -0.26e-6,   0.00e-6 ],
[[ 1,  0,  0,  0, -1,  0,  0,  0],    -0.25e-6,   0.00e-6 ],
[[ 1,  0,  2,  0,  1,  0,  0,  0],     0.22e-6,   0.00e-6 ],
[[ 2,  0,  0, -2,  0,  0,  0,  0],    -0.21e-6,   0.00e-6 ],
[[ 2,  0, -2,  0, -1,  0,  0,  0],     0.20e-6,   0.00e-6 ],
[[ 0,  0,  2,  2,  2,  0,  0,  0],     0.17e-6,   0.00e-6 ],
[[ 2,  0,  2,  0,  2,  0,  0,  0],     0.13e-6,   0.00e-6 ],
[[ 2,  0,  0,  0,  0,  0,  0,  0],    -0.13e-6,   0.00e-6 ],
[[ 1,  0,  2, -2,  2,  0,  0,  0],    -0.12e-6,   0.00e-6 ],
[[ 0,  0,  2,  0,  0,  0,  0,  0],    -0.11e-6,   0.00e-6 ]]
#
cios3=[[[ 0,  0,  0,  0,  1,  0,  0,  0],     0.30e-6, -23.51e-6 ],
[[ 0,  0,  2, -2,  2,  0,  0,  0],    -0.03e-6,  -1.39e-6 ],
[[ 0,  0,  2,  0,  2,  0,  0,  0],    -0.01e-6,  -0.24e-6 ],
[[ 0,  0,  0,  0,  2,  0,  0,  0],     0.00e-6,   0.22e-6 ]]
#
cios4=[[[ 0,  0,  0,  0,  1,  0,  0,  0],    -0.26e-6,  -0.01e-6 ]]
#
ciosp=[94.00e-6, 3808.35e-6, -119.94e-6, -72574.09e-6, 27.70e-6, 15.61e-6]

