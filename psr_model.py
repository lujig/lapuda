import numpy as np
import time_eph as te
import subprocess as sp
import psr_read as pr
#
dm_const=4148.8064224
dm_const_si=7437506.761
kpc2m = 3.08568025e19
mas_yr2rad_s = 1.536281850e-16
pxconv = 1.74532925199432958E-2/3600.0e3
au_dist=149597870691
aultsc=499.00478364 #au_dist/te.sl
gg=6.67259e-11
#
class psr_timing:
	def __init__(self,psr,time,freq):
		self.psr=psr
		self.time=time
		if time.size==np.array(freq).size:
			self.freq=freq
		elif np.array(freq).size==1:
			self.freq=np.ones(time.size)*freq
		else:
			raise 
		self.compute_te_ssb()
		self.bat=self.time.tcb.add(self.dt_ssb)
		self.compute_shklovskii_delay()
		self.compute_phase()
	#
	def copy(self):
		return cp.deepcopy(self)
	#
	def compute_shklovskii_delay(self):
		t0=self.bat.minus(self.psr.posepoch).mjd*86400
		self.shklovskii=t0**2/2/te.sl*self.psr.dshk*kpc2m*(self.psr.pmra**2+self.psr.pmdec**2)*mas_yr2rad_s**2
		self.bbat=self.bat.add(-self.shklovskii)
	#
	def compute_tropospheric_delay(self):
		if hasattr(self.time,'zenith'):
			source_elevation=np.arcsin(self.time.zenith.dot(self.psr.pos_equ)/self.time.site_grs80.z*te.sl)
		else:
			self.tropo_delay=np.zeros(self.time.size)
			return
		pressure=92.5 #101.325 is the normal value # unit in kPa
		zenith_delay_hydrostatic = 0.022768 * pressure / (te.sl * (1.0-0.00266*np.cos(self.time.site_grs80.y) -2.8e-7*self.time.site_grs80.z))
		#
		avgs_a=[1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3]
		avgs_b=[2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3]
		avgs_c=[62.610505e-3, 62.837393e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3]
		amps_a=[0.0,          1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5]
		amps_b=[0.0,          2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 11.723375e-5]
		amps_c=[0.0,          0.0128400e-5, 4.3497037e-5, 84.795348e-5, 170.37206e-5]
		a_h, b_h, c_h = 2.53e-5, 5.49e-3, 1.14e-3
		aas = [5.8021897e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641693e-4]
		bs = [1.4275268e-3,  1.5138625e-3, 1.4572752e-3, 1.5007428e-3, 1.7599082e-3]
		cs = [4.3472961e-2,  4.6729510e-2, 4.3908931e-2, 4.4626982e-2, 5.4736038e-2]
		sin_elevation=np.sin(source_elevation)
		absolute_latitude=self.time.site_grs80.y/np.pi*180
		ilat1=np.int32(np.floor(absolute_latitude/15.0)-1)
		cos_phase=np.cos((self.time.ut1.mjd-53398)*2.0*np.pi/365.25)*np.sign(self.time.site_grs80.y)
		frac=0
		if ilat1<0 or ilat1>=4: 
			if ilat1<0: ilat1=ilat2=0
			elif ilat1>=4: ilat1=ilat2=4
			a = avgs_a[ilat1] - amps_a[ilat1] * cos_phase
			b = avgs_b[ilat1] - amps_b[ilat1] * cos_phase
			c = avgs_c[ilat1] - amps_c[ilat1] * cos_phase
			a1 = aas[ilat1]
			b1 = bs[ilat1]
			c1 = cs[ilat1]
		else:
			frac=absolute_latitude / 15.0 - 1.0 - ilat1
			ilat2=ilat1+1
			a = frac*avgs_a[ilat2] + (1.0-frac)*avgs_a[ilat1] - (frac*amps_a[ilat2] + (1.0-frac)*amps_a[ilat1]) * cos_phase
			b = frac*avgs_b[ilat2] + (1.0-frac)*avgs_b[ilat1] - (frac*amps_b[ilat2] + (1.0-frac)*amps_b[ilat1]) * cos_phase
			c = frac*avgs_c[ilat2] + (1.0-frac)*avgs_c[ilat1] - (frac*amps_c[ilat2] + (1.0-frac)*amps_c[ilat1]) * cos_phase
			a1 = frac*aas[ilat2] + (1.0-frac)*aas[ilat1]
			b1 = frac*bs[ilat2] + (1.0-frac)*bs[ilat1]
			c1 = frac*bs[ilat2] + (1.0-frac)*cs[ilat1]
		basic = (1.0+a/(1.0+b/(1.0+c))) /(sin_elevation+a/(sin_elevation+b/(sin_elevation+c)))
		height_correction = self.time.site_grs80.z * 1.0e-3 * (1.0/sin_elevation - (1.0+a_h/(1.0+b_h/(1.0+c_h))) /(sin_elevation+a_h/(sin_elevation+b_h/(sin_elevation+c_h))))
		mapping_hydrostatic = basic + height_correction
		#
		zenith_delay_wet = 0.0 # ???
		mapping_wet = (1.0+a1/(1.0+b1/(1.0+c1))) /(sin_elevation+a1/(sin_elevation+b1/(sin_elevation+c1)))
		self.tropo_delay=zenith_delay_hydrostatic * mapping_hydrostatic + zenith_delay_wet * mapping_wet
	#
	def compute_dm_delay(self,delt):
		pos=self.psr.pos.add(self.psr.vel.multi(delt,type0='pos'))
		pospos=pos.length()
		self.psr_location=pos.multi(1/pospos)
		rsa=self.time.pos[2].add(self.time.sitepos).minus(self.time.pos[10])
		vobs=self.time.vel[2].add(self.time.sitevel)
		r=rsa.length()
		ctheta=self.psr_location.dot(rsa)/r
		voverc=self.psr_location.dot(vobs)
		self.vchange=(1-voverc)/self.time.einsteinrate
		freqf=self.freq*1e6*self.vchange
		self.freqSSB=freqf
		vobs.ecl2equ()
		dt=(self.time.tcb.minus(self.psr.dmepoch).mjd)/365.25
		yrs=(dt.reshape(-1,1)/np.arange(1,9)).cumprod(axis=1)
		dmdot=(np.array([self.psr.dm1,self.psr.dm2,self.psr.dm3])*yrs[:,:3]).sum(1)
		if self.psr.dmmodel:
			dmval=self.psr.dmmodel+np.interp(self.time.tcb.mjd,self.psr.dmoffs_mjd,self.psr.dmoffs)
			mean_dmoff=self.psr.dmoffs.mean()
			self.psr.dmoffs-=mean_dmoff
		elif hasattr(self.psr,'dm'):
			dmval=self.psr.dm
			dmval+=np.where((self.time.tcb.mjd.reshape(-1,1)>self.psr.dmxr1)&(self.time.tcb.mjd.reshape(-1,1)<self.psr.dmxr2), self.psr.dmx.repeat(self.time.size).reshape(-1,self.time.size).T,0).sum(1)
		dmval+=dmdot # ??? this phrase in this position?
		if self.psr.dm_s1yr:
			dmval+=self.psr.dm_s1yr*np.sin(2*np.pi*dt)+self.psr.dm_c1yr*np.cos(2*np.pi*dt)
		self.tdis1=dmval*dm_const/freqf**2*1e12
		if self.psr.cm1:
			cmval=(np.array([self.psr.cm1,self.psr.cm2,self.psr.cm3])*yrs[:,:3]).sum()
			self.tdis1+=cmval*dm_const*(freqf*1e-6)**(-self.psr.cmidx)
		if self.psr.fddc:
			self.tdis1+=self.psr.fddc/(freqf*1e-6)**self.psr.fddi
			self.tdis1+=np.polyval(self.psr.fd[::-1],(freqf*1e-9))*(freqf*1e-9)
		#
		#if self.psr.ephver==5: ne_sw=4  # ???
		#else: ne_sw=9.961
		ne_sw=4
		self.tdis2=1.0e6*te.au_dist**2/te.sl/dm_const_si*ne_sw*np.arccos(ctheta)/r/np.sqrt(1.0-ctheta**2)/freqf**2
		#self.tdis2=self.solarWindModel()/dm_const/1.0e-12/freqf/freqf   #More complex solar wind model introduced by Xiaopeng You and William Coles, but the data has been changed.
	#
	def solarWindModel(self):
		deg2rad = np.pi/180.0
		yr,mm,dd,secs,iday=te.mjd2datetime(self.time.ut1.date)
		zlonan = 1.28573 + 7.89327 * (yr -1900 + iday/365.+ 50. )/ 32400.
		zkl=np.cos(self.psr.elat)
		dj= 365*(yr-1900) + (yr-1901)/4 + iday + self.time.ut1.second/86400.0 - 0.5
		g=(358.475845+0.985600267*dj)%360.0*deg2rad
		ya= ( (279.696678+0.9856473354*dj)%360.0 +(1.91946-0.004789*dj/36525.0)*np.sin(g)+0.020094*np.sin(2.*g))*deg2rad
		cl0 = np.cos(ya-self.psr.elong)*zkl
		dp = np.sqrt(1.0-cl0**2)
		theta=np.arange(0,90,5)*deg2rad
		cl=cl0.reshape(-1,1)+dp.reshape(-1,1)*np.tan(theta).reshape(1,-1)
		nLineOfSight = (cl>0).sum(1)+1
		nLineOfSight[nLineOfSight>36]=36
		cl=cl[np.arange(self.time.size),nLineOfSight]
		clan=np.cos(zlonan)
		slan=np.sin(zlonan)
		x=np.array([[clan,slan,0.0],[-0.9920049497*slan,0.9920049497*clan,0.1261989691],[0.1261989691*slan,-0.1261989691*clan,0.9920049497]])
		sl=np.sqrt((1-cl0**2)+(cl0-cl)**2)
		sn=np.array([-np.cos(ya),-np.sin(ya),np.zeros_like(ya)]).reshape(-1,3,1)
		a=np.array([np.cos(self.psr.elong)*zkl,np.sin(self.psr.elong)*zkl,np.sin(self.psr.elat)*np.ones_like(zkl)]).reshape(-1,3,1)
		b=a*cl.reshape(self.time.size,1,1)+sn
		c=x@a
		c1c2=np.sqrt(c[:,0]**2+c[:,1]**2)
		beta=np.arctan(c[:,2],c1c2)
		a=x@b
		helat=np.arctan2(a[:,2],np.sqrt(a[:,0]**2+a[:,1]**2)).reshape(-1)
		b=x@sn
		b1b2=np.sqrt(b[:,0]**2+b[:,1]**2)
		helate=np.arctan2(b[:,2],b1b2)
		cep=-((b[:,0]*c[:,0]+b[:,1]*c[:,1])/b1b2/c1c2).reshape(-1)
		cep[cep>1]=1
		delng=np.arctan2(np.sqrt(1-cep**2),cep)
		blp=np.arctan2(a[:,1],a[:,0]).reshape(-1)%(2*np.pi)
		ble=np.arctan2(b[:,1],b[:,0]).reshape(-1)%(2*np.pi)
		zlagtm=te.au_dist/1000*sl/3600.0/400 # 400 is the velocity of the solarwind
		zhr = (24.*((yr-1969)*365.+(yr-1969)//4+iday-341.0) + self.time.ut1.second/3600. -14.77).reshape(-1,1)
		zle1 = 228.42-zhr/1.81835
		zle2 = 228.42 - zhr/1.692002 + ble/deg2rad
		crlne=zle2%360
		rote1=1556.-zle1/360.0
		rote2 = 1556.-zle2/360.0
		rote=np.floor(rote1)+rote2%1
		rote[rote1-rote>4/360]+=1
		rote[rote1-rote<=-4/360]-=1
		dlon=blp-ble
		delcrl=zlagtm/1.692002+dlon/deg2rad
		zlpp=zle2+delcrl
		crlon=zlpp%360
		rots=np.floor(rote)-crlon/360
		rots[crlon>crlne]=np.floor(rote)+1-crlon/360
		rots[crlon<3.6]=np.floor(rote)-0.01
		crlon*=deg2rad
		crlne*=deg2rad
		integral = np.zeros(self.time.size)
		for k in np.arange(self.time.size):
			filenum=rots[k]
			fname='solarWindModel/CS'+str(filenum)+'.txt'
			f=open(fname)
			currentLon=f.read()
			currentLat=f.read()  #The file is missing
			f.close()
			tmp=np.sin((currentLat*deg2rad-helat[k])/2)**2+np.cos(helat[k])*np.cos(crlon[k])*np.sin((currentLon*deg2rad-crlon[k])/2)**2
			closestAngle=np.min(2*np.arctan2(np.sqrt(tmp),np.sqrt(1-tmp)))/deg2rad
			if closestAngle < 20:  # fast_slow_angle
				integral[k]=10*nLineOfSight  #slow_ne
			else:
				integral[k]=2.5*nLineOfSight # fast_ne
		dm_sun=integral*5*deg2rad*4.85e-6/dp
		return dm_sun

	def compute_shapiro_delay(self,delt):
		num=list(range(11))
		num.pop(3)
		self.shapiro_solar_system=0
		pos=self.psr.pos.add(self.psr.vel.multi(delt,type0='pos'))
		self.psr_location=pos.multi(1/pos.length())
		self.shapiro=[]
		for i in np.arange(11):
			if i==10: mass=self.time.cons['GMS']*te.iftek
			elif i==9: mass=self.time.cons['GMB']/(1+self.time.cons['EMRAT'])*te.iftek
			elif i==2: continue
			else: mass=self.time.cons['GM'+str(i+1)]*te.iftek
			rsa=self.time.pos[2].add(self.time.sitepos).minus(self.time.pos[i])
			r=rsa.length()
			ctheta=self.psr_location.dot(rsa)/r
			self.shapiro.append(-2.0*mass*1e4*aultsc/te.sl*np.log(r/aultsc*(1+ctheta)))
			self.shapiro_solar_system+=-2.0*mass*1e4*aultsc/te.sl*np.log(r/aultsc*(1+ctheta))
	#
	def compute_te_ssb(self):
		self.compute_tropospheric_delay()
		#self.compute_ionospheric_delay() #???
		rca=self.time.sitepos.add(self.time.pos[2])
		#
		pmrvrad=self.psr.pmrv*np.pi/180/36000 # from mas/yr to radians/century
		dt_ssb=0
		loop=0
		while True:
			dt_ssb_old=dt_ssb
			rcos1=self.psr.pos.dot(rca)
			rr=rca.dot(rca)
			delt=(self.time.tcb.minus(self.psr.posepoch).mjd+dt_ssb/86400)/36525
			pmtrans_rcos2=self.psr.vel.dot(rca)
			# proper motion
			dt_pm=delt*pmtrans_rcos2
			dt_pmtt=-0.5*self.psr.vel.dot(self.psr.vel)*delt**2*rcos1 # second order of pm
			dt_pmtr=-delt**2*pmrvrad*pmtrans_rcos2
			# proper motion acceleration
			acctrans_rcos2=self.psr.acc.dot(rca)
			dt_acctrans=0.5*delt**2*acctrans_rcos2
			# parallax
			dt_px=-0.5*self.psr.px*pxconv*(rr-rcos1**2)/aultsc
			# roemer delay
			self.roemer=rcos1+dt_pm+dt_pmtt+dt_pmtr+dt_acctrans+dt_px
			#
			self.compute_shapiro_delay(delt)
			self.compute_dm_delay(delt)
			#
			dt_ssb=self.roemer-self.tdis1-self.tdis2-self.shapiro_solar_system
			if loop>1000 or np.max(np.abs(dt_ssb-dt_ssb_old))<1e-10:  # or self.veryfast
				break
			loop+=1
		#
		rca.change_unit(te.sl)
		rcaxyz=rca.xyz()
		self.dtdposequ=rcaxyz@self.psr.dpos('pos','ecl','equ').T
		self.dtdposecl=rcaxyz@self.psr.dpos('pos','ecl','ecl').T
		self.dtdvelequ=(rcaxyz*(delt-delt**2*pmrvrad).reshape(-1,1)-self.psr.vel.xyz()*(delt**2*rcos1).reshape(-1,1))@self.psr.dpos('vel','ecl','equ').T
		self.dtdvelecl=(rcaxyz*(delt-delt**2*pmrvrad).reshape(-1,1)-self.psr.vel.xyz()*(delt**2*rcos1).reshape(-1,1))@self.psr.dpos('vel','ecl','ecl').T
		self.dtdaccequ=0.5*delt.reshape(-1,1)**2*rcaxyz@self.psr.dpos('acc','ecl','equ').T
		self.dtdaccecl=0.5*delt.reshape(-1,1)**2*rcaxyz@self.psr.dpos('acc','ecl','ecl').T
		self.dt_ssb=dt_ssb-self.tropo_delay # -self.iono_delay # ???
	#
	def phase_der_para(self,paras):
		nparas=len(paras)
		deriv=np.zeros([nparas,self.time.size])
		if not pr.para_with_err.intersection(paras):
			raise Exception('One or more parameters is invalid.')
		binary_paras=list(pr.paras_binary.intersection(paras))
		lbp=len(binary_paras)
		if lbp:
			if not self.psr.binary:
				raise Exception('One or more parameters belongs to binary parameters, but the pulsar '+self.psr.name+' is not a binary pulsar.')
			binary_der=self.compute_binary_der()
			all_bparas_set=pr.__getattribute__('paras_'+self.psr.binary)
			all_bparas=np.array(all_bparas_set['necessary']+all_bparas_set['optional'])
			for i in np.arange(lbp):
				para=binary_paras[i]
				if para not in all_bparas:
					raise Exception("Parameter "+para+" is not a parameter for "+self.psr.binary+" model.")
				ind1=np.where(np.array(paras)==para)[0][0]
				ind2=np.where(all_bparas==para)[0][0]
				deriv[ind1]=-binary_der[ind2]*self.dphasedt
		nfparas={'f0','f1','f2','f3','f4','f5'}.intersection(paras)
		for para in list(nfparas):
			ind=np.where(np.array(paras)==para)[0]
			deriv[ind]=self.__getattribute__('dphased'+para)
		npparas={'raj','decj','pmra','pmdec','pmra2','pmdec2','elong','elat','pmelong','pmelat','pmelong2','pmelat2'}.intersection(paras)
		if npparas:
			dphasedraj,dphaseddecj=self.dtdposequ.T*self.dphasedt
			dphasedpmra,dphasedpmdec=self.dtdvelequ.T*self.dphasedt
			dphasedpmra2,dphasedpmdec2=self.dtdaccequ.T*self.dphasedt
			dphasedelong,dphasedelat=self.dtdposecl.T*self.dphasedt
			dphasedpmelong,dphasedpmelat=self.dtdvelecl.T*self.dphasedt
			dphasedpmelong2,dphasedpmelat2=self.dtdaccecl.T*self.dphasedt
			for para in list(npparas):
				ind=np.where(np.array(paras)==para)[0]
				deriv[ind]=eval('dphased'+para)
		return deriv
	#
	def compute_phase(self):
		coalesceFlag,coalesceFlag_p=0,0
		deltaT=self.bbat.minus(self.psr.pepoch)
		if self.psr.binary:
			self.torb=self.compute_binary()
			deltaT=deltaT.add(self.torb)
		ftpd=deltaT.second/deltaT.unit
		ntpd=deltaT.date*1.0
		tmp=str(self.psr.f0).split('.')
		nf0=int(tmp[0])
		ff0=np.float64('0.'+tmp[1])
		phase2=nf0*ftpd+ff0*ntpd+ftpd*ff0
		phase2*=86400
		phase0=te.phase(ntpd*nf0*86400,np.zeros_like(ntpd))
		phase0=phase0.add(phase2)
		#
		deltat=deltaT.mjd*deltaT.unit
		self.dphasedf0,self.dphasedf1,self.dphasedf2,self.dphasedf3,self.dphasedf4,self.dphasedf5=(deltat/np.arange(1,7).reshape(-1,1)).cumprod(axis=0)
		phase3=1/2*deltat**2*(self.psr.f1+1/3*deltat*(self.psr.f2+1/4*deltat*(self.psr.f3+1/5*deltat*(self.psr.f4+1/6*deltat*self.psr.f5))))  # precision???
		self.dphasedt=self.psr.f0+deltat*(self.psr.f1+1/2*deltat*(self.psr.f2+1/3*deltat*(self.psr.f3+1/4*deltat*(self.psr.f4+1/5*deltat*self.psr.f5))))
		self.period_now=1/self.dphasedt
		if 'mode_switch' in self.psr.paras: phase2state=self.psr.mode_switch
		else: phase2state=0
		if 'brake' in self.psr.paras:
			arg3=deltat**3
			arg4=deltat*arg3
			f1=self.psr.f1
			f0=self.psr.f0
			bindex=self.psr.brake
			f2brake=bindex*f1**2/f0
			f3brake=bindex*(2*bindex-1)*f1**3/f0**2
			phase3+=f2brake*arg3/6+f3brake*args4/24
		#
		phase4=0
		if 'glep' in self.psr.paras:
			for k in np.arange(len(self.psr.glep)):
				tp=deltat*86400
				tgl=self.psr.glep[k].minus(self.psr.pepoch).mjd*86400
				if tp>=tgl:
					dt1=tp-tgl
					if self.psr.gltd[k]: expf=np.exp(-dt1/86400/self.psr.gltd[k])
					else: expf=1.0
					phase4+=self.psr.glph[k]+dt1*(self.psr.glf0[k]+1/2*dt1*(self.psr.glf1[k]+1/3*dt1*self.psr.glf2[k]))+self.psr.glf0d[k]*self.psr.gltd[k]*86400*(1-expf)
		# glep,gltd,glph,glf0,glf1,glf2,glf0d,expep,expindex,expph,exptau,gausep,gausamp,gaussig,gausindex,wave_om,wave_om_dm,wave_epoch,wave_sin,wave_cos,wave_scale,wave_sin_dm,wave_cos_dm
		if 'expep' in self.psr.paras:
			for k in np.arange(len(self.psr.expep)):
				dt=self.bbat.minus(self.psr.expep[k]).mjd
				if dt>0:
					if self.psr.expindex[k]: gamma=self.psr.expindex[k]
					else: gamma=-2
					freq=self.freqSSB/1.4e9
					dm=self.psr.expph[k]
					tau=self.psr.exptau[k]
					phase4+=dm*freq**gamma*np.exp(-dt/tau)*self.psr.f0
		#
		if 'gausep' in self.psr.paras:
			for i in np.arange(len(self.psr.gausep)):
				freq=self.freqSSB/1.4e9
				dt=self.bbat.minus(self.psr.gausep[k]).mjd
				amp=self.psr.gausamp[k]
				sig=self.psr.gaussig[k]
				if self.psr.gausindex[k]: gamma=self.psr.gausindex[k]
				else: gamma=0
				val=amp*freq**gamma*np.exp(-dt**2/sig**2/2)
				phase4+=val*self.psr.f0
		#
		phaseJ=0
		if 'obsNjump' in self.psr.paras:
			for k in np.arange(len(self.psr.njumps)):
				for l in np.arange(len(self.obsNjump)):
					if self.jump[l]==k and self.psr.jumpSAT[l]==0:
						phaseJ+=self.psr.jumpval*self.psr.f0
			for k in np.arange(len(self.psr.nfdjumps)):
				for l in np.arange(len(self.obsNfdjump)):
					if self.fdjump[l]==k:
						idx=self.psr.fdjumpidx[k]
						phaseJ+=self.psr.fdjumpval[k]*(self.freqSSB/1e9)**idx*self.psr.f0
		#
		phaseW=0
		if 'wave_om' in self.psr.paras:
			dt=self.bbat.minus(self.psr.wave_epoch).mjd
			om=self.psr.wave_om
			for k in np.arange(len(self.psr.nwhite)):
				om_eff=om*(1+k)
				if self.psr.wave_scale==0:
					phaseW+=self.psr.wave_sin[k]*np.sin(om_eff*dt)*self.psr.f0+self.psr.wave_cos[k]*np.cos(om_eff*dt)*self.psr.f0
				elif self.psr.wave_scale==1:
					freq=self.freq
					phaseW+=(self.psr.wave_sin[k]*np.sin(om_eff*dt)*self.psr.f0+self.psr.wave_cos[k]*np.cos(om_eff*dt)*self.psr.f0)*dm_const/freq**2
				elif self.psr.wave_scale==2:
					freq=self.freq
					phaseW+=self.psr.wave_sin[k]*np.sin(om_eff*dt)*self.psr.f0+self.psr.wave_cos[k]*np.cos(om_eff*dt)*self.psr.f0
					phaseW+=(self.psr.wave_sin_dm[k]*np.sin(om_eff*dt)*self.psr.f0+self.psr.wave_cos_dm[k]*np.cos(om_eff*dt)*self.psr.f0)*dm_const/freq**2
		if 'wave_om_dm' in self.psr.paras:
			dt=self.bbat.minus(self.psr.wave_epoch).mjd
			om=self.psr.wave_om_dm
			om_eff=om*(1+k)
			freq=self.freq
			phaseW+=(self.psr.wave_sin_dm[k]*np.sin(om_eff*dt)*self.psr.f0+self.psr.wave_cos_dm[k]*np.cos(om_eff*dt)*self.psr.f0)*dm_const/freq**2
		# quad_om
		if 'quad_om' in self.psr.paras:
			dt=self.bbat.minus(self.psr.quadepoch).mjd*86400
			resp,resc,costheta=calculate_gw(self.psr.raj,self.psr.decj,self.psr.quadra,self.psr.quaddec)
			for k in np.arange(self.psr.nquad):
				omega_g=self.psr.quad_om*(k+1)
				tmp_r=self.psr.quad_aplus_r[k]*resp+self.psr.quad_across_r[k]*resc
				tmp_i=self.psr.quad_aplus_i[k]*resp+self.psr.quad_across_i[k]*resc
				res_r,res_i=tmp_r*np.sin(omgea_g*dt),tmp_i*np.cos(omgea_g*dt)
				if self.psr.gwsrc_psrdist>0:
					tmp=omega_g*dt-(1-costheta)*self.psr.gwsrc_psrdist/te.sl*omega_g
					res_r-=tmp_r*np.sin(tmp)
					res_i-=tmp_i*np.cos(tmp)
				if costheta==1: res_r,res_i=0,0
				else:
					res_r=1/(2*omega_g*(1-costheta))*res_r
					res_i=1/(2*omega_g*(1-costheta))*res_i
				phaseW+=(res_r+res_i)*self.psr.f0
		#
		if 'gwecc' in self.psr.paras:
			ra_p=self.psr.raj
			dec_p=self.psr.decj
			ra_g=self.psr.gwecc_ra
			dec_g=self.psr.gwecc_dec
			cosmu=np.cos(dec_g)*np.cos(dec_p)*np.cos(ra_g-ra_p)+np.sin(dec_g)*np.sin(dec_p)
			prev_p,prev_p_p=self.psr.gwecc_orbital_period*365.25*86400,self.psr.gwecc_orbital_period*365.25*86400
			prev_e,prev_e_p=self.psr.gwecc_e,self.psr.gwecc_e
			prev_epoch=self.psr.gwecc_epoch
			prev_epoch_p=prev_epoch+1/86400*(self.psr.gwecc_psrdist*3.08568e19/te.sl)*(1-cosmu)
			prev_theta,prev_theta_p=self.psr.gwecc_theta_0
			prev_a=(6.67e-11*(self.psr.gwecc_m1+self.psr.gwecc_m2)*19891e30*(2*np.pi/prev_p)**-2)**(1/3)
			prev_a_p=(6.67e-11*(self.psr.gwecc_m1+self.psr.gwecc_m2)*19891e30*(2*np.pi/prev_p)**-2)**(1/3)
			if self.psr.gwecc_pulsar_term_on==0:
				phaseW+=self.psr.gwecc*(self.eccRes(prev_p,prev_e,prev_a,prev_epoch,prev_theta))*self.psr.f0
			elif self.psr.gwecc_pulsar_term_on==1:
				phaseW+=self.psr.gwecc*(self.eccRes(prev_p,prev_e,prev_a,prev_epoch,prev_theta)-self.eccRes(prev_p_p,prev_e_p,prev_a_p,prev_epoch_p,prev_theta_p))*self.psr.f0
			elif self.psr.gwecc_pulsar_term_on==2:
				phaseW+=self.psr.gwecc*(-self.eccRes(prev_p_p,prev_e_p,prev_a_p,prev_epoch_p,prev_theta_p))*self.psr.f0
		#
		if ('gwsingle' in self.psr.paras) or ('cgw' in self.psr.paras):
			dt=self.bbat.minus(self.psr.gwsrc_epoch).mjd*86400
			resp,resc,costheta=calculate_gw(self.psr.raj,self.psr.decj,self.psr.gwsrc_ra,self.psr.gwsrc_dec)
			if 'gwsingle' in self.psr.paras:
				omega_g=self.psr.gwsingle
				tmp_r=self.psr.gwsrc_aplus_r[k]*resp+self.psr.gwsrc_across_r[k]*resc
				tmp_i=self.psr.gwsrc_aplus_i[k]*resp+self.psr.gwsrc_across_i[k]*resc
				res_r,res_i=tmp_r*np.sin(omgea_g*dt),tmp_i*np.cos(omgea_g*dt)
				if self.psr.gwsrc_psrdist>0:
					tmp=omega_g*dt-(1-costheta)*self.psr.gwsrc_psrdist/te.sl*omega_g
					res_r+=tmp_r*np.sin(tmp)
					res_i+=tmp_i*np.cos(tmp)
				if costheta==1: res_r,res_i=0,0
				else:
					res_r=1/(2*omega_g*(1-costheta))*res_r
					res_i=1/(2*omega_g*(1-costheta))*res_i
				phaseW+=(res_r+res_i)*self.psr.f0
			else:
				omega_g=self.psr.cgw
				res_r=(self.psr.cgw_h0/omega_g*((1+self.psr.cgw_cosinc**2)*np.cos(2*self.psr.cgw_angpol)*np.sin(omega_g*dt)+2*self.psr.cgw_cosinc*np.sin(2*self.psr.cgw_angpol) *np.cos(omega_g*dt)))*resp+(self.psr.cgw_h0/omega_g*((1+self.psr.cgw_cosinc**2)*np.sin(2*self.psr.cgw_angpol)*np.sin(omega_g*dt)+2*self.psr.cgw_cosinc*np.cos(2*self.psr.cgw_angpol) *np.cos(omega_g*dt)))*resp
				res_i=0
				if self.psr.gwsrc_psrdist>0:
					if self.psr.cgw_mc==0:
						omega_prime_g,h0_prime=omega_g,self.psr.cgw_h0
					else:
						omega_prime_g=2*np.pi*((1-costheta)*self.psr.gwsrc_psrdist/te.sl*256/5/te.sl**5*np.pi**(8/3)*(1.3271243999e20*self.psr.cgw_mc)**(5/3)+(omega_g/2.0/np.pi)**(-8/3))**(-3/8)
						h0_prime=self.psr.cgw_h0*(omega_prime_g/omega_g)**(2/3)
					tmp1=omega_prime_g*dt-(1-cosTheta)*self.psr.gwsrc_psrdist/te.sl*omega_prime_g
					tmp2=2*self.psr.cgw_angpol
					tmp_p=h0_prime/omega_prime_g*((1+self.psr.cgw_cosinc**2)*np.cos(tmp2)*np.sin(tmp1)+2*self.psr.cgw_cosinc*np.sin(tmp2)*np.cos(tmp1))
					tmp_c=h0_prime/omega_prime_g*((1+self.psr.cgw_cosinc**2)*np.sin(tmp2)*np.sin(tmp1)-2*self.psr.cgw_cosinc*np.cos(tmp2)*np.cos(tmp1))
					res_r-=tmp_p*res+tmp_c*resc
				if costheta==1: res_r,res_i=0,0
				else:
					res_r=1/(2*omega_g*(1-costheta))*res_r
					res_i=1/(2*omega_g*(1-costheta))*res_i
				phaseW+=(res_r+res_i)*self.psr.f0
		#
		if 'amp' in self.psr.paras:
			self.psr.quad_ifunc_geom_p=0
			self.psr.quad_ifunc_geom_c=0
			if self.psr.amp[0] and not self.psr.amp[1]:
				dt=self.bbat.minus(self.psr.gwm_epoch).mjd*86400
				lambda_p=self.psr.raj
				beta_p=self.psr.decj
				lamb=self.psr.gwm_raj
				beta=self.psr.gwm_decj
				clp,slp,cbp,sbp,cl,sl,cb,sb=np.cos(lambda_p),np.sin(lambda_p),np.cos(beta_p),np.sin(beta_p),np.cos(lamb),np.sin(lamb),np.cos(beta),np.sin(beta)
				vn=np.array([[clp*cbp,slp*cbp,sbp]])
				vg=np.array([[-cl*cb,sl*cb,sb]])
				costheta=cb*cbp*np.cos(lamb-lambda_p)+sb*sbp
				vd=np.array([[0,0,1]])
				if self.time.tcb.mjd>self.psr.gwm_epoch.mjd:
					if beta>0: vd=te.normalize(vg*np.cos(np.pi/2-beta)+vd)
					else: vd=te.normalize(vg*np.cos(-np.pi/2-beta)-vd)
					va=te.normalize(te.multiply(vd,vg))
					vm=vd*np.cos(self.psr.gwm_phi)+va*np.sin(self.psr.gwm_phi)
					if np.abs(costheta)!=1.0: 
						vg*=costheta
						vl=vn-vg
						cosphi=(vl*vm).sum(-1)/(vl**2).sum(-1)
						cos2phi=2*cosphi**2-1
					else:
						cos2phi=0
					dt0=self.time.tcb.minus(self.psr.gwm_epoch).mjd*86400
					scale=-0.5*cos2phi*(1-costheta)
					phaseW+=scale*(self.psr.amp[0]*self.psr.f0)*dt0+self.psr.gwm_dphase
			#
			if self.psr.amp[0] and self.psr.amp[1]:
				if self.time.tcb.mjd>self.psr.gwm_epoch.mjd:
					dt=self.bbat.minus(self.psr.gwm_epoch).mjd*86400
					resp,resc,costheta=calculate_gw(self.psr.raj,self.psr.decj,self.psr.gwm_ra,self.psr.gwm_dec)
					if costheta==1.0: resp,resc=0.0,0.0
					else:
						resp=1/(2*(1-costheta))*resp
						resc=1/(2*(1-costheta))*resc
					self.psr.quad_ifunc_geom_p=resp
					self.psr.quad_ifunc_geom_c=resc
					phaseW+=(self.psr.gwm_amp[0]*resp+self.psr.gwm_amp[1]*resc)*self.psr.f0*dt
		else:
			self.psr.quad_ifunc_geom_p=0
			self.psr.quad_ifunc_geom_c=0
		#
		if 'gwcs_amp' in self.psr.paras:
			dt=self.bbat.minus(self.psr.gwcs_epoch).mjd*86400
			resp,resc,costheta=calculate_gw(self.psr.raj,self.psr.decj,self.psr.gwcs_raj,self.psr.gwcs_dec)
			if costheta==1.0: resp,resc=0.0,0.0
			else:
				resp=1/(2*(1-costheta))*resp
				resc=1/(2*(1-costheta))*resc
			self.psr.gwcs_geom_p=resp
			self.psr.gwcs_geom_c=resc
			width=self.psr.gwcs_width*86400
			width_day=self.psr.gwcs_width
			if self.time.tcb.mjd<(self.psr.gwcs_epoch.mjd-width_day/2.0): extra1=extra2=0
			elif self.time.tcb.mjd<self.psr.gwcs_epoch.mjd:
				tmp=(3/4*((0.5*width)**(4/3)-np.abs(dt)**(4/3))-(0.5*width)**(1/3)*(dt+0.5*width))
				extra=self.psr.gwcs_amp*tmp
			elif self.time.tcb.mjd<=(self.psr.gwcs_epoch.mjd+width_day/2.0):
				tmp=(3/4*((0.5*width)**(4/3)+np.abs(dt)**(4/3))-(0.5*width)**(1/3)*(dt+0.5*width))
				extra=self.psr.gwcs_amp*tmp
			else:
				extra=-0.25*(0.5**(1/3)*self.psr.gwcs_amp*width**(4/3))
			phaseW=(extra[0]*resp+extra[1]*resc)*self.psr.f0
		else:
			self.psr.gwcs_geom_p=0
			self.psr.gwcs_geom_c=0
		#
		if 'clk_offsT' in self.psr.paras:
			for k in np.arange(len(self.psr.clk_offsT)-1):
				if self.time.bbat.mjd>=self.psr.clk_offsT[k] and self.time.bbat.mjd<self.psr.clk_offsT[k+1]:
					phaseW+=self.psr.clk_offsV[k]*self.psr.f0
		#
		if 'ifunc' in self.psr.paras:
			if self.psr.ifunc==1: #sinc interpolation
				speriod=self.psr.ifuncT[1]-self.psr.ifuncT[0]
				for k in np.arange(len(self.psr.ifuncT)):
					dt=self.psr.bbat.minus(self.psr.ifuncT[k]).mjd
					wi=1
					if dt==0: t1=wi*self.psr.ifuncV[k]
					else:
						tt=np.pi/speriod*dt
						t1=wi*self.psr.ifuncV[k]*np.sin(tt)/tt
				phaseW+=t1*self.psr.f0
			elif self.psr.ifunc==2: # linear interpolation
				ival=np.interp(self.time.tcb.mjd,self.psr.ifuncT,self.psr.ifuncV)
				phaseW+=ival*self.psr.f0
			elif self.psr.ifunc==0: # no interpolation
				for k in np.arange(len(self.psr.ifuncT)):
					if self.time.tcb.mjd>self.psr.ifuncT[k]:
						ival=self.ifuncV[k]
						break
				phaseW+=ival*self.psr.f0
		#
		if 'quad_ifunc_p' in self.psr.paras:
			if (self.psr.quad_ifunc_geom_p == 0) & (self.psr.quad_ifunc_p in [0,1,3]) :
				if self.psr.quad_ifunc_p==3: self.psr.quad_ifunc_geom_p =1
				else:
					resp,resc,costheta=calculate_gw(self.psr.raj,self.psr.decj,self.psr.quad_ifunc_p_ra,self.psr.quad_ifunc_p_dec)
					if costheta==1: resp=0
					else: resp=resp/(2*(1-costheta))
					self.psr.quad_ifunc_geom_p = resp
				ival=np.interp(self.time.tcb,self.psr.quad_ifuncT_p,self.psr.quad_ifuncV_p)
				phaseW+=self.psr.f0*ival*self.psr.quad_ifunc_geom_p
			if (self.psr.quad_ifunc_p==2):
				if self.psr.quad_ifunc_geom_p == 0:
					if self.psr.simflag: lp,bp=self.psr.rasim,self.psr.decsim
					else: lp,bp=self.psr.raj,self.psr.decj
					lg,bg=self.psr.quad_ifunc_p_ra,self.psr.quad_ifunc_p_dec
					clp,slp,cbp,sbp,clg,slg,cbg,sbg=np.cos(lp),np.sin(lp),np.cos(bp),np.sin(bp),np.cos(lg),np.sin(lg),np.cos(bg),np.sin(bg)
					vp=np.array([cbp*clp,cbp*slp,sbp])
					vg=np.array([cbg*clg,cbg*slg,sbg])
					ctheta=(vp*vg).sum(-1)
					vrho=p-ctheta*vg
					vpol=np.array([sbg*clg,sbg*slg,-cbg])
					rho=np.sqrt((vrho**2).sum(-1))
					pol=np.sqrt((vpol**2).sum(-1))
					cphi=(vrho*vpol).sum()/(rho*pol)
					phi=np.arccos(cphi)
					c2phi=2*cphi**2-1
					resp=0.5*(1-ctheta)*c2phi
					self.psr.quad_ifunc_geom_p = resp
				ival=np.interp(self.time.tcb,self.psr.quad_ifuncT_p,self.psr.quad_ifuncV_p)
				phaseW+=self.psr.f0*ival*self.psr.quad_ifunc_geom_p
		#
		if 'quad_ifunc_c' in self.psr.paras:
			if (self.psr.quad_ifunc_geom_c == 0) & (self.psr.quad_ifunc_c in [0,1,3]) :
				if self.psr.quad_ifunc_c==3: self.psr.quad_ifunc_geom_c =1
				else:
					resp,resc,costheta=calculate_gw(self.psr.raj,self.psr.decj,self.psr.quad_ifunc_c_ra,self.psr.quad_ifunc_c_dec)
					if costheta==1: resc=0
					else: resc=resc/(2*(1-costheta))
					self.psr.quad_ifunc_geom_c = resc
				ival=np.interp(self.time.tcb,self.psr.quad_ifuncT_c,self.psr.quad_ifuncV_c)
				phaseW+=self.psr.f0*ival*self.psr.quad_ifunc_geom_c
			if (self.psr.quad_ifunc_c==2):
				if self.psr.quad_ifunc_geom_c == 0:
					if self.psr.simflag: lp,bp=self.psr.rasim,self.psr.decsim
					else: lp,bp=self.psr.raj,self.psr.decj
					lg,bg=self.psr.quad_ifunc_c_ra,self.psr.quad_ifunc_c_dec
					clp,slp,cbp,sbp,clg,slg,cbg,sbg=np.cos(lp),np.sin(lp),np.cos(bp),np.sin(bp),np.cos(lg),np.sin(lg),np.cos(bg),np.sin(bg)
					vp=np.array([cbp*clp,cbp*slp,sbp])
					vg=np.array([cbg*clg,cbg*slg,sbg])
					ctheta=(vp*vg).sum(-1)
					vrho=vp-ctheta*vg
					vtheta=np.array([sbg*clg,sbg*slg,-cbg])
					vphi=np.array([-slg,clg,0])
					vpol=vtheta+vphi
					rho=np.sqrt((vrho**2).sum(-1))
					pol=np.sqrt((vpol**2).sum(-1))
					cphi=(vrho*vpol).sum()/(rho*pol)
					phi=np.arccos(cphi)
					c2phi=2*cphi**2-1
					resc=0.5*(1-ctheta)*c2phi
					self.psr.quad_ifunc_geom_c = resc
				ival=np.interp(self.time.tcb,self.psr.quad_ifuncT_c,self.psr.quad_ifuncV_c)
				phaseW+=self.psr.f0*ival*self.psr.quad_ifunc_geom_c
		#
		if 'gwb_amp' in self.psr.paras:
			if self.psr.gwb_amp[0] and self.psr.gwb_amp[1]:
				resp,resc,costheta=calculate(self.psr.raj,self.psr.decj,self.psr.gwb_raj,self.psr.gwb_decj)
				if costheta==1: resp,resc=0,0
				else: 
					resp=resp/(2*(1-costheta))
					resc=resc/(2*(1-costheta))
				self.psr.gwb_geom_p=resp
				self.psr.gwb_geom_c=resc
				dt=(self.psr.bbat.minus(self.psr.gwb_epoch).mjd)/self.psr.gwb_width
				prefac=dt*np.exp(-dt**2/2)
				phaseW+=self.psr.f0*prefac*(self.psr.gwb_amp[0]*resp+self.psr.gwb_amp[1]*resc)
		#
		phase1=phase3+phase4+phaseJ+phaseW+phase2state
		#print(nf0*ntpd[0]*86400.0, phase2[0],phase3[0],phase1[0]+phase2[0])
		self.phase=phase0.add(phase1)
	#
	def eccRes(self,prev_p,prev_e,prev_a,prev_epoch,prev_theta):
		msun=self.time.cons['GMS']/gg*aultsc*te.sl**2*1e4*te.iftek
		mpc_to_m=kpc2m*1e3
		ra_p=self.psr.raj
		dec_p=self.psr.decj
		ra_g=self.psr.gwecc_ra
		dec_g=self.psr.gwecc_dec
		m1=self.psr.gwecc_m1*msun
		m2=self.psr.gwecc_m2*msun
		inc=self.psr.gwecc_inc
		theta_n=self.psr.gwecc_theta_nodes
		phi=self.psr.gwecc_nodes_orientation
		z=self.psr.gwecc_redshift
		dd=self.psr.gwess_distance*mpc_to_m
		t_0=self.psr.gwecc_epoch.mjd
		rs_m1=2*gg*m1/te.sl**2
		time=(self.time.tcb.minus(self.residual).minus(prev_epoch).mjd)/(1+z)
		tint=86400*time/np.abs(100*np.floor(time))
		nt=np.int64(np.abs(100*np.floor(time)))
		if nt>1e6:
			tint=86400*time/1e6
			nt=1e6
		theta=prev_theta*1
		ec=prev_e*1
		ax=prev_a*1
		ih=0
		coalesceflag=0
		while ih<nt and coalesceflag==0:
			k1=tint*(-64/5)*(gg/ax)**3*m1*m2*(m1+m2)*te.sl**(-5)*(1-ec**2)**(-3.5)*(1+73*ec**2/24+37*ec**4/96)			
			l1=tint*(-304/15)*gg**3*m1*m2*(m1+m2)*te.sl**(-5)*ax**(-4)*(1-ec**2)**(-2.5)*ec*(1+121*ec**2/304)
			if ax+k1/2.<1.5*rs_m1: coalesceflag=1
			k2=tint*(-64./5.)*(gg/(ax+k1/2))**3*m1*m2*(m1+m2)*te.sl**(-5)*(1-(ec+l1/2)**2)**(-3.5)*(1+73*(ec+l1/2)**2/24+37*(ec+l1/2)**4/96)
			l2=tint*(-304/15)*gg**3*m1*m2*(m1+m2)*te.sl**(-5)*(ax+k1/2)**(-4)*(1-(ec+l1/2)**2)**(-2.5)*(ec+l1/2)*(1+121*(ec+l1/2)**2/304)
			if ax+k2/2.<1.5*rs_m1: coalesceflag=1
			k3=tint*(-64./5.)*(gg/(ax+k2/2))**3*m1*m2*(m1+m2)*te.sl**(-5)*(1-(ec+l2/2)**2)**(-3.5)*(1+73*(ec+l2/2)**2/24+37*(ec+l2/2)**4/96)		
			l3=tint*(-304/15)*gg**3*m1*m2*(m1+m2)*te.sl**(-5)*(ax+k2/2)**(-4)*(1-(ec+l2/2)**2)**(-2.5)*(ec+l2/2)*(1+121*(ec+l2/2)**2/304)
			if ax+k3/2.<1.5*rs_m1: coalesceflag=1
			k4=tint*(-64./5.)*(gg/(ax+k3/2))**3*m1*m2*(m1+m2)*te.sl**(-5)*(1-(ec+l3/2)**2)**(-3.5)*(1+73*(ec+l3/2)**2/24+37*(ec+l3/2)**4/96)
			l4=tint*(-304/15)*gg**3*m1*m2*(m1+m2)*te.sl**(-5)*(ax+k3/2)**(-4)*(1-(ec+l3/2)**2)**(-2.5)*(ec+l3/2)*(1+121*(ec+l3/2)**2/304)
			ax+=(k1+2.*k2+2.*k3+k4)/6
			ec+=(l1+2.*l2+2.*l3+l4)/6
			theta+=tint*2*np.pi/prev_p
			if ax<1.5*rs_m1: coalesceflag=1
			prev_p=1/(np.sqrt(gg*(m1+m2)*ax**(-3))/(2*np.pi))
			ih+=1
		if coalesceflag==1:
			pert=0
		else:
			prev_e=ec
			prev_epoch=self.time.tcb.minus(self.residual)
			prev_theta=theta
			prev_a=ax
			aag=4*gg**2*m1*m2/(te.sl**4*dd*(1-ec**2)*ax)
			frontterm=-aag*(1-ec**2)**(3/2)*prev_p*np.sin(theta)/(32*np.pi*(1+ec*np.cos(theta)))
			rplus=frontterm*(ec*np.cos(2*(inc-phi))+ec*np.cos(2*(inc+phi))-16*np.cos(inc)*np.sin(2*phi)*np.sin(theta-2*theta_n)+8*ec*np.cos(inc)*np.sin(2*phi)*np.sin(2*theta_n)+2*np.cos(2*phi)*(-ec+(3+np.cos(2*inc))*(ec+2*np.cos(theta))*np.cos(2*theta_n)+2*(3+np.cos(2*inc))*np.sin(theta)*np.sin(2*theta_n)))
			rcross=frontterm*(ec*np.cos(2*(inc-phi))+ec*np.cos(2*(inc+phi))+8*np.cos(inc)*np.cos(2*phi)*(-2*np.sin(theta-2*theta_n)+ec*np.sin(2*theta_n))-2*np.sin(2*phi)*(-ec+(3+np.cos(2*inc))*(ec+2*np.cos(theta))*np.cos(2*theta_n)+2*(3+np.cos(2*inc))*np.sin(theta)*np.sin(2*theta_n)))
			dalpha=ra_g-ra_p
			gplus=(-(1+3*np.cos(2*dec_p))*np.cos(dec_g)**2+np.cos(2*dalpha)*(-3+np.cos(2*dec_g))*np.sin(dec_p)**2+2*np.cos(dalpha)*np.sin(2*dec_g)*np.sin(2*dec_p))/(8*(-1+np.cos(dec_p)*np.sin(dec_g)+np.cos(dalpha)*np.cos(dec_g)*np.sin(dec_p)))
			gcross=((-np.cos(dec_g)*np.cos(dec_p)+np.cos(dalpha)*np.sin(dec_g)*np.sin(dec_p))*np.sin(dalpha)*np.sin(dec_p))/(-1+np.cos(dec_p)*np.sin(dec_g)+np.cos(dalpha)*np.cos(dec_g)*np.sin(dec_p))
			pert=1e9*(rplus*gplus+rcross*gcross)
		return pert
	#
	def compute_binary(self):
		if self.psr.binary=='BT': return self.BTmodel()
		if self.psr.binary=='BTJ': return self.BTJmodel()
		if self.psr.binary=='BTX': return self.BTXmodel()
		if self.psr.binary=='ELL1': return self.ELL1model()
		if self.psr.binary=='ELL1H': return self.ELL1Hmodel()
		if self.psr.binary=='ELL1k': return self.ELL1kmodel()
		if self.psr.binary=='DD': return self.DDmodel()
		if self.psr.binary=='DDH': return self.DDHmodel()
		if self.psr.binary=='DDK': return self.DDKmodel()
		if self.psr.binary=='DDS': return self.DDSmodel()
		if self.psr.binary=='DDGR': return self.DDGRmodel()
		if self.psr.binary=='MSS': return self.MSSmodel()
		if self.psr.binary=='T2': return self.T2model()
		if self.psr.binary=='T2_PTA': return self.T2_PTAmodel()
	# 
	def compute_binary_der(self):
		if self.psr.binary=='BT': return self.BTmodel(der=True)
		if self.psr.binary=='BTJ': return self.BTJmodel(der=True)
		if self.psr.binary=='BTX': return self.BTXmodel(der=True)
		if self.psr.binary=='ELL1': return self.ELL1model(der=True)
		if self.psr.binary=='ELL1H': return self.ELL1Hmodel(der=True)
		if self.psr.binary=='ELL1k': return self.ELL1kmodel(der=True)
		if self.psr.binary=='DD': return self.DDmodel(der=True)
		if self.psr.binary=='DDH': return self.DDHmodel(der=True)
		if self.psr.binary=='DDK': return self.DDKmodel(der=True)
		if self.psr.binary=='DDS': return self.DDSmodel(der=True)
		if self.psr.binary=='DDGR': return self.DDGRmodel(der=True)
		if self.psr.binary=='MSS': return self.MSSmodel(der=True)
		if self.psr.binary=='T2': return self.T2model(der=True)
		if self.psr.binary=='T2_PTA': return self.T2_PTAmodel(der=True)
	# 
	def BTmodel(self,der=False):
		tt0=self.bbat.minus(self.psr.t0).mjd*86400
		pb=self.psr.pb*86400
		edot=0.0
		ecc=self.psr.ecc+edot*tt0
		pbdot=self.psr.pbdot
		xpbdot=0.0
		xdot=self.psr.a1dot
		asini=self.psr.a1+xdot*tt0
		omdot=self.psr.omdot
		omega=(self.psr.om+omdot*tt0/(86400*365.25))/180*np.pi
		gamma=self.psr.gamma
		orbits=tt0/pb-0.5*(pbdot+xpbdot)*(tt0/pb)**2
		self.orbits=orbits
		phase=2*np.pi*(orbits%1)
		ep=phase*1.0
		dep=1
		while np.min(np.abs(dep))>1e-12:
			dep=(phase-(ep-ecc*np.sin(ep)))/(1-ecc*np.cos(ep))
			ep+=dep
		som=np.sin(omega)
		com=np.cos(omega)
		alpha=asini*np.sin(omega)
		tt=np.sqrt(1-ecc**2)
		beta=asini*np.cos(omega)*tt
		sbe,cbe=np.sin(ep),np.cos(ep)
		depdorbit=1/(1-ecc*cbe)
		q=alpha*(cbe-ecc)+(beta+gamma)*sbe
		vr_c=(-alpha*sbe+beta*cbe)*depdorbit*(2*np.pi/pb)
		torb=-q+q*vr_c
		if not der:
			return torb
		else:
			dtdpb=-(vr_c+gamma*cbe*depdorbit*2*np.pi)*(tt0/pb-(pbdot+xpbdot)*(tt0/pb)**2)*86400
			dtda1=(cbe-ecc)*som+sbe*com*tt
			dtdecc=-alpha*(1+sbe**2*depdorbit)+(beta+gamma)*cbe*sbe*depdorbit-beta*sbe*ecc/tt**2
			dtdom=asini*(com*(cbe-ecc)-som*tt*sbe)/180*np.pi
			dtdt0=-vr_c*(1-(pbdot+xpbdot)*tt0/pb)*86400
			dtdpbdot=-(vr_c+gamma*cbe*depdorbit*2*np.pi/pb)*(tt0/pb)**2
			dtda1dot=dtda1*tt0
			dtdomdot=dtdom*tt0/(86400*365.25)
			dtdgamma=sbe
			return np.array([dtdt0,dtdpb,dtdom,dtdecc,dtda1,dtdpbdot,np.zeros_like(dtdt0),dtda1dot,dtdomdot,dtdgamma])
	#
	def BTJmodel(self,der=False):
		tt0=self.bbat.minus(self.psr.t0).mjd*86400
		pb=self.psr.pb*86400
		edot=0.0
		ecc=self.psr.ecc+edot*tt0
		pbdot=self.psr.pbdot
		xpbdot=0.0
		xdot=self.psr.a1dot
		asini=self.psr.a1+xdot*tt0
		omdot=self.psr.omdot
		omega=(self.psr.om+omdot*tt0/(86400*365.25))/180*np.pi
		gamma=self.psr.gamma
		torb=np.zeros(self.time.size)
		asini=np.ones(self.time.size)*asini
		ecc=np.ones(self.time.size)*ecc
		omega=np.ones(self.time.size)*omega
		pb=np.ones(self.time.size)*pb
		for i in np.arange(self.psr.bpjep.size):
			jj=(self.bbat.mjd>self.psr.bpjep[i])
			torb[jj]-=self.psr.bpjph[i]/self.psr.f0
			asini[jj]+=self.psr.bpja1[i]
			ecc[jj]+=self.psr.bpjec[i]
			omega[jj]+=self.psr.bpjom[i]
			pb[jj]+=self.psr.bpjpb[i]
		orbits=tt0/pb-0.5*(pbdot+xpbdot)*(tt0/pb)**2
		self.orbits=orbits
		phase=2*np.pi*(orbits%1)
		ep=phase*1.0
		dep=1
		while np.min(np.abs(dep))>1e-12:
			dep=(phase-(ep-ecc*np.sin(ep)))/(1-ecc*np.cos(ep))
			ep+=dep
		som=np.sin(omega)
		com=np.cos(omega)
		alpha=asini*np.sin(omega)
		tt=np.sqrt(1-ecc**2)
		beta=asini*com*tt
		sbe,cbe=np.sin(ep),np.cos(ep)
		depdorbit=1/(1-ecc*cbe)
		q=alpha*(cbe-ecc)+(beta+gamma)*sbe
		vr_c=(-alpha*sbe+beta*cbe)*depdorbit*(2*np.pi/pb)
		torb=-q+q*vr_c+torb
		if not der:
			return torb
		else:
			dtdpb=-(vr_c+gamma*cbe*depdorbit*2*np.pi)*(tt0/pb-(pbdot+xpbdot)*(tt0/pb)**2)*86400
			dtda1=(cbe-ecc)*som+sbe*com*tt
			dtdecc=-alpha*(1+sbe**2*depdorbit)+(beta+gamma)*cbe*sbe*depdorbit-beta*sbe*ecc/tt**2
			dtdom=asini*(com*(cbe-ecc)-som*tt*sbe)/180*np.pi
			dtdt0=-vr_c*(1-(pbdot+xpbdot)*tt0/pb)*86400
			dtdpbdot=-(vr_c+gamma*cbe*depdorbit*2*np.pi/pb)*(tt0/pb)**2
			dtda1dot=dtda1*tt0
			dtdomdot=dtdom*tt0/(86400*365.25)
			dtdgamma=sbe
			dtdbpjph=np.zeros([self.psr.bpjep.size,self.time.size])
			dtdbpja1=np.zeros([self.psr.bpjep.size,self.time.size])
			dtdbpjec=np.zeros([self.psr.bpjep.size,self.time.size])
			dtdbpjom=np.zeros([self.psr.bpjep.size,self.time.size])
			dtdbpjpb=np.zeros([self.psr.bpjep.size,self.time.size])
			for i in np.arange(self.psr.bpjep.size):
				jj=(self.bbat.mjd>self.psr.bpjep[i])
				dtdbpjph[i,jj]=self.psr.f0
				dtdbpja1[i,jj]=dtda1[jj]
				dtdbpjec[i,jj]=dtdecc[jj]
				dtdbpjom[i,jj]=dtdom[jj]
				dtdbpjpb[i,jj]=dtdpb[jj]
			return np.array([dtdt0,dtdpb,dtdom,dtdecc,dtda1,np.zeros_like(dtdt0),dtdbpjph,dtdbpja1,dtdbpjec,dtdbpjom,dtdbpjpb,np.zeros_like(dtdt0),dtdpbdot,dtda1dot,dtdomdot,dtdgamma])
	#
	def BTXmodel(self,der=False):
		tt0=self.bbat.minus(self.psr.t0).mjd*86400
		edot=0.0
		ecc=self.psr.ecc+edot*tt0
		xdot=self.psr.a1dot
		asini=self.psr.a1+xdot*tt0
		omdot=self.psr.omdot
		omega=(self.psr.om+omdot*tt0/(86400*365.25))/180*np.pi
		gamma=self.psr.gamma
		orbits=tt0*(self.psr.fb0+1/2*tt0*(self.psr.fb1+1/3*tt0*(self.psr.fb2+1/4*tt0*self.psr.fb3)))
		self.orbits=orbits
		phase=2*np.pi*(orbits%1)
		ep=phase*1.0
		dep=1
		while np.min(np.abs(dep))>1e-12:
			dep=(phase-(ep-ecc*np.sin(ep)))/(1-ecc*np.cos(ep))
			ep+=dep
		som=np.sin(omega)
		com=np.cos(omega)
		alpha=asini*np.sin(omega)
		tt=np.sqrt(1-ecc**2)
		beta=asini*com*tt
		sbe,cbe=np.sin(ep),np.cos(ep)
		depdorbit=1/(1-ecc*cbe)
		q=alpha*(cbe-ecc)+(beta+gamma)*sbe
		vr_c=(-alpha*sbe+beta*cbe)*depdorbit*(2*np.pi*self.psr.fb0)
		torb=-q+q*vr_c
		if not der:
			return torb
		else:
			tmp=vr_c+gamma*cbe*depdorbit*2*np.pi
			dtdfb0=tmp*tt0
			dtdfb1=tmp*tt0**2/2
			dtdfb2=tmp*tt0**3/6
			dtdfb3=tmp*tt0**4/24
			dtda1=(cbe-ecc)*som+sbe*com*tt
			dtdecc=-alpha*(1+sbe**2*depdorbit)+(beta+gamma)*cbe*sbe*depdorbit-beta*sbe*ecc/tt**2
			dtdom=asini*(com*(cbe-ecc)-som*tt*sbe)/180*np.pi
			dtdt0=-vr_c/self.psr.fb0*(1+tt0*(self.psr.fb1+1/2*tt0*(self.psr.fb2+1/3*tt0*self.psr.fb3)))*86400
			dtda1dot=dtda1*tt0
			dtdomdot=dtdom*tt0/(86400*365.25)
			dtdgamma=sbe
			return np.array([dtdt0,dtdfb0,dtdom,dtdecc,dtda1,np.zeros_like(dtdt0),dtdfb1,dtdfb2,dtdfb3,np.zeros_like(dtdt0),dtda1dot,dtdomdot,dtdgamma])
	# 
	def ELL1model(self,der=False):
		sunmass=self.time.cons['GMS']*aultsc/te.sl*1e4*te.iftek
		a0,b0=0.0,0.0
		pbdot=self.psr.pbdot
		si=self.psr.sini
		t0asc=self.psr.tasc
		tt0=self.bbat.minus(t0asc).mjd*86400
		if si>1.0: 
			si=1.0
			self.psr.sini=1.0
		xdot=self.psr.a1dot
		x0=self.psr.a1+xdot*tt0
		m2=self.psr.m2*sunmass
		xpbdot=0.0
		eps1dot=self.psr.eps1dot
		eps2dot=self.psr.eps2dot
		e1=self.psr.eps1+eps1dot*tt0
		e2=self.psr.eps2+eps2dot*tt0
		if self.psr.fb0:
			dorbits_dt=self.psr.fb0+1/2*tt0*(self.psr.fb1+1/3*tt0*(self.psr.fb2+1/4*tt0*self.psr.fb3))
			orbits=tt0*dorbits_dt
		else:
			pb=self.psr.pb*86400
			orbits=tt0/pb-0.5*(pbdot+xpbdot)*(tt0/pb)**2
			dorbits_dt=1/pb-(pbdot+xpbdot)*tt0/pb**2
		if 'orbifuncT' in self.psr.paras:
			if self.psr.orbifunc==1:
				t1=0.0
				speriod=self.psr.orbifuncT.mjd[1]-self.psr.obrifuncT.mjd[0]
				for k in np.arange(len(self.psr.orbifuncV)):
					dt=self.bbat.mjd-self.psr.orbifuncT.mjd[k]
					wi=1
					if dt==0: t1+=self.psr.orbifuncV[k]
					else: 
						tt=np.pi/speriod*dt
						t1+=self.psr.orbifuncV[k]*np.sin(tt)/tt
				orbits+=t1
			elif self.psr.orbifunc==2:
				orbits+=np.interp(self.time.tcb.mjd,self.psr.orbifuncT.mjd,self.orbifuncV)
			elif self.psr.orbifunc==0 and self.psr.orbifuncT:
				order=np.argsort(self.psr.orbifuncT.mjd)
				orbifuncT=self.psr.orbifuncT.mjd[order]
				orbifuncV=self.psr.orbifuncV[order]
				for k in np.arange(len(orbits)):
					orbits[k]+=orbifuncV[np.argmax(self.time.tcb.mjd[k]>orbifuncT)]
		self.orbits=orbits
		phase=2*np.pi*(orbits%1)
		dre=x0*(np.sin(phase)-0.5*(e1*np.cos(2*phase)-e2*np.sin(2*phase)))
		brace=1-si*np.sin(phase)
		dlogbr=np.log(brace)
		ds=-2*m2*dlogbr
		da=a0*np.sin(phase)+b0*np.cos(phase) # abbretion is zero in ell1 model
		dphase_dt=2*np.pi*dorbits_dt
		ddre_dphase,d2dre_dphase2=x0*np.cos(phase),-x0*np.sin(phase)
		ddre_dt,d2dre_dt2=ddre_dphase*dphase_dt,d2dre_dphase2*(dphase_dt)**2		
		d2bar=dre*(1-ddre_dt+ddre_dt**2+0.5*dre*d2dre_dt2)+ds+da
		torb=-d2bar
		if not der:
			return torb
		else:
			tmp=x0*np.cos(phase)
			dtda1=np.sin(phase)
			dtdeps1=-0.5*x0*np.cos(2*phase)
			dtdeps2=tmp*dtda1
			dtdtasc=-tmp*dphase_dt*86400
			dtdeps1dot=dtdeps1*tt0
			dtdeps2dot=dtdeps2*tt0
			dtda1dot=dtda1*tt0
			dtdsini=2*m2*dtda1/brace
			dtdm2=-2*dlogbr*sunmass
			if self.psr.fb0:
				dtdfb0=tmp*2*np.pi*tt0
				dtdfb1=dtdfb0*tt0/2
				dtdfb2=dtdfb1*tt0/3
				dtdfb3=dtdfb2*tt0/4
				return np.array([dtdtasc,dtdfb0,dtdeps1,dtdeps2,dtda1,np.zeros_like(dtdtasc),dtdfb1,dtdfb2,dtdfb3,np.zeros_like(dtdtasc),dtdsini,dtda1dot,dtdm2,dtdeps1dot,dtdeps2dot, np.zeros_like(dtdtasc),np.zeros_like(dtdtasc),np.zeros_like(dtdtasc),np.zeros_like(dtdtasc)])
			else:
				dtdpb=-tmp*dphase_dt*tt0/pb*86400
				dtdpbdot=-0.5*tmp*2*np.pi*(tt0/pb)**2
				return np.array([dtdtasc,np.zeros_like(dtdtasc),dtdeps1,dtdeps2,dtda1,dtdpb,np.zeros_like(dtdtasc),np.zeros_like(dtdtasc),np.zeros_like(dtdtasc),dtdpbdot,dtdsini, dtda1dot,dtdm2,dtdeps1dot,dtdeps2dot,np.zeros_like(dtdtasc),np.zeros_like(dtdtasc),np.zeros_like(dtdtasc),np.zeros_like(dtdtasc)])
	#
	def ELL1Hmodel(self,der=False):
		sunmass=self.time.cons['GMS']*aultsc/te.sl*1e4*te.iftek
		a0,b0=0.0,0.0
		pbdot=self.psr.pbdot
		si=self.psr.sini
		t0asc=self.psr.tasc
		tt0=self.bbat.minus(t0asc).mjd*86400
		if si>1.0: 
			si=1.0
			self.psr.sini=1.0
		xdot=self.psr.a1dot
		x0=self.psr.a1+xdot*tt0
		m2=self.psr.m2*sunmass
		xpbdot=0.0
		eps1dot=self.psr.eps1dot
		eps2dot=self.psr.eps2dot
		e1=self.psr.eps1+eps1dot*tt0
		e2=self.psr.eps2+eps2dot*tt0
		pb=self.psr.pb*86400
		h3=self.psr.h3
		nharm=4
		if 'h4' in self.psr.paras:
			h4=self.psr.h4
			mode=2
			if 'nharm' in self.psr.paras:
				nharm=self.nharm
			if self.psr.stig: print("Warning: Both H4 and STIG in par file, then ignoring STIG")
			m2=h3**4/h4**3
		else:
			if 'stig' in self.psr.paras:
				stig=self.psr.stig
				mode=1
				m2=h3/stig**3
			else:
				mode,h4,nharm,stig=0,0,3,0
		if si>1.0: si=1.0
		orbits=tt0/pb-0.5*(pbdot+xpbdot)*(tt0/pb)**2
		dorbits_dt=1/pb-(pbdot+xpbdot)*tt0/pb**2
		self.orbits=orbits
		phase=2*np.pi*(orbits%1)
		trueanom=phase*1.0
		if mode==0: ds=-4/3*h3*np.sin(trueanom)
		elif mode==1:
			fs=1.0+stig**2-2*stig*np.sin(trueanom)
			lgf=np.log(fs)
			lsc=lgf+2.0*stig*np.sin(trueanom)-stig**2*np.cos(2*trueanom)
			ds=-2*m2*lsc
		else: ds=calcdh(trueanom,h3,h4,nharm,0)
		dre=x0*(np.sin(phase)-0.5*(e1*np.cos(2*phase)-e2*np.sin(2*phase)))
		da=a0*np.sin(phase)+b0*np.cos(phase) # abbretion is zero in ell1 model
		dphase_dt=2*np.pi*dorbits_dt
		ddre_dphase,d2dre_dphase2=x0*np.cos(phase),-x0*np.sin(phase)
		ddre_dt,d2dre_dt2=ddre_dphase*dphase_dt,d2dre_dphase2*(dphase_dt)**2		
		d2bar=dre*(1-ddre_dt+ddre_dt**2+0.5*dre*d2dre_dt2)+ds+da
		torb=-d2bar
		if not der:
			return torb
		else:
			tmp=x0*np.cos(phase)
			dtdpb=-tmp*dphase_dt*tt0/pb*86400
			dtdpbdot=-0.5*tmp*2*np.pi*(tt0/pb)**2
			dtda1=np.sin(phase)
			dtdeps1=-0.5*x0*np.cos(2*phase)
			dtdeps2=tmp*dtda1
			dtdtasc=-tmp*dphase_dt*86400
			dtdeps1dot=dtdeps1*tt0
			dtdeps2dot=dtdeps2*tt0
			dtda1dot=dtda1*tt0
			if mode==0:
				dtdh4,dtdstig=np.zeros_like(dtdtasc),np.zeros_like(dtdtasc)
				dtdh3=-4/3*np.sin(trueanom)
			elif mode==1:
				dtdh4=np.zeros_like(dtdtasc)
				dtdh3=-2*lsc/stig**3
				dtdstig=-2*m2*(1/fs*(2*stig-2*np.sin(trueanom))+2*np.sin(trueanom)-2*stig*np.cos(trueanom))+6*lsc*h3/stig**4
			else:
				dtdstig=np.zeros_like(dtdtasc)
				dtdh3,dtdh4=calcdh(trueanom,h3,h4,nharm,1)
			return np.array([dtdtasc,dtdpb,dtdeps1,dtdeps2,dtda1,dtdh3,dtdpbdot,np.zeros_like(dtdtasc),np.zeros_like(dtdtasc),dtda1dot,np.zeros_like(dtdtasc),dtdeps1dot, dtdeps2dot,dtdh4,np.zeros_like(dtdtasc),dtdstig,np.zeros_like(dtdtasc)])
	#
	def ELL1kmodel(self,der=False):
		sunmass=self.time.cons['GMS']*aultsc/te.sl*1e4*te.iftek
		a0,b0=0.0,0.0
		pbdot=self.psr.pbdot
		omdot=self.psr.omdot*np.pi/(180*365.25*86400)
		pb=self.psr.pb*86400
		si=self.psr.sini
		t0asc=self.psr.tasc
		tt0=self.bbat.minus(t0asc).mjd*86400
		if si>1.0: 
			si=1.0
			self.psr.sini=1.0
		xdot=self.psr.a1dot
		x0=self.psr.a1+xdot*tt0
		m2=self.psr.m2*sunmass
		xpbdot=0.0
		eps1=self.psr.eps1
		eps2=self.psr.eps2
		dom=omdot*tt0
		e1=eps1*np.cos(dom)+eps2*np.sin(dom)
		e2=eps2*np.cos(dom)-eps1*np.sin(dom)
		orbits=tt0/pb
		orbits-=0.5*(pbdot+xpbdot)*(tt0/pb)**2
		dorbits_dt=1/pb-(pbdot+xpbdot)*tt0/pb**2
		self.orbits=orbits
		phase=2*np.pi*(orbits%1)
		dre=x0*(np.sin(phase)-0.5*(e1*np.cos(2*phase)-e2*np.sin(2*phase)))
		brace=1-si*np.sin(phase)
		dlogbr=np.log(brace)
		ds=-2*m2*dlogbr
		da=a0*np.sin(phase)+b0*np.cos(phase) # abbretion is zero in ell1 model
		dphase_dt=2*np.pi*dorbits_dt-omdot
		ddre_dphase,d2dre_dphase2=x0*np.cos(phase),-x0*np.sin(phase)
		ddre_dt,d2dre_dt2=ddre_dphase*dphase_dt,d2dre_dphase2*(dphase_dt)**2		
		d2bar=dre*(1-ddre_dt+ddre_dt**2+0.5*dre*d2dre_dt2)+ds+da
		torb=-d2bar
		if not der:
			return torb
		else:
			tmp=x0*np.cos(phase)
			dtda1=np.sin(phase)
			dtdeps1=-0.5*x0*np.cos(2*phase)
			dtdeps2=tmp*dtda1
			dtdtasc=-tmp*dphase_dt*86400
			dtda1dot=dtda1*tt0
			dtdsini=2*m2*dtda1/brace
			dtdm2=-2*dlogbr*sunmass
			dtdpb=-tmp*dphase_dt*tt0/pb*86400
			dtdpbdot=-0.5*tmp*2*np.pi*(tt0/pb)**2
			dtdomdot=(-0.5*x0*(np.cos(2*phase)*e2+np.sin(2*phase)*e1)*tt0+dre*(ddre_dphase-2*ddre_dt*ddre_dphase-dre*d2dre_dphase2*dphase_dt))*np.pi/(180*365.25*86400)
			return np.array([dtdtasc,dtdpb,dtdeps1,dtdeps2,dtda1,dtdpbdot,np.zeros_like(dtdtasc),dtdsini,dtda1dot,dtdm2,dtdomdot])
	#
	def DDmodel(self,der=False):
		sunmass=self.time.cons['GMS']*aultsc/te.sl*1e4*te.iftek
		dr,dth=0.0,0.0
		si=self.psr.sini
		if si>1.0: 
			si=1.0
			self.psr.sini=1.0
		m2=self.psr.m2*sunmass
		pb=self.psr.pb*86400
		an=2*np.pi/pb
		k=self.psr.omdot*np.pi/(180*365.25*86400)/an
		tt0=self.bbat.minus(self.psr.t0).mjd*86400
		gamma=self.psr.gamma
		a0,b0=0,0
		omz=self.psr.om
		xdot=self.psr.a1dot
		pbdot=self.psr.pbdot
		edot=self.psr.edot
		xpbdot=self.psr.xpbdot
		x=self.psr.a1+xdot*tt0
		ecc=self.psr.ecc+edot*tt0
		er,eth=ecc*(1+dr),ecc*(1+dth)
		orbits=tt0/pb-0.5*(pbdot+xpbdot)*(tt0/pb)**2
		self.orbits=orbits
		phase=2*np.pi*(orbits%1)
		u=phase*1.0
		du=1
		while np.min(np.abs(du))>1e-12:
			du=(phase-(u-ecc*np.sin(u)))/(1-ecc*np.cos(u))
			u+=du
		su,cu=np.sin(u),np.cos(u)
		onemecu=1-ecc*cu
		cae=(cu-ecc)/onemecu
		sae=np.sqrt(1-ecc**2)*su/onemecu
		ae=np.arctan2(sae,cae)
		ae[ae<0.0]+=2*np.pi
		ae=2*np.pi*orbits+ae-phase
		omega=omz*np.pi/180+k*ae
		sw,cw=np.sin(omega),np.cos(omega)
		alpha=x*sw
		beta=x*np.sqrt(1-eth**2)*cw
		bg=beta+gamma
		dre=alpha*(cu-er)+bg*su
		drep=-alpha*su+bg*cu
		drepp=-alpha*cu-bg*su
		anhat=an/onemecu
		sqr1me2=np.sqrt(1-ecc**2)
		cume=cu-ecc
		brace=onemecu-si*(sw*cume+sqr1me2*cw*su)
		dlogbr=np.log(brace)
		ds=-2*m2*dlogbr
		da=a0*(np.sin(omega+ae)+ecc*sw)+b0*(np.cos(omega+ae)+ecc*cw)
		d2bar=dre*(1-anhat*drep+anhat**2*(drep**2+0.5*dre*drepp-0.5*ecc*su*dre*drep/onemecu))+ds+da
		torb=-d2bar
		if not der:
			return torb
		else:
			dphase_dt=-2*np.pi*(1/pb-0.5*(pbdot+xpbdot)*tt0/pb**2)
			dphase_dpb=dphase_dt*tt0/pb
			tmp=(-x*sw*su+bg*cu)/onemecu
			dtdpb=tmp*dphase_dpb*86400
			dtda1=sw*(cu-er)+np.sqrt(1-eth**2)*cw*su
			dtda1dot=dtda1*tt0
			dtdecc=-alpha*(1+dr)-x*su*cw*ecc*(1+dth)**2/np.sqrt(1-eth**2)+tmp*su
			dtdedot=dtdecc*tt0
			dtdom=(x*cw*(cu-er)-x*sw*su*np.sqrt(1-eth**2))*np.pi/180
			dtdomdot=dtdom*ae/(365.25*86400)/an
			dtdt0=tmp*dphase_dt*86400
			dtdpbdot=-tmp*2*np.pi*0.5*(tt0/pb)**2
			dtdsini=2*m2*(sw*cume+sqr1me2*cw*su)/brace
			dtdgamma=su
			dtdm2=-2*dlogbr*sunmass
			return np.array([dtdt0,dtdpb,dtdecc,dtdom,dtda1,dtdpbdot,np.zeros_like(dtdt0),dtdsini,dtdomdot,dtda1dot,dtdm2,np.zeros_like(dtdt0),dtdedot,dtdgamma])
	#
	def DDHmodel(self,der=False):
		sunmass=self.time.cons['GMS']*aultsc/te.sl*1e4*te.iftek
		dr,dth=0.0,0.0
		h3=self.psr.h3
		stig=self.psr.stig
		si=2*stig/(1+stig**2)
		m2=h3/stig**3
		if si>1.0: 
			si=1.0
			self.psr.sini=1.0
		pb=self.psr.pb*86400
		an=2*np.pi/pb
		k=self.psr.omdot*np.pi/(180*365.25*86400)/an
		tt0=self.bbat.minus(self.psr.t0).mjd*86400
		gamma=self.psr.gamma
		a0,b0=0,0
		omz=self.psr.om
		xdot=self.psr.a1dot
		pbdot=self.psr.pbdot
		edot=self.psr.edot
		xpbdot=self.psr.xpbdot
		x=self.psr.a1+xdot*tt0
		ecc=self.psr.ecc+edot*tt0
		er,eth=ecc*(1+dr),ecc*(1+dth)
		orbits=tt0/pb-0.5*(pbdot+xpbdot)*(tt0/pb)**2
		self.orbits=orbits
		phase=2*np.pi*(orbits%1)
		u=phase*1.0
		du=1
		while np.min(np.abs(du))>1e-12:
			du=(phase-(u-ecc*np.sin(u)))/(1-ecc*np.cos(u))
			u+=du
		su,cu=np.sin(u),np.cos(u)
		onemecu=1-ecc*cu
		cae=(cu-ecc)/onemecu
		sae=np.sqrt(1-ecc**2)*su/onemecu
		ae=np.arctan2(sae,cae)
		ae[ae<0]+=2*np.pi
		ae=2*np.pi*orbits+ae-phase
		omega=omz*np.pi/180+k*ae
		sw,cw=np.sin(omega),np.cos(omega)
		alpha=x*sw
		beta=x*np.sqrt(1-eth**2)*cw
		bg=beta+gamma
		dre=alpha*(cu-er)+bg*su
		drep=-alpha*su+bg*cu
		drepp=-alpha*cu-bg*su
		anhat=an/onemecu
		sqr1me2=np.sqrt(1-ecc**2)
		cume=cu-ecc
		brace=onemecu-si*(sw*cume+sqr1me2*cw*su)
		dlogbr=np.log(brace)
		ds=-2*m2*dlogbr
		da=a0*(np.sin(omega+ae)+ecc*sw)+b0*(np.cos(omega+ae)+ecc*cw)
		d2bar=dre*(1-anhat*drep+anhat**2*(drep**2+0.5*dre*drepp-0.5*ecc*su*dre*drep/onemecu))+ds+da
		torb=-d2bar
		if not der:
			return torb
		else:
			dphase_dt=-2*np.pi*(1/pb-0.5*(pbdot+xpbdot)*tt0/pb**2)
			dphase_dpb=dphase_dt*tt0/pb
			tmp=(-x*sw*su+bg*cu)/onemecu
			dtdpb=tmp*dphase_dpb*86400
			dtda1=sw*(cu-er)+np.sqrt(1-eth**2)*cw*su
			dtda1dot=dtda1*tt0
			dtdecc=-alpha*(1+dr)-x*su*cw*ecc*(1+dth)**2/np.sqrt(1-eth**2)+tmp*su
			dtdedot=dtdecc*tt0
			dtdom=(x*cw*(cu-er)-x*sw*su*np.sqrt(1-eth**2))*np.pi/180
			dtdomdot=dtdom*ae/(365.25*86400)/an
			dtdt0=tmp*dphase_dt*86400
			dtdpbdot=-tmp*2*np.pi*0.5*(tt0/pb)**2
			dtdgamma=su
			dtdsini=2*m2*(sw*cume+sqr1me2*cw*su)/brace
			dtdh3=-2*dlogbr/stig**3  # tempo2 result??
			dtdstig=-3*dtdh3*h3/stig+2*dtdsini*(1-stig**2)/(1+stig**2)**2
			return np.array([dtdt0,dtdpb,dtdecc,dtdom,dtda1,dtdh3,dtdstig,dtdpbdot,np.zeros_like(dtdt0),dtdomdot,dtda1dot,np.zeros_like(dtdt0),dtdedot,dtdgamma])
	#
	def DDKmodel(self,der=False):
		sunmass=self.time.cons['GMS']*aultsc/te.sl*1e4*te.iftek
		dr,dth=0.0,0.0
		kom=self.psr.kom*np.pi/180
		sin_omega,cos_omega=np.sin(kom),np.cos(kom)
		kin=self.psr.kin*np.pi/180
		si,ci=np.sin(kin),np.cos(kin)
		pmra=self.psr.pmra*np.pi/(180.0*3600.0e3)/(365.25*86400.0)
		pmdec=self.psr.pmdec*np.pi/(180.0*3600.0e3)/(365.25*86400.0)
		dpara=self.psr.px*pxconv
		psr_location_equ=self.psr_location.copy()
		psr_location_equ.ecl2equ()
		sin_delta=psr_location_equ.z
		cos_delta=np.cos(np.arcsin(sin_delta))
		sin_alpha=psr_location_equ.y/cos_delta
		cos_alpha=psr_location_equ.x/cos_delta
		m2=self.psr.m2*sunmass
		pb=self.psr.pb*86400
		an=2*np.pi/pb
		k=self.psr.omdot*np.pi/(180*365.25*86400)/an
		tt0=self.bbat.minus(self.psr.t0).mjd*86400
		gamma=self.psr.gamma
		a0,b0=0,0
		omz=self.psr.om*np.pi/180.0
		xdot=self.psr.a1dot
		pbdot=self.psr.pbdot
		edot=self.psr.edot
		xpbdot=self.psr.xpbdot
		asi=self.psr.a1+xdot*tt0
		ecc=self.psr.ecc+edot*tt0
		er,eth=ecc*(1+dr),ecc*(1+dth)
		ki_dot=-pmra*sin_omega+pmdec*cos_omega
		ki=kin+ki_dot*tt0
		sini,cosi=np.sin(ki),np.cos(ki)
		tani=sini/cosi
		if 'kom' in self.paras:
			asi+=(asi*ki_dot/tani)*tt0
			omz+=(pmra*cos_omega+pmdec*sin_omega)/sini*tt0
		earth_pos=self.time.pos[2].copy()
		earth_pos.ecl2equ()
		delta_i0=earth_pos.dot(te.vector(-sin_alpha,cos_alpha,np.zeros_like(sin_alpha),center='bary',scale='si',coord='equ',unit=te.sl,type0='pos'))/aultsc
		delta_j0=earth_pos.dot(te.vector(-sin_delta*cos_alpha,-sin_delta*sin_alpha,cos_delta,center='bary',scale='si',coord='equ',unit=te.sl,type0='pos'))/aultsc
		xpr=delta_i0*sin_omega-delta_j0*cos_omega
		ypr=delta_i0*cos_omega+delta_j0*sin_omega
		if 'kom' in self.paras:
			asi+=asi/tani*dpara*xpr
			si+=ci*dpara*xpr
			omz-=1.0/si*dpara*ypr
		orbits=tt0/pb-0.5*(pbdot+xpbdot)*(tt0/pb)**2
		self.orbits=orbits
		phase=2*np.pi*(orbits%1)
		u=phase*1.0
		du=1
		while np.min(np.abs(du))>1e-12:
			du=(phase-(u-ecc*np.sin(u)))/(1-ecc*np.cos(u))
			u+=du
		su,cu=np.sin(u),np.cos(u)
		onemecu=1-ecc*cu
		cae=(cu-ecc)/onemecu
		sae=np.sqrt(1-ecc**2)*su/onemecu
		ae=np.arctan2(sae,cae)
		ae[ae<0.0]+=2*np.pi
		ae=2*np.pi*orbits+ae-phase
		omega=omz+k*ae
		sw,cw=np.sin(omega),np.cos(omega)
		alpha=asi*sw
		beta=asi*np.sqrt(1-eth**2)*cw
		bg=beta+gamma
		dre=alpha*(cu-er)+bg*su
		drep=-alpha*su+bg*cu
		drepp=-alpha*cu-bg*su
		anhat=an/onemecu
		sqr1me2=np.sqrt(1-ecc**2)
		cume=cu-ecc
		brace=onemecu-si*(sw*cume+sqr1me2*cw*su)
		dlogbr=np.log(brace)
		ds=-2*m2*dlogbr
		da=a0*(np.sin(omega+ae)+ecc*sw)+b0*(np.cos(omega+ae)+ecc*cw)
		d2bar=dre*(1-anhat*drep+anhat**2*(drep**2+0.5*dre*drepp-0.5*ecc*su*dre*drep/onemecu))+ds+da
		torb=-d2bar
		if not der:
			return torb
		else:
			dphase_dt=-2*np.pi*(1/pb-0.5*(pbdot+xpbdot)*tt0/pb**2)
			dphase_dpb=dphase_dt*tt0/pb
			tmp=(-x*sw*su+bg*cu)/onemecu
			dtdpb=tmp*dphase_dpb*86400
			dtda1=sw*(cu-er)+np.sqrt(1-eth**2)*cw*su
			dtda1dot=dtda1*tt0
			dtdecc=-alpha*(1+dr)-x*su*cw*ecc*(1+dth)**2/np.sqrt(1-eth**2)+tmp*su
			dtdedot=dtdecc*tt0
			dtdom=(x*cw*(cu-er)-x*sw*su*np.sqrt(1-eth**2))*np.pi/180
			dtdomdot=dtdom*ae/(365.25*86400)/an
			dtdt0=tmp*dphase_dt*86400
			dtdpbdot=-tmp*2*np.pi*0.5*(tt0/pb)**2
			dtdgamma=su
			dtdm2=-2*dlogbr*sunmass
			return np.array([dtdt0,dtdpb,dtdecc,dtdom,dtda1,np.zeros_like(dtdt0),np.zeros_like(dtdt0),dtdpbdot,np.zeros_like(dtdt0),dtdomdot,dtda1dot,dtdm2,np.zeros_like(dtdt0), dtdedot,dtdgamma])
	#
	def DDSmodel(self,der=False):
		sunmass=self.time.cons['GMS']*aultsc/te.sl*1e4*te.iftek
		dr,dth=0.0,0.0
		shapmax=self.psr.shapmax
		m2=self.psr.m2*sunmass
		pb=self.psr.pb*86400
		an=2*np.pi/pb
		k=self.psr.omdot*np.pi/(180*365.25*86400)/an
		tt0=self.bbat.minus(self.psr.t0).mjd*86400
		gamma=self.psr.gamma
		a0,b0=0,0
		omz=self.psr.om
		xdot=self.psr.a1dot
		pbdot=self.psr.pbdot
		edot=self.psr.edot
		xpbdot=self.psr.xpbdot
		x=self.psr.a1+xdot*tt0
		ecc=self.psr.ecc+edot*tt0
		er,eth=ecc*(1+dr),ecc*(1+dth)
		orbits=tt0/pb-0.5*(pbdot+xpbdot)*(tt0/pb)**2
		self.orbits=orbits
		phase=2*np.pi*(orbits%1)
		u=phase*1.0
		du=1
		while np.min(np.abs(du))>1e-12:
			du=(phase-(u-ecc*np.sin(u)))/(1-ecc*np.cos(u))
			u+=du
		su,cu=np.sin(u),np.cos(u)
		onemecu=1-ecc*cu
		cae=(cu-ecc)/onemecu
		sae=np.sqrt(1-ecc**2)*su/onemecu
		ae=np.arctan2(sae,cae)
		ae[ae<0.0]+=2*np.pi
		ae=2*np.pi*orbits+ae-phase
		omega=omz*np.pi/180+k*ae
		sw,cw=np.sin(omega),np.cos(omega)
		alpha=x*sw
		beta=x*np.sqrt(1-eth**2)*cw
		bg=beta+gamma
		dre=alpha*(cu-er)+bg*su
		drep=-alpha*su+bg*cu
		drepp=-alpha*cu-bg*su
		anhat=an/onemecu
		sqr1me2=np.sqrt(1-ecc**2)
		cume=cu-ecc
		sdds=1-np.exp(-shapmax)
		brace=onemecu-sdds*(sw*cume+sqr1me2*cw*su)
		dlogbr=np.log(brace)
		ds=-2*m2*dlogbr
		da=a0*(np.sin(omega+ae)+ecc*sw)+b0*(np.cos(omega+ae)+ecc*cw)
		d2bar=dre*(1-anhat*drep+anhat**2*(drep**2+0.5*dre*drepp-0.5*ecc*su*dre*drep/onemecu))+ds+da
		torb=-d2bar
		if not der:
			return torb
		else:
			dphase_dt=-2*np.pi*(1/pb-0.5*(pbdot+xpbdot)*tt0/pb**2)
			dphase_dpb=dphase_dt*tt0/pb
			tmp=(-x*sw*su+bg*cu)/onemecu
			dtdpb=tmp*dphase_dpb*86400
			dtda1=sw*(cu-er)+np.sqrt(1-eth**2)*cw*su
			dtda1dot=dtda1*tt0
			dtdecc=-alpha*(1+dr)-x*su*cw*ecc*(1+dth)**2/np.sqrt(1-eth**2)+tmp*su
			dtdedot=dtdecc*tt0
			dtdom=(x*cw*(cu-er)-x*sw*su*np.sqrt(1-eth**2))*np.pi/180
			dtdomdot=dtdom*ae/(365.25*86400)/an
			dtdt0=tmp*dphase_dt*86400
			dtdpbdot=-tmp*2*np.pi*0.5*(tt0/pb)**2
			dtdshapmax=2*m2*(sw*cume+sqr1me2*cw*su)/brace*(1-sdds)
			dtdgamma=su
			dtdm2=-2*dlogbr*sunmass
			return np.array([dtdt0,dtdpb,dtdecc,dtdom,dtda1,dtdpbdot,np.zeros_like(dtdt0),dtdshapmax,dtdomdot,dtda1dot,dtdm2,np.zeros_like(dtdt0),dtdedot,dtdgamma])
	#
	def DDGRmodel(self,der=False):
		sunmass=self.time.cons['GMS']*aultsc/te.sl*1e4*te.iftek
		tt0=self.bbat.minus(self.psr.t0).mjd*86400
		f0=self.psr.f0
		xomdot=0
		afac=0
		si=self.psr.sini
		m=self.psr.mtot*sunmass
		m2=self.psr.m2*sunmass
		if m==0:
			m1=1.4*sunmass
			m=m1+m2
			print("Strong Warning: The mass of the pulsar is not given in parfile, use 1.4 times of solarmass in calculation.")
		else:
			m1=m-m2
		pb=self.psr.pb*86400
		an=2*np.pi/pb
		omz=self.psr.om
		xdot=self.psr.a1dot
		pbdot=self.psr.pbdot
		edot=self.psr.edot
		xpbdot=self.psr.xpbdot
		x=self.psr.a1+xdot*tt0
		ecc=self.psr.ecc+edot*tt0
		arr0=(m/an**2)**(1/3)
		arr,arrold=arr0,0
		while np.abs((arr-arrold)/arr)>1e-10:
			arrold=arr
			arr=arr0*(1+(m1*m2/m**2-9.0)*0.5*m/arr)**(2/3)
		arr=arr0*(1+(m1*m2/m**2-9.0)*0.5*m/arr)**(2/3)
		ar=arr*m2/m
		si=x/ar
		k=3*m/(arr*(1-ecc**2))
		gamma=ecc*m2*(m1+2*m2)/(an*arr*m)
		pbdot=-(96*2*np.pi/5)*an**(5/3)*(1-ecc**2)**(-3.5)*(1+(73/24)*ecc**2+(37/96)*ecc**4)*m1*m2*m**(-1/3)
		dr=(3*m1**2+6*m1*m2+2*m2**2)/(arr*m)
		dth=(3.5*m1**2+6*m1*m2+2*m2**2)/(arr*m)
		er,eth=ecc*(1+dr),ecc*(1+dth)
		orbits=tt0/pb-0.5*(pbdot+xpbdot)*(tt0/pb)**2
		self.orbits=orbits
		phase=2*np.pi*(orbits%1)
		u=phase*1.0
		du=1
		while np.min(np.abs(du))>1e-14:
			du=(phase-(u-ecc*np.sin(u)))/(1-ecc*np.cos(u))
			u+=du
		su,cu=np.sin(u),np.cos(u)
		onemecu=1-ecc*cu
		cae=(cu-ecc)/onemecu
		sae=np.sqrt(1-ecc**2)*su/onemecu
		ae=np.arctan2(sae,cae)
		ae[ae<0.0]+=2*np.pi
		ae=2*np.pi*orbits+ae-phase
		omega=omz*np.pi/180+(k+xomdot/(an*np.pi/180*365.25*86400))*ae
		sw,cw=np.sin(omega),np.cos(omega)
		alpha=x*sw
		beta=x*np.sqrt(1-eth**2)*cw
		bg=beta+gamma
		dre=alpha*(cu-er)+bg*su
		drep=-alpha*su+bg*cu
		drepp=-alpha*cu-bg*su
		anhat=an/onemecu
		sqr1me2=np.sqrt(1-ecc**2)
		cume=cu-ecc
		brace=onemecu-si*(sw*cume+sqr1me2*cw*su)
		dlogbr=np.log(brace)
		ds=-2*m2*dlogbr
		a0aligned=an*ar/(2*np.pi*f0*si*sqr1me2)
		a0=afac*a0aligned
		b0=0.0
		da=a0*(np.sin(omega+ae)+ecc*sw)+b0*(np.cos(omega+ae)+ecc*cw)
		d2bar=dre*(1-anhat*drep+anhat**2*(drep**2+0.5*dre*drepp-0.5*ecc*su*dre*drep/onemecu))+ds+da
		torb=-d2bar
		if not der:
			return torb
		else:
			dphase_dt=-2*np.pi*(1/pb-0.5*(pbdot+xpbdot)*tt0/pb**2)
			dphase_dpb=dphase_dt*tt0/pb
			tmp=(-x*sw*su+bg*cu)/onemecu
			dtdpb=tmp*dphase_dpb*86400
			dtda1=sw*(cu-er)+np.sqrt(1-eth**2)*cw*su
			dtda1dot=dtda1*tt0
			dtdecc=-alpha*(1+dr)-x*su*cw*ecc*(1+dth)**2/np.sqrt(1-eth**2)+tmp*su
			dtdedot=dtdecc*tt0
			dtdom=(x*cw*(cu-er)-x*sw*su*np.sqrt(1-eth**2))*np.pi/180
			dtdt0=tmp*dphase_dt*86400
			dtdpbdot=-tmp*2*np.pi*0.5*(tt0/pb)**2
			dtdgamma=su
			dtdeth=-x*su*cw*eth/np.sqrt(1-eth**2)
			dtdsini=2*m2*(sw*cume+sqr1me2*cw*su)/brace
			darrdm=(m1*m2/m**2-9.0+m2/m-2*m1*m2/m**2)*0.5/(2.5*(arr/arr0)**1.5-1)
			darrdm2=(m1-m2)/m*0.5/(2.5*(arr/arr0)**1.5-1)
			ddrdm=((6*m1+6*m2)*arr*m-(3*m1**2+6*m1*m2+2*m2**2)*(arr+darrdm*m))/(arr*m)**2
			ddrdm2=(-2*m2*arr*m-(3*m1**2+6*m1*m2+2*m2**2)*darrdm2*m)/(arr*m)**2
			ddthdm=((7*m1+6*m2)*arr*m-(3.5*m1**2+6*m1*m2+2*m2**2)*(arr+darrdm*m))/(arr*m)**2
			ddthdm2=((-m1-2*m2)*arr*m-(3.5*m1**2+6*m1*m2+2*m2**2)*darrdm2*m)/(arr*m)**2
			dgammadm=ecc*(m2*arr*m-m2*(m1+2*m2)*(arr+darrdm*m))/an/(arr*m)**2
			dgammadm2=ecc*((m1+3*m2)*arr*m-m2*(m1+2*m2)*darrdm2*m)//an/(arr*m)**2
			dpbdotdm=-(96*2*np.pi/5)*an**(5/3)*(1-ecc**2)**(-3.5)*(1+(73/24)*ecc**2+(37/96)*ecc**4)*(m2/m**(1/3)-m1*m2*m**(-4/3)/3)
			dpbdotdm2=-(96*2*np.pi/5)*an**(5/3)*(1-ecc**2)**(-3.5)*(1+(73/24)*ecc**2+(37/96)*ecc**4)*(m1-m2)*m**(-1/3)
			dsinidm=x*(arr*m2-darrdm*m2*m)/(arr*m2)**2
			dsinidm2=x*m*(-darrdm2*m2-arr)/(arr*m2)**2
			dkdm=3*(arr-darrdm*m)/(arr**2*(1-ecc**2))
			dkdm2=3*m(-darrdm2)/(arr**2*(1-ecc**2))
			dtdmtot=(-alpha*ecc*ddrdm+dtdeth*ecc*ddthdm+dtdgamma*dgammadm+dtdpbdot*dpbdotdm+dsdsini*dsinidm+dtdom*ae*dkdm)*sunmass
			dtdm2=(-alpha*ecc*ddrdm2+dtdeth*ecc*ddthdm2+dtdgamma*dgammadm2+dtdpbdot*dpbdotdm2+dsdsini*dsinidm2+dtdom*ae*dkdm2)*sunmass
			return np.array([dtdt0,dtdpb,dtdecc,dtdom,dtda1,np.zeros_like(dtdt0),np.zeros_like(dtdt0),np.zeros_like(dtdt0),dtda1dot,dtdmtot,dtdm2,np.zeros_like(dtdt0),dtdedot])
	#
	def MSSmodel(self,der=False):
		sunmass=self.time.cons['GMS']*aultsc/te.sl*1e4*te.iftek
		tt0=self.bbat.minus(self.psr.t0).mjd*86400
		m2=self.psr.m2*sunmass
		pbdot=self.psr.pbdot
		xdot=self.psr.a1dot
		edot=self.psr.edot
		si=self.psr.sini
		omdot=self.psr.omdot
		gamma=self.psr.gamma
		shapmax=self.psr.shapmax
		x2dot=self.psr.a2dot
		e2dot=self.psr.e2dot*1e-20
		orbpx=self.psr.orbpx/3.086e21
		dth=self.psr.dtheta
		dr=self.psr.dr
		a0=self.psr.a0
		b0=self.psr.b0
		om2dot=self.psr.om2dot
		pb=self.psr.pb*86400
		ecc0=self.psr.ecc
		x0=self.psr.a1
		omega=self.psr.om
		an=2*np.pi/pb
		omega0=omega/180*np.pi
		k=omdot/an/(180/np.pi*365.25*86400)
		xi=xdot/an
		orbits=tt0/pb-0.5*pbdot*(tt0/pb)**2
		self.orbits=orbits
		phase=2*np.pi*(orbits%1)
		ecc=ecc0+edot*tt0+0.5*e2dot*tt0**2
		er,eth=ecc*(1+dr),ecc*(1+dth)
		u=phase*1.0
		du=1
		while np.min(np.abs(du))>1e-14:
			du=(phase-(u-ecc*np.sin(u)))/(1-ecc*np.cos(u))
			u+=du
		su,cu=np.sin(u),np.cos(u)
		onemecu=1-ecc*cu
		cae=(cu-ecc)/onemecu
		sae=np.sqrt(1-ecc**2)*su/onemecu
		ae=np.arctan2(sae,cae)
		ae[ae<0.0]+=2*np.pi
		ae=2*np.pi*orbits+ae-phase
		xii=1e-20*x2dot/an**2
		oii=1e-20*om2dot/an**2
		x=x0+xi*ae+0.5*xii*ae**2
		omega=omega0+k*ae+0.5*oii*ae**2
		sw,cw=np.sin(omega),np.cos(omega)
		alpha=x*sw
		beta=x*np.sqrt(1-eth**2)*cw
		bg=beta+gamma
		dre=alpha*(cu-er)+bg*su
		drep=-alpha*su+bg*cu
		drepp=-alpha*cu-bg*su
		anhat=an/onemecu
		sqr1me2=np.sqrt(1-ecc**2)
		cume=cu-ecc
		brace=onemecu-si*(sw*cume+sqr1me2*cw*su)
		dlogbr=np.log(brace)
		ds=-2*m2*dlogbr
		a0=self.psr.a0
		b0=self.psr.b0
		da=a0*(np.sin(omega+ae)+ecc*sw)+b0*(np.cos(omega+ae)+ecc*cw)
		d2bar=dre*(1-anhat*drep+anhat**2*(drep**2+0.5*dre*drepp-0.5*ecc*su*dre*drep/onemecu))+ds+da
		torb=-d2bar
		shapparam=-np.log(1-ecc*cu-(np.sin(omega)*(cu-ecc)+np.sqrt(1-ecc**2)*np.cos(omega)*su)*si)
		torb-=shapmax*shapparam
		if si==0: cpx=0
		else: cpx=1e2*te.sl*x**2/2*(1/si**2-0.5+0.5*ecc**2*(1+sw**2-3/si**2)-2*ecc*(1/si**2-sw**2)*cume+sqr1me2*np.sin(2*omega)*(ecc*su-0.5*np.sin(2*u))+0.5*(np.cos(2*omega)+ecc**2*(1./si**2+cw**2))*np.cos(2*u))
		torb-=cpx*orbpx
		if not der:
			return torb
		else:
			dphase_dt=-2*np.pi*(1/pb-0.5*(pbdot+xpbdot)*tt0/pb**2)
			dphase_dpb=dphase_dt*tt0/pb
			tmp=(-x*sw*su+bg*cu)/onemecu
			dtdpb=tmp*dphase_dpb*86400
			dtda1=sw*(cu-er)+np.sqrt(1-eth**2)*cw*su
			dtda1dot=dtda1*ae/an
			dtda2dot=dtda1*0.5e-20*(ae/an)**2
			dtdecc=-alpha*(1+dr)-x*su*cw*ecc*(1+dth)**2/np.sqrt(1-eth**2)+tmp*su
			dtdedot=dtdecc*tt0
			dtde2dot=dtdecc*0.5e-20*tt0**2
			dtdom=(x*cw*(cu-er)-x*sw*su*np.sqrt(1-eth**2))*np.pi/180
			dtdomdot=dtdom*180/np.pi*ae/an/(180/np.pi*365.25*86400)
			dtdom2dot=dtdom*180/np.pi*0.5e-20*(ae/an)**2
			dtdt0=tmp*dphase_dt*86400
			dtdpbdot=-tmp*2*np.pi*0.5*(tt0/pb)**2
			dtdshapmax=shapparam
			dtdgamma=su
			dtdm2=-2*dlogbr*sunmass
			dtdorbpx=cpx/3.086e21
			dtdsini=-2*orbpx*1e2*te.sl*x**2/2*(1+1.5*ecc**2-2*ecc*cume+0.5*ecc**2*np.cos(2*u))/si**3+shapmax*(np.sin(omega)*(cu-ecc)+np.sqrt(1-ecc**2)*np.cos(omega)*su)/(1-ecc*cu-(np.sin(omega)*(cu-ecc)+np.sqrt(1-ecc**2)*np.cos(omega)*su)*si)+2*m2/brace*(sw*cume+sqr1me2*cw*su)
			dtddr=-alpha*ecc
			dtddtheta=-x*su*cw*eth/np.sqrt(1-eth**2)*ecc
			dtda0=(np.sin(omega+ae)+ecc*sw)
			dtdb0=(np.cos(omega+ae)+ecc*cw)
			return np.array([dtdt0,dtdpb,dtdecc,dtda1,dtdom,dtdpbdot,np.zeros_like(dtdt0),dtdshapmax,dtda1dot,dtdomdot,dtdm2,dtdsini,dtdedot,dtdgamma,dtda2dot,dtde2dot,dtdorbpx, dtddr,dtddtheta,dtda0,dtdb0,dtdom2dot])
	#
	def T2model(self,der=False):
		sunmass=self.time.cons['GMS']*aultsc/te.sl*1e4*te.iftek
		pb=self.psr.pb*86400
		t0=self.psr.t0
		ecc=self.psr.ecc
		omz=self.psr.om
		x=self.psr.a1
		eps1=self.psr.eps1
		eps2=self.psr.eps2
		t0asc=self.psr.tasc
		shapmax=self.psr.shapmax
		kom=self.psr.kom*np.pi/180
		ki=self.psr.kin*np.pi/180
		torb=0.0
		for i in np.arange(self.psr.bpjep.size):
			jj=(self.bbat.mjd>self.psr.bpjep[i])
			torb[jj]-=self.psr.bpjph[i]/self.psr.f0
			x[jj]+=self.psr.bpja1[i]
			ecc[jj]+=self.psr.bpjec[i]
			omz[jj]+=self.psr.bpjom[i]
			pb[jj]+=self.psr.bpjpb[i]
		an=2*np.pi/pb
		sin_omega=np.sin(kom)
		cos_omega=np.cos(kom)
		si=self.psr.sini
		if si>1.0: 
			si=1.0
			self.psr.sini=1.0
		if si<-1.0: 
			si=-1.0
			self.psr.sini=1.0
		m2=self.psr.m2*sunmass
		mtot=self.psr.mtot*sunmass
		omdot=self.psr.omdot/(180/np.pi*365.25*86400*an)
		gamma=self.psr.gamma
		xdot=self.psr.a1dot
		xpbdot=self.psr.xpbdot
		pbdot=self.psr.pbdot
		pb2dot=self.psr.pb2dot
		edot=self.psr.edot
		pmra=self.psr.pmra/(180/np.pi*3600e3*365.25*86400)
		pmdec=self.psr.pmdec/(180/np.pi*3600e3*365.25*86400)
		dpara=self.psr.px*pxconv
		dr=self.psr.dr
		dth=self.psr.dth
		a0=self.psr.a0
		b0=self.psr.b0
		xomdot=self.psr.xomdot/(180/np.pi*365.25*86400*an)
		afac=self.psr.afac
		eps1dot=self.psr.eps1dot
		eps2dot=self.psr.eps2dot
		daop=self.psr.daop*1e-3/pxconv
		ddaop,ddsr,ddop=0.0,0.0,0.0
		m1=mtot-m2
		er=ecc*(1+dr)
		eth=ecc*(1+dth)
		if mtot:
			arr0=(mtot/an**2)**(1/3)
			arr,arrold=arr0,0
			while np.abs((arr-arrold)/arr)>1e-10:
				arrold=arr
				arr=arr0*(1+(m1*m2/mtot**2-9.0)*0.5*mtot/arr)**(2/3)
			arr=arr0*(1+(m1*m2/mtot**2-9.0)*0.5*mtot/arr)**(2/3)
			ar=arr*m2/mtot
			si=x/ar
			xk=3*mtot/(arr*(1-ecc**2))
			gamma=ecc*m2*(m1+2*m2)/(an*arr*mtot)
			pbdot=-(96*2*np.pi/5)*an**(5/3)*(1-ecc**2)**(-3.5)*(1+(73/24)*ecc**2+(37/96)*ecc**4)*m1*m2*mtot**(-1/3)
			dr=(3*m1**2+6*m1*m2+2*m2**2)/(arr*mtot)
			dth=(3.5*m1**2+6*m1*m2+2*m2**2)/(arr*mtot)
			er=ecc*(1+dr)
			eth=ecc*(1+dth)
			a0aligned=an*ar/(2*np.pi*self.psr.f0*si*np.sqrt(1-ecc**2))
			a0=afac*a0aligned
			b0=0.0
			omdot=xomdot+xk
		if t0:
			tt0=self.bbat.minus(t0).mjd*86400
		elif t0asc:
			tt0=self.bbat.minus(t0asc).mjd*86400
		else:
			raise Exception('No T0 or T0ASC in pulsar paras.')
		ecc+=edot*tt0
		x+=xdot*tt0
		eps1+=eps1dot*tt0
		eps2+=eps2dot*tt0
		ecc[(ecc<0)|(ecc>=1)]=0.0
		orbits  = tt0/pb - 0.5*(pbdot+xpbdot)*(tt0/pb)**2 - 1./6.*pb2dot*(tt0/pb)**3*pb
		self.orbits=orbits
		phase=2*np.pi*(orbits%1)
		if 'ecc' in self.psr.paras:
			u=phase*1.0
			du=1
			while np.min(np.abs(du))>1e-14:
				du=(phase-(u-ecc*np.sin(u)))/(1-ecc*np.cos(u))
				u+=du
			su,cu=np.sin(u),np.cos(u)
			onemecu=1-ecc*cu
			cae=(cu-ecc)/onemecu
			sae=np.sqrt(1-ecc**2)*su/onemecu
			ae=np.arctan2(sae,cae)
			ae[ae<0.0]+=2*np.pi
			ae=2*np.pi*orbits+ae-phase
			omega=omz*np.pi/180+omdot*ae
			sw,cw=np.sin(omega),np.cos(omega)
			sqr1me2=np.sqrt(1-ecc**2)
			cume=cu-ecc
			if ('kin' in self.psr.paras) and ('kom' in self.psr.paras) and ('pmra' in self.psr.paras):
				sini,cosi=np.sin(ki),np.cos(ki)
				tani=sini/cosi
				psr_location_equ=self.psr_location.copy()
				psr_location_equ.ecl2equ()
				sin_delta=psr_location_equ.z
				cos_delta=np.cos(np.arcsin(sin_delta))
				sin_alpha=psr_location_equ.y/cos_delta
				cos_alpha=psr_location_equ.x/cos_delta
				earth_pos=self.time.pos[2].copy()
				earth_pos.ecl2equ()
				delta_i0=earth_pos.dot(te.vector(-sin_alpha,cos_alpha,np.zeros_like(sin_alpha),center='bary',scale='si',coord='equ',unit=te.sl,type0='pos'))/aultsc
				delta_j0=earth_pos.dot(te.vector(-sin_delta*cos_alpha,-sin_delta*sin_alpha,cos_delta,center='bary',scale='si',coord='equ',unit=te.sl,type0='pos'))/aultsc
				if daop: tmp=1/daop
				else: tmp=dpara
				dk011=-x*tmp/sini*delta_i0*sin_omega
				dk012=-x*tmp/sini*delta_j0*cos_omega
				dk021= x*tmp/tani*delta_i0*cos_omega
				dk022=-x*tmp/tani*delta_j0*sin_omega
				dk031= x*tt0/sini*pmra*sin_omega
				dk032= x*tt0/sini*pmdec*cos_omega
				dk041= x*tt0/tani*pmra*cos_omega
				dk042=-x*tt0/tani*pmdec*sin_omega
				cc=(cw*(cu-er)-np.sqrt(1-eth**2)*sw*su)
				ss=(sw*(cu-er)+cw*np.sqrt(1-eth**2)*su)
				ddaop=(dk011+dk012)*cc+(dk021+dk022)*ss
				ddsr=(dk031+dk032)*cc+(dk041+dk042)*ss
				ddop=dpara/aultsc/2.0*x**2*(1/sini**2-0.5+0.5*ecc**2*(1+sw**2-3/sini**2)-2*ecc*(1/sini**2-sw**2)*cume-sqr1me2*2*sw*cw*su*cume+0.5*(np.cos(2*omega)+ecc**2*(1/sini**2+cu**2))*np.cos(2*u))
			if self.psr.shapmax:
				sdds=1-exp(-shapmax)
				brace=onemecu-sdds*(sw*cume+sqr1me2*cw*su)
			else: brace=onemecu-si*(sw*cume+sqr1me2*cw*su)
			da=a0*(np.sin(omega+ae) + ecc*sw) + b0*(np.cos(omega+ae) + ecc*cw)
			alpha=x*sw
			beta=x*np.sqrt(1-eth**2)*cw
			bg=beta+gamma
			dre=alpha*(cu-er)+bg*su
			drep=-alpha*su+bg*cu
			drepp=-alpha*cu-bg*su
			anhat=an/onemecu
		elif 'eps1' in self.psr.paras:
			dre  = x*(np.sin(phase)-0.5*(eps1*np.cos(2.0*phase)-eps2*np.sin(2.0*phase)))
			drep = x*np.cos(phase)
			drepp=-x*np.sin(phase)
			if ('kin' in self.psr.paras) and ('kom' in self.psr.paras) and ('pmra' in self.psr.paras):
				sini,cosi=np.sin(ki),np.cos(ki)
				tani=sini/cosi
				psr_location_equ=self.psr_location.copy()
				psr_location_equ.ecl2equ()
				sin_delta=psr_location_equ.z
				cos_delta=np.cos(np.arcsin(sin_delta))
				sin_alpha=psr_location_equ.y/cos_delta
				cos_alpha=psr_location_equ.x/cos_delta
				earth_pos=self.time.pos[2].copy()
				earth_pos.ecl2equ()
				delta_i0=earth_pos.dot(te.vector(-sin_alpha,cos_alpha,np.zeros_like(sin_alpha),center='bary',scale='si',coord='equ',unit=te.sl,type0='pos'))/aultsc
				delta_j0=earth_pos.dot(te.vector(-sin_delta*cos_alpha,-sin_delta*sin_alpha,cos_delta,center='bary',scale='si',coord='equ',unit=te.sl,type0='pos'))/aultsc
				if daop: tmp=1/daop
				else: tmp=dpara
				dk011=-x*tmp/sini*delta_i0*sin_omega
				dk012=-x*tmp/sini*delta_j0*cos_omega
				dk021= x*tmp/tani*delta_i0*cos_omega
				dk022=-x*tmp/tani*delta_j0*sin_omega
				dk031= x*tt0/sini*pmra*sin_omega
				dk032= x*tt0/sini*pmdec*cos_omega
				dk041= x*tt0/tani*pmra*cos_omega
				dk042=-x*tt0/tani*pmdec*sin_omega
				cc=(cw*(cu-er)-np.sqrt(1-eth**2)*sw*su)
				ss=(sw*(cu-er)+cw*np.sqrt(1-eth**2)*su)
				ddaop=(dk011+dk012)*cc+(dk021+dk022)*ss
				ddsr=(dk031+dk032)*cc+(dk041+dk042)*ss
				ddop=0.0
			brace=1-si*np.sin(phase)
			da=a0*np.sin(phase)+b0*np.cos(phase)
			anhat=an
			ecc=0.0
			if ('h3' in self.psr.paras) and (('h4' in self.psr.paras) or ('stig' in self.psr.paras)):
				ecc=np.sqrt(eps1**2+eps2**2)
				omega=np.arctan2(eps1,eps2)
		else:
			raise Exception('Either DD or ELL1 model cannot be used.')
		#
		if ('h3' in self.psr.paras) and (('h4' in self.psr.paras) or ('stig' in self.psr.paras)):
			h3=self.psr.h3
			nharm=4
			if self.psr.h4:
				h4=self.psr.h4
				mode=2
				if 'nharm' in self.psr.paras:
					nharm=self.nharm
				if self.psr.stig: print("Warning: Both H4 and STIG in par file, then ignoring STIG")
				si=2*h3*h4/(h3**2+h4**2)
				m2=h3**4/h4**3
			else:
				if 'stig' in self.psr.paras:
					stig=self.psr.stig
					mode=1
					si=2*stig/(1+stig**2)
					m2=h3/stig**3
				else:
					mode,h4,nharm,stig=0,0,3,0
			if si>1.0: si=1.0
			brace=1-si*np.sin(phase)
			dlogbr=np.log(brace)
			trueanom=phase*1.0
			if mode==0: ds=-4/3*h3*np.sin(trueanom)
			elif mode==1:
				fs=1.0+stig**2-2*stig*np.sin(trueanom)
				lgf=np.log(fs)
				lsc=lgf+2.0*stig*np.sin(trueanom)-stig**2*np.cos(2*trueanom)
				ds=-2*m2*lsc
			else: ds=calcdh(trueanom,h3,h4,nharm,0)
		else:
			dlogbr=np.log(brace)
			ds=-2*m2*dlogbr
		if 'ecc' in self.psr.paras:
			d2bar=dre*(1-anhat*drep+anhat**2*(drep**2+0.5*dre*drepp-0.5*ecc*su*dre*drep/onemecu))+ds+da+ddaop+ddsr+ddop
		else:
			d2bar=dre*(1-anhat*drep+anhat**2*(drep**2+0.5*dre*drepp))+ds+da+ddaop+ddsr+ddop
		torb-=d2bar
		if not der:
			return torb
		else:
			if 'ecc' in self.psr.paras:
				dphase_dt=-2*np.pi*(1/pb-0.5*(pbdot+xpbdot)*tt0/pb**2)
				dphase_dpb=dphase_dt*tt0/pb
				tmp=(-x*sw*su+bg*cu)/onemecu
				dtdpb=tmp*dphase_dpb*86400
				dtda1=sw*(cu-er)+np.sqrt(1-eth**2)*cw*su
				dtdecc=-alpha*(1+dr)-x*su*cw*ecc*(1+dth)**2/np.sqrt(1-eth**2)+tmp*su
				dtdom=(x*cw*(cu-er)-x*sw*su*np.sqrt(1-eth**2))*np.pi/180
				dtdomdot=dtdom*ae/(365.25*86400)/an
				dtdt0=tmp*dphase_dt*86400
				dtdpbdot=-tmp*2*np.pi*0.5*(tt0/pb)**2
				dtdgamma=su
				dtdm2=-2*dlogbr*sunmass
				dtdtasc=np.zeros_like(dtdt0)
				dtdeps1=np.zeros_like(dtdt0)
				dtdeps2=np.zeros_like(dtdt0)
				dtddr=-alpha*ecc
				dtddth=-x*su*cw*eth/np.sqrt(1-eth**2)*ecc
				dtda0=(np.sin(omega+ae)+ecc*sw)
				dtdb0=(np.cos(omega+ae)+ecc*cw)
			elif 'eps1' in self.psr.paras:
				tmp=x*np.cos(phase)
				dtda1=np.sin(phase)
				dtdeps1=-0.5*x*np.cos(2*phase)
				dtdeps2=tmp*dtda1
				dtdtasc=-tmp*dphase_dt*86400
				dtdsini=2*m2*dtda1/brace
				dtdm2=-2*dlogbr*sunmass
				dtdpb=-tmp*dphase_dt*tt0/pb*86400
				dtdpbdot=-0.5*tmp*2*np.pi*(tt0/pb)**2
				dtdomdot=(-0.5*x*(np.cos(2*phase)*e2+np.sin(2*phase)*e1)*tt0+dre*(ddre_dphase-2*ddre_dt*ddre_dphase-dre*d2dre_dphase2*dphase_dt))*np.pi/(180*365.25*86400)
				dtdt0=np.zeros_like(dtdtasc)
				dtdecc=np.zeros_like(dtdtasc)
				dtdom=np.zeros_like(dtdtasc)
				dtdgamma=np.zeros_like(dtdasc)
				dtddr,dtddth=np.zeros_like(dtdasc),np.zeros_like(dtdasc)
				dtda0,dtdb0=np.sin(phase),np.cos(phase)
			dtda1dot=dtda1*tt0
			dtdedot=dtdecc*tt0
			dtdpb2dot=dtdpbdot/3*tt0
			dtdeps1dot=dtdeps1*tt0
			dtdeps2dot=dtdeps2*tt0
			if 'shapmax' in self.psr.paras:
				dtdshapmax=2*m2*(sw*cume+sqr1me2*cw*su)/brace*(1-sdds)
				dtdsini=np.zeros_like(dtdt0)
			else:
				dtdshapmax=np.zeros_like(dtdt0)
				dtdsini=2*m2*(sw*cume+sqr1me2*cw*su)*np.cos(ki)/brace
			if 'h3' in self.psr.paras:
				if mode==0:
					dtdh4,dtdstig=np.zeros_like(dtdtasc),np.zeros_like(dtdtasc)
					dtdh3=-4/3*np.sin(trueanom)
				elif mode==1:
					dtdh4=np.zeros_like(dtdtasc)
					dtdh3=-2*lsc/stig**3
					dtdstig=-2*m2*(1/fs*(2*stig-2*np.sin(trueanom))+2*np.sin(trueanom)-2*stig*np.cos(trueanom))+6*lsc*h3/stig**4
				else:
					dtdstig=np.zeros_like(dtdtasc)
					dtdh3,dtdh4=calcdh(trueanom,h3,h4,nharm,1)
			else:
				dtdh3,dtdh4,dtdstig=np.zeros_like(dtdtasc),np.zeros_like(dtdtasc),np.zeros_like(dtdtasc)
			if ('kin' in self.psr.paras) and ('kom' in self.psr.paras):
				if daop: tmp=1/daop
				else: tmp=dpara
				dk013 =-x*tmp/si*delta_i0*cos_omega
				dk014 = x*tmp/si*delta_j0*sin_omega
				dk023 = x*tmp/tani*delta_i0*sin_omega
				dk024 = x*tmp/tani*delta_j0*cos_omega
				dk033 = x*tt0/si*pmra*cos_omega
				dk034 =-x*tt0/si*pmdec*sin_omega
				dk043 =-x*tt0/tani*pmra*sin_omega
				dk044 =-x*tt0/tani*pmdec*cos_omega
				dtdkom=cc* (dk033+dk034+dk013+dk014) + ss*(dk043+dk044+dk023+dk024)
				dtdkin=cc/np.sin(ki)*(dk043+dk044+dk023+dk024)-ss/np.sin(ki)*(dk013+dk014+dk033+dk034)
				if 'ecc' in self.psr.paras:
					dtdkin+=dpara/aultsc/2.0*x**2*np.cos(ki)*np.sin(ki)**-3.0*(ecc**2*(3-np.cos(2*u))+4*ecc*cume-2)
				elif 'eps1' in self.psr.paras:
					dtdkin+=dtdsini
			else:
				dtdkom=np.zeros_like(dtdt0)
				dtdkin=np.zeros_like(dtdt0)
			dtdbpjph=np.zeros([self.psr.bpjep.size,self.time.size])
			dtdbpja1=np.zeros([self.psr.bpjep.size,self.time.size])
			dtdbpjec=np.zeros([self.psr.bpjep.size,self.time.size])
			dtdbpjom=np.zeros([self.psr.bpjep.size,self.time.size])
			dtdbpjpb=np.zeros([self.psr.bpjep.size,self.time.size])
			if 'bpjph' in self.psr.paras:
				for i in np.arange(self.psr.bpjep.size):
					jj=(self.bbat.mjd>self.psr.bpjep[i])
					dtdbpjph[i,jj]=self.psr.f0
					dtdbpja1[i,jj]=dtda1[jj]
					dtdbpjec[i,jj]=dtdecc[jj]
					dtdbpjom[i,jj]=dtdom[jj]
					dtdbpjpb[i,jj]=dtdpb[jj]
			if 'mtot' in self.psr.paras:
				m=mtot
				dtdeth=-x*su*cw*eth/np.sqrt(1-eth**2)
				dtdsini=2*m2*(sw*cume+sqr1me2*cw*su)/brace
				darrdm=(m1*m2/m**2-9.0+m2/m-2*m1*m2/m**2)*0.5/(2.5*(arr/arr0)**1.5-1)
				darrdm2=(m1-m2)/m*0.5/(2.5*(arr/arr0)**1.5-1)
				ddrdm=((6*m1+6*m2)*arr*m-(3*m1**2+6*m1*m2+2*m2**2)*(arr+darrdm*m))/(arr*m)**2
				ddrdm2=(-2*m2*arr*m-(3*m1**2+6*m1*m2+2*m2**2)*darrdm2*m)/(arr*m)**2
				ddthdm=((7*m1+6*m2)*arr*m-(3.5*m1**2+6*m1*m2+2*m2**2)*(arr+darrdm*m))/(arr*m)**2
				ddthdm2=((-m1-2*m2)*arr*m-(3.5*m1**2+6*m1*m2+2*m2**2)*darrdm2*m)/(arr*m)**2
				dgammadm=ecc*(m2*arr*m-m2*(m1+2*m2)*(arr+darrdm*m))/an/(arr*m)**2
				dgammadm2=ecc*((m1+3*m2)*arr*m-m2*(m1+2*m2)*darrdm2*m)//an/(arr*m)**2
				dpbdotdm=-(96*2*np.pi/5)*an**(5/3)*(1-ecc**2)**(-3.5)*(1+(73/24)*ecc**2+(37/96)*ecc**4)*(m2/m**(1/3)-m1*m2*m**(-4/3)/3)
				dpbdotdm2=-(96*2*np.pi/5)*an**(5/3)*(1-ecc**2)**(-3.5)*(1+(73/24)*ecc**2+(37/96)*ecc**4)*(m1-m2)*m**(-1/3)
				dsinidm=x*(arr*m2-darrdm*m2*m)/(arr*m2)**2
				dsinidm2=x*m*(-darrdm2*m2-arr)/(arr*m2)**2
				dkdm=3*(arr-darrdm*m)/(arr**2*(1-ecc**2))
				dkdm2=3*m(-darrdm2)/(arr**2*(1-ecc**2))
				dtdmtot=(-alpha*ecc*ddrdm+dtdeth*ecc*ddthdm+dtdgamma*dgammadm+dtdpbdot*dpbdotdm+dsdsini*dsinidm+dtdom*ae*dkdm)*sunmass
				dtdm2=(-alpha*ecc*ddrdm2+dtdeth*ecc*ddthdm2+dtdgamma*dgammadm2+dtdpbdot*dpbdotdm2+dsdsini*dsinidm2+dtdom*ae*dkdm2)*sunmass
			else:
				dtdmtot=np.zeros_like(dtdt0)
			return np.array([dtdt0, dtdpb, np.zeros_like(dtdt0), dtdecc, dtda1, dtdom, dtdtasc, dtdeps1, dtdeps2, dtdshapmax, dtdkom, dtdkin, dtdh3, dtdstig, dtdh4, np.zeros_like(dtdt0), dtdm2, dtdmtot, np.zeros_like(dtdt0), dtdsini, dtdpbdot, dtda1dot, dtdomdot, dtdgamma, np.zeros_like(dtdt0), dtdbpjph, dtdbpja1, dtdbpjec, dtdbpjom, dtdbpjpb, dtdedot, dtddr, dtddth, dtda0, dtdb0, dtdeps1dot, dtdeps2dot, np.zeros_like(dtdt0), np.zeros_like(dtdt0), np.zeros_like(dtdt0), dtdpb2dot])
	#
	def T2_PTAmodel(self,der=False):
		return 0
#
def calculate_gw(ra1,dec1,ra2,dec2):
	lambda_p,beta_p,lamb,beta=ra1,dec1,ra2,dec2
	clp,slp,cbp,sbp,cl,sl,cb,sb=np.cos(lambda_p),np.sin(lambda_p),np.cos(beta_p),np.sin(beta_p),np.cos(lamb),np.sin(lamb),np.cos(beta),np.sin(beta)
	vn=np.array([[clp*cbp,slp*cbp,sbp]])
	costheta=cb*cbp*np.cos(lamb-lambda_p)+sb*sbp
	e11p,e21p,e31p,e22p,e32p,e33p=sl**2-cl**2*sb**2,-sl*cl*(sb**2+1),cl*sb*cb,cl**2-sl**2*sb**2,sl*sb*cb,-cb**2
	mep=np.array([[e11p,e12p,e13p],[e12p,e22p,e23p],[e13p,e23p,e33p]])
	resp=vn@mep@vn.T
	e11c,e21c,e31c,e22c,e32c,e33c=2*sl*cl*sb,-(cl**2-sl**2)*sb,-sl*cb,-2*sl*cl*sb,cl*cb,0
	mec=np.array([[e11c,e12c,e13c],[e12c,e22c,e23c],[e13c,e23c,e33c]])
	resc=vn@mec@vn.T
	return resp,resc,costheta
#
def calcdh(ae,h3,h4,nharm,sel):
	s=h4/h3
	firsth3=-4/3*np.sin(3*ae)
	secondh3=0.0
	firsth4=np.cos(4*ae)
	secondh4=0.0
	sd3=-4/3*h3*np.sin(3*ae)
	sd4=h4*np.cos(4*ae)
	sd5=0
	fs=s*np.sin(5*ae)/5.0
	dfsds=np.sin(5*ae)/5.0
	if nharm>5:
		for evenct in np.arange(6,nharm,2):
			count=evenct
			fs+=(-1)**np.int(count/2)/count*s**(count-4)*np.cos(count*ae)
			dfsds+=(-1)**np.int(count/2)/count*s**(count-5)*np.cos(count*ae)*(count-4)
	if nharm>6:
		for oddct in np.arange(7,nharm,2):
			count=oddct
			fs+=(-1)**np.int((count-1)/2)/count*s**(count-4)*np.cos(count*ae)
			dfsds+=(-1)**np.int((count-1)/2)/count*s**(count-5)*np.cos(count*ae)*(count-4)
	if nharm>4:
		secondh3=-4*dfsds*s**2
		secondh4=4*(fs+dfsds*s)
		sd5=4*h4*fs
	if sel==1:
		return firsth3+secondh3,firsth4+secondh4
	elif sel==0:
		return sd3+sd4+sd5


