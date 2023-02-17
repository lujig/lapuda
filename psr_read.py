import numpy as np
import time_eph as te
import subprocess as sp
import copy as cp
import os
#
class psr:
	def __init__(self,name,parfile=False,glitch=False):
		self.name=name
		self.readpara(parfile=parfile,glitch=glitch)
		if self.units=='TDB':
			self.change_units()
		if 'raj' in self.paras: self.cal_pos()
		elif 'elong' in self.paras: self.cal_pos_ecl()
	#
	def copy(self):
		return cp.deepcopy(self)
	#
	def modify(self,para,paraval=1e200):
		if para not in self.paras:
			self.paras.append(para)
			if paraval>1e190:
				return
			if para in {'p2', 'p1', 'f1', 'p3', 'f2', 'f3'}:
				if para=='p1': self.paras.append('f1')
				elif para=='f1': self.paras.append('p1')
				elif para=='p2': self.paras.append('f2')
				elif para=='f2': self.paras.append('p2')
				elif para=='p3': self.paras.append('f3')
				elif para=='f3': self.paras.append('p3')
		if not paraval>1e190:
			if para=='binary':
				if self.binary:
					paradict0=eval('paras_'+self.binary)
					paralist0=paradict0['necessary']+paradict0['optional']
					paradict=eval('paras_'+paraval)
					paralist=paradict['necessary']+paradict['optional']
					for i in paralist0:
						if (i in self.paras) and (i not in paralist): 
							delattr(self, i)
							self.paras.remove(i)
			elif para in paras_binary:
				if self.binary:
					paradict=eval('paras_'+self.binary)
					paralist=paradict['necessary']+paradict['optional']
					if para not in paralist: raise Exception('The '+self.binary+' model does not have parameter '+para+'.')
				else:
					raise Exception('The parameter '+para+' is a parameter in binary model, but the pulsar '+self.name+' is not a binary pulsar.')
			elif para in ['name',  'ephver', 'ephem', 'units']:
				raise Exception('The parameter '+para+' cannot be modified.')
			elif para in {'p0', 'f0', 'p2', 'p1', 'f1', 'p3', 'f2', 'f3'}:
				if para=='p0': self.f0=1/paraval
				elif para=='f0': self.p0=1/paraval
				elif para=='p1': self.f1=-paraval*self.f0**2
				elif para=='f1': self.p1=-paraval*self.p0**2
				elif para=='p2': self.f2=-paraval*self.f0**2-2*self.p1*self.f0*self.f1
				elif para=='f2': self.p2=-paraval*self.p0**2-2*self.f1*self.p0*self.p1
				elif para=='p3': self.f3=-paraval*self.f0**2-4*self.p2*self.f0*self.f1-2*self.p1*self.f1**2-2*self.p1*self.f0*self.f2
				elif para=='f3': self.p3=-paraval*self.p0**2-4*self.f2*self.p0*self.p1-2*self.f1*self.p1**2-2*self.f1*self.p0*self.p2
			elif para not in all_paras:
				raise Exception('The parameter '+para+' cannot be recognized.')
			if para in paras_time:
				if type(paraval) is not te.time:
					paraval=te.time(paraval,0,scale=self.units.lower())
			self.__setattr__(para,paraval)
			if para in ['raj','decj','pmra','pmdec']: self.cal_pos()
			elif para in ['elong','elat','pmelong','pmelat']: self.cal_pos_ecl()
		else:
			if para=='binary':
				paradict0=eval('paras_'+self.binary)
				paralist0=paradict0['necessary']+paradict0['optional']
				for i in paralist0:
					if (i in self.paras): 
						delattr(self, i)
						self.paras.remove(i)
			elif para in paras_binary:
				paradict=eval('paras_'+self.binary)
				paralist=paradict['necessary']
				if para in paralist: raise Exception('The parameter '+para+' is a necessary parameter in '+self.binary+' model and cannot be removed.')
			elif para in ['name',  'ephver', 'ephem', 'p0', 'f0', 'dm', 'units']:
				raise Exception('The parameter '+para+' cannot be removed.')
			elif para in {'p2', 'p1', 'f1', 'p3', 'f2', 'f3'}:
				if para=='p1': 
					self.f1=0
					self.paras.remove('f1')
				elif para=='f1':
					self.p1=0
					self.paras.remove('p1')
				elif para=='p2': 
					self.f2=0
					self.paras.remove('f2')
				elif para=='f2': 
					self.p2=0
					self.paras.remove('p2')
				elif para=='p3': 
					self.f3=0
					self.paras.remove('f3')
				elif para=='f3': 
					self.p3=0
					self.paras.remove('p3')
			elif para=='pepoch':
				if ('f1' in self.paras) or ('f2' in self.paras) or ('f3' in self.paras) or ('f4' in self.paras) or ('f5' in self.paras):
					raise Exception('The parameter pepoch cannot be removed if frequency derivative exists.')
			elif para in ['dmepoch','posepoch']:
				paraval=self.pepoch
			self.__setattr__(para,paraval)				
	#
	def __eq__(self,other):
		if type(other) is not psr:
			return False
		if other.paras!=self.paras:
			return False
		for i in self.paras:
			if self.__getattribute__(i)!=other.__getattribute__(i):
				return False
		return True
	#
	def cal_pos(self):
		alpha=self.raj
		delta=self.decj
		ca=np.cos(alpha)
		sa=np.sin(alpha)
		cd=np.cos(delta)
		sd=np.sin(delta)
		self.pos_equ=te.vector(ca*cd,sa*cd,sd,center='bary',scale='si',coord='equ',unit=te.sl,type0='pos')
		self.pos=self.pos_equ.copy()
		self.pos.equ2ecl()
		x,y,z=self.pos.xyz()
		self.elong=np.arctan2(y,x)%(np.pi*2)
		self.elat=np.arctan2(z,np.sqrt(x**2+y**2))
		convert = 1.0/1000.0/60.0/60.0*np.pi/180.0*100.0
		# note that the mas/yr in PMRA is not the true mas, but to be the mas of RA. similar in PMRA2
		if ('pmra' in self.paras):
			self.vel_equ=te.vector(convert*(-self.pmra*sa-self.pmdec*ca*sd),convert*(self.pmra*ca-self.pmdec*sa*sd),convert*self.pmdec*cd,center='bary',scale='si',coord='equ',unit=te.sl,type0='vel')
			self.vel=self.vel_equ.copy()
			self.vel.equ2ecl()
		else:
			self.vel=te.vector(0,0,0,center='bary',scale='si',coord='ecl',unit=te.sl,type0='vel')
			self.vel_equ=te.vector(0,0,0,center='bary',scale='si',coord='equ',unit=te.sl,type0='vel')
		if ('pmra2' in self.paras):
			self.acc_equ=te.vector(convert*(-self.pmra2*sa-self.pmdec2*ca*sd),convert*(self.pmra2*ca-self.pmdec2*sa*sd),convert*self.pmdec2*cd,center='bary',scale='si',coord='equ',unit=te.sl,type0='vel') # something wrong???
			self.acc=self.acc_equ.copy()
			self.acc.equ2ecl()
		else:
			self.acc=te.vector(0,0,0,center='bary',scale='si',coord='ecl',unit=te.sl,type0='acc')
			self.acc_equ=te.vector(0,0,0,center='bary',scale='si',coord='equ',unit=te.sl,type0='acc')
	#
	def cal_pos_ecl(self):
		alpha=self.elong
		delta=self.elat
		ca=np.cos(alpha)
		sa=np.sin(alpha)
		cd=np.cos(delta)
		sd=np.sin(delta)
		self.pos=te.vector(ca*cd,sa*cd,sd,center='bary',scale='si',coord='ecl',unit=te.sl,type0='pos')
		self.pos_equ=self.pos.copy()
		self.pos_equ.ecl2equ()
		x,y,z=self.pos_equ.xyz()
		self.raj=np.arctan2(y,x)%(np.pi*2)
		self.decj=np.arctan2(z,np.sqrt(x**2+y**2))
		convert = 1.0/1000.0/60.0/60.0*np.pi/180.0*100.0
		if hasattr(self,'pmelong'):
			self.vel=te.vector(convert*(-self.pmelong*sa-self.pmelat*ca*sd),convert*(self.pmelong*ca-self.pmelat*sa*sd),convert*self.pmelat*cd,center='bary',scale='si',coord='ecl',unit=te.sl,type0='vel')
			self.vel_equ=self.vel.copy()
			self.vel_equ.ecl2equ()
			x,y,z=self.vel_equ.xyz()
			self.pmra=np.sqrt(x**2+y**2)/np.cos(self.decj)/convert
			self.pmdec=z/convert
		else:
			self.vel=te.vector(0,0,0,center='bary',scale='si',coord='ecl',unit=te.sl,type0='vel')
			self.vel_equ=te.vector(0,0,0,center='bary',scale='si',coord='equ',unit=te.sl,type0='vel')
		if hasattr(self,'pmelong2'):
			self.acc=te.vector(convert*(-self.pmelong2*sa-self.pmelat2*ca*sd),convert*(self.pmelong2*ca-self.pmelat2*sa*sd),convert*self.pmelat2*cd,center='bary',scale='si',coord='ecl',unit=te.sl,type0='vel') # something wrong???
			self.acc_equ=self.acc.copy()
			self.acc_equ.ecl2equ()
		else:
			self.acc=te.vector(0,0,0,center='bary',scale='si',coord='ecl',unit=te.sl,type0='acc')
			self.acc_equ=te.vector(0,0,0,center='bary',scale='si',coord='equ',unit=te.sl,type0='acc')
	#
	def dpos(self,vectype,coord1,coord2):
		arcsec2rad=np.pi/648000.0
		obliq=84381.4059*arcsec2rad
		ce=np.cos(obliq)
		se=np.sin(obliq)
		convert = 1.0/1000.0/60.0/60.0*np.pi/180.0*100.0
		if coord2=='ecl':
			alpha=self.elong
			delta=self.elat
			coordconv=np.array([[1,0,0],[0,ce,se],[0,-se,ce]])
		elif coord2=='equ':
			alpha=self.raj
			delta=self.decj
			coordconv=np.array([[1,0,0],[0,ce,-se],[0,se,ce]])
		ca=np.cos(alpha)
		sa=np.sin(alpha)
		cd=np.cos(delta)
		sd=np.sin(delta)
		if vectype=='pos': dvecdcoord=np.array([[-sa*cd,ca*cd,0],[-ca*sd,-sa*sd,cd]])
		elif vectype=='vel' or vectype=='acc': dvecdcoord=np.array([[-sa,ca,0],[-ca*sd,-sa*sd,cd]])*convert
		if coord1==coord2: return dvecdcoord
		else: return (dvecdcoord@coordconv)
	#
	def change_units(self):
		if self.units=='TCB':
			return
		for i in self.paras:
			if hasattr(self,i):
				val=self.__getattribute__(i)
			else:
				continue
			if i in para_glitch: pass
			elif i in paras_p1: factor=te.iftek
			elif i in paras_m1: factor=1/te.iftek
			elif i in paras_m2: factor=1/te.iftek**2
			elif i in paras_m3: factor=1/te.iftek**3
			elif i in paras_m4: factor=1/te.iftek**4
			elif i in paras_m5: factor=1/te.iftek**5
			elif i in paras_m6: factor=1/te.iftek**6
			elif i in paras_time:
				factor=1
				self.__setattr__(i,val.tcb())
			elif i in paras_time_array:
				factor=1
				val_new=[]
				for k in val:
					val_new.append(k.tcb())
				self.__setattr__(i,val_new)
			elif i in paras_eph:
				factor=1
				self.__setattr__(i,te.time(val,np.zeros_like(val),scale='tdb').tcb().mjd[0])
			else:
				continue
			if factor!=1: self.__setattr__(i,self.__getattribute__(i)*factor)
		self.units='TCB'
	#
	def __str__(self):
		string=''
		for i in self.paras:
			if not hasattr(self,i): continue
			if i=='name':
				string+='{:12s} {:25s}'.format('PSRJ',self.__getattribute__(i))+'\n'
			elif i in para_with_err:
				val=self.__getattribute__(i)
				if type(val) is te.time:
					val=val.mjd
				if hasattr(self,i+'_err'): err=self.__getattribute__(i+'_err')
				else: err=''
				if type(val) is np.ndarray:
					err=np.array(err).reshape(-1)
					for k in np.arange(val.size):
						if k==0: string+='{:12s} '.format(i.upper())
						else: string+='{:12s} '.format('')
						val_str=str(val[k])
						err_str=str(err[k])
						string+='{:25s} {:25s}'.format(val_str,err_str)+'\n'
				else:
					if i in paras_time:
						val_str=str(val.mjd[0])
					else:
						val_str=str(val)
					err_str=str(err)
					string+='{:12s} {:25s} {:25s}'.format(i.upper(),val_str,err_str)+'\n'
			else:
				val=self.__getattribute__(i)
				if type(val)==te.time:
					val=val.mjd
				if type(val) is np.ndarray:
					for k in np.arange(val.size):
						if k==0: string+='{:12s} '.format(i.upper())
						else: string+='{:12s} '.format('')
						val_str=str(val[k])
						string+='{:25s} '.format(val_str)+'\n'
				else:
					val_str=str(val)
					string+='{:12s} {:25s}'.format(i.upper(),val_str)+'\n'
		return string
	#
	def __repr__(self):
		return self.__str__()
	#
	def tdb_par(self):
		tmp=self.copy()
		for i in tmp.paras:
			if hasattr(tmp,i):
				val=tmp.__getattribute__(i)
			else:
				continue
			if i in para_glitch: pass
			elif i in paras_p1: factor=1/te.iftek
			elif i in paras_m1: factor=te.iftek
			elif i in paras_m2: factor=te.iftek**2
			elif i in paras_m3: factor=te.iftek**3
			elif i in paras_m4: factor=te.iftek**4
			elif i in paras_m5: factor=te.iftek**5
			elif i in paras_m6: factor=te.iftek**6
			elif i in paras_time:
				factor=1
				tmp.__setattr__(i,val.tdb())
			elif i in paras_time_array:
				factor=1
				val_new=[]
				for k in val:
					val_new.append(k.tdb())
				tmp.__setattr__(i,val_new)
			elif i in paras_eph:
				factor=1
				tmp.__setattr__(i,te.time(val,np.zeros_like(val),scale='tcb').tdb().mjd[0])
			else:
				continue
			if factor!=1: tmp.__setattr__(i,tmp.__getattribute__(i)*factor)
		tmp.units='TDB'
		return tmp.__str__()
	#
	def writepar(self,parfile):
		tmp=self.copy()
		if 'raj' in tmp.paras:
			tmp.raj=tmp.raj/np.pi*12
			if hasattr(tmp,'raj_err'): tmp.raj_err=tmp.raj_err/np.pi*12
		if 'decj' in tmp.paras:
			tmp.decj=tmp.decj/np.pi*180
			if hasattr(tmp,'decj_err'): tmp.decj_err=tmp.decj_err/np.pi*180
		if 'elong' in tmp.paras:
			tmp.elong=tmp.elong/np.pi*180
			if hasattr(tmp,'elong_err'): tmp.elong_err=tmp.elong_err/np.pi*180
		if 'elat' in tmp.paras:
			tmp.elat=tmp.elat/np.pi*180
			if hasattr(tmp,'elat_err'): tmp.elat_err=tmp.elat_err/np.pi*180
		p=open(parfile,'w')
		p.write(tmp.__str__())
		p.close()
	#
	def readpara(self,parfile=False,glitch=False):
		if parfile: paras=open(self.name).read().strip().split('\n')
		elif type(self.name) is str: paras=sp.getoutput('psrcat -e '+self.name).split('\n')
		elif type(self.name) is list: paras=self.name
		paras=dict(list(map(lambda x: [x[0],np.array(x[1].split())],map(lambda x: x.split(None,1),paras))))
		paras_key=paras.keys()
		self.paras=[]
		if 'PSRJ' in paras_key:
			self.name=paras['PSRJ'][0]
			self.paras.extend(['name'])
		elif 'NAME' in paras_key:
			self.name=paras['NAME'][0]
			self.paras.extend(['name'])
		else:
			raise Exception('No pulsar name in par file.')
		#
		if 'UNITS' in paras_key:
			i=paras['UNITS']
			self.units=i[0]
			self.paras.append('units')
		else:
			raise Exception('No para units in par file.')
		#
		if ('RAJ' in paras_key) and ('DECJ' in paras_key):
			i=paras['RAJ']
			ra=np.float64(i[0].split(':'))
			lra=len(ra)
			unit=np.array([3600,60,1])/3600/12*np.pi
			if lra in [1,2,3]:
				self.raj=(ra*unit[:lra]).sum()
				if len(i)==2:
					self.raj_err=np.float64(i[1])/60**lra*5*np.pi
				else:
					print('Warning: The parameter raj has no error.')
					self.raj_err=0
			else:
				raise Exception('The format of RA is wrong.')
			i=paras['DECJ']
			dec=np.float64(i[0].split(':'))
			ldec=len(dec)
			if i[0][0]=='-':
				unit=np.array([3600,-60,-1])/3600/180*np.pi
			else:
				unit=np.array([3600,60,1])/3600/180*np.pi
			if ldec in [1,2,3]:
				self.decj=(dec*unit[:ldec]).sum()
				if len(i)==2:
					self.decj_err=np.float64(i[1])/60**ldec/3*np.pi
				else:
					print('Warning: The parameter decj has no error.')
					self.decj_err=0
			else:
				raise Exception('The format of DEC is wrong.')
			self.paras.extend(['raj','decj'])
		elif ('ELONG' in paras_key) and ('ELAT' in paras_key):
			self.paras.extend(['elong','elat'])
			i=paras['ELONG']
			self.elong=np.float64(i[0])/180*np.pi
			if len(i)==2: self.elong_err=np.float64(i[1])/180*np.pi
			else:
				print('Warning: The parameter elong has no error.')
				self.elong_err=0
			i=paras['ELAT']
			self.elat=np.float64(i[0])/180*np.pi
			if len(i)==2: self.elat_err=np.float64(i[1])/180*np.pi
			else:
				print('Warning: The parameter elat has no error.')
				self.elat_err=0
		else:
			raise Exception('No pulsar position in par file.')
		#
		if 'F0' in paras_key:
			i=paras['F0']
			self.f0=np.float64(i[0])
			self.p0=1/self.f0
			if len(i)==2:
				self.f0_err=np.float64(i[1])
				self.p0_err=self.f0_err/self.f0**2
			else:
				print('Warning: The parameter f0 has no error.')
				self.f0_err=0
				self.p0_err=0
			self.paras.extend(['f0','p0'])
		elif 'P0' in paras_key:
			i=paras['P0']
			self.p0=np.float64(i[0])
			self.f0=1/self.p0
			if len(i)==2:
				self.p0_err=np.float64(i[1])
				self.f0_err=self.p0_err/self.p0**2
			else:
				print('Warning: The parameter p0 has no error.')
				self.f0_err=0
				self.p0_err=0
			self.paras.extend(['f0','p0'])
		else:
			raise Exception('No pulsar period in par file.')
		#
		if 'F1' in paras_key:
			i=paras['F1']
			self.f1=np.float64(i[0])
			self.p1=-self.f1*self.p0**2
			if len(i)==2:
				self.f1_err=np.float64(i[1])
				self.p1_err=np.sqrt((self.f1_err*self.p0**2)**2+(2*self.f1*self.p0*self.p0_err)**2)
			else:
				print('Warning: The parameter f1 has no error.')
				self.f1_err=0
				self.p1_err=0
			self.paras.extend(['f1','p1'])
		elif 'P1' in paras_key:
			i=paras['P1']
			self.p1=np.float64(i[0])
			self.f1=-self.p1*self.f0**2
			if len(i)==2:
				self.p1_err=np.float64(i[1])
				self.f1_err=np.sqrt((self.p1_err*self.f0**2)**2+(2*self.p1*self.f0*self.f0_err)**2)
			else:
				print('Warning: The parameter p1 has no error.')
				self.f1_err=0
				self.p1_err=0
			self.paras.extend(['f1','p1'])
		else:
			self.f1=0
			self.p1=0
		#
		if 'F2' in paras_key:
			if 'f1' not in self.paras:
				raise Exception('The parameter F2 is in parfile without F1.')
			i=paras['F2']
			self.f2=np.float64(i[0])
			self.p2=-self.f2*self.p0**2-2*self.f1*self.p0*self.p1
			if len(i)==2:
				self.f2_err=np.float64(i[1])
				self.p2_err=np.sqrt((self.f2_err*self.p0**2)**2+((self.f2*self.p0+self.f1*self.p1)*2*self.p0_err)**2+(2*self.f1*self.p0*self.p1_err)**2+(2*self.p0*self.p1*self.f1_err)**2)
			else:
				print('Warning: The parameter f2 has no error.')
				self.f2_err=0
				self.p2_err=0
			self.paras.extend(['f2','p2'])
		elif 'P2' in paras_key:
			if 'f1' not in self.paras:
				raise Exception('The parameter P2 is in parfile without P1.')
			i=paras['P2']
			self.p2=np.float64(i[0])
			self.f2=-self.p2*self.f0**2-2*self.p1*self.f0*self.f1
			if len(i)==2:
				self.p2_err=np.float64(i[1])
				self.f2_err=np.sqrt((self.p2_err*self.f0**2)**2+((self.p2*self.f0+self.p1*self.f1)*2*self.p0_err)**2+(2*self.p1*self.f0*self.f1_err)**2+(2*self.f0*self.f1*self.p1_err)**2)
			else:
				print('Warning: The parameter p2 has no error.')
				self.f2_err=0
				self.p2_err=0
			self.paras.extend(['f2','p2'])
		else:
			self.f2=0
			self.p2=0
		#
		if 'F3' in paras_key:
			if 'f2' not in self.paras:
				raise Exception('The parameter F3 is in parfile without F2.')
			i=paras['F3']
			self.f3=np.float64(i[0])
			self.p3=-self.f3*self.p0**2-4*self.f2*self.p0*self.p1-2*self.f1*self.p1**2-2*self.f1*self.p0*self.p2
			if len(i)==2:
				self.f3_err=np.float64(i[1])
				self.p3_err=np.sqrt((self.f3_err*self.p0**2)**2+((self.f3*self.p0+2*self.f2*self.p1+self.f1*self.p2)*2*self.p0_err)**2+((self.f2*self.p0+self.f1*self.p1)*4*self.p1_err)**2+(2*self.p0*self.p1*self.f2_err)**2+((self.p1**2+self.p0*self.p2)*2*self.f1_err)**2+(2*self.f1*self.p0*self.p2_err)**2)
			else:
				print('Warning: The parameter f3 has no error.')
				self.f3_err=0
				self.p3_err=0
			self.paras.extend(['f3','p3'])
		elif 'P3' in paras_key:
			if 'f2' not in self.paras:
				raise Exception('The parameter P3 is in parfile without P2.')
			i=paras['P3']
			self.p3=np.float64(i[0])
			self.f3=-self.p3*self.f0**2-4*self.p2*self.f0*self.f1-2*self.p1*self.f1**2-2*self.p1*self.f0*self.f2
			if len(i)==2:
				self.p3_err=np.float64(i[1])
				self.f3_err=np.sqrt((self.p3_err*self.f0**2)**2+((self.p3*self.f0+2*self.p2*self.f1+self.p1*self.f2)*2*self.f0_err)**2+((self.p2*self.f0+self.p1*self.f1)*4*self.f1_err)**2+(2*self.f0*self.f1*self.p2_err)**2+((self.f1**2+self.f0*self.f2)*2*self.p1_err)**2+(2*self.p1*self.f0*self.f2_err)**2)
			else:
				print('Warning: The parameter p3 has no error.')
				self.f3_err=0
				self.p3_err=0
			self.paras.extend(['f3','p3'])
		else:
			self.f3=0
			self.p3=0
		#
		self.deal_para('f4',paras,paras_key,err_case=['f3' not in self.paras],err_exc=['The parameter F4 is in parfile without F3.'])
		self.deal_para('f5',paras,paras_key,err_case=['f4' not in self.paras],err_exc=['The parameter F5 is in parfile without F4.'])
		#
		if ('f1' in self.paras) and ('PEPOCH' not in paras_key): raise Exception('No PEPOCH in par file.')
		self.deal_para('pepoch',paras,paras_key)
		#
		if 'DM' in paras_key:
			i=paras['DM']
			self.dm=np.float64(i[0])
			if len(i)>1:
				self.dm_err=np.float64(i[1])
			else:
				print('Warning: The parameter dm has no error.')
				self.dm_err=0
			self.paras.extend(['dm'])
			self.deal_para('dm1',paras,paras_key)
			self.deal_para('dm2',paras,paras_key,err_case=['dm1' not in self.paras],err_exc=['The parameter DM2 is in parfile without DM1.'])
			self.deal_para('dm3',paras,paras_key,err_case=['dm2' not in self.paras],err_exc=['The parameter DM3 is in parfile without DM2.'])
			self.dmmodel=0
			if 'DMX' in paras_key:
				i=paras['DM'].reshape(-1,4).T
				self.dmxr1=np.array(list(map(lambda x:te.time(np.float64(x),0,scale=self.units.lower()),np.reshape(i[0],-1))))
				self.dmxr2=np.array(list(map(lambda x:te.time(np.float64(x),0,scale=self.units.lower()),np.reshape(i[1],-1))))
				self.dmx=np.float64(i[2]).reshape(-1)
				self.dmx_err=np.float64(i[3]).reshape(-1)
				self.paras.extend(['dmx','dmxr1','dmxr2'])
			else:
				self.dmx=np.array([])
				self.dmxr1=np.array([])
				self.dmxr2=np.array([])
		elif 'DMMODEL' in paras_key:
			if 'DMOFF' not in paras_key:
				raise Exception('The DMMODEL para is in parfile without DMOFF.')
			i=paras['DMMODEL']
			self.dmmodel=np.float64(i[0])
			self.dmmodel_err=np.float64(i[1])
			self.paras.append('dmmodel')
			i=paras['DMOFF'].reshape(-1,3).T
			self.dmoffs_mjd=np.array(list(map(lambda x:te.time(np.float64(x),0,scale=self.units.lower()),np.reshape(i[0],-1))))
			self.dmoffs=np.float64(i[1]).reshape(-1)
			self.dmoffs_err=np.float64(i[2]).reshape(-1)
			self.paras.extend(['dmmodel','dmoffs','dmoffs_mjd'])
		else:
			self.dm=0
			self.dm1=self.dm2=self.dm3=0
			self.dmmodel=0
			self.dmx=np.array([])
			self.dmxr1=np.array([])
			self.dmxr2=np.array([])
			print('Strong Warning: No pulsar DM in par file, set DM as 0!')
		#
		self.deal_para('dm_s1yr',paras,paras_key,err_case=['DM_C1YR' not in paras_key],err_exc=['The parameter DM_S1YR is in parfile without DM_C1YR.'])
		self.deal_para('dm_c1yr',paras,paras_key,err_case=['DM_S1YR' not in paras_key],err_exc=['The parameter DM_C1YR is in parfile without DM_S1YR.'])
		self.deal_para('fddc',paras,paras_key,err_case=['FDDI' not in paras_key],err_exc=['The parameter FDDC is in parfile without FDDI.'])
		self.deal_para('fddi',paras,paras_key,err_case=['FDDC' not in paras_key],err_exc=['The parameter FDDI is in parfile without FDDC.'])
		self.deal_para('fd',paras,paras_key,err_case=['FDDI' not in paras_key],err_exc=['The parameter FD is in parfile without FDDI.'])
		self.deal_para('cm1',paras,paras_key,err_case=['CMIDX' not in paras_key],err_exc=['The parameter CM is in parfile without CMIDX.'])
		self.deal_para('cmidx',paras,paras_key,err_case=['CM' not in paras_key],err_exc=['The parameter CMIDX is in parfile without CM.'])
		self.deal_para('cm2',paras,paras_key,err_case=['cm1' not in self.paras],err_exc=['The parameter CM2 is in parfile without CM1.'])
		self.deal_para('cm3',paras,paras_key,err_case=['cm2' not in self.paras],err_exc=['The parameter CM3 is in parfile without CM2.'])
		self.deal_para('dmepoch',paras,paras_key,exce='Warning: No DMEPOCH in the parfile, using PEPOCH instead.',value=self.pepoch.mjd)
		self.deal_para('pmra',paras,paras_key,err_case=['PMDEC' not in paras_key],err_exc=['Strong Warning: The parameter PMRA is in parfile without PMDEC.'])
		self.deal_para('pmdec',paras,paras_key,err_case=['PMRA' not in paras_key],err_exc=['Strong Warning: The parameter PMDEC is in parfile without PMRA.'])
		self.deal_para('pmelong',paras,paras_key,err_case=['PMELAT' not in paras_key],err_exc=['Strong Warning: The parameter PMELONG is in parfile without PMELAT.'])
		self.deal_para('pmelat',paras,paras_key,err_case=['PMELONG' not in paras_key],err_exc=['Strong Warning: The parameter PMELAT is in parfile without PMELONG.'])
		self.deal_para('pmra2',paras,paras_key,err_case=['pmra' not in self.paras,'PMDEC2' not in paras_key],err_exc=['The parameter PM2 is in parfile without PM.','The parameter PMRA2 is in parfile without PMDEC2.'])
		self.deal_para('pmdec2',paras,paras_key,err_case=['pmdec' not in self.paras,'PMRA2' not in paras_key],err_exc=['The parameter PM2 is in parfile without PM.','The parameter PMDEC2 is in parfile without PMRA2.'])
		self.deal_para('pmelong2',paras,paras_key,err_case=['pmelong' not in self.paras,'PMELAT2' not in paras_key],err_exc=['The parameter PM2 is in parfile without PM.','The parameter PMELONG2 is in parfile without PMELAT2.'])
		self.deal_para('pmelat2',paras,paras_key,err_case=['pmelat' not in self.paras,'PMELONG2' not in paras_key],err_exc=['The parameter PM2 is in parfile without PM.','The parameter PMELAT2 is in parfile without PMELONG2.'])
		self.deal_para('pmrv',paras,paras_key)
		self.deal_para('px',paras,paras_key)
		self.deal_para('posepoch',paras,paras_key,exce='Warning: No POSEPOCH in the parfile, using PEPOCH instead.',value=self.pepoch.mjd)
		self.deal_para('binary',paras,paras_key)
		if self.binary:
			if self.binary=='BT2P' or self.binary=='BT1P': self.binary='T2'
			elif self.binary=='T2-PTA': self.binary='T2_PTA'
			for i in eval('paras_'+self.binary)['necessary']:
				self.deal_para(i,paras,paras_key, exce='Strong Warning: No '+i.upper()+' parameter for '+self.binary+'model.')
			for i in eval('paras_'+self.binary)['optional']:
				self.deal_para(i,paras,paras_key)
			if not (self.pb or self.fb0):
				raise Exception('The binary orbit period is not given in par file.')
			if self.binary in ['BTX', 'ELL1']:
				if not self.fb0:
					self.fb0=1/(self.pb*86400)
					self.fb0_err=self.pb_err*self.fb0**2/86400
					self.paras.append('fb0')
				if (not self.fb1) and self.binary=='BTX' and self.pbdot:
					self.fb1=-self.pbdot*self.fb0**2/86400**2
					self.fb1_err=np.sqrt((self.pbdot_err*self.fb0**2)**2+(2*self.pbdot*self.fb0*self.fb0_err)**2)/86400**2
					self.paras.append('fb1')
			else:
				if not self.pb:
					self.pb=1/(self.fb0*86400)
					self.pb_err=self.fb0_err*self.pb**2/86400
				if (not self.pbdot) and hasattr(self,'fb1'):
					self.pbdot=-self.fb1*self.pb**2/86400**2
					self.pbdot_err=np.sqrt((self.fb1_err*self.pb**2)**2+(2*self.fb1*self.pb*self.pb_err)**2)/86400**2
			if self.binary[:2]=='BT' or self.binary[:2]=='DD' or self.binary=='MSS':
				if not (('om' in self.paras) and ('t0' in self.paras)): 
					if ('tasc' in self.paras): raise Exception('There is no T0 and OM parameters for '+self.binary+' binary model, please use ELL type binary model.')
					else: raise Exception('There is no T0 and OM parameters for '+self.binary+' binary model.')
				if self.binary=='DD':
					if 'H3' in paras_key: print('Waring: The parameter H3 is found in parfile, maybe DDH model can be adopted.')
					if 'KOM' in paras_key: print('Waring: The parameter KOM is found in parfile, maybe DDK model can be adopted.')
				elif self.binary=='DDK':
					if 'KOM' not in paras_key: 
						self.binary='DD'
						self.sini=0
						print('Waring: The parameter KOM for DDK model is not found in parfile, use DD model instead.')
				elif self.binary=='DDH':
					if 'H3' not in paras_key: 
						self.binary='DD'
						self.sini=0
						self.m2=0
						print('Waring: The parameter H3 for DDH model is not found in parfile, use DD model instead.')
				elif self.binary=='DDGR':
					if 'M2' not in paras_key: 
						self.binary='DD'
						self.gamma=0
						self.omdot=0
						print('Waring: The parameter M2 for DDGR model is not found in parfile, use DD model instead.')
			if self.binary[:3]=='ELL':
				if not (self.tasc): 
					if (self.om and self.t0): raise Exception('There is no TASC parameter in parfile for ELL type binary model, please use another binary model.')
					else: raise Exception('There is no T0 and OM parameters for ELL type binary model.')
				if self.binary=='ELL1' or self.binary=='ELL1k':
					if 'H3' in paras_key: print('Waring: The parameter H3 is found in parfile, maybe ELL1H model can be adopted.')
				elif self.binary=='ELL1H':
					if 'H3' not in paras_key: 
						self.binary='ELL1k'
						print('Waring: The parameter H3 for ELL1H model is not found in parfile, use ELL1k model instead.')
			if self.binary=='T2':
				if not (('T0' in paras_key) or ('TASC' in paras_key)): 
					raise Exception('No T0 or T0ASC in pulsar paras.')
		self.deal_para('rm',paras,paras_key)
		self.deal_para('dshk',paras,paras_key)
		#
		if glitch:
			pwd0=sp.getoutput('which psrcat')
			pwd1='/'.join(pwd0.split('/')[:-1])
			glitchf=''
			if os.path.isfile(pwd1+'/glitch.db'):
				glitchf=pwd1+'/glitch.db'
			elif sp.getoutput('readlink '+pwd0):
				pwd2='/'.join(sp.getoutput('readlink '+pwd0).split('/')[:-1])
				if pwd2[0]=='.':
					pwd2=pwd1+'/'+pwd2
				if os.path.isfile(pwd2+'/glitch.db'):
					glitchf=pwd2+'/glitch.db'
			if glitchf:
				glitchfi=sp.getoutput('grep '+self.name+' '+glitchf)
				if glitchfi:
					def dealglitch(txt):
						if '(' in txt:
							txt,b=txt.split('(')
							b=b[:-1]
							if b.isdecimal():
								tmp=np.float64(txt.replace('.',''))/np.float64(txt)
								b=np.float64(b)/tmp
							elif b.replace('.','').isdecimal():
								b=np.float64(b)
							else:
								b=0
						else:
							b=0
						if txt.replace('.','').isdecimal():
							a=np.float64(txt)
						else:
							a=0
						return a,b
					glitchi=np.array(list(map(lambda x: x.split(),glitchfi.split('\n'))))[:,2:-1]
					nglitch=len(glitchi)
					gldata=np.zeros([nglitch,10])
					self.glf2=np.zeros(nglitch)
					self.glf2_err=np.zeros(nglitch)
					for i in np.arange(nglitch):
						gldata[i,0:2]=dealglitch(glitchi[i,0])
						gldata[i,2:4]=dealglitch(glitchi[i,1])
						gldata[i,4:6]=dealglitch(glitchi[i,2])
						gldata[i,6:8]=dealglitch(glitchi[i,3])
						gldata[i,8:10]=dealglitch(glitchi[i,4])
					self.paras.extend(['glep','glf0'])
					self.glep=te.time(gldata[:,0],np.zeros(nglitch),scale='tcb')
					self.glep_err=gldata[:,1]
					glf0=gldata[:,2]*1e9*self.f0
					glf0_err=gldata[:,3]*1e9*self.f0+glf0*self.f0_err
					self.glf1=gldata[:,4]*1e3*self.f1
					self.glf1_err=gldata[:,5]*1e3*self.f1+self.glf1*self.f1_err
					self.gltd=gldata[:,8]
					self.gltd_err=gldata[:,9]
					self.glf0d=gldata[:,6]*glf0
					self.glf0d_err=gldata[:,7]*glf0+gldata[:,6]*glf0_err
					self.glf0=glf0-self.glf0d
					self.glf0_err=np.sqrt(glf0_err**2+self.glf0d_err**2)
					if (gldata[:,4]!=0).sum()>0:
						self.paras.append('glf1')
					if (gldata[:,8]!=0).sum()>0:
						self.paras.extend(['glf0d','gltd'])
		else:
			self.deal_paralist('glitch',paras,paras_key)
		#
		self.deal_para('ephver',paras,paras_key,exce='Warning: No parameter version in the parfile.', value='2')
		self.deal_para('ephem',paras,paras_key,exce='Warning: No ephemris version in the parfile.', value='DE405')
		self.deal_para('clk',paras,paras_key,value='TT')
		self.unused_paras=set(paras_key)-set(map(lambda x: x.upper(),self.paras))-set(['PSRJ'])
		if len(self.unused_paras)>=2: print('Warning: The parameters '+', '.join(list(self.unused_paras))+' in the parfile are not used.')
		elif len(self.unused_paras)==1: print('Warning: The parameter '+', '.join(list(self.unused_paras))+' in the parfile is not used.')
	#
	def deal_para(self,paraname,paras,paras_key,exce=False,value=0,err_case=[],err_exc=[]):
		paraname0=paraname.upper()
		if (paraname in aliase_keys) and (paraname0 not in paras_key):
			for i in aliase[paraname]:
				i=i.upper()
				if i in paras_key: 
					self.paras.append(i.lower())
					paraname0=i
		if paraname0 in paras_key:
			for i,k in zip(err_case,err_exc): 
				if i: 
					if k[:15]=='Strong Warning:': print(k)
					else: raise Exception(k)
			i=paras[paraname0]
			if paraname in paras_float:
				i=np.array(np.float64(i)).reshape(-1)
			elif paraname in paras_float_array:
				i=np.float64(i).reshape(-1,2).T
			elif paraname in paras_time:
				if (paraname in para_with_err) and (len(i)==2): i=[te.time(np.float64(i[0]),0,scale=self.units.lower()),np.float64(i[1])]
				else: i=np.array([te.time(np.float64(i[0]),0,scale=self.units.lower())])
			elif paraname in paras_time_array:
				i=np.array([list(map(lambda x:te.time(np.float64(x),0,scale=self.units.lower()),np.reshape(i[0],-1)))])
			self.__setattr__(paraname,i[0])
			if paraname in para_with_err:
				if len(i)==2:
					self.__setattr__(paraname+'_err',i[1])
				else:
					print('Warning: The parameter '+paraname+' has no error.')
					self.__setattr__(paraname+'_err',0)
			self.paras.append(paraname)
		elif exce:
			if exce[:7]=='Warning' or exce[:15]=='Strong Warning:': 
				if (paraname in paras_float):
					self.__setattr__(paraname,0)
				elif (paraname in paras_time):
					self.__setattr__(paraname,te.time(value,0,scale=self.units.lower()))
				elif (paraname in paras_float_array) or (paraname in paras_time_array):
					self.__setattr__(paraname,np.array([]))
				elif paraname in paras_text:
					self.__setattr__(paraname,'')
				print(exce)
			else:
				raise Exception(exce)
		else: 
			if (paraname in paras_float):
				self.__setattr__(paraname,0)
			elif (paraname in paras_time):
				self.__setattr__(paraname,te.time(value,0,scale='tcb'))
			elif (paraname in paras_float_array) or (paraname in paras_time_array):
				self.__setattr__(paraname,np.array([]))
			elif paraname in paras_text:
				self.__setattr__(paraname,'')
	#
	def deal_paralist(self,paralist_name,paras,paras_key,listlimit=100,exce=False,value=0,err_case=[],err_exc=[]):
		paralist=eval('para_'+paralist_name)
		listindex=[]
		for i in np.arange(listlimit):
			if list(paralist)[0].upper()+'_'+str(i) in paras_key: listindex.append(str(i))
		listnum=len(listindex)
		if listnum==0: return
		for i in np.arange(len(paralist)):
			paraname=paralist[i]
			if paraname in paras_float_array: para=np.zeros(listnum)
			elif paraname in paras_time_array: para=np.zeros(listnum,dtype=te.time)
			para_err=np.zeros(listnum)
			para_count=0
			for k in np.arange(listnum):
				paraname0=paraname.upper()+'_'+listindex[k]
				if paraname0 in paras_key:
					data=np.array(np.float64(paras[paraname0])).reshape(-1)
					if paraname in paras_float_array: para[k]=data[0]
					elif paraname in paras_time_array: para[k]=te.time(data[0],0,scale=self.units.lower())
					if data.size==2: para_err[k]==data[1]
					elif paraname in para_with_err: print('Warning: The parameter '+paraname0+' has no error.')
					para_count+=1
			self.__setattr__(paraname,para)
			if paraname in para_with_err: self.__setattr__(paraname+'_err',para_err)
			if para_count: self.paras.append(paraname)
#
all_paras={'p0', 'pmra', 'pmdec', 'pmrv', 'f0', 'p2', 'p1', 'pmra2', 'pmdec2', 'f1', 'p3', 'f2', 'f3', 'f4', 'f5', 'dm', 'cm', 'dmmodel', 'dmoffs', 'dmx', 'dm_s1yr', 'dm_c1yr', 'fddc', 'fd', 'raj', 'decj', 'px' ,'cmidx', 'fddi', 'rm', 'pepoch', 'posepoch', 'dmepoch', 'dmoffs_mjd', 'dmxr1', 'dmxr2', 'name', 'dshk', 'binary','t0', 'pb', 'ecc', 'pbdot', 'a1dot', 'a1', 'omdot', 'om', 'gamma','bpjep','bpjph','bpja1','bpjec','bpjom','bpjpb', 'fb0', 'fb1', 'fb2', 'fb3', 'tasc', 'eps1', 'eps2', 'sini', 'm2', 'eps1dot', 'eps2dot', 'orbifunc', 'xpbdot', 'edot', 'kom', 'kin', 'mtot', 'a2dot', 'e2dot', 'orbpx', 'dr', 'dtheta', 'dth', 'a0', 'b0', 'om2dot', 'xomdot','afac','daop', 'pb2dot', 'ephver', 'ephem', 'dm1', 'dm2', 'dm3', 'cm1', 'cm2', 'cm3', 'glep', 'glph', 'gltd', 'glf0', 'glf1', 'glf2', 'glf0d', 'elong', 'elat', 'pmelong', 'pmelat', 'pmelong2', 'pmelat2'}
# with or without error
para_with_err={'p0', 'pmra', 'pmdec', 'pmrv', 'f0', 'p2', 'p1', 'pmra2', 'pmdec2', 'f1', 'p3', 'f2', 'f3', 'f4', 'f5', 'dm', 'cm', 'dmmodel', 'dmoffs', 'dmx', 'dm_s1yr', 'dm_c1yr', 'fddc', 'fd', 'raj', 'decj', 'px', 'cmidx', 'fddi', 'rm', 'dshk', 't0', 'pb', 'ecc', 'pbdot', 'a1dot', 'a1', 'omdot', 'om', 'gamma', 'fb0', 'fb1', 'fb2', 'fb3', 'bpjph','bpja1','bpjec','bpjom','bpjpb', 'tasc', 'eps1', 'eps2', 'sini', 'm2', 'eps1dot', 'eps2dot', 'orbifuncV','h3', 'h4','nharm','stig', 'xpbdot', 'edot', 'kom', 'kin', 'mtot', 'a2dot', 'e2dot', 'orbpx', 'dr', 'dtheta', 'dth', 'a0', 'b0', 'om2dot', 'glep', 'glph', 'gltd', 'glf0', 'glf1', 'glf2', 'glf0d', 'pmelong', 'pmelat', 'pmelong2', 'pb2dot', 'pmelat2', 'xomdot', 'shapmax'}
para_without_err={'pepoch','posepoch','dmepoch','dmoffs_mjd','dmxr1','dmxr2','name', 'binary', 'bpjep', 'orbifunc', 'orbifuncT', 'afac', 'daop', 'ephver', 'ephem'}
# tdb to tcb
paras_p1={'p0','dshk', 'pb', 'bpjpb', 'm2', 'h3', 'mtot', 'a0', 'b0', 'a1', 'gamma', 'gltd'}
paras_m1={'pmra','pmdec','pmrv','f0','p2', 'omdot', 'a1dot', 'eps1dot', 'eps2dot', 'edot', 'xomdot', 'pb2dot', 'fb0', 'dm', 'cm', 'px', 'glf0', 'glf0d', 'pmelong', 'pmelat'}
paras_m2={'pmra2','pmdec2','f1','p3', 'a2dot', 'e2dot', 'om2dot', 'fb1', 'dm1', 'cm1', 'glf1', 'pmelong2', 'pmelat2'}
paras_m3={'f2', 'fb2', 'dm2', 'cm2', 'glf2'}
paras_m4={'f3', 'fb3', 'dm3', 'cm3'}
paras_m5={'f4'}
paras_m6={'f5'}
paras_eph={'pepoch','posepoch','dmepoch','dmoffs_mjd','dmxr1','dmxr2', 't0', 'tasc', 'orbifuncT'}
uncertain_pm={'dmmodel','dmoffs','dmx','dm_s1yr','dm_c1yr','fddc','fd','rm'}
# para type
paras_float={'p0', 'pmra', 'pmdec', 'pmrv', 'f0', 'p2', 'p1', 'pmra2', 'pmdec2', 'f1', 'p3', 'f2', 'f3', 'f4', 'f5', 'dm','cm', 'dmmodel', 'dm_s1yr', 'dm_c1yr', 'fddc', 'raj', 'decj', 'px' ,'cmidx', 'fddi', 'rm', 'dshk',  'pb', 'ecc', 'pbdot', 'a1dot', 'a1', 'omdot', 'om', 'gamma', 'eps1', 'eps2', 'sini', 'm2', 'eps1dot', 'eps2dot', 'orbifunc','h3', 'h4','nharm','stig', 'xpbdot', 'edot', 'kom', 'kin', 'mtot', 'a2dot', 'e2dot', 'orbpx', 'dr', 'dtheta', 'dth', 'a0', 'b0', 'om2dot', 'xomdot','afac','daop', 'pb2dot', 'fb0', 'fb1', 'fb2', 'fb3', 'dm1', 'dm2', 'dm3', 'cm1', 'cm2', 'cm3', 'shapmax', 'pmelong', 'pmelat', 'pmelong2', 'pmelat2'}
paras_float_array={'dmx', 'dmoffs', 'fd','bpjph','bpja1','bpjec','bpjom','bpjpb', 'orbifuncV', 'glph', 'gltd', 'glf0', 'glf1', 'glf2', 'glf0d'}
paras_time={'pepoch', 'posepoch', 'dmepoch', 't0', 'tasc'}
paras_time_array={'dmoffs_mjd', 'dmxr1', 'dmxr2', 'bpjep', 'orbifuncT', 'glep'}
paras_text={'name', 'binary', 'ephver', 'ephem'}
# binary
paras_binary={'t0', 'pb', 'fb0', 'ecc', 'a1', 'om', 'fb1', 'fb2', 'fb3', 'tasc', 'eps1', 'eps2', 'shapmax', 'kom', 'kin', 'h3', 'stig', 'h4','nharm', 'm2', 'mtot', 'xpbdot', 'sini', 'pbdot', 'a1dot', 'omdot', 'gamma', 'bpjep', 'bpjph', 'bpja1', 'bpjec', 'bpjom', 'bpjpb', 'edot', 'orbpx', 'dr', 'dtheta', 'dth', 'a0', 'b0', 'eps1dot', 'eps2dot', 'xomdot','afac','daop', 'pb2dot', 'orbifunc', 'orbifuncT', 'orbifuncV', 'om2dot'}
paras_BT={'necessary':['t0', 'pb', 'om', 'ecc', 'a1'], 'optional':['pbdot', 'fb0', 'a1dot', 'omdot', 'gamma']}
paras_BTJ={'necessary':['t0', 'pb', 'om', 'ecc', 'a1','bpjep','bpjph','bpja1','bpjec','bpjom','bpjpb'], 'optional':['fb0', 'pbdot', 'a1dot', 'omdot', 'gamma']}
paras_BTX={'necessary':['t0', 'fb0', 'om', 'ecc', 'a1'], 'optional':['pb', 'fb1', 'fb2', 'fb3', 'pbdot', 'a1dot', 'omdot', 'gamma']}
paras_ELL1={'necessary':['tasc', 'fb0', 'eps1', 'eps2', 'a1'], 'optional':['pb', 'fb1', 'fb2', 'fb3', 'pbdot', 'sini', 'a1dot', 'm2', 'eps1dot', 'eps2dot', 'orbifunc', 'orbifuncT', 'orbifuncV']}
paras_ELL1H={'necessary':['tasc', 'pb', 'eps1', 'eps2', 'a1', 'h3'], 'optional':['pbdot', 'fb0', 'sini', 'a1dot', 'm2', 'eps1dot', 'eps2dot', 'h4','nharm','stig', 'omdot']}
paras_ELL1k={'necessary':['tasc', 'pb', 'eps1', 'eps2', 'a1'], 'optional':['pbdot', 'fb0', 'sini', 'a1dot', 'm2', 'omdot']}
paras_DD={'necessary':['t0', 'pb', 'ecc', 'om', 'a1'], 'optional':['pbdot', 'fb0', 'sini', 'omdot', 'a1dot', 'm2', 'xpbdot', 'edot', 'gamma']}
paras_DDH={'necessary':['t0', 'pb', 'ecc', 'om', 'a1', 'h3', 'stig'], 'optional':['pbdot', 'fb0', 'omdot', 'a1dot', 'xpbdot', 'edot', 'gamma']}
paras_DDK={'necessary':['t0', 'pb', 'ecc', 'om', 'a1', 'kom', 'kin'], 'optional':['pbdot', 'fb0', 'omdot', 'a1dot', 'm2', 'xpbdot', 'edot', 'gamma']}
paras_DDS={'necessary':['t0', 'pb', 'ecc', 'om', 'a1'], 'optional':['pbdot', 'fb0', 'shapmax', 'omdot', 'a1dot', 'm2', 'xpbdot', 'edot', 'gamma']}
paras_DDGR={'necessary':['t0', 'pb', 'ecc', 'om', 'a1'], 'optional':['pbdot', 'fb0', 'sini', 'a1dot', 'mtot', 'm2', 'xpbdot', 'edot']}
paras_MSS={'necessary':['t0', 'pb', 'ecc', 'a1','om'], 'optional':['pbdot', 'fb0', 'shapmax', 'a1dot', 'omdot', 'm2', 'sini', 'edot', 'gamma', 'a2dot', 'e2dot', 'orbpx', 'dr', 'dtheta', 'a0', 'b0', 'om2dot']}
paras_T2={'necessary':[],'optional':['t0', 'pb', 'fb0', 'ecc', 'a1', 'om', 'tasc', 'eps1', 'eps2', 'shapmax', 'kom', 'kin', 'h3', 'stig', 'h4','nharm', 'm2', 'mtot', 'xpbdot', 'sini', 'pbdot', 'a1dot', 'omdot', 'gamma', 'bpjep', 'bpjph', 'bpja1', 'bpjec', 'bpjom', 'bpjpb', 'edot', 'dr', 'dth', 'a0', 'b0', 'eps1dot', 'eps2dot', 'xomdot','afac','daop', 'pb2dot']}
# paras lists
para_glitch={'glep', 'glph', 'gltd', 'glf0', 'glf1', 'glf2', 'glf0d'}
#
aliase={'ecc':['e'],'a1dot':['xdot'],'a2dot':['x2dot','a12dot'],'daop':['d_aop'],'fb0':['fb'], 'dth':['dtheta'],'edot':['eccdot'],'cm1':['cm']}
aliase_keys=aliase.keys()

