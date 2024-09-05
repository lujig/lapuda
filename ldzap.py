#!/usr/bin/env python
import numpy as np
import numpy.ma as ma
import numpy.fft as fft
import argparse as ap
from matplotlib.figure import Figure
import matplotlib.lines as ln
import ld,os,copy,sys,shutil,time
import adfunc as af
import matplotlib.pyplot as plt
import psutil as pu
import asyncio as ai
import threading as th
plt.rcParams['font.family']='Serif'
dirname=os.path.split(os.path.realpath(__file__))[0]
sys.path.append(dirname+'/doc')
import text
#
text=text.output_text('ldzap')
version='JigLu_20240315'
parser=ap.ArgumentParser(prog='ldzap',description=text.help,epilog='Ver '+version,add_help=False,formatter_class=lambda prog: ap.RawTextHelpFormatter(prog, max_help_position=50))
parser.add_argument('-h', '--help', action='help', default=ap.SUPPRESS,help=text.help_h)
parser.add_argument('-v','--version',action='version',version=version,help=text.help_v)
parser.add_argument("-z","--zap",dest="zap_file",default=0,help=text.help_z)
parser.add_argument('-n',action='store_true',default=False,dest='norm',help=text.help_n)
parser.add_argument('-a',action='store_true',default=False,dest='mean',help=text.help_a)
parser.add_argument('-r',action='store_true',default=False,dest='redo',help=text.help_r)
parser.add_argument('-p',action='store_true',default=False,dest='dprof',help=text.help_p)
parser.add_argument("-o","--output",dest="output",help=text.help_o)
parser.add_argument("filename",help=text.help_filename)
args=(parser.parse_args())
#
class tree:
	def __init__(self,lddat,dat=0):
		self.ld=lddat
		info=self.ld.read_info()
		self.mem=mem
		if self.mem=='mem0': self.ftdat=self.ld.read_data(pol=0)[:,:,:,0]
		self.dat=dat
		self.nf,self.nt,self.nb=lddat.read_shape()[:3]
		self.weight=weight0
		self.zf,self.zt,self.zft=np.ones([3,self.nf,self.nt],dtype=bool)
		self.npref,self.nnextf=self.calpn(self.nf)
		self.npret,self.nnextt=self.calpn(self.nt)
		self.tsize=8*(self.nf-1)*self.nt*self.nb
		self.fsize=8*(self.nt-1)*self.nf*self.nb
		#self.ftsize=8*self.nt*self.nf*self.nb
		self.t=np.zeros([self.nt,self.nb])
		self.f=np.zeros([self.nf,self.nb])
		self.nstate=''
		self.nlevel=0
	#
	def calpn(self,n):
		npre=[]
		nnext=np.zeros(2*n-2,dtype=np.int32)
		last=[]
		tmp0,tmp1=n,0
		tmpc0=-n
		tmpc1=0
		for i in np.arange(-n,n-2):
			ind=int((i-tmpc0)//2)+tmpc1
			if ind>=len(npre):npre.append([])
			if i<tmpc1-1:
				nnext[i+n]=ind
				npre[ind].append(i)
			else:
				if tmp0%2==0:
					nnext[i+n]=ind
					npre[ind].append(i)
				else:
					if len(last)==1:
						nnext[i+n]=ind
						nnext[last[0]+n]=ind
						npre[ind].extend([last[0],i])
						last=[]
					else:
						last.append(i)
				tmp0,tmp1=np.divmod(tmp0+tmp1,2)
				tmpc0,tmpc1=tmpc1,tmpc1+tmp0
		return np.array(npre),nnext
	#
	def calpnft(self,nf,nt):
		npre=np.ones([self.nf,self.nt,2,2],dtype=np.int32)*max(self.nf,self.nt)
		nnext=np.zeros([self.nf,self.nt,2],dtype=np.int32)*max(self.nf,self.nt)
		last=[]
		tmp0,tmp1=n,0
		tmpc0=-n
		tmpc1=0
		for i in np.arange(-n,n-2):
			ind=int((i-tmpc0)//2)+tmpc1
			if ind>=len(npre):npre.append([])
			if i<tmpc1-1:
				nnext[i+n]=ind
				npre[ind].append(i)
			else:
				if tmp0%2==0:
					nnext[i+n]=ind
					npre[ind].append(i)
				else:
					if len(last)==1:
						nnext[i+n]=ind
						nnext[last[0]+n]=ind
						npre[ind].extend([last[0],i])
						last=[]
					else:
						last.append(i)
				tmp0,tmp1=np.divmod(tmp0+tmp1,2)
				tmpc0,tmpc1=tmpc1,tmpc1+tmp0
		return np.array(npre),nnext
	#
	def nchange(self,j0,n0,nnext):
		ind0=np.arange(n0)[j0>0]
		ind1=list(nnext[ind0])
		ind=[]
		while ind1:
			tmp=ind1.pop(0)
			if tmp<n0-2:
				ind1.append(nnext[tmp+n0])
			ind.append(tmp)
		return np.sort(np.unique(ind))
	#
	def gent(self,z0=0,field='freq',first=False):
		if (0).__eq__(z0) is True: z1=np.zeros([self.nf,self.nt],dtype=bool)
		else: z1=np.array(z0)
		if first: jjmat=z1!=True
		else: jjmat=self.zt!=z1
		jjf=jjmat.sum(1)>0
		jjt=jjmat.sum(0)>0
		if jjf.sum()==0: return
		if field=='time':
			zf=(z1==0).sum(1)
			clist=self.nchange(jjf,self.nf,self.nnextf)
			for i in clist:
				c0,c1=self.npref[i]
				if c0<0:
					if zf[c0+self.nf]==0: d0=np.zeros([self.nt,self.nb])
					elif self.mem=='mem0': d0=self.ftdat[c0+self.nf]*self.weight[c0+self.nf,:,np.newaxis]
					else:
						d0=self.ld.read_chan(c0+self.nf,pol=0)[:,:,0]*self.weight[c0+self.nf,:,np.newaxis]
						d0[z1[c0+self.nf]]=0
				elif self.dat==0: d0=self.t0[c0]
				else:
					t=np.memmap(self.dat,dtype=np.float64,mode='r',shape=(self.nf-1,self.nt,self.nb))
					d0=np.array(t[c0])
					del t
				if c1<0:
					if zf[c1+self.nf]==0: d1=np.zeros([self.nt,self.nb])
					elif self.mem=='mem0': d1=self.ftdat[c1+self.nf]*self.weight[c1+self.nf,:,np.newaxis]
					else:
						d1=self.ld.read_chan(c1+self.nf,pol=0)[:,:,0]*self.weight[c1+self.nf,:,np.newaxis]
						d1[z1[c1+self.nf]]=0
				elif self.dat==0: d1=self.t0[c1]
				else:
					t=np.memmap(self.dat,dtype=np.float64,mode='r',shape=(self.nf-1,self.nt,self.nb))
					d1=np.array(t[c1])
					del t
				if not hasattr(self,'t0'): self.t0=np.zeros([self.nf-1,self.nt,self.nb])
				if self.dat==0: self.t0[i]=d0+d1
				else:
					t=np.memmap(self.dat,dtype=np.float64,mode='r+',shape=(self.nf-1,self.nt,self.nb))
					t[i]=d0+d1
					del t
		else:
			slist=np.arange(self.nt)[jjt>0]
			clist=self.nchange(jjf,self.nf,self.nnextf)
			for i in slist:
				if (z1[:,i]==0).sum()!=0:
					if self.mem=='mem0': da=self.ftdat[:,i]*self.weight[:,i,np.newaxis]
					else: da=self.ld.read_period(i,pol=0)[:,:,0]*self.weight[:,i,np.newaxis]
					da[z1[:,i]!=0]=0
				else: da=np.zeros([self.nf,self.nb])
				if not hasattr(self,'t0'): self.t0=np.zeros([self.nf-1,self.nt,self.nb])
				if self.dat: t=np.memmap(self.dat,dtype=np.float64,mode='r+',shape=(self.nf-1,self.nt,self.nb))
				for k in clist:
					c0,c1=self.npref[k]
					if c0<0: d0=da[c0+self.nf]
					else:
						if self.dat: d0=t[c0,i]
						else: d0=self.t0[c0,i]
					if c1<0: d1=da[c1+self.nf]
					else:
						if self.dat: d1=t[c1,i]
						else: d1=self.t0[c1,i]
					if self.dat==0: self.t0[k,i]=d0+d1
					else: t[k,i]=d0+d1
				if self.dat: del t
		self.zt=z1
		if self.dat:
			t=np.memmap(self.dat,dtype=np.float64,mode='r',shape=(self.nf-1,self.nt,self.nb))
			self.t=np.array(t[-1])
			del t
		else: self.t=self.t0[-1]
		if not hasattr(self,'tnoise'): self.tnoise=np.min(noiselevel(self.t),axis=1)*np.ones(self.nf).reshape(-1,1)
	#
	def genf(self,z0=0,field='freq',first=False):
		if (0).__eq__(z0) is True: z1=np.zeros([self.nf,self.nt],dtype=bool)
		else: z1=np.array(z0)
		if first: jjmat=z1!=True
		else: jjmat=self.zf!=z1
		jjf=jjmat.sum(1)>0
		jjt=jjmat.sum(0)>0
		if jjf.sum()==0: return
		#if not hasattr(self,'ft'): ft=np.zeros([self.nf,self.nt])
		if field=='time':
			zt=(z1==0).sum(0)
			slist=self.nchange(jjt,self.nt,self.nnextt)
			for i in slist:
				c0,c1=self.npret[i]
				if c0<0:
					if zt[c0+self.nt]==0: d0=np.zeros([self.nf,self.nb])
					elif self.mem=='mem0': d0=self.ftdat[:,c0+self.nt]*self.weight[:,c0+self.nt,np.newaxis]
					else:
						d0=self.ld.read_period(c0+self.nt,pol=0)[:,:,0]*self.weight[:,c0+self.nt,np.newaxis]
						d0[z1[:,c0+self.nt]]=0
					#if not hasattr(self,'ft'): ft[:,c0+self.nt]=d0.sum(1)
				elif self.dat==0: d0=self.f0[:,c0]
				else:
					f=np.memmap(self.dat,dtype=np.float64,mode='r',shape=(self.nf,self.nt-1,self.nb),offset=self.tsize)
					d0=np.array(f[:,c0])
					del f
				if c1<0:
					if zt[c1+self.nt]==0: d1=np.zeros([self.nf,self.nb])
					elif self.mem=='mem0': d1=self.ftdat[:,c1+self.nt]*self.weight[:,c1+self.nt,np.newaxis]
					else:
						d1=self.ld.read_period(c1+self.nt,pol=0)[:,:,0]*self.weight[:,c0+self.nt,np.newaxis]
						d1[z1[:,c1+self.nt]]=0
					#if not hasattr(self,'ft'): ft[:,c1+self.nt]=d1.sum(1)
				elif self.dat==0: d1=self.f0[:,c1]
				else:
					f=np.memmap(self.dat,dtype=np.float64,mode='r',shape=(self.nf,self.nt-1,self.nb),offset=self.tsize)
					d1=np.array(f[:,c1])
					del f
				if not hasattr(self,'f0'): self.f0=np.zeros([self.nf,self.nt-1,self.nb])
				if self.dat==0: self.f0[:,i]=d0+d1
				else:
					f=np.memmap(self.dat,dtype=np.float64,mode='r+',shape=(self.nf,self.nt-1,self.nb),offset=self.tsize)
					f[:,i]=d0+d1
					del f
		else:
			clist=np.arange(self.nf)[jjf]
			slist=self.nchange(jjt,self.nt,self.nnextt)
			for i in clist:
				if (z1[i]==0).sum()!=0:
					if self.mem=='mem0': da=self.ftdat[i]*self.weight[i,:,np.newaxis]
					else: da=self.ld.read_chan(i,pol=0)[:,:,0]*self.weight[i,:,np.newaxis]
					da[z1[i]!=0]=0
				else: da=np.zeros([self.nt,self.nb])
				if not hasattr(self,'f0'): self.f0=np.zeros([self.nf,self.nt-1,self.nb])
				if self.dat: f=np.memmap(self.dat,dtype=np.float64,mode='r+',shape=(self.nf,self.nt-1,self.nb),offset=self.tsize)
				for k in slist:
					c0,c1=self.npret[k]
					if c0<0:
						d0=da[c0+self.nt]
						#if not hasattr(self,'ft'): ft[i,c0+self.nt]=d0.sum()
					else:
						if self.dat: d0=f[i,c0]
						else: d0=self.f0[i,c0]
					if c1<0:
						d1=da[c1+self.nt]
						#if not hasattr(self,'ft'): ft[i,c1+self.nt]=d1.sum()
					else:
						if self.dat: d1=f[i,c1]
						else: d1=self.f0[i,c1]
					if self.dat: f[i,k]=d0+d1
					else: self.f0[i,k]=d0+d1
				if self.dat: del f
		self.zf=z1
		if self.dat:
			f=np.memmap(self.dat,dtype=np.float64,mode='r',shape=(self.nf,self.nt-1,self.nb),offset=self.tsize)
			self.f=np.array(f[:,-1])
			del f
		else: self.f=self.f0[:,-1]
		if not hasattr(self,'ft'): 
			profile=self.f.mean(0)
			base_nbin=int(self.nb/10)
			base,bin0=af.baseline(profile,pos=True)
			basebin=np.sort(np.arange(self.nb).reshape(1,-1).repeat(2,axis=0).reshape(-1)[bin0:(bin0+base_nbin)])
			self.ft=np.ma.masked_where(self.weight==0,self.ld.bin_scrunch(select_bin=basebin.tolist(),pol=0)[:,:,0])
			self.dmod='off-pulse'
		if not hasattr(self,'fnoise'):
			self.fnoise=np.min(noiselevel(self.f),axis=1).reshape(-1,1)*np.ones(self.nt)
			self.ftnoise=noiselevel(self.ft)
	#
	def genft(self,z0=0,field='',first=False):
		if (0).__eq__(z0) is True: z1=np.zeros([self.nf,self.nt],dtype=bool)
		else: z1=np.array(z0)
		self.zft=z1
		if first:
			self.dmod=first
			profile=self.f.mean(0)
			if self.dmod=='on-pulse': selbin=af.radipos(profile)
			elif self.dmod=='off-pulse':
				base_nbin=int(self.nb/10)
				base,bin0=af.baseline(profile,pos=True)
				selbin=np.sort(np.arange(self.nb).reshape(1,-1).repeat(2,axis=0).reshape(-1)[bin0:(bin0+base_nbin)])
			self.ft=np.ma.masked_where(self.zf,self.ld.bin_scrunch(select_bin=selbin.tolist(),pol=0)[:,:,0])
			self.ftnoise=noiselevel(self.ft)
			self.ft.mask=z1
		else: self.ft.mask=z1
	#
	def ftprof(self,z0=0):
		if not hasattr(self,'profile'): self.profile=self.f.sum(0)
		if (0).__eq__(z0) is True: z1=np.zeros([self.nf,self.nt],dtype=bool)
		else: z1=np.array(z0)
		jjmat0=z1&np.logical_not(self.zp)
		jjmat1=self.zp&np.logical_not(z1)
		if self.mem=='mem0':
			profile0=(self.ftdat[jjmat0]*self.weight[jjmat0,np.newaxis]).sum(0)
			if profile0.size==0: profile0=np.zeros(self.nb)
			profile1=(self.ftdat[jjmat1]*self.weight[jjmat1,np.newaxis]).sum(0)
			if profile1.size==0: profile1=np.zeros(self.nb)
		else:
			jjf0=jjmat0.sum(1)>0
			jjf1=jjmat1.sum(1)>0
			profile0=np.zeros(self.nb)
			for i in np.arange(self.nf)[jjf0]:
				subs0=np.arange(self.nt)[jjmat0[i]]
				profile0+=(self.ld.read_data(select_chan=[i],start_period=subs0[0],end_period=subs0[-1]+1,pol=0)[0,subs0-subs0[0],:,0]*self.weight[i,subs0,np.newaxis]).sum(0)
			profile1=np.zeros(self.nb)
			for i in np.arange(self.nf)[jjf1]:
				subs1=np.arange(self.nt)[jjmat1[i]]
				profile1+=(self.ld.read_data(select_chan=[i],start_period=subs1[0],end_period=subs1[-1]+1,pol=0)[0,subs1-subs1[0],:,0]*self.weight[i,subs1,np.newaxis]).sum(0)
		self.zp=z1
		self.profile+=-profile0+profile1
		return self.profile
#
# read data
if not os.path.isfile(args.filename):
	parser.error('The input file is unexist.')
d=ld.ld(args.filename)
info=d.read_info()
#
if args.output:
	name=args.output
	if len(name)>3:
		if name[-3:]=='.ld':
			name=name[:-3]
	if os.path.isfile(name+'.ld'):
		tmp=1
		name0=name+'_'+str(tmp)
		while os.path.isfile(name0+'.ld'):
			name0=name+'_'+str(tmp)
			tmp+=1
		name=name0
#
# read the frequently used infomation 
nchan=int(info['data_info']['nchan'])
nbin=int(info['data_info']['nbin'])
nsub=int(info['data_info']['nsub'])
npol=int(info['data_info']['npol'])
length=float(info['data_info']['length'])
if 'weights' in info['data_info'].keys(): weight='weights'
else: weight='chan_weight'
#
max_noiselevel=5
def noiselevel(data):
	nx,ny=data.shape
	data=data.reshape(-1)
	ind=np.argsort(data)
	data=data[ind]
	tmp=np.diff(data)
	tmpind=np.argsort(tmp)
	tmpsort=tmp[tmpind]
	crit=tmpsort[int(data.size*0.25):int(data.size*0.75)].mean()+tmpsort[int(data.size*0.25):int(data.size*0.75)].std()*1000*10**np.arange(max_noiselevel).reshape(-1,1)
	level=max_noiselevel-(tmp>crit).sum(0)
	level=np.concatenate((level[:int(level.size/2)],[5],level[int(level.size/2):]))
	level1=level.copy()
	level1[:int(level.size/2)]=max_noiselevel
	level[int(level.size/2):]=max_noiselevel
	critn=np.zeros([2,max_noiselevel],dtype=np.int32)
	for i in max_noiselevel-np.arange(max_noiselevel)-1:
		ctmp0=np.where(level1==i)[0]
		ctmp1=np.where(level==i)[0]
		if ctmp0.size>0: critn[0,i]=ctmp0.min()
		if ctmp1.size>0: critn[1,i]=ctmp1.max()
	for i in max_noiselevel-np.arange(max_noiselevel)-1:
		if critn[0,i]>0: level1[critn[0,i]:]=i
		if critn[1,i]>0: level[:critn[1,i]]=i
	level[int(level.size/2):]=level1[int(level.size/2):]
	return level[np.argsort(ind)].reshape(nx,ny)
#
if args.redo:
	if 'chan_weight_raw' in info['additional_info'].keys():
		weight0=np.reshape(info['additional_info']['chan_weight_raw'],(-1,1))*np.ones([1,nsub])
	else:
		weight0=np.reshape(info['data_info']['chan_weight'],(-1,1))*np.ones([1,nsub])
	weight0[:,-1]*=(info['data_info']['sub_nperiod_last']/info['data_info']['sub_nperiod'])
else:
	if 'weights' in info['data_info'].keys(): weight0=np.array(info['data_info']['weights'])
	else:
		weight0=np.reshape(info['data_info']['chan_weight'],(-1,1))*np.ones([1,nsub])
		weight0[:,-1]*=(info['data_info']['sub_nperiod_last']/info['data_info']['sub_nperiod'])
#
cali=False
calmark=False
if 'calibration_info' in info.keys():
	nchan0=info['data_info']['nchan']
	if info['calibration_info']['cal_mode']=='single':
		cal=np.array(info['calibration_info']['cal'])[0].T
	else:
		noisedt=info['data_info']['stt_time']+info['data_info']['length']/2/86400-info['calibration_info']['noise_time0']
		cal=np.polyval(np.reshape(info['calibration_info']['cal'],(2,4,nchan0)),noisedt).T
	if cal.shape[0]==nchan0:
		calmark=True
#
fd=pu.disk_usage('.').free
fm=pu.virtual_memory().available
if nchan*nsub*nbin*38<fm: mem='mem0'
elif nchan*nsub*nbin*30<fm: mem='mem'
elif nchan*nsub*nbin*25<fd: mem='disk'
else: mem='none'
#
if mem!='mem0' and args.dprof:
	print(text.warning_longt)
#
zapn=weight0==0
if mem!='none':
	if mem=='disk':
		datname=uuid.uuid4().hex+'_tmp.dat'
		mtree=tree(d,datname)
		dat=np.memmap(datname,dtype=np.int8,mode='w+',shape=(mtree.fsize+mtree.tsize,))
		del dat
	elif mem[:3]=='mem':
		mtree=tree(d)
	mtree.genf(z0=zapn,field='freq',first=True)
#
zaplist=[]
state='freq'
#
async def gent(z0,field,first):
	mtree.gent(z0,field,first)
#
async def genf(z0,field,first):
	mtree.genf(z0,field,first)
#
async def genft(z0,field,first):
	mtree.genft(z0,field,first)
#
async def task0():
	global tlist
	while state:
		if tlist:
			tlist[0].append('working')
			func,z0,field,first,_=*(tlist[0]),
			await func(z0=z0,field=field,first=first)
			tlist.remove(tlist[0])
		else:
			await ai.sleep(0.02)
#
def task():
	ai.run(task0())
#
if mem!='none':
	tlist=[[gent,zapn,'freq',True]]
	threading=th.Thread(target=task,daemon=True)
	threading.start()
#
# get the information of frequecny domain
freq_start,freq_end=info['data_info']['freq_start'],info['data_info']['freq_end']
flim0=[freq_start,freq_end]
channelwidth=(freq_end-freq_start)/nchan
freq0=freq_start
#
# get the information of time domain
tsub=length/nsub
tlim0=[0,length]
#
def init(state0):
	global data,data0,data1,pxlim,pylim,yaxis,zapn0,spec0,yaxis0,profile,state,limlist,limn,dmod
	if mem!='none':
		if state0=='freq': func=genf
		elif state0=='time': func=gent
		elif state0=='dyn': func=genft
		else: return
		while True:
			mark=True
			for i in tlist:
				if func==i[0]: mark=False
			if mark: break
			else:
				state='busy'
				time.sleep(0.01)
	state=state0
	limlist=[]
	limn=0
	fig.lines=[ly,lx]
	lx.set_xdata([0,0])
	if state in ['freq','time']:
		if state=='freq':
			zapn0=np.logical_not((zapn==0).sum(1)>0)
			if mem=='none':
				data=np.zeros([nchan,nbin])
				for i in np.arange(nchan):
					if zapn0[i]: continue
					dtmp=d.read_chan(i,pol=0)[:,:,0]*weight0[i,np.newaxis]
					dtmp[zapn[i]]=0
					data[i]=dtmp.sum(1)
			else: data=np.ma.masked_where(False,mtree.f)
			pylim=flim0
			yaxis=np.arange(*flim0,channelwidth)
		else:
			zapn0=np.logical_not((zapn==0).sum(0)>0)
			zapnf=np.logical_not((zapn==0).sum(1)>0)
			if mem=='none':
				data=np.zeros([nsub,nbin])
				for i in np.arange(nchan):
					if zapnf[i]: continue
					dtmp=d.read_period(i,pol=0)[:,:,0]*weight0[i,np.newaxis]
					dtmp[zapn[i]]=0
					data+=dtmp
			else: data=np.ma.masked_where(False,mtree.t)
			pylim=tlim0
			yaxis=np.arange(*tlim0,tsub)
		pxlim=[0,1]
		base_nbin=int(nbin/10)
		profile=data.sum(0)
		base,bin0=af.baseline(profile,pos=True)
		data1=np.concatenate((data,data),axis=1)[:,bin0:(bin0+base_nbin)]
		data0=data-data1.mean(1).reshape(-1,1)
		profile-=base
		data=copy.deepcopy(data0)
		if args.norm:
			std=data1.std(1)
			data[std==0]=0
			data[std!=0]=data0[std!=0]/std[std!=0].reshape(-1,1)
		data.mask=np.array([zapn0]).T.repeat(nbin,axis=1)
		if state=='freq' and cali:
			cal0=copy.deepcopy(cal)
			cal0[zapn0]=0
			spec0=np.concatenate((np.zeros([1,4]),cal0.repeat(2,axis=0),np.zeros([1,4])),axis=0)
		else:
			if args.mean:
				spec=data1.mean(1)
			else: spec=data1.std(1)
			spec=spec.data
			spec=spec-np.min(spec)
			spec[zapn0]=0
			#profile=data.sum(0)
			spec0=np.append(0,np.append(spec.repeat(2),0))
		yaxis0=np.linspace(pylim[0],pylim[1],data0.shape[0]+1).repeat(2)
		if hasattr(mtree,'profile'): mtree.__delattr__('profile')
	else:
		if mem=='none':
			dmod='off-pulse'
			profiletmp=np.zeros(nbin)
			for i in np.arange(nchan):
				dtmp=d.read_chan(i,pol=0)[:,:,0]*weight0[i,np.newaxis]
				dtmp[zapn[i]]=0
				profiletmp+=dtmp.sum(1)
			base_nbin=int(nbin/10)
			base,bin0=af.baseline(profiletmp,pos=True)
			selbin=np.sort(np.arange(nbin).reshape(1,-1).repeat(2,axis=0).reshape(-1)[bin0:(bin0+base_nbin)])
			data=d.bin_scrunch(select_bin=selbin.tolist(),pol=0)[:,:,0]
			data[zapn]=0
		else:
			data=mtree.ft
			mtree.zp=zapn.copy()
		zapn0=zapn
		data=np.ma.masked_where(zapn0,data)
		pxlim=tlim0
		pylim=flim0
	#
	limlist.append([[pxlim[0],pylim[0]],[pxlim[1],pylim[1]]])
	plotimage(pxlim,pylim)
#
# creat a figure
fig=Figure(figsize=(120,60),dpi=80)
fig.clf()
colormap='jet'
# creat four windows 
x0,x1,x2=0.1,0.50,0.61
y0,y1=0.11,0.96
ax=fig.add_axes([x0,y0,x1-x0,y1-y0])
ax1=fig.add_axes([x1,y0,x2-x1,y1-y0])
x3,x4=0.65,0.97
yt1,yt2=0.48,0.55
axs=fig.add_axes([x3,y0,x4-x3,yt1-y0])
axp=fig.add_axes([x3,yt2,x4-x3,y1-yt2])
axs.set_xlabel('Phase',fontsize=20)
axs.set_yticks([])
ly=ln.Line2D([0,(x2+x3)/2],[0.5,0.5],color='k',transform=fig.transFigure,figure=fig)
lx=ln.Line2D([0,0],[0,1],color='k',transform=fig.transFigure,figure=fig)
fig.lines.extend([ly,lx])
#
def plotimage(xlim,ylim):
	if state not in ['freq','time','dyn']: return
	ax.cla()
	ax1.cla()
	ax.imshow(data,origin='lower',aspect='auto',interpolation='nearest',extent=(pxlim[0],pxlim[1],pylim[0],pylim[1]),cmap=colormap)
	ax.set_ylim(ylim[0],ylim[1])
	ax.set_xlim(xlim[0],xlim[1])
	if state in ['freq','time']:
		if state=='freq' and cali:
			ax1.plot(spec0,yaxis0,'-')
			ax1.set_xlim(np.min(spec0)*1.1,np.max(spec0)*1.1)
		else:
			ax1.plot(spec0,yaxis0,'k-')
			ax1.set_xlim(0,np.max(spec0)*1.1)
		ax1.set_ylim(ylim[0],ylim[1])
		ax.set_xlabel('Pulse Phase',fontsize=20)
		if state=='freq':
			ax.set_ylabel('Frequency (MHz)',fontsize=20)
		else:
			ax.set_ylabel('Time (s)',fontsize=20)
	else:
		ax.set_ylabel('Frequency (MHz)',fontsize=20)
		ax.set_xlabel('Time (s)',fontsize=20)
	axp.cla()
	axp.set_xlabel('Phase',fontsize=20)
	if state in ['freq','time'] or args.dprof or mem=='mem0':
		base=af.baseline(profile,pos=False)
		profile0=profile-base
		if profile0.max()>0: profile0/=profile0.max()
		axp.plot(np.arange(nbin)/nbin,profile0,'k')
	ax1.set_xticks([])
	ax1.set_yticks([])
	canvas.draw()
#
def leftclick(event):
	global limn,limlist
	if state not in ['freq','time','dyn']: return
	if state in ['freq','time']:
		if event.x<ax.bbox.extents[0] or event.x>ax1.bbox.extents[2] or event.y<fig.bbox.extents[1] or event.y>fig.bbox.extents[3]: return
	elif state=='dyn':
		if event.x<fig.bbox.extents[0] or event.x>ax1.bbox.extents[2] or event.y<fig.bbox.extents[1] or event.y>fig.bbox.extents[3]: return
	if limn==0: # click to choose a channel
		y=(fig.bbox.extents[3]-event.y)/fig.bbox.extents[3]
		ly2=ln.Line2D([0,(x2+x3)/2],[y,y],color='k',transform=fig.transFigure,figure=fig)
		fig.lines.append(ly2)
		if state=='dyn':
			x=event.x/fig.bbox.extents[2]
			lx2=ln.Line2D([x,x],[0,1],color='k',transform=fig.transFigure,figure=fig)
			fig.lines.append(lx2)
			limn=[xcal(event.x),ycal(event.y)]
		else: limn=[0,ycal(event.y)]
		canvas.draw()
	else:   # click again to zoom in
		if state in ['freq','time']:
			ylim=np.sort([limn[1],ycal(event.y)])
			limlist.append([[0,ylim[0]],[1,ylim[1]]])
			ax.set_ylim(ylim[0],ylim[1])
			ax1.set_ylim(ylim[0],ylim[1])
		else:
			xlim=np.sort([limn[0],xcal(event.x)])
			ylim=np.sort([limn[1],ycal(event.y)])
			limlist.append([[xlim[0],ylim[0]],[xlim[1],ylim[1]]])
			ax.set_xlim(xlim[0],xlim[1])
			ax.set_ylim(ylim[0],ylim[1])
			ax1.set_ylim(ylim[0],ylim[1])
		fig.lines=[ly,lx]
		canvas.draw()
		limn=0
#
def rightclick(event):
	global limn,zaplist,zapn,zapn0
	if state not in ['freq','time','dyn']: return
	if state in ['freq','time']:
		if event.x<ax.bbox.extents[0] or event.x>ax1.bbox.extents[2] or event.y<fig.bbox.extents[1] or event.y>fig.bbox.extents[3]: return
	elif state=='dyn':
		if event.x<fig.bbox.extents[0] or event.x>ax1.bbox.extents[2] or event.y<fig.bbox.extents[1] or event.y>fig.bbox.extents[3]: return
	py=ycal(event.y)
	if state=='freq': ybin=min(max(0,chancal(py)),nchan-1)
	elif state=='time': ybin=min(max(0,subcal(py)),nsub-1)
	else:
		ybin=min(max(0,chancal(py)),nchan-1)
		xbin=min(max(0,subcal(xcal(event.x))),nsub-1)
	if limn==0: # click to remove one channel
		if state=='freq':
			zaplist.append([state,ybin,ybin+1,0,nsub])
			zapn0[ybin]=True
			zapn[ybin]=True
		elif state=='time':
			zaplist.append([state,0,nchan,ybin,ybin+1])
			zapn0[ybin]=True
			zapn[:,ybin]=True
		else:
			zaplist.append([state,ybin,ybin+1,xbin,xbin+1])
			zapn[ybin,xbin]=True
	else:   # click after a left click to remove the region between two clicks
		if state=='freq':
			ybin0=min(max(0,chancal(limn[1])),nchan-1)
			ybin,ybin0=np.sort([ybin,ybin0])
			zaplist.append([state,ybin,ybin0+1,0,nsub])
			zapn0[ybin:(ybin0+1)]=True
			zapn[ybin:(ybin0+1)]=True
		elif state=='time':
			ybin0=min(max(0,subcal(limn[1])),nsub-1)
			ybin,ybin0=np.sort([ybin,ybin0])
			zaplist.append([state,0,nchan,ybin,ybin0+1])
			zapn0[ybin:(ybin0+1)]=True
			zapn[:,ybin:(ybin0+1)]=True
		else:
			ybin0=min(max(0,chancal(limn[1])),nchan-1)
			xbin0=min(max(0,subcal(limn[0])),nsub-1)
			ybin,ybin0=np.sort([ybin,ybin0])
			xbin,xbin0=np.sort([xbin,xbin0])
			zaplist.append([state,ybin,ybin0+1,xbin,xbin0+1])
			zapn[ybin:(ybin0+1),xbin:(xbin0+1)]=True
		limn=0
		fig.lines=[lx,ly]
	update_image()
#
def update_image(state0=state):	# update the parameters to plot the figure
	global limlist,spec0,data,data0,profile,tlist,spec,state,data1
	if mem=='none':
		if state0!=state and state in ['freq','time']:
			if state=='freq':
				data=np.zeros([nchan,nbin])
				for i in np.arange(nchan):
					if zapn0[i]: continue
					dtmp=d.read_chan(i,pol=0)[:,:,0]*weight0[i,np.newaxis]
					dtmp[zapn[i]]=0
					data[i]=dtmp.sum(1)
			else:
				zapnf=np.logical_not((zapn==0).sum(1)>0)
				data=np.zeros([nsub,nbin])
				for i in np.arange(nchan):
					if zapnf[i]: continue
					dtmp=d.read_period(i,pol=0)[:,:,0]*weight0[i,np.newaxis]
					dtmp[zapn[i]]=0
					data+=dtmp
			base_nbin=int(nbin/10)
			base,bin0=af.baseline(profile,pos=True)
			data1=np.concatenate((data,data),axis=1)[:,bin0:(bin0+base_nbin)]
			data0=data-data1.mean(1).reshape(-1,1)
			data=copy.deepcopy(data0)
			if args.norm:
				std=data1.std(1)
				data[std==0]=0
				data[std!=0]=data0[std!=0]/std[std!=0].reshape(-1,1)
	else:
		if state==state0:
			if state=='freq': tl0=[[gent,zapn,state0,False],[genf,zapn,state0,False],[genft,zapn,state0,False]]
			elif state=='time': tl0=[[genf,zapn,state0,False],[gent,zapn,state0,False],[genft,zapn,state0,False]]
			elif state=='dyn': tl0=[[genf,zapn,state0,False],[gent,zapn,state0,False],[genft,zapn,state0,False]]
			else: return
		else:
			if state=='freq': tl0=[[genf,zapn,state0,False],[gent,zapn,state0,False],[genft,zapn,state0,False]]
			elif state=='time': tl0=[[gent,zapn,state0,False],[genf,zapn,state0,False],[genft,zapn,state0,False]]
			elif state=='dyn': tl0=[[genft,zapn,state0,False],[genf,zapn,state0,False],[gent,zapn,state0,False]]
			else: return		
		if tlist:
			if tlist[0][-1]=='working': tlist=[tlist[0]]+tl0
			else: tlist=tl0
		else: tlist=tl0
		if state!=state0:
			if state=='freq': func=genf
			elif state=='time': func=gent
			elif state=='dyn': func=genft
			state0=state
			while True:
				mark=True
				for i in tlist:
					if func==i[0]: mark=False
				if mark: break
				else:
					state='busy'
					time.sleep(0.01)
			state=state0
	if state in ['freq','time']:
		data.mask=np.array([zapn0]).T.repeat(nbin,axis=1)
		data0.mask=np.array([zapn0]).T.repeat(nbin,axis=1)
		profile=data0.sum(0)
		if state=='freq' and cali:
			cal0=copy.deepcopy(cal)
			cal0[zapn0]=0
			spec0=np.concatenate((np.zeros([1,4]),cal0.repeat(2,axis=0),np.zeros([1,4])),axis=0)
		else:
			if args.mean:
				spec=data0.mean(1)
			else: spec=data1.std(1)
			spec=spec.data
			spec=spec-np.min(spec)
			spec[zapn0]=0
			spec0=np.append(0,np.append(spec.repeat(2),0))
	elif state=='dyn':
		data.mask=zapn
		if args.dprof or mem=='mem0':
			if mem!='none': profile=mtree.ftprof(zapn)
			else:
				profile=np.zeros(nbin)
				for i in np.arange(nchan):
					if zapn0[i]: continue
					dtmp=d.read_chan(i,pol=0)[:,:,0]*weight0[i,np.newaxis]
					dtmp[zapn[i]]=0
					profile+=dtmp.sum(1)
	lim0=limlist[-1]
	plotimage([lim0[0][0],lim0[1][0]],[lim0[0][1],lim0[1][1]])
#
def midclick(event):
	py=ycal(event.y)
	axs.cla()
	#axs.set_yticks([])
	if state=='freq':
		ybin=chancal(py)
		if nchan<=ybin or ybin<0: return
		#if zapn0[ybin]==True: return
		prof0=data0[ybin]
	elif state=='time':
		ybin=subcal(py)
		if ybin>=nsub or ybin<0: return
		#if zapn0[ybin]==True: return
		prof0=data0[ybin]
	elif state=='dyn':
		ybin=chancal(py)
		if nchan<=ybin or ybin<0: return
		xbin=subcal(xcal(event.x))
		if xbin>=nsub or xbin<0: return
		#if zapn[ybin,xbin]==True: return
		prof0=d.read_data(select_chan=[ybin],start_period=xbin,end_period=xbin+1,pol=0)[0,0,:,0]
	else: return
	axs.plot(np.arange(0,1,1/nbin),prof0,label='Selected')
	axs.set_xlabel('Phase',fontsize=20)
	axs.legend(loc='upper right')
	canvas.draw()
#
def xcal(x):
	xlim=ax.get_xlim()
	return (x-ax.bbox.extents[0])/ax.bbox.bounds[2]*(xlim[1]-xlim[0])+xlim[0]
#
def ycal(y):
	ylim=ax.get_ylim()
	return (fig.bbox.extents[3]-y-ax.bbox.extents[1])/ax.bbox.bounds[3]*(ylim[1]-ylim[0])+ylim[0]
#
def chancal(y):	# calculate the channel index corresponding to the y-value
	return np.int32((y-freq0)/channelwidth)
#
def subcal(y):
	return np.int32((y)/tsub)
#
def move(event):
	# plot a hline on the muose place
	if state not in ['freq','time','dyn']: return
	if not (event.x>fig.bbox.extents[0] and event.x<ax1.bbox.extents[2] and event.y<fig.bbox.extents[3] and event.y>fig.bbox.extents[1]): return
	y=(fig.bbox.extents[3]-event.y)/fig.bbox.extents[3]
	ly.set_ydata([y,y])
	pos_now_y= round(ycal(event.y),2)
	showposy=fig.text(0.005,y+0.01,pos_now_y,fontsize=12)
	if state=='dyn':
		x=event.x/fig.bbox.extents[2]
		lx.set_xdata([x,x])
		pos_now_x=round(xcal(event.x),2)
		showposx=fig.text(x+0.005,0.99,pos_now_x,fontsize=12,va='top')
	canvas.draw()  
	mpl.artist.Artist.remove(showposy)
	if state=='dyn':
		mpl.artist.Artist.remove(showposx)
#
def press(event):
	keymotion(event.keysym)
#
def keymotion(a):
	global limn,state,limlist,zapn0,zapn,zaplist,data0,data,data1,dmod
	if state not in ['freq','time','dyn']: return
	if a=='f':
		if state not in ['time','dyn']: return
		lx.set_xdata([0,0])
		init('freq')
	elif a=='t':
		if state not in ['freq','dyn']: return
		lx.set_xdata([0,0])
		init('time')
	elif a=='d':
		if state not in ['freq','time']: return
		init('dyn')
	elif a=='r':	# press 'r' to reset the zoom in region to the state before last zoom
		if len(limlist)>1:
			limlist.pop()  # remove the last one in the list 
		else: return
		lim0=limlist[-1]
		ax.set_ylim(lim0[0][1],lim0[1][1])
		ax1.set_ylim(lim0[0][1],lim0[1][1])
		if state=='dyn': ax.set_xlim(lim0[0][0],lim0[1][0])
		fig.lines=[lx,ly]
		canvas.draw()
		limn=0
	elif a=='z':	# press 'z' to zap distinct noise or rise the noise critical level
		if mem=='none': return
		if mtree.nstate and state!=mtree.nstate:
			print(text.info_mulmode)
			return
		if mtree.nlevel>=max_noiselevel: return
		if mtree.nlevel==0: mtree.nstate=state
		mtree.nlevel+=1
		if state=='freq':
			if (mtree.fnoise<mtree.nlevel).min(): return
			zapn[mtree.fnoise<mtree.nlevel]=True
			zapn0=np.logical_not((zapn==0).sum(1)>0)
		elif state=='time':
			if (mtree.tnoise<mtree.nlevel).min(): return
			zapn[mtree.tnoise<mtree.nlevel]=True
			zapn0=np.logical_not((zapn==0).sum(0)>0)
		elif state=='dyn':
			if (mtree.ftnoise<mtree.nlevel).min(): return
			zapn[mtree.ftnoise<mtree.nlevel]=True
			data.mask=zapn
		else: return
		if state in ['freq','time']:
			data.mask=np.array([zapn0]).T.repeat(nbin,axis=1)
			data0.mask=np.array([zapn0]).T.repeat(nbin,axis=1)
		update_image(state)
	elif a=='x':	# press 'x' to withdraw zappping distinct noise or lower the noise critical level
		if mem=='none':return
		if mtree.nlevel<=0: return
		mtree.nlevel-=1
		if mtree.nstate=='freq': zapn=mtree.fnoise<mtree.nlevel
		elif mtree.nstate=='time': zapn=mtree.tnoise<mtree.nlevel
		elif mtree.nstate=='dyn': zapn=mtree.ftnoise<mtree.nlevel
		else: raise
		for i in zaplist:
			zapn[i[1]:i[2],i[3]:i[4]]=True
		if state=='freq': zapn0=np.logical_not((zapn==0).sum(1)>0)
		elif state=='time': zapn0=np.logical_not((zapn==0).sum(0)>0)
		if state in ['freq','time']:
			data.mask=np.array([zapn0]).T.repeat(nbin,axis=1)
			data0.mask=np.array([zapn0]).T.repeat(nbin,axis=1)
		elif state=='dyn':
			data.mask=zapn
		fig.lines=[lx,ly]
		limn=0
		statetmp=mtree.nstate
		if mtree.nlevel==0: mtree.nstate=''
		update_image(statetmp)
	elif a=='u':	# press 'u' to withdraw the last zap
		if zaplist: state0=zaplist.pop()[0]
		else: return
		if mem=='none': zapn[:]=weight0==0
		elif mtree.nstate=='freq': zapn=mtree.fnoise<mtree.nlevel
		elif mtree.nstate=='time': zapn=mtree.tnoise<mtree.nlevel
		elif mtree.nstate=='dyn': zapn=mtree.ftnoise<mtree.nlevel
		else: zapn[:]=weight0==0
		#if state in ['freq','time']: zapn0[:]=False
		for i in zaplist:
			zapn[i[1]:i[2],i[3]:i[4]]=True
		if state=='freq': zapn0=np.logical_not((zapn==0).sum(1)>0)
		elif state=='time': zapn0=np.logical_not((zapn==0).sum(0)>0)
		if state in ['freq','time']:
			data.mask=np.array([zapn0]).T.repeat(nbin,axis=1)
			data0.mask=np.array([zapn0]).T.repeat(nbin,axis=1)
		elif state=='dyn':
			data.mask=zapn
		fig.lines=[lx,ly]
		limn=0
		update_image(state0)
	elif a=='q':
		state=''
		threading.join()
		root.destroy()
		zapchan=list(np.arange(nchan)[np.logical_not((zapn==0).sum(1)>0)])
		if zapchan: np.savetxt(args.filename[:-3]+'_zapchan.txt',zapchan,fmt='%i')
	elif a=='s':
		root.destroy()
		if zapn.sum()==0: return
		np.savetxt(args.filename[:-3]+'_zap.txt',zapn,fmt='%i')
		print(text.info_save % args.filename[:-3])
		if args.redo:
			chanweight=(info['additional_info']['chan_weight_raw']*(np.logical_not(zapn)).mean(1)).tolist()
		else:
			chanweight=(info['data_info']['chan_weight']*(np.logical_not(zapn)).mean(1)).tolist()
		weights=(weight0*np.logical_not(zapn)).tolist()
		if 'chan_weight_raw' not in info['additional_info'].keys():
			info['additional_info']['chan_weight_raw']=info['data_info']['chan_weight']
		info['data_info']['chan_weight']=chanweight
		info['data_info']['weights']=weights
		if args.output:
			newld=ld.ld(name+'.ld')
			print(text.info_saved % (args.output+'.ld'))
			newld.write_shape([nchan,nsub,nbin,npol])
			for i in np.arange(nchan):
				if chanweight[i]==0: continue
				dtmp=d.read_chan(i)
				dtmp[zapn[i]]=0
				newld.write_chan(i,dtmp)
			newld.write_info(info)
		else:
			d.write_info(info)
			print(text.info_savew % (args.filename))
	elif a=='w':
		root.destroy()
		if args.redo:
			chanweight=(info['additional_info']['chan_weight_raw']*(np.logical_not(zapn)).mean(1)).tolist()
		else:
			chanweight=(info['data_info']['chan_weight']*(np.logical_not(zapn)).mean(1)).tolist()
		weights=(weight0*np.logical_not(zapn)).tolist()
		if 'chan_weight_raw' not in info['additional_info'].keys():
			info['additional_info']['chan_weight_raw']=info['data_info']['chan_weight']
		info['data_info']['chan_weight']=chanweight
		info['data_info']['weights']=weights
		d.write_info(info)
		print(text.info_savew % (args.filename))
	elif a=='c':
		if calmark:
			global cali
			cali=not cali
			if state=='freq': update_image()
	elif a=='o':
		if state!='dyn': return
		if mem=='none':
			profiletmp=np.zeros(nbin)
			for i in np.arange(nchan):
				dtmp=d.read_chan(i,pol=0)[:,:,0]*weight0[i,np.newaxis]
				dtmp[zapn[i]]=0
				profiletmp+=dtmp.sum(1)
			if dmod=='off-pulse': 
				dmod='on-pulse'
				selbin=af.radipos(profiletmp)
			elif dmod=='on-pulse': 
				dmod='off-pulse'
				base_nbin=int(nbin/10)
				base,bin0=af.baseline(profiletmp,pos=True)
				selbin=np.sort(np.arange(nbin).reshape(1,-1).repeat(2,axis=0).reshape(-1)[bin0:(bin0+base_nbin)])
			data=d.bin_scrunch(select_bin=selbin.tolist(),pol=0)[:,:,0]
			data[zapn]=0
		else:
			if mtree.dmod=='off-pulse': mtree.genft(zapn,first='on-pulse')
			elif mtree.dmod=='on-pulse': mtree.genft(zapn,first='off-pulse')
			data=mtree.ft
		lim0=limlist[-1]
		plotimage([lim0[0][0],lim0[1][0]],[lim0[0][1],lim0[1][1]])
	elif a=='h':
		print(text.info_help.replace('\\n','\n'))
#
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import matplotlib as mpl
mpl.use('TkAgg')
root=tk.Tk()
root.title(args.filename)
root.geometry('1200x600+100+100')
canvas=FigureCanvasTkAgg(fig,master=root)
canvas.get_tk_widget().grid()  
canvas.get_tk_widget().pack(fill='both')
root.bind('<KeyPress>',press)
root.bind('<ButtonPress-1>',leftclick)
root.bind('<Double-Button-1>',midclick)
root.bind('<ButtonPress-2>',midclick)
root.bind('<ButtonPress-3>',rightclick)
root.bind('<Motion>',move)
canvas.draw()
init(state)
root.mainloop()
#
if mem=='disk': os.remove(datname)
