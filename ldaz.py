#!/usr/bin/env python
import numpy as np
import numpy.fft as fft
import numpy.polynomial.chebyshev as nc
import scipy.optimize as so
import scipy.signal as ss
import argparse as ap
import os,ld,sys
import subprocess as sp
import time
dirname=os.path.split(os.path.realpath(__file__))[0]
sys.path.append(dirname+'/doc')
import text
#
text=text.output_text('ldaz')
version='JigLu_20220711'
parser=ap.ArgumentParser(prog='ldaz',description=text.help,epilog='Ver '+version,add_help=False,formatter_class=lambda prog: ap.RawTextHelpFormatter(prog, max_help_position=50))
parser.add_argument('-h', '--help', action='help', default=ap.SUPPRESS,help=text.help_h)
parser.add_argument('-v','--version',action='version',version=version,help=text.help_v)
parser.add_argument('--verbose', action="store_true",default=False,help=text.help_verbose)
parser.add_argument("filename",nargs='+',help=text.help_filename)
parser.add_argument('-j','--crit',dest='crit',default=1,type=np.float64,help=text.help_j)
parser.add_argument("-p","--polynum",dest="polynum",default=7,type=int,help=text.help_p)
parser.add_argument("--cr",dest='chanrange',default='',help=text.help_cr)
parser.add_argument('-m',action='store_true',default=False,dest='modify',help=text.help_m)
parser.add_argument('-c',dest='cal',action='store_true',default=False,help=text.help_c)
args=(parser.parse_args())
command=['ldaz.py']
#
if args.verbose:
	print(text.info_ana)
filelist=args.filename
filenum=len(filelist)
#
def ld_check(fname,notfirst=True):	# check the validity of file
	fnametmp='___'+fname
	if fnametmp[-3:]!='.ld':
		parser.error(text.error_nld % fname)
	if not os.path.isfile(fname):
		parser.error(text.error_nvn % fname)
	try:
		f=ld.ld(filelist[i])
		finfo=f.read_info()
	except:
		parser.error(text.error_nvld % fname)
	global nchan
	if not ((finfo['data_info']['mode']=='single')|(finfo['data_info']['mode']=='subint')):
		parser.error(text.error_npd % fname)
	if not notfirst:
		nchan=np.int16(finfo['data_info']['nchan'])
	else:
		if nchan!=np.int16(finfo['data_info']['nchan']):
			parser.error(text.error_dcn)
#
for i in np.arange(filenum):
	ld_check(filelist[i],notfirst=i)
#
if args.chanrange:
	command.append('--cr '+args.chanrange)
	chanstart,chanend=np.int16(args.chanrange.split(','))
	if chanstart>chanend:
		parser.error(text.error_ncn)
	elif chanstart<0 or chanend>nchan:
		parser.error(text.error_ocr)
else:
	chanstart,chanend=0,nchan
#
command.append('-j '+str(args.crit))
command.append('-p '+str(args.polynum))
if args.modify:
	command.append('-m')
command=' '.join(command)
#
def func(x0,y0,c0):	# the ellipticity of polarization should have a linear relation with frequency
	c=[0.0015304279396914935,-5.142979752233712]
	def lineres(p):
		k,b=p
		y1=(k*x0+b-y0)%(2*np.pi)
		y1[y1>np.pi]-=np.pi*2
		return y1
	#
	a=so.leastsq(lineres,x0=c,full_output=1)
	k,b=a[0]
	y1=k*x0+b
	y0[(y0-y1)>np.pi]-=2*np.pi
	y0[(y0-y1)<-np.pi]+=2*np.pi
	jj=np.arange(len(x0))[np.abs(y0-y1)<c0]
	return k,b,jj
#
def calzap(cal,crit):	# consider the smoothness of aa and bb and the elliptical angle calculated with cr and ci
	aa,bb,cr,ci=cal
	aa=aa/aa[chanstart:chanend].mean()
	bb=bb/bb[chanstart:chanend].mean()
	b,a=ss.butter(10,0.1,'lowpass')
	aa0,bb0=ss.filtfilt(b,a,aa),ss.filtfilt(b,a,bb)
	da=aa-aa0
	da/=da.std()
	db=bb-bb0
	db/=db.std()
	jj=np.zeros_like(da,dtype=bool)
	x0=np.arange(nchan)
	j1a=np.zeros(nchan,dtype=bool)
	j1t=np.ones(nchan,dtype=bool)
	loop=0
	while not np.all(j1t==j1a):
		j1t=j1a.copy()
		j0a=np.abs(da)<=3
		while not np.all(j0a|j1a):
			j1a=np.abs(da)>3
			da/=da[j0a].std()
			j0a=np.abs(da)<=3
			loop+=1
			if loop>1000: break
		if loop>1000: return np.zeros(nchan,dtype=bool)
		xtmp=x0[j0a]
		aa1=np.interp(x0,xtmp,aa[j0a])
		aa0=ss.filtfilt(b,a,aa1)
		da=aa1-aa0
		da/=da.std()
		j1a=j1a|j1t
	j0a=np.logical_not(j1a)
	x2=x0[j0a]
	aa2=aa[j0a]
	y0=nc.chebval(x2,nc.chebfit(x2,aa2,args.polynum))
	dy=aa2-y0
	j0a=np.abs(dy)<=0.1
	j2a=np.zeros(len(dy),dtype=bool)
	while not np.all(j0a|j2a):
		j2a=np.abs(dy)>0.1
		xtmp=x2[j0a]
		y0=nc.chebval(x2,nc.chebfit(xtmp,aa2[j0a],args.polynum))
		dy=aa2-y0
		j0a=np.abs(dy)<=0.1
	j01a=np.logical_not(j1a)
	j02a=np.logical_not(j2a)
	#
	j1b=np.zeros(nchan,dtype=bool)
	j1t=np.ones(nchan,dtype=bool)
	while not np.all(j1t==j1b):
		j1t=j1b.copy()
		j0b=np.abs(db)<=3
		while not np.all(j0b|j1b):
			j1b=np.abs(db)>3
			db/=db[j0b].std()
			j0b=np.abs(db)<=3
		xtmp=x0[j0b]
		bb1=np.interp(x0,xtmp,bb[j0b])
		bb0=ss.filtfilt(b,a,bb1)
		db=bb1-bb0
		db/=db.std()
		j1b=j1b|j1t
	j0b=np.logical_not(j1b)
	x2=x0[j0b]
	bb2=bb[j0b]
	y0=nc.chebval(x2,nc.chebfit(x2,bb2,args.polynum))
	dy=bb2-y0
	j0b=np.abs(dy)<=crit
	j2b=np.zeros(len(dy),dtype=bool)
	while not np.all(j0b|j2b):
		j2b=np.abs(dy)>crit
		xtmp=x2[j0b]
		y0=nc.chebval(x2,nc.chebfit(xtmp,bb2[j0b],args.polynum))
		dy=bb2-y0
		j0b=np.abs(dy)<=crit
	j01b=np.logical_not(j1b)
	j02b=np.logical_not(j2b)
	#
	y0=np.arctan2(ci,cr)
	k,b0,jj=func(x0,y0,crit*4)
	k,b0,jj1=func(x0[jj],y0[jj],crit*2)
	jj=jj[jj1]
	k,b0,jj1=func(x0[jj],y0[jj],crit)
	jj=jj[jj1]
	y1=k*x0+b0
	#
	tmp=np.zeros(nchan,dtype=bool)
	tmp[jj]=True
	tmp1a=np.arange(nchan,dtype=np.int32)
	tmp1a=tmp1a[j01a][j02a]
	j0a=tmp1a.copy()
	tmp2a=np.zeros(nchan,dtype=bool)
	tmp2a[j0a]=True
	tmp1b=np.arange(nchan,dtype=np.int32)
	tmp1b=tmp1b[j01b][j02b]
	j0b=tmp1b.copy()
	tmp2b=np.zeros(nchan,dtype=bool)
	tmp2b[j0b]=True
	jj=tmp&tmp2a&tmp2b
	return jj
#
max_noiselevel=5
def noise(data,crit_noiselevel=5):
	nx,ny=data.shape
	data=data.reshape(-1)
	ind=np.argsort(data)
	data=data[ind]
	tmp=np.diff(data)
	tmpind=np.argsort(tmp)
	tmpsort=tmp[tmpind]
	crit=tmpsort[int(data.size*0.25):int(data.size*0.75)].mean()+tmpsort[int(data.size*0.25):int(data.size*0.75)
].std()*1000*10**np.arange(max_noiselevel).reshape(-1,1)
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
	#print(critn)
	for i in max_noiselevel-np.arange(max_noiselevel)-1:
		if critn[0,i]>0: level1[critn[0,i]:]=i
		if critn[1,i]>0: level[:critn[1,i]]=i
	#print((level==1).sum(),(level==2).sum(),(level==3).sum(),(level==4).sum(),(level==5).sum())
	level[int(level.size/2):]=level1[int(level.size/2):]
	noiselevel=level[np.argsort(ind)].reshape(nx,ny)
	fnoise=np.min(noiselevel,axis=1)
	return fnoise>=crit_noiselevel
#
for i in filelist:
	dfile=ld.ld(i)
	d=dfile.period_scrunch()[:,:,0]
	info=dfile.read_info()
	d*=np.array(info['data_info']['chan_weight']).reshape(-1,1)
	nchan,nperiod,nbin0,npol=dfile.read_shape()
	if args.cal:
		if 'calibration_info' in info.keys():
			cal=np.array(info['calibration_info']['cal'])[-1].reshape(4,-1)
			jc=calzap(cal,(np.exp((10-args.crit)/4.5)+2)/50)&(d.mean(1)>0)
			#print(jc.sum(),(d.mean(1)>0).sum(),d.shape)
			if jc.sum()==0:
				continue
			#d=d[jc]
		else:
			print(text.warning_nc)
			jc=np.ones(nchan,dtype=bool)
	else:
		jc=np.ones(nchan,dtype=bool)
	#
	j1=noise(d,args.crit)
	#
	zchan0=np.arange(nchan,dtype=np.int32)[jc&j1]
	zchan1=np.where(info['data_info']['chan_weight']==0)[0]
	zchan=list(set(zchan0).union(zchan1))
	if args.modify:
		weight=info['data_info']['chan_weight']
		weight[zchan]=0
		info['data_info']['chan_weight']=weight
		if 'history_info' in info.keys():
			info['history_info']['history'].append(command)
			info['history_info']['file_time'].append(time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime()))
		else:
			info['history_info']={}
			info['history_info']['history']=[command]
			info['history_info']['file_time']=[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())]
		dfile.write_info(info)
	else:
		np.savetxt(i[:-3]+'.txt',zchan,fmt='%i')
