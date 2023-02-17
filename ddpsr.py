#!/usr/bin/env python
import numpy as np
import numpy.fft as fft
import numpy.polynomial.chebyshev as nc
import argparse as ap
import os,sys,time,ld
import mpmath as mm
mm.mp.dps=30
import astropy.io.fits as ps
import multiprocessing as mp
import ctypes as ct
import gc
import psr_read as pr
import time_eph as te
import psr_model as pm
import subprocess as sp
#
version='JigLu_20221212'
#
parser=ap.ArgumentParser(prog='ddpsr',description='Dedisperse and Fold the psrfits data.',epilog='Ver '+version)
parser.add_argument('-v','--version', action='version', version=version)
parser.add_argument('--verbose', action="store_true",default=False,help="print detailed information")
parser.add_argument("filename",nargs='+',help="name of file or filelist")
parser.add_argument("-a","--cal",dest="cal",nargs='+',help="name of calibration file or calibration filelist")
parser.add_argument("--cal_period",dest="cal_period",default=0,type=np.float64,help="period of the calibration fits file (s)")
parser.add_argument("--subi",action="store_true",default=False,help="take one subint as the calibration unit")
parser.add_argument("--cal_para",dest='cal_para',default='',help="the time range of the calibration file")
parser.add_argument("-t","--trend",action="store_true",default=False,help="fit the calibration parameter evolution")
parser.add_argument("-o","--output",dest="output",default="psr",help="output file name")
parser.add_argument("-f","--frequency",dest='freqrange',default=0,help="output frequency range (MHz) in form start_freq,end_freq")
parser.add_argument('-d','--dm',dest='dm',default=0,type=np.float64,help="dispersion measure")
parser.add_argument('-n','--pulsar_name',default=0,dest='psr_name',help='input pulsar name')
parser.add_argument('-e','--pulsar_ephemeris',default=0,dest='par_file',help='input pulsar parameter file')
parser.add_argument("-z","--zap",dest="zap_file",default=0,help="file recording zap channels")
parser.add_argument("-r","--reverse",action="store_true",default=False,help="reverse the band")
parser.add_argument("-m","--multi",dest="multi",default=0,type=int,help="number of processes")
parser.add_argument("-w","--overwrite",action="store_true",default=False,help="overwrite the existed output file")
args=(parser.parse_args())
command=['ddpsr.py']
#
if args.verbose:
	sys.stdout.write('Analyzing the arguments...\n')
filelist=args.filename
filenum=len(filelist)
file_t0=[]
file_time=[]
file_len=[]
#
def file_error(para,filetype):
	parser.error("Fits "+filetype+" have different parameters: "+para+".")
#
telename,pol_type,npol,nchan,freq,bandwidth,tsamp,nsblk,bw_sign,stt_imjd,stt_smjd,stt_offs,nsub,offs_sub='','',0,0,0,0.0,0.0,0,True,0,0,0.0,0,0.0
def file_check(fname,notfirst=True,filetype='data'):
	if not os.path.isfile(fname):
		parser.error('Fits '+filetype+' name is invalid.')
	try:
		f=ps.open(fname,mmap=True)
	except:
		parser.error('Fits '+filetype+' is invalid.')
	head=f['PRIMARY'].header
	subint=f['SUBINT']
	subint_header=subint.header
	subint_data=subint.data[0]
	global telename,pol_type,npol,nchan,freq,bandwidth,tsamp,nsblk,bw_sign,stt_imjd,stt_smjd,stt_offs,nsub,offs_sub
	if not notfirst:
		telename=head['TELESCOP']
		npol=subint_header['NPOL']
		nchan=head['OBSNCHAN']
		freq=head['OBSFREQ']
		bandwidth=subint_header['NCHAN']*subint_header['CHAN_BW']
		bw_sign=(bandwidth>0)
		bandwidth=np.abs(bandwidth)
		tsamp=subint_header['TBIN']
		nsblk=subint_header['NSBLK']
		pol_type=subint_header['POL_TYPE']
	else:
		if telename!=head['TELESCOP']:
			file_error('telescope name',filetype)
		if pol_type!=subint_header['POL_TYPE']:
			file_error('polarisation type',filetype)
		if npol!=subint_header['NPOL']:
			file_error('number of polorisations',filetype)
		if nchan!=head['OBSNCHAN']:
			file_error('number of channels',filetype)
		if freq!=head['OBSFREQ']:
			file_error('central frequency',filetype)
		if bandwidth!=np.abs(subint_header['NCHAN']*subint_header['CHAN_BW']):
			file_error('bandwidth',filetype)
		if tsamp!=subint_header['TBIN']:
			file_error('sampling time',filetype)
		#
	stt_imjd=head['STT_IMJD']
	stt_smjd=head['STT_SMJD']
	stt_offs=head['STT_OFFS']
	nsub=subint_header['NAXIS2']
	offs_sub=subint_data['OFFS_SUB']
	del subint_data
	f.close()
#
for i in np.arange(filenum):
	file_check(filelist[i],notfirst=i)
	#
	subint_t0=(offs_sub-tsamp*nsblk/2.0+stt_smjd+stt_offs)/86400.0+stt_imjd
	file_time.append([offs_sub-tsamp*nsblk/2.0,stt_smjd,stt_offs,stt_imjd])
	file_len.append(nsub)
	file_t0.append(subint_t0)
#
file_len,file_t0,filelist,file_time=np.array(file_len),np.array(file_t0),np.array(filelist),np.array(file_time)
sorts=np.argsort(file_t0)
file_len,file_t0,filelist,file_time=file_len[sorts],np.sort(file_t0),filelist[sorts],file_time[sorts]
if len(file_len)>1:
	if np.max(np.abs((file_len*nsblk*tsamp/86400.0+file_t0)[:-1]-file_t0[1:]))>(tsamp/86400.0):
		parser.error("Data files are not continuous.")
#
if args.cal:
	command.append('-a ')
	if args.cal_para:
		cal_para=args.cal_para.split(',')
		if len(cal_para)==1:
			try: tmp=np.float64(cal_para[0])
			except: parser.error("The calculation parameter is invalid.")
			cal_trend_eff=tmp
			cal_seg_eff=tmp
		elif len(cal_para)==2:
			try: cal_trend_eff,cal_seg_eff=np.float64(cal_para)
			except: parser.error("The calculation parameter is invalid.")
			if cal_seg_eff>cal_trend_eff:
				sys.stdout.write('Warning: The allowed time range of calibration file for single value need to be not more than that for trend fitting, and the former has been forced to be equal to the latter.\n')
		else: parser.error("The calculation parameter is invalid.")
		command.append('--cal_para '+args.cal_para)
	else:
		cal_trend_eff=1.5
		cal_seg_eff=0.5
	if len(args.cal)==1:
		if args.cal[0][-3:]=='.ld':
			noise_mark='ld'
			if not os.path.isfile(args.cal[0]):
				parser.error('Calibration file name is invalid.')
			noise=ld.ld(args.cal[0])
			noise_info=noise.read_info()
			if noise_info['mode']!='cal':
				parser.error("LD file is not caliration file.")
			elif telename!=noise_info['telename']:
				parser.error("LD calibration file has different telescope name.")
			elif nchan!=noise_info['nchan']:
				parser.error("LD calibration file has different channel number.")
		else:
			noise_mark='fits'
	else:
		noise_mark='fits'
	if noise_mark=='fits':
		if not args.cal_period:
			parser.error("Noise period is not given.")
		noiselist=args.cal
		noisenum=len(noiselist)
		noise_t0,noise_len=[],[]
		for i in np.arange(noisenum):
			file_check(noiselist[i],filetype='noise')
			subint_t0=(offs_sub-tsamp*nsblk/2.0+stt_smjd+stt_offs)/86400.0+stt_imjd
			noise_len.append(nsub)
			noise_t0.append(subint_t0)
		#
		noise_t0,noise_len,noiselist=np.array(noise_t0),np.array(noise_len),np.array(noiselist)
		sorts=np.argsort(noise_t0)
		noise_t0,noise_len,noiselist=noise_t0[sorts],noise_len[sorts],noiselist[sorts]
else:
	noise_mark=''
#
channel_width=bandwidth*1.0/nchan
if args.freqrange:
	command.append('-f '+args.freqrange)
	freq_start,freq_end=np.float64(args.freqrange.split(','))
	chanstart,chanend=np.int16(np.round((np.array([freq_start,freq_end])-freq)/channel_width+0.5*nchan))
	if freq_start>freq_end:
		parser.error("Starting frequency larger than ending frequency.")
	elif freq_start<(freq-bandwidth/2.0) or freq_end>(freq+bandwidth/2.0):
		parser.error("Input frequency is overrange.")
else:
	chanstart,chanend=0,nchan
nchan_new=chanend-chanstart
#
nbin=file_len.sum()*nsblk
stt_time=file_t0[0]
freq_start,freq_end=(np.array([chanstart,chanend])-0.5*nchan)*channel_width+freq
info={'nbin_origin':int(nbin),'telename':telename,'freq_start':freq_start,'freq_end':freq_end,'nchan_origin':int(chanend-chanstart),'nchan':1,'tsamp_origin':tsamp,'stt_time_origin':stt_time,'stt_time':stt_time,'npol':int(npol),'freq_align':freq_end-channel_width}
#
if args.psr_name and args.par_file:
	parser.error('At most one of flags -n and -p is required.')
elif args.psr_name or args.par_file:
	if args.dm:
		parser.error('With pulsar name or ephemeris, DM value is needless.')
	elif args.psr_name:
		command.append('-n '+args.psr_name)
		psr_name=args.psr_name
		psr=pr.psr(psr_name)
		psr_par=sp.getoutput('psrcat -e '+psr_name).split('\n')
		if len(psr_par)<3:
			parser.error('A valid pulsar name is required.')
		for i in range(len(psr_par)): psr_par[i]=psr_par[i]+'\n'
	else:
		command.append('-e')
		par_file=open(args.par_file,'r')
		psr_par=par_file.readlines()
		par_file.close()
		psr=pr.psr(args.par_file,parfile=True)
	info['psr_par']=psr_par
	pepoch=False
	for line in psr_par:
		elements=line.split()
		if elements[0]=='PSRJ' or elements[0]=='NAME':
			psr_name=elements[1].strip('\n')
			info['psr_name']=psr_name
		elif elements[0]=='DM':
			if not args.dm:
				dm=np.float64(elements[1])
			else:
				dm=args.dm
		elif elements[0]=='PEPOCH':
			pepoch=True
else:
	if not args.dm:
		parser.error('DM should be provided.')
	dm=args.dm
	command.append(' -d '+str(args.dm))
#
info['mode']='unfold'
info['dm']=dm
#
if args.zap_file:
	command.append('-z')
	if not os.path.isfile(args.zap_file):
		parser.error('The zap channel file is invalid.')
	zchan=np.loadtxt(args.zap_file,dtype=int)
	if np.max(zchan)>=nchan or np.min(zchan)<0:
		parser.error('The zapped channel number is overrange.')
	info['zchan']=zchan.tolist()
else:
	zchan=[]
name=args.output
if len(name)>3:
	if name[-3:]=='.ld':
		name=name[:-3]
if os.path.isfile(name+'.ld'):
	if not args.overwrite:
		tmp=1
		name0=name+'_'+str(tmp)
		while os.path.isfile(name0+'.ld'):
			name0=name+'_'+str(tmp)
			tmp+=1
		name=name0
		#parser.error('The name of output file already existed. Please provide a new name.')
#
if args.reverse:
	command.append('-r')
#
if args.multi:
	if args.multi>20:
		parser.error('The processes number is too large!')
	command.append('-m '+str(args.multi))
#
command=' '.join(command)
info['history']=[command]
#
def deal_seg(n1,n2):
	cumsub=0
	noise_data=np.zeros([noisen,npol,nchan])
	noise_cum=np.zeros(noisen)
	for n in np.arange(n1,n2):
		f=ps.open(noiselist[n],mmap=True)
		fsub=f['SUBINT'].header['naxis2']
		for i in np.arange(fsub):
			dtmp=f['SUBINT'].data[i]
			data=np.int16(dtmp['DATA'].reshape(nsblk,npol,nchan)*dtmp['dat_scl'].reshape(1,npol,nchan)+dtmp['dat_offs'].reshape(1,npol,nchan))
			del f['SUBINT'].data
			if args.reverse or (not bw_sign):
				data=data[:,:,::-1]
			if args.subi:
				noise_t=np.int64(cumsub%noisen)
				noise_data[noise_t]+=data.mean(0)
				noise_cum[noise_t]+=1
			else:
				noise_t=np.int64((np.arange(nsblk)+cumsub*nsblk)*tsamp%args.cal_period//tsamp)
				for k in np.arange(nsblk):
					tmp_noise_t=noise_t[k]
					if tmp_noise_t==noisen:
						continue
					noise_data[tmp_noise_t]+=data[k]
					noise_cum[tmp_noise_t]+=1
			cumsub+=1
		f.close()
	tmp_noise=noise_data[:,0].sum(1)/noise_cum
	sorts=np.argsort(tmp_noise)
	noise_data,noise_cum=noise_data[sorts],noise_cum[sorts]
	noisen_center=np.int64(noisen//2)
	if noisen>6:
		noise_off=noise_data[2:(noisen_center-1)].sum(0)/noise_cum[2:(noisen_center-1)].sum().reshape(-1,1)
		noise_on=noise_data[(noisen_center+2):-2].sum(0)/noise_cum[(noisen_center+2):-2].sum().reshape(-1,1)-noise_off
	elif noisen>2:
		sys.stdout.write('Warning: The noise data used in calulation is too short to get accurate calibration parameters.\n')
		noise_off=noise_data[:noisen_center].sum(0)/noise_cum[:noisen_center].sum().reshape(-1,1)
		noise_on=noise_data[(noisen_center+1):].sum(0)/noise_cum[(noisen_center+1):].sum().reshape(-1,1)-noise_off
	else:
		parser.error('The noise data used in calulation is too short.')
	noise_a12,noise_a22=noise_on[:2]
	noise_dphi=np.arctan2(noise_on[3],noise_on[2])
	noise_cos,noise_sin=np.cos(noise_dphi),np.sin(noise_dphi)
	return np.array([noise_a12,noise_a22,noise_cos,noise_sin])
#
if noise_mark=='fits':
	if args.verbose:
		sys.stdout.write('Processing the noise file...\n')
	if args.subi:
		noisen=np.int64(args.cal_period//(tsamp*nsblk))
	else:
		noisen=np.int64(args.cal_period//tsamp)
	jumps=np.abs((noise_len*tsamp/86400.0+noise_t0)[:-1]-noise_t0[1:])>(tsamp/86400.0)
	file_nseg=jumps.sum()+1
	jumps=np.concatenate(([0],np.where(jumps)[0]+1,[noisenum]))
	cumlen_noise=noise_t0-noise_t0[0]
	if args.trend:
		if (file_t0[-1]+1/1440.)<noise_t0[0] or (file_t0[0]-1/1440.)>noise_t0[-1]:
			parser.error('The calibration file time is out of the interpolating range.')
		if args.cal_para:
			if (file_t0[0]-noise_t0[0])>cal_trend_eff/24. or (noise_t0[-1]-file_t0[-1])>cal_trend_eff:
				parser.error('The calibration file time is out of the interpolating range.')
		noise_time0=noise_t0[0]
		if file_nseg>1:
			noise_time=np.zeros(file_nseg)
			noise_data=np.zeros([file_nseg,4,nchan])
			for i in np.arange(file_nseg):
				noise_data[i]=deal_seg(jumps[i],jumps[i+1])
				noise_time[i]=(cumlen_noise[jumps[i+1]-1]+cumlen_noise[jumps[i]]+noise_len[jumps[i]]*tsamp/86400)/2
			cal_mode='trend'
		else:
			if noisenum>1:
				noise_time=(cumlen_noise[-1]+noise_len[-1]*tsamp/86400)/2
				noise_data=np.zeros([noisenum,4,nchan])
				for i in np.arange(noisenum):
					noise_data[i]=deal_seg(i,i+1)
				cal_mode='trend'
			else:
				sys.stdout.write('Warning: Only one file is used to do the calibration and the calibration parameters are adopted without regard to the evolution.')
				if args.cal_para:
					if (file_t0[0]-noise_t0[0])>cal_seg_eff/24. or (noise_t0[-1]-file_t0[-1])>cal_seg_eff:
						parser.error('The calibration file time is out of the allowed range.')
				noise_data=deal_seg(0)
				cal_mode='single'
		noise_data=np.polyfit(noise_time,noise_data.reshape(file_nseg,-1),1).reshape(2,4,nchan)
	else:
		if args.cal_para:
			if (file_t0[0]-noise_t0[0])>cal_seg_eff/24. or (noise_t0[-1]-file_t0[-1])>cal_seg_eff:
				parser.error('The calibration file time is out of the allowed range.')
		noise_data=np.zeros([file_nseg,4,nchan])
		for i in np.arange(file_nseg):
			noise_data[i]=deal_seg(jumps[i],jumps[i+1])
		noise_data=noise_data.mean(0)
		cal_mode='single'
elif noise_mark=='ld':
	if noise_info['cal_mode']=='trend':
		noise_time0=noise_info['stt_time']
		noise_time=noise_info['seg_time']
		if file_t0[0]<((1.25*noise_time[0]-0.25*noise_time[-1])+noise_time0) or file_t0[-1]>((1.25*noise_time[-1]-0.25*noise_time[0])+noise_time0):
			parser.error('The file time is out of the extrapolating range.')
		noise_data=noise.read_data().reshape(nchan,2,4).transpose(1,2,0)
		cal_mode='trend'
	elif noise_info['cal_mode']=='seg':
		noise_time0=noise_info['stt_time']
		noise_time=noise_info['seg_time']
		noise_time_judge=((noise_time+noise_time0-file_t0[0])>(-cal_trend_eff/24.))&((noise_time+noise_time0-file_t0[-1])<(cal_trend_eff/24.))
		noise_time_judge_1=((noise_time+noise_time0-file_t0[0])>(-cal_seg_eff/24.))&((noise_time+noise_time0-file_t0[-1])<(cal_seg_eff/24.))
		noise_time_index=np.arange(len(noise_time))[noise_time_judge]
		noise_time_index_1=np.arange(len(noise_time))[noise_time_judge_1]
		nseg=len(noise_time_index)
		nseg_1=len(noise_time_index_1)
		if not nseg_1:
			parser.error('No valid calibration segment closed to the observation data.')
		elif nseg<=1:
			if args.trend:
				sys.stdout.write('Warning: The calibration file has only one segment and the calibration parameters are adopted without regard to the evolution.\n')
			noise_data=noise.read_period(noise_time_index_1[0]).reshape(nchan,4).T
			cal_mode='single'
		elif args.trend:
			if file_t0[-1]<(noise_time[noise_time_index[0]]+noise_time0) or file_t0[0]>(noise_time[noise_time_index[-1]]+noise_time0):
				sys.stdout.write('Warning: The file time of effective calibration segments are out of the interpolating range and the calibration parameters are adopted without regard to the evolution.')
				noise_data=np.zeros([nseg_1,4,nchan])
				for i in np.arange(nseg_1):
					noise_data[i]=noise.read_period(noise_time_index_1[i]).reshape(nchan,4).T
				noise_data=noise_data.mean(0)
				cal_mode='single'
			else:
				noise_data=np.zeros([nseg,4,nchan])
				for i in np.arange(nseg):
					noise_data[i]=noise.read_period(noise_time_index[i]).reshape(nchan,4).T
				noise_data=np.polyfit(noise_time[noise_time_judge],noise_data.reshape(nseg,-1),1).reshape(2,4,nchan)
				cal_mode='trend'
		else:
			cal_mode='single'
			noise_data=np.zeros([nseg_1,4,nchan])
			for i in np.arange(nseg_1):
				noise_data[i]=noise.read_period(noise_time_index_1[i]).reshape(nchan,4).T
			noise_data=noise_data.mean(0)
	else:
		parser.error('The calibration file mode is unknown.')
if args.cal:
	info['cal_mode']=cal_mode
	info['cal']=noise_data.reshape(-1,nchan).tolist()
	if cal_mode=='single':
		noise_a12,noise_a22,noise_cos,noise_sin=noise_data
		noise_a12=np.where(noise_a12>0,1./noise_a12,0)
		noise_a22=np.where(noise_a22>0,1./noise_a22,0)
		noise_a1a2=np.sqrt(noise_a12*noise_a22)
		noise_cos=noise_cos*noise_a1a2
		noise_sin=noise_sin*noise_a1a2
#
if args.verbose:
	sys.stdout.write('Constructing the output file...\n')
#
nbin_old=nbin
freq0=freq_start
freq1=freq_end
nbin0=nbin
#
roach_delay=8.192e-6*3
gps_delay=1e-5
transline_delay=2e-5
light_delay=(300.0+141.96)/3.0e8
delay=transline_delay+light_delay-gps_delay-roach_delay
#
stt_time=info['stt_time']
end_time=stt_time+tsamp*nbin0/86400.0+60./86400
stt_time-=60./86400
time0=file_time[0]
#
dbin=int(np.ceil(dm*pm.dm_const*(1/freq0**2-1/freq1**2)/tsamp))
nbin=nbin0-dbin
phase=np.arange(nbin)*tsamp
#
info['phase0']=int(0)
stt_sec=time0[:-1].sum()-delay
stt_date=time0[-1]+stt_sec//86400
stt_sec=stt_sec%86400
#
info['stt_sec']=stt_sec
info['stt_date']=int(stt_date)
info['stt_time']=stt_date+stt_sec/86400.0
info['nbin']=int(nbin)
info['length']=nbin0*tsamp
#
df=freq_start+np.arange(nchan_new)*channel_width
#
info['nsub']=int(1)
info['file_time']=time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())
#
def write_data(n,cumsub,fsub,data,lock=0):
	if args.multi: lock.acquire()
	sbin=cumsub*nsblk
	nn=fsub*nsblk
	if n==(filenum-1): nn-=dbin
	ebin=sbin+nn
	d[sbin:ebin]+=data[:,:nn].T
	if n>0:
		ntmp=min(sbin,dbin)
		d[(sbin-ntmp):sbin]+=data[:,-ntmp:].T
	if args.multi: lock.release()
#
if args.verbose:
	sys.stdout.write('Dedispersing and folding the data...\n')
#
def dealdata(filelist,n,lock=0):
	global noise_a12,noise_a22,noise_cos,noise_sin
	if args.verbose:
		if args.multi: lock.acquire()
		sys.stdout.write('Processing the '+str(n+1)+'th fits file...\n')
		if args.multi: lock.release()
		timemark=time.time()
	tpsub=0
	tpsubn=0
	cumsub=np.int64(file_len[:n].sum())
	f=ps.open(filelist[n],mmap=True)
	fsub=f['SUBINT'].header['naxis2']
	nbin_tmp=2**np.int16(np.log2(fsub*nsblk+dbin)+1)
	dout=np.zeros([npol,nbin_tmp])	
	scl,offs=f['SUBINT'].data['dat_scl'].reshape(fsub,1,npol,nchan),f['SUBINT'].data['dat_offs'].reshape(fsub,1,npol,nchan)
	del f['SUBINT'].data
	if args.cal:
		if cal_mode=='trend':
			dt=(np.arange(nsblk*fsub)+nsblk*fsub*cumsub)*tsamp/86400+stt_time-noise_time0
			noise_a12,noise_a22,noise_cos,noise_sin=np.polyval(noise_data,dt.reshape(-1,1,1)).transpose(1,2,0)
			noise_a12=np.where(noise_a12>0,1./noise_a12,0)
			noise_a22=np.where(noise_a22>0,1./noise_a22,0)
			noise_a1a2=np.sqrt(noise_a12*noise_a22)
			noise_cos=noise_cos*noise_a1a2
			noise_sin=noise_sin*noise_a1a2
	for i in np.arange(chanstart,chanend):
		dtmp=f['SUBINT'].data['data'][:,:,:,i].reshape(fsub,nsblk,npol)
		if i in zchan: continue
		dphase=dm*pm.dm_const*(1/df[i-chanstart]**2-1/freq_end**2)/tsamp/nbin_tmp*2*np.pi
		data=(dtmp*scl[:,:,:,i]+offs[:,:,:,i]).reshape(fsub*nsblk,npol)
		del f['SUBINT'].data
		if args.cal:
			if pol_type=='AABBCRCI':
				aa0,bb0,cr0,ci0=noise_a12[i]*data[:,0],noise_a22[i]*data[:,1],noise_cos[i]*data[:,2]+noise_sin[i]*data[:,3],-noise_sin[i]*data[:,2]+noise_cos[i]*data[:,3]
				ii,qq,uu,vv=aa0+bb0,aa0-bb0,2*cr0,2*ci0
			elif pol_type=='IQUV':
				noise_a1p2=(noise_a12+noise_a22)/2.0
				noise_a1m2=(noise_a12-noise_a22)/2.0
				ii,qq,uu,vv=noise_a1p2*data[:,0]-noise_a1m2*data[:,1],noise_a1p2*data[:,1]-noise_a1m2*data[:,0],noise_cos*data[:,2]+noise_sin*data[:,3],-noise_sin*data[:,2]+noise_cos*data[:,3]
			data=np.array([ii,qq,uu,vv])
		else:
			data=data.T
		dtmp=np.zeros([npol,nbin_tmp])
		dtmp[:,:(fsub*nsblk)]=data
		dtmp=fft.rfft(dtmp,axis=1)
		ldtmp=dtmp.shape[1]
		dout+=fft.irfft(np.exp(np.arange(ldtmp)*dphase*1j)*dtmp,axis=1)
	write_data(n,cumsub,fsub,dout,lock)
	f.close()
	gc.collect()
	if args.verbose:
		if args.multi: lock.acquire()
		sys.stdout.write('Processing the '+str(n+1)+'th fits file takes '+str(time.time()-timemark)+' second.\n')
		if args.multi: lock.release()
#
if args.multi:
	shared=mp.Array(ct.c_double,int(nbin*npol),lock=True)
	d=np.frombuffer(shared.get_obj(),dtype=ct.c_double).reshape(nbin,npol)
	pool=mp.Pool(processes=args.multi)
	lock=mp.Manager().Lock()
else:
	d=np.zeros([nbin,npol])
#
for n in np.arange(filenum):
	if args.multi:
		pool.apply_async(dealdata,(filelist,n,lock))
	else:
		dealdata(filelist,n)
if args.multi:
	pool.close()
	pool.join()
#
if args.cal:
	info['pol_type']='IQUV'
	if cal_mode=='trend':
		info['noise_time0']=noise_time0
else:
	info['pol_type']=pol_type
#
d0=ld.ld(name+'.ld')
d0.write_shape([1,info['nsub'],nbin,npol])
d0.write_chan(d,0)
d0.write_info(info)
#

