#!/usr/bin/env python
import numpy as np
import numpy.fft as fft
import numpy.polynomial.chebyshev as nc
import argparse as ap
import os,sys,time,ld,platform
import mpmath as mm
mm.mp.dps=30
import astropy.io.fits as ps
from multiprocessing import Pool,Manager
import gc
import psr_read as pr
import time_eph as te
import psr_model as pm
import subprocess as sp
import adfunc as af
#
version='JigLu_20200925'
#
parser=ap.ArgumentParser(prog='dfpsr',description='Dedisperse and Fold the psrfits data.',epilog='Ver '+version)
parser.add_argument('-v','--version', action='version', version=version)
parser.add_argument('--verbose', action="store_true",default=False,help="print detailed information")
parser.add_argument("filename",nargs='+',help="name of file or filelist")
parser.add_argument("-a","--cal",dest="cal",nargs='+',help="name of calibration file or calibration filelist")
parser.add_argument("--cal_period",dest="cal_period",default=0,type=np.float64,help="period of the calibration fits file (s)")
parser.add_argument("--subi",action="store_true",default=False,help="take one subint as the calibration unit")
parser.add_argument("--cal_para",dest='cal_para',default='',help="the time range of the calibration file")
parser.add_argument("--trend",action="store_true",default=False,help="fit the calibration parameter evolution")
parser.add_argument("-o","--output",dest="output",default="psr",help="output file name")
parser.add_argument("--fr",dest='freqrange',default=0,help="output frequency range (MHz) in form start_freq,end_freq")
parser.add_argument('-d','--dm',dest='dm',default=-1,type=np.float64,help="dispersion measure")
parser.add_argument('-p','--period',dest='period',default=0,type=np.float64,help="pulsar period (s)")
parser.add_argument('-n','--pulsar_name',default=0,dest='psr_name',help='input pulsar name')
parser.add_argument('-e','--pulsar_ephemeris',default=0,dest='par_file',help='input pulsar parameter file')
parser.add_argument("-c","--coefficients_num",dest="ncoeff",default=12,type=int,help="numbers of Chebyshev polynomial coefficients on time axis")
parser.add_argument("-b","--nbin",dest="nbin",default=0,type=int,help="number of phase bins in each period")
parser.add_argument("-s","--sublen",dest="subint",default=0,type=np.float64,help="length of sub-integration (s)")
parser.add_argument("--nsub",dest="nsub",default=0,type=int,help="total number of sub-integrations")
parser.add_argument("--sub_nperiod",dest="sub_nperiod",default=0,type=int,help="the number of period in one sub-integration")
parser.add_argument("-z","--zap",dest="zap_file",default=0,help="file recording zap channels")
parser.add_argument("-r","--reverse",action="store_true",default=False,help="reverse the band")
parser.add_argument("-l","--large_mem",action="store_true",default=False,help="large RAM")
parser.add_argument("-m","--multi",dest="multi",default=0,type=int,help="number of processes")
parser.add_argument("-w","--overwrite",action="store_true",default=False,help="overwrite the existed output file")
args=(parser.parse_args())
command=['dfpsr.py']
#
if args.verbose:
	sys.stdout.write('Analyzing the arguments...\n')
#
platf=platform.system()
if platf=='Windows':
	if os.path.isfile(args.filename[0]): filelist=args.filename
	else:
		filepath,filere=os.path.split(args.filename[0])
		filere=filere.replace('.','\.').replace('+','\+').replace('*','(.*)')
		if filepath=='':filepath='.'
		files0=os.listdir(filepath)
		filelist=[]
		for i in files0:
			import re
			ii=re.match(filere,i)
			if ii: filelist.append(ii.string)
else:
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
#
def file_check(fname,notfirst=True,filetype='data'):	# check the file consistency
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
if len(file_len)>1:	# check the file continuity
	if np.max(np.abs((file_len*nsblk*tsamp/86400.0+file_t0)[:-1]-file_t0[1:]))>(tsamp/86400.0):
		parser.error("Data files are not continuous.")
#
if args.cal:	# check the calibration file and parameters
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
			if noise_info['data_info']['mode']!='cal':
				parser.error("LD file is not caliration file.")
			elif telename!=noise_info['telescope_info']['telename']:
				parser.error("LD calibration file has different telescope name.")
			elif nchan!=noise_info['data_info']['nchan']:
				parser.error("LD calibration file has different channel number.")
		else:
			noise_mark='fits'
	else:
		noise_mark='fits'
	if noise_mark=='fits':
		if not args.cal_period:
			parser.error("Noise period is not given.")
		if platf=='Windows':
			if os.path.isfile(args.cal[0]): noiselist=args.cal
			else:
				filepath,filere=os.path.split(args.cal[0])
				filere=filere.replace('.','\.').replace('+','\+').replace('*','(.*)')
				if filepath=='':filepath='.'
				files0=os.listdir(filepath)
				noiselist=[]
				for i in files0:
					import re
					ii=re.match(filere,i)
					if ii: noiselist.append(ii.string)
		else:
			noiselist=args.cal
		noisenum=len(noiselist)
		noise_t0,noise_len=[],[]
		for i in np.arange(noisenum):
			file_check(noiselist[i],filetype='noise')
			subint_t0=(offs_sub-tsamp*nsblk/2.0+stt_smjd+stt_offs)/86400.0+stt_imjd
			noise_len.append(nsub*nsblk)
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
	command.append('--fr '+args.freqrange)
	freq_start,freq_end=np.float64(args.freqrange.split(','))
	chanstart,chanend=np.int16(np.round((np.array([freq_start,freq_end])-freq)/channel_width+0.5*nchan+0.5))
	if chanstart>chanend:
		parser.error("Starting frequency larger than ending frequency.")
	elif chanstart<0 or chanend>nchan:
		parser.error("Input frequency is overrange.")
else:
	chanstart,chanend=0,nchan
nchan_new=chanend-chanstart
#
nbin0=file_len.sum()*nsblk
stt_time=file_t0[0]
freq_start,freq_end=(np.array([chanstart,chanend])-0.5*nchan-0.5)*channel_width+freq
#
info={'data_info':{'freq_start':freq_start,'freq_end':freq_end,'nchan':int(chanend-chanstart),'stt_time':stt_time,'npol':int(npol)},'original_data_info':{'nbin_origin':int(nbin0),'freq_start_origin':freq-bandwidth*0.5-channel_width*0.5,'freq_end_origin':freq+bandwidth*0.5-channel_width*0.5,'nchan_origin':int(nchan),'tsamp_origin':tsamp,'stt_time_origin':stt_time},'telescope_info':{'telename':telename}}
#
if args.psr_name and args.par_file:	# check the conflict of folding flags
	parser.error('At most one of flags -n and -e is required.')
elif args.psr_name or args.par_file:
	if args.period:
		parser.error('With pulsar name or ephemeris, period is needless.')
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
	info['pulsar_info']={'psr_par':psr_par}
	pepoch=False
	if args.dm>=0:
		psr.dm=args.dm
	for line in psr_par:
		elements=line.split()
		if elements[0]=='PSRJ' or elements[0]=='NAME':
			psr_name=elements[1].strip('\n')
			info['pulsar_info']['psr_name']=psr_name
		elif elements[0]=='F0':
			period=1./np.float64(elements[1])
		elif elements[0]=='DM':
			if args.dm<0:
				dm=np.float64(elements[1])
			else:
				dm=args.dm
		elif elements[0]=='PEPOCH':
			pepoch=True
else:
	if not args.period and args.dm<0:
		parser.error('Pulsar Parameter should be provided.')
	if not args.period or args.dm<0:
		parser.error('Both DM and period should be provided.')
	period=args.period
	dm=args.dm
	command.append('-p '+str(args.period)+' -d '+str(args.dm))
#
if args.subint or args.nsub or args.sub_nperiod:	# check the subint flags
	if (args.subint and args.nsub) or (args.subint and args.sub_nperiod) or (args.sub_nperiod and args.nsub):
		parser.error('At most one of flags --sublen, --nsub and --sub_nperiod is required.')
	if args.subint:
		if args.subint<period:
			parser.error('Duration time of a subint is too short')
		elif args.subint<(1.5*period):
			sys.stdout.write('Warning: Duration time of a subint is too short, then the out put file is indeed single pulse mode.\n')
			info['data_info']['mode']='single'
		else:
			info['data_info']['mode']='subint'
			sub_nperiod=np.int64(round(args.subint/period))
			info['data_info']['sublen']=period*sub_nperiod
			command.append('-s '+str(args.subint))
	elif args.nsub:
		if args.nsub*period>nbin0*tsamp:
			parser.error('Duration time of a subint is too short')
		elif args.nsub*period*1.5>nbin0*tsamp:
			sys.stdout.write('Warning: Duration time of a subint is too short, then the out put file is indeed single pulse mode.\n')
			info['data_info']['mode']='single'
		elif args.nsub<0:
			parser.error('The sub-integration number can not be negative.')
		else:
			info['data_info']['mode']='subint'
			sub_nperiod=np.int64(round(nbin0*tsamp/args.nsub/period))
			info['data_info']['sublen']=period*sub_nperiod
			command.append('--nsub '+str(args.nsub))
	else:
		if args.sub_nperiod==1:
			sys.stdout.write('Warning: Period number in one sub-integration is too short, then the out put file is indeed single pulse mode.\n')
			info['data_info']['mode']='single'
		elif args.sub_nperiod<=0:
			parser.error('The period number in one sub-integration must be positive.')
		else:
			info['data_info']['mode']='subint'
			sub_nperiod=np.int64(args.sub_nperiod)
			info['data_info']['sublen']=period*sub_nperiod
			command.append('--sub_nperiod '+str(args.sub_nperiod))
else:
	info['data_info']['mode']='single'
#
info['data_info']['dm']=dm
#
command.append('-c '+str(args.ncoeff))
if args.nbin:	# check the nbin flag
	command.append('-b '+str(args.nbin))
	if args.nbin>(period/tsamp):
		if args.subint:
			if args.nbin>(period/tsamp*sub_nperiod):
				parser.error('Provided phase bin number in each period is too large.')
		else:
			parser.error('Provided phase bin number in each period is too large.')
#
if args.zap_file:
	command.append('-z')
	if not os.path.isfile(args.zap_file):
		parser.error('The zap channel file is invalid.')
	zchan=np.loadtxt(args.zap_file,dtype=int)
	if np.max(zchan)>=nchan or np.min(zchan)<0:
		parser.error('The zapped channel number is overrange.')
	weight=np.ones(nchan,dtype=np.float64)
	weight[zchan]=0
	weight=weight[chanstart:chanend]
else:
	zchan=[]
	weight=np.ones(chanend-chanstart,dtype=np.float64)
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
	if platf=='Windows' and args.multi>1:
		parser.error('The multi-processes function is not supported on Windows!')
	if args.multi>20:
		parser.error('The processes number is too large!')
	command.append('-m '+str(args.multi))
#
command=' '.join(command)
info['history_info']={'history':[command]}
#
def tmp_deal(n,noise_data,noise_cum,fsub,cumsub,n1,n2,multi,lock=0):
	f=ps.open(noiselist[n],mmap=True)
	for i in np.arange(fsub[n-n1]):
		dtmp=f['SUBINT'].data[i]
		data=np.int16(dtmp['DATA'].reshape(nsblk,npol,nchan)*dtmp['dat_scl'].reshape(1,npol,nchan)+dtmp['dat_offs'].reshape(1,npol,nchan))
		del f['SUBINT'].data
		if args.reverse or (not bw_sign):
			data=data[:,:,::-1]
		if args.subi:
			noise_t=np.int64((cumsub[n-n1]+i)%noisen)
			if multi: lock.acquire()
			noise_data[noise_t]+=data.mean(0)
			noise_cum[noise_t]+=1
			if multi: lock.release()
		else:
			noise_t=np.int64((np.arange(nsblk)+(cumsub[n-n1]+i)*nsblk)*tsamp%args.cal_period//tsamp)
			for k in np.arange(nsblk):
				tmp_noise_t=noise_t[k]
				if tmp_noise_t==noisen:
					continue
				if multi: lock.acquire()
				noise_data[tmp_noise_t]+=data[k]
				noise_cum[tmp_noise_t]+=1
				if multi: lock.release()
	f.close()
#
def deal_seg(n1,n2,multi=0):	# processing the noise data segments
	cumsub=0
	if multi:
		manager=Manager()
		noise_data=manager.list([np.zeros([npol,nchan])]*noisen)
		noise_cum=manager.list(np.zeros(noisen))
		pool0=Pool(processes=multi)
		lock=manager.Lock()
	else:
		noise_data=np.zeros([noisen,npol,nchan])
		noise_cum=np.zeros(noisen)
	fsub=[]
	for n in np.arange(n1,n2):
		f=ps.open(noiselist[n],mmap=True)
		fsub.append(f['SUBINT'].header['naxis2'])
		f.close()
	cumsub=np.cumsum(fsub)
	#
	for n in np.arange(n1,n2):
		if multi: pool0.apply_async(tmp_deal,(n,noise_data,noise_cum,fsub,cumsub,n1,n2,multi,lock),error_callback=lambda x:print(x))
		else: tmp_deal(n,noise_data,noise_cum,fsub,cumsub,n1,n2,multi)
	if multi:
		pool0.close()
		pool0.join()
		noise_data=np.array(noise_data)
		noise_cum=np.array(noise_cum)
	#
	tmp_noise=noise_data[:,0].sum(1)/noise_cum
	sorts=np.argsort(tmp_noise)
	noise_data,noise_cum=noise_data[sorts],noise_cum[sorts]
	noisen_center=np.int64(noisen//2)
	if noisen>6:
		noise_off=noise_data[2:(noisen_center-1)].sum(0)/noise_cum[2:(noisen_center-1)].sum().reshape(-1,1)
		noise_on=noise_data[(noisen_center+2):-2].sum(0)/noise_cum[(noisen_center+2):-2].sum().reshape(-1,1)-noise_off
	elif noisen>=2:
		if not args.subi: sys.stdout.write('Warning: The noise data used in calulation is too short to get accurate calibration parameters.\n')
		if noisen==2: tmp=0
		else: tmp=1
		noise_off=noise_data[:noisen_center].sum(0)/noise_cum[:noisen_center].sum().reshape(-1,1)
		noise_on=noise_data[(noisen_center+tmp):].sum(0)/noise_cum[(noisen_center+tmp):].sum().reshape(-1,1)-noise_off
	else:
		parser.error('The noise data used in calulation is too short.')
	noise_a12,noise_a22=noise_on[:2]
	noise_dphi=np.arctan2(noise_on[3],noise_on[2])
	noise_cos,noise_sin=np.cos(noise_dphi),np.sin(noise_dphi)
	return np.array([noise_a12,noise_a22,noise_cos,noise_sin])
#
if noise_mark=='fits':	# the calibration data can be original fits data of noise or processed parameters in ld file
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
				noise_data[i]=deal_seg(jumps[i],jumps[i+1],multi=args.multi)
				noise_time[i]=(cumlen_noise[jumps[i+1]-1]+cumlen_noise[jumps[i]]+noise_len[jumps[i]]*tsamp/86400)/2
			cal_mode='trend'
		else:
			if noisenum>1:
				noise_time=cumlen_noise+noise_len*tsamp/86400/2
				if args.multi:
					def trend_deal(i):
						noise_data[i]=deal_seg(i,i+1)
					noise_data=Manager().list([np.zeros([4,nchan])]*noisenum)
					pool1=Pool(processes=args.multi)
				else: noise_data=np.zeros([noisenum,4,nchan])
				for i in np.arange(noisenum):
					if args.multi: pool1.apply_async(trend_deal,(i,))
					else: noise_data[i]=deal_seg(i,i+1)
				if args.multi:
					pool1.close()
					pool1.join()
					noise_data=np.array(noise_data)
				cal_mode='trend'
				file_nseg=noisenum
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
			noise_data[i]=deal_seg(jumps[i],jumps[i+1],multi=args.multi)
		noise_data=noise_data.mean(0)
		cal_mode='single'
elif noise_mark=='ld':
	if noise_info['calibration_info']['cal_mode']=='trend':
		noise_time0=noise_info['data_info']['stt_time']
		noise_time=np.array(noise_info['calibration_info']['seg_time'])
		if file_t0[0]<((1.25*noise_time[0]-0.25*noise_time[-1])+noise_time0) or file_t0[-1]>((1.25*noise_time[-1]-0.25*noise_time[0])+noise_time0):
			parser.error('The file time is out of the extrapolating range.')
		noise_data=noise.read_data().reshape(nchan,2,4).transpose(1,2,0)
		cal_mode='trend'
	elif noise_info['calibration_info']['cal_mode']=='seg':
		noise_time0=noise_info['data_info']['stt_time']
		noise_time=np.array(noise_info['calibration_info']['seg_time'])
		noise_time_judge=((noise_time+noise_time0-file_t0[0])>(-cal_trend_eff/24.))&((noise_time+noise_time0-file_t0[-1])<(cal_trend_eff/24.))
		noise_time_judge_1=((noise_time+noise_time0-file_t0[0])>(-cal_seg_eff/24.))&((noise_time+noise_time0-file_t0[-1])<(cal_seg_eff/24.))
		noise_time_index=np.arange(len(noise_time))[noise_time_judge]
		noise_time_index_1=np.arange(len(noise_time))[noise_time_judge_1]
		nseg=len(noise_time_index)
		nseg_1=len(noise_time_index_1)
		#print(noise_time,file_t0[0]-noise_time0)
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
	info['calibration_info']={'cal_mode':cal_mode}
	info['calibration_info']['cal']=noise_data.reshape(-1,npol,nchan)[:,:,chanstart:chanend].tolist()
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
# the delay in telescope line transmission
roach_delay=8.192e-6*3
gps_delay=1e-5
transline_delay=2e-5
light_delay=(300.0+141.96)/3.0e8
delay=transline_delay+light_delay-gps_delay-roach_delay
#
stt_time=info['data_info']['stt_time']
end_time=stt_time+tsamp*nbin0/86400.0+60./86400
stt_time-=60./86400
time0=file_time[0]
freq0=freq_start+channel_width*0.5
freq1=freq_end+channel_width*0.5
if args.period or (not pepoch):
	if args.period:
		period=args.period
	phase=np.arange(nbin0)*tsamp/period
	info['additional_info']={'phase0':0}
	nperiod=int(np.ceil(np.max(phase)))
	stt_sec=time0[:-1].sum()-delay+offs_sub-tsamp*nsblk/2.0-dm/df[f]**2*pm.dm_const
	stt_date=time0[-1]+stt_sec//86400
	stt_sec=stt_sec%86400
else:	# generate the Chebyshev polynomials based on the time and frequency
	chebx_test0=nc.chebpts1(args.ncoeff)
	chebx_test=np.concatenate(([-1],chebx_test0,[1]),axis=0)
	second_test=(chebx_test+1)/2*nbin0*tsamp+file_time[0][:-1].sum()-delay+offs_sub-tsamp*nsblk/2.0
	time_test=te.time(file_time[0][-1]*np.ones(args.ncoeff+2),second_test,scale=telename)
	times_test=te.times(time_test)
	timing_test_end=pm.psr_timing(psr,times_test,freq1)
	timing_test_start=pm.psr_timing(psr,times_test,freq0)
	phase_start=timing_test_end.phase.integer[0]+1
	phase_end=timing_test_start.phase.integer[-1]
	phase=timing_test_end.phase.integer-phase_start+timing_test_end.phase.offset
	nperiod=phase_end-phase_start
	period=((time_test.date[-1]-time_test.date[0])*time_test.unit+time_test.second[-1]-time_test.second[0])/(timing_test_end.phase.integer[-1]-timing_test_end.phase.integer[0]+timing_test_end.phase.offset[-1]-timing_test_end.phase.offset[0])
	cheb_end=nc.chebfit(chebx_test,timing_test_end.phase.integer-phase_start+timing_test_end.phase.offset,args.ncoeff-1)
	roots=nc.chebroots(cheb_end)
	roots=np.real(roots[np.isreal(roots)])
	root=roots[np.argmin(np.abs(roots))]
	stt_time_test=af.cal_time(psr,int(phase_start),ttest=file_time[0][-1]+((root+1)/2*nbin0*tsamp+file_time[0][:-1].sum()-delay+offs_sub-tsamp*nsblk/2.0)/86400,telescope=telename,freq=np.inf)
	stt_sec=stt_time_test.second[0]
	stt_date=stt_time_test.date[0]
	info['additional_info']={'phase0':int(phase_start)}
	ncoeff_freq=10
	phase_tmp=np.zeros([ncoeff_freq,args.ncoeff+2])
	disp_tmp=np.zeros(ncoeff_freq)
	cheby=nc.chebpts1(ncoeff_freq)
	freqy=(cheby+1)/2*bandwidth+freq0
	for i in np.arange(ncoeff_freq):
		timing_test=pm.psr_timing(psr,times_test,freqy[i])
		disp_tmp[i]=((timing_test.tdis1+timing_test.tdis2)/period).mean()
		phase_tmp[i]=timing_test.phase.integer-phase_start+timing_test.phase.offset+disp_tmp[i]
	coeff_freq=np.polyfit(1/freqy,disp_tmp,4)
	coeff=nc.chebfit(chebx_test,nc.chebfit(cheby,phase_tmp,1).T,args.ncoeff-1)
	info['folding_info']={'predictor':coeff.tolist()}
	info['folding_info']['predictor_freq']=coeff_freq.tolist()
#
info['data_info']['stt_sec']=stt_sec
info['data_info']['stt_date']=int(stt_date)
info['data_info']['stt_time']=stt_date+stt_sec/86400.0
#
info['data_info']['nperiod']=int(nperiod)
info['data_info']['period']=period
#
nbin_max=(nbin0-1)/(np.max(phase)-np.min(phase))
if args.nbin:
	nbin=args.nbin
	if nbin>nbin_max:
		temp_multi=1
	else:
		temp_multi=2**(np.int16(np.log2(nbin_max/nbin)))
else:
	nbin=2**np.int16(np.log2(nbin_max))
	temp_multi=1
info['data_info']['nbin']=int(nbin)
info['data_info']['length']=period*nperiod
#
totalbin=nperiod*nbin*temp_multi
dphasebin=1./(nbin*temp_multi)
df=freq0+np.arange(nchan_new)*channel_width
df0=(df-freq0)/(freq1-freq0)*2-1
#
d=ld.ld(name+'.ld')
if info['data_info']['mode']=='subint':
	info['data_info']['nsub']=int(np.ceil(nperiod*1.0/sub_nperiod))
	tpsub=np.zeros([nchan_new,npol,nbin],dtype=np.float64)
	sub_nperiod_last=(nperiod-1)%sub_nperiod+1
	info['data_info']['sub_nperiod']=int(sub_nperiod)
	info['data_info']['sub_nperiod_last']=int(sub_nperiod_last)
	tpsubn=np.zeros(nchan_new)
elif info['data_info']['mode']=='single':
	sub_nperiod=1
	info['data_info']['nsub']=int(nperiod)
	info['data_info']['sub_nperiod']=1
	info['data_info']['sub_nperiod_last']=1
info['data_info']['chan_weight']=(weight*(2**int(np.log2(period*info['data_info']['sub_nperiod']/nbin/tsamp))*1.5)).tolist()
info['history_info']['file_time']=[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())]
d.write_shape([nchan_new,info['data_info']['nsub'],nbin,npol])
#
def write_data(ldfile,data,startbin,channum,lock=0):
	#if args.multi: lock.acquire()
	d.__write_chanbins_add__(data.T,startbin,channum)
	#if args.multi: lock.release()
#
def gendata(cums,data,tpsub=0,tpsubn=0,last=False,first=True,lock=0):	# analyze the phase bin of the data
	if args.reverse or (not bw_sign):
		if nchan==chanend:
			data=data[(nchan-chanstart-1)::-1]
		else:
			data=data[(nchan-chanstart-1):(nchan-chanend-1):-1]
	else:
		data=data[chanstart:chanend]
	if args.period:
		dt=np.array([np.arange(nsblk*cums-1.5,nsblk*cums+nsblk+2.5)*tsamp]*nchan_new)
	else:
		dt=np.arange(nsblk*cums-1.5,nsblk*cums+nsblk+2.5)/nbin0*2-1
	for f in np.arange(nchan_new):
		if f+chanstart in zchan: continue
		if args.period:
			if dm:
				phase=(dt+dm/df[f]**2*pm.dm_const)/period
			else:
				phase=dt/period
		else:
			phase=nc.chebval2d(dt,df0[f]*np.ones_like(dt,dtype=np.float64),coeff)-np.polyval(coeff_freq,1/df[f])
		newphase=np.arange(phase[0]//dphasebin+1,phase[-1]//dphasebin+1,dtype=np.int64)
		if newphase[-1]<0 or newphase[0]>=totalbin:
			continue
		newphase=newphase[(newphase>=0) & (newphase<totalbin)]
		tpdata=np.zeros([npol,len(newphase)])
		for p in np.arange(npol):
			tpdata[p]=np.interp(newphase,phase/dphasebin,data[f,p,:])
		if temp_multi>1:
			startphase,phaseres0=divmod(newphase[0],temp_multi)
			phaseres0=np.int64(phaseres0)
			nphase=np.int64(np.ceil((newphase[-1]+1-newphase[0]+phaseres0)*1.0/temp_multi))
			tpdata=np.concatenate((np.zeros([npol,phaseres0]),tpdata,np.zeros([npol,nphase*temp_multi-newphase[-1]-1+newphase[0]-phaseres0])),axis=1).reshape(npol,nphase,temp_multi).mean(2)
		else:
			startphase=newphase[0]
			nphase=newphase.size
		if info['data_info']['mode']=='single':
			write_data(d,tpdata,startphase,f,lock=lock)
		else:
			startperiod,phaseres1=divmod(startphase,nbin)
			phaseres1=np.int64(phaseres1)
			file_nperiod=np.int64(np.ceil((nphase+phaseres1)*1.0/nbin))
			startsub,periodres=divmod(startperiod,sub_nperiod)
			periodres=np.int64(periodres)
			file_nsub=np.int64(np.ceil((file_nperiod+periodres)*1.0/sub_nperiod))
			if file_nsub>1 or newphase[-1]==totalbin-1:
				file_sub_data=np.zeros([npol,file_nsub,nbin])
				if newphase[-1]==totalbin-1:
					if newphase[0]==0:
						tpdata=tpdata.reshape(npol,-1,nbin)
						file_sub_data[:,:-1]=tpdata[:,:(-sub_nperiod_last)].reshape(npol,file_nsub-1,sub_nperiod,nbin).sum(2)
						file_sub_data[:,-1]=tpdata[:,(-sub_nperiod_last):].sum(1)
					elif file_nsub>1:
						tpsub[f,:,phaseres1:]+=tpdata[:,:(nbin-phaseres1)]
						tpdata=tpdata[:,(nbin-phaseres1):].reshape(npol,-1,nbin)
						file_sub_data[:,1:-1]=tpdata[:,(sub_nperiod-periodres-1):(-sub_nperiod_last)].reshape(npol,file_nsub-2,sub_nperiod_last,nbin).sum(2)
						file_sub_data[:,0]=tpsub[f]+tpdata[:,:(sub_nperiod-periodres-1)].sum(1)
						file_sub_data[:,-1]=tpdata[:,(-sub_nperiod_last):].sum(1)
					else:
						tpsub[f,:,phaseres1:]+=tpdata[:,:(nbin-phaseres1)]
						tpdata=tpdata[:,(nbin-phaseres1):].reshape(npol,-1,nbin)
						file_sub_data[:,0]=tpsub[f]+tpdata.sum(1)
					write_data(d,file_sub_data.reshape(npol,-1)/sub_nperiod,startsub*nbin,f,lock=lock)
				else:
					periodres_left=sub_nperiod-periodres
					periodres_right=(file_nsub-1)*sub_nperiod-periodres
					phaseres_left=periodres_left*nbin-phaseres1
					phaseres_right=periodres_right*nbin-phaseres1
					phaseres_left1=nbin-phaseres1
					phaseres_right1=(file_nperiod-1)*nbin-phaseres1
					file_sub_data[:,1:-1]=tpdata[:,phaseres_left:phaseres_right].reshape(npol,file_nsub-2,sub_nperiod,nbin).sum(2)
					file_sub_data[:,0]=tpdata[:,phaseres_left1:phaseres_left].reshape(npol,sub_nperiod-periodres-1,nbin).sum(1)
					file_sub_data[:,0,phaseres1:]+=tpdata[:,:phaseres_left1]
					file_sub_data[:,-1]=tpdata[:,phaseres_right:phaseres_right1].reshape(npol,file_nperiod-1-periodres_right,nbin).sum(1)
					file_sub_data[:,-1,:(nphase-phaseres_right1)]+=tpdata[:,phaseres_right1:]
					if tpsubn[f]<=startsub:
						file_sub_data[:,0]+=tpsub[f]
					else:
						file_sub_data[:,1]+=tpsub[f]
					write_data(d,file_sub_data[:,:-1].reshape(npol,-1)/sub_nperiod,startsub*nbin,f,lock=lock)
					tpsub[f]=file_sub_data[:,-1]
					tpsubn[f]=startsub+file_nsub-1
					if last:
						write_data(d,tpsub[f]/sub_nperiod,tpsubn[f]*nbin,f,lock=lock)
			else:
				if file_nperiod>1:
					phaseres_left=nbin-phaseres1
					phaseres_right=(file_nperiod-1)*nbin-phaseres1
					tpsub[f,:,phaseres1:]+=tpdata[:,:phaseres_left]
					tpsub[f,:,:(nphase-phaseres_right)]+=tpdata[:,phaseres_right:]
					if file_nperiod>2:
						tpsub[f]+=tpdata[:,phaseres_left:phaseres_right].reshape(npol,file_nperiod-2,nbin).sum(1)
				else:
					tpsub[f,:,phaseres1:(phaseres1+tpdata.shape[1])]+=tpdata
				if last:
					write_data(d,tpsub[f]/sub_nperiod,startsub*nbin,f,lock=lock)
	return tpsub,tpsubn
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
	if info['data_info']['mode']=='subint':
		tpsub=np.zeros([nchan_new,npol,nbin],dtype=np.float64)
		tpsubn=np.zeros(nchan_new,dtype=np.int64)
	elif info['data_info']['mode']=='single':
		tpsub=0
		tpsubn=0
	f=ps.open(filelist[n],mmap=True)
	fsub=f['SUBINT'].header['naxis2']
	if args.large_mem:
		cumsub=np.int64(file_len[:n].sum())
		dtmp=f['SUBINT'].data
		data=np.float64((dtmp['DATA']).reshape(fsub,nsblk,npol,nchan)*dtmp['dat_scl'].reshape(fsub,1,npol,nchan)+dtmp['dat_offs'].reshape(fsub,1,npol,nchan)).reshape(fsub*nsblk,npol,nchan)
		del f['SUBINT'].data
		f.close()
		tmp=gendata(cumsub,file_len[n]*nsblk,data,tpsub,tpsubn,last=True,lock=lock)
	else:
		for i in np.arange(fsub):
			cumsub=np.int64(file_len[:n].sum()+i)
			dtmp=f['SUBINT'].data[i]
			data=np.int16(dtmp['DATA'].reshape(nsblk,npol,nchan)*dtmp['dat_scl'].reshape(1,npol,nchan)+dtmp['dat_offs'].reshape(1,npol,nchan))
			del f['SUBINT'].data
			if args.cal:
				if cal_mode=='trend':
					dt=(np.arange(nsblk)+nsblk*cumsub)*tsamp/86400+stt_time-noise_time0
					noise_a12,noise_a22,noise_cos,noise_sin=np.polyval(noise_data,dt.reshape(-1,1,1)).transpose(1,0,2)
					noise_a12=np.where(noise_a12>0,1./noise_a12,0)
					noise_a22=np.where(noise_a22>0,1./noise_a22,0)
					noise_a1a2=np.sqrt(noise_a12*noise_a22)
					noise_cos=noise_cos*noise_a1a2
					noise_sin=noise_sin*noise_a1a2
				if pol_type=='AABBCRCI':
					aa0,bb0,cr0,ci0=noise_a12*data[:,0],noise_a22*data[:,1],noise_cos*data[:,2]+noise_sin*data[:,3],-noise_sin*data[:,2]+noise_cos*data[:,3]
					data=np.zeros([nchan,npol,nsblk+4])
					data[:,0,2:-2],data[:,1,2:-2],data[:,2,2:-2],data[:,3,2:-2]=(aa0+bb0).T,(aa0-bb0).T,(2*cr0).T,(2*ci0).T
				elif pol_type=='IQUV':
					noise_a1p2=(noise_a12+noise_a22)/2.0
					noise_a1m2=(noise_a12-noise_a22)/2.0
					ii,qq,uu,vv=noise_a1p2*data[:,0]-noise_a1m2*data[:,1],noise_a1p2*data[:,1]-noise_a1m2*data[:,0],noise_cos*data[:,2]+noise_sin*data[:,3],-noise_sin*data[:,2]+noise_cos*data[:,3]
					data=np.zeros([nchan,npol,nsblk+4])
					data[:,0,2:-2],data[:,1,2:-2],data[:,2,2:-2],data[:,3,2:-2]=ii.T,qq.T,uu.T,vv.T
			else:
				data_tmp=np.zeros([nchan,npol,nsblk+4])
				data_tmp[:,:,2:-2]=data.transpose(2,1,0)
				data=data_tmp
			tpsub,tpsubn=gendata(cumsub,data,tpsub,tpsubn,last=(i==(fsub-1)),first=i==0,lock=lock)
		f.close()
	gc.collect()
	if args.verbose:
		if args.multi: lock.acquire()
		sys.stdout.write('Processing the '+str(n+1)+'th fits file takes '+str(time.time()-timemark)+' second.\n')
		if args.multi: lock.release()
#
if args.multi:
	pool=Pool(processes=args.multi)
	lock=Manager().Lock()
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
if info['data_info']['sub_nperiod']!=info['data_info']['sub_nperiod_last']:
	data=d.read_period(info['data_info']['nsub']-1)
	data*=(info['data_info']['sub_nperiod']/info['data_info']['sub_nperiod_last'])
	d.write_period(data,info['data_info']['nsub']-1)
#
if args.cal:
	info['data_info']['pol_type']='IQUV'
	if cal_mode=='trend':
		info['calibration_info']['noise_time0']=noise_time0
else:
	info['data_info']['pol_type']=pol_type
d.write_info(info)
#

