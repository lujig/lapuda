#!/usr/bin/env python
import numpy as np
import numpy.polynomial.chebyshev as nc
import argparse as ap
import numpy.fft as fft
import os,ld,time,sys,copy
import adfunc as af
import time_eph as te
import psr_read as pr
import psr_model as pm
from itertools import product
#
version='JigLu_20231117'
parser=ap.ArgumentParser(prog='ldslc',description='slice the ld file.',epilog='Ver '+version)
parser.add_argument('-v','--version',action='version',version=version)
parser.add_argument("filename",help="input file to be compressed")
parser.add_argument('--cr','--channel_range',dest='crange',default=0,help="limit the frequency channel range CHAN0,CHAN1")
parser.add_argument('--fr','--frequency_range',dest='frange',default=0,help='limit the frequency (MHz) range FREQ0,FREQ1')
parser.add_argument('--sr','--subint_range',dest='srange',default=0,help='limit the sub-integration range SUBINT0,SUBINT1')
parser.add_argument('--tr','--time_range',dest='trange',default=0,help="limit the time (s) range TIME0,TIME1")
parser.add_argument('--pr','--polarization',dest='polar',default=0,help='select the output polarization index/indices')
parser.add_argument('--nt',dest='nt',default=0,type=np.int16,help="slice the file in NT pieces in time domain")
parser.add_argument('--nf',dest='nf',default=0,type=np.int16,help="slice the file in NF pieces in frequency domain")
parser.add_argument('--lt',dest='lt',default=0,type=np.float64,help="slice the file in pieces with length LT (s)")
parser.add_argument('--lf',dest='lf',default=0,type=np.float64,help="slice the file in pieces with bandwidth LF (MHz)")
parser.add_argument('--ns',dest='ns',default=0,type=np.float64,help="slice the file in pieces with NS sub-integrations")
parser.add_argument('--nc',dest='nc',default=0,type=np.float64,help="slice the file in pieces with NC frequency channgels")
parser.add_argument("-o","--output",dest="output",default="compress",help="the output file name")
args=(parser.parse_args())
command=['ldslc.py']
#
if not os.path.isfile(args.filename):
	parser.error('A valid ld file name is required.')
d0=ld.ld(args.filename)
info0=d0.read_info()
if info0['data_info']['mode'] not in ['single','subint','test']:
	parser.error('The data format does not support slicing .')
#
fflag=np.sum(list(map(bool,[args.crange,args.frange])))
tflag=np.sum(list(map(bool,[args.srange,args.trange])))
pflag=bool(args.polar)
stflag=np.sum(list(map(bool,[args.nt,args.lt,args.ns])))
sfflag=np.sum(list(map(bool,[args.nf,args.lf,args.nc])))
#
if fflag+tflag+pflag+stflag+sfflag==0:
	parser.error('At least one of data ranging or slicing flags is required.')
elif fflag==2:
	parser.error('At most one of flags --cr and --fr is required.')
elif tflag==2:
	parser.error('At most one of flags --sr and --tr is required.')
elif stflag>=2:
	parser.error('At most one of flags --nt, --lt and --ns is required.')
elif sfflag>=2:
	parser.error('At most one of flags --nf, --lf and --nc is required.')
#
nchan0=info0['data_info']['nchan']
nbin0=info0['data_info']['nbin']
nsub0=info0['data_info']['nsub']
npol0=info0['data_info']['npol']
pol0=info0['data_info']['pol_type']
length0=info0['data_info']['length']
freq_start0=info0['data_info']['freq_start']
freq_end0=info0['data_info']['freq_end']
sublen=info0['data_info']['sublen']
if 'chan_weight' in info0['data_info'].keys():
	chan_weight0=info0['data_info']['chan_weight']
if 'weights' in info0['data_info'].keys():
	weights0=info0['data_info']['weights']
bandwidth=freq_end0-freq_start0
channel_width=bandwidth/nchan0
#
# frange
if args.frange:
	command.append('--fr '+args.frange)
	freq_start,freq_end=np.float64(args.frange.split(','))
	if freq_start>freq_end:
		parser.error("Starting frequency larger than ending frequency.")
	elif freq_start<freq_start0 or freq_end>freq_end0:
		parser.error("Input frequency is overrange.")
	chanstart,chanend=np.int16(np.round((np.array([freq_start,freq_end])-freq_start0)/channel_width))
elif args.crange:
	command.append('--cr '+args.crange)
	chanstart,chanend=np.int16(args.crange.split(','))
	if chanstart>chanend:
		parser.error("Starting channel larger than ending channel.")
	elif chanstart<0 or chanend>nchan0:
		parser.error("Input channel number is overrange.")
else:
	chanstart,chanend=0,nchan0
freq_start,freq_end=np.array([chanstart,chanend])*channel_width+freq_start0
if chanend==nchan0: nchan_last=True
else: nchan_last=False
#
# trange
if args.trange:
	command.append('--tr '+args.trange)
	time_start,time_end=np.float64(args.trange.split(','))
	if time_start>time_end:
		parser.error("Starting time larger than ending time.")
	elif time_start<0 or time_end>length0:
		parser.error("Input time is overrange.")
	substart,subend=np.int16(np.round(np.array([time_start,time_end])/sublen))
elif args.srange:
	command.append('--sr '+args.srange)
	substart,subend=np.int16(args.srange.split(','))
	if substart>subend:
		parser.error("Starting frequency larger than ending frequency.")
	elif substart<0 or subend>nsub0:
		parser.error("Input frequency is overrange.")
else:
	substart,subend=0,nsub0
time_start,time_end=np.array([substart,subend])*sublen
time_end=min(time_end,length0)
if subend==nsub0: nsub_last=True
else: nsub_last=False
#
# polar
if args.polar:
	command.append('--pr '+args.polar)
	polar=np.sort(np.array(np.int8(args.polar.split(','))).reshape(-1))
	if polar[0]<0 or polar[-1]>=npol0:
		parser.error("Input polarization is overrange.")
	info0['data_info']['pol_type']=''.join(np.array(list(pol0)).reshape(npol0,-1)[polar].reshape(-1))
	info0['data_info']['npol']=len(polar)
else:
	polar=np.arange(npol0)
#
# slicet
if args.nt:
	command.append('--nt '+str(args.nt))
	if args.nt<0: parser.error('The time domain piece number cannot be negative.')
	elif args.nt>=subend-substart: print("Warning: The time domain piece number should be smaller than original sub-integration number.")
	nsub=int(np.round((subend-substart)/args.nt))
elif args.lt:
	command.append('--lt '+str(args.lt))
	if args.lt<0: parser.error('The time span of each piece cannot be negative.')
	elif args.lt<sublen: print('Warning: The time span of each piece should be larger than 1 sub-integration.')
	elif args.lt>=(time_end-time_start): print('Warning: The time span of each piece should be smaller than the original length.')
	nsub=int(np.round(max(sublen,min(time_end-time_start,args.lt))/sublen))
elif args.ns:
	command.append('--ns '+str(args.ns))
	if args.ns<0: parser.error('The sub-integration number in each piece cannot be negative.')
	elif args.ns>(subend-substart): print('Warning: The sub-integration number in each piece should be smaller than the original sub-integration number.')
	nsub=int(min(args.ns,subend-substart))
else:
	nsub=subend-substart
ntime=int(np.ceil((subend-substart)/nsub))
#
# slicef
if args.nf:
	command.append('--nf '+str(args.nf))
	if args.nf<0: parser.error('The frequency domain piece number cannot be negative.')
	elif args.nf>=chanend-chanstart: print("Warning: The frequency domain piece number should be smaller than original channel number.")
	nchan=int(np.round((chanend-chanstart)/args.nf))
elif args.lf:
	command.append('--lf '+str(args.lf))
	if args.lf<0: parser.error('The bandwidth of each piece cannot be negative.')
	elif args.lf<channel_width: print('Warning: The bandwidth of each piece should be larger than 1 channel.')
	elif args.lf>=(freq_end-freq_start): print('Warning: The bandwidth of each piece should be smaller than the original bandwidth.')
	nchan=int(np.round(max(channel_width,min(freq_end-freq_start,args.lf))/channel_width))
elif args.nc:
	command.append('--nc '+str(args.nc))
	if args.nc<0: parser.error('The frequency channel number in each piece cannot be negative.')
	elif args.nc>(chanend-chanstart): print('Warning: The frequency channel number in each piece should be smaller than the original channel number.')
	nchan=int(min(args.nc,chanend-chanstart))
else:
	nchan=chanend-chanstart
nfreq=int(np.ceil((chanend-chanstart)/nchan))
#
command=' '.join(command)
#
# output
name=args.output
if len(name)>3:
	if name[-3:]=='.ld':
		name=name[:-3]
if ntime*nfreq==1:
	if os.path.isfile(name+'.ld'): parser.error('The name of output file already existed. Please provide a new name.')
	names=[name+'.ld']
else:
	if ntime>1 and nfreq==1: names=list(map(lambda x:name+'_'+str(x)+'t.ld',range(ntime)))
	elif nfreq>1 and ntime==1: names=list(map(lambda x:name+'_'+str(x)+'f.ld',range(nfreq)))
	elif nfreq>1 and ntime>1:
		names=list(map(lambda x:name+'_'+str(x[0])+'t_'+str(x[1])+'f.ld',product(range(ntime),range(nfreq))))
	for i in names:
		if os.path.isfile(i): parser.error('The name of one of the output file already existed. Please provide a new name.')
#
for k in np.arange(nfreq):
	if k==nfreq-1:
		nchan1=chanend-chanstart-(nfreq-1)*nchan
	else:
		nchan1=nchan
	freq_start1=freq_start+nchan*channel_width*k
	freq_end1=freq_start1+nchan1*channel_width
	chanstart1=chanstart+k*nchan
	chanend1=chanstart1+nchan1
	files=[]
	infos=[]
	substarts=[]
	subends=[]
	for i in np.arange(ntime):
		phase0=info0['additional_info']['phase0']+info0['data_info']['sub_nperiod']*nsub*(i+substart)
		if 'pulsar_info' in info0.keys():
			psr=pr.psr(info0['pulsar_info']['psr_par'])
			stt_time=af.cal_time(psr,te.phase(int(phase0),0),telescope=info0['telescope_info']['telename'],ttest=info0['data_info']['stt_time']+sublen*nsub*(i+substart)/86400,freq=np.inf)
		else: stt_time=te.time(info0['data_info']['stt_date'],info0['data_info']['stt_sec']+sublen*nsub*(i+substart))
		stt_date=stt_time.date[0]
		stt_sec=stt_time.second[0]
		stt_time=stt_time.mjd[0]
		if nsub_last:
			if nsub1==1: sub_nperiod=info0['data_info']['sub_nperiod_last']
			else: sub_nperiod=info0['data_info']['sub_nperiod']
			sub_nperiod_last=info0['data_info']['sub_nperiod_last']
		else:
			sub_nperiod=info0['data_info']['sub_nperiod']
			sub_nperiod_last=info0['data_info']['sub_nperiod']
		#		
		if i==ntime-1 and nsub_last:
			nsub1=subend-substart-(ntime-1)*nsub
			length=time_end-time_start-(ntime-1)*sublen
			nperiod=sub_nperiod*(nsub1-1)+sub_nperiod_last
		else:
			nsub1=nsub
			length=sublen*nsub1
			nperiod=sub_nperiod*nsub1
		substart1=substart+nsub*i
		subend1=substart1+nsub1
		info=copy.deepcopy(info0)
		info['additional_info']['phase0']=int(phase0)
		info['data_info'].update({'freq_start':freq_start1,'freq_end':freq_end1,'nchan':int(nchan1),'length':length,'nperiod':int(nperiod),'nsub':int(nsub1),'sub_nperiod':int(sub_nperiod), 'sub_nperiod_last':int(sub_nperiod_last),'stt_date':int(stt_date),'stt_sec':stt_sec,'stt_time':stt_time})
		#
		if 'chan_weight' in info['data_info'].keys():
			info['data_info']['chan_weight']=chan_weight0[chanstart1:chanend1]
		if 'weights' in info['data_info'].keys():
			info['data_info']['weights']=np.ndarray(weights0)[chanstart1:chanend1,substart1:subend1].tolist()
		#
		if 'history_info' in info.keys():
			info['history_info']['history'].append(command)
			info['history_info']['file_time'].append(time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime()))
		else:
			info['history_info']={}
			info['history_info']['history']=[command]
			info['history_info']['file_time']=[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())]
		#
		substarts.append(substart1)
		subends.append(subend1)
		infos.append(info)
		files.append(ld.ld(names[i*nfreq+k]))
		files[i].write_shape([nchan1,nsub1,nbin0,len(polar)])
	#
	for s in np.arange(chanstart1,chanend1):
		data=d0.read_chan(s,pol=polar)
		for i in np.arange(ntime):
			files[i].write_chan(data[substarts[i]:subends[i]],s-chanstart1)
	#
	for i in np.arange(ntime):
		for s in infos[i].keys():
			for t in infos[i][s].keys():
				if type(infos[i][s][t])==np.int64: print(t)
		files[i].write_info(infos[i])		
#
