#!/usr/bin/env python
import numpy as np
import numpy.polynomial.chebyshev as nc
import argparse as ap
import numpy.fft as fft
import os,ld,time
import adfunc as ad
import psr_model as pm
#
version='JigLu_20180506'
parser=ap.ArgumentParser(prog='ldcomp',description='Compress the ld file.',epilog='Ver '+version)
parser.add_argument('-v','--version',action='version',version=version)
parser.add_argument("filename",help="input file to be compressed")
parser.add_argument('-d',dest='dm',default=np.inf,type=np.float64,help="modify the dispersion measure to this value while compressing")
parser.add_argument('-f',dest='nchan_new',default=0,type=np.int16,help="frequency scrunch to NCHAN_NEW channels")
parser.add_argument('-F',action='store_true',default=False,dest='fscrunch',help='frequency scrunch to one channel')
parser.add_argument('-t',dest='nsub_new',default=0,type=np.int16,help="time scrunch to NSUB_NEW subints")
parser.add_argument('-T',action='store_true',default=False,dest='tscrunch',help='time scrunch to one subint')
parser.add_argument('-b',dest='nbin_new',default=0,type=np.int16,help="bin scrunch to NBIN_NEW bins")
parser.add_argument('-B',action='store_true',default=False,dest='bscrunch',help='bin scrunch to one bin')
parser.add_argument('-P',action='store_true',default=False,dest='pscrunch',help='only reserve intensity')
parser.add_argument('-r','--frequency_range',default=0,dest='freqrange',help='limit the frequency rangeFREQ0,FREQ1')
parser.add_argument("-z","--zap",dest="zap_file",default=0,help="file recording zap channels")
parser.add_argument("-o","--output",dest="output",default="compress",help="outputfile name")
args=(parser.parse_args())
command=['ldcomp.py']
#
if not os.path.isfile(args.filename):
	parser.error('A valid ld file name is required.')
d=ld.ld(args.filename)
info=d.read_info()
#
fflag=np.sum(list(map(np.bool,[args.nchan_new,args.fscrunch])))
tflag=np.sum(list(map(np.bool,[args.nsub_new,args.tscrunch])))
bflag=np.sum(list(map(np.bool,[args.nbin_new,args.bscrunch])))
pflag=np.int16(args.pscrunch)
if fflag+tflag+bflag+pflag==0:
	parser.error('At least one of flags -f, -F, -t, -T, -b and -B is required.')
elif fflag==2:
	parser.error('At most one of flags -f and -F is required.')
elif tflag==2:
	parser.error('At most one of flags -t and -T is required.')
elif fflag==2:
	parser.error('At most one of flags -b and -B is required.')
elif np.sum(map(np.bool,[args.fscrunch,args.tscrunch,args.bscrunch]))==3:
	parser.error('What do you want to do? To obtain a point?')
#
if 'compressed' in info.keys():
	nchan=int(info['nchan_new'])
	nbin=int(info['nbin_new'])
	nperiod=int(info['nsub_new'])
	npol=int(info['npol_new'])
else:
	nchan=int(info['nchan'])
	nbin=int(info['nbin'])
	nperiod=int(info['nsub'])
	npol=int(info['npol'])
#
if args.nchan_new:
	nchan_new=args.nchan_new
	command.append('-f '+str(nchan_new))
	if nchan_new>nchan:
		parser.error('The input channel number is larger than the channel number of dat file.')
elif args.fscrunch:
	command.append('-F')
	nchan_new=1
else:
	nchan_new=nchan
#
if args.nsub_new:
	nsub_new=args.nsub_new
	command.append('-t '+str(nsub_new))
	if nsub_new>nperiod:
		parser.error('The input subint number is larger than the period number of dat file.')
	elif nsub_new<nperiod:
		info['mode']='subint'
elif args.tscrunch:
	nsub_new=1
	info['mode']='subint'
	command.append('-T')
else:
	nsub_new=nperiod
#
if args.nbin_new:
	nbin_new=args.nbin_new
	command.append('-b '+str(nbin_new))
	if nbin_new>nbin:
		parser.error('The input bin number is larger than the bin number of dat file.')
elif args.bscrunch:
	nbin_new=1
	command.append('-B')
else:
	nbin_new=nbin
#
if args.pscrunch:
	npol_new=1
	command.append('-P')
	info['pol_type']='I'
else:
	npol_new=npol
#
if args.zap_file:
	command.append('-z')
	if not os.path.isfile(args.zap_file):
		parser.error('The zap channel file is invalid.')
	zchan=np.loadtxt(args.zap_file,dtype=np.int32)
	if np.max(zchan)>=nchan or np.min(zchan)<0:
		parser.error('The zapped channel number is overrange.')
	if 'zchan' in info.keys():
		info['zchan']=list(set(info['zchan']).union(zchan))
	else:
		info['zchan']=list(zchan)
	if nchan_new>1 and nchan_new!=nchan:
		info.pop('zchan')
else:
	if 'zchan' in info.keys():
		if nchan_new>1 and nchan_new!=nchan:
			info.pop('zchan')
	zchan=[]
#
freq_start,freq_end=info['freq_start'],info['freq_end']
freq=(freq_start+freq_end)/2.0
bandwidth=freq_end-freq_start
channel_width=bandwidth/nchan
if args.freqrange:
	command.append('-r '+args.freqrange)
	freq_start,freq_end=np.float64(args.freqrange.split(','))
	chanstart,chanend=np.int16(np.round((np.array([freq_start,freq_end])-freq)/channel_width+0.5*nchan))
	if freq_start>freq_end:
		parser.error("Starting frequency larger than ending frequency.")
	elif freq_start<(freq-bandwidth/2.0) or freq_end>(freq+bandwidth/2.0):
		parser.error("Input frequency is overrange.")
else:
	chanstart,chanend=0,nchan
#i
if args.dm is not np.inf:
	dmmodi=True
	new_dm=args.dm
	command.append('-d '+str(args.dm))
elif 'best_dm' in info.keys():
	dmmodi=True
	new_dm=info['best_dm'][0]
else:
	dmmodi=False
command=' '.join(command)
#
name=args.output
if os.path.isfile(name):
	parser.error('The name of output file already existed. Please provide a new name.')
if len(name)>3:
	if name[-3:]=='.ld':
		name=name[:-3]
d1=ld.ld(name+'.ld')
if 'history' in info.keys():
	info['history'].append(command)
	info['file_time'].append(time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime()))
else:
	info['history']=[command]
	info['file_time']=[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())]
#
d1.write_shape([nchan_new,nsub_new,nbin_new,npol_new])
#
def shift(y,x):
	nperiod,nbin,npol=y.shape
	if info['mode']=='subint':
		fftp=fft.rfft(y,axis=1)
		ffts=fftp*np.exp(x*nperiod*1j*np.arange(fftp.shape[1])).reshape(1,-1,1)
		fftr=fft.irfft(ffts,axis=1)
	elif info['mode']=='single':
		fftp=fft.rfft(y.reshape(-1,npol),axis=0)
		ffts=fftp*np.exp(x*1j*np.arange(fftp.shape[0])).reshape(-1,1)
		fftr=fft.irfft(ffts,axis=0).reshape(nperiod,nbin,npol)
	return fftr
#
nchan0=chanend-chanstart
if dmmodi:
	freq0=freq_start+channel_width/2.0
	freq1=freq_end-channel_width/2.0
	if 'dm' in info.keys():
		dm_old=info['dm']
	else:
		dm_old=0
	disp_time=1/np.linspace(freq0,freq1,nchan0)**2*np.float64(new_dm-dm_old)*pm.dm_const
	disp=disp_time*np.pi*2.0/info['period']/nperiod
	disp=disp-np.min(disp)
	info['dm']=new_dm
res=nchan0
tpdata=np.zeros([nperiod,nbin,npol_new])
i_new=0
for i in np.arange(chanstart,chanend):
	if res>nchan_new:
		res-=nchan_new
		if i in zchan: continue
		data0=d.read_chan(i)
		if npol_new==1:
			data0=data0[:,:,0].reshape(nperiod,nbin,npol_new)
		if dmmodi:
			tpdata+=np.float64(shift(data0,disp[i-chanstart]))
		else:
			tpdata+=data0
	else:
		if i in zchan:
			chan_data=np.zeros([nperiod,nbin,npol_new])
		else:
			data0=d.read_chan(i)
			if npol_new==1:
				data0=data0[:,:,0].reshape(nperiod,nbin,npol_new)
			if dmmodi:
				chan_data=np.float64(shift(data0,disp[i-chanstart]))
			else:
				chan_data=data0
		tpdata+=chan_data*(res*1.0/nchan_new)
		if nsub_new!=nperiod:
			if nsub_new==1:
				tpdata=tpdata.sum(0).reshape(1,nbin,npol_new)
			else:
				tpdata=fft.rfft(tpdata,axis=0)*np.exp(-(0.5/nsub_new-0.5/nperiod)*1j*np.arange(nperiod/2+1)).reshape(-1,1,1)
				if 2*nsub_new>=nperiod:
					tpdata=fft.irfft(np.concatenate((tpdata,np.zeros([nsub_new+1-tpdata.shape[0],nbin])),axis=0),axis=0).reshape(nsub_new,2,nbin,npol_new).sum(1)
				else:
					tpdata=fft.irfft(tpdata[:(nsub_new+1),:],axis=0).reshape(nsub_new,2,nbin,npol_new).sum(1)
		if nbin_new!=nbin:
			#tpdata=fft.irfft(fft.rfft(tpdata,axis=1)*np.exp(-(0.5/nbin_new-0.5/nbin)*1j*np.arange(nbin/2+1)).reshape(1,-1,1),axis=1)
			#if 2*nbin_new>=nbin:
			#	tpdata=fft.irfft(np.concatenate((tpdata,np.zeros([nsub_new,nbin_new+1-tpdata.shape[1]])),axis=1),axis=1).reshape(nsub_new,nbin_new,2,npol_new).sum(2)
			#else:
			#	tpdata=fft.irfft(tpdata[:,:(nbin_new+1)],axis=1).reshape(nsub_new,nbin_new,2,npol_new).sum(2)
			tpdata=tpdata.reshape(nsub_new,nbin_new,-1,npol_new).sum(2)
		d1.write_chan(tpdata,i_new)
		i_new+=1
		tpdata=chan_data*((nchan_new-res)*1.0/nchan_new)
		res=nchan0-(nchan_new-res)
#
if nchan_new==1:
	data=d.period_scrunch()[:,:,0]
	data[zchan]=0
	base,bin0=ad.baseline(data.mean(0),pos=True)
	spec=np.concatenate((data,data),axis=1)[:,bin0:(bin0+10)].mean(1)
	info['spec']=list(spec)
#
info['nchan_new']=int(nchan_new)
info['nsub_new']=int(nsub_new)
info['nbin_new']=int(nbin_new)
info['npol_new']=int(npol_new)
info['freq_start']=freq_start
info['freq_end']=freq_end
info['compressed']=True
d1.write_info(info)
#!/usr/bin/env python
