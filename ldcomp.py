#!/usr/bin/env python
import numpy as np
import numpy.polynomial.chebyshev as nc
import argparse as ap
import numpy.fft as fft
import os,ld,time,sys,copy
import adfunc as ad
import psr_model as pm
dirname=os.path.split(os.path.realpath(__file__))[0]
sys.path.append(dirname+'/doc')
import text
#
text=text.output_text('ldcomp')
version='JigLu_20180506'
parser=ap.ArgumentParser(prog='ldcomp',description=text.help,epilog='Ver '+version,add_help=False,formatter_class=lambda prog: ap.RawTextHelpFormatter(prog, max_help_position=50))
parser.add_argument('-h', '--help', action='help', default=ap.SUPPRESS,help=text.help_h)
parser.add_argument('-v','--version',action='version',version=version,help=text.help_v)
parser.add_argument("filename",nargs='+',help=text.help_filename)
parser.add_argument('-d',dest='dm',default=np.inf,type=np.float64,help=text.help_d)
parser.add_argument('--nchan',dest='nchan_new',default=0,type=np.int16,help=text.help_nchan)
parser.add_argument('-F',action='store_true',default=False,dest='fscrunch',help=text.help_F)
parser.add_argument('--nsub',dest='nsub_new',default=0,type=np.int16,help=text.help_nsub)
parser.add_argument('-T',action='store_true',default=False,dest='tscrunch',help=text.help_T)
parser.add_argument('-b','--nbin',dest='nbin_new',default=0,type=np.int16,help=text.help_b)
parser.add_argument('-B',action='store_true',default=False,dest='bscrunch',help=text.help_B)
parser.add_argument('-P',action='store_true',default=False,dest='pscrunch',help=text.help_P)
parser.add_argument('--fr','--frequency_range',default=0,dest='freqrange',help=text.help_fr)
parser.add_argument("-z","--zap",dest="zap_file",default=0,help=text.help_z)
parser.add_argument("-w","--weights",dest="weights",action='store_true',default=False,help=text.help_w)
parser.add_argument("-o","--output",dest="output",default='',help=text.help_o)
parser.add_argument("-e","--extension",dest="ext",default='',help=text.help_e)
args=(parser.parse_args())
command0=['ldcomp.py']
#
filelist0=args.filename
#
if args.output:
	if len(filelist0)!=1:
		parser.error(text.error_ofn)
	if args.ext:
		parser.error(text.error_ofne)
	output_mark=True
	output=args.output
	if len(output)>3:
		if output[-3:]=='.ld': output=output[:-3]
	if os.path.isfile(output):
		parser.error(text.error_ofno)
elif args.ext:
	output_mark=False
	ext=args.ext
	if ext[0]=='_': ext=ext[1:]
	if len(ext)>3:
		if ext[-3:]=='.ld': ext=ext[:-3]
else:
	output_mark=False
	ext='comp'	
#
errorfile=[]
filelist=[]
nchan,nsub,nbin,freq_start0,freq_end0=10**8,10**8,10**8,0,1e100
for i in filelist0:	# check the files
	if not os.path.isfile(i):
		parser.error(text.error_ue % i)
	else:
		try:
			filei=ld.ld(i)
			info=filei.read_info()
			if info['data_info']['mode'] not in ['single','subint']:
				raise
			nchan=min(nchan,info['data_info']['nchan'])
			nsub=min(nsub,info['data_info']['nsub'])
			nbin=min(nbin,info['data_info']['nbin'])
			freq_start0=max(freq_start0,info['data_info']['freq_start'])
			freq_end0=min(freq_end0,info['data_info']['freq_end'])
			filelist.append(i)
		except:
			errorfile.append(i)
#
if errorfile:
	print(text.warning_fn % ', '.join(errorfile))
#
fflag=np.sum(list(map(bool,[args.nchan_new,args.fscrunch])))
tflag=np.sum(list(map(bool,[args.nsub_new,args.tscrunch])))
bflag=np.sum(list(map(bool,[args.nbin_new,args.bscrunch])))
pflag=np.int16(args.pscrunch)
if fflag+tflag+bflag+pflag==0:
	parser.error(text.error_1fr)
elif fflag==2:
	parser.error(text.error_mff)
elif tflag==2:
	parser.error(text.error_mft)
elif bflag==2:
	parser.error(text.error_mfb)
elif np.sum(map(bool,[args.fscrunch,args.tscrunch,args.bscrunch]))==3:
	parser.error(text.error_joke)
#
if args.nchan_new:
	nchan_new=args.nchan_new
	command0.append('--nchan '+str(nchan_new))
	if nchan_new>nchan:
		parser.error(text.error_cnl)
elif args.fscrunch:
	command0.append('-F')
#
if args.nsub_new:
	command0.append('--nsub '+str(args.nsub_new))
	if args.nsub_new>nsub:
		parser.error(text.error_snl)
elif args.tscrunch:
	command0.append('-T')
#
if args.nbin_new:
	nbin_new=args.nbin_new
	command0.append('-b '+str(nbin_new))
	if nbin_new>nbin:
		parser.error(text.error_bnl)
elif args.bscrunch:
	command0.append('-B')
#
if args.pscrunch:
	command0.append('-P')
#
if args.zap_file:
	command0.append('-z')
	if not os.path.isfile(args.zap_file):
		parser.error(text.error_nzf)
	zchan=np.loadtxt(args.zap_file,dtype=np.int32)
	if np.max(zchan)>=nchan or np.min(zchan)<0:
		parser.error(text.error_zno)
#
if args.freqrange:
	command0.append('--fr '+args.freqrange)
	freq_start,freq_end=np.float64(args.freqrange.split(','))
	if freq_start>freq_end:
		parser.error(text.error_sfl)
	elif freq_start<freq_start0 or freq_end>freq_end0:
		parser.error(text.error_ifo)
#
for filei in filelist:
	command=copy.deepcopy(command0)
	d=ld.ld(filei)
	info=d.read_info()
	#
	nchan=int(info['data_info']['nchan'])
	nbin=int(info['data_info']['nbin'])
	nsub=int(info['data_info']['nsub'])
	npol=int(info['data_info']['npol'])
	#
	if args.nchan_new: nchan_new=args.nchan_new
	elif args.fscrunch: nchan_new=1
	else: nchan_new=nchan
	#
	if args.nsub_new:
		if args.nsub_new<nsub:
			info['data_info']['mode']='subint'
		nsub_new=int(np.ceil(nsub/np.ceil(nsub/args.nsub_new)))
		if nsub_new!=args.nsub_new:
			print(text.warning_snf.format(str(args.nsub_new),filei,str(nsub_new)))
	elif args.tscrunch:
		nsub_new=1
		info['data_info']['mode']='subint'
	else: nsub_new=nsub
	#
	if args.nbin_new: nbin_new=args.nbin_new
	elif args.bscrunch: nbin_new=1
	else: nbin_new=nbin
	#
	if args.pscrunch:
		info['data_info']['pol_type']='I'
		npol_new=1
	else: npol_new=npol
	#
	weights_mark=False
	weights_comp=False
	if args.weights:
		if 'weights' not in info['data_info'].keys():
			print(text.warning_ifw % filei)
		else:
			command.append('-w')
			weights_comp=True
			weights_mark=True
			weights=np.array(info['data_info']['weights'])
	elif 'weights' in info['data_info'].keys():
			weights_mark=True
	#
	weight=np.array(info['data_info']['chan_weight'])
	if args.zap_file:
		if weights_comp:
			zchan=list(set(np.where(weights.sum(1)==0)[0]).union(zchan))
		else:
			zchan=list(set(np.where(weight==0)[0]).union(zchan))
		if weights_mark: weights[zchan]=0		
		weight[zchan]=0
	else:
		if weights_comp: zchan=list(np.where(weights.sum(1)==0)[0])
		else: zchan=list(np.where(weight==0)[0])
	#
	freq_start0,freq_end0=info['data_info']['freq_start'],info['data_info']['freq_end']
	freq=(freq_start0+freq_end0)/2.0
	bandwidth=freq_end0-freq_start0
	channel_width=bandwidth/nchan
	if args.freqrange:
		freq_start,freq_end=np.float64(args.freqrange.split(','))
		chanstart,chanend=np.int16(np.round((np.array([freq_start,freq_end])-freq_start0)/channel_width))
	else:
		chanstart,chanend=0,nchan
	freq_start,freq_end=np.array([chanstart,chanend])*channel_width+freq_start0
	#i
	if args.dm is not np.inf:
		dmmodi=True
		new_dm=args.dm
		command.append('-d '+str(args.dm))
	elif 'additional_info' in info.keys():
		if 'best_dm' in info['additional_info'].keys():
			dmmodi=True
			new_dm=info['additional_info']['best_dm'][0]
		else: dmmodi=False
	else:
		dmmodi=False
	#
	command=' '.join(command)
	#
	if output_mark:
		d1=ld.ld(output+'.ld')
	else:
		output=filei[:-3]+'_'+ext
		tmpnum=1
		output0=output
		while os.path.isfile(output0+'.ld'):
			output0=output+'_'+str(tmpnum)
			tmpnum+=1
		if output!=output0:
			print(text.warning_sofn % (filei,output0))
		d1=ld.ld(output0+'.ld')
	#	
	if 'history_info' in info.keys():
		info['history_info']['history'].append(command)
		info['history_info']['file_time'].append(time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime()))
	else:
		info['history_info']={}
		info['history_info']['history']=[command]
		info['history_info']['file_time']=[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())]
	#
	d1.write_shape([nchan_new,nsub_new,nbin_new,npol_new])
	#
	def shift(y,x):	# cyclically shift y with phase difference x
		nsub,nbin,npol=y.shape
		if info['data_info']['mode']=='subint':
			fftp=fft.rfft(y,axis=1)
			ffts=fftp*np.exp(x*nsub*1j*np.arange(fftp.shape[1])).reshape(1,-1,1)
			fftr=fft.irfft(ffts,axis=1)
		elif info['data_info']['mode']=='single':
			fftp=fft.rfft(y.reshape(-1,npol),axis=0)
			ffts=fftp*np.exp(x*1j*np.arange(fftp.shape[0])).reshape(-1,1)
			fftr=fft.irfft(ffts,axis=0).reshape(nsub,nbin,npol)
		return fftr
	#
	nchan0=chanend-chanstart
	if dmmodi:	# use new DM value to modify data
		freq0=freq_start+channel_width/2.0
		freq1=freq_end+channel_width/2.0
		if 'dm' in info['data_info'].keys():
			dm_old=info['data_info']['dm']
		else:
			dm_old=0
		disp_time=1/np.arange(freq0,freq1,channel_width)**2*np.float64(new_dm-dm_old)*pm.dm_const
		disp=disp_time*np.pi*2.0/info['data_info']['period']/nsub
		disp=disp-np.min(disp)
		info['data_info']['dm']=new_dm
	res=nchan0
	tpdata=np.zeros([nsub,nbin,npol_new])
	weight_new=np.zeros(nchan_new)
	tpweight=0
	if weights_mark:
		weights_new=np.zeros(nchan_new,nsub_new)
		tpweights=np.zeros(nsub)
	i_new=0
	for i in np.arange(chanstart,chanend):	# fold in different frequency channel
		if res>nchan_new:
			res-=nchan_new
			if i in zchan: continue
			weight0=weight[i]
			tpweight+=weight0
			if weights_mark:
				weights0=weights[i]
				tpweights+=weights0
			if weights_comp: data0=d.read_chan(i)*weights0.reshape(-1,1,1)
			else: data0=d.read_chan(i)*weight0
			if npol_new==1:
				data0=data0[:,:,0].reshape(nsub,nbin,npol_new)
			if dmmodi:
				tpdata+=np.float64(shift(data0,disp[i-chanstart]))
			else:
				tpdata+=data0
		else:
			if i in zchan:
				chan_data=np.zeros([nsub,nbin,npol_new])
				weight0=0
				if weights_mark: weights0=np.zeros(nsub)
			else:
				weight0=weight[i]
				if weights_mark:
					weights0=weights[i]
				if weights_comp: data0=d.read_chan(i)*weights0.reshape(-1,1,1)
				else: data0=d.read_chan(i)*weight0
				if npol_new==1:
					data0=data0[:,:,0].reshape(nsub,nbin,npol_new)
				if dmmodi:
					chan_data=np.float64(shift(data0,disp[i-chanstart]))
				else:
					chan_data=data0
			tpweight+=weight0*(res*1.0/nchan_new)
			if weights_mark: tpweights+=weights0*(res*1.0/nchan_new)
			tpdata+=chan_data*(res*1.0/nchan_new)
			if nsub_new!=nsub:	# compress in sub-integration dimension
				sub_nperiod=np.ones(nsub)*info['data_info']['sub_nperiod']
				sub_nperiod[-1]=info['data_info']['sub_nperiod_last']
				if not weights_comp: tpdata*=sub_nperiod.reshape(-1,1,1)
				if nsub_new==1:
					sub_nperiod=[int(sub_nperiod.sum())]
					tpdata=tpdata.sum(0).reshape(1,nbin,npol_new)
					if weights_mark: tpweights=np.array(tpweights.sum()).reshape(-1)
					if not weights_comp: tpdata/=sub_nperiod[0]
					sub_nsub=sub_nperiod[0]/info['data_info']['sub_nperiod']
				else:
					sub_nsub=int(np.ceil(nsub/nsub_new))
					sub_nsub_resi=sub_nsub*nsub_new-nsub
					tpdata0=np.zeros([sub_nsub*nsub_new,nbin,npol_new])
					tpdata0[:nsub]=tpdata
					tpdata=tpdata0.reshape(nsub_new,-1,nbin,npol_new).sum(1)
					sub_nperiod=np.append(sub_nperiod,np.zeros(sub_nsub_resi)).reshape(nsub_new,-1).sum(1)
					if weights_mark: tpweights=np.append(tpweights,np.zeros(sub_nsub_resi)).reshape(nsub_new,-1).sum(1)
					if not weights_comp: tpdata/=sub_nperiod.reshape(-1,1,1)
			else: sub_nsub=1
			if nbin_new!=nbin:	# compress in phase bin dimension
				#tpdata=fft.irfft(fft.rfft(tpdata,axis=1)*np.exp(-(0.5/nbin_new-0.5/nbin)*1j*np.arange(nbin/2+1)).reshape(1,-1,1),axis=1)
				#if 2*nbin_new>=nbin:
				#	tpdata=fft.irfft(np.concatenate((tpdata,np.zeros([nsub_new,nbin_new+1-tpdata.shape[1]])),axis=1),axis=1).reshape(nsub_new,nbin_new,2,npol_new).sum(2)
				#else:
				#	tpdata=fft.irfft(tpdata[:,:(nbin_new+1)],axis=1).reshape(nsub_new,nbin_new,2,npol_new).sum(2)
				tpdata=tpdata.reshape(nsub_new,nbin_new,-1,npol_new).mean(2)
			if weights_mark: weights_new[i_new]=tpweights*(nbin/nbin_new)
			if weights_comp:
				jj=tpweights!=0
				tpdata[jj]/=tpweights[jj]
			else:
				if tpweight!=0: tpdata/=tpweight
			weight_new[i_new]=tpweight*sub_nsub*(nbin/nbin_new)
			d1.write_chan(tpdata,i_new)
			i_new+=1
			tpdata=chan_data*((nchan_new-res)*1.0/nchan_new)
			tpweight=weight0*((nchan_new-res)*1.0/nchan_new)
			res=nchan0-(nchan_new-res)
	#
	if nchan_new==1:	# create a spectrum for frequency-scrunched data
		data=d.period_scrunch()[:,:,0]*weight.reshape(-1,1)
		data[zchan]=0
		base,bin0=ad.baseline(data.mean(0),pos=True)
		spec=np.concatenate((data,data),axis=1)[:,bin0:(bin0+10)].mean(1)
		if 'additional_info' in info.keys(): info['additional_info']['spec']=list(spec)
		else: info['additional_info']={'spec':list(spec)}
	#
	if nchan_new!=nchan:
		if 'chan_weight_raw' in info['additional_info'].keys():
			info['additional_info'].pop('chan_weight_raw')
	#
	if nsub_new!=nsub:	# adjust the information which describe the weight in each sub-integration
		info['data_info']['sub_nperiod']=sub_nperiod[0]
		info['data_info']['sub_nperiod_last']=sub_nperiod[-1]
		info['data_info']['sublen']=float(info['data_info']['period']*sub_nperiod[0])
	#
	info['data_info']['chan_weight']=weight_new.tolist()	# adjust information
	if weights_mark: info['data_info']['weights']=weights_new.tolist()
	info['data_info']['nchan']=int(nchan_new)
	info['data_info']['nsub']=int(nsub_new)
	info['data_info']['nbin']=int(nbin_new)
	info['data_info']['npol']=int(npol_new)
	info['data_info']['freq_start']=freq_start
	info['data_info']['freq_end']=freq_end
	info['data_info']['compressed']=True
	d1.write_info(info)
	#!/usr/bin/env python
