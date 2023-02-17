#!/usr/bin/env python
import numpy as np
import numpy.fft as fft
import numpy.polynomial.chebyshev as nc
import argparse as ap
import os,sys,time,ld
try:
	import astropy.io.fits as ps
except:
	import pyfits as ps
#
version='JigLu_20220221'
#
parser=ap.ArgumentParser(prog='ldcal',description='Dedisperse and Fold the psrfits data.',epilog='Ver '+version)
parser.add_argument('-v','--version', action='version', version=version)
parser.add_argument('--verbose', action="store_true",default=False,help="print detailed information")
parser.add_argument("filename",nargs='+',help="name of file or filelist")
parser.add_argument("--cal_period",dest="cal_period",default=0,type=np.float64,help="period of the calibration fits file (s)")
parser.add_argument("-o","--output",dest="output",default="psr",help="output file name")
parser.add_argument("-r","--reverse",action="store_true",default=False,help="reverse the band")
parser.add_argument("-t","--trend",action="store_true",default=False,help="fit the calibration parameter evolution")
parser.add_argument("-s","--subi",action="store_true",default=False,help="take one subint as the calculation unit")
parser.add_argument("-w","--overwrite",action="store_true",default=False,help="overwrite the existed output file")
args=(parser.parse_args())
command=['ldcal.py']
#
if args.verbose:
	sys.stdout.write('Analyzing the arguments...\n')
filelist=args.filename
filenum=len(filelist)
if filelist[0][-3:]=='.ld':
	mark='ld'
	file_times=[]
	segs=[]
	file_len=[]
	telename,npol,nchan,freq_start,freq_end,stt_time,seg_time,flen='',0,0,0,0,0,[],0
else:
	mark='fits'
	file_t0=[]
	file_time=[]
	file_len=[]
	telename,pol_type,npol,nchan,freq,bandwidth,tsamp,nsblk,bw_sign,stt_imjd,stt_smjd,stt_offs,nsub,offs_sub='','',0,0,0,0.0,0.0,0,True,0,0,0.0,0,0.0
#
def file_error(para,filetype):
	parser.error(filetype+" have different parameters: "+para+".")
#
def fits_check(fname,notfirst=True,filetype='Fits file'):
	if not os.path.isfile(fname):
		parser.error(filetype+' name '+fname+' '+'is invalid.')
	try:
		f=ps.open(filelist[i],mmap=True)
	except:
		parser.error(filetype+' '+fname+' '+' is invalid.')
	head=f['PRIMARY'].header
	subint=f['SUBINT']
	subint_header=subint.header
	subint_data=subint.data[0]
	global telename,pol_type,npol,nchan,freq,bandwidth,tsamp,nsblk,bw_sign,stt_imjd,stt_smjd,stt_offs,nsub
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
def ld_check(fname,notfirst=True,filetype='Ld file'):
	if not os.path.isfile(fname):
		parser.error(filetype+' name '+fname+' '+'is invalid.')
	try:
		f=ld.ld(filelist[i])
		finfo=f.read_info()
	except:
		parser.error(filetype+' '+fname+' is invalid.')
	global telename,npol,nchan,freq_start,freq_end,stt_time,seg_time,flen
	if finfo['mode']=='cal':
		if not finfo['cal_mode']=='seg':
			parser.error(filetype+' '+fname+' is not segmented calibration file.')
	else:
		parser.error(filetype+' '+fname+' is not calibration file.')
	if not notfirst:
		telename,npol,nchan,freq_start,freq_end=finfo['telename'],finfo['npol'],finfo['nchan'],finfo['freq_start'],finfo['freq_end']
	else:
		if telename!=finfo['telename']:
			file_error('telescope name',filetype)
		if npol!=finfo['npol']:
			file_error('number of polorisations',filetype)
		if nchan!=finfo['nchan']:
			file_error('number of channels',filetype)
		if (freq_start-finfo['freq_start'])>1e-3:
			file_error('start frequency',filetype)
		if (freq_end-finfo['freq_end'])>1e-3:
			file_error('end frequency',filetype)
		#
	flen=finfo['length']
	stt_time=finfo['stt_time']
	seg_time=list(np.float64([finfo['seg_time']]).reshape(-1)+stt_time)
#
for i in np.arange(filenum):
	if mark=='fits':
		fits_check(filelist[i],notfirst=i)
		#
		subint_t0=(offs_sub-tsamp*nsblk/2.0+stt_smjd+stt_offs)/86400.0+stt_imjd
		file_len.append(nsub*nsblk)
		file_t0.append(subint_t0)
	elif mark=='ld':
		ld_check(filelist[i],notfirst=i)
		file_times.append(stt_time)
		segs.append(seg_time)
		file_len.append(flen)
#
if mark=='fits':
	file_len,file_t0,filelist=np.array(file_len),np.array(file_t0),np.array(filelist)
	sorts=np.argsort(file_t0)
	file_len,file_t0,filelist=file_len[sorts],np.sort(file_t0),filelist[sorts]
	#
	channel_width=bandwidth*1.0/nchan
	#
	nbin=file_len.sum()
	stt_time=file_t0[0]
	freq_start,freq_end=np.array([-0.5,0.5])*nchan*channel_width+freq
	info={'nbin_origin':int(nbin),'telename':telename,'freq_start':freq_start,'freq_end':freq_end,'nchan':int(nchan),'tsamp_origin':tsamp,'stt_time':stt_time,'npol':int(npol),'mode':'cal','length':nbin*tsamp}
	if args.reverse:
		command.append('-r')
	if args.subi:
		command.append('-s')
		noisen=np.int64(args.cal_period//(tsamp*nsblk))
	else:
		noisen=np.int64(args.cal_period//tsamp)
elif mark=='ld':
	stt_time=np.min(file_times)
	seg_times=np.concatenate(segs)
	segis=list(map(lambda x: list(range(len(x))),segs))
	seg_sorts=np.argsort(seg_times)
	seg_sorts1=np.argsort(seg_sorts)
	lseg=np.cumsum(list(map(len,segs)))
	nseg=len(seg_times)
	for i in np.arange(nseg-1,0,-1):
		if np.abs(seg_times[i]-seg_times[seg_sorts1[seg_sorts[i]-1]])<1e-10:
			segi=np.where((lseg-i)>0)[0][0]
			tmpi=len(segs[segi])+i-lseg[segi]
			segs[segi].pop(tmpi)
			segis[segi].pop(tmpi)
	seg_times=np.concatenate(segs)
	seg_sorts=np.argsort(seg_times)
	lseg=list(map(len,segs))
	nseg=len(seg_times)
	info={'telename':telename,'freq_start':freq_start,'freq_end':freq_end,'nchan':int(nchan),'npol':int(npol),'mode':'cal','stt_time':stt_time,'length':np.sum(file_len)}
	if args.reverse:
		sys.stdout.write('Warning: Band of the ld file will not be reversed, and the flag \'-r\' will be ignored.\n')
	if args.subi:
		sys.stdout.write('Warning: The flag \'-s\' will be ignored.\n')
#
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
#
if args.verbose:
	sys.stdout.write('Constructing the output file...\n')
#
d=ld.ld(name+'.ld')
#
if args.trend:
	command.append('-t')
command=' '.join(command)
info['history']=[command]
#
sys.stdout.write('Processing the noise file...\n')
#
def deal_seg(n1,n2):
	cumsub=0
	noise_data=np.zeros([noisen,npol,nchan])
	noise_cum=np.zeros(noisen)
	for n in np.arange(n1,n2):
		f=ps.open(filelist[n],mmap=True)
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
	noise_off=noise_data[2:(noisen_center-1)].sum(0)/noise_cum[2:(noisen_center-1)].sum().reshape(-1,1)
	noise_on=noise_data[(noisen_center+2):-2].sum(0)/noise_cum[(noisen_center+2):-2].sum().reshape(-1,1)-noise_off
	noise_a12,noise_a22=noise_on[:2]
	noise_dphi=np.arctan2(noise_on[3],noise_on[2])
	noise_cos,noise_sin=np.cos(noise_dphi),np.sin(noise_dphi)
	return np.array([noise_a12,noise_a22,noise_cos,noise_sin])
#
if mark=='fits':
	jumps=np.abs((file_len*tsamp/86400.0+file_t0)[:-1]-file_t0[1:])>(tsamp/86400.0)
	file_nseg=jumps.sum()+1
	jumps=np.concatenate(([0],np.where(jumps)[0]+1,[filenum]))
	cumlen=file_t0-file_t0[0]
	if args.trend:
		if file_nseg>1:
			noise_time=np.zeros(file_nseg)
			noise_info=np.zeros([file_nseg,npol,nchan])
			for i in np.arange(file_nseg):
				noise_info[i]=deal_seg(jumps[i],jumps[i+1])
				noise_time[i]=(cumlen[jumps[i+1]-1]+cumlen[jumps[i]]+file_len[jumps[i]]*tsamp/86400)/2
		else:
			if filenum==1:
				parser.error('The evolution of calibration parameters cannot be obtained from only one calibration file.')
			noise_time=(cumlen[-1]+file_len[-1]*tsamp/86400)/2
			noise_info=np.zeros([filenum,npol,nchan])
			for i in np.arange(filenum):
				noise_info[i]=deal_seg(i,i+1)
		noise_info=np.polyfit(noise_time,noise_info.reshape(file_nseg,-1),1).reshape(2,npol,nchan)
		info['cal_mode']='trend'
		d.write_shape([nchan,2,1,npol])
		d.write_period(noise_info[0].T,0)
		d.write_period(noise_info[1].T,1)
	else:
		info['cal_mode']='seg'
		noise_time=np.zeros(file_nseg)
		d.write_shape([nchan,file_nseg,1,npol])
		for i in np.arange(file_nseg):
			noise_info=deal_seg(jumps[i],jumps[i+1])
			noise_time[i]=(cumlen[jumps[i+1]-1]+cumlen[jumps[i]]+file_len[jumps[i]]*tsamp/86400)/2
			d.write_period(noise_info.T,i)
elif mark=='ld':
	noise_data=np.zeros([nseg,nchan,npol])
	cum_seg=0
	for i in np.arange(filenum):
		ldi=ld.ld(filelist[i])
		for k in np.arange(lseg[i]):
			noise_data[cum_seg]=ldi.read_period(segis[i][k]).reshape(nchan,npol)
			cum_seg+=1
	noise_data=noise_data[seg_sorts]
	noise_time=seg_times[seg_sorts]-stt_time
	if args.trend:
		noise_info=np.polyfit(noise_time,noise_data.reshape(nseg,-1),1).reshape(2,nchan,npol)
		info['cal_mode']='trend'
		d.write_shape([nchan,2,1,npol])
		d.write_period(noise_info[0],0)
		d.write_period(noise_info[1],1)
	else:
		info['cal_mode']='seg'
		d.write_shape([nchan,nseg,1,npol])
		for i in np.arange(nseg):
			d.write_period(noise_data[i],i)
#
info['seg_time']=list(noise_time)
info['file_time']=time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())
d.write_info(info)
