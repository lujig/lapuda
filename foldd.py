#!/usr/bin/env python
import numpy as np
import numpy.fft as fft
import numpy.polynomial.chebyshev as nc
import argparse as ap
import os,sys,time,ld
import mpmath as mm
mm.mp.dps=30
import astropy.io.fits as ps
import gc
#
version='JigLu_20200925'
#
parser=ap.ArgumentParser(prog='dfpsr',description='Dedisperse and Fold the psrfits data.',epilog='Ver '+version)
parser.add_argument('-v','--version', action='version', version=version)
parser.add_argument('--verbose', action="store_true",default=False,help="print detailed information")
parser.add_argument("filename",help="name of file or filelist")
parser.add_argument("-o","--output",dest="output",default="psr",help="output file name")
parser.add_argument('-e','--pulsar_ephemeris',default=0,dest='par_file',help='input pulsar parameter file')
parser.add_argument("-p","--period",dest="period",default=0,type=np.float64,help="pulsar period (s)")
parser.add_argument("-c","--coefficients_num",dest="ncoeff",default=12,type=int,help="numbers of Chebyshev polynomial coefficients on time axis")
parser.add_argument("-b","--nbin",dest="nbin",default=0,type=int,help="number of phase bins in each period")
parser.add_argument("-s","--sublen",dest="subint",default=0,type=np.float64,help="length of a subint (s)")
args=(parser.parse_args())
command=['dfpsr.py']
#
if args.verbose:
	sys.stdout.write('Analyzing the arguments...\n')
#
if args.filename[-3:]=='.ld':
	d0=ld.ld(args.filename)
	info=d0.read_info()
	if info['mode']!="dedisperse":
		parser.error("The input ld file should be \'dedisperse\' mode.")
else:
	parser.error("The input file should be ld format.")
#
telename,psr_name,npol,nbin0,nchan,tsamp=info['telename'],info['psr_name'],np.int32(info['npol']),np.int32(info['nbin_origin']),np.int32(info['nchan']),np.float64(info['tsamp_origin'])
freq_start,freq_end,stt_time,stt_date,stt_sec=np.float64(info['freq_start']),np.float64(info['freq_end']),np.float64(info['stt_time_origin']),np.float64(info['stt_date']),np.float64(info['stt_sec'])
#
bandwidth=freq_end-freq_start
channel_width=bandwidth*1.0/nchan
freq=(freq_start+freq_end)/2.0
#
if args.par_file and args.period:
	parser.error('At most one of flags -n and -p is required.')
elif not args.period:
	if not args.par_file:
		os.system('psrcat -e '+psr_name+' > psr.par')
		par_file=open('psr.par','r')
		psr_par=par_file.readlines()
		par_file.close()
		if len(psr_par)<3:
			parser.error('A valid pulsar name is required.')
		par_file='psr.par'
	else:
		command.append('-e')
		par_file=open(args.par_file,'r')
		psr_par=par_file.readlines()
		par_file.close()
		par_file=args.par_file
		info['psr_par']=psr_par
	pepoch=False
	for line in psr_par:
		elements=line.split()
		if elements[0]=='PSRJ':
			psr_name=elements[1].strip('\n')
			info['psr_name']=psr_name
		elif elements[0]=='F0':
			period=1./np.float64(elements[1])
		elif elements[0]=='PEPOCH':
			pepoch=True
else:
	period=args.period
	command.append('-p '+str(args.period))
#
if args.subint:
	if args.subint<period:
		parser.error('Duration time of a subint is too short')
	elif args.subint<(1.5*period):
		sys.stdout.write('Warning: Duration time of a subint is too short, then the out put file is indeed single pulse mode.\n')
		info['mode']='single'
	else:
		info['mode']='subint'
		sub_nperiod=np.int64(round(args.subint/period))
		info['sublen']=period*sub_nperiod
		command.append('-s '+str(args.subint))
else:
	info['mode']='single'
#
command.append('-c '+str(args.ncoeff))
if args.nbin:
	command.append('-b '+str(args.nbin))
	if args.nbin>(period/tsamp):
		if args.subint:
			if args.nbin>(period/tsamp*sub_nperiod):
				parser.error('Provided phase bin number in each period is too large.')
		else:
			parser.error('Provided phase bin number in each period is too large.')
#
name=args.output
if os.path.isfile(name):
	parser.error('The name of output file already existed. Please provide a new name.')
if len(name)>3:
	if name[-3:]=='.ld':
		name=name[:-3]
#
command=' '.join(command)
info['history']+=' ; '+command
#
if args.verbose:
	sys.stdout.write('Constructing the output file...\n')
#
freq0=freq_start
freq1=freq_end
#
end_time=stt_time+tsamp*nbin0/86400.0+60./86400
stt_time-=60./86400
stt_time=str(int(stt_time))+str(stt_time%1)[1:]
end_time=str(int(end_time))+str(end_time%1)[1:]
if args.period or (not pepoch):
	if args.period:
		period=args.period
	phase=np.arange(nbin0)*tsamp/period
	info['phase0']=0
	nperiod=int(np.ceil(np.max(phase)))
else:
	os.popen('tempo2 -f '+par_file+' -pred \"'+telename+' '+stt_time+' '+end_time+' '+str(freq_start)+' '+str(freq_end)+' '+str(args.ncoeff)+' 2 '+str(int(tsamp*nbin0)+150)+'\"').close()
	predictor_file='t2pred.dat'
	#
	polyco=open(predictor_file,'r')
	lines=polyco.readlines()
	polyco.close()
	os.remove(predictor_file)
	os.remove('pred.tim')
	if not args.par_file:
		os.remove(par_file)
	coeff=[]
	predictor=[]
	for line in lines:
		predictor.append(line)
		elements=line.split()
		if elements[0]=='TIME_RANGE':
			t0=mm.mpf(elements[1])
			t1=mm.mpf(elements[2])
		elif elements[0]=='FREQ_RANGE':
			f0=np.float64(elements[1])
			f1=np.float64(elements[2])
		elif elements[0]=='DISPERSION_CONSTANT':
			dispc=np.float64(elements[1])
		elif elements[0]=='COEFFS':
			coeff.append(list(map(mm.mpf,elements[1:])))
		elif line=="ChebyModel END\n" or line=="ChebyModel END":
			break
	info['predictor']=predictor[1:]
	coeff=np.array(coeff)
	coeff[0,:]/=2.0
	coeff[:,0]/=2.0
	tmp=int(coeff[0,0])
	coeff[0,0]-=tmp
	coeff=np.float64(coeff)
	t0=np.float64(t0-stt_date)*86400.0
	t1=np.float64(t1-stt_date)*86400.0
	dt=(np.array([0,nbin0])*tsamp+stt_sec-t0)/(t1-t0)*2-1
	coeff1=coeff.sum(1)
	phase=nc.chebval(dt,coeff1)+dispc/freq_end**2
	phase0=np.ceil(phase[0])
	nperiod=int(np.floor(phase[-1]-phase0+dispc/freq_start**2-dispc/freq_end**2))
	coeff[0,0]-=phase0
	coeff1=coeff.sum(1)
	roots=nc.chebroots(coeff1)
	roots=np.real(roots[np.isreal(roots)])
	root=roots[np.argmin(np.abs(roots-dt[0]))]
	period=1./np.polyval(np.polyder(nc.cheb2poly(coeff1)[::-1]),root)/2.0*(t1-t0)
	stt_sec=(root+1)/2.0*(t1-t0)+t0
	stt_date=stt_date+stt_sec//86400
	stt_sec=stt_sec%86400
	info['phase0']=int(phase0)+tmp
	phase-=phase0
#
info['nperiod']=nperiod
info['period']=period
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
info['nbin']=nbin
info['length']=period*nperiod
#
totalbin=nperiod*nbin*temp_multi
dphasebin=1./(nbin*temp_multi)
df=freq_start+np.arange(nchan)*channel_width
#
d=ld.ld(name+'.ld')
if info['mode']=='subint':
	info['nsub']=int(np.ceil(nperiod*1.0/sub_nperiod))
	tpsub=np.zeros([nchan_new,npol,nbin],dtype=np.float64)
	sub_nperiod_last=(nperiod-1)%sub_nperiod+1
	info['sub_nperiod']=sub_nperiod
	info['sub_nperiod_last']=sub_nperiod_last
	tpsubn=np.zeros(nchan_new)
elif info['mode']=='single':
	info['nsub']=nperiod
info['file_time']=time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())
d.write_shape([nchan,info['nsub'],nbin,npol])
#
if args.period:
	dt=np.array([np.arange(nbin0)*tsamp]*nchan_new)
else:
	dt=(np.arange(nbin0)*tsamp-tsamp+np.float64(info['stt_sec'])-t0)/(t1-t0)*2-1
for f in np.arange(nchan):
	if args.period:
		phase=dt/period
	else:
		phase=nc.chebval2d(dt,np.ones_like(dt),coeff)+dispc/freq_end**2
	newphase=np.arange(totalbin)
	data=d0.read_chan(f)[0].T
	tpdata=np.zeros([npol,totalbin])
	for p in np.arange(npol):
		tpdata[p]=np.interp(newphase,phase/dphasebin,data[p,:])
	if temp_multi>1:
		tpdata=tpdata.reshape(npol,-1,temp_multi).sum(2)
	if info['mode']=='single':
		d.write_chan(tpdata.T,f)
	else:
		tpdata=np.concatenate((tpdata,np.zeros([npol,(sub_nperiod-sub_nperiod_last)*nbin])),axis=1).reshape(npol,nsub,sub_nperiod,nbin).sum(2).reshape(npol,nsub*nbin).T
		d.write_chan(tpdata.T,f)
#
info['stt_sec']=stt_sec
info['stt_date']=stt_date
info['stt_time']=stt_date+stt_sec/86400.0
#
d.write_info(info)
#

