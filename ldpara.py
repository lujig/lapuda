#!/usr/bin/env python
import numpy as np
import argparse as ap
import os,time,ld,sys
#
version='JigLu_20200930'
parser=ap.ArgumentParser(prog='ldpara',description='Show the parameters of ld file.',epilog='Ver '+version)
parser.add_argument('-v','--version',action='version',version=version)
parser.add_argument("filename",nargs='+',help="input ld file")
parser.add_argument('-c',dest='paras',default=0,help="parameter name list, include nsub, nchan, nbin, npol, stt_time, file_time, psr_name, period, nperiod, dm, freq, bw and length")
#
args=(parser.parse_args())
filelist=args.filename
for ldfile in filelist:
	if not os.path.isfile(ldfile):
		parser.error('Valid file names are required.')
	sys.stdout.write(ldfile+'\n')
	d=ld.ld(ldfile)
	info=d.read_info()
	#
	if not args.paras:
		quit()
	plist=args.paras.split(',')
	for pname in plist:
		if pname=='nsub':
			if 'compressed' in info.keys():
				sys.stdout.write(pname+' '+str(info['nsub_new'])+'\n')
			else:
				sys.stdout.write(pname+' '+str(info['nsub'])+'\n')
		elif pname=='nchan':
			if 'compressed' in info.keys():
				sys.stdout.write(pname+' '+str(info['nchan_new'])+'\n')
			else:
				sys.stdout.write(pname+' '+str(info['nchan'])+'\n')
		elif pname=='nbin':
			if 'compressed' in info.keys():
				sys.stdout.write(pname+' '+str(info['nbin_new'])+'\n')
			else:
				sys.stdout.write(pname+' '+str(info['nbin'])+'\n')
		elif pname=='npol':
			if 'compressed' in info.keys():
				sys.stdout.write(pname+' '+str(info['npol_new'])+'\n')
			else:
				sys.stdout.write(pname+' '+str(info['npol'])+'\n')
		elif pname=='shape':
			sys.stdout.write(pname+' '+str(tuple(d.read_shape()))+'\n')
		elif pname in ['stt_time', 'file_time', 'psr_name', 'nperiod', 'period', 'dm', 'length', 'mode']:
			sys.stdout.write(pname+' '+info[pname]+'\n')
		elif pname=='freq':
			sys.stdout.write(pname+' '+str((info['freq_end']+info['freq_start'])/2)+'\n')
		elif pname=='bw':
			sys.stdout.write(pname+' '+str(info['freq_end']-info['freq_start'])+'\n')
		elif pname=='begin_end':
			sys.stdout.write(pname+' '+str(info['phase0'])+'  '+str(info['phase0']+info['nperiod'])+'\n')
		elif pname=='phase0':
			sys.stdout.write(pname+' '+str(info['phase0'])+'\n')
		elif pname in info.keys():
			if pname=='zchan':
				sys.stdout.write('The zapped channels can not be shown.'+'\n')
			para=info[pname]
			if type(para) is not list:
				sys.stdout.write(pname+' '+para+'\n')
			elif len(filelist)==1 and len(plist)==1:
				sys.stdout.write(pname+'\n')
				for line in para:
					sys.stdout.write(line+'\n')
			else:
				sys.stdout.write('Parameter '+pname+' is a list, and it must be checked individually.'+'\n')
		else:
			sys.stdout.write('Parameter '+pname+' can not be found.'+'\n')
