#!/usr/bin/env python
import argparse as ap
import numpy as np
import astropy.io.fits as ps
import os,ld,time
#
version='JigLu_20201012'
parser=ap.ArgumentParser(prog='ldconv',description='Convert the ld file to other data format.',epilog='Ver '+version)
parser.add_argument('-v','--version', action='version', version=version)
parser.add_argument("filename",help="name of ld file to convert")
parser.add_argument('-m',dest='mode',default='dat',help="data format to convert: ld->dat, ToA->tim")
parser.add_argument("-o","--output",dest="output",default="",help="output file name")
#
args=(parser.parse_args())
if not os.path.isfile(args.filename):
	parser.error('A valid ld file name is required.')
d=ld.ld(args.filename)
shape=tuple(d.read_shape())
info=d.read_info()
if not args.output: output=args.filename[:-3]
else: output=args.output
#
if args.mode=='dat':	# save the four dimension data to a dat file
	output_tmp='    '+output
	if output_tmp[-4:]=='.dat':output=output[:-4]
	d1=np.memmap(output+'.dat',dtype=np.float64,mode='w+',shape=shape)
	del d1
	weight=info['data_info']['chan_weight']
	for i in range(shape[0]):
		d1=np.memmap(output+'.dat',dtype=np.float64,mode='r+',shape=shape)
		data=d.read_chan(i)*weight[i]
		d1[i]=data
		del d1
elif info['data_info']['mode']=='ToA' and args.mode=='tim':
	output_tmp='    '+output
	if not output_tmp[-4:]=='.tim': output=output+'.tim'
	result=d.read_chan(0)[:,:,0]
	fout=open(output,'w')
	fout.write('FORMAT 1\n')
	nind=max(4,np.ceil(np.log10(len(result)+1)))
	for i in np.arange(len(result)):
		ftmp='ToA_'+info['pulsar_info']['psr_name']+'_'+str(i).zfill(nind)+'.dat'
		fout.write('{:26s} {:10.6f} {:28s} {:4f} {:8s}'.format(ftmp,(result[i,5]+result[i,6])/2,str(int(result[i,0]))+str(result[i,1]/86400)[1:],result[i,2]*1e6,info['telescope_info']['telename'])+'\n')
	fout.close()
else:
	parser.error('The output file format is unrecognized.')
