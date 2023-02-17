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
parser.add_argument('-m',dest='mode',default='dat',help="data format to convert: dat")
parser.add_argument("-o","--output",dest="output",default="conv",help="output file name")
#
args=(parser.parse_args())
if not os.path.isfile(args.filename):
	parser.error('A valid ld file name is required.')
d=ld.ld(args.filename)
shape=tuple(d.read_shape())
info=d.read_info()
#
if args.mode=='dat':
	d1=np.memmap(args.output+'.dat',dtype=np.float64,mode='w+',shape=shape)
	del d1
	for i in range(shape[0]):
		d1=np.memmap(args.output+'.dat',dtype=np.float64,mode='r+',shape=shape)
		data=d.read_chan(i)
		d1[i]=data
		del d1
else:
	parser.error('The output file format is unrecognized.')
