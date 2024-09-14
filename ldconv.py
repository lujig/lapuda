#!/usr/bin/env python
import argparse as ap
import numpy as np
import astropy.io.fits as ps
import os,ld,time,sys
dirname=os.path.split(os.path.realpath(__file__))[0]
sys.path.append(dirname+'/doc')
import text
#
text=text.output_text('ldconv')
version='JigLu_20201012'
parser=ap.ArgumentParser(prog='ldconv',description=text.help,epilog='Ver '+version,add_help=False,formatter_class=lambda prog: ap.RawTextHelpFormatter(prog, max_help_position=50))
parser.add_argument('-h', '--help', action='help', default=ap.SUPPRESS,help=text.help_h)
parser.add_argument('-v','--version',action='version',version=version,help=text.help_v)
parser.add_argument("filename",help=text.help_filename)
parser.add_argument('-m',dest='mode',default='dat',help=text.help_m)
parser.add_argument("-o","--output",dest="output",default="",help=text.help_o)
#
args=(parser.parse_args())
if not os.path.isfile(args.filename):
	parser.error(text.error_nfn)
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
	nind=max(5,np.ceil(np.log10(len(result)+1)))
	for i in np.arange(len(result)):
		if 'original_data_info' in info.keys():
			ftmp=info['original_data_info']['filenames'][i][0].split('/')[-1]
		else:
			ftmp='ToA_'+info['pulsar_info']['psr_name']+'_'+str(i).zfill(nind)+'.dat'
		fout.write('{:26s} {:10.6f} {:28s} {:4f} {:8s}'.format(ftmp,(result[i,5]+result[i,6])/2,str(int(result[i,0]))+str(result[i,1]/86400)[1:],result[i,2]*1e6,info['telescope_info']['telename'])+'\n')
	fout.close()
else:
	parser.error(text.error_nof)
