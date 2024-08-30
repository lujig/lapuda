#!/usr/bin/env python
import numpy as np
import argparse as ap
import os,time,ld,sys
import adfunc as af
dirname=os.path.split(os.path.realpath(__file__))[0]
sys.path.append(dirname+'/doc')
import text
#
text=text.ldpara_text()
version='JigLu_20200930'
parser=ap.ArgumentParser(prog='ldpara',description=text.help,epilog='Ver '+version,add_help=False,formatter_class=lambda prog: ap.RawTextHelpFormatter(prog, max_help_position=50))
parser.add_argument('-h', '--help', action='help', default=ap.SUPPRESS,help=text.help_h)
parser.add_argument('-v','--version',action='version',version=version,help=text.help_v)
parser.add_argument("filename",nargs='+',help=text.help_filename)
parser.add_argument('-c',dest='paras',default=0,help=text.help_c)
parser.add_argument('-g',dest='group',default=0,help=text.help_g)
parser.add_argument('-H',dest='list',action="store_true",default=False,help=text.help_H)
parser.add_argument('-a',dest='all',action="store_true",default=False,help=text.help_a)
#
args=(parser.parse_args())
filelist=args.filename
#
def paral(info):
	set0=set(af.json2dic(info).keys())
	set0.add('shape')
	if {'freq_end','freq_start'}.issubset(set0):
		set0.update({'freq','bw'})
	if {'phase0','nperiod'}.issubset(set0):
		set0.add('begin_end')
	#set0.difference_update({'weight', 'chan_weight', 'cal'})
	return set0
#
def sout(info,plist,check=False):
	plist=np.sort(list(plist))
	li,ldic=af.parakey()
	if check: pout=[]
	else: pout=set(plist).difference(set(af.json2dic(info).keys()))
	for pname in plist:
		if pname in ['nsub', 'nchan', 'nbin', 'npol', 'stt_time', 'file_time', 'psr_name', 'nperiod', 'period', 'dm', 'length', 'mode','freq_start','freq_end','phase0']:
			sys.stdout.write(pname+' '+str(info[ldic[pname]][pname])+'\n')
		elif pname=='shape':
			sys.stdout.write(pname+' '+str(tuple(d.read_shape()))+'\n')
		elif pname=='freq':
			sys.stdout.write(pname+' '+str((info['data_info']['freq_end']+info['data_info']['freq_start'])/2)+'\n')
		elif pname=='bw':
			sys.stdout.write(pname+' '+str(info['data_info']['freq_end']-info['data_info']['freq_start'])+'\n')
		elif pname=='begin_end':
			sys.stdout.write(pname+' '+str(info['additional_info']['phase0'])+'  '+str(info['additional_info']['phase0']+info['data_info']['nperiod'])+'\n')
		elif pname not in pout:
			para=info[ldic[pname]][pname]
			if type(para) is not list:
				sys.stdout.write(pname+' '+str(para)+'\n')
			elif len(filelist)==1 and len(plist)==1:
				sys.stdout.write(pname+'\n')
				for line in para:
					print(line)
			else:
				print(pname,np.array(para))
		else:
			print(text.info_np % pname)
#
for ldfile in filelist:
	if not os.path.isfile(ldfile):
		parser.error(text.error_nf)
	sys.stdout.write(ldfile+'\n')
	d=ld.ld(ldfile)
	info=d.read_info()
	#
	if args.list:
		print(*paral(info))
	elif args.all:
		sout(info,paral(info),check=True)
	elif args.group:
		dirname=os.path.split(os.path.realpath(__file__))[0]
		sys.path.append(dirname+'/doc')
		from functools import reduce
		import class_doc as cd
		li=cd.ld_file_info
		glist=args.group.split(',')
		gdif=list(set(glist).difference(set(li.keys())))
		if gdif: print(text.info_ng % ', '.join(gdif))
		ginter=list(set(glist).intersection(set(li.keys())))
		pall=paral(info)
		for gname in ginter:
			sys.stdout.write(gname+'\n')
			sout(info,pall.intersection(set(li[gname].keys())),check=True)
	elif args.paras:
		plist=args.paras.split(',')
		sout(info,plist)

