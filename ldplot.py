#!/usr/bin/env python
import numpy as np
import numpy.ma as ma
import numpy.fft as fft
from matplotlib.figure import Figure
import argparse as ap
import os,time,ld,sys
import warnings as wn
import adfunc as af
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Serif'
import subprocess as sp
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
mpl.use('TkAgg')
import warnings
from scipy import ndimage
import psutil
#
version='JigLu_20240530'
parser=ap.ArgumentParser(prog='ldplot',description='Plot the ld data file.')
parser.add_argument('-v','--version',action='version',version=version)
parser.add_argument('filename', nargs='+', help='input ld files')
parser.add_argument('-f',action='store_true',default=False,dest='fdomain',help='show the frequency domain image')
parser.add_argument('-t',action='store_true',default=False,dest='tdomain',help='show the time domain image')
parser.add_argument('-p',action='store_true',default=False,dest='profile',help='show the average pulse profile')
parser.add_argument('-d',action='store_true',default=False,dest='dysp',help='show a simple dynamic spectrum image')
parser.add_argument('--sec',action='store_true',default=False,dest='second',help='show the second spectrum image of the dynamic spectrum')
parser.add_argument('-l',action='store_true',default=False,dest='polarization',help='show the polarization infos')
parser.add_argument('-N',default=None, dest='mulplot',help='x,y divide the window into x by y panels')
parser.add_argument('--fr',default='',dest='frange',help='limit the frequency range to [f1,f2] MHz')
parser.add_argument('--cr',default='',dest='crange',help='limit the frequency range to [chan1,chan2]')
parser.add_argument('--tr',default='',dest='trange',help='limit the time range to [t1,t2] Seconds')
parser.add_argument('--sr',default='',dest='srange',help='limit the time range to [sub1,sub2]')
parser.add_argument('--br',default='',dest='prange',help='limit the phase range to [p1,p2], The full range is [0,1]')
parser.add_argument('--binr',default='',dest='brange',help='limit the phase range to [bin1,bin2]')
parser.add_argument('--polar',default=0,dest='polar',type=int,help='plot the specified polarization (0123 for IQUV,default=0)')
parser.add_argument('-r','--shift', type=float, default=0, help='shift the phase [-1,1], positive towards right, negative towards left')
#parser.add_argument('--rtail',action='store_true',default=False,dest='removetail',help='remove the last sub-integration')
parser.add_argument('-n','--norm',action='store_true',default=False,dest='norm',help='Normalize the data')
parser.add_argument('-s', '--savefigure',default='', help='save figures as the specified name, and the file type could be PNG, EPS or PDF')
args = parser.parse_args()
warnings.filterwarnings("ignore")
#
filelist0=args.filename
#
fflag=np.sum(list(map(bool,[args.crange,args.frange])))
tflag=np.sum(list(map(bool,[args.srange,args.trange])))
bflag=np.sum(list(map(bool,[args.prange,args.brange,args.shift])))
if fflag==2:
	parser.error('At most one of flags --cr and --fr is required.')
elif tflag==2:
	parser.error('At most one of flags --sr and --tr is required.')
elif bflag==2:
	parser.error('At most one of flags --br, --binr and -r is required.')
#
if args.crange:
	chanstart0,chanend0=np.int32(args.crange.split(','))
	if chanend0<=chanstart0 and chanend0!=-1: parser.error("Starting channel number larger than ending channel number.")
	elif chanstart0<0: parser.error("Input channel number is overrange.")
elif args.frange:
	freq_start0,freq_end0=np.float64(args.frange.split(','))
	if freq_end0<=freq_start0: parser.error("Starting frequency larger than ending frequency.")
	elif freq_start0<0: parser.error("Input frequency is overrange.")
#
if args.srange:
	substart0,subend0=np.int32(args.srange.split(','))
	if subend0<=substart0 and subend0!=-1 : parser.error("Starting sub-integration number larger than sub-integration number.")
	elif substart0<0: parser.error("Input sub-integration number is overrange.")
elif args.trange:
	time_start0,time_end0=np.float64(args.trange.split(','))
	if time_end0<=time_start0: parser.error("Starting time larger than ending time.")
	elif time_start0<0 or time_end0<0: parser.error("Input time is overrange.")
#
if args.brange:
	binstart0,binend0=np.int32(args.brange.split(','))
	if binstart0<0 or (binend0<0 and binend0!=-1): parser.error("Input bin number is overrange.")
elif args.prange:
	phase_start0,phase_end0=np.float64(args.frange.split(','))
	if phase_start0<0 or phase_end0<0 or phase_start0>1 or phase_end0>1: parser.error("Input channel number is overrange.")
	if phase_start0==phase_end0: parser.error("Starting phase must be unequal to ending phase.")
#
if args.polar:
	if args.polar>=4: parser.error("Input polarization is overrange.")
	if args.polarization:
		print("Warning: showing polarization needs not to specify polarization index.")
if args.polarization: dx,dy=6,8
else: dx,dy=8,6
#
plotflag=np.sum(list(map(bool,[args.fdomain,args.tdomain,args.profile,args.dysp,args.second,args.polarization])))
if plotflag>1:
	parser.error('At most one of flags -f, -t, -p, -d, -l and --sec is required.')
elif plotflag==0:
	parser.error('At least one of flags -f, -t, -p, -d, -l and --sec is required.')
#
if args.mulplot:
	row,column=np.int8(args.mulplot.split(','))
	if row>6 or column>10: parser.error('The multi-plot parameters is too large.')
else: row,column=1,1
# check the files
filelist=[]
ranges=[]
for fname in filelist0:
	if not os.path.isfile(fname):
		print('The ld file '+fname+' name does not exsit.')
		continue
	d=ld.ld(fname)
	info=d.read_info()
	if info['data_info']['mode'] not in ['single','subint']:
		print(fname+' is not a showable file.')
		continue
	#
	nchan=int(info['data_info']['nchan'])
	nbin=int(info['data_info']['nbin'])
	nsub=int(info['data_info']['nsub'])
	npol=int(info['data_info']['npol'])
	length=np.float64(info['data_info']['length'])
	freq_start,freq_end=np.float64(info['data_info']['freq_start']),np.float64(info['data_info']['freq_end'])
	channel_width=(freq_end-freq_start)/nchan #MHz
	sublen=info['data_info']['sublen'] # seconds
	#
	if args.frange:
		if freq_start>freq_end0 or freq_end<freq_start0:
			print('Input frequency for file '+fname+' is overrange, and this file is ignored.')
			continue
		if freq_start>freq_start0 or freq_end<freq_end0:
			print('Input frequency for file '+fname+' is overrange, and the intersection band is used instead.')
		freq_start1=max(freq_start,freq_start0)
		freq_end1=min(freq_end,freq_end0)
		chanstart,chanend=np.int16(np.round((np.array([freq_start1,freq_end1])-freq_start)/channel_width))
		if chanstart==chanend:
			print('Input frequency range for file '+fname+' is too narrow, and this file is ignored.')
			continue
	elif args.crange:
		if chanstart0>nchan:
			print('Input frequency for file '+fname+' is overrange, and this file is ignored.')
			continue
		if chanend0>nchan:
			print('Input frequency for file '+fname+' is overrange, and the intersection band is used instead.')
		chanstart=max(chanstart0,0)
		chanend=min(chanend0,nchan)
		if chanend0==-1: chanend=nchan
	else:
		chanstart,chanend=0,nchan
	freq_start,freq_end=np.array([chanstart,chanend])*channel_width+freq_start
	#
	if args.trange:
		if time_start0>length:
			print("Input time for file "+fname+" is overrange, and this file is ignored.")
			continue
		if time_end0>length:
			print('Input time for file '+fname+' is overrange, and the intersection epoch is used instead.')
		time_start1=time_start0
		time_end1=min(length,time_end0)
		substart,subend=np.int16(np.round(np.array([time_start1,time_end1])/sublen))
		if substart==subend:
			print('Input time range for file '+fname+' is too short, and this file is ignored.')
			continue
	elif args.srange:
		if substart0>nsub:
			print("Input time for file "+fname+" is overrange, and this file is ignored.")
			continue
		if subend0>nsub:
			print('Input time for file '+fname+' is overrange, and the intersection epoch is used instead.')
		substart=max(substart0,0)
		subend=min(subend0,nsub)
		if subend0==-1: subend=nsub
	else:
		substart,subend=0,nsub
	time_start,time_end=np.array([substart,subend])*sublen
	time_end=min(time_end,length)
	#
	if args.prange:
		binstart,binend=np.int16(np.round(np.array([phase_start0,phase_end0])*nbin))
		if binstart==binnend:
			print('Input phase range for file '+fname+' is too narrow, and this file is ignored.')
			continue
	elif args.brange:
		if binstart0>nbin:
			print("Input bin for file "+fname+" is overrange, and this file is ignored.")
			continue
		if binend0>nbin:
			print('Input bin for file '+fname+' is overrange, and the intersection interval is used instead.')
		binstart=binstart0
		binend=min(binend0,nbin)
		if binend0==-1: binend=nbin
	else:
		binstart,binend=0,nbin
	phase_start,phase_end=np.array([binstart,binend])/nbin
	if phase_start>phase_end: phase_end+=1
	#
	if args.polar:
		if args.polar not in np.arange(npol):
			print("Input polarization for file "+fname+" is overrange, and this file is ignored.")
			continue
	polar=args.polar
	filelist.append(fname)
	ranges.append([freq_start,freq_end,chanstart,chanend,time_start,time_end,substart,subend,phase_start,phase_end,binstart,binend,polar])
	#
	if args.polarization:
		if npol<4:
			print("The polarization number for file "+fname+" does not support showing polarization, and this file is ignored.")
			continue
	#
	if args.second:
		if nchan<2 or nbin<2:
			print("The sub-integration or frequency channel number for file "+fname+" does not support showing the second spectrum image of the dynamic spectrum, and this file is ignored.")
			continue
#
nfile=len(filelist)
if len(filelist0)>1:
	print('There are '+str(nfile)+' valid files in total.')
	if nfile>0: print('Processing the data ...')
	else: print('Program aborted.')
#
if args.mulplot:
	if (row-1)*column>=nfile or row*(column-1)>=nfile:
		print('The multi-plot parameters could be smaller, and it has been adjusted automatically')
		row=(nfile>np.array([0,2,6,15,24,40])).sum()
		column=int(np.ceil(nfile/row))
		nout=1
	else: nout=int(np.ceil(nfile/(row*column)))
else: nout=nfile
#
root=tk.Tk()
sw=root.winfo_screenwidth()
sh=root.winfo_screenheight()
dpi=sw/root.winfo_screenmmwidth()*25.4
labelsize=50
if args.mulplot:
	width0=int(sw*0.8)
	height0=int(sh*0.8)
	width1=column*dpi*dx
	height1=row*dpi*dy
	if width0<width1 or height0<height1:
		if width0/height0<width1/height1:
			width=width0
			height=width0/(column*dx)*row*dy
		else:
			width=height0/(row*dy)*column*dx
			height=height0
		widthmg=int(sw*0.1)
		heightmg=int(sh*0.1)
	else:
		width=width1
		height=height1
		widthmg,heightmg=100,100
else:
	if sw>dpi*dx+200 and sh>dpi*dy+160: 
		width=int(dpi*8)
		height=int(dpi*6)
		widthmg,heightmg=100,100
	else:
		width=int(sw*0.8)
		height=int(sh*0.8)
		widthmg=int(sw*0.1)
		heightmg=int(sh*0.1)
fonts=min(height/row/20,22)
root.geometry(str(int(width+labelsize))+'x'+str(int(height+labelsize))+'+'+str(widthmg)+'+'+str(heightmg))				
figures=[]
for i in np.arange(nout): figures.append(Figure(figsize=((width+labelsize)/dpi,(height+labelsize)/dpi),dpi=dpi))
#
if args.savefigure:
	fignames=[]
	name=args.savefigure
	if len(name)>4:
		if name[-4:] in ['.png','.pdf','.eps','.PNG','.EPS','.PDF']:
			name,ext=name[:-4],name[-4:]
		else: ext='.png'
	#
	if nout==1:
		fignames.append(name)
		if os.path.isfile(name+ext): parser.error('The name of output figure file already existed. Please provide a new name.')
	else:
		for i in np.arange(nout):
			fignames.append(name+'_'+str(i))
			if os.path.isfile(name+'_'+str(i)+ext): parser.error('The name of output figure file already existed. Please provide a new name.')
#
def shift(y,x,axis=1):
	fftp=fft.rfft(y,axis=axis)
	shape=fftp.shape
	shape1=np.ones(len(shape),dtype=int)
	shape1[axis]=shape[axis]
	ffts=fftp*np.exp(-2*np.pi*x*1j*np.arange(np.shape(fftp)[axis])).reshape(*shape1)
	fftr=fft.irfft(ffts,axis=axis)
	return fftr
#
for i in np.arange(nfile):
	d=ld.ld(filelist[i])
	info=d.read_info()
	nchan=int(info['data_info']['nchan'])
	nbin=int(info['data_info']['nbin'])
	nsub=int(info['data_info']['nsub'])
	npol=int(info['data_info']['npol'])
	length=np.float64(info['data_info']['length'])
	freq_start,freq_end=np.float64(info['data_info']['freq_start']),np.float64(info['data_info']['freq_end'])
	channel_width=(freq_end-freq_start)/nchan  #MHz
	sublen=info['data_info']['sublen'] # seconds
	sublen_last=info['data_info']['sub_nperiod_last']*info['data_info']['period'] # seconds
	#
	if 'weights' in info['data_info'].keys():
		weight=np.array(info['data_info']['weights'])
		wmark='weights'
	else:
		if 'chan_weight' in info['data_info'].keys():
			weight = np.array(info['data_info']['chan_weight'])
			wmark='chan_weight'
		else:
			weight = np.ones(nchan)
			wmark='None'
		weight=weight.reshape(-1,1).repeat(nsub,axis=1)
	filer=ranges[i]
	if filer[10]>filer[11]: bins=np.append(np.arange(filer[10],nbin),np.arange(0,filer[11]))
	else: bins=np.arange(filer[10],filer[11])
	ifig=int(i//(row*column))
	irow=int(i%(row*column)//column)
	icolumn=int(i%column)
	x1,x2=labelsize/(labelsize+width),1-labelsize/(labelsize+width)
	y1,y2=labelsize/(labelsize+height),1-labelsize/(labelsize+height)
	if not args.polarization: ax=figures[ifig].add_axes([x1+(icolumn+0.15)*x2/column,y1+(row-irow-0.91)*y2/row,x2/column*0.8,y2/row*0.82])
	#
	if args.fdomain:
		data=d.period_scrunch(select_chan=np.arange(filer[2],filer[3]),start_period=filer[6],end_period=filer[7],weighted=wmark,pol=filer[-1]).reshape(filer[3]-filer[2],nbin)
		data=ma.masked_where(data==0,data)
		base_nbin = int(nbin/10)
		_,bin0=af.baseline(data.mean(0),pos=True)
		data0=np.concatenate((data,data),axis=1)[:,bin0:(bin0+base_nbin)]
		data-=data0.mean(1).reshape(-1,1)
		if args.norm: #normalize the fdata
			std=data0.std(1)
			data[std!=0]/=std[std!=0].reshape(-1,1)
		if args.shift: data=shift(data,args.shift)
		else: data=data[:,bins]
		#
		ax.imshow(data,origin='lower',aspect='auto',interpolation='nearest',extent=(filer[8],filer[9],filer[0],filer[1]),cmap='jet')
		ylabel,xlabel,suffix='Frequency (MHz)','Pulse Phase','_fdomain',
	elif args.tdomain:
		data=d.chan_scrunch(select_chan=np.arange(filer[2],filer[3]),start_period=filer[6],end_period=filer[7],weighted=wmark,pol=filer[-1])
		data=data.reshape(filer[7]-filer[6],nbin)
		data=ma.masked_where(data==0,data)
		base_nbin = int(nbin/10)
		_,bin0=af.baseline(data.mean(0),pos=True)
		data0=np.concatenate((data,data),axis=1)[:,bin0:(bin0+base_nbin)]
		data-=data0.mean(1).reshape(-1,1)
		if args.norm: #normalize the fdata
			std=data0.std(1)
			data[std!=0]/=std[std!=0].reshape(-1,1)
		if args.shift: data=shift(data,args.shift)
		else: data=data[:,bins]
		#
		ax.imshow(data,origin='lower',aspect='auto',interpolation='nearest',extent=(filer[8],filer[9],filer[4],filer[5]),cmap='jet')
		ylabel,xlabel,suffix='Time (Second)','Pulse Phase','_tdomain'
	elif args.profile:
		data=d.profile(select_chan=np.arange(filer[2],filer[3]),start_period=filer[6],end_period=filer[7],weighted=wmark).reshape(-1,npol)
		base_nbin = int(nbin/10)
		_,bin0=af.baseline(data[:,0],pos=True)
		data0=np.concatenate((data,data),axis=0)[bin0:(bin0+base_nbin)]
		data=data[:,filer[-1]]-data0[:,filer[-1]].mean()
		phasex=np.arange(filer[8],filer[9],1/nbin)
		if args.norm: data/=data0[:,filer[-1]].std()
		else: data/=data.max()
		if args.shift: data=shift(data.reshape(-1,1),args.shift,axis=0)
		else: data=data[bins]
		#
		ax.plot(phasex,data,'k-')
		ylabel,xlabel,suffix='Intensity (arbi.)','Pulse Phase','_tdomain'
	elif args.dysp:
		template=d.profile(select_chan=np.arange(filer[2],filer[3]),start_period=filer[6],end_period=filer[7],weighted=wmark,pol=0).reshape(-1)
		# determine the on- and off- pulse areas.
		template_med=np.sort(template)[int(0.25*nbin):int(1.5*(nbin//2))]
		mean=np.mean(template_med)
		std=np.std(template_med)
		SN=(template-mean)/std
		on_gates=np.argwhere(SN>7.).squeeze()
		data=d.bin_scrunch(select_chan=np.arange(filer[2],filer[3]),start_period=filer[6],end_period=filer[7],select_bin=on_gates,pol=filer[-1]).reshape(filer[3]-filer[2],filer[7]-filer[6])
		data*=weight[filer[2]:filer[3],filer[6]:filer[7]]
		data=ma.masked_where(data==0,data)
		# normalize the intensity in each sub
		if args.norm: data/=np.mean(data,axis=0,keepdims=True)
		else:
			data-=np.min(data)
			data/=np.max(data)
		ax.imshow(data,aspect='auto',cmap='jet',origin='lower',extent=[filer[4],filer[5],filer[0],filer[1]],interpolation='nearest')
		ylabel,xlabel,suffix='Frequency (MHz)','Time (Second)','_dysp'
	elif args.second:
		template=d.profile(select_chan=np.arange(filer[2],filer[3]),start_period=filer[6],end_period=filer[7],weighted=wmark,pol=0).reshape(-1)
		# determine the on- and off- pulse areas.
		template_med=np.sort(template)[int(0.25*nbin):int(1.5*(nbin//2))]
		mean=np.mean(template_med)
		std=np.std(template_med)
		SN=(template-mean)/std
		on_gates=np.argwhere(SN>7.).squeeze()
		data=d.bin_scrunch(select_chan=np.arange(filer[2],filer[3]),start_period=filer[6],end_period=filer[7],select_bin=on_gates,pol=filer[-1]).reshape(filer[3]-filer[2],filer[7]-filer[6])
		data*=weight[filer[2]:filer[3],filer[6]:filer[7]]
		data=ma.masked_where(data==0,data)
		if sublen!=sublen_last:
			nsub_u=nsub-1
			data=data[:,:nsub_u]
		mhzperbin,secperbin=channel_width,sublen
		dynspec=data-np.mean(data[~np.isnan(data)])  # subtract mean to make dynspec on zero level
		nf,nt=dynspec.shape
		#padding
		window_frac=0.1
		cw=np.hamming(np.floor(window_frac*nt))
		sw=np.hamming(np.floor(window_frac*nf))
		#taper
		chan_window=np.insert(cw,int(np.ceil(len(cw)/2)),np.ones([nt-len(cw)]))
		subint_window=np.insert(sw,int(np.ceil(len(sw)/2)),np.ones([nf-len(sw)]))
		dynspec=dynspec*chan_window*subint_window.reshape(-1,1)
		# find the right fft lengths for rows and columns
		nrfft=int(2**(np.ceil(np.log2(nf))+1))
		ncfft=int(2**(np.ceil(np.log2(nt))+1))
		#create secondary spectrum
		dynspec[np.isnan(dynspec)]=0
		simf=np.fft.rfft2(dynspec,s=[nrfft,ncfft],axes=[1,0])
		simf=np.real(simf*np.conj(simf))  # is real
		sec=np.fft.fftshift(simf,axes=(1,))  # fftshift
		#considering nyquist frequency
		Nyquist_time = (1.0/secperbin)*0.5*10**3 #in the unit of 10^{-3} hz
		Nyquist_freq = (1.0/mhzperbin)*0.5       #in the unit of us     
		# Make db
		sec = 10*np.log10(sec)
		sec = ndimage.gaussian_filter(sec, 1)
		mean = np.mean(sec)
		std = np.std(sec)
		ax.imshow(sec, aspect='auto', extent=[-1.0*Nyquist_time,Nyquist_time, 0, Nyquist_freq], vmin=mean-1.0*std, vmax = mean+3.0*std, origin='lower', cmap = 'viridis')
		ylabel,xlabel,suffix='Delay ($\mu$s)','Fringe Frequency (mHz)','_secondary'
	elif args.polarization:
		data=d.profile(select_chan=np.arange(filer[2],filer[3]),start_period=filer[6],end_period=filer[7],weighted=wmark).reshape(-1,npol)
		base_nbin = int(nbin/10)
		_,bin0=af.baseline(data[:,0],pos=True)
		data0=np.concatenate((data,data),axis=0)[bin0:(bin0+base_nbin)]
		data=data-data0.mean(0).reshape(1,-1)
		phasex=np.arange(filer[8],filer[9],1/nbin)
		if args.norm: data/=data0[:,0].std()
		else: data/=data[:,0].max()
		if args.shift: data=shift(data,args.shift,axis=0)
		else: data=data[bins]
		ii,qq,uu,vv=data.T
		ll=np.sqrt(qq**2+uu**2)
		_,bin0=af.baseline(ii,pos=True)
		bins0=np.int32(np.arange(bin0,bin0+base_nbin)%nbin)
		ppa=0.5*np.arctan2(uu,qq)/np.pi*180%180
		ppae=1/2*np.std(data[bins0])/ll
		ax=figures[ifig].add_axes([x1+(icolumn+0.15)*x2/column,y1+(row-irow-0.91)*y2/row,x2/column*0.8,y2/row*0.42])
		ax1=figures[ifig].add_axes([x1+(icolumn+0.15)*x2/column,y1+(row-irow-0.49)*y2/row,x2/column*0.8,y2/row*0.4])
		ax.plot(phasex,ii,label='I',color='black')
		ax.plot(phasex,ll,label='Lin',color='red')
		ax.plot(phasex,vv,label='Cir',color='blue')
		jj=ppae<0.5
		ax1.errorbar(phasex[jj],-ppa[jj], yerr=ppae[jj]/np.pi*180, fmt='.')
		ax1.set_xticks([])
		ax.legend()
		ylabel,xlabel,suffix='Intensity (arbi.) ; PPA ($\degree$)','Pulse Phase','_polarization'
		ax1.tick_params(axis='x',labelsize=fonts/1.5)
		ax1.tick_params(axis='y',labelsize=fonts/1.5)
		ax1.set_title(filelist[i],fontsize=fonts/1.5)
		ax.tick_params(axis='x',labelsize=fonts/1.5)
		ax.tick_params(axis='y',labelsize=fonts/1.5)
	if not args.polarization:
		if irow!=row-1: ax.set_xticks([])
		ax.set_title(filelist[i],fontsize=fonts)
		ax.tick_params(axis='x',labelsize=fonts)
		ax.tick_params(axis='y',labelsize=fonts)
#
for i in np.arange(nout):
	figures[i].text(x1*0.5,0.5,ylabel,fontsize=30,va='center',ha='center',rotation='vertical')
	figures[i].text(0.5,y1*0.5,xlabel,fontsize=30,va='center',ha='center')
#
ifig=0
def cont(event):
	global ifig,canvas
	if ifig==0:
		canvas=FigureCanvasTkAgg(figures[ifig],master=root)
		canvas.get_tk_widget().grid()
		canvas.get_tk_widget().pack(fill='both')
	elif ifig<nout:
		canvas.figure.clf()
		canvas.draw()
		canvas.figure=figures[ifig]
	else: sys.exit()
	canvas.draw()
	ifig+=1
#
if args.savefigure:
	for i in np.arange(nout): figures[i].savefig(fignames[i]+suffix+ext)
else:
	cont(0)
	root.bind('<KeyPress>',cont)
	root.mainloop()
