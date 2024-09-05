#!/usr/bin/env python
import numpy as np
import numpy.ma as ma
import numpy.fft as fft
import argparse as ap
import os,time,ld,sys
import warnings as wn
import time_eph as te
import psr_model as pm
import psr_read as pr
import adfunc as af
import time
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Serif'
dirname=os.path.split(os.path.realpath(__file__))[0]
sys.path.append(dirname+'/doc')
import text
#
text=text.output_text('lddm')
version='JigLu_20180515'
parser=ap.ArgumentParser(prog='lddm',description=text.help,epilog='Ver '+version,add_help=False,formatter_class=lambda prog: ap.RawTextHelpFormatter(prog, max_help_position=50))
parser.add_argument('-h', '--help', action='help', default=ap.SUPPRESS,help=text.help_h)
parser.add_argument('-v','--version',action='version',version=version,help=text.help_v)
parser.add_argument("filename",nargs='+',help=text.help_filename)
parser.add_argument('--fr','--frequency_range',default=0,dest='frequency',help=text.help_fr)
parser.add_argument('--sr','--subint_range',default=0,dest='subint',help=text.help_sr)
parser.add_argument('-n',action='store_true',default=False,dest='norm',help=text.help_n)
parser.add_argument('-k','--text',action='store_true',default=False,dest='text',help=text.help_k)
parser.add_argument('-o','--file',default='',dest='file',help=text.help_o)
parser.add_argument('-m',action='store_true',default=False,dest='modify',help=text.help_m)
parser.add_argument('-c',action='store_true',default=False,dest='correct',help=text.help_c)
parser.add_argument('-d','--dm_center',dest='dm',default=0,type=np.float64,help=text.help_d)
parser.add_argument('-i','--dm_zone',dest='zone',default=0,type=np.float64,help=text.help_i)
parser.add_argument('--degree','--polynomial_order',default=0,dest='n',type=int,help=text.help_degree)
parser.add_argument("-z","--zap",dest="zap_file",default=0,help=text.help_z)
parser.add_argument('-p','--precision',default=0,dest='prec',type=np.float64,help=text.help_p)
args=(parser.parse_args())
wn.filterwarnings('ignore')
command=['lddm.py']
#
filelist=args.filename
errorfile=[]
filedict={}
for i in filelist:	# check the files
	if not os.path.isfile(i):
		parser.error(text.error_fue % i)
	else:
		try:
			filei=ld.ld(i)
			psr_par=filei.read_para('psr_par')
			psr_name=pr.psr(psr_par,warning=False).name
			if psr_name in filedict.keys(): filedict[psr_name].append([i,psr_par])
			else: filedict[psr_name]=[[i,psr_par]]
		except:
			errorfile.append(i)
if errorfile:
	print(text.warning_ilf % ', '.join(errorfile))
psrlist=list(filedict.keys())
#
if args.file:
	output=open(args.file,'w')
#
if args.zap_file:
	command.append('-z')
	if not os.path.isfile(args.zap_file):
		parser.error(text.error_zfi)
	zchan0=np.loadtxt(args.zap_file,dtype=np.int32)
else:
	zchan0=np.array([])
#
if args.frequency:
	command.append('--fr '+args.frequency)
if args.subint:
	command.append('--sr '+args.subint)
if args.dm:
	command.append('-d '+str(args.dm))
if args.zone:
	command.append('-i '+str(args.zone))
if args.n:
	command.append('--degree '+str(args.n))
if args.prec:
	command.append('-p '+str(args.prec))
if args.norm:
	command.append('-n')
if args.modify:
	command.append('-m')
if args.correct:
	command.append('-c')
if args.modify and args.correct:
	parser.error(text.error_mfc)
command=' '.join(command)
#
if not args.text:
	if args.prec:
		parser.error(text.error_pvn)
	if len(psrlist)>0:
		if len(psrlist)>1 or len(filedict[psrlist[0]])>1:
			parser.error(text.error_vnm)
#
for psr_name in psrlist:
	first=True
	print(text.info_ana % psr_name)
	for filename,psr_par in filedict[psr_name]:
		psr_para=pr.psr(psr_par,warning=first)
		first=False
		d=ld.ld(filename)
		info=d.read_info()
		nchan=info['data_info']['nchan']
		nbin=info['data_info']['nbin']
		nsub=info['data_info']['nsub']
		if len(zchan0):
			if np.max(zchan0)>=nchan or np.min(zchan0)<0:
				parser.error(text.error_zno)
		dm0=info['data_info']['dm']
		period=info['data_info']['period']
		if args.dm:
			ddm=args.dm-dm0
		else:
			ddm=0
		#
		if args.zone:
			zone=args.zone/2
		else:
			zone=np.max([0.1,dm0/100])
			zone=np.min([0.5,zone])
		freq_start0=info['data_info']['freq_start']
		freq_end0=info['data_info']['freq_end']
		channel_width=(freq_end0-freq_start0)/nchan
		frequency=np.arange(freq_start0,freq_end0,channel_width)+channel_width/2.0
		if args.frequency:
			freqtmp=np.float64(args.frequency.split(','))
			if len(freqtmp)!=2:
				parser.error(text.error_nfr)
			freq_start,freq_end=freqtmp
			chanstart,chanend=np.int16(np.round((np.array([freq_start,freq_end])-freq_start0)/channel_width))
			if chanstart>chanend:
				parser.error(text.error_sfl)
			elif chanstart<0 or chanend>nchan:
				parser.error(text.error_ifo)
			freq_start,freq_end=np.array([chanstart,chanend])*channel_width+freq_start0
			chan=np.arange(chanstart,chanend)
			if len(chan)<2:
				parser.error(text.error_ibn)
			freq=frequency[chan]
		else:
			freq=frequency
			chanstart,chanend=0,nchan
			chan=[]
		#
		if args.subint:
			subint=np.int64(args.subint.split(','))
			if subint[1]<0:
				subint[1]=subint[1]+nsub
			if len(subint)!=2:
				parser.error(text.error_nsr)
			if subint[0]>subint[1]:
				parser.error(text.error_ssl)
			subint_start=max(int(subint[0]),0)
			subint_end=min(int(subint[1]+1),nsub)
		else:
			subint_start=0
			subint_end=nsub
			subint=np.array([subint_start,subint_end])
		#
		data0=d.period_scrunch(subint_start,subint_end,pol=0)*np.asarray(info['data_info']['chan_weight']).reshape(-1,1,1)
		data0=data0[:,:,0]
		if len(chan):
			data=data0[chan]
		else:
			data=data0.copy()
		#
		if not args.text:	# figure
			from matplotlib.figure import Figure
			import matplotlib.pyplot as plt
			fig=Figure(figsize=(40,30),dpi=80)
			fig.set_facecolor('white')
			x0,x1,x2=0.1,0.6,0.9
			y0,y1=0.11,0.96
			ax=fig.add_axes([x0,y0,x1-x0,y1-y0])
			ax.patch.set_facecolor('w')
			ax1=fig.add_axes([x1,y0,x2-x1,(y1-y0)/2])
			ax1.set_xlabel('Pulse Phase',fontsize=25)
			ax.set_yticks([])
			ax1.set_yticks([])
			ax1=ax1.twinx()
			ax2=fig.add_axes([x1,(y1+y0)/2,x2-x1,(y1-y0)/2])
			ax2.set_xticks([])
			ax2.set_yticks([])
			ax2=ax2.twinx()
			ax1.patch.set_facecolor('w')
			ax2.patch.set_facecolor('w')
			ax.set_xlabel('DM',fontsize=30)
			ax.set_ylabel('Relative Maxima',fontsize=30)
			ax1.set_ylabel('Frequency (MHz)',fontsize=25)
			ax2.set_ylabel('Frequency (MHz)',fontsize=25)
		#
		zchan1=np.where(info['data_info']['chan_weight']==0)[0]
		zchan=set(zchan0).union(zchan1)
		if len(chan):
			zchan=np.int32(list(zchan.intersection(chan)))-chanstart
		else:
			zchan=np.int32(list(zchan))-chanstart
		nzchan=np.array(list(set(range(chanend-chanstart))-set(zchan)))
		zaparray=np.zeros_like(data0)
		zaparray[zchan]=True
		data1=ma.masked_array(data0,mask=zaparray)
		#
		data-=data.mean(1).reshape(-1,1)
		data1-=data1.mean(1).reshape(-1,1)
		if args.norm:
			maxima=data.max(1)
			maxima[maxima==0]=1
			data/=maxima.reshape(-1,1)
			maxima=data1.max(1)
			maxima[maxima==0]=1
			data1/=maxima.reshape(-1,1)
		else:
			data*=np.asarray(info['data_info']['chan_weight']).reshape(-1,1)
			data1*=np.asarray(info['data_info']['chan_weight']).reshape(-1,1)
		#
		if not args.text: 
			ax2.imshow(data1[::-1],aspect='auto',interpolation='nearest',extent=(0,1,freq_start0,freq_end0),cmap='jet')
			if args.frequency:
				ax2.plot([0,1],[freq_start,freq_start],'k--')
				ax2.plot([0,1],[freq_end,freq_end],'k--')
		#
		def shift(y,x):
			ffts=y*np.exp(x*1j)
			fftr=fft.irfft(ffts)
			return fftr
		#
		psr=pm.psr_timing(psr_para,te.times(te.time(info['data_info']['stt_time'],info['data_info']['length']/2)),freq.mean())	# the observed frequency is different from the original signal frequency in ISM
		fftdata=fft.rfft(data,axis=1)
		tmp=np.shape(fftdata)[-1]
		const=(1/(freq*psr.vchange)**2*pm.dm_const/period*np.pi*2.0).repeat(tmp).reshape(-1,tmp)*np.arange(tmp)
		#
		if args.n:
			order=args.n
		else:
			order=7
		#
		if args.prec:	# calculate the best DM and its error
			dmmax,dmerr=af.dmdet(fftdata[nzchan],const[nzchan],ddm,zone,order,prec=args.prec)
		else:
			dmmax,dmerr,dm,value,fitvalue=af.dmdet(fftdata[nzchan],const[nzchan],ddm,zone,order,prec=args.prec)
		#
		if not args.text:
			x0=dm.min()+dm0
			x1=dm.max()+dm0
			y0=value.min()*1.1-value.max()*0.1
			y1=value.max()*1.1-value.min()*0.1
			ax.plot(dm+dm0,value,'b-')
			ax.set_xlim(x0,x1)
			ax.set_ylim(y0,y1)
		#
		if not args.text:
			ax.plot(dm+dm0,fitvalue,'k--')
		if dmerr>0:
			ndigit=int(-np.log10(dmerr))+2
			ndigit=max(ndigit,0)
			if args.file:
				output.write(psr_para.name+'  '+filename+' DM_0='+str(dm0)+', Best DM='+str(np.round(dmmax+dm0,ndigit))+'+-'+str(np.round(dmerr,ndigit))+'\n')
			if args.modify:
				if 'additional_info' in info.keys(): info['additional_info']['best_dm']=[dmmax+dm0,dmerr]
				else: info['additional_info']={'best_dm':[dmmax+dm0,dmerr]}
				if 'history_info' in info.keys():
					info['history_info']['history'].append(command)
					info['history_info']['file_time'].append(time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime()))
				else:
					info['history_info']={}
					info['history_info']['history']=[command]
					info['history_info']['file_time']=[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())]
				d.write_info(info)
			elif args.correct:
				info['data_info']['dm']=dmmax+dm0
				if 'additional_info' in info.keys(): info['additional_info']['best_dm']=[dmmax+dm0,dmerr]
				else: info['additional_info']={'best_dm':[dmmax+dm0,dmerr]}
				if 'history_info' in info.keys():
					info['history_info']['history'].append(command)
					info['history_info']['file_time'].append(time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime()))
				else:
					info['history_info']={}
					info['history_info']['history']=[command]
					info['history_info']['file_time']=[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())]
				for i in np.arange(nchan):
					data_tmp=d.read_chan(i)
					fftdata0=fft.rfft(data_tmp,axis=1)
					tmp=np.shape(fftdata)[1]
					frac=1/(frequency[i]*psr.vchange)**2*pm.dm_const/period*dmmax
					const=((frac*np.pi*2.0)*np.arange(tmp)).reshape(1,tmp,1)
					ffts=fftdata0*np.exp(const*1j)
					data_tmp=fft.irfft(ffts,axis=1)
					d.write_chan(data_tmp,i)
				d.write_info(info)
			#
			if not args.text:
				ax.plot([dm0+dmmax,dm0+dmmax],[y0,y1],'k:')
				ax.text(dm0+ddm,y0*0.95+y1*0.05,'DM$_0$='+str(dm0)+'\nBest DM='+str(np.round(dmmax+dm0,3))+'$\\pm$'+str(np.round(dmerr,3)),horizontalalignment='center',verticalalignment='bottom',fontsize=25)
				fftdata0=fft.rfft(data0,axis=1)
				tmp=np.shape(fftdata)[-1]
				frac=1/(frequency*psr.vchange)**2*pm.dm_const/period*dmmax
				const=(frac*np.pi*2.0).repeat(tmp).reshape(-1,tmp)*np.arange(tmp)
				data1=shift(fftdata0,const)
				data1=ma.masked_array(data1,mask=zaparray)
				ax1.imshow(data1[::-1],aspect='auto',interpolation='nearest',extent=(0,1,freq_start0,freq_end0),cmap='jet')
				ax1.plot(np.ones_like(frequency)*0.5,frequency,'r--')
				ax2.plot((frac+0.5)%1,frequency,'r--')
				if args.frequency:
					ax1.plot([0,1],[freq_start,freq_start],'k--')
					ax1.plot([0,1],[freq_end,freq_end],'k--')
			else:
				print(psr_para.name+'  '+filename+' DM_0='+str(dm0)+', Best DM='+str(np.round(dmmax+dm0,ndigit))+'+-'+str(np.round(dmerr,ndigit)))
		else:
			if not args.text: ax.text(dm0+ddm,y0*0.95+y1*0.05,'The best DM cannot be found',horizontalalignment='center',verticalalignment='bottom',fontsize=25)
			else: print(text.info_bdm % (filename,psr_para.name))
		#
		if not args.text:
			ax2.set_ylim(freq_start0,freq_end0)
			ax2.set_xlim(0,1)
			ax1.set_ylim(freq_start0,freq_end0)
			ax1.set_xlim(0,1)
			#
			def save_fig():
				figname=input(text.input_fn)
				if figname.split('.')[-1] not in ['ps','eps','png','pdf','pgf']:
					figname+='.pdf'
				fig.savefig(figname)
				print(text.info_save % figname)
			#
			try:
				import gtk
				from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg
				window=gtk.Window()
				window.set_title(args.filename)
				window.set_size_request(1000,600)
				box=gtk.VBox()
				canvas=FigureCanvasGTKAgg(fig)
				box.pack_start(canvas)
				window.add(box)
				window.modify_bg('normal',gtk.gdk.Color('#fff'))
				window.show_all()
				window.connect('destroy',gtk.main_quit)
				def save_gtk(window,event):
					if gtk.gdk.keyval_name(event.keyval)=='s':
						save_fig()
				window.connect('key-press-event',save_gtk)
				gtk.main()
			except:
				import matplotlib as mpl
				from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
				import tkinter as tk
				mpl.use('TkAgg')
				ax.tick_params(axis='x',labelsize=15)
				ax.tick_params(axis='y',labelsize=15)
				root=tk.Tk()
				root.title(args.filename)
				root.geometry('1000x600+100+100')
				canvas=FigureCanvasTkAgg(fig,master=root)
				canvas.get_tk_widget().grid()
				canvas.get_tk_widget().pack(fill='both')
				canvas.draw()
				def save_tk(event):
					if event.keysym=='s':
						save_fig()
				root.bind('<KeyPress>',save_tk)
				root.mainloop()
#
if args.file:
	output.close()
