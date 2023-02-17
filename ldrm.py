#!/usr/bin/env python
import numpy as np
import argparse as ap
import numpy.fft as fft
import os,ld,time,sys
import scipy.optimize as so
import time_eph as te
import psr_read as pr
import psr_model as pm
import warnings as wn
import adfunc as af
wn.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Serif'
#
version='JigLu_20201202'
parser=ap.ArgumentParser(prog='ldrm',description='Calculate the best RM value. Press \'s\' in figure window to save figure.',epilog='Ver '+version)
parser.add_argument('-v','--version',action='version',version=version)
parser.add_argument("filename",nargs='+',help="input ld file or files")
parser.add_argument('-r','--frequency_range',default=0,dest='frequency',help='limit the frequency rangeFREQ0,FREQ1')
parser.add_argument('-o','--rmi',default=0,type=np.float64,dest='rmi',help='the initial RM value')
parser.add_argument('-b','--base',default='',dest='base',help='the baseline range PHASE0,PHASE1 or pulse-off phase width')
parser.add_argument('-s','--subint_range',default=0,dest='subint',help='limit the subint range SUBINT0,SUBINT1')
parser.add_argument('-t',action='store_true',default=False,dest='text',help='only print the result in text-form instead of plot')
parser.add_argument('-f',default='',dest='file',help='output the results into a file')
parser.add_argument('-d','--dm_center',dest='dm',default=0,type=np.float64,help="center of the fitting dispersion measure")
parser.add_argument('-i','--dm_zone',dest='zone',default=0,type=np.float64,help="total range of the fitting dispersion measure")
parser.add_argument('-c',action='store_true',default=False,dest='correct',help='correct the data with the best dm')
parser.add_argument('-n',type=np.int16,default=0,dest='compf',help='scrunch the frequency by a factor n')
parser.add_argument("-z","--zap",dest="zap_file",default=0,help="file recording zap channels")
args=(parser.parse_args())
wn.filterwarnings('ignore')
command=['ldrm.py']
#
c=299792458.0
filelist=args.filename
errorfile=[]
filedict={}
for i in filelist:
	if not os.path.isfile(i):
		parser.error(i+' is unexist.')
	else:
		try:
			filei=ld.ld(i)
			if filei.read_para('pol_type')!='IQUV':
				parser.error('The data in file '+i+' is unpolarized/uncalibrated.')
			if filei.read_shape()[0]<=2:
				parser.error('The channel numbers for file '+i+' is too small to fit the rotation measure.')
			psr_par=filei.read_para('psr_par')
			psr_name=pr.psr(psr_par).name
			if psr_name in filedict.keys(): filedict[psr_name].append([i,psr_par])
			else: filedict[psr_name]=[[i,psr_par]]
		except:
			errorfile.append(i)
if errorfile:
	print('Warning: '+', '.join(errorfile)+' is/are not valid ld file')
psrlist=list(filedict.keys())
#
if args.file:
	output=open(args.file,'w')
#
if args.zap_file:
	command.append('-z')
	if not os.path.isfile(args.zap_file):
		parser.error('The zap channel file is invalid.')
	zchan0=np.loadtxt(args.zap_file,dtype=np.int32)
else:
	zchan0=np.array([],dtype=np.int32)
#
if args.frequency:
	command.append('-r '+args.frequency)
if args.rmi:
	command.append('-o '+str(args.rmi))
if args.base:
	command.append('-b '+args.base)
if args.subint:
	command.append('-s '+args.subint)
if args.correct:
	command.append('-c')
if args.compf:
	command.append('-n '+str(args.compf))
if args.dm:
	command.append('-d '+str(args.dm))
if args.zone:
	command.append('-i '+str(args.zone))
command=' '.join(command)
#
if not args.text:
	if len(psrlist)>0:
		if len(psrlist)>1 or len(filedict[psrlist[0]])>1:
			parser.error('The visualized results cannot be manifested for multi-files.')
#
for psr_name in psrlist:
	for filename,psr_par in filedict[psr_name]:
		psr_para=pr.psr(psr_par)
		d=ld.ld(filename)
		info=d.read_info()
		if 'rm' in info.keys(): rm0=info['rm'][0]
		else: rm0=0
		if 'compressed' in info.keys():
			nchan=info['nchan_new']
			nbin=info['nbin_new']
			nsub=info['nsub_new']
		else:
			nchan=info['nchan']
			nbin=info['nbin']
			nsub=info['nsub']
		if len(zchan0):
			if np.max(zchan0)>=nchan or np.min(zchan0)<0:
				parser.error('The zapped channel number is overrange.')
		dm0=info['dm']
		period=info['period']
		#
		if args.zone:
			zone=args.zone/2
		else:
			zone=np.max([0.1,dm0/100])
			zone=np.min([0.5,zone])
		freq_start0=info['freq_start']
		freq_end0=info['freq_end']
		channel_width=(freq_end0-freq_start0)/nchan
		freq0=np.arange(freq_start0,freq_end0,channel_width)+channel_width/2.0
		if args.frequency:
			frequency=np.float64(args.frequency.split(','))
			if len(frequency)!=2:
				parser.error('A valid frequency range should be given.')
			if frequency[0]>frequency[1]:
				parser.error("Starting frequency larger than ending frequency.")
			freq_start=max(frequency[0],freq_start0)
			freq_end=min(frequency[1],freq_end0)
			chanstart,chanend=np.int16(np.round((np.array([freq_start,freq_end])-(freq_start+freq_end)/2.0)/channel_width+0.5*nchan))
			chan=np.arange(chanstart,chanend)
			if len(chan)<=2:
				parser.error('Input bandwidth is too narrow.')
			freq=freq0[chan]
		else:
			freq=freq0
			chanstart,chanend=0,nchan
			chan=np.arange(0,nchan)
		#
		if args.subint:
			subint=np.int64(args.subint.split(','))
			if subint[1]<0:
				subint[1]=subint[1]+nsub
			if len(subint)!=2:
				parser.error('A valid subint range should be given.')
			if subint[0]>subint[1]:
				parser.error("Starting subint is larger than ending subint.")
			subint_start=max(int(subint[0]),0)
			subint_end=min(int(subint[1]+1),nsub)
		else:
			subint_start=0
			subint_end=nsub
			subint=np.array([subint_start,subint_end])
		#
		data0=d.period_scrunch(subint_start,subint_end)
		data0[zchan0]=0
		data0=data0[chan]
		i=data0[:,:,0]
		q=data0[:,:,1]
		u=-data0[:,:,2]
		lam=c/freq/1e6
		lam2=lam**2
		base=args.base.split(',')
		if args.base:
			if len(base)==1:
				nbase=int(np.float64(base)*nbin)
				pn,bn=af.radipos(i.mean(0),base=True,base_nbin=nbase)
				bn=np.arange(bn,bn+nbase)
			elif len(base)==2:
				bn=np.arange(tuple(np.int64(np.float64(base)*nbin)))
				nbase=bn.size
				pn=af.radipos(i.mean(0),base_nbin=nbase)
			else:
				parser.error('A valid phase range/phase width should be given.')
		else:
			nbase=int(nbin/10)
			pn,bn=af.radipos(i.mean(0),base=True,base_nbin=nbase)
			bn=np.arange(bn,bn+nbase)
		bn=bn%nbin
		q-=q[:,bn].mean(1).reshape(-1,1)
		u-=u[:,bn].mean(1).reshape(-1,1)
		if args.compf:
			if len(freq)/args.compf<3:
				parser.error('The scrunch factor is too large.')
			lenf=int(len(freq)//args.compf*args.compf)
			lam2=lam2[:lenf].reshape(-1,args.compf).mean(1)
		#
		def rot(x,rm,c):
			return rm*x+c
		#
		rm,drm=0,1e10
		if args.rmi:
			rm=args.rmi
			dphi0=rot(lam**2,rm,0)*2
			q0=q*np.cos(dphi0).reshape(-1,1)+u*np.sin(dphi0).reshape(-1,1)
			u0=u*np.cos(dphi0).reshape(-1,1)-q*np.sin(dphi0).reshape(-1,1)
		else:
			q0,u0=q.copy(),u.copy()			
		while True:
			qs=q0.mean(0)
			us=u0.mean(0)
			qs-=qs[bn].mean()
			us-=us[bn].mean()
			dphi=np.arctan2(us,qs)
			qs=q0*np.cos(dphi)+u0*np.sin(dphi)
			us=u0*np.cos(dphi)-q0*np.sin(dphi)
			qs-=qs[:,bn].mean(1).reshape(-1,1)
			us-=us[:,bn].mean(1).reshape(-1,1)
			eqs,eus=qs[:,bn].std(1),us[:,bn].std(1)
			qq=qs[:,pn].sum(1)
			uu=us[:,pn].sum(1)
			if args.compf:
				qq=qq[:lenf].reshape(-1,args.compf).sum(1)
				uu=uu[:lenf].reshape(-1,args.compf).sum(1)
			phi=1/2*np.arctan2(uu,qq)
			if args.compf:
				ephi=np.sqrt((eqs**2+eus**2)[:lenf].reshape(-1,args.compf).sum(1)/2*len(pn)/(qq**2+uu**2))
			else:
				ephi=np.sqrt((eqs**2+eus**2)/2*len(pn)/(qq**2+uu**2))
			jj=(ephi>0)
			phi=phi[jj]
			ephi=ephi[jj]
			lam20=lam2[jj]
			popt,pcov=so.curve_fit(rot,lam20,phi,p0=(0,0),sigma=ephi)
			if np.abs(popt[0])<np.abs(drm):
				drm=popt[0]
				rm+=drm
			else: break
			dphi0=rot(lam**2,rm,0)*2
			q0=q*np.cos(dphi0).reshape(-1,1)+u*np.sin(dphi0).reshape(-1,1)
			u0=u*np.cos(dphi0).reshape(-1,1)-q*np.sin(dphi0).reshape(-1,1)
		#
		best_rm=rm0+rm
		rmerr=np.sqrt(pcov[0,0])
		ndigit=int(-np.log10(rmerr))+2
		ndigit=max(ndigit,0)
		if args.file:
			output.write(psr_para.name+'  '+filename+' Best RM='+str(np.round(best_rm,ndigit))+'+-'+str(np.round(rmerr,ndigit))+'\n')
		if args.text:
			sys.stdout.write(psr_para.name+'  '+filename+' Best RM='+str(np.round(best_rm,ndigit))+'+-'+str(np.round(rmerr,ndigit))+'\n')
		ndigit=int(-np.log10(rmerr))+1
		ndigit=max(ndigit,0)
		if args.correct:
			info['rm']=[best_rm,rmerr]
			lam0=c/freq0/1e6
			if 'history' in info.keys():
				info['history'].append(command)
				info['file_time'].append(time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime()))
			else:
				info['history']=[command]
				info['file_time']=[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())]
			for i in np.arange(nchan):
				data_tmp=d.read_chan(i)
				dphi0=rot(lam0[i]**2,rm,0)*2
				q_tmp,u_tmp=data_tmp[:,:,1:3].transpose(2,0,1).copy()
				q0=q_tmp*np.cos(dphi0)+u_tmp*np.sin(dphi0)
				u0=u_tmp*np.cos(dphi0)-q_tmp*np.sin(dphi0)
				data_tmp[:,:,1]=q0
				data_tmp[:,:,2]=u0
				d.write_chan(data_tmp,i)
			d.write_info(info)
		#
		if not args.text:
			from matplotlib.figure import Figure
			import matplotlib.pyplot as plt
			fig=Figure(figsize=(50,24),dpi=80)
			fig.set_facecolor('white')
			x0,x1,x2=0.09,0.5,0.93
			y0,y1=0.11,0.96
			ax=fig.add_axes([x0,y0,x1-x0,y1-y0])
			ax1=fig.add_axes([x1,y0,x2-x1,y1-y0])
			ax.patch.set_facecolor('w')
			ax1.set_xlabel('Pulse Phase',fontsize=30)
			ax1.set_yticks([])
			ax1.patch.set_facecolor('w')
			ax.set_ylabel('Polarization Angle',fontsize=30)
			ax.set_xlabel('Wavelength (m)',fontsize=30)
			ax1=ax1.twinx()
			ax1.set_ylabel('Intensity (arbi.)',fontsize=30)
			y=phi+rot(lam2[jj],rm,0)
			ax.errorbar(np.sqrt(lam2[jj]),y,yerr=ephi,fmt='b.')
			ax.plot(lam,dphi0/2,'r-')
			ymax,ymin=(y+ephi).max(),(y-ephi).min()
			ax.set_ylim(ymin*1.2-ymax*0.2,ymax*1.05-ymin*0.05)
			ax.text(lam.mean(),ymin*1.1-ymax*0.1,'Best RM='+str(np.round(best_rm,ndigit))+'$\pm$'+str(np.round(rmerr,ndigit)),horizontalalignment='center',verticalalignment='center',fontsize=25)
			#
			phase=np.arange(nbin)/nbin
			dphi0=rot(lam**2,rm,0)*2
			q0=q*np.cos(dphi0).reshape(-1,1)+u*np.sin(dphi0).reshape(-1,1)
			u0=u*np.cos(dphi0).reshape(-1,1)-q*np.sin(dphi0).reshape(-1,1)
			q1=q0.mean(0)
			u1=u0.mean(0)
			q1-=q1[bn].mean()
			u1-=u1[bn].mean()
			l1=np.sqrt(q1**2+u1**2)
			q2=q.mean(0)
			u2=u.mean(0)
			q2-=q2[bn].mean()
			u2-=u2[bn].mean()
			l2=np.sqrt(q2**2+u2**2)
			l2/=l1.max()
			l1/=l1.max()
			ax1.plot(phase,l2,'k',label='Lin. Polar. (before modifying RM)')
			ax1.plot(phase,l1,'r',label='Lin. Polar. (after modifying RM)')
			ax1.set_xlim(0,1)
			ax1.legend(fontsize=20,frameon=False)
			#
			def save_fig():
				figname=input("Please input figure name:")
				if figname.split('.')[-1] not in ['ps','eps','png','pdf','pgf']:
					figname+='.pdf'
				fig.savefig(figname)
				sys.stdout.write('Figure file '+figname+' has been saved.')
			#
			try:
				import gtk
				from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg
				window=gtk.Window()
				window.set_title(args.filename)
				window.set_size_request(1250,600)
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
				root.geometry('1250x600+100+100')
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
