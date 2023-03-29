#!/usr/bin/env python
import numpy as np
import numpy.ma as ma
import numpy.fft as fft
import argparse as ap
from matplotlib.figure import Figure
import matplotlib.lines as ln
import ld,os,copy,sys,shutil
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Serif'
#
version='JigLu_20200923'
parser=ap.ArgumentParser(prog='ldzap',description='Zap the frequency domain interference in ld file.',epilog='Ver '+version)
parser.add_argument('-v','--version',action='version',version=version)
parser.add_argument("-z","--zap",dest="zap_file",default=0,help="file recording zap channels")
parser.add_argument('-n',action='store_true',default=False,dest='norm',help='normalized the data at each channel')
parser.add_argument('-m',action='store_true',default=False,dest='mean',help='use mean value as the screening standard')
parser.add_argument('-c','--cal',action='store_true',default=False,dest='cal',help='use calibration parameter as the screening standard')
parser.add_argument("filename",help="input ld file")
args=(parser.parse_args())
#
if not os.path.isfile(args.filename):
	parser.error('A valid ld file name is required.')
d=ld.ld(args.filename)
info=d.read_info()
#
nchan=info['nchan']
if info['mode']=='test':
	nbin=1
	nperiod=d.read_shape()[1]
else:
	nbin=info['nbin']
	nperiod=info['nsub']
npol=info['npol']
if nbin!=1:
	data0=d.period_scrunch()[:,:,0]
else:
	data0=d.__read_bin_segment__(0,nperiod)[:,:,0]
data0*=info['chan_weight'].reshape(-1,1)
if nbin>128 or ((nbin==1)&(nperiod>512)):
	data0=fft.irfft(fft.rfft(data0,axis=1)[:,:257],axis=1)
if args.norm:
	data=data0-data0.mean(1).reshape(-1,1)
	std=data.std(1)
	data[std==0]=0
	data[std!=0]=data[std!=0]/std[std!=0].reshape(-1,1)
	data-=data[np.isfinite(data)].min()-1
else:
	data=data0
testdata=copy.deepcopy(data)
testdata=ma.masked_where(testdata<0,testdata)
#
zaplist=[list(np.where(info['chan_weight']==0)[0])]
zapnum=zaplist[0]
zaparray=np.zeros_like(testdata)
zaparray[zapnum,:]=True
testdata.mask=zaparray
zap0=1
#
if args.zap_file:
	if not os.path.isfile(args.zap_file):
		parser.error('The zap channel file is invalid.')
	zchan=np.loadtxt(args.zap_file,dtype=np.int32)
	if np.max(zchan)>=nchan or np.min(zchan)<0:
		parser.error('The zapped channel number is overrange.')
	zap0+=1
	zaplist.append(zchan)
	zapnum=set()
	for i in zaplist:
		zapnum.update(i)
	zapnum=np.array(list(zapnum))
	zaparray=np.zeros_like(testdata)
	zaparray[zapnum,:]=True
	testdata.mask=zaparray
#
if args.mean:
	spec=data0.mean(1)
else:
	spec=data0.std(1)
spec=spec-np.min(spec)
spec0=np.append(0,np.append(spec.repeat(2),0))
spec1=copy.deepcopy(spec0)
cali=args.cal
if args.cal:
	if 'cal' not in info.keys():
		parser.error('The file information does not contain calibration parameters.')
	nchan0=info['nchan']
	if info['cal_mode']=='single':
		cal=info['cal'].T
	else:
		noisedt=info['stt_time']+info['length']/2/86400-info['noise_time0']
		cal=np.polyval(info['cal'],noisedt).T
	cal0=np.concatenate((np.zeros([1,4]),cal.repeat(2,axis=0),np.zeros([1,4])),axis=0)
	cal1=copy.deepcopy(cal0)
freq_start,freq_end=info['freq_start'],info['freq_end']
ylim0=[freq_start,freq_end]
channelwidth=(freq_end-freq_start)/nchan
halfwidth=channelwidth/2
freq=np.linspace(ylim0[0]-halfwidth,ylim0[1]-halfwidth,len(spec)+1).repeat(2)
#
def plotimage(ylim):
	ax.imshow(testdata[::-1,:],aspect='auto',interpolation='nearest',extent=(0,1,ylim0[0]-halfwidth,ylim0[1]-halfwidth),cmap=colormap)
	if cali:
		ax1.plot(cal1,freq,'-')
		ax1.set_xlim(np.min(cal1)*1.1,np.max(cal1)*1.1)
	else:
		ax1.plot(spec1,freq,'k-')
		ax1.set_xlim(0,np.max(spec1)*1.1)
	ax.set_ylim(ylim[0]-halfwidth,ylim[1]-halfwidth)
	ax1.set_ylim(ylim[0]-halfwidth,ylim[1]-halfwidth)
	ax1.set_xticks([])
	ax1.set_yticks([])
	if nbin==1:
		ax.set_xlabel('Integration Length',fontsize=30)
	else:
		ax.set_xlabel('Pulse Phase',fontsize=30)
	ax.set_ylabel('Frequency (MHz)',fontsize=30)
	canvas.draw()
#
def ycal(y,ylim):
	return (fig.bbox.extents[3]-y-ax.bbox.extents[1])/ax.bbox.bounds[3]*(ylim[1]-ylim[0])+ylim[0]
#
def chancal(y):
	return np.int32((y-freq_start)/channelwidth)
#
fig=Figure(figsize=(40,30),dpi=80)
fig.clf()
colormap='jet'
x0,x1,x2=0.13,0.8,0.95
y0,y1=0.11,0.96
ax=fig.add_axes([x0,y0,x1-x0,y1-y0])
ax1=fig.add_axes([x1,y0,x2-x1,y1-y0])
ylim=0
ylimlist=[]
l1 = ln.Line2D([0,1],[0.5,0.5],color='k',transform=fig.transFigure,figure=fig)
fig.lines.append(l1)
#
def leftclick(event):
	global ylim
	if event.x<ax.bbox.extents[0] or event.x>ax1.bbox.extents[2] or event.y<fig.bbox.extents[1] or event.y>fig.bbox.extents[3]: return
	if ylimlist:
		ylim1=ylimlist[-1]
	else:
		ylim1=ylim0
	if ylim==0:
		y=(fig.bbox.extents[3]-event.y)/fig.bbox.extents[3]
		l2=ln.Line2D([0,1],[y,y],color='k',transform=fig.transFigure,figure=fig)
		fig.lines.append(l2)
		canvas.draw()
		ylim=ycal(event.y,ylim1)
	else:
		ylim=[ylim,ycal(event.y,ylim1)]
		ylimlist.append(ylim)
		ax.set_ylim(ylim[0]-halfwidth,ylim[1]-halfwidth)
		ax1.set_ylim(ylim[0]-halfwidth,ylim[1]-halfwidth)
		fig.lines=[l1]
		canvas.draw()
		ylim=0
#
def rightclick(event):
	global ylim
	if event.x<ax.bbox.extents[0] or event.x>ax1.bbox.extents[2] or event.y<fig.bbox.extents[1] or event.y>fig.bbox.extents[3]: return
	if ylimlist:
		ylim1=ylimlist[-1]
	else:
		ylim1=ylim0
	chan=chancal(ycal(event.y,ylim1))
	chan=min(chan,nchan-1)
	if chan>=0 and chan<nchan:
		if ylim==0:
			zaplist.append([chan])
		else:
			if chancal(ylim)<=chan:
				zaplist.append(list(range(max(0,chancal(ylim)),chan+1)))
			else:
				zaplist.append(list(range(max(0,chan),chancal(ylim)+1)))
			ylim=0
		update_image()
#
def update_image():
	global ylim,spec1
	zapnum=set()
	for i in zaplist:
		zapnum.update(i)
	zapnum=np.int32(list(zapnum))
	zaparray=np.zeros_like(testdata)
	zaparray[zapnum,:]=True
	testdata.mask=zaparray
	if ylimlist:
		ylim=ylimlist[-1]
	else:
		ylim=ylim0
	ax.cla()
	ax1.cla()
	spec1=copy.deepcopy(spec0)
	spec1[2*zapnum+1]=0
	spec1[2*zapnum+2]=0
	if args.cal:
		global cal1
		cal1=copy.deepcopy(cal0)
		cal1[2*zapnum+1]=0
		cal1[2*zapnum+2]=0
	fig.lines=[l1]
	plotimage(ylim)
	ylim=0
#
def move_tk(event):
	if event.x>ax.bbox.extents[0] and event.x<ax1.bbox.extents[2]: 
		y=(fig.bbox.extents[3]-event.y)/fig.bbox.extents[3]
		l1.set_ydata([y,y])
		canvas.draw()
#
def move_gtk(window,event):
	if event.x>ax.bbox.extents[0] and event.x<ax1.bbox.extents[2]: 
		y=(fig.bbox.extents[3]-event.y)/fig.bbox.extents[3]
		l1.set_ydata([y,y])
		canvas.draw()
#
def press_gtk(window,event):
	keymotion(gtk.gdk.keyval_name(event.keyval))
#
def press_tk(event):
	keymotion(event.keysym)
#
def keymotion(a):
	global ylim
	if a=='q':
		root.destroy()
		zapnum=set()
		for i in zaplist:
			zapnum.update(i)
		zapnum=np.sort(list(zapnum))
		zapnum=zapnum[(zapnum>=0)&(zapnum<nchan)]
		np.savetxt(args.filename[:-3]+'_zap.txt',zapnum,fmt='%i')
	elif a=='s':
		root.destroy()
		zapnum=set()
		for i in zaplist:
			zapnum.update(i)
		zapnum=np.sort(list(zapnum))
		zapnum=list(zapnum[(zapnum>=0)&(zapnum<nchan)])
		weight=info['chan_weight']
		weight[zapnum]=0
		info['chan_weight']=weight
		new_name='.'.join(args.filename.split('.')[:-1])+'_zap.ld'
		sys.stdout.write("file saving...\n\n")
		shutil.copyfile(args.filename,newname)
		save=ld.ld(newname)
		for i in zapnum:
			save.write_chan(np.zeros(nperiod*nbin*npol),i)
		command=['ldzap.py']
		if 'history' in info.keys():
			info['history'].append(command)
			info['file_time'].append(time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime()))
		else:
			info['history']=[command]
			info['file_time']=[time.strftime('%Y-%m-%dT%H:%M:%S',time.gmtime())]
		save.write_info(info)
	elif a=='r':
		if ylimlist:
			ylimlist.pop()
		else: return
		if ylimlist:
			ylim=ylimlist[-1]
		else:
			ylim=ylim0
		ax.set_ylim(ylim[0]-halfwidth,ylim[1]-halfwidth)
		ax1.set_ylim(ylim[0]-halfwidth,ylim[1]-halfwidth)
		canvas.draw()
		ylim=0
	elif a=='u':
		if len(zaplist)>zap0:
			zaplist.pop()
		else: return
		update_image()
	elif a=='c':
		global cali
		cali=not cali
		update_image()
	elif a=='h':
		sys.stdout.write("\nldzap interactive commands\n\n")
		sys.stdout.write("Mouse:\n")
		sys.stdout.write("  Left-click selects the start of a range\n")
		sys.stdout.write("    then left-click again to zoom, or right-click to zap.\n")
		sys.stdout.write("  Right-click zaps current cursor location.\n\n")
		sys.stdout.write("Keyboard:\n")
		sys.stdout.write("  h  Show this help\n")
		sys.stdout.write("  u  Undo last zap command\n")
		sys.stdout.write("  r  Reset zoom and update dynamic spectrum\n")
		sys.stdout.write("  s  Save zapped version as (filename)_zap.ld and quit\n")
		sys.stdout.write("  q  Exit program\n\n")
#
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import matplotlib as mpl
mpl.use('TkAgg')
root=tk.Tk()
root.title(args.filename)
root.geometry('800x600+100+100')
canvas=FigureCanvasTkAgg(fig,master=root)
canvas.get_tk_widget().grid()  
canvas.get_tk_widget().pack(fill='both')
root.bind('<KeyPress>',press_tk)
root.bind('<ButtonPress-1>',leftclick)
root.bind('<ButtonPress-3>',rightclick)
root.bind('<Motion>',move_tk)
canvas.draw()
plotimage(ylim0)
root.mainloop()
#except:
	#raise(Exception,"Only Tkinter available for this programme.")
	#import gtk
	#from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg
	#root=gtk.Window()
	#root.set_title(args.filename)
	#root.set_size_request(800,600)
	#box=gtk.VBox()
	#canvas=FigureCanvasGTKAgg(fig)
	#box.pack_start(canvas)
	#root.add(box)
	#root.modify_bg('normal',gtk.gdk.Color('#fff'))
	#root.show_all()
	#root.connect('destroy',gtk.main_quit)
	#root.connect('key-press-event',press_gtk)
	#root.connect('move-cursor',move_gtk)
	#gtk.main()

