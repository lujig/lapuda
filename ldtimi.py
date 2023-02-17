#!/usr/bin/env python
import numpy as np
import time_eph as te
import psr_read as pr
import psr_model as pm
import scipy.optimize as so
import warnings as wn
import argparse as ap
from matplotlib.figure import Figure
import matplotlib.patches as mp
import ld,os,copy,sys
wn.filterwarnings('ignore')
#
version='JigLu_20220633'
parser=ap.ArgumentParser(prog='ldtim',description='Get the timing solution the ToA.',epilog='Ver '+version)
parser.add_argument('-v','--version',action='version',version=version)
parser.add_argument("filename",help="input ToA file with ld or txt format")
parser.add_argument('-p','--par',dest='par',help="input par file")
#
args=(parser.parse_args())
#
if not os.path.isfile(args.filename):
	parser.error('ToA file name is invalid.')
d=ld.ld(args.filename)

data=d.read_chan(0)[:,:,0]
info=d.read_info()
if args.par:
	psr0=pr.psr(args.par,parfile=True)
else:
	psr0=pr.psr(info['psr_name'])
#
jj_list=[np.ones(len(data),dtype=np.bool)]
select_list=[np.ones(len(data),dtype=np.bool)]
time_jump=np.ones(len(data),dtype=np.int64)
plotlim_list=[]
merge_mark,restart,reset_select,fit_mark=False,True,False,False
err_limit=1.0
dtunit='phase'
xax='mjd'
yax='post'
paralist0=['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'raj', 'decj', 'pmra', 'pmdec', 'pmra2', 'pmdec2', 'elong', 'elat', 'pmelong', 'pmelat', 'pmelong2', 'pmelat2', 't0', 'tasc', 'pb', 'ecc', 'a1', 'om', 'eps1', 'eps2', 'sini', 'm2', 'fb0', 'fb1', 'fb2', 'fb3', 'pbdot', 'a1dot', 'omdot', 'edot', 'eps1dot', 'eps2dot', 'gamma', 'bpjph','bpja1','bpjec','bpjom','bpjpb', 'h3', 'h4', 'stig', 'kom', 'kin', 'mtot', 'a2dot', 'e2dot', 'orbpx', 'dr', 'dtheta', 'dth', 'a0', 'b0', 'om2dot', 'pb2dot', 'shapmax']
cur_x,cur_y,cur_x0,cur_y0=0,0,0,0
key_on,mouse_on,rect=False,False,0
#
fig=Figure(figsize=(8,6),dpi=100)
fig.clf()
x1,x2=0.18,0.95
y1,y2,y3=0.16,0.56,0.96
#
def merge(time,dt,dterr,freq,dm,dmerr,period,jj1,se):
	global mergej
	date,sec=time.local.date,time.local.second
	if 'mergej' in locals().keys():
		jj=mergej.copy()
	else:
		ttmp=time.local.mjd
		nt=len(ttmp)
		jj=np.zeros(nt,dtype=np.int32)
		j0=1
		t0=ttmp[0]
		merge_time=0.5
		for i in np.arange(nt):
			if np.abs(ttmp[i]-t0)<merge_time:
				jj[i]=j0
			else:
				t0=ttmp[i]
				j0+=1
				jj[i]=j0
		mergej=jj
	#
	dt2=[]
	dt2err=[]
	date2=[]
	sec2=[]
	dm2=[]
	dmerr2=[]
	freq2=[]
	period2=[]
	jj2=[]
	se2=[]
	for i in np.arange(jj[-1])+1:
		setj0=jj==i
		setj1=setj0&jj1&se
		if not np.any(setj1):
			setj=setj0
		else:
			setj=setj1
		t0=dt[setj]
		ta=dterr[setj]
		errtmp=np.sqrt(1/(1/ta**2).sum())+t0.std()
		date2.append(date[setj].mean())
		sec2.append(sec[setj].mean())
		dm2.append(dm[setj].mean())
		dmerr2.append(dmerr[setj].mean())
		t0a=(t0/ta**2).sum()/(1/ta**2).sum()
		dt2.append(t0a)
		dt2err.append(np.sqrt(((t0-t0a)**2/ta**2).sum()/(1/ta**2).sum()))
		freq2.append(freq[setj].mean())
		period2.append(period[setj].mean())
		jj2.append(np.any(jj1[setj])&(errtmp<err_limit))
		se2.append(np.any(se[setj]))
	date2=np.array(date2)
	sec2=np.array(sec2)
	dt2=np.array(dt2)
	dt2err=np.array(dt2err)
	freq2=np.array(freq2)
	dm2=np.array(dm2)
	dmerr2=np.array(dmerr2)
	period2=np.array(period2)
	jj2=np.array(jj2)
	se2=np.array(se2)
	return te.times(te.time(date2,sec2)),dt2,dt2err,freq2,dm2,dmerr2,period2,jj2,se2
#
def psrfit(psr,paras,time,dt,toae,freq,jj,se):
	psrt=pm.psr_timing(psr,time,freq)
	lpara=len(paras)
	x0=np.zeros(lpara+1)
	jj0=jj&se
	for i in np.arange(lpara):
		tmp=psr.__getattribute__(paras[i])
		if type(tmp)==te.time:
			x0[i]=tmp.mjd[0]
		else:
			x0[i]=tmp
	#
	def fit(para):
		psr1=psr.copy()
		for i in np.arange(lpara):
			psr1.modify(paras[i],para[i])
		psrt1=pm.psr_timing(psr1,time,freq)
		dphase=psrt1.phase.minus(psrt.phase)
		return dphase.phase+para[-1]
	#
	def dfunc(para):
		psr1=psr.copy()
		for i in np.arange(lpara):
			psr1.modify(paras[i],para[i])
		psrt1=pm.psr_timing(psr1,time,freq)
		tmp=psrt1.phase_der_para(paras).T
		return (np.concatenate((tmp,np.ones([time.size,1])),axis=1)/toae.reshape(-1,1))[jj0]
	#
	def resi(para):
		return ((fit(para)+dt)/toae)[jj0]
	#
	a=so.leastsq(resi,x0=x0,full_output=True,Dfun=dfunc)
	popt,pcov=a[0],(np.diag(a[1])*(resi(a[0])**2).sum()/(len(dt)-len(x0)))**0.5
	psr1=psr.copy()
	for i in np.arange(lpara):
		psr1.modify(paras[i],popt[i])
	psrt2=pm.psr_timing(psr1,time,freq)
	#print(fit(popt),dt,x0)
	return popt,np.sqrt(pcov),fit(popt)+dt,resi(popt),psr1,psrt2
#
def plot(time,psrt,dt,dterr,dm,dmerr,jj,se):
	global ax1,xax,yax,xaxis,yaxis,yerr,plotlim_list,lines,points,yunit
	marker_colors=np.array(['b']*len(jj))
	marker_colors[np.logical_not(se)]='c'
	marker_colors[np.logical_not(jj)]='r'
	jj0=jj&se
	fig.clf()
	if xax=='orbit':
		xaxis=psrt.orbits%1
		xlim=0,1
		xlabel='Orbital Phase'
	elif xax=='lst':
		lst=psrt.time.lst
		lst[(lst-psr.raj/np.pi/2)>0.5]-=1
		lst[(lst-psr.raj/np.pi/2)<-0.5]+=1
		xaxis=lst*24
		xlabel='Sidereal Time (h)'
	elif xax=='mjd':
		xaxis=time.local.mjd
		xlabel='MJD (d)'
	elif xax=='year':
		xaxis=te.mjd2datetime(time.local.mjd)[4]
		xlim=0,366
		xlabel='Day in a Year (d)'
	#
	if len(xaxis)>len(jj):
		xaxtmp=np.zeros(mergej[-1])
		for i in np.arange(mergej[-1]):
			setj=mergej==(i+1)
			xaxtmp[i]=xaxis[setj].mean()
		xaxis=xaxtmp
	if xax=='lst':
		xmax,xmin=np.max(xaxis[jj0]),np.min(xaxis[jj0])
		xlim=xmin*1.05-xmax*0.05,-xmin*0.05+xmax*1.05
	elif xax=='mjd':
		xlim=xaxis[jj0][0]*1.05-xaxis[jj0][-1]*0.05,-xaxis[jj0][0]*0.05+xaxis[jj0][-1]*1.05	
	#
	if yax=='prepost':
		ax1=fig.add_axes((x1,y2,x2-x1,y3-y2))
		ax2=fig.add_axes((x1,y1,x2-x1,y2-y1))
		yaxis,yerr=dt,dterr
		lines=ax1.errorbar(xaxis,yaxis,yerr,fmt='none').get_children()[0]
		points=ax1.scatter(xaxis,yaxis,marker='.')
		line2=ax2.errorbar(xaxis,res,dterr,fmt='none').get_children()[0]
		points2=ax2.scatter(xaxis,res,marker='.')
		points2.set_color(marker_colors)
		line2.set_color(marker_colors)
		ax2.set_xlim(*xlim)
		ax2.set_xlabel(xlabel,fontsize=25,family='serif')
		ax1.set_ylabel('Phase Resi.'+yunit,fontsize=25,family='serif')
		ax2.set_ylabel('Fit Resi.'+yunit,fontsize=25,family='serif')
		ax1.set_xticks([])
		mark_text='Pre-fit and Post-fit'
	else:
		ax1=fig.add_axes((x1,y1,x2-x1,y3-y1))
		if yax=='post':
			yaxis,yerr=res,dterr
			ax1.set_ylabel('Fit Resi.'+yunit,fontsize=25,family='serif')
			mark_text='Post-fit'
		if yax=='pre':
			yaxis,yerr=dt,dterr
			ax1.set_ylabel('Phase Resi.'+yunit,fontsize=25,family='serif')
			mark_text='Pre-fit'
		elif yax=='dm':
			yaxis,yerr=dm,dmerr
			ax1.set_ylabel('DM',fontsize=25,family='serif')
			mark_text='DM'
		lines=ax1.errorbar(xaxis,yaxis,yerr,fmt='none').get_children()[0]
		points=ax1.scatter(xaxis,yaxis,marker='.')
		ax1.set_xlabel(xlabel,fontsize=25,family='serif')
	lines.set_color(marker_colors)
	points.set_color(marker_colors)
	ymax,ymin=np.max((yaxis+yerr)[jj0]),np.min((yaxis-yerr)[jj0])
	ylim=ymin*1.05-ymax*0.05,-ymin*0.05+ymax*1.05
	ax1.set_xlim(*xlim)
	ax1.set_ylim(ylim)
	fig.text(x2-0.05,y3-0.05,mark_text,fontsize=25,family='serif',color='green',va='top',ha='right')
	canvas.draw()
	plotlim=[*ax1.get_xlim(),*ax1.get_ylim()]
	plotlim_list=[plotlim]
#
def adjust():
	global jj_list,psr,time,dt,res,dterr,dm,yunit,select_list,reset_select,freq,dmerr,period,restart,fit_mark,time_jump
	errentry.delete(0,np.int(np.uint32(-1)/2))
	errentry.insert(0,str(err_limit))
	if reset_select:
		select_list=[np.ones(len(data),dtype=np.bool)]
		reset_select=False
	if restart:
		date,sec,toae,dt,dterr,freq_start,freq_end,dm,dmerr,period=data.T
		jj_list=[jj_list[0]]
		jj=jj_list[0]&(dterr<err_limit)
		if np.any(jj_list[0]!=jj): jj_list.append(jj)
		select_list=[select_list[0]]
		se=select_list[-1]
		freq=(freq_start+freq_end)/2
		nt=len(date)
		time=te.times(te.time(date,sec))
		psr=psr0.copy()
		psrt=pm.psr_timing(psr,time,freq)
		phase=psrt.phase
		dt=phase.offset%1
		dterr=toae/period
		dt[dt>0.5]-=1
		restart=False
		time_jump=np.zeros_like(dt,dtype=np.int64)
	else:
		jj0=jj_list[-1]
		se=select_list[-1]
		jj=jj0&(dterr<err_limit)
		if np.any(jj0!=jj): jj_list.append(jj)
		psrt=pm.psr_timing(psr,time,freq)
		phase=psrt.phase
		dt=phase.offset%1
		dt[dt>0.5]-=1
		dt+=time_jump
	if merge_mark:
		time1,dt1,dterr1,freq1,dm1,dmerr1,period1,jj1,se1=merge(time,dt,dterr,freq,dm,dmerr,period,jj,se)
	else:
		time1,dt1,dterr1,freq1,dm1,dmerr1,period1,jj1,se1=time.copy(),dt.copy(),dterr.copy(),freq.copy(),dm.copy(),dmerr.copy(),period.copy(),jj,se
	paras=np.array(paralist)[list(pbox.curselection())]
	if len(paras) and fit_mark:
		psrp,psrpe,res,rese,psr,psrt=psrfit(psr,paras,time1,dt1,dterr1,freq1,jj1,se1)
	else:
		res=dt1
	time_jump=np.zeros_like(dt,dtype=np.int64)
	if dtunit=='time':
		period_tmp=period1.mean()*1e6
		res=res*period_tmp
		dt1=(dt1-dt1.mean())*period_tmp
		dterr1*=period_tmp
	if len(paras) and fit_mark:
		phasestd=np.sqrt((rese**2).sum()/(1/dterr1[jj1]**2).sum())
		if dtunit=='time': phasestd/=1e6
		else: phasestd*=period1.mean()
		print('RMS of the fit residuals (s):',phasestd)
		fit_mark=False
	if dtunit=='time':
		yunit=' ($\mu$s)'
	elif dtunit=='phase':
		yunit=''
	plot(time1,psrt,dt1,dterr1,dm1,dmerr1,jj1,se1)
#
def zoom(cur_x0,cur_x,cur_y0,cur_y,reset=False):
	global ax1,plotlim_list
	tmp_x0,tmp_x1=np.sort([cur_x0,cur_x])
	tmp_y0,tmp_y1=np.sort([cur_y0,cur_y])
	ax1.set_xlim(tmp_x0,tmp_x1)
	ax1.set_ylim(tmp_y0,tmp_y1)
	canvas.draw()
	if reset:
		plotlim_list=[[tmp_x0,tmp_x1,tmp_y0,tmp_y1]]
	else:
		plotlim=[*ax1.get_xlim(),*ax1.get_ylim()]
		if plotlim!=plotlim_list[-1]:
			plotlim_list.append(plotlim)
#
def delete_range():
	global jj_list,ax1,lines,points
	tmp_x0,tmp_x1=np.sort([cur_x0,cur_x])
	tmp_y0,tmp_y1=np.sort([cur_y0,cur_y])
	jj0=(xaxis<tmp_x0)|(xaxis>tmp_x1)|(yaxis<tmp_y0)|(yaxis>tmp_y1)
	if not merge_mark:
		jj=jj_list[-1]&jj0
		se=select_list[-1]
		jj0=jj.copy()
	else:
		jj=np.zeros(len(jj_list[0]),dtype=np.bool)
		se=np.zeros_like(jj0,dtype=np.bool)
		for i in np.arange(len(jj0))+1:
			setj=mergej==i
			jj0[i-1]&=np.any(jj_list[-1][setj])
			se[i-1]=np.any(select_list[-1][setj])
			jj[setj]=jj0[i-1]
	if np.any(jj!=jj_list[-1]):
		jj_list.append(jj)
		marker_colors=np.array(['b']*len(jj0))
		marker_colors[np.logical_not(se)]='c'
		marker_colors[np.logical_not(jj0)]='r'
		lines.set_color(marker_colors)
		points.set_color(marker_colors)
		canvas.draw()	
#
def fit(fit=True,merge=True):
	global select_list,fit_mark
	xlim1,xlim2,ylim1,ylim2=*ax1.get_xlim(),*ax1.get_ylim()
	se=(xaxis>xlim1)&(xaxis<xlim2)&(yaxis>ylim1)&(yaxis<ylim2)
	if not np.logical_xor(merge_mark,merge):
		se0=np.zeros(len(select_list[0]),dtype=np.bool)
		for i in np.arange(len(se))+1:
			se0[mergej==i]=se[i-1]
		se=se0.copy()
	if np.any(se!=select_list[-1]):
		select_list.append(se)
	fit_mark=fit
	adjust()
#
def click(event):
	global cur_x0,cur_y0,mouse_on,rect
	if mouse_on: return
	xlim1,xlim2,ylim1,ylim2=*ax1.get_xlim(),*ax1.get_ylim()
	axx1,axy1,axx2,axy2=ax1.get_position().extents
	kx,ky=((event.x-fig.bbox.extents[0])/fig.bbox.extents[2]-axx1)/(axx1-axx2),((fig.bbox.extents[3]-event.y)/fig.bbox.extents[3]-axy1)/(axy1-axy2)
	if kx>0 or kx<-1 or ky>0 or ky<-1: return
	mouse_on=True
	cur_x0=kx*(xlim1-xlim2)+xlim1
	cur_y0=ky*(ylim1-ylim2)+ylim1
	rect=mp.Rectangle((cur_x0,cur_y0),0,0,ec='k',fill=False,linestyle='--')
	ax1.add_patch(rect)
	canvas.draw()
#
def leftrelease(event):
	global mouse_on
	if not mouse_on: return
	mouse_on=False
	zoom(cur_x0,cur_x,cur_y0,cur_y)
	ax1.patches.clear()
	canvas.draw()
#
def rightrelease(event):
	global mouse_on
	if not mouse_on: return
	mouse_on=False
	delete_range()
	ax1.patches.clear()
	canvas.draw()
#
def move_tk(event):
	global cur_x,cur_y,rect
	xlim1,xlim2,ylim1,ylim2=*ax1.get_xlim(),*ax1.get_ylim()
	axx1,axy1,axx2,axy2=ax1.get_position().extents
	cur_x=((event.x-fig.bbox.extents[0])/fig.bbox.extents[2]-axx1)*(xlim1-xlim2)/(axx1-axx2)+xlim1
	cur_y=((fig.bbox.extents[3]-event.y)/fig.bbox.extents[3]-axy1)*(ylim1-ylim2)/(axy1-axy2)+ylim1
	if not mouse_on: return
	rect.set_height(cur_y-cur_y0)
	rect.set_width(cur_x-cur_x0)
	canvas.draw()
#
def press(event):
	global key_on
	key_on=event.keysym
	if key_on in ['r','e','a','s','u','f','m','b','h','p','q','d','x']:
		keymotion(key_on)
#
def release(event):
	global key_on
	key_on=''
#
def add_jump(a):
	global time_jump
	if a=='a': jump=1
	else: jump=-1
	jump=(xaxis>cur_x)*jump
	if merge_mark:
		time_jump0=np.zeros(len(jj_list[0]),dtype=np.bool)
		for i in np.arange(len(jump))+1:
			setj=mergej==i
			time_jump[setj]=jump[i-1]
	else:
		time_jump0=jump
	time_jump+=time_jump0
	adjust()
#
def keymotion(a):
	global plotlim_list,select_list,jj_list,lines,points,merge_mark,reset_select,restart
	if a=='r':
		if len(plotlim_list)>1:
			plotlim_list.pop()
			ax1.set_xlim(*plotlim_list[-1][:2])
			ax1.set_ylim(*plotlim_list[-1][2:])
			canvas.draw()
	elif a=='e':
		if len(select_list)>1:
			if xax=='orbit': xlim=0,1
			elif xax=='year': xlim=0,366
			else: 
				xmax,xmin=np.max(xaxis),np.min(xaxis)
				xlim=xmin*1.05-xmax*0.05,-xmin*0.05+xmax*1.05
			ymax,ymin=np.max(yaxis),np.min(yaxis)
			ylim=ymin*1.05-ymax*0.05,-ymin*0.05+ymax*1.05
			zoom(*xlim,*ylim,reset=True)
			reset_select=True
	elif a=='u':
		if len(jj_list)>1:
			jj_list.pop()
			jj0=jj_list[-1]
			se0=select_list[-1]
			if merge_mark:
				jj=np.zeros(len(xaxis),dtype=np.bool)
				se=np.zeros_like(jj,dtype=np.bool)
				for i in np.arange(len(jj))+1:
					setj=mergej==i
					jj[i-1]=np.any(jj0[setj])
					se[i-1]=np.any(se0[setj])
			else:
				jj=jj0
				se=se0
			marker_colors=np.array(['b']*len(jj))
			marker_colors[np.logical_not(se)]='c'
			marker_colors[np.logical_not(jj)]='r'
			lines.set_color(marker_colors)
			points.set_color(marker_colors)
			canvas.draw()
	elif a=='f':
		fit(fit=True)
	elif a=='a' or a=='d':
		add_jump(a)
	elif a=='m':
		merge_mark=not merge_mark
		fit(fit=False,merge=False)
	elif a=='b':
		restart=True
		adjust()	
	elif a=='p':
		print('Pulsar parameters:')
		print(psr)
	elif a=='q':
		root.destroy()
	elif a=='s':
		psrfile=tk.filedialog.asksaveasfilename(defaultextension='.par')
		if psrfile:
			psr.writepar(psrfile)
	elif a=='x':
		resfile=tk.filedialog.asksaveasfilename(defaultextension='.res')
		if resfile:
			tmp=np.array([time.local.date,time.local.second,dt,dterr,np.int8(jj_list[-1]),np.int8(select_list[-1])]).T
			np.savetxt(resfile,tmp,fmt=['%i', '%5.11f', '%.16f', '%.16f', '%i', '%i'])
	elif a=='h':
		sys.stdout.write("\nldzap interactive commands\n\n")
		sys.stdout.write("Mouse:\n")
		sys.stdout.write("  Left-click select a rectangle region to zoom in.\n")
		sys.stdout.write("  Right-click select a rectangle region to delete the ToAs therein.\n\n")
		sys.stdout.write("Keyboard:\n")
		sys.stdout.write("  h    Show this help\n")
		sys.stdout.write("  u    Undo last delete command\n")
		sys.stdout.write("  r    Reset zoom to the last selection\n")
		sys.stdout.write("  e    Reset zoom to the initial region\n")
		sys.stdout.write("  b    Restart the fitting\n")
		sys.stdout.write("  m    Merge the neighboring ToAs\n")
		sys.stdout.write("  a/d  Add/substract one period for the ToAs on the right side of the cursor\n")
		sys.stdout.write("  f    Fit the ToAs displayed currently\n")
		sys.stdout.write("  p    Print the current parfile\n")
		sys.stdout.write("  s    Save parfile to a specified file\n")
		sys.stdout.write("  x    Save the fit residuals to a specified file\n")
		sys.stdout.write("  q    Exit program\n\n")
	return
#
def xmode(mode):
	global xax,select_list
	if mode==xax: return
	xax=mode
	select_list=[select_list[0]]
	adjust()
#
def ymode(mode):
	global yax,select_list,dtunit,y5bttn
	if mode==yax: return
	select_list=[select_list[0]]
	if mode=='time':
		if dtunit=='time': dtunit,tmp='phase','Time'
		else: dtunit,tmp='time','Phase'
		y5bttn.config(text=tmp)
		adjust()
	else:
		yax=mode
		adjust()
#
def submit_err(event):
	global err_limit
	err=errentry.get()
	try:
		err_limit=np.float64(err)
	except:
		tk.messagebox.showwarning('Error!','The inputing error limit is invalid!')
	errentry.delete(0,np.int(np.uint32(-1)/2))
	errentry.insert(0,str(err_limit))
	frame0.focus()
#
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import matplotlib as mpl
mpl.use('TkAgg')
root=tk.Tk()
root.title('Timing of PSR '+psr0.name)
root.geometry('950x700+100+100')
root.configure(bg='white')
root.resizable(False,False)
frame0=tk.Frame(root,bg='white')
frame0.grid(row=0,column=0)
canvas=FigureCanvasTkAgg(fig,master=frame0)
canvas.get_tk_widget().grid()  
canvas.get_tk_widget().pack(fill='y')
root.bind('<KeyPress>',press)
root.bind('<KeyRelease>',release)
root.bind('<ButtonPress-1>',click)
root.bind('<ButtonPress-3>',click)
root.bind('<ButtonRelease-1>',leftrelease)
root.bind('<ButtonRelease-3>',rightrelease)
root.bind('<Motion>',move_tk)
sframe=tk.Frame(root,bg='white')
sframe.grid(row=0,column=1,rowspan=2)
tk.Label(sframe,text='PARA List:',bg='white',font=('serif',20)).grid(row=0,column=0)
pboxframe=tk.Frame(sframe,bg='white')
pboxframe.grid(row=1,column=0)
pbox=tk.Listbox(pboxframe,height=16,width=10,selectmode=tk.MULTIPLE,font=('serif',17),exportselection=0)
paralist=[]
for i in paralist0:
	if i in psr0.paras: paralist.append(i)
pbox.insert(0,*map(lambda x:x.upper(),paralist))
sbar1= tk.Scrollbar(pboxframe,command=pbox.yview)
sbar1.pack(side=tk.RIGHT,fill='y')
pbox.pack(fill='both',anchor='w',expand='no')
pbox.config(yscrollcommand = sbar1.set)
tk.Label(sframe,text='Err limit:',bg='white',font=('serif',20)).grid(row=2,column=0)
errentry=tk.Entry(sframe,width=10,font=('serif',17))
errentry.grid(row=3,column=0)
errentry.insert(0,str(err_limit))
errentry.bind('<Return>',submit_err)
pframe=tk.Frame(root,bg='white')
pframe.grid(row=1,column=0)
tk.Label(pframe,text='Plot mode: ',bg='white',font=('serif',20)).grid(row=0,column=0)
y1bttn=tk.Button(pframe,text='Pre-fit',command=lambda: ymode('pre'),bg='white',activebackground='#E5E35B',font=('serif',20))
y1bttn.grid(row=0,column=1)
y2bttn=tk.Button(pframe,text='Post-fit',command=lambda: ymode('post'),bg='white',activebackground='#E5E35B',font=('serif',20))
y2bttn.grid(row=0,column=2)
y3bttn=tk.Button(pframe,text='Pre&Post',command=lambda: ymode('prepost'),bg='white',activebackground='#E5E35B',font=('serif',20))
y3bttn.grid(row=0,column=3)
y4bttn=tk.Button(pframe,text='DM',command=lambda: ymode('dm'),bg='white',activebackground='#E5E35B',font=('serif',20))
y4bttn.grid(row=0,column=4)
y5bttn=tk.Button(pframe,text='Time',command=lambda: ymode('time'),bg='white',activebackground='#E5E35B',font=('serif',20))
y5bttn.grid(row=0,column=5)
tk.Label(pframe,text='X-axis ',bg='white',font=('serif',20)).grid(row=1,column=0)
x1bttn=tk.Button(pframe,text='MJD',command=lambda: xmode('mjd'),bg='white',activebackground='#E5E35B',font=('serif',20))
x1bttn.grid(row=1,column=1)
x2bttn=tk.Button(pframe,text='Orbit',command=lambda: xmode('orbit'),bg='white',activebackground='#E5E35B',font=('serif',20))
x2bttn.grid(row=1,column=2)
x3bttn=tk.Button(pframe,text='LST',command=lambda: xmode('lst'),bg='white',activebackground='#E5E35B',font=('serif',20))
x3bttn.grid(row=1,column=3)
x4bttn=tk.Button(pframe,text='Year',command=lambda: xmode('year'),bg='white',activebackground='#E5E35B',font=('serif',20))
x4bttn.grid(row=1,column=4)
#
adjust()
canvas.draw()
#
root.mainloop()

