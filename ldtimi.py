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
dirname=os.path.split(os.path.realpath(__file__))[0]
sys.path.append(dirname+'/doc')
import text
#
text=text.output_text('ldtimi')
version='JigLu_20220633'
parser=ap.ArgumentParser(prog='ldtimi',description=text.help,epilog='Ver '+version,add_help=False,formatter_class=lambda prog: ap.RawTextHelpFormatter(prog, max_help_position=50))
parser.add_argument('-h', '--help', action='help', default=ap.SUPPRESS,help=text.help_h)
parser.add_argument('-v','--version',action='version',version=version,help=text.help_v)
parser.add_argument("filename",help=text.help_filename)
parser.add_argument('-p','--par',dest='par',help=text.help_p)
parser.add_argument('--tr',dest='trange',help=text.help_tr)
#
args=(parser.parse_args())
#
if not os.path.isfile(args.filename):
	parser.error(text.error_nfn)
d=ld.ld(args.filename)
dirname,_=os.path.split(os.path.abspath(args.filename))
#
data=d.read_chan(0)[:,:,0]
info=d.read_info()
if args.par:
	psr0=pr.psr(args.par,parfile=True)
else:
	psr0=pr.psr(info['pulsar_info']['psr_name'])
#
if args.trange:
	time_start0,time_end0=np.float64(args.trange.split(','))
	if time_end0<=time_start0: parser.error(text.error_slte)
	mjdtime=data[:,0]+data[:,1]/86400
	jj_list=[-1,['Time limit',(mjdtime>time_start0)&(mjdtime<time_end0)]]
else:
	jj_list=[-1,['No Delete',np.ones(len(data),dtype=bool)]]	# the list which records the reserved ToAs
#
fileinfo=info['original_data_info']['filenames']
filenames=np.array(list(map(lambda x:x[0],fileinfo)))
uniqf=np.unique(filenames)
select_list=[-1,['All points',np.ones(len(data),dtype=bool)]]	# the list which records the ToAs in zoom in region
jump_list=[-1,['No jumps',np.zeros(len(data),dtype=np.int64)]]	# the list which records the time-jump for each ToAs
merge_mark,zoom_mark,profwindow=False,False,False	# the marks for merging ToAs, zooming region and plotting profile
err_limit=1.0
dtunit='phase'
paralist0=['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'raj', 'decj', 'pmra', 'pmdec', 'pmra2', 'pmdec2', 'elong', 'elat', 'pmelong', 'pmelat', 'pmelong2', 'pmelat2', 't0', 'tasc', 'pb', 'ecc', 'a1', 'om', 'eps1', 'eps2', 'sini', 'm2', 'fb0', 'fb1', 'fb2', 'fb3', 'pbdot', 'a1dot', 'omdot', 'edot', 'eps1dot', 'eps2dot', 'gamma', 'bpjph','bpja1','bpjec','bpjom','bpjpb', 'h3', 'h4', 'stig', 'kom', 'kin', 'mtot', 'a2dot', 'e2dot', 'orbpx', 'dr', 'dtheta', 'dth', 'a0', 'b0', 'om2dot', 'pb2dot', 'shapmax']
date,sec,toae,dt,dterr,freq_start,freq_end,dm,dmerr,period=data.T
freq=(freq_start+freq_end)/2
period=period.mean()
nt=len(date)
time=te.times(te.time(date,sec,scale=info['telescope_info']['telename']))
psr=psr0.copy()
psrt=pm.psr_timing(psr,time,np.inf)
phase=psrt.phase
dt=phase.offset%1
dterr=toae/period
dt[dt>0.5]-=1
fit_list=[[dt,dterr,freq,dm,dmerr,psr,psrt,jj_list[jj_list[0]][1]&select_list[select_list[0]][1]]]
action_list=[-1]
uintsize=int(np.array([-1]).astype(np.uint32)[0]/2)
#
cur_x,cur_y,cur_x0,cur_y0=0,0,0,0
key_on,mouse_on='',False
#
fig=Figure(figsize=(8,6),dpi=100)
fig.clf()
x1,x2=0.18,0.95
y1,y2,y3=0.16,0.56,0.96
#
def merge():	# merge the ToAs in one observation
	dt,dterr,freq,dm,dmerr,psr,psrt,fitj=copy.deepcopy(fit_list[-1])
	se=select_list[select_list[0]][1].copy()
	jj=jj_list[jj_list[0]][1].copy()
	jump=jump_list[jump_list[0]][1].copy()
	date,sec=time.local.date,time.local.second
	dt2=[]
	dt2err=[]
	date2=[]
	sec2=[]
	dm2=[]
	dmerr2=[]
	freq2=[]
	jj2=[]
	se2=[]
	for i in np.arange(uniqf.size):
		setj0=(filenames==uniqf[i])
		setj1=setj0&jj
		if not np.any(setj1):
			setj=setj0
		else:
			setj=setj1
		date2.append(date[setj].mean())
		sec2.append(sec[setj].mean())
		dm2.append(dm[setj].mean())
		dmerr2.append(dmerr[setj].mean())
		freq2.append(freq[setj].mean())
		t0=dt[setj]+jump[setj]
		ta=dterr[setj]
		errtmp=np.sqrt(1/(1/ta**2).sum())+t0.std()
		if not np.any(setj1): jj2.append(False)
		else: jj2.append(np.any(jj[setj]))
		if len(t0)==1:
			dt2.append(t0[0])
			dt2err.append(ta[0])
		else:
			t0a=(t0/ta**2).sum()/(1/ta**2).sum()
			dt2.append(t0a)
			dt2err.append(np.sqrt(((t0-t0a)**2/ta**2).sum()/(1/ta**2).sum()))
	date2=np.array(date2)
	sec2=np.array(sec2)
	dt2=np.array(dt2)
	dt2err=np.array(dt2err)
	freq2=np.array(freq2)
	dm2=np.array(dm2)
	dmerr2=np.array(dmerr2)
	jj2=np.array(jj2)
	time2=te.times(te.time(date2,sec2,scale=info['telescope_info']['telename']))
	psrt2=pm.psr_timing(psr,time2,np.inf)
	if not merge_mark:
		return [dt2,dt2err,freq2,dm2,dmerr2,jj2,np.ones_like(jj2),psr,psrt2]
	else:
		return [dt2,dt2err,freq2,dm2,dmerr2,jj2,mfit[6],psr,psrt2]
#
def psrfit(paras):	# fit the ToAs with pulsar parameters
	dt,dterr,freq,dm,dmerr,psr,psrt,fitj=copy.deepcopy(fit_list[-1])
	jj=jj_list[jj_list[0]][1].copy()
	se=select_list[select_list[0]][1].copy()
	jump=jump_list[jump_list[0]][1].copy()
	dt+=jump
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
		psrt1=pm.psr_timing(psr1,time,np.inf)
		dphase=psrt1.phase.minus(psrt.phase)
		return dphase.phase+para[-1]
	#
	def dfunc(para):
		psr1=psr.copy()
		for i in np.arange(lpara):
			psr1.modify(paras[i],para[i])
		psrt1=pm.psr_timing(psr1,time,np.inf)
		tmp=psrt1.phase_der_para(paras).T
		return (np.concatenate((tmp,np.ones([time.size,1])),axis=1)/dterr.reshape(-1,1))[jj0]
	#
	def resi(para):
		return ((fit(para)+dt)/dterr)[jj0]
	#
	a=so.leastsq(resi,x0=x0,full_output=True,Dfun=dfunc)
	popt,pcov=a[0],(np.diag(a[1])*(resi(a[0])**2).sum()/(jj0.sum()-len(x0)))**0.5
	psr1=psr.copy()
	for i in np.arange(lpara):
		psr1.modify(paras[i],popt[i])
	psrt2=pm.psr_timing(psr1,time,np.inf)
	#print(fit(popt),dt,x0)
	fit_list.append([fit(popt)+dt-jump,dterr,freq,dm,dmerr,psr1,psrt2,jj0])
#
def plot():	# plot the figure with options
	global ax1,xax,yax,points,lines,xaxis,yaxis,yerr,point0,rect
	zoom_mark=False
	if merge_mark: 
		res,dterr,freq,dm,dmerr,jj,se,psr,psrt=copy.deepcopy(mfit)
	else:
		res,dterr,freq,dm,dmerr,psr,psrt,fitj=copy.deepcopy(fit_list[-1])
		se=select_list[select_list[0]][1].copy()
		jj=jj_list[jj_list[0]][1].copy()
		jump=jump_list[jump_list[0]][1].copy()
		res+=jump
	marker_colors=np.array(['b']*len(jj))
	if len(select_list)==2 or select_list[0]==1: marker_colors[np.logical_not(se)]='c'
	else: marker_colors[se]='c'
	marker_colors[np.logical_not(jj)]='r'
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
		xaxis=psrt.time.local.mjd
		xlabel='MJD (d)'
	elif xax=='year':
		xaxis=te.mjd2datetime(psrt.time.local.mjd)[4]
		xlim=0,366
		xlabel='Day in a Year (d)'
	#
	if len(xaxis)>len(jj):
		xaxtmp=np.zeros(mergej[-1])
		for i in np.arange(mergej[-1]):
			setj=mergej==(i+1)
			xaxtmp[i]=xaxis[setj].mean()
		xaxis=xaxtmp
	if jj.sum()==0: jlim=np.ones_like(jj,dtype=np.bool)
	else: jlim=jj
	if xax=='lst':
		xmax,xmin=np.max(xaxis[jlim]),np.min(xaxis[jlim])
		xlim=xmin*1.05-xmax*0.05,-xmin*0.05+xmax*1.05
	elif xax=='mjd':
		xlim=xaxis[jlim][0]*1.05-xaxis[jlim][-1]*0.05,-xaxis[jlim][0]*0.05+xaxis[jlim][-1]*1.05	
	#
	if dtunit=='time':
		yunit=' ($\\mu$s)'
	elif dtunit=='phase':
		yunit=''
	if yax=='prepost':
		ax1=fig.add_axes((x1,y2,x2-x1,y3-y2))
		ax2=fig.add_axes((x1,y1,x2-x1,y2-y1))
		if len(fit_list)>1:
			yaxis,yerr=fit_list[-2][0].copy(),fit_list[-2][1].copy()
			yaxis+=jump
		else: yaxis,yerr=res.copy(),dterr.copy()
		if dtunit=='time':
			period_tmp=period*1e6
			yaxis*=period_tmp
			yerr*=period_tmp
			res*=period_tmp
			dterr*=period_tmp
		lines2=ax2.errorbar(xaxis,yaxis,yerr,fmt='none').get_children()[0]
		points2=ax2.scatter(xaxis,yaxis,marker='.')
		lines=ax1.errorbar(xaxis,res,dterr,fmt='none').get_children()[0]
		points=ax1.scatter(xaxis,res,marker='.')
		points2.set_color(marker_colors)
		lines2.set_color(marker_colors)
		ax1.set_xlim(*xlim)
		ax1.set_xlabel(xlabel,fontsize=25,family='serif')
		ax2.set_ylabel('Phase Resi.'+yunit,fontsize=25,family='serif')
		ax1.set_ylabel('Fit Resi.'+yunit,fontsize=25,family='serif')
		ax2.set_xticks([])
		mark_text='Pre-fit and Post-fit'
	else:
		ax1=fig.add_axes((x1,y1,x2-x1,y3-y1))
		if yax=='post':
			yaxis,yerr=res,dterr
			if dtunit=='time':
				period_tmp=period*1e6
				yaxis*=period_tmp
				yerr*=period_tmp
			ax1.set_ylabel('Fit Resi.'+yunit,fontsize=25,family='serif')
			mark_text='Post-fit'
		if yax=='pre':
			if len(fit_list)>1:
				yaxis,yerr=fit_list[-2][0].copy(),fit_list[-2][1].copy()
				yaxis+=jump
			else: yaxis,yerr=res.copy(),dterr.copy()
			if dtunit=='time':
				period_tmp=period*1e6
				yaxis*=period_tmp
				yerr*=period_tmp
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
	ymax,ymin=np.max((yaxis+yerr)[jj]),np.min((yaxis-yerr)[jj])
	ylim=ymin*1.05-ymax*0.05,-ymin*0.05+ymax*1.05
	ax1.set_xlim(*xlim)
	ax1.set_ylim(ylim)
	point0,=ax1.plot(-10000,10000,'o',c='#00FF00',zorder=-1)
	fig.text(x2-0.05,y3-0.05,mark_text,fontsize=25,family='serif',color='green',va='top',ha='right')
	rect=mp.Rectangle((0,0),0,0,ec='k',fill=False,linestyle='--')
	ax1.add_patch(rect)
	canvas.draw()
#
def restart():	# restart
	global jj_list,select_list,jump_list,fit_list,dtunit,xax,yax,merge_mark,zoom_mark,action_list
	fit_list=[fit_list[0]]
	jj_list=jj_list[0:2]
	jj_list[0]=-1
	select_list=select_list[0:2]
	select_list[0]=-1
	jump_list=jump_list[0:2]
	jump_list[0]=-1
	combo1['value']=[combo1['value'][-1]]
	combo1.current(0)
	combo2['value']=[combo2['value'][-1]]
	combo2.current(0)
	combo3['value']=[combo3['value'][-1]]
	combo3.current(0)
	dtunit='phase'
	setxax('mjd')
	setyax('post')
	merge_mark,zoom_mark=False,False
	action_list=[-1]
	err_limit=1.0
	errentry.delete(0,uintsize)
	errentry.insert(0,str(err_limit))
	frame0.focus()
	plot()
#
def select(cur_x0,cur_x,cur_y0,cur_y,reset=False):	# set the zoom in region
	jj=(xaxis>cur_x0)&(xaxis<cur_x)&(yaxis>cur_y0)&(yaxis<cur_y)
	if jj.sum()==0:
		if ind>=0: jj[ind]=True
		else: return
	if merge_mark:
		sem=mfit[6]
		if len(select_list)==2 or select_list[0]==1:
			if key_on in ['Shift_L','v']:
				change=np.zeros_like(sem)
				change[jj]=1
			else:
				change=np.ones_like(sem)
				change[jj]=0
		else:
			change=np.zeros_like(sem)
			if key_on in ['Shift_L','v']:
				change[jj&(sem==1)]=1
			else:
				change[jj&(sem==0)]=1
		sem[change]=np.logical_not(sem[change])
		sea=np.zeros(len(filenames),dtype=np.bool)
		for i in np.arange(uniqf.size):
			if not jj[i]: continue
			setj=(filenames==uniqf[i])
			sea[setj]=True
		jj=sea
	se=select_list[select_list[0]][1].copy()
	if len(select_list)==2 or select_list[0]==1:
		if key_on in ['Shift_L','v']:
			change=np.zeros_like(se)
			change[jj]=1
		else:
			change=np.ones_like(se)
			change[jj]=0
		sinfo='Cancel '
	else:
		change=np.zeros_like(se)
		if key_on in ['Shift_L','v']:
			change[jj&(se==1)]=1
			sinfo='Cancel '
		else:
			change[jj&(se==0)]=1
			sinfo='Add '
	if change.sum()==0: return
	sinfo+=str(change.sum())+' points'
	se[change]=np.logical_not(se[change])
	select_list.append([sinfo,se])
	if merge_mark: plot()
	else: update_fig()
	combo2['value']=list(map(lambda x,y:str(y)+' '+x[0],select_list[1:],np.arange(len(select_list)-1)))[::-1]
	combo2.current(0)
	select_list[0]=-1
#
def zoom(cur_x0,cur_x,cur_y0,cur_y,reset=False):
	ax1.set_xlim(cur_x0,cur_x)
	ax1.set_ylim(cur_y0,cur_y)
	zoom_mark=True
	canvas.draw()
#
def delete(cur_x0,cur_x,cur_y0,cur_y,reset=False):	# delete ToAs in set of ToA to be fitted
	global action_list
	jj0=(xaxis>cur_x0)&(xaxis<cur_x)&(yaxis>cur_y0)&(yaxis<cur_y)
	if jj0.sum()==0:
		if ind>=0: jj0[ind]=True
		else: return
	if merge_mark:
		jjm=mfit[5]
		jjm0=jjm.copy()
		change=np.zeros_like(jjm)
		if key_on in ['Shift_L','v']:
			change[jj0&(jjm==0)]=1
		else:
			change[jj0&(jjm==1)]=1
		jjm[change]=np.logical_not(jjm[change])
		jja=np.zeros(len(filenames),dtype=np.bool)
		for i in np.arange(uniqf.size):
			if not jj0[i]: continue
			setj=(filenames==uniqf[i])
			jja[setj]=True
		jj0=jja
	jj=jj_list[jj_list[0]][1].copy()
	change=np.zeros_like(jj)
	if key_on in ['Shift_L','v']:
		change[jj0&(jj==0)]=1
		jinfo='Cancel '
	else:
		change[jj0&(jj==1)]=1
		jinfo='Delete '
	if change.sum()==0: return
	jinfo+=str(change.sum())+' points'
	jj[change]=np.logical_not(jj[change])
	jj_list.append([jinfo,jj])
	if zoom_mark: update_fig()
	else: plot()
	if action_list[0]!=-1:
		action_list=action_list[:int(action_list[0]+1)]
		action_list[0]=-1
	if merge_mark: action_list.append(['add_jj',[jj_list[0],jjm0,jjm]])
	else: action_list.append(['add_jj',[jj_list[0]]])
	combo1['value']=list(map(lambda x,y:str(y)+' '+x[0],jj_list[1:],np.arange(len(jj_list)-1)))[::-1]
	combo1.current(0)
	jj_list[0]=-1
#
def fit(fit=True):	# implement the fitting process
	global action_list,mfit
	paras=np.array(paralist)[list(pbox.curselection())]
	if len(paras)==0: return
	jj=jj_list[jj_list[0]][1].copy()
	se=select_list[select_list[0]][1].copy()
	lpara=len(paras)
	jj0=jj&se
	if jj0.sum()<=(lpara+2):
		print(text.warning_ftoa)
		return
	psrfit(paras)
	if merge_mark:
		mfit=merge()
		res,dterr,freq,dm,dmerr,jj,se,psr,psrt=copy.deepcopy(mfit)
	else:
		res,dterr,freq,dm,dmerr,psr,psrt,fitj=copy.deepcopy(fit_list[-1])
		se=select_list[select_list[0]][1].copy()
		jj=jj_list[jj_list[0]][1].copy()
		jump=jump_list[jump_list[0]][1].copy()
		res+=jump
	rese=res[jj]/dterr[jj]
	phasestd=np.sqrt((rese**2).sum()/(1/dterr[jj]**2).sum())*period
	print(text.info_rms,phasestd,', chi-square/d.o.f.:',(rese**2).sum()/len(rese-1-len(paras)),'\n')
	plot()
	if action_list[0]!=-1:
		action_list=action_list[:int(action_list[0]+1)]
		action_list[0]=-1
	action_list.append(['fit',[paras]])
#
def click(event):	# handle the click event
	global cur_x0,cur_y0,mouse_on,rect
	if mouse_on: return
	xlim1,xlim2,ylim1,ylim2=*ax1.get_xlim(),*ax1.get_ylim()
	axx1,axy1,axx2,axy2=ax1.get_position().extents
	kx,ky=((event.x-fig.bbox.extents[0])/fig.bbox.extents[2]-axx1)/(axx1-axx2),((fig.bbox.extents[3]-event.y)/fig.bbox.extents[3]-axy1)/(axy1-axy2)
	if kx>0 or kx<-1 or ky>0 or ky<-1: return
	mouse_on=True
	cur_x0=kx*(xlim1-xlim2)+xlim1
	cur_y0=ky*(ylim1-ylim2)+ylim1
	rect.set_xy((cur_x0,cur_y0))
	canvas.draw()
#
def leftrelease(event):
	global mouse_on,cur_x0,cur_y0
	if not mouse_on: return
	mouse_on=False
	rect.set_height(0)
	rect.set_width(0)
	rect.set_xy((0,0))
	cur_x0,cur_x1=np.sort([cur_x0,cur_x])
	cur_y0,cur_y1=np.sort([cur_y0,cur_y])
	if key_on in ['Control_L','z']: zoom(cur_x0,cur_x1,cur_y0,cur_y1)
	else: select(cur_x0,cur_x1,cur_y0,cur_y1)
#
def rightrelease(event):
	global mouse_on,cur_x0,cur_y0
	if not mouse_on: return
	mouse_on=False
	cur_x0,cur_x1=np.sort([cur_x0,cur_x])
	cur_y0,cur_y1=np.sort([cur_y0,cur_y])
	delete(cur_x0,cur_x1,cur_y0,cur_y1)
	rect.set_height(0)
	rect.set_width(0)
	rect.set_xy((0,0))
#
def middleclick(event):
	global profwindow,profax,profcanvas,prof
	if ind<0: return
	if merge_mark:
		setj0=(filenames==uniqf[ind])
		jj=jj_list[jj_list[0]][1].copy()
		setj1=setj0&jj
		mjj=jj[setj0]
		profinfo=fileinfo[np.arange(nt)[setj1][0]]
		mprof=np.array(fileinfo)[setj1]
	else:
		profinfo=fileinfo[ind]
	profname=profinfo[0]
	if not os.path.isfile(profname):
		profname=os.path.join(dirname,os.path.split(profinfo[0])[1])
		if os.path.isfile(profname):
			print(text.warning_ndfti)
		else:
			profname=os.path.split(profinfo[0])[1]
			if os.path.isfile(profname):
				print(text.warning_ndfni)
			else:
				print(text.info_ndf)
				return
	ldfile=ld.ld(profname)
	finfo=ldfile.read_info()
	freq_start1,freq_end1=np.float64(finfo['data_info']['freq_start']),np.float64(finfo['data_info']['freq_end'])
	nchan1=int(finfo['data_info']['nchan'])
	nbin1=int(finfo['data_info']['nbin'])
	channel_width1=(freq_end1-freq_start1)/nchan1 #MHz
	chanstart,chanend=np.int16(np.round((np.array([freq_start[ind],freq_end[ind]])-freq_start1)/channel_width1))
	substart=profinfo[1]
	if len(profinfo)==3: subend=profinfo[2]
	elif merge_mark: subend=int(mprof[-1,1])+1
	else: subend=substart+1
	profdata=ldfile.chan_scrunch(np.arange(chanstart,chanend),substart,subend,pol=0).reshape(subend-substart,nbin1)
	if subend-substart>1:
		if merge_mark:
			profdata=profdata[mjj[substart:subend]].sum(0)
		else: profdata=profdata.sum(0)
	else: profdata=profdata[0]
	if not profwindow:
		proffig=Figure(figsize=(6,4.5),dpi=100)
		proffig.clf()
		profax=proffig.add_axes((0.18,0.16,0.77,0.8))
		prof=tk.Toplevel(root)
		prof.title('')
		prof.geometry('600x450+180+180')
		prof.configure(bg='white')
		prof.resizable(False,False)
		prof.attributes('-type','splash')
		prof.attributes('-topmost',1)
		profframe=tk.Frame(prof,bg='white')
		profframe.grid(row=0,column=0)
		profcanvas=FigureCanvasTkAgg(proffig,master=profframe)
		def close(event):
			global profwindow
			prof.destroy()
			profwindow=False
		prof.bind('<ButtonPress-1>',close)
		prof.bind('<Escape>',close)
		profcanvas.get_tk_widget().pack(side=tk.TOP,fill='both',expand=1)
		#prof.mainloop()
		profwindow=True
	profax.cla()
	profax.plot(np.arange(nbin1)/nbin1,profdata)
	profax.set_xlabel('Phase',fontsize=20)
	profax.set_ylabel('Flux (arbi.)',fontsize=20)
	profcanvas.draw()
	prof.focus_set()
#
def dist(x,y,dx,dy):
	d=((xaxis-x)/dx)**2+((yaxis-y)/dy)**2
	#print(xaxis,event)
	ind=np.argmin(d)
	if d[ind]<0.0001: return ind
	else: return -1
#
def move_tk(event):
	global cur_x,cur_y,rect,ind
	xlim1,xlim2,ylim1,ylim2=*ax1.get_xlim(),*ax1.get_ylim()
	axx1,axy1,axx2,axy2=ax1.get_position().extents
	cur_x=((event.x-fig.bbox.extents[0])/fig.bbox.extents[2]-axx1)*(xlim1-xlim2)/(axx1-axx2)+xlim1
	cur_y=((fig.bbox.extents[3]-event.y)/fig.bbox.extents[3]-axy1)*(ylim1-ylim2)/(axy1-axy2)+ylim1
	if yax=='post':
		ind=dist(cur_x,cur_y,np.abs(xlim2-xlim1)*0.75,np.abs(ylim2-ylim1))
		if ind>=0: point0.set_data([xaxis[ind]],[yaxis[ind]])
		else: point0.set_data([-10000],[10000])
	if mouse_on:
		rect.set_height(cur_y-cur_y0)
		rect.set_width(cur_x-cur_x0)
	canvas.draw()
#
def press(event):
	global key_on
	key_on=event.keysym
	if key_on in ['r','a','e','s','u','y','f','m','b','h','p','q','d','x']:
		keymotion(key_on)
#
def release(event):
	global key_on
	key_on=''
#
def add_jump(a):	# add or minus a one-period jump for the ToAs of the right side of the cursor
	global mfit,action_list
	if merge_mark: se=mfit[6]
	else: se=select_list[select_list[0]][1].copy()
	if a=='a': 
		jump=1
		jinfo='+1'
	else: 
		jump=-1
		jinfo='-1'
	if se.sum()>0 and se.sum()!=se.size:jump=se*jump
	else: jump=(xaxis>cur_x)*jump
	if merge_mark:
		time_jump=np.zeros(len(filenames),dtype=np.int64)
		for i in np.arange(uniqf.size):
			if not jump[i]: continue
			setj=(filenames==uniqf[i])
			time_jump[setj]=jump[i]
	else:
		time_jump=jump
	time_jump0=jump_list[jump_list[0]][1].copy()
	jinfo=str(int(np.abs(time_jump.sum())))+' points '+jinfo
	time_jump+=time_jump0
	jump_list.append([jinfo,time_jump])
	if merge_mark: mfit=merge()
	plot()
	if action_list[0]!=-1:
		action_list=action_list[:int(action_list[0]+1)]
		action_list[0]=-1
	action_list.append(['add_jump',[jump_list[0]]])
	combo3['value']=list(map(lambda x,y:str(y)+' '+x[0],jump_list[1:],np.arange(len(jump_list)-1)))[::-1]
	combo3.current(0)
	jump_list[0]=-1
#
def merge_act():
	global merge_mark,mfit,select_list
	if not merge_mark: mfit=merge()
	merge_mark=not merge_mark
	select_list=select_list[0:2]
	combo2['value']=[combo2['value'][-1]]
	combo2.current(0)
	plot()
#
def undo():
	global mfit,err_limit
	if len(action_list)+action_list[0]<=0: return
	action=action_list[action_list[0]]
	if action[0]=='add_jump':
		tmp=jump_list.pop()
		combo3['value']=list(map(lambda x,y:str(y)+' '+x[0],jump_list[1:],np.arange(len(jump_list)-1)))[::-1]
		combo3.current(len(jump_list)-1-action[1][0]%len(jump_list))
		jump_list[0]=action[1][0]
		if merge_mark: mfit=merge()
		action[1]=[tmp]
		plot()
	elif action[0]=='jump_combo':
		combo3.current(len(jump_list)-1-action[1][0]%len(jump_list))
		tmp=jump_list[0]
		jump_list[0]=action[1][0]
		action[1]=[tmp]
		if merge_mark: mfit=merge()
		if zoom_mark: update_fig()
		else: plot()
	elif action[0]=='add_jj':
		tmp=jj_list.pop()
		if merge_mark: mfit[5]=action[1][1]
		if zoom_mark: update_fig()
		else: plot()
		jj_list[0]=action[1][0]
		combo1['value']=list(map(lambda x,y:str(y)+' '+x[0],jj_list[1:],np.arange(len(jj_list)-1)))[::-1]
		combo1.current(len(jj_list)-1-action[1][0]%len(jj_list))
		if merge_mark: action[1]=[tmp,action[1][2]]
		else: action[1]=[tmp]
	elif action[0]=='jj_combo':
		combo1.current(len(jj_list)-1-action[1][0]%len(jj_list))
		tmp=jj_list[0]
		jj_list[0]=action[1][0]
		action[1]=[tmp]
		if merge_mark: mfit=merge()
		if zoom_mark: update_fig()
		else: plot()
	elif action[0]=='submit_err':
		tmp=err_limit
		tmp1=jj_list.pop()
		errentry.delete(0,uintsize)
		errentry.insert(0,str(action[1][0]))
		combo1['value']=list(map(lambda x,y:str(y)+' '+x[0],jj_list[1:],np.arange(len(jj_list)-1)))[::-1]
		combo1.current(len(jj_list)-1-action[1][1]%len(jj_list))
		err_limit=action[1][0]
		jj_list[0]=action[1][1]
		action[1]=[tmp,tmp1]
		frame0.focus()
		if merge_mark: mfit=merge()
		if zoom_mark: update_fig()
		else: plot()
	elif action[0]=='merge':
		merge_act()
	elif action[0]=='fit':
		tmp=fit_list.pop()
		if merge_mark: mfit=merge()
		for i in np.arange(len(paralist)):
			pbox.selection_clear(i)
			if paralist[i] in action[1][0]:
				pbox.selection_set(i)
		action[1]=[tmp]
		plot()
	action_list[0]-=1
#
def redo():
	global mfit,err_limit
	if action_list[0]==-1: return
	action=action_list[int(action_list[0]+1)]
	if action[0]=='add_jump':
		jump_list.append(action[1][0])
		if merge_mark: mfit=merge()
		plot()
		combo3['value']=list(map(lambda x,y:str(y)+' '+x[0],jump_list[1:],np.arange(len(jump_list)-1)))[::-1]
		combo3.current(0)
		jump_list[0]=-1
	elif action[0]=='jump_combo':
		combo3.current(len(jump_list)-1-action[1][0]%len(jump_list))
		jump_list[0]=action[1][0]
		if merge_mark: mfit=merge()
		if zoom_mark: update_fig()
		else: plot()
	elif action[0]=='add_jj':
		jj_list.append(action[1][0])
		if merge_mark: mfit[5]=action[1][1]
		if zoom_mark: update_fig()
		else: plot()
		combo1['value']=list(map(lambda x,y:str(y)+' '+x[0],jj_list[1:],np.arange(len(jj_list)-1)))[::-1]
		combo1.current(0)
		jj_list[0]=-1
	elif action[0]=='jj_combo':
		combo1.current(len(jj_list)-1-action[1][0]%len(jj_list))
		jj_list[0]=action[1][0]
		if merge_mark: mfit=merge()
		if zoom_mark: update_fig()
		else: plot()
	elif action[0]=='submit_err':
		jj_list.append(action[1][1])
		errentry.delete(0,uintsize)
		errentry.insert(0,str(action[1][0]))
		combo1['value']=list(map(lambda x,y:str(y)+' '+x[0],jj_list[1:],np.arange(len(jj_list)-1)))[::-1]
		combo1.current(0)
		err_limit=action[1][0]
		jj_list[0]=-1	
		frame0.focus()
		if merge_mark: mfit=merge()
		if zoom_mark: update_fig()
		else: plot()
	elif action[0]=='merge':
		merge_act()
	elif action[0]=='fit':
		fit_list.append(action[1][0])
		if merge_mark: mfit=merge()
		plot()
	action_list[0]+=1
#
def keymotion(a):
	global select_list,jj_list,action_list,mfit
	if a=='r':	# reset the zoom in region
		plot()
	elif a=='u':	# undo the last delete
		undo()
	elif a=='y':	# undo the last delete
		redo()
	elif a=='f':	# fit
		fit()
	elif a=='a' or a=='d':	# add or minus a jump
		add_jump(a)
	elif a=='m':	# merge or unmerge the ToAs
		merge_act()
		if action_list[0]!=-1:
			action_list=action_list[:int(action_list[0]+1)]
			action_list[0]=-1
		action_list.append(['merge'])
	elif a=='e':
		select_list=select_list[0:2]
		select_list[0]=-1
		if merge_mark: mfit[6]=np.ones(len(mfit[6]))
		combo2['value']=[combo2['value'][-1]]
		combo2.current(0)
		plot()
	elif a=='b':	# restart the fitting process
		restart()
	elif a=='p':	# print present selected pulsar parameters
		psr=fit_list[-1][-3]
		paras=np.array(paralist)[list(pbox.curselection())]
		print(text.info_pp)
		psr.output(paras)
	elif a=='q':	# quit
		root.destroy()
	elif a=='s':	# save the pulsar parameters to file
		psr=fit_list[-1][-3]
		psrfile=tk.filedialog.asksaveasfilename(defaultextension='.par')
		if psrfile:
			psr.writepar(psrfile)
	elif a=='x':	# save the timing residuals to file
		resfile=tk.filedialog.asksaveasfilename(defaultextension='.res')
		if resfile:
			tmp=np.array([time.local.date,time.local.second,dt,dterr,fit_list[-1][-1]]).T
			np.savetxt(resfile,tmp,fmt=['%i', '%5.11f', '%.16f', '%.16f', '%i'])
			#psr.writepar(resfile[:-4]+'.par')
	elif a=='h':
		print(text.info_help.replace('\\n','\n'))
	return
#
def xmode(mode):	# choose the x-axis mode
	global xax,select_list
	if mode==xax: return
	if mode=='orbit' and 'binary' not in psr0.paras:
		print(text.warning_nb)
		return
	setxax(mode)
	select_list=select_list[0:2]
	plot()
#
def ymode(mode):	# choose the y-axis mode
	global select_list,dtunit
	if mode==yax: return
	select_list=select_list[0:2]
	if mode=='time':
		if dtunit=='time': dtunit,tmp='phase','Y-Unit\nPhase'
		else: dtunit,tmp='time','Y-Unit\nTime'
		y5bttn.config(text=tmp)
		plot()
	else:
		setyax(mode)
		plot()
#
def setyax(mode):
	global yax
	yaxlist=['pre','post','prepost','dm']
	ind=yaxlist.index(mode)
	yax=mode
	y1bttn.config(bg='white')
	y2bttn.config(bg='white')
	y3bttn.config(bg='white')
	y4bttn.config(bg='white')
	exec('y'+str(ind+1)+'bttn.config(bg=\'#E5E35B\')')
#
def setxax(mode):
	global xax
	yaxlist=['mjd','orbit','lst','year']
	ind=yaxlist.index(mode)
	xax=mode
	x1bttn.config(bg='white')
	x2bttn.config(bg='white')
	x3bttn.config(bg='white')
	x4bttn.config(bg='white')
	exec('x'+str(ind+1)+'bttn.config(bg=\'#E5E35B\')')
#
def submit_err(event):	# screen the large noise ToAs
	global err_limit,mfit,action_list
	err=errentry.get()
	try:
		err_limit0=np.float64(err)
	except:
		tk.messagebox.showwarning('Error!',text.tk_niel)
		return
	jj=jj_list[jj_list[0]][1]&(dterr<err_limit0)
	if np.any(jj_list[jj_list[0]][1]!=jj): jj_list.append(['Error_limit',jj])
	errentry.delete(0,uintsize)
	errentry.insert(0,str(err_limit0))
	combo1['value']=list(map(lambda x,y:str(y)+' '+x[0],jj_list[1:],np.arange(len(jj_list)-1)))[::-1]
	combo1.current(0)
	if action_list[0]!=-1:
		action_list=action_list[:int(action_list[0]+1)]
		action_list[0]=-1
	action_list.append(['submit_err',[err_limit,jj_list[0]]])
	err_limit=err_limit0
	jj_list[0]=-1	
	frame0.focus()
	if merge_mark:
		mfit=merge()
		#print(mfit[5])
	if zoom_mark: update_fig()
	else: plot()
#
def update_fig():
	if merge_mark: jj,se=mfit[5:7]
	else:
		se=select_list[select_list[0]][1].copy()
		jj=jj_list[jj_list[0]][1].copy()
	marker_colors=np.array(['b']*len(jj))
	if len(select_list)==2 or select_list[0]==1: marker_colors[np.logical_not(se)]='c'
	else: marker_colors[se]='c'
	marker_colors[np.logical_not(jj)]='r'
	lines.set_color(marker_colors)
	points.set_color(marker_colors)
	canvas.draw()
#
def jjcombo(event):
	global mouse_on,mfit,action_list
	value=list(combo1['value'])
	index=value.index(combo1.get())
	combo1.current(index)
	if action_list[0]!=-1:
		action_list=action_list[:int(action_list[0]+1)]
		action_list[0]=-1
	action_list.append(['jj_combo',[jj_list[0]]])
	jj_list[0]=len(value)-index
	mouse_on=False
	if merge_mark: mfit=merge()
	if zoom_mark: update_fig()
	else: plot()
#
def selcombo(event):
	global mouse_on,mfit
	value=list(combo2['value'])
	index=value.index(combo2.get())
	combo2.current(index)
	select_list[0]=len(value)-index
	mouse_on=False
	if merge_mark: mfit=merge()
	update_fig()
#
def jumpcombo(widget):
	global mouse_on,mfit,action_list
	value=list(combo3['value'])
	index=value.index(combo3.get())
	combo3.current(index)
	if action_list[0]!=-1:
		action_list=action_list[:int(action_list[0]+1)]
		action_list[0]=-1
	action_list.append(['jump_combo',[jump_list[0]]])
	jump_list[0]=len(value)-index
	mouse_on=False
	if merge_mark: mfit=merge()
	if zoom_mark: update_fig()
	else: plot()
#
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import matplotlib as mpl
mpl.use('TkAgg')
root=tk.Tk()
root.title('Timing of PSR '+psr0.name)
root.geometry('950x750+100+100')
root.configure(bg='white')
root.resizable(False,False)
frame0=tk.Frame(root,bg='white')
frame0.grid(row=1,column=0)
canvas=FigureCanvasTkAgg(fig,master=frame0)
canvas.get_tk_widget().grid()  
canvas.get_tk_widget().pack(fill='y')
root.bind('<KeyPress>',press)
root.bind('<KeyRelease>',release)
root.bind('<ButtonPress-1>',click)
root.bind('<ButtonPress-3>',click)
root.bind('<ButtonPress-2>',middleclick)
root.bind('<ButtonRelease-1>',leftrelease)
root.bind('<ButtonRelease-3>',rightrelease)
root.bind('<Motion>',move_tk)
sframe=tk.Frame(root,bg='white')
sframe.grid(row=1,column=1,rowspan=2)
#
font0=tk.font.Font(root,font=('serif',20))
font0width=font0.measure('0')
font0height=font0.metrics('linespace')
font1=tk.font.Font(root,font=('serif',17))
font1width=font1.measure('0')
font1height=font1.metrics('linespace')
combowidth=int((950-2-10-(8*font0width+2)*3-(13+2)*3)/3//font0width)
listboxheight=int((750-2-10-4*(2+font0height)-2)//(font1height+1))
listboxwidth=int((950-2-10-800-13-2)//font1width)
#
tk.Label(sframe,text='PARA List:',bg='white',font=('serif',20)).grid(row=0,column=0)
pboxframe=tk.Frame(sframe,bg='white')
pboxframe.grid(row=1,column=0)
pbox=tk.Listbox(pboxframe,height=listboxheight,width=listboxwidth,selectmode=tk.MULTIPLE,font=('serif',17),exportselection=0)
paralist=[]
for i in paralist0:
	if i in psr0.paras: paralist.append(i)
if 'fb0' in paralist and 'pb' in paralist: paralist.remove('pb')
pbox.insert(0,*map(lambda x:x.upper(),paralist))
sbar1= tk.Scrollbar(pboxframe,command=pbox.yview)
sbar1.pack(side=tk.RIGHT,fill='y')
pbox.pack(fill='both',anchor='w',expand='no')
pbox.config(yscrollcommand = sbar1.set)
tk.Label(sframe,text='Err limit:',bg='white',font=('serif',20)).grid(row=2,column=0)
errentry=tk.Entry(sframe,width=listboxwidth,font=('serif',17))
errentry.grid(row=3,column=0)
errentry.insert(0,str(err_limit))
errentry.bind('<Return>',submit_err)
pframe=tk.Frame(root,bg='white')
pframe.grid(row=2,column=0)
tk.Label(pframe,text='Plot mode: ',bg='white',font=('serif',20)).grid(row=0,column=0)
y1bttn=tk.Button(pframe,text='Pre-fit',command=lambda: ymode('pre'),bg='white',activebackground='#E5E35B',font=('serif',20))
y1bttn.grid(row=0,column=1)
y1bttn.config(width=5)
y2bttn=tk.Button(pframe,text='Post-fit',command=lambda: ymode('post'),bg='white',activebackground='#E5E35B',font=('serif',20))
y2bttn.grid(row=0,column=2)
y2bttn.config(width=6)
y3bttn=tk.Button(pframe,text='Pre&Post',command=lambda: ymode('prepost'),bg='white',activebackground='#E5E35B',font=('serif',20))
y3bttn.grid(row=0,column=3)
y3bttn.config(width=7)
y4bttn=tk.Button(pframe,text='DM',command=lambda: ymode('dm'),bg='white',activebackground='#E5E35B',font=('serif',20))
y4bttn.grid(row=0,column=4)
y4bttn.config(width=3)
tk.Label(pframe,width=1,text=' ',bg='white',font=('serif',20)).grid(row=0,column=5,rowspan=2)
y5bttn=tk.Button(pframe,text='Y-Unit\nPhase',command=lambda: ymode('time'),bg='white',activebackground='#E5E35B',font=('serif',20))
y5bttn.grid(row=0,column=6,rowspan=2)
y5bttn.config(width=5)
tk.Label(pframe,text='X-axis ',bg='white',font=('serif',20)).grid(row=1,column=0)
x1bttn=tk.Button(pframe,text='MJD',command=lambda: xmode('mjd'),bg='white',activebackground='#E5E35B',font=('serif',20))
x1bttn.grid(row=1,column=1)
x1bttn.config(width=5)
x2bttn=tk.Button(pframe,text='Orbit',command=lambda: xmode('orbit'),bg='white',activebackground='#E5E35B',font=('serif',20))
x2bttn.grid(row=1,column=2)
x2bttn.config(width=6)
x3bttn=tk.Button(pframe,text='LST',command=lambda: xmode('lst'),bg='white',activebackground='#E5E35B',font=('serif',20))
x3bttn.grid(row=1,column=3)
x3bttn.config(width=7)
x4bttn=tk.Button(pframe,text='Year',command=lambda: xmode('year'),bg='white',activebackground='#E5E35B',font=('serif',20))
x4bttn.grid(row=1,column=4)
x4bttn.config(width=3)
pframe1=tk.Frame(root,bg='white')
pframe1.grid(row=0,column=0,columnspan=2)
tk.Label(pframe1,width=8,text='Delete:',bg='white',font=('serif',20)).grid(row=0,column=0)
combo1=ttk.Combobox(pframe1,width=combowidth,font=('serif',20),state="readonly")
combo1['value']=list(map(lambda x,y:str(y)+' '+x[0],jj_list[1:],np.arange(len(jj_list)-1)))[::-1]
combo1.current(0)
combo1.grid(row=0,column=1)
combo1.bind("<<ComboboxSelected>>",jjcombo)
tk.Label(pframe1,width=8,text='Select:',bg='white',font=('serif',20)).grid(row=0,column=2)
combo2=ttk.Combobox(pframe1,width=combowidth,font=('serif',20),state="readonly")
combo2['value']=list(map(lambda x,y:str(y)+' '+x[0],select_list[1:],np.arange(len(select_list)-1)))[::-1]
combo2.current(0)
combo2.grid(row=0,column=3)
combo2.bind("<<ComboboxSelected>>",selcombo)
tk.Label(pframe1,width=8,text='Jump:',bg='white',font=('serif',20)).grid(row=0,column=4)
combo3=ttk.Combobox(pframe1,width=combowidth,font=('serif',20),state="readonly")
combo3['value']=list(map(lambda x,y:str(y)+' '+x[0],jump_list[1:],np.arange(len(jump_list)-1)))[::-1]
combo3.current(0)
combo3.grid(row=0,column=5)
combo3.bind("<<ComboboxSelected>>",jumpcombo)
#
setxax('mjd')
setyax('post')
plot()
canvas.draw()
#
root.mainloop()
#
