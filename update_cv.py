#!/usr/bin/env python
import urllib.request as ur
import time_eph as rd
import numpy as np
import os,time,re
import argparse as ap
#import matplotlib.pyplot as plt
#plt.rcParams['font.family']='Serif'
#
version='JigLu_20220317'
#
parser=ap.ArgumentParser(prog='update_tdiff',description='Updating the time difference { (1) gps2utc, (2) leap, (3) polarmotion, (4) tai2ut1, (5) tai2tt }.',epilog='Ver '+version)
parser.add_argument('-v','--version', action='version', version=version)
parser.add_argument('--verbose', action="store_true",default=False,help="print detailed information")
parser.add_argument('-u','--update',dest='update',default='',help='update the specified time difference.')
parser.add_argument('-s','--show',dest='show',default='',help='show the specified time difference, contain { (6) leaptxt, (7) utc2ut1 }.')
args=(parser.parse_args())
#
dirname=os.path.split(os.path.realpath(__file__))[0]
#
def update_leap():
	# utc2tai, leap second
	a=ur.urlopen('https://hpiers.obspm.fr/eop-pc/earthor/utc/UTC-offsets_tab.html').readlines()
	b=list(map(lambda x:x.strip().decode(),a))
	lenb=len(b)
	for i in range(lenb):
		num=lenb-1-i
		t=b[num]
		if t:
			if not t[0:4].isdigit(): 
				b.pop(num)
			elif int(t[0:4])<1972:
				b.pop(num)
			else:
				b[num]=t.replace('Jan.','1').replace('Jul.','7').replace('1s','').strip()+'1\n'
				if b[num].split()[:2]==['1972','1']:
					b.pop(num)
		else:
			b.pop(num)
	f=open(dirname+'/conventions/'+'leap.txt','w')
	f.writelines(b)
	f.close()
#
def show_leaptxt():
	f=open(dirname+'/conventions/'+'leap.txt')
	s=f.read()
	f.close()
	print(s)
#
def show_leap(ax):
	f=np.loadtxt(dirname+'/conventions/'+'leap.txt')
	leap_time=np.array(list(map(lambda x:rd.datetime2mjd([x[0],x[1],x[2],0,0,0]).mjd,f))).reshape(-1)
	today=rd.datetime2mjd(np.int32(time.strftime('%Y %m %d 0 0 0',time.localtime()).split())).mjd
	leap_time=np.concatenate(([37700],leap_time.repeat(2),today))
	leap_sec=f[:,3].cumsum()
	leap_sec=-np.append([0,0],leap_sec.repeat(2))+10
	ax.plot(leap_time,leap_sec)
	return 'MJD','TAI-UTC (s)'
#
def update_polarmotion():
	# utc2ut1
	a=ur.urlopen('https://hpiers.obspm.fr/eoppc/eop/eopc04/eopc04.62-now').readlines()
	b=list(map(lambda x:x.strip().decode(),a))
	lenb=len(b)
	for i in range(lenb):
		num=lenb-1-i
		t=b[num]
		if t:
			if not t[0:4].isdigit(): 
				b.pop(num)
			else:
				b[num]=t+'\n'
		else:
			b.pop(num)
	f=open(dirname+'/conventions/'+'eopc.txt','w')
	f.writelines(b)
	f.close()
#
def show_polarmotion(ax):
	mjd,dx,dy=np.loadtxt(dirname+'/conventions/'+'eopc.txt')[:,3:6].T
	ax.plot(mjd,dx,label='dX')
	ax.plot(mjd,dy,label='dY')
	ax.legend(fontsize=20)
	return 'MJD','Polarmotion (\'\')'
#
def show_utc2ut1(ax):
	mjd,dt=np.loadtxt(dirname+'/conventions/'+'eopc.txt')[:,[3,6]].T
	ax.plot(mjd,dt)
	return 'MJD','UT1-UTC (s)'
#
def show_pmxy(ax):
	dx,dy=np.loadtxt(dirname+'/conventions/'+'eopc.txt')[:,4:6].T
	ax.plot(dx,dy)
	return 'PMDX (\'\')','PMDY (\'\')'
#
def update_tai2ut1():
	# tai2ut1
	a=ur.urlopen('https://hpiers.obspm.fr/eoppc/eop/eopc04/eopc04.62-now').readlines()
	b=list(map(lambda x:x.strip().decode(),a))
	lenb=len(b)
	for i in range(lenb):
		num=lenb-1-i
		t=b[num]
		if t:
			if not t[0:4].isdigit(): 
				b.pop(num)
			else:
				b[num]=t+'\n'
		else:
			b.pop(num)
	mjd,deltat=np.float64(list(map(lambda x:x.split(),b)))[:,[3,6]].T
	taimjd=(rd.time(mjd,np.zeros_like(mjd),scale='utc')).tai().mjd
	f=np.loadtxt(dirname+'/conventions/'+'leap.txt')
	leap_time0=np.array(list(map(lambda x:rd.datetime2mjd([x[0],x[1],x[2],0,0,0]).mjd,f))).reshape(-1)
	leap_time=np.array(list(map(lambda x:f[leap_time0<=x,3].sum(),mjd)))
	deltat+=leap_time-10
	f=open(dirname+'/conventions/'+'tai2ut1.txt','w')
	f.writelines(list(map(lambda x: str(x[0])+' '+str(x[1])+' '+str(x[2])+'\n',zip(mjd,taimjd,deltat))))
	f.close()
#
def show_tai2ut1(ax):
	mjd,dut1=np.loadtxt(dirname+'/conventions/'+'tai2ut1.txt')[:,[0,2]].T
	ax.plot(mjd,dut1)
	return 'MJD','UT1-TAI (s)'
#
def update_tai2tt():
	# tai2tt
	a=ur.urlopen('ftp://ftp2.bipm.org/pub/tai/ttbipm/').readlines()
	def year(x):
		tmp=re.compile(r'TTBIPM.(\d)(\d)(\d)(\d)').match(x.decode().split()[8])
		if tmp: return int(tmp.group()[7:11])
		else: return 0
	b=str(np.max(np.array(list(map(year,a)))))
	a=ur.urlopen('ftp://ftp2.bipm.org/pub/tai/ttbipm/TTBIPM.'+b).readlines()
	b=list(map(lambda x:x.decode(),a))
	lenb=len(b)
	for i in range(lenb):
		num=lenb-1-i
		t=b[num]
		if t:
			if not t[0:5].isdigit(): 
				b.pop(num)
		else:
			b.pop(num)
	f=open(dirname+'/conventions/'+'tai2tt.txt','w')
	f.writelines(b)
	f.close()
#
def show_tai2tt(ax):
	mjd0,deltat=np.loadtxt(dirname+'/conventions/'+'tai2tt.txt')[:,[0,2]].T
	ax.plot(mjd0,deltat)
	return 'MJD','TT-TAI-32.184s ($\mu$s)'
#
def update_gps2utc():
	# gps2utc
	a=ur.urlopen('ftp://ftp2.bipm.org/pub/tai/other-products/utcgnss/utc-gnss').readlines()[20:]
	b=list(map(lambda x:x.decode().strip(),a))
	lenb=len(b)
	for i in range(lenb):
		num=lenb-1-i
		t=b[num]
		if t:
			if not t[0:5].isdigit(): 
				b.pop(num)
			else:
				b[num]=t[:23]+'\n'
		else:
			b.pop(num)
	f=open(dirname+'/conventions/'+'gps2utc.txt','w')
	f.writelines(b)
	f.close()
#
def show_gps2utc(ax):
	f=np.loadtxt(dirname+'/conventions/'+'gps2utc.txt')
	mjd,dt=f[:,0:2].T
	ax.plot(mjd,dt)
	return 'MJD','UTC-GPS (ns)'
#
slist=['1','2','3','4','5','utc2tai']
tlist=['gps2utc','leap','polarmotion','tai2ut1','tai2tt','leap']
sdict=dict(np.array([slist,tlist]).T)
if args.update:
	if args.update in slist:
		update_select=sdict[args.update]
	elif args.update in tlist:
		update_select=args.update
	else:
		parser.error("The selected clock difference to be updated cannot be recognized.")
else:
	update_select=''
#
slist.extend(['6','7','8'])
tlist.extend(['leaptxt','utc2ut1','pmxy'])
sdict=dict(np.array([slist,tlist]).T)
if args.show:
	if args.show in slist:
		show_select=sdict[args.show]
	elif args.show in tlist:
		show_select=args.show
	else:
		parser.error("The selected clock difference to be shown cannot be recognized.")
else:
	show_select=''
#
if update_select:
	eval('update_'+update_select+'()')
elif not show_select:
	slist2=['gps2utc','leap','polarmotion','tai2ut1','tai2tt']
	for i in slist2:
		eval('update_'+i+'()')
#
if show_select:
	if show_select=='leaptxt':
		show_leaptxt()
	else:
		fig=plt.figure(1)
		ax=fig.add_axes([0.13,0.14,0.82,0.81])
		xlabel,ylabel=eval('show_'+show_select+'(ax)')
		ax.set_xlabel(xlabel,fontsize=20)
		ax.set_ylabel(ylabel,fontsize=20)
		plt.show()
		

