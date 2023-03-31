import numpy as np
import time_eph as te
import psr_read as pr
import psr_model as pm
import ld
import _io
#
# ld.py
ld={
	'__size__':[np.ndarray,[5],int],	# the size of the LD file
	'file':_io.TextIOWrapper,	# the file handle of LD file
	'name':str,	# the name of the LD file
	#
	'__init__()':(dict(name=str),
		[]),
	'__read_size__()':(dict(),		# read the size of the LD file, containing the file size and the shape along the 4 dimensions of the data
		[[np.ndarray,[5],int]]),
	'__refresh_size__()':(dict(),		# refresh the file size value in the LD file
		[]),
	'__read_bin_segment__()':(dict(bin_start=int,bin_num=int),	# read data segment with specified starting bin index and total bin numbers in all frequency channels
		[[np.ndarray,['nchan','nbin','npol'],float]]),
	'__read_chan0__()':(dict(chan_num=int,ndata_chan0=int),	# discarded
		[[np.ndarray,['ndata'],float]]),
	'__write_bin_segment__()':(dict(data=[np.ndarray,['nchan','ndata_per_channel'],float],bin_start=int),	# write the data segment with all frequency channels into LD file at the specified starting bin index
		[]),
	'__write_chanbins_add__()':(dict(data=[np.ndarray,['nbin','npol'],float],bin_start=int,chan_num=int),	# add the data series onto specific frequency channel of LD file at specified starting bin index
		[]),
	'__write_chanbins__()':(dict(data=[np.ndarray,['nbin','npol'],float],bin_start=int,chan_num=int),		# write the data series into specific frequency channel of LD file at specified starting bin index
		[]),
	'__write_size__()':(dict(size=[np.ndarray,[5],int]),		# write the size array into LD file, containing the file size and the shape along the 4 dimensions of the data
		[]),
	'chan_scrunch()':(dict(select_chan=[list,['number_of_selected_channels'],int],start_period=int,end_period=int,chan_weighted=bool),	# scrunch the data in LD file along frequency axis
		[[np.ndarray,['nsub','nbin','npol'],float]]),		# scrunched data
	'period_scrunch()':(dict(start_period=int,end_period=int,select_chan=[list,['number_of_selected_channels'],int]),	# scrunch the data in LD file along subint axis
		[[np.ndarray,['nchan','nbin','npol'],float]]),	# scrunched data
	'read_chan()':(dict(chan_num=int),	# read the data in specific channel index in LD file
		[[np.ndarray,['nsub','nbin','npol'],float]]),
	'read_data()':(dict(),		# read all data in LD file
		[[np.ndarray,['nchan','nsub','nbin','npol'],float]]),
	'read_info()':(dict(),		# read the information of LD file
		[dict]),
	'read_para()':(dict(key=str),		# read the value of the specified key in the information of LD file
		['alterable']),
	'read_period()':(dict(p_num=int),		# read the data in specific sub-integration index in LD file
		[[np.ndarray,['nchan','nbin','npol'],float]]),
	'read_shape()':(dict(),		# read the shape along the 4 dimensions of the data
		[[np.ndarray,[4],int]]),
	'write_chan()':(dict(data=[np.ndarray,['ndata_per_channel'],float],chan_num=int),	# write data into specific channel index in LD file
		[]),
	'write_info()':(dict(info=dict),		# write the information dictionary into the LD file
		[]),
	'write_para()':(dict(key=str,value='alterable'),		# write or modify the value of a specific key into the information of LD file
		[]),
	'write_period()':(dict(data=[np.ndarray,['ndata_per_subint'],float],p_num=int),	# write data into specific sub-integration index in LD file
		[]),
	'write_shape()':(dict(shape=[np.ndarray,[4],int]),		# write the data shape into the LD file
		[])
}
#
# ld.py
ld_file_info={
	'best_dm':[list,[2],float],		# the best DM value and its error determined with the program lddm.py
	'cal':[list,['1 or 2',4,'nchan'],float],		# the calibration array used to calibrate the data
	'cal_mode':str,			# the calibration mode, 'single' or 'seg' or 'trend'
	'chan_weight':[list,['nchan'],float],	# the weight of each frequency channel
	'compressed':bool,			# whether the data is compressed
	'dm':float,				# the DM value of the pulsar which used in the preliminary dedispersion
	'file_time':[list,['number_of_processing'],str],	# the file creation time
	'freq_end':float,			# the upper bound of the data frequency
	'freq_end_origin':float,		# the frequency upper bound of the observed data
	'freq_start':float,			# the lower bound of the data frequency
	'freq_start_origin':float,		# the frequency lower bound of the observed data
	'history':[list,['number_of_processing'],str],	# the commands used to obtain the current data from the original data
	'krange':[list,[2],float],		#  
	'length':float,			# the duration of the data
	'mode':str,				# the mode of the data, 'single' or 'subint' or 'cal' or 'template' or 'ToA' or 'test'
	'nbin':int,				# the bin number of the data
	'nbin_origin':int,			# the bin number of the observed data
	'nchan':int,				# the number of the frequency channels
	'nchan_origin':int,			# the number of the frequency channels in the observed data
	'noise_time0':float,			# the zero time of the noise diode data
	'nperiod':int,				# the number of the pulse periods in the data
	'npol':int,				# the number of the polarizations in the data
	'nsub':int,				# the number of the sub-integrations in the data
	'period':float,			# the actually observational pulse period at the observing time
	'phase0':int,				# the integer cycles for pulsar rotating from the PEPOCH0 to the observation time
	'pol_type':str,			# the polarization type, 'IQUV' or 'I' or 'AABB' or 'AABBCRCI'
	'predictor':[list,['ncoeff',2],float],		# the coefficients of Chebishev polynomials to predict the pulse phase
	'predictor_freq':[list,[5],float],			# the coefficients of Chebishev polynomials to predict the dispersion
	'psr_name':str,					# the pulsar name
	'psr_par':[list,['number_of_lines'],str],		# the parameters of pulsar
	'rm':[list,[2],float],			# the RM value and its error determined with the program ldrm.py
	'seg_time':[list,['number_of_noise_segments'],float],	# the recorded time of the noise diode data from 'noise_time0'
	'spec':[list,['nchan'],float],	# the spectra of the data
	'stt_date':int,			# the start date of the data
	'stt_sec':float,			# the start sec of the data at 'stt_data'
	'stt_time':float,			# the start time of the data
	'stt_time_origin':float,		# the start time of the observed data
	'sub_nperiod':int,			# the number of the pulse periods in one sub-integration
	'sub_nperiod_last':int,		# the number of the pulse periods in the last sub-integration
	'sublen':float,			# the duration of one sub-integration
	'telename':str,			# the name of the observing telescope
	'tsamp_origin':float			# the sample time of the observed data
}
#
# time_eph.py
#
functions_variables_time_eph={
	'au_dist':int,			# astronomical unit (m)
	'cios0':[list,[33],list],	# the 0th order of the precession term parameters
	'cios1':[list,[3],list],	# the 1th order of the precession term parameters
	'cios2':[list,[25],list],	# the 2th order of the precession term parameters
	'cios3':[list,[4],list],	# the 3th order of the precession term parameters
	'cios4':[list,[1],list],	# the 4th order of the precession term parameters
	'ciosp':[list,[6],float],	# the isotropic term of the precession parameters
	'dirname':str,			# the directory of the programs
	'ephname':str,			# the ephemeris file name
	'iftek':float,			# the linear coefficient for TCB over TDB
	'km1':float,			# the linear difference between TDB and TCB
	'lc':float,			# the average linear difference between TCB and TCG
	'lg':float,			# the linear difference between TT and TCG
	'mjd0':float,			# the bifurcation MJD time for TT and TCG
	'nutarray':[np.ndarray,[77,11],float],		# the nutation parameters
	'pi':float,		# PI
	'pi_mm':float,		# PI in mpmath
	'sl':float,		# speed of light
	'tdb0':float,		# the time difference between TDB and TCB at time 'mjd0'
	'time_scales':[list,[10],str],	# all time scales
	#
	'datetime2mjd':(dict(datetime=[np.ndarray,['ndatetime',6],float]),	# change datetime (year, month, day, hour, minute, and second) to be MJD time
		[[np.ndarray,['ndatetime'],float]]),
	'get_precessionMatrix':(dict(et0=[np.ndarray,['net0'],float],nut=[np.ndarray,['net0'],float]),	# calculate the precession matrix with UT1 time and nutation parameters
		[[np.ndarray,['net0',3,3],float]]),
	'lmst':(dict(mjd=[np.ndarray,['nmjd'],float],olong=float),		# calculate the local sidereal time with MJD time and longitude
		[[np.ndarray,['nmjd'],float]]),
	'mjd2datetime':(dict(mjd=[np.ndarray,['nmjd'],float]),		# change MJD time to be date time (year, month, day, second in one day, and day in one year)
		[[np.ndarray,['nmjd'],int],[np.ndarray,['nmjd'],int],[np.ndarray,['nmjd'],int],[np.ndarray,['nmjd'],float],[np.ndarray,['nmjd'],int]]),
	'multiply':(dict(a=[np.ndarray,['nvec',3],float],b=[np.ndarray,['nvec',3],float]),			# return the cross product of two groups of vectors
		[[np.ndarray,['nvec',3],float]]),
	'normalize':(dict(a=[np.ndarray,['nvec',3],float]),			# normalize the vectors
		[[np.ndarray,['nvec',3],float]]),
	'readeph':(dict(et=te.time,ephname=str),		# read the parameters from ephemeris
		[[np.ndarray,[13],te.vector],[np.ndarray,[13],te.vector],[np.ndarray,[13],te.vector],[np.ndarray,['net'],float],dict]),
	'rotx':(dict(phi=[np.ndarray,['nphi'],float],mat=[np.ndarray,[3,3],float]),		# rotate the matrix around x-axis
		[[np.ndarray,['nphi',3,3],float]]),
	'roty':(dict(theta=[np.ndarray,['ntheta'],float],mat=[np.ndarray,[3,3],float]),	# rotate the matrix around y-axis
		[[np.ndarray,['ntheta',3,3],float]]),
	'rotz':(dict(psi=[np.ndarray,['npsi'],float],mat=[np.ndarray,[3,3],float]),		# rotate the matrix around z-axis
		[[np.ndarray,['npsi',3,3],float]]),
}
#
# time_eph.py
vector={
	'center':str,	# center of the vector, 'geo' or 'bary'
	'coord':str,	# coordinates of the vector, 'equ' or 'ecl'
	'scale':str,	# 'si' or 'tdb' for 'center' being 'bary, or 'itrs' or 'grs80' for 'center' being 'geo'
	'size':int,	# the size of the vectors
	'type':str,	# position 'pos' or velocity 'vel' or acceleration 'acc'
	'unit':float,	# unit of the value, 1.0 or sl or au_dist
	'x':[np.ndarray,['size'],float],	# x-coordinate of the vector
	'y':[np.ndarray,['size'],float],	# y-coordinate of the vector
	'z':[np.ndarray,['size'],float],	# z-coordinate of the vector
	#
	'__eq__()':(dict(other=te.vector),	# determine whether the vector is equal to another vector
		[bool]),
	'__init__()':(dict(x=[np.ndarray,['nvec'],float],y=[np.ndarray,['nvec'],float],z=[np.ndarray,['nvec'],float],center=str,scale=str,coord=str,unit=float,type0=str),
		[]),
	'__repr__()':(dict(),			# return a string about this vector instance
		[str]),
	'__str__()':(dict(),			# return a string about this vector instance
		[str]),
	'add()':(dict(vec=te.vector),		# calculate the summation of the vector and another vector
		[te.vector]),
	'angle()':(dict(vec=te.vector),	# calculate the angle between the vector and another vector
		[[np.ndarray,['nvec'],float]]),
	'change_unit()':(dict(unit=float),	# change unit of the vector (e.g., from m to km)
		[]),
	'copy()':(dict(),			# produce a counterpart of this vector instance
		[te.vector]),
	'dot()':(dict(vec=te.vector),		# calculate the scalar product of the vector and another vector
		[[np.ndarray,['nvec'],float]]),
	'ecl2equ()':(dict(),			# transform the coordinates from ecliptic to equatorial
		[]),
	'equ2ecl()':(dict(),			# transform the coordinates from equatorial to ecliptic
		[]),
	'grs802itrs()':(dict(),		# transform the coordinates from GRS80 to ITRS
		[]),
	'itrs2grs80()':(dict(),		# transform the coordinates from ITRS to GRS80
		[]),
	'length()':(dict(),			# calculate the length of the vectors
		[[np.ndarray,['nvec'],float]]),	
	'minus()':(dict(vec=te.vector),	# calculate the difference between the vector and another vector
		[te.vector]),
	'multi()':(dict(factor=float,type0=str),	# scalar-multiplication of the vector and a scalar
		[te.vector]),
	'si2tdb()':(dict(),			# change scale from TCB to TDB
		[]),
	'tdb2si()':(dict(),			# change scale from TDB to TCB
		[]),
	'xyz()':(dict(),			# return the coodinates of the vectors
		[[np.ndarray,['nvec',3],float]])
}
#
# time_eph.py
time={
	'date':[np.ndarray,['size'],float],	# integer part of the MJD date
	'scale':str,	# scale of the time instance, such as shown in te.time_scales
	'second':[np.ndarray,['size'],float],	# seconds in one day
	'size':int,	# size of the time
	'mjd':[np.ndarray,['size'],float],	# MJD time
	'unit':float,	# the total seconds in one day
	#
	'__eq__()':(dict(other=te.time),	# determine whether the time is equal to another time
		[bool]),
	'__init__()':(dict(date=[np.ndarray,['size'],float],second=[np.ndarray,['size'],float],scale=str,unit=int),
		[]),
	'__repr__()':(dict(),			# return a string about this time instance
		[str]),
	'__str__()':(dict(),			# return a string about this time instance
		[str]),
	'add()':(dict(dt=te.time,scale=int),	# add a time interval onto this time instance
		[te.time]),
	'copy()':(dict(),			# produce a counterpart of this time instance
		[te.time]),
	'local2unix()':(dict(),		# return the corresponding UNIX timestamp from local MJD time
		[te.time]),
	'local2utc()':(dict(),			# calculate the difference between local MJD time and UTC
		[[np.ndarray,['size'],float]]),
	'minus()':(dict(time1=te.time),	# calculate the difference between this time instance and another time instance
		[te.time]),
	'tai()':(dict(),		# return the corresponding TAI
		[te.time]),
	'tai2tt()':(dict(),			# calculate the difference between TAI and TT(BIPM)
		[[np.ndarray,['size'],float]]),
	'tai2ut1()':(dict(),			# calculate the difference between TAI and UT1
		[[np.ndarray,['size'],float]]),
	'tai2utc()':(dict(),			# calculate the difference between TAI and UTC
		[[np.ndarray,['size'],float]]),
	'tcb()':(dict(),		# return the corresponding TCB
		[te.time]),
	'tcb2tdb()':(dict(),			# calculate the difference between TCB and TDB
		[[np.ndarray,['size'],float]]),
	'tdb()':(dict(),		# return the corresponding TDB
		[te.time]),
	'tdb2tcb()':(dict(),			# calculate the difference between TDB and TCB
		[[np.ndarray,['size'],float]]),
	'tt()':(dict(),		# return the corresponding TT(BIPM)
		[te.time]),
	'tt2tai()':(dict(),			# calculate the difference between TT(BIPM) and TAI
		[[np.ndarray,['size'],float]]),
	'unix2local()':(dict(scale=str),	# return the local MJD time from UNIX timestamp
		[te.time]),
	'update()':(dict(),			# update the value of the time instance
		[]),
	'ut1()':(dict(),		# return the corresponding UT1
		[te.time]),
	'utc()':(dict(),		# return the corresponding UTC
		[te.time]),
	'utc2tai()':(dict(),			# calculate the difference between UTC and TAI
		[[np.ndarray,['size'],float]]),
	'utc2tt()':(dict(),			# calculate the difference between UTC and TT(BIPM)
		[[np.ndarray,['size'],float]])
}
#
# time_eph.py
phase={
	'integer':[np.ndarray,['size'],float],	# integer part of the phase
	'offset':[np.ndarray,['size'],float],	# fractional part of the phase
	'phase':[np.ndarray,['size'],float],	# total phase
	'scale':str,	# scale of the phase instance, always be 'phase'
	'size':int,	# size of the phase
	#
	'__eq__()':(dict(other=te.phase),	# determine whether the time is equal to another time
		[bool]),
	'__init__()':(dict(integer=[np.ndarray,['size'],float],offset=[np.ndarray,['size'],float],scale=str),
		[]),
	'__repr__()':(dict(),			# return a string about this time instance
		[str]),
	'__str__()':(dict(),			# return a string about this time instance
		[str]),
	'add()':(dict(dt=te.phase,scale=int),	# add a time interval onto this time instance
		[te.phase]),
	'copy()':(dict(),			# produce a counterpart of this time instance
		[te.phase]),
	'minus()':(dict(phase1=te.phase),	# calculate the difference between this time instance and another time instance
		[te.phase]),
	'update()':(dict(),			# update the value of the time instance
		[])
}
#
# time_eph.py
times={
	'acc':[np.ndarray,[13],te.vector],	# celestial body acceleration relative to barycenter
	'cons':dict,		# the constants used to calculate the ephemeris
	'earthacc':te.vector,	# the acceleration of the Earth relative to barycenter
	'earthpos':te.vector,	# the position of the Earth relative to barycenter
	'earthvel':te.vector,	# the velocity of the Earth relative to barycenter
	'einsteinrate':[np.ndarray,['size'],float],	# the derivative of TCB to TT
	'ephem':str,		# ephemeris file name
	'ephver':int,		# ephemeris version, 2 for TEMPO, 5 for TEMPO2
	'local':te.time,	# local time
	'lst':[np.ndarray,['size'],float],	# local sidereal time
	'nut':[np.ndarray,['net'],float],	# the nutation of the Earth and its derivative
	'pos':[np.ndarray,[13],te.vector],	# celestial body position relative to barycenter
	'site_grs80':te.vector,		# the site coordinate in GRS80 coordinates
	'site_pos':te.vector,	# the position of the telescope relative to the center of the Earth
	'site_vel':te.vector,	# the velocity of the telescope relative to the center of the Earth
	'size':int,		# size of the times
	'tai':te.time,		# TAI time
	'tcb':te.time,		# TCB time
	'tdb':te.time,		# TDB time
	'tt':te.time,		# TT time
	'unix':te.time,	# UNIX timestamp
	'ut1':te.time,		# UT1 time
	'utc':te.time,		# UTC time
	'vel':[np.ndarray,[13],te.vector],	# celestial body velocity relative to barycenter
	'zenith':te.vector,	# the unit vector of the zenith direction 
	#
	'__eq__()':(dict(other=te.times),	# determine whether this times instance is equal to another times instance
		[]),
	'__init__()':(dict(time0=te.time,ephem=str,ephver=int),
		[]),
	'copy()':(dict(),			# produce a counterpart of this times instance
		[te.times]),
	'deltat_fb()':(dict(),			# the time difference between TT and TCB induced by gravitational and the Earth motion effects (Fairhead and Bretagnon, 1990)
		[[np.ndarray,['size'],float]]),
	'deltat_if()':(dict(),			# the time difference (and its derivative) between TT and TCB induced by gravitational and the Earth motion effects (Irwin and Fukushima, 1999)
		[[np.ndarray,[2,'size'],float]]),
	'ephem_compute()':(dict(ephname=str),	# read ephemeris from file
		[]),
	'readpos()':(dict(),			# read telescope position from file
		[[np.ndarray,[3],float]]),
	'sitecalc()':(dict(),			# calculate telescope position relative to barycenter
		[]),
	'sitecalc_old()':(dict(),		# calculate telescope position relative to barycenter (discarded)
		[]),
	'tcb2tt()':(dict(),			# return the corresponding TT from TCB
		[te.time]),
	'tdb2tt()':(dict(),			# return the corresponding TT from TDB
		[te.time]),
	'tt2tdb()':(dict(),			# transform TT to TDB and TCB
		[]),
	've_if()':(dict(),			# calculate the motion of the Earth
		[])
}
#
# psr_read.py
functions_variables_psr_read={
	'aliase':dict,			# the pulsar parameters and their aliases
	'aliase_keys':type(dict().keys()),	# the pulsar parameters which have aliases
	'all_paras':set,		# all pulsar parameters
	'para_glitch':set,		# pulsar parameters which describe glitch
	'para_with_err':set,		# pulsar parameters which have errors
	'para_without_err':set,	# pulsar parameters which have no error
	'paras_binary':set,		# pulsar parameters which describe binary motion
	'paras_eph':set,		# pulsar parameters which describe time
	'paras_float':set,		# pulsar parameters which are floats
	'paras_float_array':set,	# pulsar parameters which are float arrays
	'paras_m1':set,		# pulsar parameters which has same dimension as the sixth power of frequency in natural unit system
	'paras_m2':set,		# pulsar parameters which has same dimension as the fifth power of frequency in natural unit system
	'paras_m3':set,		# pulsar parameters which has same dimension as the fourth power of frequency in natural unit system
	'paras_m4':set,		# pulsar parameters which has same dimension as the third power of frequency in natural unit system
	'paras_m5':set,		# pulsar parameters which has same dimension as the square of frequency in natural unit system
	'paras_m6':set,		# pulsar parameters which has same dimension as the frequency in natural unit system
	'paras_p1':set,		# pulsar parameters which has same dimension as the time in natural unit system
	'paras_time':set,		# pulsar parameters which are te.time instances
	'paras_time_array':set,	# pulsar parameters which are arrays of te.time instances
	'paras_text':set,		# pulsar parameters which are texts
	'paras_BT':dict,		# binary pulsar parameters which used in BT model
	'paras_BTJ':dict,		# binary pulsar parameters which used in BTJ model
	'paras_BTX':dict,		# binary pulsar parameters which used in BTX model
	'paras_DD':dict,		# binary pulsar parameters which used in DD model
	'paras_DDH':dict,		# binary pulsar parameters which used in DDH model
	'paras_DDK':dict,		# binary pulsar parameters which used in DDK model
	'paras_DDS':dict,		# binary pulsar parameters which used in DDS model
	'paras_DDGR':dict,		# binary pulsar parameters which used in DDGR model
	'paras_ELL1':dict,		# binary pulsar parameters which used in ELL1 model
	'paras_ELL1k':dict,		# binary pulsar parameters which used in ELL1k model
	'paras_ELL1H':dict,		# binary pulsar parameters which used in ELL1H model
	'paras_MSS':dict,		# binary pulsar parameters which used in MSS model
	'paras_T2':dict,		# binary pulsar parameters which used in T2 model
	'uncertain_pm':set		# pulsar parameters whose dimension is uncertain
}
#
# psr_read.py
psr={
	'acc':te.vector,			# pulsar acceleration vector in ecliptic coordinates
	'acc_equ':te.vector,			# pulsar acceleration vector in equatorial coordinates
	'paras':[list,['nparas'],str],	# list of parameter names
	'pos':te.vector,			# pulsar position vector in ecliptic coordinates
	'pos_equ':te.vector,			# pulsar position vector in equatorial coordinates
	'vel':te.vector,			# pulsar velocity vector in ecliptic coordinates
	'vel_equ':te.vector,			# pulsar velocity vector in equatorial coordinates
	# the list of all parameters in pulsar ephemeris and their types can be found in psr_read.py
	#
	'__eq__()':(dict(other=pr.psr),	# determine whether this psr instance is equal to another psr instance
		[bool]),
	'__init__()':(dict(name=str,parfile=bool,glitch=bool),
		[]),
	'__repr__()':(dict(),			# return a string about this psr instance
		[str]),
	'__str__()':(dict(),			# return a string about this psr instance
		[str]),
	'cal_pos()':(dict(),			# calculate the pulsar position in ecliptic coordinates
		[]),
	'cal_pos_ecl()':(dict(),		# calculate the pulsar position in equatorial coordinates
		[]),
	'change_units()':(dict(),		# change the unit of pulsar parameter to be TCB
		[]),
	'copy()':(dict(),			# produce a counterpart of this psr instance
		[pr.psr]),
	'deal_para()':(dict(paraname=str,paras=dict,paras_key=type(dict().keys()),exce=bool,value=int,err_case=[list,['ncase'],bool],err_exc=[list,['ncase'],str]),	# analyze parameter
		[]),
	'deal_paralist()':(dict(paralist_name=str,paras=dict,paras_key=type(dict().keys()),listlimit=int,exce=bool,value=int,err_case=[list,['ncase'],bool],err_exc=[list,['ncase'],str]),	# analyze parameter list
		[]),
	'dpos()':(dict(vectype=str,coord1=str,coord2=str),	# calculate the derivative of pulsar position along different coordinate axes
		[np.ndarray,[3,3],float]),
	'modify()':(dict(para=str,paraval=float),	# modify the specified parameter to the given value
		[]),
	'readpara()':(dict(parfile=bool,glitch=bool),		# read parameters from a file or a string
		[]),
	'tdb_par()':(dict(),			# return a TDB unit psr instance counterpart
		[pr.psr]),
	'writepar()':(dict(parfile=str),	# write the psr instance into a file
		[])
}
#
# psr_model.py
functions_variables_psr_model={
	'au_dist':int,		# astronomical unit (m)
	'aultsc':float,	# au_dist/te.sl
	'dm_const':float,	# the constant to calculate the dispersion delay
	'dm_const_si':float,	# the constant to calculate the dispersion delay (in SI unit)
	'gg':float,		# gravitational constant
	'kpc2m':float,		# the constant to convert the distance unit kpc to m
	'mas_yr2rad_s':float,	# the constant to convert the proper motion unit mas/yr to rad/s
	'pxconv':float,	# the constant to convert the parallax to distance
	#
	'calcdh()':(dict(ae=float,h3=float,h4=float,nharm=int,sel=int),	# calculate the Shapiro delay with expanded harmonic fitting (Freire & Wex, 2010)
		[[np.ndarray,['nae'],float]]),
	'calculate_gw()':(dict(ra1=float,dec1=float,ra2=float,dec2=float),	# calculate the influence of a gravitational wave source on the ToA
		[[np.ndarray,[1,1],float],[np.ndarray,[1,1],float],float])
}
#
# psr_model.py
psr_timing={
	'__init__()':(dict(psr=pr.psr,time=te.times,freq=float),
		[]),
	'copy()':(dict(),			# produce a counterpart of this psr instance
		[pm.psr_timing]),
	'compute_binary()':(dict(),		# calculate the binary system induced delay
		[]),
	'compute_binary_der()':(dict(),	# calculate the derivative of binary system induced delay to different binary parameters
		[]),
	'compute_dm_delay()':(dict(delt=float),	# calculate the dispersion delay
		[]),
	'compute_phase()':(dict(),		# calculate the pulse phase at specified time and frequency
		[]),
	'compute_shapiro_delay()':(dict(),	# calculate the Shapiro delay
		[]),
	'compute_shklovskii_delay()':(dict(),	# calculate the Shklovskii delay
		[]),
	'compute_te_ssb()':(dict(),		# calculate the delay in solar system
		[]),
	'compute_tropospheric_delay()':(dict(),	# calculate the tropospheric delay
		[]),
	'eccRes()':(dict(prev_p=float,prev_e=float,prev_a=float,prev_epoch=float,prev_theta=float),	# calculate the gravitational wave influence on ToA induced by a elliptic orbit source
		[[np.ndarray,['size'],float]]),
	'phase_der_para()':(dict(paras=[np.ndarray,['nparas'],str]),		# calculate the derivative of pulse phase to different parameters
		[[np.ndarray,['nparas','size'],float]]),
	'solarWindModel()':(dict(),		# calculate the solar wind induced variation of DM
		[[np.ndarray,['size'],float]]),
	'BTmodel()':(dict(der=bool),		# binary pulsar BT model
		[[np.ndarray,['size'],float]]),
	'BTJmodel()':(dict(der=bool),		# binary pulsar BTJ model
		[[np.ndarray,['size'],float]]),
	'BTXmodel()':(dict(der=bool),		# binary pulsar BTX model
		[[np.ndarray,['size'],float]]),
	'DDmodel()':(dict(der=bool),		# binary pulsar DD model
		[[np.ndarray,['size'],float]]),
	'DDGRmodel()':(dict(der=bool),	# binary pulsar DDGR model
		[[np.ndarray,['size'],float]]),
	'DDHmodel()':(dict(der=bool),		# binary pulsar DDH model
		[[np.ndarray,['size'],float]]),
	'DDKmodel()':(dict(der=bool),		# binary pulsar DDK model
		[[np.ndarray,['size'],float]]),
	'DDSmodel()':(dict(der=bool),		# binary pulsar DDS model
		[[np.ndarray,['size'],float]]),
	'ELL1model()':(dict(der=bool),	# binary pulsar ELL1 model
		[[np.ndarray,['size'],float]]),
	'ELL1kmodel()':(dict(der=bool),	# binary pulsar ELL1k model
		[[np.ndarray,['size'],float]]),
	'ELL1Hmodel()':(dict(der=bool),	# binary pulsar ELL1H model
		[[np.ndarray,['size'],float]]),
	'MSSmodel()':(dict(der=bool),		# binary pulsar MSS model
		[[np.ndarray,['size'],float]]),
	'T2model()':(dict(der=bool),		# binary pulsar T2 model
		[[np.ndarray,['size'],float]]),
	'T2_PTAmodel()':(dict(der=bool),	# binary pulsar T2_PTA model
		[[np.ndarray,['size'],float]]),
}
#
# adfunc.py
functions_variables_ad_func={
	'baseline()':(dict(data=[np.ndarray,['nbin'],float],base_nbin=int,pos=bool),	# determine the baseline of the data
		[float]),
	'baseline0()':(dict(data=[np.ndarray,['nbin'],float]),	# determine the baseline of the data (old version)
		[float]),
	'dmdet()':(dict(fftdata=[np.ndarray,['nchan','nbin'],complex],dmconst=[np.ndarray,['nbin'],float],dm0=float,dmw=float,polynum=int,prec=float),	# determine the best DM with the specified precision
		[float,float,float,[np.ndarray,['ntest'],float],[np.ndarray,['ntest'],float]]),
	'radipos()':(dict(data=[np.ndarray,['nbin'],float],crit=float,base=bool,base_nbin=int),	# determine the radiation position of the data
		[[np.ndarray,['nradipos'],int]]),
	'reco()':(dict(x=str),	# recognize the telescope name from its aliases
		[str]),
	'shift()':(dict(y=[np.ndarray,['ny'],complex],x=float),	# assistant function for dmdet(); return the shifted counterpart of the Fourier-domain array
		[[np.ndarray,['ny*2-2'],float]])
}
