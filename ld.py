import os,platform
import numpy as np
import struct as st
import psutil
if platform.system()!='Windows':
	import fcntl
import json
#
class ld():
	def __init__(self,name):
		self.name=name
		if not os.path.isfile(name):
			self.__size__=np.array([24,0,0,0,0],dtype=np.int64) # [filesize, nchan, nperiod, nbin, npol] [Q, I, I, I, I] 24 bytes in total
			self.file=open(name,'wb')
			self.file.close()
			self.__write_size__(self.__size__)
		else:
			self.__size__=self.__read_size__()
	#
	def __write_size__(self,size):	# write the data shape into the LD file
		self.file=open(self.name,'rb+')
		self.file.seek(0)
		if len(size)!=5:
			raise Exception('Worng size of data.')
		self.file.write(st.pack('>Q4I',*size))
		self.file.flush()
		self.file.close()
		self.__size__=size
	#
	def __read_size__(self):	# read the size of the LD file, containing the file size and the shape along the 4 dimensions of the data
		self.file=open(self.name,'rb')
		size=np.array(st.unpack('>Q4I',self.file.read(24)))
		self.file.seek(0,2)
		if size[0]==self.file.tell():
			return size
		else:
			raise Exception('Invalid ld file.')
	#
	def __refresh_size__(self):	# refresh the file size value in the LD file
		self.file=open(self.name,'rb+')
		self.file.seek(0,2)
		self.__size__[0]=self.file.tell()
		self.__write_size__(self.__size__)
		self.file.close()
		#print self.__size__
	#
	def write_shape(self,shape):	# write the data shape into the LD file
		if len(shape)!=4:
			raise Exception('Data shape should be 4 integers.')
		self.__size__[1:]=np.int32(shape)
		self.__write_size__(self.__size__)
		self.file=open(self.name,'rb+')
		self.file.seek(24,0)
		self.file.truncate()
		self.file.flush()
		self.file.close()
	#
	def read_shape(self):	# read the shape along the 4 dimensions of the data
		return self.__size__[1:]
	#
	def write_chan(self,data,chan_num):	# write data into specific channel index in LD file
		data=data.reshape(self.__size__[2:])
		ndata_chan=np.int64(np.array(self.__size__[2:]).prod())
		if ndata_chan!=data.size:
			raise Exception('Data size unmatches the file.')
		if chan_num>self.__size__[1]:
			raise Exception('The input channel number is larger than total channel number of file.')
		d0=np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r+',dtype='>d',order='C')
		d0[chan_num]=data
		d0.flush()
		del d0
		self.__refresh_size__()
	#
	def read_chan(self,chan_num,pol=-1):	# read the data in specific channel index in LD file
		if chan_num>self.__size__[1]:
			raise Exception('The input channel number is larger than total channel number of file.')
		if (-1).__eq__(pol) is True: pol=np.arange(self.__size__[4])
		else: pol=np.array(pol).reshape(-1)
		data=np.array(np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r',dtype='>d',order='C')[chan_num][:,:,pol])
		return data
	#
	def read_pol(self,pol_num):	# read the data in specific polarization index in LD file
		if pol_num>self.__size__[4]:
			raise Exception('The input channel number is larger than total channel number of file.')
		data=np.array(np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r',dtype='>d',order='C')[:,:,:,pol_num])
		return data
	#
	def read_data(self,select_chan=[],start_period=0,end_period=0,start_bin=0,end_bin=0,pol=-1):	# read all data in LD file
		if (-1).__eq__(pol) is True: pol=np.arange(self.__size__[4])
		else: pol=np.array(pol).reshape(-1)
		if not set(pol).issubset(set(np.arange(self.__size__[4]))):
			raise Exception('The input polarization number is overrange.')
		#
		if len(select_chan)==0:
			select_chan=np.arange(self.__size__[1])
		elif not set(select_chan).issubset(set(np.arange(self.__size__[1]))):
			raise Exception('The input channel number is overrange.')
		if end_period==0: end_period=self.__size__[2]
		if end_bin==0: end_bin=self.__size__[3]
		if start_period>end_period or start_period<0 or end_period>self.__size__[2]:
			raise Exception('The number of subintegration range is not right')
		if start_bin==end_bin or start_bin<0 or end_bin>self.__size__[3]:
			raise Exception('The number of phase bin range is not right')
		if start_bin>end_bin: bins=np.append(np.arange(int(start_bin),self.__size__[3]),np.arange(int(end_bin)))
		else: bins=np.arange(start_bin,end_bin)
		memory=psutil.virtual_memory()
		avlmemory = memory.available/1024/1024/1024  #Gb
		dsize=len(select_chan)*(end_period-start_period)*len(bins)*8/2**30  #Gb
		if dsize>avlmemory*0.8:
			raise Exception('The data is larger than the available memory')
		data=np.array(np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r',dtype='>d',order='C')[:,start_period:end_period][select_chan][:,:,bins][:,:,:,pol])
		return data
	#
	def write_data(self,data):	# write all data of a LD file
		if data.size!=np.prod(self.__size__[1:]):
			raise Exception('Data size unmatches the file.')
		data=data.reshape(*self.__size__[1:])
		d0=np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r+',dtype='>d',order='C')
		d0[:]=data
		d0.flush()
		del d0
		self.__refresh_size__()
	#
	def __read_chan0__(self,chan_num,ndata_chan0):	# discarded
		if chan_num>self.__size__[1]:
			raise Exception('The input channel number is larger than total channel number of file.')
		data=np.array(np.memmap(self.name,offset=24,shape=(chan_num+1,ndata_chan0),mode='r',dtype='>d',order='C')[chan_num])
		return data
	#
	def write_period(self,data,p_num):	# write data into specific sub-integration index in LD file
		ndata_period=np.int64(self.__size__[3]*self.__size__[4])
		if (self.__size__[1]*ndata_period)!=data.size:
			raise Exception('Data size unmatches the file.')
		if p_num>self.__size__[2]:
			raise Exception('The input period number is larger than total period number of file.')
		data=data.reshape(self.__size__[1],self.__size__[3],self.__size__[4])
		d0=np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r+',dtype='>d',order='C')
		d0[:,p_num]=data
		d0.flush()
		del d0
		self.__refresh_size__()
	#
	def read_period(self,p_num,pol=-1):	# read the data in specific sub-integration index in LD file
		if p_num>self.__size__[2]:
			raise Exception('The input period number is larger than total period number of file.')
		if (-1).__eq__(pol) is True: pol=np.arange(self.__size__[4])
		else: pol=np.array(pol).reshape(-1)
		data=np.array(np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r',dtype='>d',order='C')[:,p_num:p_num+1,:,pol])
		return data.reshape(*np.array(data.shape)[[0,2,3]])
	#
	def __write_bin_segment__(self,data,bin_start):	# write the data segment with all frequency channels into LD file at the specified starting bin index
		ndata_chan=np.int64(np.array(self.__size__[2:]).prod())
		if self.__size__[1]!=len(data):
			raise Exception('Data size unmatches the file.')
		if (bin_start*self.__size__[4]+np.array(data.shape[1:]).prod())>ndata_chan or bin_start<0:
			raise Exception('The input bin number is overrange.')
		data=data.reshape(self.__size__[1],-1,self.__size__[4])
		d0=np.memmap(self.name,offset=24,shape=(self.__size__[1],self.__size__[2]*self.__size__[3],self.__size__[4]),mode='r+',dtype='>d',order='C')
		d0[:,bin_start:(bin_start+data.shape[1])]=data
		d0.flush()
		del d0
		self.__refresh_size__()
	#
	def __write_chanbins_add__(self,data,bin_start,chan_num):	# add the data series onto specific frequency channel of LD file at specified starting bin index
		ndata_chan=np.int64(np.array(self.__size__[2:]).prod())
		if self.__size__[4]!=data.shape[1]:
			raise Exception('Data size unmatches the file.')
		if (bin_start*self.__size__[4]+np.array(data.shape).prod())>ndata_chan or bin_start<0:
			raise Exception('The input bin number is overrange.')
		data=data.reshape(-1)
		size=data.size
		with open(self.name,'rb+') as self.file:
			if platform.system()!='Windows': fcntl.flock(self.file.fileno(),fcntl.LOCK_EX)
			self.file.seek(24+ndata_chan*chan_num*8+bin_start*8*self.__size__[4])
			data0=np.zeros_like(data,dtype=np.float64)
			data0tmp=self.file.read(size*8)
			length0=np.int64(len(data0tmp)/8)
			data0[:length0]=np.array(st.unpack('>'+str(length0)+'d',data0tmp))
			self.file.seek(24+ndata_chan*chan_num*8+bin_start*8*self.__size__[4])
			self.file.write(st.pack('>'+str(size)+'d',*(data+data0)))
			self.file.flush()
			#self.file.close()
		self.__refresh_size__()
	#
	def __write_chanbins__(self,data,bin_start,chan_num):	# write the data series into specific frequency channel of LD file at specified starting bin index
		ndata_chan=np.int64(np.array(self.__size__[2:]).prod())
		if self.__size__[4]!=data.shape[1]:
			raise Exception('Data size unmatches the file.')
		if (bin_start*self.__size__[4]+np.array(data.shape).prod())>ndata_chan or bin_start<0:
			raise Exception('The input bin number is overrange.')
		data=data.reshape(-1,self.__size__[4])
		d0=np.memmap(self.name,offset=24,shape=(self.__size__[1],self.__size__[2]*self.__size__[3],self.__size__[4]),mode='r+',dtype='>d',order='C')
		d0[chan_num,bin_start:(bin_start+data.shape[1])]=data
		self.__refresh_size__()
	#
	def __read_bin_segment__(self,bin_start,bin_num,pol=-1):	# read data segment with specified starting bin index and total bin numbers in all frequency channels
		if ((bin_start+bin_num)*self.__size__[4])>ndata_chan or bin_start<0:
			raise Exception('The input bin number is overrange.')
		if (-1).__eq__(pol) is True: pol=np.arange(self.__size__[4])
		else: pol=np.array(pol).reshape(-1)
		data=np.array(np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r',dtype='>d',order='C')[:,:,bin_start:(bin_start+bin_num),pol])
		return data
	#
	def bin_scrunch(self,select_chan=[],start_period=0,end_period=0,select_bin=[],baselinepos=0,pol=-1):
		if (-1).__eq__(pol) is True: pol=np.arange(self.__size__[4])
		else: pol=np.array(pol).reshape(-1)
		if not set(pol).issubset(set(np.arange(self.__size__[4]))):
			raise Exception('The input polarization number is overrange.')
		#
		if len(select_chan)==0: select_chan=np.arange(self.__size__[1])
		elif not set(select_chan).issubset(set(np.arange(self.__size__[1]))):
			raise Exception('The input channel number is overrange.')
		#
		if len(select_bin)==0: select_bin=np.arange(self.__size__[3])
		elif not set(select_bin).issubset(set(np.arange(self.__size__[3]))):
			raise Exception('The input phase bin number is overrange.')
		#
		start_period=int(start_period)
		end_period=int(end_period)
		if end_period ==0: end_period=self.__size__[2]
		if start_period > end_period or start_period<0 or end_period>self.__size__[2]:
			raise Exception('The number of subintegration range is not right')
		#
		memory=psutil.virtual_memory()
		avlmemory = memory.available/1024/1024/1024  #Gb
		#
		nbin=self.__size__[3]
		base_nbin = int(nbin/10)
		if baselinepos==0:
			import adfunc as af
			info=self.read_info()
			if 'weights' in info['data_info'].keys():
				template=self.profile(weighted='weights',pol=0).reshape(nbin)
			else: template=self.profile(weighted='chan_weight',pol=0).reshape(nbin)
			_,bin0=af.baseline(template,pos=True)
		else: bin0=baselinepos
		if bin0+base_nbin>nbin: bins=np.append(np.arange(bin0+base_nbin-nbin),np.arange(bin0,nbin))
		else: bins=np.arange(bin0,bin0+base_nbin)
		data=np.zeros([len(select_chan),end_period-start_period,len(pol)])
		nblock=int(np.ceil(self.__size__[1]*(end_period-start_period)*self.__size__[3]*self.__size__[4]/2**27/(avlmemory*0.1)))
		cblock=int(np.ceil(self.__size__[1]/nblock))
		for i in range(nblock):
			sc=list(np.array(list(select_chan))[cblock*i:(cblock*i+cblock)])
			if len(sc)>0:
				d0=np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r',dtype='>d',order='C')[:,start_period:end_period,:,:]
				base=d0[sc][:,:,bins][:,:,:,pol].sum(2)
				data[cblock*i:(cblock*i+cblock)]=d0[sc][:,:,select_bin][:,:,:,pol].sum(2)-base
				del d0
		return data
	#
	def profile(self,select_chan=[],start_period=0,end_period=0,weighted='None',pol=-1):
		if (-1).__eq__(pol) is True: pol=np.arange(self.__size__[4])
		else: pol=np.array(pol).reshape(-1)
		if not set(pol).issubset(set(np.arange(self.__size__[4]))):
			raise Exception('The input polarization number is overrange.')
		#
		scmark=True
		if len(select_chan)==0: 
			select_chan=np.arange(self.__size__[1])
			scmark=False
		elif not set(select_chan).issubset(set(np.arange(self.__size__[1]))):
			raise Exception('The input channel number is overrange.')
		#
		start_period=int(start_period)
		end_period=int(end_period)
		if end_period ==0: end_period=self.__size__[2]
		if start_period > end_period or start_period<0 or end_period>self.__size__[2]:
			raise Exception('The number of subintegration range is not right')
		#
		memory=psutil.virtual_memory()
		avlmemory = memory.available/1024/1024/1024  #Gb
		#
		info=self.read_info()
		#
		if weighted == 'None':
			if scmark:
				weight=np.zeros(len(weight))
				weight[select_chan]=1
		elif weighted == 'weights' or weighted == 'chan_weight':
			if weighted == 'chan_weight':
				if 'chan_weight' not in info['data_info'].keys(): raise Exception('the chan_weight is not given')
				weight=np.array(self.read_para('chan_weight'))
				scchan=np.zeros(len(weight))
				scchan[select_chan]=1
				if (weight==np.mean(weight)).all(): chan_weight=weight[0]
				else: scmark=True
				weight=(weight*scchan).reshape(-1,1,1,1)
			elif weighted == 'weights':
				if 'weights' not in info['data_info'].keys(): raise Exception('the weights is not given')
				weight=np.array(self.read_info()['data_info']['weights'])[:,start_period:end_period].reshape(-1,end_period-start_period,1,1)
		else: raise Exception('Please give the right weighted info')
		#
		chans=np.arange(self.__size__[1])
		sc0=set(chans).intersection(select_chan)
		if scmark or weighted == 'weights': 
			data=np.zeros([self.__size__[3],*np.shape(pol)])
			datasize=data.size*(end_period-start_period)
			nblock=int(np.ceil(self.__size__[1]*datasize/2**27/(avlmemory*0.1)))
			cblock=int(np.ceil(self.__size__[1]/nblock))
			for i in range(nblock):
				sc=list(sc0.intersection(np.arange(cblock*i,(cblock*i+cblock))))
				if sc:
					d0=np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r',dtype='>d',order='C')[:,start_period:end_period]
					data+=(d0[sc][:,:,:,pol]*weight[sc]).sum((0,1))
					del d0
		else: 
			d0=np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r',dtype='>d',order='C')[:,start_period:end_period]
			data=d0.sum((0,1))[:,pol]
			del d0
		return data
	#
	def chan_scrunch(self,select_chan=[],start_period=0,end_period=0,weighted='None',pol=-1):	# scrunch the data in LD file along frequency axis
		if (-1).__eq__(pol) is True: pol=np.arange(self.__size__[4])
		else: pol=np.array(pol).reshape(-1)
		if not set(pol).issubset(set(np.arange(self.__size__[4]))):
			raise Exception('The input polarization number is overrange.')
		#
		scmark=True
		if len(select_chan)==0: 
			select_chan=np.arange(self.__size__[1])
			scmark=False
		elif not set(select_chan).issubset(set(np.arange(self.__size__[1]))):
			raise Exception('The input channel number is overrange.')
		#
		start_period=int(start_period)
		end_period=int(end_period)
		if end_period ==0: end_period=self.__size__[2]
		if start_period > end_period or start_period<0 or end_period>self.__size__[2]:
			raise Exception('The number of subintegration range is not right')
		#
		memory=psutil.virtual_memory()
		avlmemory = memory.available/1024/1024/1024  #Gb
		#
		info=self.read_info()
		#
		if weighted == 'None':
			if scmark:
				weight=np.zeros(len(weight))
				weight[select_chan]=1
		elif weighted == 'weights' or weighted == 'chan_weight':
			if weighted == 'chan_weight':
				if 'chan_weight' not in info['data_info'].keys(): raise Exception('the chan_weight is not given')
				weight=np.array(self.read_para('chan_weight'))
				scchan=np.zeros(len(weight))
				scchan[select_chan]=1
				if (weight==np.mean(weight)).all(): chan_weight=weight[0]
				else: scmark=True
				weight=(weight*scchan).reshape(-1,1,1,1)
			elif weighted == 'weights':
				if 'weights' not in info['data_info'].keys(): raise Exception('the weights is not given')
				weight=np.array(self.read_info()['data_info']['weights'])[:,start_period:end_period].reshape(-1,end_period-start_period,1,1)
		else: raise Exception('Please give the right weighted info')
		#
		if weighted == 'weights' or scmark: 
			data=np.zeros([end_period-start_period,self.__size__[3],*np.shape(pol)])
			datasize=data.size
			nblock=int(np.ceil(self.__size__[1]*datasize/2**27/(avlmemory*0.1)))
			cblock=int(np.ceil(self.__size__[1]/nblock))
			for i in range(nblock):
				sc=list(set(select_chan).intersection(np.arange(cblock*i,(cblock*i+cblock))))
				if sc:
					d0=np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r',dtype='>d',order='C')[:,start_period:end_period]
					data+=(d0[sc][:,:,:,pol]*weight[sc]).sum(0)
					del d0
		else: 
			d0=np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r',dtype='>d',order='C')[:,start_period:end_period]
			data=d0.sum(0)[:,:,pol]
			del d0
			if weighted == 'chan_weight': data*=chan_weight
		return data
	#
	def period_scrunch(self,start_period=0,end_period=0,select_chan=[],weighted='None',pol=-1):	# scrunch the data in LD file along subint axis
		if (-1).__eq__(pol) is True: pol=np.arange(self.__size__[4])
		else: pol=np.array(pol).reshape(-1)
		if not set(pol).issubset(set(np.arange(self.__size__[4]))):
			raise Exception('The input polarization number is overrange.')
		#
		scmark=True
		if len(select_chan)==0: 
			select_chan=np.arange(self.__size__[1])
			scmark=False
		elif not set(select_chan).issubset(set(np.arange(self.__size__[1]))):
			raise Exception('The input channel number is overrange.')
		#
		start_period=int(start_period)
		end_period=int(end_period)
		if end_period ==0: end_period=self.__size__[2]
		if start_period > end_period or start_period<0 or end_period>self.__size__[2]:
			raise Exception('The number of subintegration range is not right')
		#
		memory=psutil.virtual_memory()
		avlmemory = memory.available/1024/1024/1024  #Gb
		#
		info=self.read_info()
		#
		if weighted == 'None': 
			if scmark:
				weight=np.zeros(len(weight))
				weight[select_chan]=1
		elif weighted == 'weights' or weighted == 'chan_weight':
			if weighted == 'chan_weight':
				if 'chan_weight' not in info['data_info'].keys(): raise Exception('the chan_weight is not given')
				weight=np.array(self.read_para('chan_weight'))
				scchan=np.zeros(len(weight))
				scchan[select_chan]=1
				if (weight==np.mean(weight)).all(): chan_weight=weight[0]
				else: scmark=True
				weight=(weight*scchan).reshape(-1,1,1,1)
			elif weighted == 'weights':
				if 'weights' not in info['data_info'].keys(): raise Exception('the weights is not given')
				weight=np.array(self.read_info()['data_info']['weights'])[:,start_period:end_period].reshape(-1,end_period-start_period,1,1)
		else: raise Exception('Please give the right weighted info')
		#
		chans=np.arange(self.__size__[1])
		sc0=set(chans).intersection(select_chan)
		if weighted == 'weights' or scmark: 
			data=np.zeros([len(sc0),self.__size__[3],*np.shape(pol)])
			datasize=data.size*(end_period-start_period)
			nblock=int(np.ceil(datasize/2**27/(0.1*avlmemory)))
			cblock=int(np.ceil(self.__size__[1]/nblock))
			for i in range(nblock):
				sc=list(np.array(list(sc0))[cblock*i:(cblock*i+cblock)])
				if sc:
					d0=np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r',dtype='>d',order='C')[:,start_period:end_period]
					data[cblock*i:(cblock*i+cblock)]=(d0[sc][:,:,:,pol]*weight[sc]).sum(1)
					del d0
		else: 
			d0=np.memmap(self.name,offset=24,shape=tuple(self.__size__[1:]),mode='r',dtype='>d',order='C')[:,start_period:end_period]
			data=d0.sum(1)[:,:,pol]
			del d0
			if weighted == 'chan_weight': data*=chan_weight
		return data
	#
	def read_info(self):	# read the information of LD file
		self.file=open(self.name,'r')
		self.file.seek(24+np.array(self.__size__[1:]).prod()*8)
		infotext=self.file.read()
		self.file.close()
		info=json.loads(infotext)
		return info
	#
	def write_info(self,info):	# write the information dictionary into the LD file
		for i in info.values():
			if type(i) is not dict: raise Exception('The information format is wrong.')
		self.file=open(self.name,'r+')
		self.file.seek(24+np.array(self.__size__[1:]).prod()*8)
		self.file.write(json.dumps(info,indent=1))
		self.file.truncate()
		self.file.flush()
		self.file.close()
		self.__refresh_size__()
	#
	def change_info(self):
		import adfunc as af
		self.file=open(self.name,'r')
		self.file.seek(24+np.array(self.__size__[1:]).prod()*8)
		infotext=self.file.readlines()
		self.file.close()
		info={}
		for line in infotext:
			if line[0]==' ':
				if key not in info.keys():
					info[key]=line.strip('\n').strip()
				elif type(info[key])==list:
					info[key].append(line.strip('\n').strip())
				else:
					info[key]=[info[key],line.strip('\n').strip()]
			else:
				key=line.strip('\n').strip()
		js=af.dic2json(info)
		if 'freq_start' in js['data_info'].keys():
			js['data_info']['freq_start']-=1000/16384
			js['data_info']['freq_end']-=1000/16384
		self.write_info(js)
	#
	def write_para(self,key,value):	# write or modify the value of a specific key into the information of LD file
		import adfunc as af
		li,ldic=af.parakey()
		info=self.read_info()
		info[ldic[key]][key]=value
		self.write_info(info)
	#
	def read_para(self,key):	# read the value of the specified key in the information of LD file
		import adfunc as af
		info=af.json2dic(self.read_info())
		if key in info.keys():
			return info[key]
		else:
			raise Exception('Wrong parameter name.')
#
