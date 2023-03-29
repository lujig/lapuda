import os
import numpy as np
import struct as st
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
	def __write_size__(self,size):
		self.file=open(self.name,'rb+')
		self.file.seek(0)
		if len(size)!=5:
			raise Exception('Worng size of data.')
		self.file.write(st.pack('>Q4I',*size))
		self.file.flush()
		self.file.close()
		self.__size__=size
	#
	def __read_size__(self):
		self.file=open(self.name,'rb')
		size=np.array(st.unpack('>Q4I',self.file.read(24)))
		self.file.seek(0,2)
		if size[0]==self.file.tell():
			return size
		else:
			raise Exception('Invalid ld file.')
	#
	def __refresh_size__(self):
		self.file=open(self.name,'rb+')
		self.file.seek(0,2)
		self.__size__[0]=self.file.tell()
		self.__write_size__(self.__size__)
		self.file.close()
		#print self.__size__
	#
	def write_shape(self,shape):
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
	def read_shape(self):
		return self.__size__[1:]
	#
	def write_chan(self,data,chan_num):
		data=data.reshape(-1)
		ndata_chan=np.int64(np.array(self.__size__[2:]).prod())
		if ndata_chan!=data.size:
			raise Exception('Data size unmatches the file.')
		if chan_num>self.__size__[1]:
			raise Exception('The input channel number is larger than total channel number of file.')
		self.file=open(self.name,'rb+')
		self.file.seek(24+chan_num*ndata_chan*8)
		self.file.write(st.pack('>'+str(ndata_chan)+'d',*data))
		#print self.file.tell(),ndata_chan,data.shape,chan_num
		self.file.flush()
		self.file.close()
		self.__refresh_size__()
	#
	def read_chan(self,chan_num):
		ndata_chan=np.int64(np.array(self.__size__[2:]).prod())
		if chan_num>self.__size__[1]:
			raise Exception('The input channel number is larger than total channel number of file.')
		self.file=open(self.name,'rb')
		self.file.seek(24+chan_num*ndata_chan*8)
		data=np.array(st.unpack('>'+str(ndata_chan)+'d',self.file.read(ndata_chan*8)))
		self.file.close()
		return data.reshape(self.__size__[2:])
	#
	def read_data(self):
		data_shape=np.int64(np.array(self.__size__[1:]))
		data=np.zeros(data_shape,dtype=np.float64)
		ndata_chan=np.int64(np.array(self.__size__[2:]).prod())
		self.file=open(self.name,'rb')
		for chan_num in range(self.__size__[1]):
			self.file.seek(24+chan_num*ndata_chan*8)
			data[chan_num]=np.array(st.unpack('>'+str(ndata_chan)+'d',self.file.read(ndata_chan*8))).reshape(self.__size__[2:])
		return data
	#
	def __read_chan0__(self,chan_num,ndata_chan0):
		if chan_num>self.__size__[1]:
			raise Exception('The input channel number is larger than total channel number of file.')
		self.file=open(self.name,'rb')
		self.file.seek(24+chan_num*ndata_chan0*8)
		data=np.array(st.unpack('>'+str(ndata_chan0)+'d',self.file.read(ndata_chan0*8)))
		self.file.close()
		return data
	#
	def write_period(self,data,p_num):
		ndata_period=np.int64(self.__size__[3]*self.__size__[4])
		ndata_chan=np.int64(ndata_period*self.__size__[2])
		if (self.__size__[1]*ndata_period)!=data.size:
			raise Exception('Data size unmatches the file.')
		if p_num>self.__size__[2]:
			raise Exception('The input period number is larger than total period number of file.')
		data=data.reshape(self.__size__[1],self.__size__[3]*self.__size__[4])
		self.file=open(self.name,'rb+')
		for i in range(self.__size__[1]):
			self.file.seek(24+ndata_chan*i*8+p_num*ndata_period*8)
			self.file.write(st.pack('>'+str(ndata_period)+'d',*data[i]))
		#print self.file.tell(),data.shape,ndata_chan,ndata_period,p_num
		self.file.flush()
		self.file.close()
		self.__refresh_size__()
	#
	def read_period(self,p_num):
		ndata_period=np.int64(self.__size__[3]*self.__size__[4])
		ndata_chan=np.int64(ndata_period*self.__size__[2])
		if p_num>self.__size__[2]:
			raise Exception('The input period number is larger than total period number of file.')
		data=np.zeros([self.__size__[1],self.__size__[3],self.__size__[4]])
		self.file=open(self.name,'rb')
		for i in range(self.__size__[1]):
			self.file.seek(24+ndata_chan*i*8+p_num*ndata_period*8)
			data[i]=np.array(st.unpack('>'+str(ndata_period)+'d',self.file.read(ndata_period*8))).reshape(self.__size__[3],self.__size__[4])
		self.file.close()
		return data
	#
	def __write_bin_segment__(self,data,bin_start):
		ndata_chan=np.int64(np.array(self.__size__[2:]).prod())
		if self.__size__[1]!=len(data):
			raise Exception('Data size unmatches the file.')
		if (bin_start*self.__size__[4]+np.array(data.shape[1:]).prod())>ndata_chan or bin_start<0:
			raise Exception('The input bin number is overrange.')
		data=data.reshape(self.__size__[1],-1)
		self.file=open(self.name,'rb+')
		for i in range(self.__size__[1]):
			self.file.seek(24+ndata_chan*i*8+bin_start*8*self.__size__[4])
			self.file.write(st.pack('>'+str(data.shape[1])+'d',*data[i]))
		self.file.flush()
		self.file.close()
		self.__refresh_size__()
	#
	def __write_chanbins_add__(self,data,bin_start,chan_num):
		ndata_chan=np.int64(np.array(self.__size__[2:]).prod())
		if self.__size__[4]!=data.shape[1]:
			raise Exception('Data size unmatches the file.')
		if (bin_start*self.__size__[4]+np.array(data.shape).prod())>ndata_chan or bin_start<0:
			raise Exception('The input bin number is overrange.')
		data=data.reshape(-1)
		size=data.size
		with open(self.name,'rb+') as self.file:
			fcntl.flock(self.file.fileno(),fcntl.LOCK_EX)
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
	def __write_chanbins__(self,data,bin_start,chan_num):
		ndata_chan=np.int64(np.array(self.__size__[2:]).prod())
		if self.__size__[4]!=data.shape[1]:
			raise Exception('Data size unmatches the file.')
		if (bin_start*self.__size__[4]+np.array(data.shape).prod())>ndata_chan or bin_start<0:
			raise Exception('The input bin number is overrange.')
		data=data.reshape(-1)
		self.file=open(self.name,'rb+')
		self.file.seek(24+ndata_chan*chan_num*8+bin_start*8*self.__size__[4])
		self.file.write(st.pack('>'+str(data.size)+'d',*data))
		self.file.flush()
		self.file.close()
		self.__refresh_size__()
	#
	def __read_bin_segment__(self,bin_start,bin_num):
		ndata_chan=np.int64(np.array(self.__size__[2:]).prod())
		if ((bin_start+bin_num)*self.__size__[4])>ndata_chan or bin_start<0:
			raise Exception('The input bin number is overrange.')
		data=np.zeros([self.__size__[1],bin_num,self.__size__[4]])
		self.file=open(self.name,'rb')
		for i in range(self.__size__[1]):
			self.file.seek(24+ndata_chan*i*8+bin_start*8*self.__size__[4])
			data[i]=np.array(st.unpack('>'+str(bin_num*self.__size__[4])+'d',self.file.read(bin_num*self.__size__[4]*8))).reshape(bin_num,self.__size__[4])
		self.file.close()
		return data
	#
	def chan_scrunch(self,select_chan=[],start_period=0,end_period=0):
		ndata_chan=np.int64(np.array(self.__size__[2:]).prod())
		if len(select_chan)==0:
			select_chan=np.arange(self.__size__[1])
		elif not set(select_chan).issubset(set(np.arange(self.__size__[1]))):
			raise Exception('The input channel number is overrange.')
		elif start_period<0 or end_period>self.__size__[2]:
			raise Exception('The input period number is overrange.')
		elif start_period>end_period:
			raise Exception('The starting period number is larger than ending period number.')
		if end_period==0:
			end_period=self.__size__[2]
		bin_start=start_period*self.__size__[3]*self.__size__[4]
		bin_num=(end_period-start_period)*self.__size__[3]*self.__size__[4]
		data=np.zeros(bin_num)
		self.file=open(self.name,'rb')
		info=self.read_info()
		if 'weight' in info.keys(): weight=self.read_para('chan_weight')
		else: weight=np.ones(self.__size__[1])
		for i in select_chan:
			self.file.seek(24+ndata_chan*i*8+bin_start*8)
			data+=np.array(st.unpack('>'+str(bin_num)+'d',self.file.read(bin_num*8)))*weight[i]
		data=data.reshape([end_period-start_period,self.__size__[3],self.__size__[4]])
		return data
	#
	def period_scrunch(self,start_period=0,end_period=0,select_chan=[]):
		ndata_chan=np.int64(np.array(self.__size__[2:]).prod())
		if len(select_chan)==0:
			select_chan=np.arange(self.__size__[1])
		elif not set(select_chan).issubset(set(np.arange(self.__size__[1]))):
			raise Exception('The input channel number is overrange.')
		elif start_period<0 or end_period>self.__size__[2]:
			raise Exception('The input period number is overrange.')
		elif start_period>end_period:
			raise Exception('The starting period number is larger than ending period number.')
		if end_period==0:
			end_period=self.__size__[2]
		bin_start=start_period*self.__size__[3]*self.__size__[4]
		bin_num=(end_period-start_period)*self.__size__[3]*self.__size__[4]
		data=np.zeros([len(select_chan),self.__size__[3],self.__size__[4]])
		self.file=open(self.name,'rb')
		for i in range(len(select_chan)):
			self.file.seek(24+ndata_chan*select_chan[i]*8+bin_start*8)
			data[i]=np.array(st.unpack('>'+str(bin_num)+'d',self.file.read(bin_num*8))).reshape(end_period-start_period,self.__size__[3],self.__size__[4]).sum(0)
		return data
	#
	def read_info(self):
		self.file=open(self.name,'r')
		self.file.seek(24+np.array(self.__size__[1:]).prod()*8)
		infotext=self.file.read()
		self.file.close()
		info=json.loads(infotext)
		return info
	#
	def write_info(self,info):
		self.file=open(self.name,'r+')
		self.file.seek(24+np.array(self.__size__[1:]).prod()*8)
		self.file.write(json.dumps(info,indent=1))
		self.file.truncate()
		self.file.flush()
		self.file.close()
		self.__refresh_size__()
	#
	def write_para(self,key,value):
		info=self.read_info()
		info[key]=value
		self.write_info()
	#
	def read_para(self,key):
		info=self.read_info()
		if key in info.keys():
			return info[key]
		else:
			raise Exception('Wrong parameter name.')
#
