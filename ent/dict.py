import numpy as np
import struct as st
import os
#
dirname=os.path.split(os.path.realpath(__file__))[0]
#
def dict_write(fname,data):
    f=open(fname,'wb+')
    for i in list(data.keys()):
        ln=len(i)
        f.write(st.pack('>I',ln))
        f.write(i)
        d=np.array(data[i])
        shape=np.array(d.shape)
        ls=len(shape)
        f.write(st.pack('>I',ls))
        f.write(st.pack('>'+str(ls)+'I',*shape))
        if d.dtype==np.bool: dtype='?'
        elif d.dtype==np.float32: dtype='f'
        elif d.dtype==np.float64: dtype='d'
        elif d.dtype==np.int8: dtype='b'
        elif d.dtype==np.int16: dtype='h'
        elif d.dtype==np.int32: dtype='i'
        elif d.dtype==np.int64: dtype='q'
        f.write(dtype.encode())
        f.write(st.pack('>'+str(d.size)+dtype,*d.reshape(-1)))
    f.close()
#
def dict_load(fname):
    f=open(fname,'rb')
    dic={}
    a=f.read(4)
    while a:
        ln=st.unpack('>I',a)[0]
        name=f.read(ln)
        ls=st.unpack('>I',f.read(4))[0]
        shape=np.array(st.unpack('>'+str(ls)+'I',f.read(4*ls)))
        dtype=f.read(1).decode()
        if dtype=='?': nbin=1
        elif dtype=='f': nbin=4
        elif dtype=='d': nbin=8
        elif dtype=='b': nbin=1
        elif dtype=='h': nbin=2
        elif dtype=='i': nbin=4
        elif dtype=='q': nbin=8
        ld=int(shape.prod())
        data=np.array(st.unpack('>'+str(ld)+dtype,f.read(ld*nbin)))
        dic[name]=data.reshape(*shape)
        a=f.read(4)
    f.close()
    return dic
#
def save_write(save):
    f=open(dirname+'/src/save.dat','wb+')
    for i in save.keys():
        f.write(i)
        dif,level,j=save[i]
        f.write(st.pack('>I',dif))
        f.write(st.pack('>I',level))
        f.write(st.pack('>?',j))
#
def save_load():
    f=open(dirname+'/src/save.dat','rb')
    save={}
    for i in range(3):
        name=f.read(5)
        dif,level,j=st.unpack('>II?',f.read(9))
        save[name]=[dif,level,j]
    return save
