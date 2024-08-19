import numpy as np
import numpy.random as nr
import random,time
import itertools as it
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
mpl.use('TkAgg')  
#
class tetris_class:
	def __init__(self,name,width,height,fig,axis,canvas,fig1,axis1,canvas1,ctrl,quitbutton,stopbutton,frame,root,save,saven,chdict,nextfunc,next,quitfunc):
		self.width=width
		self.height=height
		self.quitlevel=10
		self.displaying=False
		self.nextfunc=nextfunc
		self.next=next
		self.quitfunc=quitfunc
		self.save=save
		self.saven=saven
		self.removenum=0
		self.deltat0=0.3
		self.hard=np.min([np.max([0,int(self.save[self.saven][0])]),9])
		self.levelcmpt()
		self.root=root
		self.fig=fig
		self.axis=self.fig.axes[0]
		self.fig1=fig1
		self.axis1=self.fig1.axes[0]
		chnum,chname,chnumc,chnamec=chdict
		chnumhgt=np.shape(chnum)[0]
		chnumwdth=np.shape(chnum)[1]
		chnamehgt=np.shape(chname)[0]
		chnamewdth=np.shape(chname)[1]
		self.axis.imshow(chnum,extent=(15-2*chnumwdth/chnumhgt,15+2*chnumwdth/chnumhgt,18,22),cmap=mpl.colors.ListedColormap([chnumc,"#FFFFFF"]))
		self.axis.imshow(chname,extent=(15-5*chnamewdth/chnamehgt,15+5*chnamewdth/chnamehgt,6,16),cmap=mpl.colors.ListedColormap([chnamec,"#FFFFFF"]))
		self.axis1.imshow(chnum,extent=(5-1*chnumwdth/chnumhgt,5+1*chnumwdth/chnumhgt,6,8),cmap=mpl.colors.ListedColormap([chnumc,"#FFFFFF"]))
		self.axis1.imshow(chname,extent=(5-2.*chnamewdth/chnamehgt,5+2.*chnamewdth/chnamehgt,1,5),cmap=mpl.colors.ListedColormap([chnamec,"#FFFFFF"]))
		self.canvas=canvas
		self.canvas1=canvas1
		self.canvas1.draw()
		self.label=tk.Label(frame)
		self.label.grid(row=0,column=0)
		self.label.config(text=' Level : 0 ',font=('serif',20))
		self.ctrl=ctrl
		self.ctrl.config(text='Start',command=self.mainloop,bg='#D5E0EE',activebackground='#E5E35B',font=('serif',20))
		self.quitbutton=quitbutton
		self.quitbutton.config(text='Quit',command=self.quit,bg='#D5E0EE',activebackground='#E5E35B',font=('serif',20))
		self.stopbutton=stopbutton
		self.stopbutton.config(text='Stop',command=self.stop,bg='#D5E0EE',activebackground='#E5E35B',font=('serif',20))
		self.root.title(name)
		self.wall=self.axis.add_patch(plt.Rectangle((6.5,0.5),self.width-13,self.height-1,edgecolor='k',linewidth=5,linestyle='-',fill=False))
		self.wall.zorder=0
		self.mask=plt.Rectangle((0.5,0.5),self.width-1,self.height-1,edgecolor='k',linewidth=5,linestyle='-',facecolor='w')
		self.mask.zorder=3
		self.canvas.draw()
		self.shapebox=[[[[1,0],[1,1],[1,2],[1,3]],[[0,1],[1,1],[2,1],[3,1]],[[1,0],[1,1],[1,2],[1,3]],[[0,1],[1,1],[2,1],[3,1]]],[[[1,0],[1,1],[2,1],[2,2]],[[2,0],[1,0],[1,1],[0,1]],[[1,0],[1,1],[2,1],[2,2]],[[2,0],[1,0],[1,1],[0,1]]],[[[2,0],[2,1],[1,1],[1,2]],[[0,0],[1,0],[1,1],[2,1]],[[2,0],[2,1],[1,1],[1,2]],[[0,0],[1,0],[1,1],[2,1]]],[[[0,0],[1,0],[1,1],[2,0]],[[1,1],[2,1],[2,2],[2,0]],[[0,1],[1,0],[1,1],[2,1]],[[1,0],[1,1],[1,2],[2,1]]],[[[1,0],[1,1],[2,1],[2,0]],[[1,0],[1,1],[2,1],[2,0]],[[1,0],[1,1],[2,1],[2,0]],[[1,0],[1,1],[2,1],[2,0]]],[[[1,0],[1,1],[1,2],[2,2]],[[2,1],[2,0],[0,1],[1,1]],[[1,0],[2,0],[2,1],[2,2]],[[2,0],[1,0],[0,0],[0,1]]],[[[2,1],[0,0],[0,1],[1,1]],[[2,0],[2,1],[1,2],[2,2]],[[0,0],[1,0],[2,0],[2,1]],[[1,2],[2,0],[1,0],[1,1]]]]
		self.colorbox=['#FFFF00','#FF00FF','#00FFFF','#FF0000','#0000FF','#00FF00']
		self.shapelen=len(self.shapebox)
		self.sequencelen=6
		self.direct=0
		self.cubelist={}
		self.shape_sequence=[]
		self.end=False
		self.movemark=False
		self.startmark=False
		self.pressnum=0
		self.stopmark=False
		self.quitmark=False
		self.textmark=True
		self.root.bind('<KeyPress>',self.judge_press)
		self.root.bind('<KeyRelease>',self.judge_release)
	#
	def levelcmpt(self):
		self.level=int(self.removenum/(int(self.hard/2)+1))+1
		self.deltat=(1-self.hard*0.01)**self.level*self.deltat0
	#
	def update_shape_sequence(self):
		shape_type=nr.randint(self.sequencelen)
		direct_type=nr.randint(4)
		if self.level<=3:
			color='#000000'
		elif self.level>8 and self.hard==9:
			color=nr.choice(self.colorbox)
		else:
			color=nr.choice(self.colorbox[:3])
		self.shape_sequence.append([shape_type,direct_type,self.shapebox[shape_type][direct_type],color,[0,0]])
		if len(self.shape_sequence)>self.sequencelen:
			res=self.shape_sequence.pop(0)
		else: res=None
		for i in range(len(self.shape_sequence)):
			self.shape_sequence[i][4]=[25,5*i+1]
			if i==len(self.shape_sequence)-1:
				cubelist=[]
				for k in range(4):
					pos=np.array(self.shape_sequence[i][4])+np.array(self.shape_sequence[i][2][k])
					cubelist.append(self.axis.add_patch(plt.Rectangle(tuple(pos),1,1,edgecolor='w',linewidth=.5,facecolor=self.shape_sequence[i][3])))
					cubelist[-1].zorder=2
				self.shape_sequence[i].append(cubelist)
			else:
				for k in range(4):
					pos=np.array(self.shape_sequence[i][4])+np.array(self.shape_sequence[i][2][k])
					self.shape_sequence[i][5][k].xy=tuple(pos)
		return res
	#
	def reshape(self):
		self.reshapemark=True
		shape_type=self.current_cube[0]
		direct_type=(self.current_cube[1]+1)%4
		cubepos=np.array(self.shapebox[shape_type][direct_type])+np.array(self.current_cube[4])
		if not self.judge_over(cubepos):
			self.current_cube[1]=direct_type
			self.current_cube[2]=self.shapebox[shape_type][direct_type]
			for i in np.arange(4):
				self.current_cube[5][i].xy=tuple(cubepos[i])
		self.imshow()
	#	
	def judge_over(self,cubepos):
		cubemark=False
		for i in np.arange(4):
			if tuple(cubepos[i]) in self.cubelist.keys():
				if not overable(self.current_cube[3],self.cubelist[tuple(cubepos[i])][0]):
					cubemark=True
					break
		wallmark=(np.sum(cubepos[:,1]<=0)+np.sum(cubepos[:,0]<7)+np.sum(cubepos[:,0]>22))!=0
		return cubemark|wallmark
	#
	def move(self):
		if self.quitmark:
			self.root.destroy()
			return
		cubepos=np.array(self.current_cube[2])+np.array(self.current_cube[4])-np.array([[0,1]]*4)
		if self.judge_over(cubepos):
			self.fall()
		else:
			self.current_cube[4][1]-=1
			for i in np.arange(4):
				self.current_cube[5][i].xy=tuple(cubepos[i])
		self.display()
	#
	def trans_move(self,mark):
		if self.quitmark:
			self.root.destroy()
			return
		direct=self.direct
		if direct not in [1,-1]:
			self.pressnum-=1
			return
		if (not mark) and self.pressnum>1:
			self.pressnum-=1
			return
		cubepos=np.array(self.current_cube[2])+np.array(self.current_cube[4])+np.array([[direct,0]]*4)
		if self.judge_over(cubepos):
			pass
		else:
			self.current_cube[4][0]+=direct
			for i in np.arange(4):
				self.current_cube[5][i].xy=tuple(cubepos[i])
		self.imshow()
		if mark:
			self.root.after(200,self.trans_move,False)
		else:
			self.root.after(10,self.trans_move,False)
	#
	def fall(self):
		cubepos=np.array(self.current_cube[2])+np.array(self.current_cube[4])
		if (cubepos[:,1]>=29).sum()==0:
			for i in np.arange(4):
				pos=tuple(cubepos[i])
				if pos in self.cubelist.keys():
					self.axis.patches.remove(self.cubelist[pos][1])
					self.cubelist[pos]=[calcolor(self.current_cube[3],self.cubelist[pos][0]),self.current_cube[5][i]]
				else:
					self.cubelist[pos]=[self.current_cube[3],self.current_cube[5][i]]
			self.eliminate()
			self.current_cube=self.update_shape_sequence()
			self.current_cube[4]=[self.width/2-2,self.height-1]
			cubepos=np.array(self.current_cube[2])+np.array(self.current_cube[4])
			for i in np.arange(4):
				self.current_cube[5][i].xy=tuple(cubepos[i])			
		else:
			self.fail()
			self.end=True
	#
	def eliminate(self):
		cubepos=np.int8([i for i in self.cubelist.keys() if self.cubelist[i][0]=='#000000'])
		mat=np.zeros([16,28])
		mat[tuple(cubepos.T[0]-7),tuple(cubepos.T[1]-1)]=1
		line=np.arange(28)[mat.sum(0)==16]+1
		if not len(line):
			return
		cubepos=[]
		timespace=0.05
		for i in it.product(np.arange(7,23),line):
			cubepos.append(i)
		for i in cubepos:
			self.cubelist[i][1].set_fc('#FFD700')
		self.canvas.draw()
		time.sleep(timespace)
		for i in cubepos:
			self.cubelist[i][1].set_fc('#000000')
		self.canvas.draw()
		time.sleep(timespace)
		for i in cubepos:
			self.cubelist[i][1].set_fc('#FFD700')
		self.canvas.draw()
		time.sleep(timespace)
		for i in cubepos:
			self.axis.patches.remove(self.cubelist.pop(i)[1])
		self.canvas.draw()
		time.sleep(timespace)
		cubepos=np.sort(np.array(list(self.cubelist.keys()),dtype=[('x',int),('y',int)]),order='y')
		for i in cubepos:
			step=(line<i[1]).sum()
			if step:
				self.cubelist[tuple(i)][1].xy=(i[0],i[1]-step)
				self.cubelist[(i[0],i[1]-step)]=self.cubelist.pop(tuple(i))
		self.removenum+=1
	#
	def imshow(self):
		self.levelcmpt()
		if not self.end: 
			try: 
				self.label.config(text=' Level : '+str(self.level)+' ')
			except:
				pass
		if self.cubelist:
			cubepos=list(map(list,np.array(self.current_cube[2])+np.array(self.current_cube[4])))
			for i in np.arange(4):
				pos=tuple(cubepos[i])
				if pos in self.cubelist.keys():
					self.current_cube[5][i].set_fc(calcolor(self.current_cube[3],self.cubelist[pos][0]))
				else:
					self.current_cube[5][i].set_fc(self.current_cube[3])
		self.canvas.draw()
	#
	def display(self):
		self.canvas.draw()
		self.imshow()
		if self.level>self.quitlevel:
			self.end=True
			self.stopbutton.deletecommand(self.stopbutton['command'])
			self.stopbutton.config(text='Next\nChapter',command=self.nextchapter)
			self.axis.text(self.width/2.0,self.height/2.0,'You win!',family='serif', fontsize=40,color='g',verticalalignment='center', horizontalalignment='center')
			self.canvas.draw()
			self.label.config(text=' Level : '+str(self.quitlevel)+' ')
			if int(self.save[self.saven][1])<=2: self.save[self.saven][1]=3
			np.save('save',self.save)
		elif self.end: 
			self.displaying=False
		elif self.stopmark: pass
		elif self.quitmark:
			self.root.destroy()
		elif self.direct==2:
			self.root.after(1,self.move)
		else:
			self.root.after(np.int16(self.deltat*1000),self.move)
	#
	def nextchapter(self):
		self.axis.images.clear()
		self.axis.patches.clear()
		self.axis.texts.clear()
		self.axis1.images.clear()
		self.axis1.patches.clear()
		self.axis1.texts.clear()
		self.canvas.draw()
		self.canvas1.draw()
		self.label.config(text=' Level : 0 ')
		self.nextfunc(self.next)	
	#
	def stop(self):
		if not self.startmark or self.textmark or self.end: pass
		elif self.stopmark:
			self.stopmark=False
			if self.mask in self.axis.patches: 
				self.axis.patches.remove(self.mask)
				self.axis.texts.clear()
			self.stopbutton.config(text='Stop')
			if not self.displaying:
				self.displaying=True
				self.display()
		elif self.stopbutton['text']!='Next\nChapter':
			self.stopmark=True
			self.displaying=False
			self.mask=self.axis.add_patch(self.mask)
			self.axis.text(self.width/2.0,self.height/2.0,'Stopped',family='serif', fontsize=40,color='r',verticalalignment='center', horizontalalignment='center')
			self.stopbutton.config(text='Continue')
	#
	def restart(self):
		if not self.startmark or self.textmark: return
		self.axis.texts.clear()
		self.axis.patches.clear()
		self.axis.add_patch(self.wall)
		self.removenum=0
		self.levelcmpt()
		self.direct=0
		self.cubelist={}
		self.shape_sequence=[]
		self.stopbutton.config(text='Stop',command=self.stop)
		while len(self.shape_sequence)<self.sequencelen:
			self.update_shape_sequence()
		self.current_cube=self.update_shape_sequence()
		self.current_cube[4]=[self.width/2-2,self.height-1]
		cubepos=np.array(self.current_cube[2])+np.array(self.current_cube[4])
		for i in np.arange(4):
			self.current_cube[5][i].xy=tuple(cubepos[i])			
		if self.stopmark:
			self.stop()
		if self.end:
			self.end=False
			# if not self.thread.is_alive():
				# self.thread=th.Thread(target=self.display)
				# self.thread.start()
			if not self.displaying:
				self.displaying=True
				self.display()
		#self.canvas.draw()
	#
	def fail(self):
		self.axis.text(self.width/2.0,self.height/2.0,'You Lose!',family='serif', fontsize=30,color='r',verticalalignment='center', horizontalalignment='center')
	#
	def judge_press(self,event):
		a=event.keysym
		if a=='a' or a=='Left':
			if not self.direct:
				self.direct=-1
				self.pressnum+=1
				self.trans_move(True)
		elif a=='s' or a=='Down': 
			if not self.direct:
				self.direct=2
		elif a=='d' or a=='Right': 
			if not self.direct:
				self.direct=1
				self.pressnum+=1
				self.trans_move(True)
		elif a=='w' or a=='Up': 
			self.reshape()
		elif a=='Return':
			self.mainloop()
		elif a=='space':
			self.stop()
		elif a=='r':
			self.restart()
		elif a=='q':
			self.quit()
	#
	def judge_release(self,event):
		a=event.keysym
		if a=='a' or a=='Left' or  a=='d' or a=='Right' or a=='s' or a=='Down':
			self.direct=0
	#
	def quit(self):
		if self.stopmark or self.end or self.textmark or (not self.startmark):
			self.stopmark=False
			if self.mask in self.axis.patches: 
				self.axis.patches.remove(self.mask)
				self.axis.texts.clear()
			self.stopbutton.config(text='Stop')
			self.root.destroy()
		else: self.quitmark=True
		self.quitfunc()
	#
	def mainloop(self):
		self.axis.images.clear()
		if self.startmark: return
		if self.textmark:
			self.axis.text(self.width/30.0,self.height/2,'This is the chapter \nintroduction.\nYou can use \'wasd\' or \nUp/Down/Left/Right to play.\nPress \'Space\' to stop/continue \nand press \'Q\' to quit.\n\nNow click \'Start\' again \nor press \'Enter\' to start the \ngame.',family='serif', fontsize=13,color='b',verticalalignment='center', horizontalalignment='left')
			self.canvas.draw()
			self.textmark=False
			return
		self.ctrl.config(text='Restart',command=self.restart)
		self.startmark=True
		self.axis.texts.clear()
		while len(self.shape_sequence)<self.sequencelen:
			self.update_shape_sequence()
		self.current_cube=self.update_shape_sequence()
		self.current_cube[4]=[self.width/2-2,self.height-1]
		cubepos=np.array(self.current_cube[2])+np.array(self.current_cube[4])
		for i in np.arange(4):
			self.current_cube[5][i].xy=tuple(cubepos[i])			
		# self.thread=th.Thread(target=self.display)
		# self.thread.start()
		self.displaying=True
		self.display()
#
def calcolor(c1,c2):
	color=np.array(list('000000'))
	calarray=(np.array(list(c1[1:]))!='0')&(np.array(list(c2[1:]))!='0')
	if calarray.sum()==2:
		color[calarray]=['F','F']
	else:
		color[calarray]='F'
	return '#'+''.join(color)
#
def overable(c1,c2):
	calarray=(np.array(list(c1[1:]))=='0')&(np.array(list(c2[1:]))=='0')
	return calarray.sum()<=1
