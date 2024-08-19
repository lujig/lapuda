import numpy as np
import numpy.random as nr
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
mpl.use('TkAgg')  
#
class snake_class:
	def __init__(self,name,width,height,fig,axis,canvas,fig1,axis1,canvas1,ctrl,quitbutton,stopbutton,frame,root,save,saven,bombimg,bangimg,chdict,nextfunc,next,quitfunc):
		self.width=width
		self.height=height
		self.quitlevel=10
		self.bombimg=bombimg
		self.bangimg=bangimg
		self.nextfunc=nextfunc
		self.next=next
		self.quitfunc=quitfunc
		self.save=save
		self.saven=saven
		self.hard=np.min([np.max([0,int(self.save[self.saven][0])]),9])
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
		self.wall=self.axis.add_patch(plt.Rectangle((0.5,0.5),self.width-1,self.height-1,edgecolor='k',linewidth=5,linestyle='-',fill=False))
		self.mask=plt.Rectangle((0.5,0.5),self.width-1,self.height-1,edgecolor='k',linewidth=5,linestyle='-',facecolor='w')
		self.mask.zorder=3
		self.canvas.draw()   
		self.deltat0=0.3
		self.pos=[[self.width/2,self.height/2+1],[self.width/2,self.height/2],[self.width/2,self.height/2-1]]
		self.direct=[0,1]
		self.direct0=[0,1]
		if self.hard==9:
			self.safe=2
		else:
			self.safe=4
		self.cube={}
		self.bomb={}
		self.levelcmpt()
		self.end=False
		self.startmark=False
		self.stopmark=False
		self.quitmark=False
		self.textmark=True
		self.root.bind('<KeyPress>',self.judge)
	#
	def levelcmpt(self):
		self.level=int((len(self.pos)-3)/(int(self.hard/3)+4))+1
		self.deltat=(0.9-self.hard*0.015)**self.level*self.deltat0
	#
	def move(self):
		if self.quitmark:
			self.root.destroy()
			return
		self.direct0=[self.direct[0],self.direct[1]]
		newx=self.pos[0][0]+self.direct[0]
		newy=self.pos[0][1]+self.direct[1]
		if newx>0 and newx<self.width-1 and newy>0 and newy<self.height-1 :
			if ([newx,newy] not in self.pos):
				if (newx,newy) in self.bomb.keys():
					self.fail('bomb')
					self.axis.imshow(self.bangimg,extent=(newx-1,newx+2,newy-1,newy+2))
					self.axis.images.remove(self.bomb.pop((newx,newy))[1])
					self.end=True
				else:
					if (newx,newy) in self.cube.keys():
						self.axis.patches.remove(self.cube.pop((newx,newy)))
						self.newcube()
					else:
						self.pos.pop()
						self.axis.patches.remove(self.snakecube.pop())
					self.pos.insert(0,[newx,newy])
					self.snakecube.insert(0,self.axis.add_patch(plt.Rectangle((newx,newy),1,1,edgecolor='k',linewidth=.5,facecolor='b')))
					self.head.center=(newx+0.5,newy+0.5)
					self.head.zorder=2
					self.head.stale=True
			else:
				self.fail('self')
				self.end=True
		else:
			self.fail('wall')
			self.end=True
		self.display()
	#
	def gencube(self):
		newx=nr.randint(self.width-2)+1
		newy=nr.randint(self.height-2)+1
		while ([newx,newy] in self.pos) or ((newx,newy) in self.cube.keys()) or ((newx,newy) in list(self.bomb.keys())):
			newx=nr.randint(self.width-2)+1
			newy=nr.randint(self.height-2)+1
		if self.level>2 and np.abs(newx-self.pos[0][0])+np.abs(newy-self.pos[0][1])>self.safe and nr.rand()>0.5:
			self.bomb[(newx,newy)]=[False,self.axis.add_patch(plt.Rectangle((newx,newy),1,1,edgecolor='k',linewidth=.5,facecolor='y'))]
		else:
			self.cube[(newx,newy)]=self.axis.add_patch(plt.Rectangle((newx,newy),1,1,edgecolor='k',linewidth=.5,facecolor='y'))
	#
	def newcube(self):
		num=self.level/2+1
		if len(self.bomb)>0: 
			lenbomb=len(self.bomb)-np.array(list(self.bomb.values()))[:,0].sum()
		else: lenbomb=0
		lencube=lenbomb+len(self.cube)
		if (num<lencube+1) or (not self.startmark) or (lencube>=self.level-1): self.gencube()
		else:
			for i in np.arange(1,nr.randint(2)+2):
				self.gencube()
	#
	def display(self):
		self.levelcmpt()
		if not self.end: self.label.config(text=' Level : '+str(self.level)+' ')
		for i in list(self.bomb.keys()):
			if np.abs(i[0]-self.pos[0][0])+np.abs(i[1]-self.pos[0][1])<=self.safe and not self.bomb[i][0]:
				self.newcube()
				self.bomb[i][0]=True
				self.axis.patches.remove(self.bomb[i][1])
				self.bomb[i][1]=self.axis.imshow(self.bombimg,extent=(i[0],i[0]+1,i[1],i[1]+1),cmap='Greys_r')
		self.canvas.draw()
		if self.level>self.quitlevel:
			self.end=True
			self.stopbutton.deletecommand(self.stopbutton['command'])
			self.stopbutton.config(text='Next\nChapter',command=self.nextchapter)
			self.axis.text(self.width/2.0,self.height/2.0,'You win!',family='serif', fontsize=40,color='g',verticalalignment='center', horizontalalignment='center')
			self.canvas.draw()
			self.label.config(text=' Level : '+str(self.quitlevel)+' ')
			if int(self.save[self.saven][1])<=1: self.save[self.saven][1]=2
			np.save('save',self.save)
			return
		if self.end or self.stopmark: return
		if self.quitmark:
			self.root.destroy()
			return
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
		if not self.startmark or self.textmark or self.end: return
		if self.stopmark:
			self.stopmark=False
			if self.mask in self.axis.patches: 
				self.axis.patches.remove(self.mask)
				self.axis.texts.clear()
			self.stopbutton.config(text='Stop')
			self.display()
		elif self.stopbutton['text']!='Next\nChapter':
			self.stopmark=True
			self.axis.add_patch(self.mask)
			self.axis.text(self.width/2.0,self.height/2.0,'Stopped',family='serif', fontsize=40,color='r',verticalalignment='center', horizontalalignment='center')
			self.stopbutton.config(text='Continue')
	#
	def restart(self):
		if not self.startmark or self.textmark: return
		self.axis.texts.clear()
		self.axis.patches.clear()
		self.pos=[[self.width/2,self.height/2+1],[self.width/2,self.height/2],[self.width/2,self.height/2-1]]
		self.head=self.axis.add_patch(plt.Circle((self.pos[0][0]+0.5,self.pos[0][1]+0.5),0.75,linewidth=0,facecolor='r'))
		self.head.zorder=2
		self.snakecube=[]
		self.axis.add_patch(self.wall)
		for i in self.pos:
			self.snakecube.append(self.axis.add_patch(plt.Rectangle((i[0],i[1]),1,1,edgecolor='k',linewidth=.5,facecolor='b')))
		self.direct=[0,1]
		self.direct0=[0,1]
		self.cube={}
		self.bomb={}
		self.axis.images.clear()
		self.levelcmpt()
		self.newcube()
		self.stopbutton.config(text='Stop',command=self.stop)
		if self.stopmark:
			self.stop()
		if self.end:
			self.end=False
			self.display()
	#
	def fail(self,info):
		if info=='self':
			self.axis.text(self.width/2.0,self.height/2.0,'You hit yourself!',family='serif', fontsize=30,color='r',verticalalignment='center', horizontalalignment='center')
		elif info=='wall':
			self.axis.text(self.width/2.0,self.height/2.0,'You hit wall!',family='serif', fontsize=30,color='r',verticalalignment='center', horizontalalignment='center')
		elif info=='bomb':
			self.axis.text(self.width/2.0,self.height/2.0,'You are bombed!',family='serif', fontsize=30,color='r',verticalalignment='center', horizontalalignment='center')
	#
	def judge(self,event):
		a=event.keysym
		if a=='a' or a=='Left': 
			if self.direct0!=[1,0]: self.direct=[-1,0]
		elif a=='s' or a=='Down': 
			if self.direct0!=[0,1]: self.direct=[0,-1]
		elif a=='d' or a=='Right': 
			if self.direct0!=[-1,0]: self.direct=[1,0]
		elif a=='w' or a=='Up': 
			if self.direct0!=[0,-1]: self.direct=[0,1]
		elif a=='Return':
			self.mainloop()
		elif a=='space':
			self.stop()
		elif a=='r':
			self.restart()
		elif a=='q':
			self.quit()
	#
	def quit(self):
		if self.stopmark or self.end or self.textmark or not self.startmark:
			self.stop()
			self.root.destroy()
		else: self.quitmark=True
		self.quitfunc()
	#
	def mainloop(self):
		self.axis.images.clear()
		if self.startmark: return
		if self.textmark:
			self.axis.text(self.width/20.0,self.height/2,'This is the chapter \nintroduction.\nYou can use \'wasd\' or \nUp/Down/Left/Right to play.\nPress \'Space\' to stop/continue \nand press \'Q\' to quit.\n\nNow click \'Start\' again \nor press \'Enter\' to start the \ngame.',family='serif', fontsize=13,color='k',verticalalignment='center', horizontalalignment='left')
			self.canvas.draw()
			self.textmark=False
			return
		self.ctrl.config(text='Restart',command=self.restart)
		self.startmark=True
		self.axis.texts.clear()
		self.head=self.axis.add_patch(plt.Circle((self.pos[0][0]+0.5,self.pos[0][1]+0.5),0.75,linewidth=0,facecolor='r'))
		self.head.zorder=2
		self.snakecube=[]
		for i in self.pos:
			self.snakecube.append(self.axis.add_patch(plt.Rectangle((i[0],i[1]),1,1,edgecolor='k',linewidth=.5,facecolor='b')))
		self.newcube()
		self.display()
#
