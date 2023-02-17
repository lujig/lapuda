import numpy as np
import numpy.random as nr
import time
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
mpl.use('TkAgg')  
#
class bricks_class:
    def __init__(self,name,width,height,fig,axis,canvas,fig1,axis1,canvas1,ctrl,quitbutton,stopbutton,frame,root,save,saven,chdict,nextfunc,next,quitfunc):
        self.width=width
        self.height=height
        self.quitlevel=10
        self.nextfunc=nextfunc
        self.next=next
        self.displaying=False
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
        self.axis.imshow(chnum,extent=(15-2*chnumwdth/chnumhgt,15+2*chnumwdth/chnumhgt,18,22),cmap=mpl.colors.ListedColormap([chnumc,'#458B00']))
        self.axis.imshow(chname,extent=(15-5*chnamewdth/chnamehgt,15+5*chnamewdth/chnamehgt,6,16),cmap=mpl.colors.ListedColormap([chnamec,'#458B00']))
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
        self.wall=self.axis.add_patch(plt.Rectangle((0.5,0.5),self.width-1,self.height-1,edgecolor='k',linewidth=5,linestyle='-',fc='#458B00'))
        self.wall.zorder=0
        self.tableedge=1.5
        self.ballradius=0.6
        self.vellim=40
        self.trans_direct=0
        self.platevel=15
        if self.hard==9:
            self.tablewidth=self.height*0.9
            self.pocketradius=self.ballradius*1.4
            self.acc=0.1
            self.platelen0=2
        else:
            self.tablewidth=self.height*(self.hard*0.03+0.47)
            self.pocketradius=self.ballradius*(3.6-self.hard*0.2)
            self.acc=0.01+0.005*self.hard
            self.platelen0=6-self.hard*0.33
        self.tablecorner=[(self.tableedge,self.height-self.tablewidth-self.tableedge),(self.width-self.tableedge,self.height-self.tablewidth-self.tableedge),(self.width-self.tableedge,self.height-self.tableedge),(self.tableedge,self.height-self.tableedge)]
        self.mask=plt.Rectangle((0.5,0.5),self.width-1,self.height-1,edgecolor='k',linewidth=5,linestyle='-',facecolor='w')
        self.mask.zorder=3
        self.canvas.draw()
        self.freshtime=0.02
        self.deltat=0.01
        self.removenum=0
        self.levelcmpt()
        self.colorbox=['#FFFF00','#FF00FF','#00FFFF','#FF0000','#0000FF','#00FF00']
        self.end=False
        self.startmark=False
        self.stopmark=False
        self.quitmark=False
        self.textmark=True
        self.root.bind('<KeyPress>',self.judge_press)
        self.root.bind('<KeyRelease>',self.judge_release)
    #
    def levelcmpt(self):
        if hasattr(self,'level'): tmp=self.level
        else: tmp=0
        if self.removenum<=9:
            self.level=int(self.removenum/3+1)
        else:
            self.level=int((self.removenum-9)/(2+self.hard/9.0))+4
        self.platelen=self.platelen0*(1-0.05*self.level)
        if hasattr(self,'ballvel'):
        	if self.level>tmp: self.ballvel+=1+self.hard*0.1
        if not (tmp==3 and self.level==4): return
        self.table=self.axis.add_patch(plt.Rectangle((self.tableedge,self.height-self.tablewidth-self.tableedge),self.width-2*self.tableedge,self.tablewidth,linewidth=2,edgecolor='#FFD700',facecolor='#458B00'))
        self.table.zorder=1
        self.pocket=[]
        for i in np.arange(4):
            self.pocket.append(self.axis.add_patch(mpl.patches.Wedge(self.tablecorner[i],self.pocketradius,i*90,i*90+90,linewidth=2,edgecolor='#FFD700',facecolor='k')))
            self.pocket[-1].zorder=1
    #
    def ball_move(self):
        ball,vx,vy=self.ball
        x,y=ball.center
        dx=0
        if vx==0 and vy==0: return
        if y<1.4:
            self.fail()
            self.end=True
            return
        xnew=x+vx*self.deltat
        ynew=y+vy*self.deltat
        nball=len(self.billiards)
        vx_old,vy_old=vx,vy
        elim=[]
        for i in np.arange(nball):
            ball0,vx0,vy0=self.billiards[i]
            x0,y0=ball0.center
            d1=np.sqrt((xnew-x0)**2+(ynew-y0)**2)
            d0=np.sqrt((x-x0)**2+(y-y0)**2)
            if d1<2*self.ballradius:
                vxpa,vypa=(vx*(x-x0)+vy*(y-y0))/d0**2*np.array([x-x0,y-y0])
                vxpe,vype=vx-vxpa,vy-vypa
                vx0pa,vy0pa=(vx0*(x-x0)+vy0*(y-y0))/d0**2*np.array([x-x0,y-y0])
                vx,vy=vxpe+vx0pa,vype+vy0pa
                if self.level>3:
                    vx0pe,vy0pe=vx0-vx0pa,vy0-vy0pa
                    vx0,vy0=vx0pe+vxpa,vy0pe+vypa
                    self.billiards[i][1:]=[vx0,vy0]
                else:
                    elim.append(i)
        if elim:
            for i in elim:
                self.axis.patches.remove(self.billiards.pop(i)[0])
                self.removenum+=1
                self.levelcmpt()
                self.plate.set_width(self.platelen)
        if (xnew-1)<self.ballradius or (self.width-1-xnew)<self.ballradius: vx=-vx
        if (self.height-1-ynew)<self.ballradius: vy=-vy
        platex=self.plate.xy[0]+self.platelen/2
        if ynew-1.9<self.ballradius:
            if ynew>1.9:
                if xnew>(platex-self.platelen/2) and xnew<(platex+self.platelen/2):
                    vy=-vy
                    vx+=self.platevel*self.trans_direct
                elif xnew>(platex+self.platelen/2) and np.sqrt((ynew-1.9)**2+(xnew-platex-self.platelen/2)**2)<self.ballradius:
                    vx1=vx-self.platevel*self.trans_direct
                    x0,y0=platex+self.platelen/2,1.9
                    d0=np.sqrt((x-x0)**2+(y-y0)**2)
                    vxpa,vypa=(vx1*(x-x0)+vy*(y-y0))/d0**2*np.array([x-x0,y-y0])
                    vxpe,vype=vx1-vxpa,vy-vypa
                    vx,vy=vxpe-vxpa+self.platevel*self.trans_direct,vype-vypa
                    dx=self.platevel*self.trans_direct*self.deltat*2
                elif xnew<(platex-self.platelen/2) and np.sqrt((ynew-1.9)**2+(xnew-platex+self.platelen/2)**2)<self.ballradius:
                    vx1=vx-self.platevel*self.trans_direct
                    x0,y0=platex-self.platelen/2,1.9
                    d0=np.sqrt((x-x0)**2+(y-y0)**2)
                    vxpa,vypa=(vx1*(x-x0)+vy*(y-y0))/d0**2*np.array([x-x0,y-y0])
                    vxpe,vype=vx1-vxpa,vy-vypa
                    vx,vy=vxpe-vxpa+self.platevel*self.trans_direct,vype-vypa
                    dx=self.platevel*self.trans_direct*self.deltat*2
            elif ynew<1.9 and ynew>1.4:
                if 0<(xnew-platex-self.platelen/2)<self.ballradius or 0>(xnew-platex+self.platelen/2)>-self.ballradius:
                    vx=-vx+self.platevel*2*self.trans_direct
                    dx=self.platevel*self.trans_direct*self.deltat*2
        if vx==vx_old and vy==vy_old:
            ball.center=(xnew,ynew)
        else:
            ball.center=(x+dx,y)
            if np.abs(vy)*10<np.abs(vx):
                vy=np.abs(vx/10)*(2*int(vy>0)-1)
            times=self.ballvel/np.sqrt(vx**2+vy**2)
            vx,vy=vx*times,vy*times
        self.ball=[ball,vx,vy]
    #
    def billiards_move(self):
        nball=len(self.billiards)
        elim=[]
        for i in np.arange(nball):
            ball,vx,vy=self.billiards[i]
            if vx==0 and vy==0: continue
            x,y=ball.center
            mark=0
            for x1,y1 in self.tablecorner:
                if np.sqrt((x-x1)**2+(y-y1)**2)<self.pocketradius: mark=1
            if mark:
                elim.append(i)
            xnew=x+vx*self.deltat
            ynew=y+vy*self.deltat
            vx_old,vy_old=vx,vy
            for k in np.arange(nball):
                if k==i: continue
                ball0,vx0,vy0=self.billiards[k]
                x0,y0=ball0.center
                d1=np.sqrt((xnew-x0)**2+(ynew-y0)**2)
                d0=np.sqrt((x-x0)**2+(y-y0)**2)
                if d1<2*self.ballradius:
                    vxpa,vypa=(vx*(x-x0)+vy*(y-y0))/d0**2*np.array([x-x0,y-y0])
                    vxpe,vype=vx-vxpa,vy-vypa
                    vx0pa,vy0pa=(vx0*(x-x0)+vy0*(y-y0))/d0**2*np.array([x-x0,y-y0])
                    vx0pe,vy0pe=vx0-vx0pa,vy0-vy0pa
                    vx0,vy0=vx0pe+vxpa,vy0pe+vypa
                    vx,vy=vxpe+vx0pa,vype+vy0pa
                    self.billiards[k][1:]=[vx0,vy0]
            ball0,vx0,vy0=self.ball
            x0,y0=ball0.center
            d1=np.sqrt((xnew-x0)**2+(ynew-y0)**2)
            d0=np.sqrt((x-x0)**2+(y-y0)**2)
            if d1<2*self.ballradius:
                vxpa,vypa=(vx*(x-x0)+vy*(y-y0))/d0**2*np.array([x-x0,y-y0])
                vxpe,vype=vx-vxpa,vy-vypa
                vx0pa,vy0pa=(vx0*(x-x0)+vy0*(y-y0))/d0**2*np.array([x-x0,y-y0])
                vx0pe,vy0pe=vx0-vx0pa,vy0-vy0pa
                vx0,vy0=vx0pe+vxpa,vy0pe+vypa
                vx,vy=vxpe+vx0pa,vype+vy0pa
                self.ball[1:]=[vx0,vy0]
                if np.abs(vy0)*10<np.abs(vx0):
                    vy0=np.abs(vx0/10)*(2*int(vy0>0)-1)
                times=self.ballvel/np.sqrt(vx0**2+vy0**2)
                vx0,vy0=vx0*times,vy0*times
                self.ball[1:]=[vx0,vy0]
            if (xnew-self.tableedge)<self.ballradius or (self.width-self.tableedge-xnew)<self.ballradius: vx=-vx
            if (self.height-self.tableedge-ynew)<self.ballradius or (ynew-self.height+self.tableedge+self.tablewidth)<self.ballradius: vy=-vy
            if vx==vx_old and vy==vy_old:
                ball.center=(xnew,ynew)
            vv=np.sqrt(vx**2+vy**2)
            if vv>self.vellim: vvnew=self.vellim
            else: vvnew=max(vv-self.acc,0)
            vx,vy=vx*vvnew/vv,vy*vvnew/vv
            self.billiards[i]=[ball,vx,vy]
        if elim:
            for i in elim:
                self.axis.patches.remove(self.billiards.pop(i)[0])
                self.removenum+=1
                self.levelcmpt()
                self.plate.set_width(self.platelen)
    #
    def trans_move(self):
        if self.trans_direct:
            x,y=self.plate.xy
            xnew=self.platevel*self.trans_direct*self.deltat+x
            if xnew>1 and (xnew+self.platelen)<(self.width-1):
                self.plate.xy=(xnew,y)
    #
    def imshow(self):
        self.levelcmpt()
        if not self.end: 
            try: 
                self.label.config(text=' Level : '+str(self.level)+' ')
            except:
                pass
        self.canvas.draw()
    #
    def display(self):
        self.imshow()
        self.trans_move()
        self.ball_move()
        self.billiards_move()
        if self.level>=self.quitlevel:
            self.end=True
            self.stopbutton.deletecommand(self.stopbutton['command'])
            self.stopbutton.config(text='Next\nChapter',command=self.nextchapter)
            self.axis.text(self.width/2.0,self.height/2.0,'You win!',family='serif', fontsize=40,color='r',verticalalignment='center', horizontalalignment='center')
            self.canvas.draw()
            self.label.config(text=' Level : '+str(self.quitlevel)+' ')
            if int(self.save[self.saven][1])<=2: self.save[self.saven][1]=3
            np.save('save',self.save)
            return
        if self.end or self.stopmark: return
        if self.quitmark:
            self.root.destroy()
            return
        self.root.after(np.int16(self.deltat),self.display)
    #
    def nextchapter(self):
        self.axis.images=[]
        self.axis.patches=[]
        self.axis.texts=[]
        self.axis1.images=[]
        self.axis1.patches=[]
        self.axis1.texts=[]
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
                self.axis.texts=[]
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
        self.axis.texts=[]
        self.axis.patches=[]
        self.axis.add_patch(self.wall)
        self.freshtime=0.02
        self.deltat=0.01
        self.removenum=0
        self.levelcmpt()
        self.stopbutton.config(text='Stop',command=self.stop)
        ball=plt.Circle((self.width/2.0,2.5),self.ballradius,edgecolor='k',facecolor='w')
        plate=plt.Rectangle((self.width/2.0-self.platelen/2.0,1.4),self.platelen,0.5,edgecolor=None,facecolor='k')
        billiards=[]
        self.billiards=[]
        for i in range(10):
            billiards.append([plt.Circle((6+i*2.0,22),self.ballradius,facecolor=self.colorbox[nr.randint(6)],ec='#c0c0c0',linewidth=0.5),0,0])
        for i in range(9):
            billiards.append([plt.Circle((7.+i*2.0,20.27),self.ballradius,facecolor=self.colorbox[nr.randint(6)],ec='#c0c0c0',linewidth=0.5),0,0])
        for i in range(8):
            billiards.append([plt.Circle((8+i*2.0,18.54),self.ballradius,facecolor=self.colorbox[nr.randint(6)],ec='#c0c0c0',linewidth=0.5),0,0])
        self.ball=[self.axis.add_patch(ball),0,0]
        if self.hard==9: self.ballvel=20
        else: self.ballvel=9+self.hard
        self.plate=self.axis.add_patch(plate)
        self.ball[0].zorder=2
        self.plate.zorder=2
        for i in np.arange(len(billiards)):
            self.billiards.append([self.axis.add_patch(billiards[i][0]),billiards[i][1],billiards[i][2]])
            self.billiards[i][0].zorder=2
        if self.ball[1]==0 and self.ball[2]==0:
            theta=np.random.rand()*np.pi/2+np.pi/4
            self.ball[1:3]=[self.ballvel*np.cos(theta),self.ballvel*np.sin(theta)]
        if self.stopmark:
            self.stop()
        if self.end:
            self.end=False
            self.displaying=True
            self.display()
    #
    def fail(self):
        self.axis.text(self.width/2.0,self.height/2.0,'You Lose!',family='serif', fontsize=30,color='r',verticalalignment='center', zorder=5,horizontalalignment='center')
        self.canvas.draw()
    #
    def judge_press(self,event):
        a=event.keysym
        if a=='a' or a=='Left':
            if not self.trans_direct:
                self.trans_direct=-1
                self.trans_move()
        elif a=='d' or a=='Right': 
            if not self.trans_direct:
                self.trans_direct=1
                self.trans_move()
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
        if a=='a' or a=='Left' or  a=='d' or a=='Right':
            self.trans_direct=0
    #
    def quit(self):
        if self.stopmark or self.end or self.textmark or (not self.startmark):
            self.stopmark=False
            if self.mask in self.axis.patches: 
                self.axis.patches.remove(self.mask)
                self.axis.texts=[]
            self.stopbutton.config(text='Stop')
            self.root.destroy()
        else: self.quitmark=True
        self.quitfunc()
    #
    def mainloop(self):
        self.axis.images=[]
        if self.startmark: return
        if self.textmark:
            self.axis.text(self.width/20.0,self.height/2,'This is the chapter \nintroduction.\nYou can use \'ad\' or Left/Right \nto play.\nPress \'Space\' to stop/continue \nand press \'Q\' to quit.\n\nNow click \'Start\' again \nor press \'Enter\' to start the \ngame.',family='serif', fontsize=13,color='k',verticalalignment='center', horizontalalignment='left')
            self.canvas.draw()
            self.textmark=False
            return
        self.ctrl.config(text='Restart',command=self.restart)
        self.startmark=True
        self.axis.texts=[]
        ball=plt.Circle((self.width/2.0,2.5),self.ballradius,edgecolor='k',facecolor='w')
        plate=plt.Rectangle((self.width/2.0-self.platelen/2.0,1.4),self.platelen,0.5,edgecolor=None,facecolor='k')
        billiards=[]
        self.billiards=[]
        for i in range(10):
            billiards.append([plt.Circle((6+i*2.0,22),self.ballradius,facecolor=self.colorbox[nr.randint(6)],ec='#c0c0c0',linewidth=0.5),0,0])
        for i in range(9):
            billiards.append([plt.Circle((7.+i*2.0,20.27),self.ballradius,facecolor=self.colorbox[nr.randint(6)],ec='#c0c0c0',linewidth=0.5),0,0])
        for i in range(8):
            billiards.append([plt.Circle((8+i*2.0,18.54),self.ballradius,facecolor=self.colorbox[nr.randint(6)],ec='#c0c0c0',linewidth=0.5),0,0])
        self.ball=[self.axis.add_patch(ball),0,0]
        if self.hard==9: self.ballvel=20
        else: self.ballvel=9+self.hard
        self.plate=self.axis.add_patch(plate)
        self.ball[0].zorder=2
        self.plate.zorder=2
        for i in np.arange(len(billiards)):
            self.billiards.append([self.axis.add_patch(billiards[i][0]),billiards[i][1],billiards[i][2]])
            self.billiards[i][0].zorder=2
        if self.ball[1]==0 and self.ball[2]==0:
            theta=np.random.rand()*np.pi/2+np.pi/4
            self.ball[1:3]=[self.ballvel*np.cos(theta),self.ballvel*np.sin(theta)]
        self.displaying=True
        self.display()
#
