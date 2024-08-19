#coding=UTF-8
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tkinter as tk
mpl.use('TkAgg')  
import snake
import tetris
import bricks
import dict as di
import os
#
dirname=os.path.split(os.path.realpath(__file__))[0]
#
def test():
    return
#
def delsave(saven):
    global save,s1bttn,s2bttn,s3bttn,titleaxis,ch,titlecanvas
    save[saven]=[0,1,False]
    s1bttn.config(text='Slot1: '+dealsave(b'save1'))
    s2bttn.config(text='Slot2: '+dealsave(b'save2'))
    s3bttn.config(text='Slot3: '+dealsave(b'save3'))
    if np.max([save[b'save1'][1],save[b'save2'][1],save[b'save3'][1]])>10:
        titleaxis.imshow(ch[b'title2'],extent=(0,15,0,10),aspect='auto')
    else: titleaxis.imshow(ch[b'title1'],extent=(0,15,0,10),aspect='auto')
    titleaxis.set_xlim(0,15)
    titleaxis.set_ylim(0,10)
    titlecanvas.draw()
    di.save_write(save)
#
def dealsave(saven):
    global save
    savefile=save[saven]
    if savefile[2]:
        if savefile[1]>10:
            return 'Difficulty '+str(savefile[0])+', cleared!'
        else:
            return 'Difficulty '+str(savefile[0])+', Level '+str(savefile[1])
    else: return 'Empty'
#
def createsave(saven,difnum):
    global save
    save[saven]=[difnum[0],1,True]
    di.save_write(save)
    mainfunc(saven,1)
#
def create_button(ftk,saven,row,column,gamenum):
    global save,chapterlist,chdict
    gamename=chapterlist[gamenum-1]
    chnum=chdict[gamename][0]
    chname=chdict[gamename][1]
    chnumc=chdict[gamename][2]
    chnamec=chdict[gamename][3]
    chnumhgt=np.shape(chnum)[0]
    chnumwdth=np.shape(chnum)[1]
    chnamehgt=np.shape(chname)[0]
    chnamewdth=np.shape(chname)[1]
    btimage=Figure(figsize=(1,1), dpi=150)
    btcanvas=FigureCanvasTkAgg(btimage,master=ftk)
    btcanvas.get_tk_widget().config(bd=0)
    btcanvas.get_tk_widget().grid(row=row,column=column)
    btimage.add_axes([0.05,0.05,0.95,0.95])
    btaxis=btimage.axes[0]
    btaxis.set_xlim(0,10)
    btaxis.set_ylim(0,10)
    btaxis.set_xticks([])
    btaxis.set_yticks([])
    if save[saven][1]>=gamenum:
        btaxis.imshow(chnum,extent=(5-1*chnumwdth/chnumhgt,5+1*chnumwdth/chnumhgt,6,8),cmap=mpl.colors.ListedColormap([chnumc,"#FFFFFF"]))
        btaxis.imshow(chname,extent=(5-2.*chnamewdth/chnamehgt,5+2.*chnamewdth/chnamehgt,1,5),cmap=mpl.colors.ListedColormap([chnamec,"#FFFFFF"]))
        btcanvas.get_tk_widget().bind("<Button-1>",lambda event:mainfunc(saven,gamenum))
    else:
        btaxis.text(5,5,'???',family='serif', fontsize=30,color='r',verticalalignment='center', horizontalalignment='center')
#
def start(saven):
    global stk,ftk,save
    stk.destroy()
    ftk=tk.Tk()
    ftk.resizable(False,False)
    if save[saven][2]:
        ftk.title('select chapter')
        tk.Label(ftk,text='Pls select the Chapter (Diff : '+str(save[saven][0])+'): ',font=('serif',20)).grid(row=0,column=0,columnspan=3)
        create_button(ftk,saven,row=1,column=0,gamenum=1)
        create_button(ftk,saven,row=1,column=1,gamenum=2)
        create_button(ftk,saven,row=1,column=2,gamenum=3)
        create_button(ftk,saven,row=2,column=0,gamenum=4)
        create_button(ftk,saven,row=2,column=1,gamenum=5)
        create_button(ftk,saven,row=2,column=2,gamenum=6)
        create_button(ftk,saven,row=3,column=0,gamenum=7)
        create_button(ftk,saven,row=3,column=1,gamenum=8)
        create_button(ftk,saven,row=3,column=2,gamenum=9)
        create_button(ftk,saven,row=4,column=1,gamenum=10)
    else:
        ftk.title('select difficulty')
        tk.Label(ftk,text='Please select the difficulty: ',font=('serif',20)).grid(row=0,column=0)
        dif=tk.Listbox(ftk,selectmode=tk.BROWSE,font=('serif',20))
        dif.grid(row=1,column=0)
        dif.insert(0,u'0    平地打滚的难度','1    步行的难度','2    骑自行车的难度','3    开车的难度','4    开飞机的难度','5    开火箭的难度','6    开时光机的难度','7    超出想象的难度','8    编程的难度','9    Why so serious?')
        dif.select_set(0)
        tk.Button(ftk,text='Start Game',command=lambda :createsave(saven,dif.curselection()),font=('serif',20)).grid(row=2,column=0)
        ftk.bind('<KeyPress-Return>',lambda event:createsave(saven,dif.curselection()))
        ftk.mainloop()
#
def chapter_select(chapternum,width,height,fig,axis,canvas,fig1,axis1,canvas1,ctrl,quitbutton,stopbutton,frame,root,saven):
    global img,save,chapterlist,chdict
    chaptername=chapterlist[chapternum-1]
    def gameselect(chaptername):
        if chaptername=='snake':
            snake.snake_class('snake',width,height,fig,axis,canvas,fig1,axis1,canvas1,ctrl,quitbutton,stopbutton,frame,root,save,saven,bombimg=img[b'bomb'],bangimg=img[b'bang'],chdict=chdict['snake'],nextfunc=gameselect,next='tetris',quitfunc=maingame)
        elif chaptername=='tetris':
            tetris.tetris_class('tetris',width,height,fig,axis,canvas,fig1,axis1,canvas1,ctrl,quitbutton,stopbutton,frame,root,save,saven,chdict=chdict['tetris'],nextfunc=gameselect,next='bricks',quitfunc=maingame)
        elif chaptername=='bricks':
            bricks.bricks_class('bricks',width,height,fig,axis,canvas,fig1,axis1,canvas1,ctrl,quitbutton,stopbutton,frame,root,save,saven,chdict=chdict['bricks'],nextfunc=gameselect,next=0,quitfunc=maingame)
        else:
            test()
    gameselect(chaptername)
#
def mainfunc(saven,chapternum):
    global ftk,save
    ftk.destroy()
    root=tk.Tk()
    root.resizable(False,False)
    width=30
    height=30
    fig=Figure(figsize=(3,3), dpi=200)
    fig.add_axes([0,0,1,1])
    axis=fig.axes[0]
    axis.set_xlim(0,width)
    axis.set_ylim(0,height)
    axis.set_xticks([])
    axis.set_yticks([])
    fig1=Figure(figsize=(0.5,0.5), dpi=200)
    fig1.add_axes([0,0,1,1])
    axis1=fig1.axes[0]
    axis1.set_xlim(0,10)
    axis1.set_ylim(0,10)
    axis1.set_xticks([])
    axis1.set_yticks([])
    canvas=FigureCanvasTkAgg(fig,master=root)
    canvas.get_tk_widget().grid(row=0,column=0,rowspan=12)  
    canvas1=FigureCanvasTkAgg(fig1,master=root)
    canvas1.get_tk_widget().grid(row=0,column=1,rowspan=2)
    tk.Label(root,text='Diff : '+str(save[saven][0]),font=('serif',20)).grid(row=2,column=1)
    frame=tk.Frame(root)
    frame.grid(row=3,column=1,rowspan=2)
    ctrl=tk.Button(root)
    quitbutton=tk.Button(root)
    quitbutton.grid(row=9,column=1,rowspan=2)  
    stopbutton=tk.Button(root)
    ctrl.grid(row=5,column=1,rowspan=2)
    stopbutton.grid(row=7,column=1,rowspan=2)
    chapter_select(chapternum,width,height,fig,axis,canvas,fig1,axis1,canvas1,ctrl,quitbutton,stopbutton,frame,root,saven)
    root.mainloop()
#
def maingame():
    global img,save,chapterlist,chdict,stk,s1bttn,s2bttn,s3bttn,titleaxis,ch,titlecanvas
    img=di.dict_load(dirname+'/src/img.dat')
    ch=di.dict_load(dirname+'/src/ch.dat')
    save=di.save_load()
    chapterlist=['snake','tetris','bricks',4,5,6,7,8,9,10]
    chdict={}
    chdict['snake']=[ch[b'ch1'],ch[b'lejiuguo'],"#000000","#FF0000"]
    chdict['tetris']=[ch[b'ch2'],ch[b'liujinguan'],"#000000","#FF0000"]
    chdict['bricks']=[ch[b'ch3'],ch[b'lujiaoguan'],"#000000","#FF0000"]
    chdict[4]=[ch[b'ch4'],ch[b'linjiangge'],"#000000","#FF0000"]
    chdict[5]=[ch[b'ch5'],ch[b'lingjiangong'],"#000000","#FF0000"]
    chdict[6]=[ch[b'ch6'],ch[b'luanjungang'],"#000000","#FF0000"]
    chdict[7]=[ch[b'ch7'],ch[b'longjinggu'],"#000000","#FF0000"]
    chdict[8]=[ch[b'ch8'],ch[b'langjuangang'],"#000000","#FF0000"]
    chdict[9]=[ch[b'ch9'],ch[b'luojigou'],"#000000","#FF0000"]
    chdict[10]=[ch[b'ch10'],ch[b'lianjuege'],"#000000","#FF0000"]
    #
    stk=tk.Tk()
    stk.wm_attributes('-topmost',1)
    stk.resizable(False,False)
    stk.title('select save')
    sframe=tk.Frame(stk,bg='white')
    sframe.grid(row=0,column=0)
    titleimage=Figure(figsize=(3,2), dpi=200,facecolor='white')
    titlecanvas=FigureCanvasTkAgg(titleimage,master=sframe)
    titlecanvas.get_tk_widget().config(bd=0,highlightthickness=0)
    titlecanvas.get_tk_widget().grid(row=0,column=0,columnspan=2)
    titleimage.add_axes([0,0,1,1])
    titleaxis=titleimage.axes[0]
    titleaxis.axis('off')
    if np.max([save[b'save1'][1],save[b'save2'][1],save[b'save3'][1]])>10:
        titleaxis.imshow(ch[b'title2'],extent=(0,15,0,10),aspect='auto')
    else: titleaxis.imshow(ch[b'title1'],extent=(0,15,0,10),aspect='auto')
    titleaxis.set_xlim(0,15)
    titleaxis.set_ylim(0,10)
    titlecanvas.draw()
    s1bttn=tk.Button(sframe,text='Slot1: '+dealsave(b'save1'),command=lambda:start(b'save1'),bg='#D5E0EE',activebackground='#E5E35B',font=('serif',20))
    s1bttn.grid(row=1,column=0)
    s2bttn=tk.Button(sframe,text='Slot2: '+dealsave(b'save2'),command=lambda:start(b'save2'),bg='#D5E0EE',activebackground='#E5E35B',font=('serif',20))
    s2bttn.grid(row=2,column=0)
    s3bttn=tk.Button(sframe,text='Slot3: '+dealsave(b'save3'),command=lambda:start(b'save3'),bg='#D5E0EE',activebackground='#E5E35B',font=('serif',20))
    s3bttn.grid(row=3,column=0)
    tk.Button(sframe,text='Delete',command=lambda:delsave(b'save1'),bg='#D5E0EE',activebackground='#FF0000',font=('serif',20)).grid(row=1,column=1)
    tk.Button(sframe,text='Delete',command=lambda:delsave(b'save2'),bg='#D5E0EE',activebackground='#FF0000',font=('serif',20)).grid(row=2,column=1)
    tk.Button(sframe,text='Delete',command=lambda:delsave(b'save3'),bg='#D5E0EE',activebackground='#FF0000',font=('serif',20)).grid(row=3,column=1)
    stk.mainloop()
maingame()
