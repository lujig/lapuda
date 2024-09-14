# LAPUDA
**LAPUDA**: 简洁的脉冲星数据分析程序包（LAconic Program Units for pulsar Data Analysis）

## Table of Contents
- [编写背景](#编写背景)
- [安装方法](#安装方法)
- [使用方法](#使用方法)
- [维护人员](#维护人员)
- [贡献](#贡献)
	- [作出贡献者](#作出贡献者)
- [许可](#许可)

## 编写背景

FAST已经落成数年，目前能够提供超高信噪比的脉冲星数据。目前FAST脉冲星数据记录格式为搜寻模式的**PSRFITS**格式。在使用这些数据进行脉冲星分析前，需要先对数据进行预处理，比如利用已知的脉冲星星历参数通过消色散与折叠生成单脉冲或子积分格式的数据、利用定标数据对观测数据进行偏振与流量定标、消除数据中的射频干扰、修正数据色散与法拉第旋转、利用脉冲轮廓模板获得脉冲到达时间、拟合脉冲星测时模型等。

上面所列出的数据预处理过程实际上是连贯的，然而通常所用的预处理程序是较为分散的，并且是使用不同的程序编写而成的。因此，在使用这些软件进行预处理时需要核对不同软件之间参数的一致性及兼容性。此外，有部分现有程序是使用C、C++等编译型语言写成，这使得使用者在检查调用预处理程序中的变量时更为困难。为了更加方便地处理FAST脉冲星数据，我们编写了**LAPUDA**。**LAPUDA**使用Python编写而成，其中包含了脉冲星数据预处理的各项功能，并提供了关于时间、空间、脉冲星测时模型的不同可调用模块。

## 安装方法

**LAPUDA**无需进行安装，下载配置好环境后可直接进行使用。

目前程序信赖的软件与Python库包括：

	软件：PSRCAT（可选）

	Python库：psutil，numpy，matplotlib，scipy，astropy，scikit-learn（可选），mpmath（可选）与tkinter

为了获得更好的使用体验，可将**LAPUDA**所在目录加入环境变量**PATH**后再使用。

目前程序支持中文与英文两种语言，默认为中文。使用者可通过修改doc/text.py中的字典**language**的键值，分别更改输出、绘图与界面中的对应语言。

## 使用方法

dfpsr.py: 
	根据脉冲星名（或PSRCAT式脉冲星星历表，或色散量与周期）对搜寻模式的PSRFITS数据进行折叠与消色散。程序将生成LD格式（Laconic Data）的数据，它是由本程序包中的ld.py所定义的一种新的数据格式。LD格式文件可用于保存脉冲星数据与信息，其中数据部分有4个维度（默认为频率通道、子积分、脉冲相位与偏振）。

	dfpsr.py [-f FREQ_RANGE] [-d DM] [-p PERIOD] [-n PSR_NAME] [-e PAR_FILE] [-b NBIN] [-a CAL [CAL ...]] [--cal_period CAL_PERIOD] [-s SUBINT] [-m MULTI] filename [filename ...]

ldcomp.py:
	将LD格式数据按指定的频率通道数、子积分数、脉冲相位点数进行压缩并保存到新的LD文件。

	ldcomp.py [-f NCHAN] [-F] [-t NSUB] [-T] [-b NBIN] [-B] [-P] [-r FREQ_RANGE] [-s SUBINT_RANGE] filename

ldpara.py:
	查看LD格式文件中记录的各参数信息。

	ldpara.py [-c PARAMETER_NAME_LIST] filename

ldplot.py:
	绘制LD格式数据所对应的图像，包括频域、时域、脉冲轮廓、动态谱等。

	ldplot.py [-f] [-t] [-p] [-b PHASE_RANGE] [-r FREQ_RANGE] [-s SUBINT_RANGE] filename

ldcal.py:
	利用周期性的噪声定标数据或其它定标模式的LD格式数据获得记录定标参数的LD格式文件。

	ldcal.py [--cal_period CAL_PERIOD] filename [filename ...]

ldzap.py:
	消除LD格式数据中的射频干扰。

	ldzap.py filename

lddm.py:
	计算LD格式数据对应的最佳色散量。

	lddm.py [-r FREQUENCY] [-s SUBINT] [-n] [-d DM] [-z ZONE] filename

ldconv.py:
	将LD格式数据转换成其它格式。

	ldconv.py [-m MODE] filename

ld.py:
	提供访问LD格式的各种函数。利用这些函数，使用者可以对LD文件进行数据与信息的读写。

time_eph.py:
	提供关于时间标准与空间坐标记录的Python类，可用于计算望远镜在各种参考系中的位置及观测时间点在各种时间标准下的对应值。

psr_read.py:
	提供用于分析读取PSRCAT类脉冲星星历参数表的Python类。

psr_timing.py:
	提供根据脉冲星星历参数、观测时间与频率预测计算脉冲星脉冲相位的Python类。
	
update_cv.py
	用于更新时间标准转换参数与EOPC参数。

### 钟差改正

钟差文件放置于materials/clock文件夹中，每一个钟差文件都是以望远镜名称命名的TXT格式文件。

钟差文件包含两列，第一列是UNIX时间戳（相邻两行的UNIX时间戳之差必需相等），第二列是对应的以秒为单位的时间修正值。钟差中的每一行均为30个字符宽度（包括空格在内），且每一UNIX时间戳均为10个字符宽度。

特别地，FAST钟差可以使用update\_tdiff\_FAST.py程序进行处理获得。使用者可从FAST数据中心下载FAST钟差文件，并将它们置于materials/clock/FAST\_tmp文件夹中。然后运行update\_tdiff\_FAST.py即可在materials/clock文件夹中生成FAST\_poly.txt文件，其中记录了FAST钟差的多项式拟合结果。随后程序即可通过读取该多项式文件计算FAST的对应钟差。此外，如果能够获得FAST与某标准时间的共视时间比对数据，也可运行update\_tdiff\_FAST1.py以生成相应的多项式系数。

## 维护人员

卢吉光[@JiguangLu](mailto:lujig@nao.cas.cn)

## 贡献

包括软件设计、开发、测试与维护。

### 作出贡献者

党世军

刘雨兰

卢吉光

许睿安

王正理

## 许可

免费给予获得本软件和相关文档文件副本的任何人处理本软件————包括使用、复制、修改、合并、发布、分发和再许可本软件副本————的权利，但须遵守以下条件：上述版权声明和本许可声明应包含在本软件的所有副本或主要部分中。
