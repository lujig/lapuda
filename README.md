# LAPUDA
**LAPUDA**: LAconic Program Units for pulsar Data Analysis

## Table of Contents
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
	- [Contributors](#contributors)
- [License](#license)

## Background

FAST has been under construction for 5 years, and has the potential to provide large quantity pulsar data with high signal-to-noise ratio. The FAST pulsar data format is the search-mode unfold **PSRFITS**. To use these data, we have to do the pre-process on the raw data, such as de-dispersing and folding the data to single pulse mode or sub-integration mode data with known pulsar ephemeris parameters, calibrating the data on polarization and flux with calibration data, remove the radio-frequency interference automatically or interactively, modify the dispersion measure and rotation measure based on the data, obtain timing of arrival with the template, and fitting timing model with ToAs (time of arrival).

The pre-processes listed above are coherent in one continuous line. However, the commonly used software for pre-processing is scattered, and written in different programming languages. As a result, the users need to note and reconcile the parameters used in data-processing with different software. Additionally, some of the software is written in compiled languages such as C and C++, making it difficult for users to check or call the variables in pre-processing. To pre-process FAST pulsar data more conveniently, the program units **LAPUDA** (LAconic Program Units for pulsar Data Analysis) was written. **LAPUDA** is a software written in Python that implements the integrated pre-processing for pulsar data (not only for FAST data), and provides callable modules for the time-space information and pulsar timing model.

## Install

The software LAPUDA can be used directly without install, of course, with some basic dependent packages installed. 

Dependence: 

	Software: PSRCAT (optional)

	Python module: psutil, numpy, matplotlib, scipy, astropy, scikit-learn (optional), mpmath (optional) and tkinter

For a better experience, the user can add the programme directory to the environment variable **PATH**, alternatively. 

## Usage

dfpsr.py: 
	Dedisperse and fold the search mode psrfits file with pulsar name (or PSRCAT ephemris file, or DM and period). The folded file is saved as LD format, which is a new format to save pulsar data and information. The data in LD file always have 4 dimensions (nchan, nsub, nbin, npol).

	dfpsr.py [-f FREQ_RANGE] [-d DM] [-p PERIOD] [-n PSR_NAME] [-e PAR_FILE] [-b NBIN] [-a CAL [CAL ...]] [--cal_period CAL_PERIOD] [-s SUBINT] [-m MULTI] filename [filename ...]

ldcomp.py:
	Compress the ld format file with given nchan, nsub, nbin, and save resutls in a new LD file.

	ldcomp.py [-f NCHAN] [-F] [-t NSUB] [-T] [-b NBIN] [-B] [-P] [-r FREQ_RANGE] [-s SUBINT_RANGE] filename

ldpara.py:
	View the information of LD format file.

	ldpara.py [-c PARAMETER_NAME_LIST] filename

ldplot.py:
	Plot the time-domain or frequency-domain image or pulse profile of a LD file.

	ldplot.py [-f] [-t] [-p] [-b PHASE_RANGE] [-r FREQ_RANGE] [-s SUBINT_RANGE] filename

ldcal.py:
	Obtain the calibration LD file with periodic noise fits files or calibration LD file fragments.

	ldcal.py [--cal_period CAL_PERIOD] filename [filename ...]

ldzap.py:
	Zap the frequency domain interference in LD file.

	ldzap.py filename

lddm.py:
	Calculate the best DM value for LD file.

	lddm.py [-r FREQUENCY] [-s SUBINT] [-n] [-d DM] [-z ZONE] filename

ldconv.py:
	Convert the LD format file to other format.

	ldconv.py [-m MODE] filename

ld.py:
	Provide some functions to access LD (Laconic Data) format data. With these functions, one can read data and information of ld file, or write data and information into an LD file.

time_eph.py:
	Provide a class to calculate the position of FAST, the ephemris of the solar system, and to convert the time between different time standard.

psr_read.py:
	Provide a class to obtain the ephemris of the pulsar based on the PSRCAT.

psr_timing.py:
	Provide a class to calculate the pulse phase of a pulsar with a given time and a given frequency.
	
update_cv.py
	Update the time-depending infomation such like the clock correction data files.

## Maintainers

[@JiguangLu](mailto:lujig@nao.cas.cn)

## Contributing

Including the programme design, development, testing, maintenance.

### Contributors

Shijun Dang

Jui-An Hsu

Yulan Liu

Jiguang Lu

Zhengli Wang

## License

[MIT](LICENSE)
