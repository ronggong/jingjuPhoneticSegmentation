# Jingju phonetic segmentation
Segmentation of jingju (Beijing opera) singing syllables into phonetic units  

This repo is for the review of "A CLASSIFICATION METHOD FOR PHONETIC SEGMENTATION OF JINGJU SINGING SYLLABLE, Rong Gong, Xavier Serra"  

## Please see this at first

To use this code, please do as follows:

1.Make sure install lfs for git before cloning this repo, because the SVM model contains large file. https://git-lfs.github.com/  

2.Download the audio dataset in https://github.com/CompMusic/jingju-lyrics-annotation/tree/master/annotations/3folds/ for fold1, fold2 and fold3. Put all audios (without directories) into ./dataset/wav  

3.You have to generate and save the feature files for each audio. The method is written in demo.py  

## Usage

Please refer to demo.py for:  

1.running the speech phonetic segmentation algorithms of Aversano, Hoang and Winebarger.  

2.reproduce the result of the pattern classification method (proposed).  

The results of Aversano and Hoang can be plot by figurePlot.py

## Dependencies

Numpy  
Scipy  
Sklearn==0.17.1  
Essentia==2.0.1  
Matplotlib

All dependencies of other versions have not been tested.

## Contact

email: rong.gong@upf.edu  

* Copyright (C) 2016  Music Technology Group - Universitat Pompeu Fabra  
*  
* pypYIN is free software: you can redistribute it and/or modify it under
* the terms of the GNU Affero General Public License as published by the Free
* Software Foundation (FSF), either version 3 of the License, or (at your
* option) any later version.  
*  
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
* details.  
*  
* You should have received a copy of the Affero GNU General Public License
* version 3 along with this program.  If not, see http://www.gnu.org/licenses/  
*  
* If you have any problem about this python version code, please contact: Rong Gong
* rong.gong@upf.edu  
*  
*  
* If you want to refer this code, please use this article:  
*  
* A CLASSIFICATION METHOD FOR PHONETIC SEGMENTATION OF JINGJU SINGING SYLLABLE,  
* Rong Gong, Xavier Serra
