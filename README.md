# Jingju phonetic segmentation
Segmentation of jingju (Beijing opera) singing syllables into phonetic units  

This repo is for the review of "A CLASSIFICATION METHOD FOR PHONETIC SEGMENTATION OF JINGJU SINGING SYLLABLE, Rong Gong, Xavier Serra"  

## Annotation units

1.This table shows the annotation units used in 'pinyin' and 'details' tiers of each .\dataset\textgrid\\*.TextGrid  

2.Chinese pyin and X-SAMPA format are given. 

3.b,p,d,t,k,j,q,x,zh,ch,sh,z,c,s initials are grouped into one representation (not X-SAMPA): c  

4.v,N,J (X-SAMPA) are three special pronunciations which do not exist in pinyin.  

<dl>
<table>
  <tr>
    <th></th>
    <th>Structure</th>
    <th>Pinyin[X-SAMPA]</th>
  </tr>
  <tr>
    <td rowspan="2">head</td>
    <td>initials</td>
    <td>m[m], f[f], n[n], l[l], g[k], h[x], r[r\'], y[j], w[w],<br>{b, p, d, t, k, j, q, x, zh, ch, sh, z, c, s} - group [c]<br>[v], [N], [J] - special pronunciations</td>
  </tr>
  <tr>
    <td>medial vowels</td>
    <td>i[i], u[u], ü[y]</td>
  </tr>
  <tr>
    <td rowspan="4">belly</td>
    <td>simple finals</td>
    <td>a[a"], o[O], e[7], ê[E], i[i], u[u], ü[y],<br>i (zhi,chi,shi) [1], i (ci,ci,si) [M],</td>
  </tr>
  <tr>
    <td>compound finals</td>
    <td>ai[aI^], ei[eI^], ao[AU^], ou[oU^]</td>
  </tr>
  <tr>
    <td>nasal finals</td>
    <td>an[an], en[@n], in[in],<br>ang[AN], eng[7N], ing[iN], ong[UN]</td>
  </tr>
  <tr>
    <td>retroflexed finals</td>
    <td>er [@][r\']</td>
  </tr>
  <tr>
    <td>tail</td>
    <td></td>
    <td>i[i], u[u], n[n], ng[N]</td>
  </tr>
</table>
</dl>

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
