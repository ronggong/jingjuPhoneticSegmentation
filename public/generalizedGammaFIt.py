'''
 * Copyright (C) 2016  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of jingjuPhoneticSegmentation
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
'''

from scipy.stats import gengamma
import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(1, 1)

a, c = 4.41623854294, 3.11930916792

r = gengamma.rvs(a, c, size=10000)

o1,o2,c1,a1 = gengamma.fit(r)

x = np.linspace(gengamma.ppf(0.01, a1, c1),gengamma.ppf(0.99, a1, c1), 100)

rv = gengamma(a1, c1)

ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)

ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

plt.show()