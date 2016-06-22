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