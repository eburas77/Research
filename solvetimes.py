# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:05:05 2016

@author: ericburas
"""

import numpy as np
import matplotlib.pylab as plt



ptimes = np.array([1.6,2.3,53,480,.33])
stimes = np.array([.51,.966,1.44,64.46,560])
total = ptimes+stimes
x = np.array([0,1,2,3,4])
xt = ('Neural','Metabolic','Protein','Facebook','Power')
plt.xticks(x, xt)
plt.plot(x,ptimes,label = 'Partitioning Times')
plt.ylabel('Time (s)')
plt.savefig('ptimes.png')

plt.xticks(x, xt)
plt.plot(x,stimes,label = 'Solve Times')
plt.ylabel('Time (s)')
plt.savefig('stimes.png')

plt.xticks(x, xt)
plt.plot(x,total,label = 'Total Solve Time')
plt.ylabel('Time (s)')
plt.savefig('total.png')


plt.legend(prop={'size':12})
plt.savefig('solvetimes.png')

x = np.array([0,1,2,3,4])
xt = ('Neural','Metabolic','Protein','Facebook','Power')
plt.xticks(x, xt)
svdtimes = np.array([.0334,.0737,.4183,38.47,72.86])
rmgtimes = np.array([.3292,.7360,.5190,23.11,106.9])
matmulttimes = np.array([.0006,.0036,.0013,.4124,285.1])
solvetimes = np.array([.0023,.0025,.0013,.0099,86.49])
plt.semilogy(x,svdtimes,'-',label = 'SVD')
plt.semilogy(x,rmgtimes,'--',label = 'r x MG')
plt.semilogy(x,matmulttimes,'-.',label = 'Matrix Mult.')
plt.semilogy(x,solvetimes,':',label = 'Matrix Solve')
plt.legend(prop={'size':10},loc = 'upper left')
plt.ylabel('Time (s)')
plt.savefig('operationtimes.png')