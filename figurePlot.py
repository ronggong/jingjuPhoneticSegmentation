import numpy as np
import matplotlib.pyplot as plt

'''
####---- plot hoang
csv_filename = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/hoa_mfccBands.csv'

data         = np.loadtxt(csv_filename,delimiter=',')

fn  = 0     # figure number l
ln  = 1     # line number h1
x  = 2     # h2

HR  = 3
OS  = 4
FAR = 5
F   = 6
R   = 7
deletion    = 8
insertion   = 9

l_to_choose = [2,4,6,8,10]
h1_to_choose = [0.6,0.8,1.0]

for l_chosen in l_to_choose:
    data_l  = data[data[:,fn]==l_chosen,:]

    data_h1 = []
    for h1 in h1_to_choose:
        h1_mat = data_l[data_l[:,ln]==h1,:]
        data_h1.append(h1_mat)

    f, axarr = plt.subplots(4, sharex=True)
    for ii, h1_mat in enumerate(data_h1):
        axarr[0].plot(h1_mat[:,x],h1_mat[:,HR],label='h1='+str(0.6+ii*0.2))

    axarr[0].set_ylabel('HR')
    axarr[0].legend()
    axarr[0].set_title('l='+str(l_chosen))

    for ii, h1_mat in enumerate(data_h1):
        axarr[1].plot(h1_mat[:,x],h1_mat[:,OS])

    axarr[1].set_ylabel('OS')

    for ii, h1_mat in enumerate(data_h1):
        axarr[2].plot(h1_mat[:,x],h1_mat[:,F])

    axarr[2].set_ylabel('F')

    for ii, h1_mat in enumerate(data_h1):
        axarr[3].plot(h1_mat[:,x],h1_mat[:,R])
    axarr[3].set_ylabel('R')
    axarr[3].set_xlabel('h2')


    plt.show()
'''

'''
####---- plot avesano
csv_filename = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/ave_bark.csv'

data         = np.loadtxt(csv_filename,delimiter=',')

fn  = 0 # a figure number
x   = 1 # b
ln  = 2 # c line number

HR  = 3
OS  = 4
FAR = 5
F   = 6
R   = 7
deletion    = 8
insertion   = 9

a_to_choose = [1,2,3,4]
c_to_choose = [3,5,7,9]

for a_chosen in a_to_choose:
    data_a  = data[data[:,fn]==a_chosen,:]
    data_c  = []
    for c in c_to_choose:
        c_mat = data_a[data_a[:,ln]==c,:]
        data_c.append(c_mat)

    f, axarr = plt.subplots(4, sharex=True)
    for ii, c_mat in enumerate(data_c):
        axarr[0].plot(c_mat[:,x],c_mat[:,HR],label=('c='+str((ii+1)*2+1)))

    axarr[0].set_ylabel('HR')
    axarr[0].legend()
    axarr[0].set_title('a='+str(a_chosen))

    for ii, c_mat in enumerate(data_c):
        axarr[1].plot(c_mat[:,x],c_mat[:,OS])


    axarr[1].set_ylabel('OS')

    for ii, c_mat in enumerate(data_c):
        axarr[2].plot(c_mat[:,x],c_mat[:,F])

    axarr[2].set_ylabel('F')

    for ii, c_mat in enumerate(data_c):
        axarr[3].plot(c_mat[:,x],c_mat[:,R])

    axarr[3].set_ylabel('R')
    axarr[3].set_xlabel('c')


    plt.show()
'''


####---- plot Win
csv_filename = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/win_plpcc.csv'

data         = np.loadtxt(csv_filename,delimiter=',')

ln          = 0 # h2
fn          = 1 # alpha
x           = 2 # lambda

HR  = 3
OS  = 4
FAR = 5
F   = 6
R   = 7
deletion    = 8
insertion   = 9

h2_to_choose    = [0.0,0.02,0.04,0.06]
alpha_to_choose = [0.2,0.4,0.6,0.8,1.0]

for a_chosen in alpha_to_choose:
    data_a  = data[data[:,fn]==a_chosen,:]

    data_h2   = []
    for h2 in h2_to_choose:
        h2_mat = data_a[data_a[:,ln]==h2,:]
        data_h2.append(h2_mat)

    f, axarr = plt.subplots(4, sharex=True)
    for ii,h2_mat in enumerate(data_h2):
        axarr[0].plot(h2_mat[:,x],h2_mat[:,HR],label=('h2='+str(ii*0.02)))

    axarr[0].set_ylabel('HR')
    axarr[0].legend()
    axarr[0].set_title('alpha='+str(a_chosen))

    for ii, h2_mat in enumerate(data_h2):
        axarr[1].plot(h2_mat[:,x],h2_mat[:,OS])
    axarr[1].set_ylabel('OS')

    for ii, h2_mat in enumerate(data_h2):
        axarr[2].plot(h2_mat[:,x],h2_mat[:,F])
    axarr[2].set_ylabel('F')

    for ii, h2_mat in enumerate(data_h2):
        axarr[3].plot(h2_mat[:,x],h2_mat[:,R])

    axarr[3].set_ylabel('R')
    axarr[3].set_xlabel('lambda')


    plt.show()




