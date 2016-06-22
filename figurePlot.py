import numpy as np
import matplotlib.pyplot as plt


####---- plot hoang
csv_filename = '/Users/gong/Documents/MTG document/Jingju arias/phonemeSeg/hoa_mfccBands.csv'

data         = np.loadtxt(csv_filename,delimiter=',')

l   = 0
h1  = 1
h2  = 2

HR  = 3
OS  = 4
FAR = 5
F   = 6
R   = 7
deletion    = 8
insertion   = 9

l_to_choose = [2,4,6,8,10]

for l_chosen in l_to_choose:
    data_l  = data[data[:,l]==l_chosen,:]

    h1_6 = data_l[data_l[:,h1]==0.6,:]
    h1_8 = data_l[data_l[:,h1]==0.8,:]
    h1_10 = data_l[data_l[:,h1]==1.0,:]


    f, axarr = plt.subplots(4, sharex=True)
    axarr[0].plot(h1_6[:,h2],h1_6[:,HR],label='h1=0.6')
    axarr[0].plot(h1_8[:,h2],h1_8[:,HR],label='h1=0.8')
    axarr[0].plot(h1_10[:,h2],h1_10[:,HR],label='h1=1.0')
    axarr[0].set_ylabel('HR')
    axarr[0].legend()
    axarr[0].set_title('l='+str(l_chosen))

    axarr[1].plot(h1_6[:,h2],h1_6[:,OS])
    axarr[1].plot(h1_8[:,h2],h1_8[:,OS])
    axarr[1].plot(h1_10[:,h2],h1_10[:,OS])
    axarr[1].set_ylabel('OS')

    axarr[2].plot(h1_6[:,h2],h1_6[:,F])
    axarr[2].plot(h1_8[:,h2],h1_8[:,F])
    axarr[2].plot(h1_10[:,h2],h1_10[:,F])
    axarr[2].set_ylabel('F')

    axarr[3].plot(h1_6[:,h2],h1_6[:,R])
    axarr[3].plot(h1_8[:,h2],h1_8[:,R])
    axarr[3].plot(h1_10[:,h2],h1_10[:,R])
    axarr[3].set_ylabel('R')
    axarr[3].set_xlabel('h2')


    plt.show()


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

for a_chosen in a_to_choose:
    data_a  = data[data[:,fn]==a_chosen,:]

    c_3 = data_a[data_a[:,ln]==3,:]
    c_5 = data_a[data_a[:,ln]==5,:]
    c_7 = data_a[data_a[:,ln]==7,:]
    c_9 = data_a[data_a[:,ln]==9,:]

    f, axarr = plt.subplots(4, sharex=True)
    axarr[0].plot(c_3[:,x],c_3[:,HR],label='c=3')
    axarr[0].plot(c_5[:,x],c_5[:,HR],label='c=5')
    axarr[0].plot(c_7[:,x],c_7[:,HR],label='c=7')
    axarr[0].plot(c_9[:,x],c_9[:,HR],label='c=9')

    axarr[0].set_ylabel('HR')
    axarr[0].legend()
    axarr[0].set_title('a='+str(a_chosen))

    axarr[1].plot(c_3[:,x],c_3[:,OS])
    axarr[1].plot(c_5[:,x],c_5[:,OS])
    axarr[1].plot(c_7[:,x],c_7[:,OS])
    axarr[1].plot(c_9[:,x],c_9[:,OS])

    axarr[1].set_ylabel('OS')

    axarr[2].plot(c_3[:,x],c_3[:,F])
    axarr[2].plot(c_5[:,x],c_5[:,F])
    axarr[2].plot(c_7[:,x],c_7[:,F])
    axarr[2].plot(c_9[:,x],c_9[:,F])

    axarr[2].set_ylabel('F')

    axarr[3].plot(c_3[:,x],c_3[:,R])
    axarr[3].plot(c_5[:,x],c_5[:,R])
    axarr[3].plot(c_7[:,x],c_7[:,R])
    axarr[3].plot(c_9[:,x],c_9[:,R])

    axarr[3].set_ylabel('R')
    axarr[3].set_xlabel('c')


    plt.show()
'''


