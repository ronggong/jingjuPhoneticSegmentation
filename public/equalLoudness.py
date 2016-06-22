import numpy as np

def equalLoudness(bandCenters):

    # Hynek's magic equal-loudness-curve formula
    fsq = np.power(bandCenters,2.0)
    ftmp = fsq + 1.6e5
    eql = (np.power((fsq/ftmp),2.0)) * ((fsq + 1.44e6)/(fsq + 9.61e6))

    return eql