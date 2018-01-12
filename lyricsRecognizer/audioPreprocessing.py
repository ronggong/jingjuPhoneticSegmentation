from madmom.processors import SequentialProcessor
from general.Fprev_sub import Fprev_sub
import numpy as np

EPSILON = np.spacing(1)

def _nbf_2D(mfcc, nlen):
    mfcc = np.array(mfcc).transpose()
    mfcc_out = np.array(mfcc, copy=True)
    for ii in range(1, nlen + 1):
        mfcc_right_shift = Fprev_sub(mfcc, w=ii)
        mfcc_left_shift = Fprev_sub(mfcc, w=-ii)
        mfcc_out = np.vstack((mfcc_right_shift, mfcc_out, mfcc_left_shift))
    feature = mfcc_out.transpose()
    return feature


class MadmomMelbankProcessor(SequentialProcessor):


    def __init__(self, fs, hopsize_t):
        from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
        from madmom.audio.stft import ShortTimeFourierTransformProcessor
        from madmom.audio.filters import MelFilterbank
        from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                              LogarithmicSpectrogramProcessor)
        # from madmom.features.onsets import _cnn_onset_processor_pad

        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=fs)
        # process the multi-resolution spec in parallel
        # multi = ParallelProcessor([])
        # for frame_size in [2048, 1024, 4096]:
        frames = FramedSignalProcessor(frame_size=2048, hopsize=int(fs*hopsize_t))
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            filterbank=MelFilterbank, num_bands=80, fmin=27.5, fmax=16000,
            norm_filters=True, unique_filters=False)
        spec = LogarithmicSpectrogramProcessor(log=np.log, add=EPSILON)

        # process each frame size with spec and diff sequentially
        # multi.append())
        single = SequentialProcessor([frames, stft, filt, spec])

        # stack the features (in depth) and pad at beginning and end
        # stack = np.dstack
        # pad = _cnn_onset_processor_pad

        # pre-processes everything sequentially
        pre_processor = SequentialProcessor([sig, single])

        # instantiate a SequentialProcessor
        super(MadmomMelbankProcessor, self).__init__([pre_processor])


def getMFCCBands2DMadmom(audio_fn, fs, hopsize_t, channel):
    madmomMelbankProc = MadmomMelbankProcessor(fs, hopsize_t)
    mfcc = madmomMelbankProc(audio_fn)

    if channel == 1:
        mfcc = _nbf_2D(mfcc, 7)
    else:
        mfcc_conc = []
        for ii in range(3):
            mfcc_conc.append(_nbf_2D(mfcc[:,:,ii], 7))
        mfcc = np.stack(mfcc_conc, axis=2)
    return mfcc


def featureReshape(feature, nlen=10):
    """
    reshape mfccBands feature into n_sample * n_row * n_col
    :param feature:
    :return:
    """

    n_sample = feature.shape[0]
    n_row = 80
    n_col = nlen*2+1

    feature_reshaped = np.zeros((n_sample,n_row,n_col),dtype='float32')
    # print("reshaping feature...")
    for ii in range(n_sample):
        # print ii
        feature_frame = np.zeros((n_row,n_col),dtype='float32')
        for jj in range(n_col):
            feature_frame[:,jj] = feature[ii][n_row*jj:n_row*(jj+1)]
        feature_reshaped[ii,:,:] = feature_frame
    return feature_reshaped

from general.parameters import *
from general.filePathHsmm import kerasScaler_path
import essentia.standard as ess
import pickle

winAnalysis     = 'hann'
N               = 2 * framesize                     # padding 1 time framesize
SPECTRUM        = ess.Spectrum(size=N)
MFCC            = ess.MFCC(sampleRate           =fs,
                           highFrequencyBound   =highFrequencyBound,
                           inputSize            =framesize + 1,
                           numberBands          =80)
WINDOW          = ess.Windowing(type=winAnalysis, zeroPadding=N - framesize)

def getMFCCBands2D(audio, framesize, hopsize, nbf=False, nlen=10):

    '''
    mel bands feature [p[0],p[1]]
    output feature for each time stamp is a 2D matrix
    it needs the array format float32
    :param audio:
    :param p:
    :param nbf: bool, if we need to neighbor frames
    :return:
    '''

    mfcc   = []
    # audio_p = audio[p[0]*fs:p[1]*fs]
    for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize):
        frame           = WINDOW(frame)
        mXFrame         = SPECTRUM(frame)
        bands,mfccFrame = MFCC(mXFrame)
        mfcc.append(bands)

    if nbf:
        mfcc = np.array(mfcc).transpose()
        mfcc_out = np.array(mfcc, copy=True)
        for ii in range(1,nlen+1):
            mfcc_right_shift    = Fprev_sub(mfcc, w=ii)
            mfcc_left_shift     = Fprev_sub(mfcc, w=-ii)
            mfcc_out = np.vstack((mfcc_right_shift, mfcc_out, mfcc_left_shift))
        feature = mfcc_out.transpose()
    else:
        feature = mfcc
    # the mel bands features
    feature = np.array(feature,dtype='float32')

    return feature


def mfccFeature_pipeline(filename_wav):
    audio               = ess.MonoLoader(downmix = 'left', filename = filename_wav, sampleRate = fs)()
    scaler              = pickle.load(open(kerasScaler_path,'rb'))

    feature             = getMFCCBands2D(audio,framesize, hopsize, nbf=True)
    mfcc                = np.log(100000 * feature + 1)
    feature             = scaler.transform(mfcc)
    feature             = featureReshape(feature)

    return feature, mfcc