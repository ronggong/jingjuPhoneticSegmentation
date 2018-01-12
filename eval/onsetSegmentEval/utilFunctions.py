def removeSilence(phn_onsets, phn_labels):
    """
    Remove silence phoneme onsets and labels
    :param phn_onsets:
    :param phn_labels:
    :return:
    """
    phn_onsets = list(phn_onsets)
    phn_labels = list(phn_labels)

    for ii in reversed(range(len(phn_labels))):
        if phn_labels[ii] == u'':
            phn_onsets.pop(ii)
            phn_labels.pop(ii)
    return phn_onsets, phn_labels