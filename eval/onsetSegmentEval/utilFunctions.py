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
        if not len(phn_labels[ii]):
            phn_onsets.pop(ii)
            phn_labels.pop(ii)