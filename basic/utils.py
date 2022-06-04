

import editdistance
import re

# Charset / labels conversion

def LM_str_to_ind(labels, str):
    return [labels.index(c) for c in str]


def LM_ind_to_str(labels, ind, oov_symbol=None):
    if oov_symbol is not None:
        res = []
        for i in ind:
            if i < len(labels):
                res.append(labels[i])
            else:
                res.append(oov_symbol)
    else:
        res = [labels[i] for i in ind]
    return "".join(res)


# OCR METRICS


def edit_cer_from_list(truth, pred):
    edit = 0
    for t, p in zip(truth, pred):
        edit += editdistance.eval(t, p)
    return edit


def format_string_for_wer(str):
    str = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str)
    str = re.sub('([ \n])+', " ", str).strip()
    return str


def edit_wer_from_list(truth, pred):
    edit = 0
    for pred, gt in zip(pred, truth):
        gt = format_string_for_wer(gt)
        pred = format_string_for_wer(pred)
        gt = gt.split(" ")
        pred = pred.split(" ")
        edit += editdistance.eval(gt, pred)
    return edit


def nb_words_from_list(list_gt):
    len_ = 0
    for gt in list_gt:
        gt = format_string_for_wer(gt)
        gt = gt.split(" ")
        len_ += len(gt)
    return len_


def nb_chars_from_list(list_gt):
    return sum([len(t) for t in list_gt])


def cer_from_list_str(str_gt, str_pred):
        len_ = 0
        edit = 0
        for pred, gt in zip(str_pred, str_gt):
            edit += editdistance.eval(gt, pred)
            len_ += len(gt)
        cer = edit / len_
        return cer


def wer_from_list_str(str_gt, str_pred):
    len_ = 0
    edit = 0
    for pred, gt in zip(str_pred, str_gt):
        gt = format_string_for_wer(gt)
        pred = format_string_for_wer(pred)
        gt = gt.split(" ")
        pred = pred.split(" ")
        edit += editdistance.eval(gt, pred)
        len_ += len(gt)
    cer = edit / len_
    return cer


def cer_from_files(file_gt, file_pred):
        with open(file_pred, "r") as f_p:
            str_pred = f_p.readlines()
        with open(file_gt, "r") as f_gt:
            str_gt = f_gt.readlines()
        return cer_from_list_str(str_gt, str_pred)


def wer_from_files(file_gt, file_pred):
    with open(file_pred, "r") as f_p:
        str_pred = f_p.readlines()
    with open(file_gt, "r") as f_gt:
        str_gt = f_gt.readlines()
    return wer_from_list_str(str_gt, str_pred)
