#!/usr/bin/python

# Evaluate the prediciton
import sys
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

def fdr(pred_pos, pred_neg, cutoff):
    tp = 0.0
    for each in pred_pos:
        if (each > cutoff):
            tp += 1

    fp = 0.0
    for each in pred_neg:
        if (each > cutoff):
            fp += 1

    s = tp + fp
    if (s == 0): s = 1

    return tp / s

def fdrNew(pred_pos, pred_neg, fdr_cutoff):
    buf = []
    for each in pred_pos:
        buf.append([each, 1])
    for each in pred_neg:
        buf.append([each, 0])
    buf = sorted(buf, key = lambda x: x[0], reverse=True)

    fpcnt = 0.0
    score_cutoff = buf[0][0]
    for i in range(len(buf)):
        each = buf[i]
        fpcnt += 1 - each[1]
        if (fpcnt / (i+1) < fdr_cutoff):
            # Take the last one satisfying this condition
            score_cutoff = each[0]

    tpcnt = 0.0
    for each in pred_pos:
        if (each > score_cutoff):
            tpcnt += 1

    s = len(pred_pos)
    if (s == 0): s = 1

    return score_cutoff, tpcnt / s

def tpfp(pred_pos, pred_neg, cutoff):
    tp = 0.0
    for each in pred_pos:
        if (each >= cutoff):
            tp += 1

    fp = 0.0
    for each in pred_neg:
        if (each >= cutoff):
            fp += 1

    l1 = len(pred_pos)
    if (l1 == 0): l1 = 1
    l2 = len(pred_neg)
    if (l2 == 0): l2 = 1

    return (tp / l1, fp / l2)

def aucCurve(pred_pos, pred_neg, step):
    aucvalue = 0
    curve = []
    cutoffs = range(int(1/step)+1)
    lasttp = 1
    lastfp = 1
    for idx in cutoffs:
        cutoff = step * idx
        tp, fp = tpfp(pred_pos, pred_neg, cutoff)
        curve.append((cutoff, tp, fp))
        aucvalue += (lastfp - fp) * (lasttp + tp) / 2
        lasttp, lastfp = tp, fp
    # The auc calculated in this way is slightly under estimated because the step based partition

    ## Use the implementation in sklearn
    ##if (len(pred_pos) == 0): pred_pos = [1]
    ##if (len(pred_neg) == 0): pred_neg = [0]
    ##y_true = np.array([1] * len(pred_pos) + [0] * len(pred_neg))
    ##y_scores = np.array(pred_pos + pred_neg)
    ##fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    ##curve = zip(thresholds, tpr, fpr)
    ##aucvalue = roc_auc_score(y_true, y_scores)

    return [aucvalue, curve]

def praucCurve(pred_pos, pred_neg, step):
    true_value = [1] * len(pred_pos)
    true_value += [0] * len(pred_neg)
    pred_value = pred_pos + pred_neg
    precision, recall, thresholds = precision_recall_curve(true_value, pred_value)
    auc = average_precision_score(true_value, pred_value)
    return [auc, zip(thresholds, precision, recall)]

def fdrCurve(pred_pos, pred_neg, step):
    """ For each prediction value cutoff, calculate the true positive rate """
    value = 0
    curve = []
    cutoffs = range(int(1/step)+1)
    for idx in cutoffs:
        cutoff = step * idx
        tp = fdr(pred_pos, pred_neg, cutoff)
        curve.append((cutoff, tp, 1-tp))
        value += step * tp
    return [value, curve]

def fdrNewCurve(pred_pos, pred_neg, step):
    """ For each FDR cutoff, calculate the recall """
    value = 0
    curve = []
    cutoffs = range(int(1/step)+1)
    for idx in cutoffs:
        cutoff = step * idx
        score_cutoff, tp = fdrNew(pred_pos, pred_neg, cutoff)
        curve.append((cutoff, tp, score_cutoff))
        value += step * tp
    return [value, curve]

def printCurve(comb, name, aucResult):
    sys.stdout.write("%15s\t%30s\t%.4f" % (comb, name, aucResult[0]))
    for cutoff, tp, fp in aucResult[1]:
        sys.stdout.write("\t%f:%.4f:%.4f" % (cutoff, tp, fp))
    sys.stdout.write("\n")

def selectCol(cols, pred, annotation, step, curve_type):
    """ Group the data by specified columns """
    if (cols[0] == 0):
        pred_pos = [float(each[1]) for each in pred if each[0] == '1']
        pred_neg = [float(each[1]) for each in pred if each[0] == '0']

        if (curve_type == 'AUC'):
            printCurve('ALL', 'ALL', aucCurve(pred_pos, pred_neg, step))
        elif (curve_type == 'FDR'):
            printCurve('ALL', 'ALL', fdrCurve(pred_pos, pred_neg, step))
        elif (curve_type == 'FDRNew'):
            printCurve('ALL', 'ALL', fdrNewCurve(pred_pos, pred_neg, step))
        elif (curve_type == 'PRAUC'):
            printCurve('ALL', 'ALL', praucCurve(pred_pos, pred_neg, step))
        else:
            sys.stderr.write('ERROR, Unknown type: %s\n' % curve_type)
            exit(1)
        return

    byids = {}
    for pv, anno in zip(pred, annotation):
        mid = ""
        for each in cols:
            mid += "|"  + anno[each-1]

        if (mid not in byids):
            byids[mid] = ([], [])

        if (pv[0] == '1'):
            byids[mid][0].append(float(pv[1]))
        else:
            byids[mid][1].append(float(pv[1]))
    for key in byids:
        if (len(byids[key][0]) == 0): 
            sys.stderr.write('Warning, ' + key + ' does not have positive true label. Supply with a pseduo positive label\n')
        if (len(byids[key][1]) == 0): 
            sys.stderr.write('Warning, ' + key + ' does not have negative true label. Supply with a pseduo negative label\n')

        if (curve_type == 'AUC'):
            printCurve(str(cols), key, aucCurve(byids[key][0], byids[key][1], step))
        elif (curve_type == 'FDR'):
            printCurve(str(cols), key, fdrCurve(byids[key][0], byids[key][1], step))
        elif (curve_type == 'FDRNew'):
            printCurve(str(cols), key, fdrNewCurve(byids[key][0], byids[key][1], step))
        elif (curve_type == 'PRAUC'):
            printCurve(str(cols), key, praucCurve(byids[key][0], byids[key][1], step))
        else:
            sys.stderr.write('ERROR, Unknown type: %s\n' % curve_type)
            exit(1)

def process_command_line(argv):
    """ Processing command line """

    if argv is None:
         argv = sys.argv[1:]
 
    parser = argparse.ArgumentParser(description="""Given the predictions, do evaluation.""")
 
    # Manual:
    parser.add_argument('-p', '--pred', dest='pred',required=True, help='The prediction result')
    parser.add_argument('-s', '--seq', dest='seq', required=True, help='The fa file used to generate the prediction.')
    parser.add_argument('-c', '--comb', dest='comb', required=True, help="Combinations to generate summary by specifying 1-based columns. 0 means ALL. E.g. 0,1.2,1.2.3,1")
    parser.add_argument('-d', '--dup', dest='dup', action='store_true', help="If the negative sequence is randomly generated, there is no annotation for negative instances. set this parameter.")
    parser.add_argument('-t', '--type', dest='type', required=True, help="The type of curve. FDRNew or FDR or AUC or PRAUC")

    args = parser.parse_args(argv)

    return args

 
def main(argv=None):
    """ The main entrance of the program. """

    args = process_command_line(argv)
    pred = args.pred
    seq = args.seq
    comb = args.comb
    step = 0.01

    comb = args.comb.split(',')
    pred = [each.strip().split() for each in open(pred)]
    annotation = [each.strip().split()[:-1] for each in open(seq)]

    if (args.dup):
        newannotation = []
        for each in annotation:
            newannotation.append(each)
            newannotation.append(each)
        annotation = newannotation

    for each in comb:
        each = each.split('.')
        each = [int(e) for e in each]
        #sys.stderr.write(str(each))
        #sys.stderr.write("\n")
        selectCol(each, pred, annotation, step, args.type)
 
if __name__ == '__main__':
    status = main()
    sys.exit(status)
