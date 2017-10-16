import numpy as np
from char_map import index_map
from utils import for_tf_or_th


def ctc_decode(acc, next_i):
    next_char = '' if next_i == for_tf_or_th(28, 0) else index_map[next_i]
    if not acc:
        return [next_char]
    if acc[-1] == next_char:
        return acc
    if acc[-1] == '':
        return acc[:-1] + [next_char]
    return acc + [next_char]


def beam_decode(probs, beam_width, eps=1e-5):
    preds = []

    for t in range(probs.shape[0]):
        best_ys = probs[t].argsort()[-beam_width:]
        if t == 0:
            pred_probs = np.log(probs[t, best_ys].clip(eps, 100))
            preds = [[i] for i in best_ys]
        else:
            level_probs = np.array([
                [pp + np.log(p.clip(eps, 100)) for p in probs[t, best_ys]]
                for pp in pred_probs]).flatten()
            best_probs = level_probs.argsort()[-beam_width:]
            preds = [preds[prob_ix // beam_width] +
                     [best_ys[prob_ix % beam_width]] for prob_ix in best_probs]
            pred_probs = level_probs[best_probs]

    # decode ctc
    predicts = []
    for pred in preds:
        prev, plist = -1, []
        for i in pred:
            if i == prev:
                continue
            elif i != for_tf_or_th(28, 0):
                plist.append(index_map[i])
            prev = i
        predicts.append(''.join(plist))

    return predicts, pred_probs


def beam_decode_u(probs, beam_width, eps=1e-5, normalize=False):
    u_preds = []  # unique predictions
    # give more characters a chance becauase we remove duplicates in each step
    mid_beam = max(beam_width+2, probs.shape[1])

    # loop over each time
    for t in range(probs.shape[0]):
        best_ys = probs[t].argsort()[-mid_beam:]
        if normalize:
            clipped_t_probs = probs[t].clip(eps, 100)
            norm_log_sum = np.log(np.exp(clipped_t_probs[best_ys]).sum())
        if t == 0:
            if normalize:
                pred_probs = clipped_t_probs[best_ys] - norm_log_sum
            else:
                pred_probs = np.log(probs[t, best_ys].clip(eps, 100))
            u_preds = [[] if i == for_tf_or_th(28, 0) else [index_map[i]]
                       for i in best_ys]
        else:
            if normalize:
                level_probs = np.array([
                    [pp + p - norm_log_sum for p in clipped_t_probs[best_ys]]
                    for pp in pred_probs]).flatten()
            else:
                level_probs = np.array([
                    [pp + np.log(p.clip(eps, 100)) for p in probs[t, best_ys]]
                    for pp in pred_probs]).flatten()
            best_probs = level_probs.argsort()[-(beam_width*2):]
            level_preds = [(prob_ix, ctc_decode(u_preds[prob_ix // mid_beam],
                                                best_ys[prob_ix % mid_beam]))
                           for prob_ix in best_probs]
            # delete duplicates
            new_preds, new_prob_ixs = [], []
            for prob_ix, pred in level_preds[::-1]:
                if pred in new_preds:
                    continue
                else:
                    new_preds.append(pred)
                    new_prob_ixs.append(prob_ix)
            u_preds = new_preds[:beam_width]
            pred_probs = level_probs[new_prob_ixs[:beam_width]]

    return [''.join(pred) for pred in u_preds], pred_probs


def beam_decode_mul(probs, beam_width):
    nodes = [[]] * beam_width
    scores = None

    for t in range(probs.shape[0]):
        best_ys = probs[t].argsort()[-beam_width:]
        if t == 0:
            best_scores = probs[t, best_ys] / 10
        else:
            best_scores = (scores[:, None] * probs[t, best_ys]/10).flatten()
        best_is = best_scores.argsort()[-beam_width:]
        nodes = [nodes[si // beam_width] + [best_ys[si % beam_width]]
                 for si in best_is]
        print (best_scores)
        scores = np.clip(best_scores[best_is], 1e-4, 1e4)
    preds = []
    for strcode in nodes:
        preds.append([])
        pred = -1
        for code in strcode:
            if code == pred:
                continue
            elif code != for_tf_or_th(28, 0):
                preds[-1].append(index_map[code])
            pred = code

    return [''.join(p) for p in preds], scores
