from math import sqrt
import sklearn.metrics as sklm
from seqeval.metrics import f1_score
import argparse
from pathlib import Path
from datasets import load_dataset
import json
import Levenshtein
import re
import numpy as np
from ..utils import massive_lang_map

np.set_printoptions(legacy='1.25')


def convert_to_bio(seq_tags, outside='Other', labels_merge=None):
    """
    Converts a sequence of tags into BIO format. EX:

        ['city', 'city', 'Other', 'country', -100, 'Other']
        to
        ['B-city', 'I-city', 'O', 'B-country', 'I-country', 'O']
        where outside = 'Other' and labels_merge = [-100]

    :param seq_tags: the sequence of tags that should be converted
    :type seq_tags: list
    :param outside: The label(s) to put outside (ignore). Default: 'Other'
    :type outside: str or list
    :param labels_merge: The labels to merge leftward (i.e. for tokenized inputs)
    :type labels_merge: str or list
    :return: a BIO-tagged sequence
    :rtype: list
    """

    seq_tags = [str(x) for x in seq_tags]

    outside = [outside] if type(outside) != list else outside
    outside = [str(x) for x in outside]

    if labels_merge:
        labels_merge = [labels_merge] if type(labels_merge) != list else labels_merge
        labels_merge = [str(x) for x in labels_merge]
    else:
        labels_merge = []

    bio_tagged = []
    prev_tag = None
    for tag in seq_tags:
        if prev_tag == None and tag in labels_merge:
            bio_tagged.append('O')
        elif tag in outside:
            bio_tagged.append('O')
            prev_tag = tag
        elif tag != prev_tag and tag not in labels_merge:
            bio_tagged.append('B-' + tag)
            prev_tag = tag
        elif tag == prev_tag or tag in labels_merge:
            if prev_tag in outside:
                bio_tagged.append('O')
            else:
                bio_tagged.append('I-' + prev_tag)

    return bio_tagged


# massive eval script
# https://github.com/alexa/massive/blob/main/src/massive/utils/training_utils.py#L427
def eval_preds(pred_intents=None, lab_intents=None, pred_slots=None, lab_slots=None,
               eval_metrics='all', labels_ignore='Other', labels_merge=None, pad='Other'):
    """
    Function to evaluate the predictions from a model

    :param pred_intents: a list of predicted intents
    :type pred_intents: list
    :param lab_intents: a list of intents labels (ground truth)
    :type lab_intents: list
    :param pred_slots: a list of predicted slots, where each entry is a list of token-based slots
    :type pred_slots: list
    :param lab_slots: a list of slots labels (ground truth)
    :type lab_slots: list
    :param eval_metrics: The metrics to include. Options are 'all', 'intent_acc', 'ex_match_acc',
                         'slot_micro_f1'
    :type eval_metrics: str
    :param labels_ignore: The labels to ignore (prune away). Default: ['Other']
    :type labels_ignore: str or list
    :param labels_merge: The labels to merge leftward (i.e. for tokenized inputs)
    :type labels_merge: str or list
    :param pad: The value to use when padding slot predictions to match the length of ground truth
    :type pad: str
    """

    results = {}

    # Check lengths
    if pred_intents is not None and lab_intents is not None:
        assert len(pred_intents) == len(lab_intents),f"pred_intents and lab_intents must be same len {len(pred_intents)} vs. {len(lab_intents)}"
    if pred_slots is not None and lab_slots is not None:
        assert len(pred_slots) == len(lab_slots), "pred_slots and lab_slots must be same length"

    if ('intent_acc' in eval_metrics) or ('all' in eval_metrics):
        intent_acc = sklm.accuracy_score(lab_intents, pred_intents)
        results['intent_acc'] = intent_acc
        # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
        results['intent_acc_stderr'] = sqrt(intent_acc*(1-intent_acc)/len(pred_intents))

    if lab_slots is not None and pred_slots is not None:
        bio_slot_labels, bio_slot_preds = [], []
        for lab, pred in zip(lab_slots, pred_slots):

            # Pad or truncate prediction as needed using `pad` arg
            if type(pred) == list:
                pred = pred[:len(lab)] + [pad]*(len(lab) - len(pred))

            # Fix for Issue 21 -- subwords after the first one from a word should be ignored
            for i, x in enumerate(lab):
                if x == -100:
                    pred[i] = -100

            # convert to BIO
            bio_slot_labels.append(
                convert_to_bio(lab, outside=labels_ignore, labels_merge=labels_merge)
            )
            bio_slot_preds.append(
                convert_to_bio(pred, outside=labels_ignore, labels_merge=labels_merge)
            )

    if ('slot_micro_f1' in eval_metrics) or ('all' in eval_metrics):

        # from seqeval
        smf1 = f1_score(bio_slot_labels, bio_slot_preds)
        results['slot_micro_f1'] = smf1
        # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
        total_slots = sum([len(x) for x in bio_slot_preds])
        results['slot_micro_f1_stderr'] = sqrt(smf1*(1-smf1)/total_slots)

    if ('ex_match_acc' in eval_metrics) or ('all' in eval_metrics):
        # calculate exact match accuracy (~0.01 seconds)
        matches = 0
        denom = 0
        for p_int, p_slot, l_int, l_slot in zip(pred_intents,
                                                bio_slot_preds,
                                                lab_intents,
                                                bio_slot_labels):

            if (p_int == l_int) and (p_slot == l_slot):
                matches += 1
            denom += 1
        emacc = matches / denom

        results['ex_match_acc'] = emacc
        # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
        results['ex_match_acc_stderr'] = sqrt(emacc*(1-emacc)/len(pred_intents))

    return results


def subfinder(orig_list, pattern):
    matches = []
    match_idx = []
    orig_list_lower = [i.lower() for i in orig_list]
    pattern_lower = [i.lower() for i in pattern]
    for i in range(len(orig_list)):
        if orig_list_lower[i] == pattern_lower[0] and orig_list_lower[i:i + len(pattern)] == pattern_lower:
            matches.append(pattern)
            match_idx.append(i)
    return matches, match_idx


def isascii(s):
    try:
        return s.isascii()
    except AttributeError:
        return all([ord(c) < 128 for c in s])


def main(args):

    assert Path(args.pred_slots_file).is_file()

    partition = args.partition
    dataset_partition = load_dataset("AmazonScience/massive", name=massive_lang_map[args.lang])[partition]

    orig_utt, label_slots = [], []

    # from https://github.com/alexa/massive/blob/main/scripts/create_hf_dataset.py#L114
    char_split = ['ja-JP', 'zh-CN', 'zh-TW']
    for row in dataset_partition:
        eyed, locale, split, utt = row['id'], row['locale'], row['partition'], row['utt']
        # domain = row['scenario'] if 'scenario' in row else ''
        # intent = row['intent'] if 'intent' in row else ''
        annot_utt = row['annot_utt'] if 'annot_utt' in row else ''

        # Split these languages by character
        if locale in char_split:
            tokens, labels = [], []
            label = 'Other'
            skip_colon = False
            if annot_utt:
                for chunk in annot_utt.split():
                    if chunk.startswith('['):
                        label = chunk.lstrip('[')
                        skip_colon = True
                        continue
                    if chunk == ':' and skip_colon is True:
                        skip_colon = False
                        continue
                    # keep latin chars together in cases of code switching
                    if isascii(chunk):
                        tokens.append(chunk.strip().rstrip(']'))
                        labels.append(label)
                    else:
                        chars = list(chunk.strip())
                        for char in chars:
                            if char == ']':
                                label = 'Other'
                            else:
                                tokens.append(char)
                                labels.append(label)
            # if no annot_utt, then make assumption latin words are space sep already
            else:
                for chunk in utt.split():
                    if isascii(chunk):
                        tokens.append(chunk.strip())
                    else:
                        chars = list(chunk.strip())
                        for char in chars:
                            tokens.append(char)

        else:
            # Create the tokens and labels by working left to right of annotated utt
            tokens = utt.split()
            labels = []
            label = 'Other'
            split_annot_utt = annot_utt.split()
            idx = 0
            while idx < len(split_annot_utt):
                if split_annot_utt[idx].startswith('['):
                    label = split_annot_utt[idx].lstrip('[')
                    idx += 2
                elif split_annot_utt[idx].endswith(']'):
                    labels.append(label)
                    label = 'Other'
                    idx += 1
                else:
                    labels.append(label)
                    idx += 1

        label_slots.append(labels)
        orig_utt.append(tokens)

    # dummy values
    label_intents = ["None" for _ in label_slots]
    pred_intents = ["None" for _ in label_slots]
    pred_slots = [['Other'] * len(l) for l in label_slots]

    with open(args.pred_slots_file, "r") as f:
        for l_idx, l in enumerate(f.readlines()):
            # contains tagged parts
            if l.strip().lower() != "none":

                tagged_parts = l.strip().split("; ")
                for p in tagged_parts:
                    # check format is "label: phrase"
                    if len(p.split(": ")) == 2:
                        utt_pred_label, utt_part = p.split(": ")
                        # find tokens in original sentence
                        utt_part_tokens = utt_part.split()

                        if utt_part:
                            if locale not in char_split:
                                _, match_idx = subfinder(orig_utt[l_idx], utt_part_tokens)
                                for i in match_idx:
                                    pred_slots[l_idx][i: i + len(utt_part_tokens)] = [utt_pred_label] * len(
                                        utt_part_tokens)
                            else:   # used char-based split (ja, zh)
                                _, match_idx = subfinder(list(orig_utt[l_idx]), list(utt_part))
                                for i in match_idx:
                                    pred_slots[l_idx][i: i + len(list(utt_part))] = \
                                        [utt_pred_label] * len(list(utt_part))
                    else:
                        print("*** WARNING: ill-formatted tagged parts: " + p)

    # postprocess predicted intents
    res = eval_preds(pred_intents=pred_intents,
                     lab_intents=label_intents,
                     pred_slots=pred_slots,
                     lab_slots=label_slots)
    print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-slots-file", required=True, type=str)
    parser.add_argument("--partition", type=str, default="validation")
    parser.add_argument("--lang", type=str, default='en', help="language used in inference")
    args = parser.parse_args()
    main(args=args)
