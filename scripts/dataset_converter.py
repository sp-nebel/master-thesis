import sys
from datasets import load_dataset
import jsonlines

DATASET = 'xnli'

BASE_INSTRUCTION = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n'
INSTRUCTION = '### Instruction:\nDetermine how the following premise and hypothesis are related. If the hypothesis follows from the premise, select "entailment". If they contradict each other, select "contradiction". If the statements do not entail nor contradict each other, select "neutral". Premise: {premise} Hypothesis: {hypothesis}\n\n### Response:'
LABEL = '{label}</s>'

ENTAILMENT_ID = 0
NEUTRAL_ID = 1
CONTRADICTION_ID = 2

language = sys.argv[1]
split = sys.argv[2]
output_file = sys.argv[3]

def convert_label(label):
    if label == ENTAILMENT_ID:
        return 'entailment'
    elif label == NEUTRAL_ID:
        return 'neutral'
    elif label == CONTRADICTION_ID:
        return 'contradiction'
    else:
        raise ValueError(f'Unknown label: {label}')


def main():
    print(f'Converting {DATASET} {language} {split} split to jsonl')
    dataset = load_dataset(DATASET, language, split=split)
    print(f'Number of examples: {len(dataset)}')

    with jsonlines.open(f'{output_file}.json', mode='w') as writer:
        for example in dataset:
            json_dict = {
                'text': LABEL.format(label=convert_label(example['label'])),
                'prefix': BASE_INSTRUCTION + INSTRUCTION.format(premise=example['premise'], hypothesis=example['hypothesis']),
            }
            writer.write(json_dict)

if __name__ == '__main__':
    main()