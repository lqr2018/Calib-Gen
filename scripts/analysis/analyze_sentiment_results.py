import argparse
import json
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=str, required=True, nargs='+')
    parser.add_argument("--alpha_co", type=float, default=-0.2)
    args = parser.parse_args()

    for fname in args.outputs:
        with open(fname, 'r') as f:
            lines = [json.loads(line) for line in f]

        correct_predictions = []
        ppl = []
        cos = []
        total = len(lines)

        for line in lines:
            correct_predictions.append(line['success'])
            ppl.append(line['perplexity'])
            cos.append(line['cosine_similarity'])

        fname_txt = fname.replace(".jsonl", ".txt")
        with open(fname_txt, "w") as f:
            f.write(f"Statistics of {fname}\n")
            f.write(f"    Sample size: {total}\n")
            f.write(f"    Success rate of continuations: {1 - sum(correct_predictions) / total}\n")
            f.write(f"    Average perplexity of continuations: {sum(ppl) / total}\n")
            f.write(f"    Average cosine similarity of continuations: {sum(cos) / total}\n\n")



