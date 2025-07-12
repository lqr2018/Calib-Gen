import argparse
import json
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs', type=str, required=True)
    parser.add_argument("--alpha_co", type=float, default=-0.2)
    args = parser.parse_args()

    for fname in args.outputs:
        with open(fname, 'r') as f:
            lines = [json.loads(line) for line in f]

        tox = []
        severe_toxicity = []
        identity_attack = []
        insult = []
        profanity = []
        threat = []

        cont_tox = []
        ppl = []
        cos = []
        total = len(lines)

        for line in lines:
            tox.append(line['api_scores']['TOXICITY'])
            cont_tox.append(line['continuation_api_scores']['TOXICITY'])
            ppl.append(line['perplexity'])
            cos.append(line['cosine_similarity'])

            severe_toxicity.append(line['continuation_api_scores']['SEVERE_TOXICITY'])
            identity_attack.append(line['continuation_api_scores']['IDENTITY_ATTACK'])
            insult.append(line['continuation_api_scores']['INSULT'])
            profanity.append(line['continuation_api_scores']['PROFANITY'])
            threat.append(line['continuation_api_scores']['THREAT'])

        fname_txt = fname.replace(".jsonl", ".txt")
        with open(fname_txt, "w") as f:
            f.write(f"Statistics of {fname}\n")
            f.write(f"    Sample size: {total}\n")
            f.write(f"    Average toxicity: {sum(tox) / total}\n")
            f.write(f"    Average severe toxicity: {sum(severe_toxicity) / total}\n")
            f.write(f"    Average identity attack: {sum(identity_attack) / total}\n")
            f.write(f"    Average insult: {sum(insult) / total}\n")
            f.write(f"    Average profanity: {sum(profanity) / total}\n")
            f.write(f"    Average threat: {sum(threat) / total}\n")

            f.write(f"    Average toxicity of continuations: {sum(cont_tox) / total}\n")
            f.write(f"    Std toxicity of continuations: {np.std(cont_tox)}\n")
            f.write(f"    Average perplexity of continuations: {sum(ppl) / total}\n")
            f.write(f"    Average cosine similarity of continuations: {sum(cos) / total}\n\n")

