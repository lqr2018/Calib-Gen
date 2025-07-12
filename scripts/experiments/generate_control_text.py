from transformers import AutoTokenizer
from utils.utils import *

from methods.prefixes import *
from methods import BaseMethod, CalibGen, Fudge, PreAdd

def generate_control_text(task,
                          method,
                          prompt,
                          prefix_setting,
                          strength,
                          max_tokens,
                          top_k,
                          top_p,
                          base_model_string,
                          fudge_model_string):
    # generate output
    prefix = all_prefixes[task][prefix_setting]
    if method == "preadd":
        preadd = PreAdd(prompt=prompt,
                        prefix=prefix,
                        strength=strength,
                        max_tokens=max_tokens,
                        top_k=top_k,
                        top_p=top_p,
                        model_string=base_model_string)
        output = preadd.generate()
    elif method == "calib":
        calib = CalibGen(prompt=prompt,
                         prefix=prefix,
                         strength=strength,
                         max_tokens=max_tokens,
                         top_k=top_k,
                         top_p=top_p,
                         model_string=base_model_string)

        output = calib.muse()

    elif method == "raw_prompting":
        raw_prompting = BaseMethod(prompt=prompt,
                                   max_tokens=max_tokens,
                                   model_string=base_model_string)
        output = raw_prompting.generate()

    elif method == "neg_prompting":
        if prefix_setting == "pos":
            prefix = all_prefixes[task]["neg"]
        elif prefix_setting == "neg":
            prefix = all_prefixes[task]["pos"]
        neg_prompting = BaseMethod(prompt=prompt,
                                   prefix=prefix,
                                   max_tokens=max_tokens,
                                   model_string=base_model_string)
        output = neg_prompting.generate()

    elif method == "fudge":
        if task == "sentiment":
            task = task + "_" + prefix_setting
        fudge = Fudge(prompt=prompt,
                                strength=1,
                                max_tokens=max_tokens,
                                model_string=base_model_string,
                                control_model_string=fudge_model_string,
                                task=task)
        output = fudge.generate()
    else:
        raise NotImplementedError

    return output