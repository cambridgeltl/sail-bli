import sys
import argparse
import pickle as pkl
import os
import numpy as np
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer Seed Pairs')

    parser.add_argument("--forward_data_dir", type=str, default="./",
                    help="forward_data_dir")
    parser.add_argument("--backward_data_dir", type=str, default="./",
                    help="backward_data_dir")
    parser.add_argument("--topk", type=int, default=1,
                    help="topk")
    parser.add_argument("--save_dir", type=str, default="./",
                    help="save_dir")

    start_time = time.time()
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    args_dict = vars(args)

    ### Load Data

    f_name = args.forward_data_dir
    with open(f_name,"rb") as f:
        forward_s2t, forward_t2s = pkl.load(f)
    f_name = args.backward_data_dir
    with open(f_name,"rb") as f:
        backward_s2t, backward_t2s = pkl.load(f)

    merged_s2t = {**forward_s2t,**backward_s2t}
    merged_t2s = {**forward_t2s,**backward_t2s}

    high_confidence_list = set()  # s2t
    for k, v in forward_s2t.items():
        s_w = k
        t_w = v[0]
        if t_w in merged_t2s:
            s_list = merged_t2s[t_w][:args.topk]
            if s_w in s_list:
                high_confidence_list.add((s_w, t_w))


    for k, v in forward_t2s.items():
        t_w = k
        s_w = v[0]
        if s_w in merged_s2t:
            t_list = merged_s2t[s_w][:args.topk]
            if t_w in t_list:
                high_confidence_list.add((s_w, t_w))
    high_confidence_list = list(high_confidence_list)
    high_confidence_list = ["\t".join(e) for e in high_confidence_list]
    # save inferred dicts
    save_txt = args.save_dir
    with open(save_txt, 'w') as f:
        for i,e in enumerate(high_confidence_list):
            f.write(e+"\n")
    end_time = time.time()
    print("Total Runtime :", end_time-start_time)
    sys.stdout.flush()
