import sys
import argparse
import pickle as pkl
import os
import numpy as np
from util import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
random = np.random
random.seed(1234)
torch.manual_seed(1234)


def normed_input(x):
    y = x/(np.linalg.norm(x,axis=1,keepdims=True) + 1e-9)
    return y

def my_sorted(x):    
    y = sorted(x,key=lambda t: t[1]) 
    y = [t[0] for t in y]
    return y

def return_definition_wn(word):
    
    try:
        synsets = wn.synsets(word)
        res = synsets[0].definition()
    except:
        res = None
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Neighbours')

    parser.add_argument("--l1", type=str, default=" ",
                    help="Language (string)")
    parser.add_argument("--l2", type=str, default=" ",
                    help="Language (string)")
    parser.add_argument("--norm_input", action="store_true", default=True,
                    help="True if unit-norm word embeddings")
    parser.add_argument("--train_size", type=int, default=5000,
                    help="train dict size")
    parser.add_argument("--direction", type=str, default="forward",
                    help="direction")
    parser.add_argument("--forward_inference_results", type=str, default=None,
                    help="forward inference results")    
    parser.add_argument("--emb_src_dir", type=str, default="./",
                    help="emb_src_dir")
    parser.add_argument("--emb_tgt_dir", type=str, default="./",
                    help="emb_tgt_dir")
    parser.add_argument("--train_dict_dir", type=str, default="./",
                    help="train_dict_dir") 
    parser.add_argument("--save_dir", type=str, default="./",
                    help="save_dir")
    parser.add_argument("--source_data", type=str, default="./",
                    help="source_data")
    parser.add_argument("--random", action="store_true", default=False,
                    help="Randomly select in-context examples (for ablation study).")

    start_time = time.time()
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    args_dict = vars(args)
    print(args_dict)
    args.str2lang = {"hr":"croatian", "en":"english","fi":"finnish","fr":"french","de":"german","it":"italian","ru":"russian","tr":"turkish","bg":"Bulgarian","ca":"Catalan","hu":"Hungarian"}


####Defining Directories
    DIR_EMB_SRC = args.emb_src_dir
    DIR_EMB_TRG = args.emb_tgt_dir
    DIR_TRAIN_DICT = args.train_dict_dir


    if DIR_TRAIN_DICT is None or DIR_TRAIN_DICT == "None":

        f_name = args.source_data + "{}_vocabulary.pkl".format(args.l1)
        with open(f_name,"rb") as f:
            voc_l1 = pkl.load(f)
        f_name = args.source_data + "{}_vocabulary.pkl".format(args.l2)
        with open(f_name,"rb") as f:
            voc_l2 = pkl.load(f)

    else:
        ####LOAD WORD EMBS
        voc_l1, embs_l1 = load_embs(DIR_EMB_SRC)#,topk=10000)
        print("L1 INPUT WORD VECTOR SPACE OF SIZE:", embs_l1.shape)
    
        args.class_num_l1 = len(voc_l1)
        print("L1 Contain", args.class_num_l1, " Words")
    
        voc_l2, embs_l2 = load_embs(DIR_EMB_TRG)#,topk=10000)
        print("L2 INPUT WORD VECTOR SPACE OF SIZE:", embs_l2.shape)
    
        args.class_num_l2 = len(voc_l2)
        print("L2 Contain", args.class_num_l2, " Words")
    
        if args.norm_input:
            embs_l1 = normed_input(embs_l1)
            embs_l2 = normed_input(embs_l2)
    
        wvs_l1 = torch.from_numpy(embs_l1.copy())
        wvs_l2 = torch.from_numpy(embs_l2.copy())
        print("Static WEs Loaded")
    
        #### LOAD DTRAIN
        file = open(DIR_TRAIN_DICT,'r')
        l1_dic = []
        l2_dic = []
        for line in file.readlines():
            pair = line[:-1].split('\t')
            l1_dic.append(pair[0].lower())
            l2_dic.append(pair[1].lower())
        file.close()
        l1_idx_sup = []
        l2_idx_sup = []
    
        train_pairs = set()
    
        for i in range(len(l1_dic)):
            l1_tok = voc_l1.get(l1_dic[i])
            l2_tok = voc_l2.get(l2_dic[i])
            if (l1_tok is not None) and (l2_tok is not None):
                l1_idx_sup.append(l1_tok)
                l2_idx_sup.append(l2_tok)
                train_pairs.add((l1_tok,l2_tok))
    
        print("Sup Set Size: ", len(l1_idx_sup), len(l2_idx_sup), len(l1_dic), len(l2_dic))
        assert len(l1_idx_sup) == len(l1_dic)
        assert len(l2_idx_sup) == len(l2_dic)
        assert len(l1_idx_sup) == len(l2_idx_sup)
        print("Sup L1 Word Frequency Ranking: ", 'min ',min(l1_idx_sup), ' max ', max(l1_idx_sup), ' average ', float(sum(l1_idx_sup))/len(l1_idx_sup))
        print("Sup L2 Word Frequency Ranking: ", 'min ',min(l2_idx_sup), ' max ', max(l2_idx_sup), ' average ', float(sum(l2_idx_sup))/len(l2_idx_sup))
        sys.stdout.flush() 
    
    
        id2w_l1 = {}
        for k,v in enumerate(l1_dic):
            id2w_l1[k] = v
    
        id2w_l2 = {}
        for k,v in enumerate(l2_dic):
            id2w_l2[k] = v
    
        s2t_train_dict = {}
        t2s_train_dict = {}
        for i,w in enumerate(l1_dic):
            if w in s2t_train_dict:
                if (l2_dic[i],l2_idx_sup[i]) not in s2t_train_dict[w]:
                    s2t_train_dict[w].append((l2_dic[i],l2_idx_sup[i]))
            else:
                s2t_train_dict[w] = []
                s2t_train_dict[w].append((l2_dic[i],l2_idx_sup[i]))
    
        for i,w in enumerate(l2_dic):
            if w in t2s_train_dict:
                if (l1_dic[i],l1_idx_sup[i]) not in t2s_train_dict[w]:
                    t2s_train_dict[w].append((l1_dic[i],l1_idx_sup[i]))            
            else:   
                t2s_train_dict[w] = []
                t2s_train_dict[w].append((l1_dic[i],l1_idx_sup[i]))
    
        for k,v in s2t_train_dict.items():
            s2t_train_dict[k] =  my_sorted(v)
    
        for k,v in t2s_train_dict.items():
            t2s_train_dict[k] = my_sorted(v)
    
        wvs_l1_train = torch.index_select(wvs_l1,0,torch.tensor(l1_idx_sup))
        wvs_l2_train = torch.index_select(wvs_l2,0,torch.tensor(l2_idx_sup))
   


    ### Get prompting dict. Forward: from most freq words. Backward: from forward pass results.
    if args.direction == "forward":
        freq_l1 = sorted(voc_l1.items(),key=lambda x:x[1],reverse=False)[:args.train_size]    
        freq_l2 = sorted(voc_l2.items(),key=lambda x:x[1],reverse=False)[:args.train_size]

        s2t_prompt_dict = {}
        t2s_prompt_dict = {}
        for e in freq_l1:
            s2t_prompt_dict[e[0]]=[]
        for e in freq_l2:
            t2s_prompt_dict[e[0]]=[]
        s2t_words = [e[0] for e in freq_l1]
        t2s_words = [e[0] for e in freq_l2]
        if not (DIR_TRAIN_DICT is None or DIR_TRAIN_DICT == "None"):
            freq_ids1 = [e[1] for e in freq_l1]
            freq_ids2 = [e[1] for e in freq_l2]
            wvs_l1_prompt = torch.index_select(wvs_l1,0,torch.tensor(freq_ids1))
            wvs_l2_prompt = torch.index_select(wvs_l2,0,torch.tensor(freq_ids2))

    else:
        # derive s2t/t2s_prompt_dict wvs_l1/2 prompt
        dicts_forward_dir = args.forward_inference_results
        with open(dicts_forward_dir,"rb") as f:
            dicts_forward = pkl.load(f)
        forward_dict_s2t, forward_dict_t2s = dicts_forward

        s2t_prompt_dict = {} # from forward_dict_t2s
        t2s_prompt_dict = {} # from forward_dict_s2t

        for k,v in forward_dict_t2s.items():
            if len(v)>0 and v[0] not in forward_dict_s2t:
                s2t_prompt_dict[v[0]]=[]
        for k,v in forward_dict_s2t.items():
            if len(v)>0 and v[0] not in forward_dict_t2s:            
                t2s_prompt_dict[v[0]]=[]
        s2t_words = list(s2t_prompt_dict.keys())
        t2s_words = list(t2s_prompt_dict.keys())
        s2t_ids = [voc_l1.get(w) for w in s2t_words]
        t2s_ids = [voc_l2.get(w) for w in t2s_words]
        if not (DIR_TRAIN_DICT is None or DIR_TRAIN_DICT == "None"):

            wvs_l1_prompt = torch.index_select(wvs_l1,0,torch.tensor(s2t_ids))
            wvs_l2_prompt = torch.index_select(wvs_l2,0,torch.tensor(t2s_ids))

    
    if DIR_TRAIN_DICT is None or DIR_TRAIN_DICT == "None":
        save_dict = {}
        save_dict["s2t_prompt_dict"] = s2t_prompt_dict
        save_dict["t2s_prompt_dict"] = t2s_prompt_dict
        save_dict["s2t_prompt_dict_in_context"] = None
        save_dict["t2s_prompt_dict_in_context"] = None

    else:
 
        s2t_prompt_dict_in_context = {}
        t2s_prompt_dict_in_context = {}
    
       
        len_ = 15
        if args.random:
            idxs_s2t_prompt = torch.topk(torch.randn(wvs_l1_prompt.size(0), wvs_l1_train.size(0)), len_*6, dim=1, largest=True, sorted=True).indices.tolist()
        else:
            idxs_s2t_prompt = torch.topk(wvs_l1_prompt @ wvs_l1_train.T, len_*6, dim=1, largest=True, sorted=True).indices.tolist()
        for i,w in enumerate(s2t_words):
            if w not in s2t_prompt_dict_in_context:
                s2t_prompt_dict_in_context[w] = []
                j = 0
                left_seen = set()
                left_seen.add(w)
                while len(s2t_prompt_dict_in_context[w])<len_:
                    left = id2w_l1[idxs_s2t_prompt[i][j]]
                    right = s2t_train_dict[left][0]
                    if left not in left_seen:
                        s2t_prompt_dict_in_context[w].append((left,right))
                        left_seen.add(left)
                    j+=1
    
    
        if args.random:                 
            idxs_t2s_prompt = torch.topk(torch.randn(wvs_l2_prompt.size(0), wvs_l2_train.size(0)), len_*6, dim=1, largest=True, sorted=True).indices.tolist()
        else:
            idxs_t2s_prompt = torch.topk(wvs_l2_prompt @ wvs_l2_train.T, len_*6, dim=1, largest=True, sorted=True).indices.tolist()   
        for i,w in enumerate(t2s_words):
            if w not in t2s_prompt_dict_in_context:
                t2s_prompt_dict_in_context[w] = []
                j = 0
                left_seen = set()
                left_seen.add(w)
                while len(t2s_prompt_dict_in_context[w])<len_:
                    left = id2w_l2[idxs_t2s_prompt[i][j]]
                    right = t2s_train_dict[left][0]
                    if left not in left_seen:
                        t2s_prompt_dict_in_context[w].append((left,right))
                        left_seen.add(left)
                    j+=1
    
        # Finally, save all the dicts.
    
        save_dict = {}
        save_dict["s2t_prompt_dict"] = s2t_prompt_dict
        save_dict["t2s_prompt_dict"] = t2s_prompt_dict
        save_dict["s2t_prompt_dict_in_context"] = s2t_prompt_dict_in_context
        save_dict["t2s_prompt_dict_in_context"] = t2s_prompt_dict_in_context
    
    
    save_pkl = args.save_dir

    with open(save_pkl, 'wb') as f:
        pkl.dump(save_dict, f)    
    
    end_time = time.time()
    print("Total Runtime :", end_time-start_time)
