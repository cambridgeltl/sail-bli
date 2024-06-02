import sys
import argparse
import pickle as pkl
import os
import numpy as np
from util import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['TRANSFORMERS_CACHE'] = '/media/cache/'
from tqdm import tqdm
from model_wrapper import Model_Wrapper

def inference(model, tokenizer, prompt_dict, args, target_voc, max_len=5, num_seq=5):
    # Input: prompt_dict
    #            key: source word
    #            value: prompt (input to transformer models)
    # Output: return_dict
    #             key: source word
    #             value: predicted target word
    return_dict = {}
    source_words, prompts = zip(*prompt_dict.items())
    predictions = []
    if "mbart" in args.model_name:
        add_spec_toks = False
    else:
        add_spec_toks = True 
    if ("<extra_id_0>" in prompts[0]) or ("<mask>" in prompts[0]):
        span_mask = True
    else:
        span_mask = False   
    if span_mask or ("t5" in args.model_name) or ("t0" in args.model_name):
        max_len = 5

    if (("xglm" in args.model_name) or ("GPT" in args.model_name) or ("llama" in args.model_name) or ("Llama" in args.model_name)) and (args.batch_eval == 1):
        dynamic_max_len = True
    else:
        dynamic_max_len = False

    for i in tqdm(np.arange(0, len(source_words), args.batch_eval)):

        # i ~ i+args.batch_eval
        TXT = prompts[i:i+args.batch_eval] 
        input_encs = tokenizer.batch_encode_plus(TXT,padding=True,add_special_tokens=add_spec_toks,return_tensors="pt")   
        if dynamic_max_len: 
            max_len = 5 + input_encs.input_ids.size(1) 
        input_encs_cuda = {k:v.cuda() for k,v in input_encs.items()}     
        outputs = model.generate(**input_encs_cuda,num_beams=num_seq, num_return_sequences=num_seq,max_length=max_len,do_sample=False)
        output_sents = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        output_grouped = [output_sents[i*num_seq:(i+1)*num_seq] for i in range((len(output_sents)//num_seq))]  
        predictions.extend(output_grouped)

    # Extract Answers From Predicted Sequences
    assert len(source_words) == len(predictions)
    for i in tqdm(range(len(source_words))):
        candidates = predictions[i]   
        for seq in candidates:
            if "mt5" in args.model_name:
                txt_body = seq.split("<extra_id_0>")[1].split("<extra_id_1>")[0].strip()
                try: word_predict = txt_body.split()[0]
                except: word_predict = None
            elif "mbart" in args.model_name:
                try: word_predict = seq.split(prompt_dict[source_words[i]][6:-4])[1].split()[0]
                except: word_predict = None
            elif "flan-t5" in args.model_name:
                try: word_predict = seq.split()[0]
                except: word_predict = None
            elif "t0" in args.model_name:
                try: word_predict = seq.split()[0]
                except: word_predict = None
            else: # GPT models
                try: word_predict = seq.split(prompt_dict[source_words[i]])[1].split()[0]
                except: word_predict = None
            # remove the dots "..." in "word..."
            if word_predict is not None:
                word_predict = word_predict.split(".")[0]             
                if len(word_predict) == 0:
                    word_predict = None
            if word_predict is not None:
                word_predict = word_predict.lower()    
                if word_predict in target_voc:
                    dict_as(return_dict, source_words[i], word_predict)
                elif (len(word_predict) > 1) and (word_predict[-1] == ".") and (word_predict[:-1] in target_voc):
                    dict_as(return_dict, source_words[i], word_predict[:-1])
    return return_dict


def count_parameters(m):
    num_all = sum(p.numel() for p in m.parameters())
    num_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return (num_all, num_trainable)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer Seed Pairs')
    parser.add_argument("--l1", type=str, default=" ",
                    help="Language (string)")
    parser.add_argument("--l2", type=str, default=" ",
                    help="Language (string)")
    parser.add_argument("--data_dir", type=str, default="./",
                    help="data_dir")    
    parser.add_argument("--save_dir", type=str, default="./",
                    help="save_dir")
    parser.add_argument("--source_data", type=str, default="./",
                    help="source_data")
    parser.add_argument("--model_name", type=str, default="./",
                    help="mt5,mbart,byt5,flant5,xglm,mgpt...")
    parser.add_argument("--print_every", type=int, default=25,
                    help="Print every k training steps")
    parser.add_argument("--eval_every", type=int, default=50,
                    help="evaluate model every k training steps")
    parser.add_argument("--batch_eval", type=int, default=100,
                    help="batch size evaluation")
    parser.add_argument("--batch_train", type=int, default=100,
                    help="batch size train")
    parser.add_argument("--n_shot", type=int, default=0,
                    help="0,1,2,3,4,5,6,7,8,9,10")    
    parser.add_argument('--finetune',  action="store_true")
    parser.add_argument('--best_template',  action="store_true")
    parser.add_argument('--max_length', default=6, type=int)
    parser.add_argument('--template_id', default=0, type=int)
    start_time = time.time()
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    args_dict = vars(args)
    print("Entering Main")

    if ("xglm" in args.model_name) or ("GPT" in args.model_name) or ("llama" in args.model_name) or ("Llama" in args.model_name):
        args.batch_eval = 1
    elif "xl" in args.model_name:
        args.batch_eval = 8

    args.str2lang = {"hr":"Croatian", "en":"English","fi":"Finnish","fr":"French","de":"German","it":"Italian","ru":"Russian","tr":"Turkish","bg":"Bulgarian","ca":"Catalan","hu":"Hungarian"}
    args.str2mbart_code = {"hr":"hr_HR", "en":"en_XX","fi":"fi_FI","fr":"fr_XX","de":"de_DE","it":"it_IT","ru":"ru_RU","tr":"tr_TR"}
 
    ### Load Data

    f_name = args.source_data + "{}_vocabulary.pkl".format(args.l1)
    with open(f_name,"rb") as f:
        voc_l1 = pkl.load(f)
    f_name = args.source_data + "{}_vocabulary.pkl".format(args.l2)
    with open(f_name,"rb") as f:
        voc_l2 = pkl.load(f)


    with open(args.data_dir,"rb") as f:
        data = pkl.load(f)


    s2t_prompt_dict = data["s2t_prompt_dict"]
    t2s_prompt_dict = data["t2s_prompt_dict"]
    s2t_prompt_dict_in_context = data["s2t_prompt_dict_in_context"]
    t2s_prompt_dict_in_context = data["t2s_prompt_dict_in_context"]

    if s2t_prompt_dict_in_context is None:
        args.n_shot = 0

    ### Load Model
    model_wrapper = Model_Wrapper()
    model, tokenizer = model_wrapper.load_model(
        path=args.model_name,
        max_length=args.max_length
    )
    
    ### Get Templates
    if args.best_template: 
        templates_s2t = get_best_template(args, args.str2lang[args.l1], args.str2lang[args.l2], args.l2)
        templates_t2s = get_best_template(args, args.str2lang[args.l2], args.str2lang[args.l1], args.l1)
    else:
        templates_s2t = get_templates(args, args.str2lang[args.l1], args.str2lang[args.l2], args.l2)
        templates_t2s = get_templates(args, args.str2lang[args.l2], args.str2lang[args.l1], args.l1)
    assert len(templates_s2t) == len(templates_t2s)
    print(len(templates_s2t)," Source to Target Templates and ",len(templates_t2s)," Target to Source Templates." )
    sys.stdout.flush()

    if "mbart" in args.model_name:
        add_spec_toks = False
    else:
        add_spec_toks = True

    for template_ids in range(len(templates_s2t)): 
        template_s2t, template_t2s = templates_s2t[template_ids], templates_t2s[template_ids]
        template_line = f"Template {template_ids} {template_s2t}"
        print(template_line)
        max_len_s2t = tokenizer([template_s2t], add_special_tokens=add_spec_toks, return_tensors="pt").input_ids.size(1) + 20
        max_len_t2s = tokenizer([template_t2s], add_special_tokens=add_spec_toks, return_tensors="pt").input_ids.size(1) + 20

        ### Apply Templates
        s2t_test_prompt = apply_template(template_s2t, s2t_prompt_dict, s2t_prompt_dict_in_context, args)
        t2s_test_prompt = apply_template(template_t2s, t2s_prompt_dict, t2s_prompt_dict_in_context, args)
    
        ### S2T BLI Evaluation
        model.eval()
        with torch.no_grad():
            inferred_dict_s2t = inference(model, tokenizer, s2t_test_prompt, args, voc_l2, max_len=max_len_s2t, num_seq=5) 
        ### T2S BLI Evaluation
        model.eval()
        with torch.no_grad():
            inferred_dict_t2s = inference(model, tokenizer, t2s_test_prompt, args, voc_l1, max_len=max_len_t2s, num_seq=5)  
    
    # save inferred dicts
    save_dict = (inferred_dict_s2t, inferred_dict_t2s)
    save_pkl = args.save_dir
    with open(save_pkl, 'wb') as f:
        pkl.dump(save_dict, f)    

    end_time = time.time()
    print("Total Runtime :", end_time-start_time)
    sys.stdout.flush()
