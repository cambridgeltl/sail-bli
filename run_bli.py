import os
import sys

lang_pairs = [('de', 'fr'),('bg', 'ca')]

XLING = set(["en","de","fr","it","ru","tr","hr","fi"])
PanLex = set(["bg","ca","hu"])

Model = "meta-llama/Llama-2-13b-hf"
size_train = 5000  # Seed dictionary size
n_shot = 5  # Number of in-context examples. Zero-shot prompting (also known as unsupervised BLI in previous BLI work): n_shot=0.
n_iter = 4
topk = 1
DATA_ROOT = "/media/data/T2TData/"
SAVE_ROOT = "/media/data/T2TModel/" # save dir 

TMP_DIR = "./TMP/"
os.system("rm -rf {}".format(TMP_DIR))
os.system("mkdir {}".format(TMP_DIR))


for (lang1, lang2) in lang_pairs:
    print(lang1, lang2)
    sys.stdout.flush()
    # --best_template
    if lang1 in XLING:
        ROOT_EMB_SRC = "/media/data/WES/fasttext.wiki.{}.300.vocab_200K.vec".format(lang1)
        ROOT_EMB_TRG = "/media/data/WES/fasttext.wiki.{}.300.vocab_200K.vec".format(lang2)
        ROOT_TEST_DICT = "/media/data/xling-eval/bli_datasets/{}-{}/yacle.test.freq.2k.{}-{}.tsv".format(lang1, lang2, lang1, lang2)
    else: 
        ROOT_EMB_SRC = "/media/data/WESPLX/fasttext.cc.{}.300.vocab_200K.vec".format(lang1)
        ROOT_EMB_TRG = "/media/data/WESPLX/fasttext.cc.{}.300.vocab_200K.vec".format(lang2)
        ROOT_TEST_DICT = "/media/data/panlex-bli/lexicons/all/{}-{}/{}-{}.test.2000.cc.trans".format(lang1, lang2, lang1, lang2)
    for i_iter in range(n_iter):
        if i_iter > 0:
            Dtrain_dir_previous = TMP_DIR + "{}2{}_Dtrain_{}.txt".format(lang1, lang2, i_iter - 1)
        else:
            Dtrain_dir_previous = None
        forward_prompt_dict_dir = TMP_DIR+"{}2{}_forward_prompt_{}.pkl".format(lang1, lang2, i_iter)
        backward_prompt_dict_dir = TMP_DIR+"{}2{}_backward_prompt_{}.pkl".format(lang1, lang2, i_iter)
        forward_results = TMP_DIR+"{}2{}_forward_inference_{}.pkl".format(lang1, lang2, i_iter)
        backward_results = TMP_DIR+"{}2{}_backward_inference_{}.pkl".format(lang1, lang2, i_iter)
        test_prompt_dict_dir = TMP_DIR+"{}2{}_test_prompt_{}.pkl".format(lang1, lang2, i_iter)
        ###Step1.construct prompt for Dtrain-forward:
        #    if Dtrain exists:
        #        few-shot prompting
        #    else:
        #        zero-shot prompting
        #    input: Dtrain in the previous iteration ; output: forward_prompt_dict
        print("###### FORWARD INFERENCE")
        sys.stdout.flush()
        os.system('python ./src/extract_bli_train_data.py --direction {} --l1 {} --l2 {} --train_size {} --emb_src_dir {} --emb_tgt_dir {} --train_dict_dir {} --save_dir {} --source_data {}'.format("forward", lang1, lang2, size_train, ROOT_EMB_SRC, ROOT_EMB_TRG, Dtrain_dir_previous, forward_prompt_dict_dir, DATA_ROOT))
        ###Step2.LLM inference (forward)
        #    input: forward_prompt_dict; output: forward_results (output dict)
        os.system('python ./src/infer_seed_dic.py --l1 {} --l2 {} --model_name {} --n_shot {} --data_dir {} --save_dir {} --source_data {} --best_template'.format(lang1, lang2, Model,  n_shot, forward_prompt_dict_dir, forward_results, DATA_ROOT))

        ###Step3.construct prompt for Dtrain-backward:
        #    if Dtrain exists:
        #        few-shot prompting
        #    else:
        #        zero-shot prompting
        #    input: Dtrain in the previous iteration,  forward_results; output: backward_prompt_dict_dir
        print("###### BACKWARD INFERENCE")
        sys.stdout.flush()
        os.system('python ./src/extract_bli_train_data.py --direction {} --l1 {} --l2 {} --forward_inference_results {} --train_size {} --emb_src_dir {} --emb_tgt_dir {} --train_dict_dir {} --save_dir {} --source_data {}'.format("backward", lang1, lang2, forward_results, size_train, ROOT_EMB_SRC, ROOT_EMB_TRG, Dtrain_dir_previous, backward_prompt_dict_dir, DATA_ROOT))
        ###Step4.LLM inference (backward)
        #    input: backward_prompt_dict_dir; output: backward_results (output dict)
        os.system('python ./src/infer_seed_dic.py --l1 {} --l2 {} --model_name {} --n_shot {} --data_dir {} --save_dir {} --source_data {} --best_template'.format(lang1, lang2, Model, n_shot, backward_prompt_dict_dir, backward_results, DATA_ROOT))

        ###Step3.Update Dtrain
        #    input: forward_results and backward_results; output:  new Dtrain
        print("###### UPDATE DTRAIN")
        Dtrain_dir_current = TMP_DIR + "{}2{}_Dtrain_{}.txt".format(lang1, lang2, i_iter)
        sys.stdout.flush()
        os.system('python ./src/update_train_dict.py --forward_data_dir {} --backward_data_dir {} --topk {} --save_dir {}'.format(forward_results, backward_results, topk, Dtrain_dir_current))


        ###Step4.construct prompt for Dtest:
        #    if Dtrain exists:
        #       few-shot prompting
        #    else:
        #       zero-shot prompting
        #    input: new Dtrain ; output:  test_prompt_dict
        print("###### EVALUATION")
        sys.stdout.flush()
        os.system('python ./src/extract_bli_test_data.py --l1 {} --l2 {} --emb_src_dir {} --emb_tgt_dir {} --train_dict_dir {} --test_dict_dir {} --save_dir {} --source_data {}'.format(lang1, lang2, ROOT_EMB_SRC, ROOT_EMB_TRG, Dtrain_dir_current, ROOT_TEST_DICT, test_prompt_dict_dir, DATA_ROOT))

        ###Step5. LLM inference and eval 
        #    input: test_prompt_dict ; output:  metric scores
        os.system('python ./src/main.py --l1 {} --l2 {} --model_name {} --train_size {} --n_shot {} --data_dir {} --test_dict_dir {} --i_iter {} --best_template'.format(lang1, lang2, Model, size_train, n_shot, DATA_ROOT, test_prompt_dict_dir, i_iter))