import os
import sys


lang_pairs = [('de', 'fr'),('bg', 'ca')]

Model = "meta-llama/Llama-2-13b-hf" 
size_train = 0  # Seed dictionary size
n_shot = 0  # Number of in-context examples. Zero-shot prompting (also known as unsupervised BLI in previous BLI work): n_shot=0.
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


    Dtrain_dir = None
    test_prompt_dict_dir = TMP_DIR+"{}2{}_test_prompt_{}.pkl".format(lang1, lang2, size_train)


    os.system('python ./src/extract_bli_test_data.py --l1 {} --l2 {} --emb_src_dir {} --emb_tgt_dir {} --train_dict_dir {} --test_dict_dir {} --save_dir {} --source_data {}'.format(lang1, lang2, ROOT_EMB_SRC, ROOT_EMB_TRG, Dtrain_dir, ROOT_TEST_DICT, test_prompt_dict_dir, DATA_ROOT))
    os.system('python ./src/main.py --l1 {} --l2 {} --model_name {} --train_size {} --n_shot {} --data_dir {} --test_dict_dir {} --best_template'.format(lang1, lang2, Model, size_train, n_shot, DATA_ROOT, test_prompt_dict_dir))
