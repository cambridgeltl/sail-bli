import os
import sys
langs = ['de','fr','it','ru','en']

for lang in langs:
    print(lang)
    sys.stdout.flush()
 
    EMB_DIR = "/media/data/WES/fasttext.wiki.{}.300.vocab_200K.vec".format(lang)
    SAVE_ROOT = "/media/data/T2TData/" # save dir

    os.system('python ./src/extract_vocabularies.py --lang {} --emb_dir {} --save_dir {}'.format(lang, EMB_DIR, SAVE_ROOT))


import os
import sys
langs = ['bg','ca','hu']

for lang in langs:
    print(lang)
    sys.stdout.flush()
    EMB_DIR = "/media/data/WESPLX/fasttext.cc.{}.300.vocab_200K.vec".format(lang) 
    SAVE_ROOT = "/media/data/T2TData/" # save dir

    os.system('python ./src/extract_vocabularies.py --lang {} --emb_dir {} --save_dir {}'.format(lang, EMB_DIR, SAVE_ROOT))