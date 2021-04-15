import json
import re 
import numpy as np 
import tqdm

# REF: https://github.com/chenyuntc/pytorch-book/blob/master/chapter09-neural_poet_RNN/data.py

def handle(para):
    result, _ = re.subn("（.*）", "", para)
    result, _ = re.subn("{.*}", "", result)
    result, _ = re.subn("《.*》", "", result)
    result, _ = re.subn("[\]\[]", "", result)
    r = ""
    for s in result:
        if s not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
            r += s
    r, _ = re.subn("。。", "。", r)
    return r

#para = "-181-村橋路不端，數里就迴湍。積壤連涇脉，高林上笋竿。早嘗甘蔗淡，生摘琵琶酸。（「琵琶」，嚴壽澄校《張祜詩集》云：疑「枇杷」之誤。）好是去塵俗，煙花長一欄。"
#print(handle(para))

#保存唐诗
data7 = []
data5 = []
for i in tqdm.tqdm(range(20)):
    path = "./json/poet.tang." + str(i*1000) + ".json"
    with open("./json/poet.song.0.json","rb") as f:
        jsondata = json.loads(f.read())
        for poetry in jsondata:
            p = poetry.get("paragraphs")
            for s in p:
                s = handle(s)
                if s == "":
                    continue
                ss = re.split('。',s)
                for w in ss:
                    if len(w) == 15 and w[7] == '，':
                        data7.append(list(w))
                    elif len(w) == 11 and w[5] == '，':
                        data5.append(list(w))

def vec2poem(s,ix2word):
    ns = ""
    for x in s:
        if x != "</s>" and x != "<END>" and x != "<STRAT>":
            ns = ns + ix2word[x]
    return ns 

def processData(data,path):
    words = {_word for _sentence in data for _word in _sentence}
    word2ix = {_word: _ix for _ix, _word in enumerate(words)}
    word2ix['<END>'] = len(word2ix)  
    word2ix['<START>'] = len(word2ix)  
    word2ix['</s>'] = len(word2ix) 
    ix2word = {_ix: _word for _word, _ix in list(word2ix.items())}

    for i in range(len(data)):
        data[i] = ["<START>"] + data[i] + ["<END>"]

    data = [[word2ix[_word] for _word in _sentence] for _sentence in data]

    print(len(data))
    print(vec2poem(data[2],ix2word))
    print(vec2poem(data[3],ix2word))

    np.save(path + "data.npy", data)   
    np.save(path + "ix2word.npy", ix2word)   
    np.save(path + "word2ix.npy", word2ix)   

processData(data7, "./datasets/tang7/")
processData(data5, "./datasets/tang5/")

 
