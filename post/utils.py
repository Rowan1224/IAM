import json
import editdistance
from transformers import pipeline
import os

def init(vocabDir):

    bert_model = pipeline('fill-mask', model='bert-large-cased')    
    bert = os.path.join(vocabDir,'bert-large-cased-vocab.txt')
    english = os.path.join(vocabDir,'words_dictionary.json')

    bert_vocab = []
    with open(bert,'r') as file:
        bert_vocab = [line.strip() for line in file]

    with open(english,'r') as json_file:
        english_words_set = json.load(json_file)

    english_words_set = list(english_words_set.keys())

    return bert_model, bert_vocab, english_words_set

def probe(sentence: str, candidates, model) :


    scores = []

    for res in model(sentence,targets=candidates):
        scores.append((res['token_str'],res['score']))


    return sorted(scores, key = lambda x: x[1],reverse=True)[0][0]



def get_correct_candidates(corrupt_str, vocab, min=3):


    'returns a list of candidate words that needs maximum 3 (min) edits from the corrupted string'

    min = 3
    candidates = []
    for word in vocab:
        d = editdistance.eval(word,corrupt_str) 
        if d <= min:

            if word in vocab:
                candidates.append(word)


    if len(candidates)==0:
        return candidates.append(corrupt_str)
    
    return candidates

def get_correct_word(corrupt_str, vocab, min = 5):

    'returns the correct word with minimun number of edits'

    clean = ''
    
    for word in vocab:
        d = editdistance.eval(word,corrupt_str) 
        if min > d:
            min = d
            clean = word


    if clean =='':
        return corrupt_str

    
    return clean

def get_best_candidate(sen, pos, candidates, model):

    'returns the edited sentence after selecting bres replacement for each incorrect word'

    

    best_candidates = {}
    for i,c in enumerate(candidates):
        inp = sen.split(" ")
        inp[pos[i]] ='[MASK]'
        inp = " ".join(inp)
        b = probe(inp,c,model)
        best_candidates[pos[i]]=b
        

    sen = sen.split(" ")
    for i,v in best_candidates.items():
        sen[i] = v

    return " ".join(sen)



def edit_neuspell(corrupt_sen, pred_sen, english_words_set):

    ''' compare the actual courrepted sentence with prediction from neuspell model 
    and only replace the word in courrepted sentence that is not found in vocab

    '''

    output = []
    corrupt_sen = corrupt_sen.split(" ")
    pred_sen = pred_sen.split(" ")
    for i,word in enumerate(corrupt_sen):
        w = word
        if w.lower() not in english_words_set:
            if i < len(pred_sen):
                output.append(pred_sen[i])
            else:
                output.append(word)
        else:
            output.append(word)

    return " ".join(output)



def edit_candidate(corrupt_sen, english_words_set, model):

    'returns the edited sentence based on finding best replacement out of a candidate list'

    output = []
    corrupt_sen = corrupt_sen.split(" ")
    candidates = []
    count = 0
    for i,word in enumerate(corrupt_sen):
        w = word
        if w.lower() not in english_words_set and len(w)>1:
            # output.append(correct_string(w))
            
            count+=1
            c = get_correct_candidates(word)
            if c is not None:
                output.append(i)
                candidates.append(c)

    
    if count>1 and len(candidates)>0:
        output = get_best_candidate(" ".join(corrupt_sen),output,candidates, model)
    elif count==1 and len(candidates)>0:
        output = get_best_candidate(" ".join(corrupt_sen),output,candidates, model)
    else:
        output = " ".join(corrupt_sen)

    return output


def edit_brute(image, vocab, english_words_set):

    output = []
    image = image.split(" ")

    for i,word in enumerate(image):
        w = word
        if w.lower() not in english_words_set and len(w)>1:
            output.append(get_correct_word(word,vocab))
        else:
            output.append(word)


    output = " ".join(output)

    return output

