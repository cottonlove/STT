import re
import os
import pandas as pd


def bracket_filter(sentence, mode='phonetic'):
    new_sentence = str()

    if mode == 'phonetic': #phonetic : 칠 십 퍼센트,  spelling : 70% (퍼센트 vs % 의 차이) -> 이거 파일명으로 확인.
        flag = False

        for ch in sentence:
            if ch == '(' and flag is False:
                flag = True
                continue
            if ch == '(' and flag is True:
                flag = False
                continue
            if ch != ')' and flag is False:
                new_sentence += ch

    elif mode == 'spelling':
        flag = True

        for ch in sentence:
            if ch == '(':
                continue
            if ch == ')':
                if flag is True:
                    flag = False
                    continue
                else:
                    flag = True
                    continue
            if ch != ')' and flag is True:
                new_sentence += ch

    else:
        raise ValueError("Unsupported mode : {0}".format(mode))

    return new_sentence


def special_filter(sentence, mode='phonetic', replace=None):
    SENTENCE_MARK = ['?', '!', '.']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', ',']

    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                continue

        if ch == '#':
            new_sentence += '샾'

        elif ch == '%':
            if mode == 'phonetic':
                new_sentence += replace
            elif mode == 'spelling':
                new_sentence += '%'

        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence

#이거 왜 안써? -> character base 아닌 phonetic, spelling에서만 쓰는거?
def sentence_filter(raw_sentence, mode, replace=None):
    return special_filter(bracket_filter(raw_sentence, mode), mode, replace)


def load_label(filepath): #labels.csv
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    freq_list = ch_labels["freq"] #안쓰임

    for (id_, char, freq) in zip(id_list, char_list, freq_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        try:
            target += (str(char2id[ch]) + ' ')
        except KeyError:
            continue

    return target[:-1]

#문장부호, 괄호 지우기
def onlyletters(x):
    # 괄호
    a = re.compile(r'\([^)]*\)')
    # 문장 부호
    b = re.compile('[^가-힣 ]')
    x = re.sub(pattern=a, repl='', string= x)
    x = re.sub(pattern=b, repl='', string= x)
    return x


#output_unit = 'character' (kospeech 코드)
#labels.csv 만드는 함수
def generate_character_labels(transcript_df, label_dest): # create vocabulary csv file #원래 labels.csv랑 비교하고 싶음! 사라진 거랑 새로 생긴거. frequency기준으로
    
    print('create_char_labels started..')
    
    label_list = list()
    label_freq = list()
    transcripts = transcript_df['text']
    for transcript in transcripts:
        for ch in transcript:
            if ch not in label_list:
                label_list.append(ch)
                label_freq.append(1)
            else:
                label_freq[label_list.index(ch)] += 1

    # sort together Using zip
    label_freq, label_list = zip(*sorted(zip(label_freq, label_list), reverse=True))
    label = {'id': [0, 1, 2], 'char': ['<pad>', '<sos>', '<eos>'], 'freq': [0, 0, 0]}

    for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
        label['id'].append(idx + 3)
        label['char'].append(ch)
        label['freq'].append(freq)

    label['id'] = label['id'][:2000]
    label['char'] = label['char'][:2000]
    label['freq'] = label['freq'][:2000]

    label_df = pd.DataFrame(label)
    label_df.to_csv(os.path.join(label_dest), encoding="utf-8", index=False)


#transcripts.txt 만드는 함수
def generate_character_script(data_df, labels_dest): #transcript_df, os.getcwd() 

    print('[INFO] create_script started..')

    char2id, id2char = load_label(os.path.join(labels_dest, "yj_labels.csv")) #labels2.csv
    with open(os.path.join(labels_dest,"yj_transcripts.txt"), "w+") as f:
    #with open(os.path.join(labels_dest,"transcripts.txt"), "w+") as f:
        for audio_path, transcript in data_df.values:
            char_id_transcript = sentence_to_target(transcript, char2id)
            f.write(f'{audio_path}\t{transcript}\t{char_id_transcript}\n') #audio_path, 한글 전사, 전사ID tab으로 구분.

#output_unit = 'subword' -> vocab_size 지정해줘야함 (필요시 구현)
#output_unit = 'grapheme' (필요시 구현)

# #main.py에서 불러진다. (yj)
# def preprocessing(ruled_transcript_df, labels_dest):
#     generate_character_script(ruled_transcript_df, labels_dest)
#     print('[INFO] Preprocessing is Done')


#원래꺼
def preprocessing(transcripts_dest, labels_dest): #label_path = os.path.join(DATASET_PATH, 'train', 'train_label'), os.getcwd()
    
    transcript_df = pd.read_csv(transcripts_dest) #이미 하나로 모은 듯! transcript_df: []

    text = transcript_df['text'] #여기에 senetence filter 해주기...?
    
    # # # yj
    # transcript_df['text'] = transcript_df['text'].map(lambda x:rule(x))  #sentence_filter(x, 'phonetic')
    # # # labels2.csv를 새로 만들기 (Vocab)
    # # generate_character_labels(transcripts, opt.vocab_dest) -> 이건 main에서 하자!
    # # generate_character_script(audio_paths, transcripts, opt.vocab_dest)
    # generate_character_script(transcript_df, labels_dest)

    print('[INFO] Preprocessing is Done')