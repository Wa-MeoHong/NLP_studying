# 정수 인코딩 (단어를 딕셔너리화 하여 숫자를 부여함)
# 조건 : 단어 중, 불용어를 제외하고, 1-2개의 길이를 가진 단어는 제외시킨다.
#        조건에 부합하는 단어 중, 빈도수가 1이하인 단어는 숫자 부여하지 않고 제외한다.

# 1. 딕셔너리를 이용해서 인코딩

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

raw_text = "A barber is a person. a barber is good person. a barber is huge \
person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. \
His barber kept his word. a barber kept his word. His barber kept his secret\
. But keeping and keeping such a huge secret to himself was driving the \
barber crazy. the barber went up a huge mountain."

sentences = sent_tokenize(raw_text)  # 문장 토큰화

vocab = {} # 딕셔너리
preprocessed_sentences = [] # 리스트
stop_words = set(stopwords.words('english')) # 불용어 리스트를 모두 들고옴

for sentence in sentences:
  # 각 문장마다 진행
  tokenized_sentence = word_tokenize(sentence)  #단어 토큰화
  result = []
  for word in tokenized_sentence:
    # 만약 단어가 불용어가 아니고, 3개 이상의 문자로 이뤄진 단어이면 
    # 결과 리스트에 word를 넣고, 만약 vocab(단어)에 리스트에 포함되어있지 않다면, 0으로 초기화하고, vocab가 몇번 나왔는지 센다
    word = word.lower() # 안하면 같은 단어라도 대소문자 구분때문에 다른단어로 인식
    if (word not in stop_words):
      if (len(word) > 2):
        result.append(word)
        # 맨 처음에 vocab에 word가 없는 경우, 딕셔너리에 추가해줌
        if (word not in vocab):
          vocab[word] = 0
        # 단어가 등장했으므로 등장 횟수를 체크
        vocab[word] += 1
  preprocessed_sentences.append(result) # 토큰화하고, 정제된 문장에 포함된 단어들의 리스트를  preprocessed_sentences에 담아줌
print(preprocessed_sentences) # 출력
#print("단어 집합 : ", vocab)    # 결과 출력시 대소문자가 구분됨을 볼 수 있음 (huge, Huge가 다르게 카운트됨)

vocab_sorted = sorted(vocab.items(), key=lambda x:x[1], reverse=True) 
# 내림차순으로 정렬, value(빈도수)의 값에 따라 정렬을 함
#print("정렬 : ",vocab_sorted)

# 정수 부여
word_to_index = {}
i = 0
# 정렬된 vocab중, 빈도수의 값에 따라 정수를 부여함. 1부터 시작함
for (word, frequency) in vocab_sorted :
  if ( frequency > 1):
    i = i + 1
    word_to_index[word] = i
#print("번호 부여 : " ,word_to_index)


# if, 빈도수 상위 5개의 단어만 사용하고 싶다면, 
vocab_size = 5
# 빈도수가 많은 단어를 가져와서 튜플 형태로 저장 후, 딕셔너리화 시킴
word_freq = [(word, index) for(word,index) in word_to_index.items() if index < vocab_size+1]
word_to_index = dict(word_freq)

# 사실 밑의 방법이 더 빠름 이유는 빈도수가 적은 것이 적기 때문
# 만약 단어의 수가 많다면 위 방법이 더 효율적일 수 있음
# word_freq = [word for (word,index) in word_to_index.items() if index >= vocab_size+1]
# for w in word_freq:
#   del word_to_index[w]
#print("상위 5개 빈도수 단어 : ", word_to_index)

# 상위 5개의 빈도수 단어를 제외한 모든 단어를 Out-of-Vocabulary로 정함

# OOV의 데이터를 딕셔너리에 추가, 맨마지막 번호로 주어지기에 word_to_index의 길이 + 1이 된다.
word_to_index['OOV'] = len(word_to_index) + 1
#print("OOV 추가 : ", word_to_index)

# 단어 인코딩(정수로 변환)
encoded_setences = []
# 각 문장마다 진행
for sentence in preprocessed_sentences:
  encoded_setence = []
  # 정제된 토큰화된 단어에
  for word in sentence:
    # 일단, word의 번호를 넣어주고, 만약 KeyError(word_to_index의 키가 아닌 경우), OOV의 값을 넣어줌
    try:
      encoded_setence.append(word_to_index[word])
    except KeyError:
      encoded_setence.append(word_to_index['OOV'])
  # encode가 끝나면 리스트에 encoded된 문장을 넣어줌(리스트형식으로 정리됨)
  encoded_setences.append(encoded_setence)
print("encoded sentences : ", encoded_setences)

"""
[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
encoded sentences :  [[1, 5], [1, 6, 5], [1, 3, 5], [6, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [6, 6, 3, 2, 6, 1, 6], [1, 6, 3, 6]]
"""

#==================================================================
# 2.Counter 모듈을 이용해서 정수 인코딩

#위의 문장을 정제한 후, Counter 모듈을 이용해 정수 인코딩 딕셔너리를 만듦

from collections import Counter
import numpy as np

# numpy를 통해 preprocessed_sentences(정제된 단어들의 리스트)를 numpy행렬로 변환가능 (1차원 배열형태)
words = np.hstack(preprocessed_sentences)
# Counter 모듈을 통해 words안에 든 단어의 빈도수를 측정 가능
vocab = Counter(words)
print(vocab)
vocab_size = 5
vocab = vocab.most_common(vocab_size)   # value가 가장 높은 상위 5개의 단어만 추출해서 다시 저장
print(vocab)

# 딕셔너리화
word_to_index = {}
i = 0
for (word, frequency) in vocab :
    i = i + 1
    word_to_index[word] = i
#print("번호 부여 : " ,word_to_index)
# 상위 5개의 빈도수 단어를 제외한 모든 단어를 Out-of-Vocabulary로 정함
# OOV의 데이터를 딕셔너리에 추가, 맨마지막 번호로 주어지기에 word_to_index의 길이 + 1이 된다.
word_to_index['OOV'] = len(word_to_index) + 1
print(word_to_index)

#이후 문장 정수 인코딩은 동일 

"""
Counter({'barber': 8, 'secret': 6, 'huge': 5, 'kept': 4, 'person': 3, 'word': 2, 'keeping': 2, 'good': 1, 'knew': 1, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1})
[('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'OOV': 6}
"""

#=======================================================================
# 3. FreqDist를 이용한 정수 인코딩
# 3번째 방법 : FreqDist이용 (enumrate도 사용 (리스트의 순서에 맞게 번호를 부여하는 함수))

from nltk import FreqDist
import numpy as np

words = np.hstack(preprocessed_sentences)
vocab = FreqDist(words)   # FreqDist를 통해 딕셔너리화 함
print(dict(vocab))

vocab_size = 5
vocab = vocab.most_common(vocab_size)   # value가 가장 높은 상위 5개의 단어만 추출해서 다시 저장

# enumerate (index를 부여함(enumerate는 인덱스가 0번부터 시작해서 1을 더해줘서 정수 부여))
word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}
# 상위 5개의 빈도수 단어를 제외한 모든 단어를 Out-of-Vocabulary로 정함

# OOV의 데이터를 딕셔너리에 추가, 맨마지막 번호로 주어지기에 word_to_index의 길이 + 1이 된다.
word_to_index['OOV'] = len(word_to_index) + 1
print(word_to_index)

# 이하 동일

"""
{'barber': 8, 'person': 3, 'good': 1, 'huge': 5, 'knew': 1, 'secret': 6, 'kept': 4, 'word': 2, 'keeping': 2, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1}
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'OOV': 6}
"""

#================================================================
# 4번째 방법 : Keras (텐서플로우 이용)
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

# fit_on_texts( 입력받은 텍스트로부터 빈도수가 높은 순서대로 인덱스번호 부여함. 1부터 시작)
# word_index(인덱스를 부여받은 단어의 딕셔너리)
# word_counts (단어의 빈도수를 튜플의 리스트 형태로 표현)

tokenizer.fit_on_texts(preprocessed_sentences)
print(tokenizer.word_index)
print(tokenizer.word_counts)
# text_to_sequences (각 단어에 부여된 번호로 preprocesssed된 단어를 인코딩함 )
print(tokenizer.texts_to_sequences(preprocessed_sentences))

# if 상위 5개의 단어만 사용
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 1) # 상위 5개만 사용하지만 숫자 0를 고려하여 실제 개수는 6개를 사용용
tokenizer.fit_on_texts(preprocessed_sentences)
# 이렇게 해도 word_index와 word_count에는 적용이 X, 실제 적용은 texts_to_sequences에서 적용
#print(tokenizer.word_index)
#print(tokenizer.word_counts)

# 숫자 0를 고려하는 이유 = 패딩의 과정을 처리할 때, 필요함

# 적용된 모습에서는 상위 5개의 단어만 인코딩 되고 나머지 단어는 인코딩을 시키지 않았음(인코딩에서 제외했기 때문에 없는 걸로 표현)
print("상위 5개의 단어만 남긴 단어를 인코딩 : ",tokenizer.texts_to_sequences(preprocessed_sentences))
 
# 만약 숫자 0를 고려하지 않고, 순수하게 단어들만 남기고 싶다면 다음과 같이 진행해도 된다.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)
vocab_size = 5
words_freq = [word for word, index in tokenizer.word_index.items() if index >= vocab_size+1]
# 상위 5개의 단어를 제외한 모든 단어의 리스트를 받아와서
for word in words_freq:
  del tokenizer.word_index[word]
  del tokenizer.word_counts[word]     # 토크나이저에 있는 word_index와 word_counts에 값을 전부 지워준다..
print()
print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(preprocessed_sentences))

# 만약 OOV를 고려하게 된다면 다음과 같이 한다. (num_words는 +2 해야함)
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 2, oov_token = 'OOV')    # oov_token을 사용하여, 빈도수가 상위 5개보다 적은 단어들은 전부 OOV로 간주함함
# 참고로 oov_token을 사용한다면 oov_token의 인덱스는 1로 한다. (즉 , 상위 5개의 빈도수 단어들의 시작이 2번부터 시작함)
tokenizer.fit_on_texts(preprocessed_sentences)
print()
print("OOV를 고려한 토크나이저: ",tokenizer.texts_to_sequences(preprocessed_sentences))

"""
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7, 'good': 8, 'knew': 9, 'driving': 10, 'crazy': 11, 'went': 12, 'mountain': 13}
OrderedDict([('barber', 8), ('person', 3), ('good', 1), ('huge', 5), ('knew', 1), ('secret', 6), ('kept', 4), ('word', 2), ('keeping', 2), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)])
[[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]
상위 5개의 단어만 남긴 단어를 인코딩 :  [[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2], [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]

{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
OrderedDict([('barber', 8), ('person', 3), ('huge', 5), ('secret', 6), ('kept', 4)])
[[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2], [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]

OOV를 고려한 토크나이저:  [[2, 6], [2, 1, 6], [2, 4, 6], [1, 3], [3, 5, 4, 3], [4, 3], [2, 5, 1], [2, 5, 1], [2, 5, 3], [1, 1, 4, 3, 1, 2, 1], [2, 1, 4, 1]]
"""

#===========================================================================
# 1번의 방법(DICT를 이용한 방법)을 4번의 방법 (Tensorflow Keras를 이용하는 방법)
# 으로 코드를 간결화 할 수 있음

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer

raw_text = "A barber is a person. a barber is good person. a barber is huge \
person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. \
His barber kept his word. a barber kept his word. His barber kept his secret\
. But keeping and keeping such a huge secret to himself was driving the \
barber crazy. the barber went up a huge mountain."

preprocessed_sentences = [] # 리스트
stop_words = set(stopwords.words('english')) # 불용어 리스트를 모두 들고옴

for sentence in sentences:
  # 각 문장마다 진행
  tokenized_sentence = word_tokenize(sentence)  #단어 토큰화
  result = []
  for word in tokenized_sentence:
    # 만약 단어가 불용어가 아니고, 3개 이상의 문자로 이뤄진 단어이면 
    # 결과 리스트에 word를 넣고, 만약 vocab(단어)에 리스트에 포함되어있지 않다면, 0으로 초기화하고, vocab가 몇번 나왔는지 센다
    word = word.lower() # 안하면 같은 단어라도 대소문자 구분때문에 다른단어로 인식
    if (word not in stop_words):
      if (len(word) > 2):
        result.append(word)
  preprocessed_sentences.append(result) # 토큰화하고, 정제된 문장에 포함된 단어들의 리스트를  preprocessed_sentences에 담아줌
print("정제화된 단어 :",preprocessed_sentences)
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 2, oov_token = 'OOV')    # oov_token을 사용하여, 빈도수가 상위 5개보다 적은 단어들은 전부 OOV로 간주함함
# 참고로 oov_token을 사용한다면 oov_token의 인덱스는 1로 한다. (즉 , 상위 5개의 빈도수 단어들의 시작이 2번부터 시작함)
tokenizer.fit_on_texts(preprocessed_sentences)

print("OOV를 고려한 토크나이저 :",tokenizer.texts_to_sequences(preprocessed_sentences))

"""
정제화된 단어 : [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
OOV를 고려한 토크나이저 : [[2, 6], [2, 1, 6], [2, 4, 6], [1, 3], [3, 5, 4, 3], [4, 3], [2, 5, 1], [2, 5, 1], [2, 5, 3], [1, 1, 4, 3, 1, 2, 1], [2, 1, 4, 1]]
"""

