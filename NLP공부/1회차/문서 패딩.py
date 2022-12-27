#==================================================================
# 패딩 (각 문서의 인코딩한 단어들을 행렬로 처리함)

import numpy as np
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

# 케라스를 이용해 정수 인코딩화 시킴
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
print("인코딩화 된 데이터 :", encoded)

#동일한 길이로 맞추기 위해 가장 긴 문장을 찾아낸다,
max_len = max(len(item) for item in encoded)
print("최대 길이 :", max_len)

for sentence in encoded :
  # 만약 길이가 max_len보다 작다면 길이를 맞추기위해 0을 추가함
  while len(sentence) < max_len: 
    sentence.append(0)
# 길이를 맞춘 encoded 2차원 리스트를 numpy 행렬로 변환시킴
padded_np = np.array(encoded)
padded_np

# 0번은 아무 의미 없는 단어로 처리한다(NULL문자로 인식하는 것 같음? )
# 따라서 데이터의 크기를 동일하게 맞추기 위해 특정값을 채워서 크기를 조정하는 것을 
# Padding (패딩)이라 하며, 0으로 맞추는 패딩을 제로 패딩 (zero padding)이라함

"""
정제화된 단어 : [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
인코딩화 된 데이터 : [[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]
최대 길이 : 7
array([[ 1,  5,  0,  0,  0,  0,  0],
       [ 1,  8,  5,  0,  0,  0,  0],
       [ 1,  3,  5,  0,  0,  0,  0],
       [ 9,  2,  0,  0,  0,  0,  0],
       [ 2,  4,  3,  2,  0,  0,  0],
       [ 3,  2,  0,  0,  0,  0,  0],
       [ 1,  4,  6,  0,  0,  0,  0],
       [ 1,  4,  6,  0,  0,  0,  0],
       [ 1,  4,  2,  0,  0,  0,  0],
       [ 7,  7,  3,  2, 10,  1, 11],
       [ 1, 12,  3, 13,  0,  0,  0]])
"""

#===================================================================================
# pad_sequences() : 문서 패딩화를 도와주는 함수
# Keras에서는 패딩을 지원하는 pad_sequences()를 지원함

from tensorflow.keras.preprocessing.sequence import pad_sequences

# 인코딩된 데이터
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
padded = pad_sequences(encoded)
# numpy 행렬로 만들어줌, 참고로, numpy의 방식과 달리 
# 앞에서부터 0으로 채워지는 모습을 볼 수 있음 (기본값)
print("케라스를 이용한 패딩화된 데이터 : \n", padded)

# 따라서 뒤로 채우고싶다면 추가 인자로 padding='post'를 추가한다
padded = pad_sequences(encoded, padding='post')
print("\n뒤에서부터 0을 채운 패딩화된 데이터 : \n",padded)

(padded == padded_np).all() # Numpy와 같은지 비교

#만약 maxlen의 인자를 주면, 길이는 제한되지만, 만약 maxlen을 넘긴 경우, 앞의 인코딩된 단어를 날림
padded = pad_sequences(encoded, padding='post', maxlen=5)
print("\nMaxlen으로 제한한 패딩된 데이터 :\n", padded)

# 따라서 먼저 인코딩된 단어를 남기고, 넘긴 단어들만 날리고 싶으면 truncating인자를 사용한다.
padded = pad_sequences(encoded, padding='post',truncating='post', maxlen=5)
print("\ntruncating으로 날릴 데이터를 결정해준 패딩된 데이터 :\n", padded)

# 참고 : 꼭 0으로 채우지 않고, 다른 번호( 인코딩 번호를 간섭하지 않는)로 패딩해도된다.
last_v = len(tokenizer.word_index) + 1 # 단어 집합 크기보다 1 큰 숫자를 사용함
padded = pad_sequences(encoded, padding='post', value = last_v)
print("\n 다른 번호로 채워진 패딩된 데이터 :\n", padded)

"""
케라스를 이용한 패딩화된 데이터 : 
 [[ 0  0  0  0  0  1  5]
 [ 0  0  0  0  1  8  5]
 [ 0  0  0  0  1  3  5]
 [ 0  0  0  0  0  9  2]
 [ 0  0  0  2  4  3  2]
 [ 0  0  0  0  0  3  2]
 [ 0  0  0  0  1  4  6]
 [ 0  0  0  0  1  4  6]
 [ 0  0  0  0  1  4  2]
 [ 7  7  3  2 10  1 11]
 [ 0  0  0  1 12  3 13]]

뒤에서부터 0을 채운 패딩화된 데이터 : 
 [[ 1  5  0  0  0  0  0]
 [ 1  8  5  0  0  0  0]
 [ 1  3  5  0  0  0  0]
 [ 9  2  0  0  0  0  0]
 [ 2  4  3  2  0  0  0]
 [ 3  2  0  0  0  0  0]
 [ 1  4  6  0  0  0  0]
 [ 1  4  6  0  0  0  0]
 [ 1  4  2  0  0  0  0]
 [ 7  7  3  2 10  1 11]
 [ 1 12  3 13  0  0  0]]

Maxlen으로 제한한 패딩된 데이터 :
 [[ 1  5  0  0  0]
 [ 1  8  5  0  0]
 [ 1  3  5  0  0]
 [ 9  2  0  0  0]
 [ 2  4  3  2  0]
 [ 3  2  0  0  0]
 [ 1  4  6  0  0]
 [ 1  4  6  0  0]
 [ 1  4  2  0  0]
 [ 3  2 10  1 11]
 [ 1 12  3 13  0]]

truncating으로 날릴 데이터를 결정해준 패딩된 데이터 :
 [[ 1  5  0  0  0]
 [ 1  8  5  0  0]
 [ 1  3  5  0  0]
 [ 9  2  0  0  0]
 [ 2  4  3  2  0]
 [ 3  2  0  0  0]
 [ 1  4  6  0  0]
 [ 1  4  6  0  0]
 [ 1  4  2  0  0]
 [ 7  7  3  2 10]
 [ 1 12  3 13  0]]

 다른 번호로 채워진 패딩된 데이터 :
 [[ 1  5 14 14 14 14 14]
 [ 1  8  5 14 14 14 14]
 [ 1  3  5 14 14 14 14]
 [ 9  2 14 14 14 14 14]
 [ 2  4  3  2 14 14 14]
 [ 3  2 14 14 14 14 14]
 [ 1  4  6 14 14 14 14]
 [ 1  4  6 14 14 14 14]
 [ 1  4  2 14 14 14 14]
 [ 7  7  3  2 10  1 11]
 [ 1 12  3 13 14 14 14]]
"""
#=============================================================================
# One-Hot Encodeing
# 메모리를 너무 많이 사용하므로 추천하는방법은 아님

# One-Hot Encoding 
# 케라스에서 원핫인코딩을 지원하기 위해 to_categorical()을 지원함
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
print('단어 집합 :',tokenizer.word_index)

# 이제 인코딩화 시킴
sub_text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print("인코딩된 sub_text : ", encoded)
# 원 핫 인코딩 진행
one_hot = to_categorical(encoded)
print ("\n원 핫 인코딩 된 데이터 : \n", one_hot)
# 근데 원 핫 인코딩 된 데이터는 이 한 문장을 표현하기 위해서 2차원배열(행렬)로 표현해야한다
# (길이는 단어의 개수만큼 ) 따라서 불필요한 메모리를 아주많이 사용하게 된다. 따라서 권장되는 방법은 아니다.

"""
단어 집합 : {'갈래': 1, '점심': 2, '햄버거': 3, '나랑': 4, '먹으러': 5, '메뉴는': 6, '최고야': 7}
인코딩된 sub_text :  [2, 5, 1, 6, 3, 7]

원 핫 인코딩 된 데이터 : 
 [[0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1.]]
"""