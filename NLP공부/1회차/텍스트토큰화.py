# 단어 토큰화 (NLTK이용)
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

print('단어 토큰화1 :',word_tokenize("Don't be fooled by the dark sounding name\
, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
print('단어 토큰화2 :',WordPunctTokenizer().tokenize("Don't be fooled by the\
dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a\
pastry shop."))
print('단어 토큰화3 :',text_to_word_sequence("Don't be fooled by the dark\
sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry\
shop."))

#============================================================================
# 트리뱅크 워드 토큰화
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()     # 트리뱅크 토크나이저저

text = "Starting a home-based restaurant may be an ideal. it doesn't have a\
food chain or restaurant of their own."
print('트리 뱅크 워드 토크나이저 :',tokenizer.tokenize(text))

#=============================================================
# 문장 토큰화
from nltk.tokenize import sent_tokenize

text = "His barber kept his word. But keeping such a huge secret to himself was\
driving him crazy. Finally, the barber went up a mountain and almost to the\
edge of a cliff. He dug a hole in the midst of some reeds. He looked about,\
to make sure no one was near."
print('문장 토큰화1 :',sent_tokenize(text))

text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print('문장 토큰화2 :',sent_tokenize(text))

#=========================================================
# 한국어 토큰화
import kss

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다.\
 이제 해보면 알걸요?'
print('한국어 문장 토큰화 :',kss.split_sentences(text))

#=========================================================
# 토큰화 및 품사 태깅
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence = word_tokenize(text)

print('단어 토큰화 :',tokenized_sentence)
print('품사 태깅 :',pos_tag(tokenized_sentence))

#=========================================================
# 한국어 토큰화 및 품사 태깅, 명사추출
# Okt와 Kkma의 방식이 다르기 때문에 해봐야 됨
from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()

print('OKT 형태소 분석 :',okt.morphs("열심히 코딩한 당신 , 연휴에는 여행을 가봐요"))
print('OKT 품사 태깅 :',okt.pos("열심히 코딩한 당신 , 연휴에는 여행을 가봐요"))
print('OKT 명사 추출 :',okt.nouns("열심히 코딩한 당신 , 연휴에는 여행을 가봐요"))

print('꼬꼬마 형태소 분석 :',kkma.morphs("열심히 코딩한 당신 , 연휴에는 여행을 가봐요"))
print('꼬꼬마 품사 태깅 :',kkma.pos("열심히 코딩한 당신 , 연휴에는 여행을 가봐요"))
print('꼬꼬마 명사 추출 :',kkma.nouns("열심히 코딩한 당신 , 연휴에는 여행을 가봐요"))

# 결과 
"""
OKT 형태소 분석 : ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
OKT 품사 태깅 : [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
OKT 명사 추출 : ['코딩', '당신', '연휴', '여행']
꼬꼬마 형태소 분석 : ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']
꼬꼬마 품사 태깅 : [('열심히', 'MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]
꼬꼬마 명사 추출 : ['코딩', '당신', '연휴', '여행']
"""