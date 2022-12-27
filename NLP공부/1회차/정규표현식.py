# 정규 표현식
import re # 정규표현식

text = "I was wondering if anyone out there could enlighten me on this car."

# 길이가 1 ~ 2인 단어들을 정규표현식을 이용하여 삭제
shortword = re.compile(r'\W*\b\w{1,2}\b') # 문자+숫자가 아닌 것들이 0번 이상 반복되는 것들중, 문자또는 숫자가 1번/2번 반복되는 문자를 backspace로 삭제함함
print(shortword.sub('', text))     # text에서, 1글자/2글자로 이루어진 글자를 앞글자 (여기선 '')로 대체함.

#=========================================
#표제어 추출
from nltk.stem import WordNetLemmatizer # 표제어 추출도구 

lemmatizer = WordNetLemmatizer()
words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies',\
         'watched', 'has', 'starting']

print('표제어 추출 전 :', words)      # words 표제어 추출 전
print('표제어 추출 후 :', [lemmatizer.lemmatize(word) for word in words]) # words에 관한 단어 하나하나 모두 표제어 추출을 진행함.

""" 
표제어 추출 전 : ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
표제어 추출 후 : ['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']
"""

print(lemmatizer.lemmatize('dies', 'v'))    # dies의 표제어 : die(죽다)
print(lemmatizer.lemmatize('watched', 'v')) # watched 표제어 : watch(보다)
print(lemmatizer.lemmatize('has', 'v'))     # has의 표제어 : have (취하다)

#=========================================
#어간 추출
from nltk.stem import PorterStemmer   # 토큰화된 단어의 어간을 추출해주는 stemmer (Porter)
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

sentence = "This was not the map we found in Billy Bones's chest, but an\
accurate copy, complete in all things--names and heights and soundings--with\
the single exception of the red crosses and the written notes."

tokenized_sentence = word_tokenize(sentence)      # 단어 토큰화

print('어간 추출 전 :', tokenized_sentence)       # 토큰화된 단어 출력
print('어간 추출 후 :',[stemmer.stem(word) for word in tokenized_sentence]) # 토큰화된 단어를 어간추출함함

"""
어간 추출 전 : ['This', 'was', 'not', 'the', 'map', 'we', 'found', 'in', 'Billy', 'Bones', "'s", 'chest', ',', 'but', 'an', 'accurate', 'copy', ',', 
'complete', 'in', 'all', 'things', '--', 'names', 'and', 'heights', 'and', 'soundings', '--', 'with', 'the', 'single', 'exception', 'of', 'the', 'red',
'crosses', 'and', 'the', 'written', 'notes', '.']

어간 추출 후 : ['thi', 'wa', 'not', 'the', 'map', 'we', 'found', 'in', 'billi',
'bone', "'s", 'chest', ',', 'but', 'an', 'accur', 'copi', ',', 'complet', '
in', 'all', 'thing', '--', 'name', 'and', 'height', 'and', 'sound', '--', '
with', 'the', 'singl', 'except', 'of', 'the', 'red', 'cross', 'and', 'the',
'written', 'note', '.']
"""

words = ['formalize', 'allowance', 'electricical']
# 포터 어간 추출 규칙 
# ~alize → ~al, ~ance → 제거(ance가 없어짐), ~ical → ~ic
# ~s, ~e, ~es→ 제거(s, e, es 제거), ~y → ~i, ~ate → 제거(ate 제거), ~tion → ~t(ion 제거), 등등

print('어간 추출 전 :',words)
print('어간 추출 후 :',[stemmer.stem(word) for word in words])

"""
어간 추출 전 : ['formalize', 'allowance', 'electricical']
어간 추출 후 : ['formal', 'allow', 'electric']
"""

#==============================================================
# Porter와 Lancaster의 방식의 차이

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', \
         'fly', 'dies', 'watched', 'has', 'starting']

# 서로 다른 어간 추출 알고리즘의 사용으로 각기 다른 어간을 추출해줌.
# 포터의 방식이 좀 더 정확하다지만, 정확한 어근을 찾는것에는 부족함.
# 따라서 정확한 어근을 찾기 위해선 어간추출보다는 표제어 추출을 권장함

print('어간 추출 전 :', words)
print('포터 스테머의 어간 추출 후:',[porter_stemmer.stem(w) for w in words])
print('랭커스터 스테머의 어간 추출 후:',[lancaster_stemmer.stem(w) for w in words])
"""
어간 추출 전 : ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
포터 스테머의 어간 추출 후: ['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']
랭커스터 스테머의 어간 추출 후: ['policy', 'doing', 'org', 'hav', 'going', 'lov', 'liv', 'fly', 'die', 'watch', 'has', 'start']
"""

#==============================================================
# 불용어 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt        # 한국어 추출용으로 Okt를 사용

stop_words_list = stopwords.words('english')    # 영어에서의 불용어를 들고옴
print('불용어 개수 :', len(stop_words_list))   # 불용어 개수 ( 영어 )

# 불용어 : 해석하는데 큰 의미가 없는 단어 (어떤 의미를 분석하는데 있어 기여하는 바가 적은 단어들 
# (주로 주어, 조사, 접미사 등))

# 다 출력해봄
for i in range(0 ,18):
  print('불용어 10개 출력 :',stop_words_list[i*10 : i*10 + 10])

# 결과로 거의 대부분 인칭대명사와, be동사, 조동사의 일부, 접속사 등이 나왔음
# re(재~, 다시 ~함), here (여기)등 없어도 별 의미 없는 단어들도 불용어 처리함
# 만약 빠짐으로써 의미가 달라지는 경우에 한해서는 예외처리를 해놓음

"""
불용어 개수 : 179
불용어 10개 출력 : ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]
불용어 10개 출력 : ["you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his']
불용어 10개 출력 : ['himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself']
불용어 10개 출력 : ['they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this']
불용어 10개 출력 : ['that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be']
불용어 10개 출력 : ['been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing']
불용어 10개 출력 : ['a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until']
불용어 10개 출력 : ['while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into']
불용어 10개 출력 : ['through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down']
불용어 10개 출력 : ['in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once']
불용어 10개 출력 : ['here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each']
불용어 10개 출력 : ['few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only']
불용어 10개 출력 : ['own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will']
불용어 10개 출력 : ['just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o']
불용어 10개 출력 : ['re', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't"]
불용어 10개 출력 : ['doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't"]
불용어 10개 출력 : ['ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn']
불용어 10개 출력 : ["shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
"""

# 불용어를 제외하고, 단어를 토큰화함

example = "Family is not an important thing. It's everything."

stop_words = set(stopwords.words('english'))      # 불용어
word_tokens = word_tokenize(example)              # 단어 토큰화

result = []     #결과 배열

for word in word_tokens:
  if word not in stop_words:    # 단어가 불용어가 아니라면
    result.append(word)         # result 리스트에 단어를 집어 넣음 

print('불용어 제거 전 :',word_tokens)
print('불용어 제거 후 :',result)

"""
불용어 제거 전 : ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
불용어 제거 후 : ['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
"""

#==============================================================
# 한국어에서의 불용어

# 한국어 불용어 제거
# 한국어는 영어와 달리 단어 하나가 가지는 의미가 영어보다 많다.
# 따라서 단어 하나가 없어지면 문장이 의미하는 바가 완전히 달라질 수 있다. 따라서 불용어의 기준이 영어보다 좀 더 정확해야함
# '그럼', '위하', '때', '일', '그것', '사실', '경우', '어떤', '을','를' 등이 존재
# 위 예시는 쓰일 때마다 의미를 전달하는데 있어서 기여하는가 적은 편이다. (아닐 수 있다.)

# 불용어는 절대적인 기준이 없음. 어떤 데이터에 쓰는지/ 어떤 문제에 적용하는지/ 어떤 토크나이저를 쓰는지에 따라 달라질 수 있음

okt = Okt()

example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한게 있지."
stop_words = "를 아무렇게나 구 우려 고 안 돼 같은 게 구울 때 는"
stop_words = set(stop_words.split(' '))
# 불용어를 다음과 같이 정의함
# '를' '아무렇게나' '구' '우려' '고' '안' '돼' '같은' '게' '구울' '때' '는'

word_tokens = okt.morphs(example)   #단어 토큰화

result = [word for word in word_tokens if not word in stop_words]
# 토큰화된 단어 중, 불용어에 포함되지 않는 단어만 result 리스트에 포함됨 

print('불용어 제거 전', word_tokens)
print('불용어 제거 후', result)

# 절대적인 기준은 아님.
# 따라서 필요한 불용어를 따로 뽑아 리스트화 (txt파일로 저장/ csv로 저장)하여 이를 불러와서 사용

"""
불용어 제거 전 ['고기', '를', '아무렇게나', '구', '우려', '고', '하면', '안', '돼', '.', '고기', '라고', '다', '같은', '게', '아니거든', '.', '예컨대', '삼겹살', '을', '구울', '때', '는', '중요한게', '있지', '.']
불용어 제거 후 ['고기', '하면', '.', '고기', '라고', '다', '아니거든', '.', '예컨대', '삼겹살', '을', '중요한게', '있지', '.']
"""

#==============================================================
# 정규표현식의 compile 규칙

import re       # 정규표현식 Regular Expression 모듈

r = re.compile("a.c") # a와 c사이에 어떤 하나의 문자를 허용함. 즉, 문자가 3개인 문자열 중, a와 c가 각 끝에 위치하면 ok
print(r.search("kkk"))     # 아무 결과 출력 안함
print(r.search("abc"))     # 매치 결과 출력
"""
None
<re.Match object; span=(0, 3), match='abc'>
"""

r = re.compile("ab*c")    # abc사이에 'b'가 x번 반복되어도, 매칭시킴
# 즉, abc의 형식을 지키되, b가 여러번 나와도된다.
# abbbc, abbbbbbbbbbc, abbbbbbbbbbbbbcccc도 가능 (이유는 abc의 형식이 지켜졌기 때문)

# 만약 abc사이에 b뒤에 다른 문자가 나오면 매칭이 안됨

print(r.search("abbbbbbbc"))     # 매치 결과 출력 X
print(r.search("abc"))     # 매치 결과 출력
print(r.search("sacaaa"))     # 매치 결과 출력
print(r.search("sabbvc"))     # 매치 결과 출력

"""
<re.Match object; span=(0, 9), match='abbbbbbbc'>
<re.Match object; span=(0, 3), match='abc'>
<re.Match object; span=(1, 3), match='ac'>
None
"""

r = re.compile("ab+c")    # abc사이에 'b'가 1번 반복되어도 매칭시킴
# 즉, abc, abbc, .... 등 b가 1번이라도 들어가 있으면 전부 매칭시킴 

print(r.search("abbc"))     # 매치 결과 출력 
print(r.search("abc"))     # 매치 결과 출력
print(r.search("acaaa"))     # 매치 결과 출력 X
print(r.search("abbbbbbbbbbbbbbbbbc"))     # 매치 결과 출력 
print()

"""
<re.Match object; span=(0, 4), match='abbc'>
<re.Match object; span=(0, 3), match='abc'>
None
<re.Match object; span=(0, 19), match='abbbbbbbbbbbbbbbbbc'>
"""
r = re.compile("^ab")    # 문자 시작이 ab로 시작되면 매칭

print(r.search("bbc"))     # 매치 결과 출력 X
print(r.search("abc"))     # 매치 결과 출력
print(r.search("acaaa"))     # 매치 결과 출력 X
print(r.search("abbbbbbbbbbbbbbbbbc"))     # 매치 결과 출력 
print(r.search("abz"))     # 매치 결과 출력 
"""
None
<re.Match object; span=(0, 2), match='ab'>
None
<re.Match object; span=(0, 2), match='ab'>
<re.Match object; span=(0, 2), match='ab'>
"""


r = re.compile("ab{2}c")    # abc중 b가 2번 반복되는 것 즉, abbc만 매칭시킴

print(r.search("ac"))     # 매치 결과 출력 X
print(r.search("abc"))     # 매치 결과 출력 X
print( r.search("abbbbbc"))     # 매치 결과 출력 X
print(r.search("dddabbc"))     # 매치 결과 출력

print()

r = re.compile("ab{2,8}c")    # abc중 b가 2번이상, 8번 이하 반복되는것만 매칭
# 즉, abbc~ abbbbbbbbc만 매칭 
print(r.search("ac"))     # 매치 결과 출력 X
print(r.search("abc"))     # 매치 결과 출력 X
print( r.search("asdasabbbbbc"))     # 매치 결과 출력 O
print(r.search("asdjsdkabbc"))     # 매치 결과 출력 O
print(r.search("abbbbbbbbbc"))     # 매치 결과 출력 X

"""
None
None
None
<re.Match object; span=(3, 7), match='abbc'>

None
None
<re.Match object; span=(5, 12), match='abbbbbc'>
<re.Match object; span=(7, 11), match='abbc'>
None
"""

r = re.compile("ab.")   # ab다음에 하나의 문자 허용

print( r.match("kkkkkabc"))     # 처음 문자열이 ab가 아니므로 매칭 X
print( r.search("kkkkkabc"))     # ab가 문자열에 포함되므로 매칭 O
print( r.match("abckkkkk"))     # 처음 문자열이 ab이므로 매칭 O

"""
None
<re.Match object; span=(5, 8), match='abc'>
<re.Match object; span=(0, 3), match='abc'>
"""

#=========================================================
# 정규표현식 split()함수

# split함수 : 기준에 맞춰서 텍스트를 분리함
# 1. 공백 (" ")기준 분리
text = "사과 딸기 수박 메론 바나나"
print(re.split(" ", text))

# 줄바꿈 (\n) 기준 분리
text = """사과
딸기
수박
메론
바나나"""
print(re.split("\n", text))

# +를 기준으로 분리 (어떤 문자를 기준으로 분리)
text = "사과+딸기+수박+메론+바나나"
re.split("\+", text)
print(re.split("\+", text))

# findall (정규표현식과 모두 매치되는 문자열 반환)
text = """
이름 : 김철수
전화번호 : 010 - 1234 - 1234
나이 : 30
성별 : 남
"""
print(re.findall("\d+", text)) # 숫자만으로 이루어진 문자열 전부 반환
print(re.findall("\d+", "문자열입니다.")) # 숫자만으로 이루어진 문자열이 존재하지 않기에 빈칸 출력

"""
['사과', '딸기', '수박', '메론', '바나나']
['사과', '딸기', '수박', '메론', '바나나']
['사과', '딸기', '수박', '메론', '바나나']
['010', '1234', '1234', '30']
['11231']
"""

#==========================================================
#텍스트 전처리 예시

#정규표현식 텍스트 전처리 예시

text = """100 John PROF
101 James STUD
102 Mac STUD"""

print(re.split('\s+', text))    
# \s : 공백 (공백을 기준으로, 텍스트를 분리함, (공백이 얼마나 들어가있든 1번 이상 들어가면 모조리 분리)) 
print(re.findall('\d+',text))
# \d : 숫자로만 이루어진 문자열을 찾기
print(re.findall('[A-Z]',text)) # 문자열중 대문자를 반환함 (이때 문자열 전체를 반환하진 않고, 문자 하나만 반환)
print(re.findall('[A-Z]{4}',text)) # 문자열중 대문자가 4번 반복되는 문자열을 반환함 (즉, 대문자로 이루어진 4개의 문자만 가진 문자열만 반환)
print(re.findall('[A-Z][a-z]+',text))
# 문자열 중,  첫 글자가 대문자이면서, 나머지글자들은 모두 소문자인 경우를 찾음 (ex : John )

"""
['100', 'John', 'PROF', '101', 'James', 'STUD', '102', 'Mac', 'STUD']
['100', '101', '102']
['J', 'P', 'R', 'O', 'F', 'J', 'S', 'T', 'U', 'D', 'M', 'S', 'T', 'U', 'D']
['PROF', 'STUD', 'STUD']
['John', 'James', 'Mac']
"""
from nltk.tokenize import RegexpTokenizer 
# 정규 표현식을 이용한 토크나이저 (RegexpTokenizer)

text = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as\
cheery as cheery goes for a pastry shop"

tokenizer1 = RegexpTokenizer("[\w]+")   
# 문자+숫자를 포함하는 문자열만 토큰화. 나머지는 날림
tokenizer2 = RegexpTokenizer("\s+", gaps=True)
# 공백을 기준으로 토큰화, 만약 gaps가 false이면 공백만 출력) 

print(tokenizer1.tokenize(text))
print(tokenizer2.tokenize(text))

"""
['Don', 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'Mr', 'Jone', 's', 'Orphanage', 'is', 'ascheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
["Don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name,', 'Mr.', "Jone's", 'Orphanage', 'is', 'ascheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
"""


