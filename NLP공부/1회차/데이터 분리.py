#데이터 분리 (pandas 모델 사용)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. zip함수를 이용

# zip을 이용하여 분리하면, X는 첫번째 인자의 리스트를, y는 두번째 인자의 리스트로 나뉘어짐
X, y = zip(['a', 1], ['b', 2], ['c', 3])
print('X 데 이 터 :',X)   
print('y 데 이 터 :',y)
print()

#이차원 리스트의 형태로도 가능
sequences = [['a', 1], ['b', 2], ['c', 3]]
X, y = zip(*sequences) #sequences를 포인터처럼 사용함
print('X 데 이 터 :',X)
print('y 데 이 터 :',y)

# Pandas의 데이터프레임을 이용한 분리 
values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨, 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']

# 데이터 프레임으로 나눌 수 있음
df = pd.DataFrame(values, columns=columns)
#df

print()
# X를 본문, y를 스팸메일 유무로 분리하여 확인 가능
X = df['메일 본문']
y = df['스팸 메일 유무']

print('X 데이터 :',X.to_list())
print('y 데이터 :',y.to_list())

# NumPy를 이용한 데이터 분리 
np_array = np.arange(0,16).reshape((4,4))
print('\n전체 데이터 :')
print(np_array)
#마지막 열을 제외한 나머지는 X데이터, 마지막열만 y로 분리할 수 있음
X = np_array[:, :3]
y = np_array[:,3]   # 행, 열

print('X 데이터 :')
print(X)
print('y 데이터 :',y)

"""
X 데 이 터 : ('a', 'b', 'c')
y 데 이 터 : (1, 2, 3)

X 데 이 터 : ('a', 'b', 'c')
y 데 이 터 : (1, 2, 3)

X 데이터 : ['당신에게 드리는 마지막 혜택!', '내일 뵐 수 있을지 확인 부탁드...', '도연씨, 잘 지내시죠? 오랜만입...', '(광고) AI로 주가를 예측할 수 있다!']
y 데이터 : [1, 0, 0, 1]

전체 데이터 :
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
X 데이터 :
[[ 0  1  2]
 [ 4  5  6]
 [ 8  9 10]
 [12 13 14]]
y 데이터 : [ 3  7 11 15]
"""

#===============================================================
#사이킷 런

#사이킷 런  (train_test_split())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)
# random_state는 임의의 값

# 임의로 X,y 생성
X, y = np.arange(10).reshape((5, 2)), range(5)
print('X 전체 데이터 :')
print(X)
print('y 전체 데이터 :')
print(list(y))
print()

# 이 데이터를 7대 3비율로 훈련 : 테스트로 분리해봄
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=1234)
print('X 훈련 데이터 :')
print(X_train)
print('X 테스트 데이터 :')
print(X_test)
print('y 훈련 데이터 :')
print(y_train)
print('y 테스트 데이터 :')
print(y_test)
print()

# 이때 random_state를 다음과 같이 바꾸면 y의 샘플은 다르게 변한다 (1로 주면 처음 30퍼센트의 데이터가 테스트로 간다.)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=1)
print('y 훈련 데이터 :')
print(y_train)
print('y 테스트 데이터 :')
print(y_test)

"""
X 전체 데이터 :
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
y 전체 데이터 :
[0, 1, 2, 3, 4]

X 훈련 데이터 :
[[2 3]
 [4 5]
 [6 7]]
X 테스트 데이터 :
[[8 9]
 [0 1]]
y 훈련 데이터 :
[1, 2, 3]
y 테스트 데이터 :
[4, 0]

y 훈련 데이터 :
[4, 0, 3]
y 테스트 데이터 :
[2, 1]
"""

#======================================================================
# 수동으로 분리

# 수동 분리 

X, y = np.arange(0,24).reshape((12,2)), range(12)
print('X 전체 데이터 :')
print(X)
print('y 전체 데이터 :')
print(list(y))
print()

num_train = int(len(X)* 0.8) # 80퍼센트는 훈련데이터
num_test = int (len(X)- num_train) # 나머지는 테스트
print("훈련 데이터 크기 : ", num_train)
print("테스트 데이터 크기 : ", num_test)
print()

# 처음의 80%는 훈련데이터, 나머지 20%는 테스트
X_test = X[num_train:]
X_train = X[:num_train]
y_test = y[num_train:]
y_train = y[:num_train]
print('X 테스트 데이터 :')
print(X_test)
print('y 테스트 데이터 :')
print(list(y_test))
print()

"""
X 전체 데이터 :
[[ 0  1]
 [ 2  3]
 [ 4  5]
 [ 6  7]
 [ 8  9]
 [10 11]
 [12 13]
 [14 15]
 [16 17]
 [18 19]
 [20 21]
 [22 23]]
y 전체 데이터 :
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

훈련 데이터 크기 :  9
테스트 데이터 크기 :  3

X 테스트 데이터 :
[[18 19]
 [20 21]
 [22 23]]
y 테스트 데이터 :
[9, 10, 11]
"""