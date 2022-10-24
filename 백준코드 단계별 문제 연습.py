# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 21:33:41 2022

@author: yoontaejun
"""
#%%
print('1단계 입출력과 사칙연산')
#%%
a = [1,1,2,0,0,0,1]
print(a,end=' ')
#%%
a = [1,1,2,0,0,0,1]
print(a)
#%%
# end = ' '를 입력함으로써 결과값이 한줄에 나옴, end없으면, 여러 행으로 나옴
A = list(map(int, input().split()))
result = [1, 1, 2, 2, 2, 8] # 원래 있어야하는 말의 개수
for i in range(len(result)):
    print(result[i] - A[i], end = ' ')
#%%
#2588번 문제
#  %은 나머지를 , // 몫을 구해주는 연산기호, 구분을 해서 구하는게 핵심
# 이전 문제들은 한 리스트 형태 [0, 1,2,3] 같이 넣어줘서 split을 이용해 각 값에 넣어줬지만 이번에는
# 472  이렇게 2행으로 이루어져있어서 a 와 b를 따로 선언하고 해줘야함
# 385
a = int(input())
b = int(input())
out1 = a*((b%100)%10)
out2 = a*((b%100)//10)
out3 = a*(b//100)
out4 = a*b
print(out1,out2,out3,out4,sep='\n')
# answer
inp1 = int(input(472))
inp2 = int(input(385))
 
out1 = inp1*((inp2%100)%10) # 1의 자리 5를 나오게 하기 위함
out2 = inp1*((inp2%100)//10) #10의 자리 나오게 하기 위해 %로 먼저 나눠서 85가 나오게 하고 //로 나눠줘서 몫인 8이 나오게 함
out3 = inp1*(inp2//100) # 100의 자리 3 이 나오게하기 위해 //를 이용해 100으로 나눠서 몫인 3이 나오게 함x
res = inp1*inp2
 
print(out1, out2, out3, res, sep='\n') # '\n'을 통해 각 행으로 출력
#%%
# 10171번 고양이 알고리즘
print("\\    /\\")
print(" )  ( ')")
print("(  /  )")
print(" \\(__)|")
# 역슬래시는 2개해줘야 역슬래시로 인식, 역슬래시 1개 하면 인식 못함.
print("\    /\")
print(" )  ( ')")
print("(  /  )")
print(" \(__)|") # 오류남
#%%
# 10172번 개 알고리즘
print("|\_/|")
print("|q p|   /}")
print('( 0 )"""\\')  # \'앞에 \을 붙여준다.
print('|"^"`    |')
print("||_/=\\__|")


print('|\\_/|')
print('|q p|   /}')
print('( 0 )"""\\') # \를 붙이고 쉼표를 처리
print('|"^"`    |')
print('||_/=\\__|')

print("|\\_/|")
print("|q p|   /}")
print('( 0 )"""\\')
print('|"^"`    |')
print("||_/=\\\\__|")
# 위의 3개도 여기서 프린타하면 개의 모양이 나오는데 다 오답으로 나옴 도대체 왜
# 이거만 정답
print('|\\_/|')
print('|q p|   /}')
print('( 0 )"""\\')
print('|"^"`    |')
print('||_/=\\\\__|') # 이 79번 행 처럼 역슬래시가 2개 나와야하므로 4개를 사용해줘야하는데 이 차이인듯하고
#%%
# 25083번 내가 제출한 답
#문제에서 한 행에 따옴표와 쌍따옴표가 섞여있어 '," 각각 따옴표가 있는 것은 쌍따옴표로 묶고
#쌍따옴표가 있는것은 따옴표로 묶어서 콤마로 분리해주고 프린트하는데 이때 결과는 각각 묶어준 코드가 
#출력될 때 한칸 띄워서 나오기 때문에 sep=' '로 구분자를 띄워쓰기로 인식하여 붙여서 나오게 출력
print("         ,r'",'\"7',sep = ' ')
print("r`-_   ,'  ,/")
print(' \. ".'," L_r'",sep = ' ')
print("   `~\\/")
print('      ||')
print('      ||') # 틀림
# 굳이 sep안써도 \를 이용해서 '와" 함께 있을 때 구분해서 출력 가능
#정답
print("         ,r\'\"7")
print("r`-_   ,\'  ,/")
print(" \\. \". L_r\'")
print("   `~\\/")
print("      |")
print("      |")
#%%
print('조건문 단계')
#%%
#1330번 두 수 비교하기
# 내 답, 정답
a = [ ]
b = [ ]
a , b = map(int,input().split())
if a>b:
    print('>')
elif a<b:
    print('<')
else:
    print('==')
#%%
# 9498번 시험 성적
# 내 답, 정답
a = int(input())
if 90<= a <=100:
    print('A')
elif 80 <= a <=89:
    print('B')
elif 70 <= a <=79:
    print('C')
elif 60 <= a <= 69:
    print('D')
else:
    print('F')
#%%
# 2753번 윤년
# 내 답
#윤년은 연도가 4의 배수이면서, 100의 배수가 아닐 때 또는 400의 배수일 때이다.
#윤년이면 1 아니면 0
# 정답
a = int(input())
if (a%4 == 0)&((a%100!=0)or(a%400==0)):
    print(1)
else:
    print(0)
#%%
#14681번 사분면 고르기
# 내 답, 틀림 3,4분면 바꿔적은 것 같음, 3,4 바꿨는데도 틀림, 2사분면도 잘 못 적었음
a = int(input())
b = int(input())
if (0<a)&(0<b):
    print(1)
elif (0<a)&(0>b):
    print(2)
elif (0>a)&(0<b):
    print(3)
elif (0>a)&(0>b):
    print(4)
# 수정한 답, 맞음
a = int(input())
b = int(input())
if (0<a)&(0<b):
    print(1)
elif (0>a)&(0<b):
    print(2)
elif (0>a)&(0>b):
    print(3)
elif (0<a)&(0>b):
    print(4)
#%%
# 2884번 문제 알람 시간 정하기
# 보고 풀었음
h,m = map(int,input().split)
if m > 44 :
    print(h,m-45)
elif (m < 45) & (h>0):
    print(h-1,m+15)
else:
    print(23,m+15)
# 런타임에러
# 비교연산자 사이 띄워주기 없애기, split하고 ()안 넣었음
h,m = map(int,input().split)
if m > 44 :
    print(h,m-45)
elif (m < 45)&(h>0):
    print(h-1,m+15)
else:
    print(23,m+15)
# 다시
h,m = map(int,input().split)
if m > 44 :
    print(h,m-45)
elif (m < 45)&(h>0):
    print(h-1,m+15)
else:
    print(23,m+15)
#%%
#2525번 오븐알람시계
# 답 틀림, 시간이 24시간이 넘어가지 않을 경우를 구해주지 않음
a,b = map(int,input().split())
c = int(input())
d = (b+c)//60
if (b+c < 60):
    print(a,(b+c))
elif ((b+c)>60)&((a+d) >=24):
    print((h+d)-24),((b+c)-(d*60))
# 다시 제출한 답
a,b = map(int,input().split())
c = int(input())
d = (b+c)//60
if (b+c < 60):
    print(a,(b+c))
elif ((b+c)>=60)&((a+d) >=24):
    print(((h+d)-24),((b+c)-(d*60)))
else:
    print(h+d,((b+c)-(d*60)))
# 다른 사람 예시랑 다른건 괄호밖에 없는데
a,b = map(int,input().split())
c = int(input())
d = (b+c)//60
if ((b+c)<60):
    print(a,b+c)
elif ((b+c)>=60)&((a+d)>=24):
    print(h+d-24),(b+c-d*60) # 여기 괄호 오류
else:
    print(h+d,(b+c-d*60)
# 도저히 모르겠다
# 딴 사람이 푼건데 나랑 다른게 뭔지 모르겠다
# 다른건 그냥 h m c d 랑 abcd인건데 
h, m = map(int, input().split())
c = int(input())
d = (m+c)//60

if m + c < 60:
    print(h, m+c)
elif (m + c >= 60)&(h + d >= 24):
    print(h+d-24, m+c-d*60)
else:
    print(h+d, m+c-d*60)
# 이 문제 한지 1시간 드디어 알아버렸음 elif구문부터 다른 사람이 한거랑 비교한다고 h,m을 넣었음
# 난 h,m에 값을 할당한게 아닌데.. 천천히 문제를 보자
#편의를 위해 d = (m+c) // 60 를 지정한다.
#  ex. 현재 1시 30분인데 요리 시간이 98분일 경우, d는 30+98을 60으로 나눈 몫인 2이다.
#  이 경우 요리가 끝나는 h는 1 + 2 = 3, m은 30 + 98 - (2*60) =  8이다.     
#%%
#2480번 문제 주사위 세게
a,b,c = map(int,input().split())
if a == b == c:
    print(10000+(a*1000))
elif a==b or a==c:
    print(1000+(a*100))
elif b==c:
    print(1000+(b*100))
else:
    print(max(a,b,c)*100)
#%% 
# 반복문
# 2739번 , 그냥 range(9)로 할당하면 0~8까지 할당됨
a = int(input())
for i in range(9):
    print(f'i*a : {i*a}')
# 다시 바꿔서
a = int(input())
for i in range(1,10):
    print(f'i*a : {i*a}') # 원하는 형식으로 출력이 안돼서 틀렸나 봄
# 다시 
a = int(input())
for i in range(1,10):
    print(f'{a}*{i} : {a*i}') # 출력은 같게 나오는데 답이 아님
# 애들이 원하는대로, 아마 포맷형식을 사이트에서 원하는데로 출력해야하는 듯
a = int(input())
for i in range(1,10):
    print(a,'*',i,'=',a*i) 
#%%
#내가 사용한 답
#2739번

a, b = map(int,input().split())
for i in range(7):
print(a[i]+b[i]-3)

#재제출 답

a, b = map(int,input().split())
while True:
print(a+b-3)
if (a+b-3):
break

#재재제출 

a = int(input())
for i in range(a): #a에 들어온 횟수만큼 돌아가게
b,c = map(int,input().split())
print(b+c)

#%%
#8393번 

#내 답

a = int(input())
for i in range(a+1): #1~ a까지 다 더해주게 범위를 a+1로 잡음
sum += i #1씩 증가하면서 더해주게
print(sum)

#틀림 sum초기값을 설정안해줌

#다시

a = int(input())
sum = 0 #반복문 안에 0 넣으면 계속 초기화돼서 1 나옴
for i in range(a+1):
    sum += i
    print(sum) #출력오류

#다시
a = int(input())
sum = 0
for i in range(a+1):
    sum += i
print(sum) # 반복문 안에 print가 있으면 더해줄때마다 출력해서 안됨, 최종값 한번만 나와야 함

#%%
#25304번, 영수증
# if가 반복문 내에서 돌아가서 각 값마다 if 적용
total = int(input()) # 영수증 총 금액
things = int(input()) # 총 개수

sum = 0 # 초기값 지정
for i in range(things+1):  # 총 개수만큼 반복하게
    a , b = map(int,input().split())  # 각 품목의 개수와 가격
    sum += (a+b) 
    
    if sum == total:
        print('Yes')
    else:
        print('No')

#다시
total = int(input()) # 영수증 총 금액
things = int(input()) # 총 개수

sum = 0 # 초기값 지정
for i in range(things+1):  # 총 개수만큼 반복하게
    a , b = map(int,input().split())  # 각 품목의 개수와 가격
    sum += (a+b) 
    
if sum == total:
    print('Yes')
else:
    print('No')
    
#다시
total = int(input()) # 영수증 총 금액
things = int(input()) # 총 개수

sum = 0 # 초기값 지정
for i in range(things):  # 총 개수만큼 반복하게, +1해주는 게 아니였음
    a , b = map(int,input().split())  # 각 품목의 개수와 가격
    sum += (a*b)  # 물건의 값은 개수 + 값이 아닌 개수*값, 제대로 보고 하기 
    
if sum == total:
    print('Yes')
else:
    print('No')
#%%
#15552번 빠른 a+b, input대신에 sys.stdin.readline사용할 수 있다.
import sys 

number = int(input()) # 첫줄에 주어진 개수 5를 사용하기 위함
for i in range(number):
    a,b = map(int,sys.stdin.readline().split()) #input대신 넣어봄
    print(a+b)
#%%
#11021번, A+B - 7
a = int(input()) #개수 받기
for i in range(a):
    q,w = map(int,input().split())
    print(f"Case #{i}:{q+w}") # 포맷을 이용한 각 i의 출력값 확인
    
#다시
a = int(input()) #개수 받기
for i in range(a+1):  # 이렇게 1, 로 지정안하고 그냥 개수만 지정하면 그 수만큼 반복한다는 뜻
    q,w = map(int,input().split())
    print(f"Case #{i}: {q+w}") # 포맷을 이용한 각 i의 출력값 확인

#다른 사람이 한 답, 나랑 별 다른거 못 느끼겠는데 정답임
t = int(input())

for i in range(1, t+1):  # 1부터 t까지
    a, b = map(int, input().split())
    print(f'Case #{i}: {a+b}')
    
#다시
a = int(input()) #개수 받기
for i in range(1,a+1):  # 이렇게 1, 로 지정안하고 그냥 개수만 지정하면 그 수만큼 반복한다는 뜻
    q,w = map(int,input().split())
    print(f"Case #{i}: {q+w}") # 포맷을 이용한 각 i의 출력값 확인
#%%
# 백준 2438번 별찍기ㄴ
a = int(input()) # 들어오는 값 문자, 정수형으로 변경
for i in range(1, a+1):  # 1부터 5까지
    print("*"*i) # i 번째마다 *을 i번 반복해서 출력
#%%
# 백준 2438번 별찍기 방법 2, 컴프리헨션 
[print('*' * i) for i in range(1, int(input())+1)]

#%%
# 백준 2438번 별찍기, 2438번과는 반대 방향으로
# 오른족에서 부터 채우기

