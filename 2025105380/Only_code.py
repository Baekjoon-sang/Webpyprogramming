import pandas as pd    #csv file을 읽기 위해 3rdparty 모듈 중 pandas를 채택했다.

rawdata = pd.read_csv('stars_raw.csv')  #획득한 데이터 원본이다.
rawdata.head()  #제대로 불러왔는지 확인차 작성했다. 실행 시 하단에 읽힌 csv 파일이 출력된다.

column_list = ["Temp","Lumin","Rad","Mag","Type","Col","Spect"] #열의 이름을 변경한 리스트 작성
rawdata.columns = column_list   #날것 데이터의 열 이름을 삽입했다.

"""
Star type는 Kaggle 사이트에서 새롭게 정의한 분류용 정수값이다.
이는 데이터 분류 논리에서 효율적이겠으나 가시적으로 문제가 있으므로 
추가적으로 Name 인덱스를 설정해 주었다.
물론 Data 학습 시에는 새롭게 index를 설정해 주기에 놔두는 것이 효율적일지도 모르겠다... 
허나 원래 index 부여는 index가 없는 Data에서 필수적으로 해야 하는 일이므로 우선 삭제했다.
"""
starlist = ["BrownDwarf","RedDwarf","WhiteDwarf","MainSequence","SuperGiant","HyperGiant"]
starlist_KOR = ["갈색왜성","적색왜성","백색왜성","주계열성","초거성","극대거성"] #출력값의 가시성을 위해 추가적으로 작성했다.

for i in range(len(starlist)):
    rawdata.loc[rawdata['Type']== i, 'Name'] = starlist[i]    

#필요없는 열을 제거한 데이터셋 새로 선언
processed_data = rawdata.drop(columns=['Rad','Type','Col','Spect'])
unit_list = ["K","L/Lo","Mv","Name"] #혹여 물리량을 표시할 필요가 있을 때를 대비해 리스트를 추가 작성함
column_list_KOR = ["표면온도","광도","절대등급","이름"]


processed_data

processed_data.to_csv("stars_new.csv",index=False) #미리 만들어둔 stars_new.csv 파일에 가공 데이터를 저장
#즉 가공한 데이터 원본이다.

#Matplotlib이라는 3rdParty 모듈을 활용하였다.

import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic') #한글폰트 불러오기: GPT에게 얻은 모듈이다.
plt.rcParams['axes.unicode_minus'] = False #마이너스 폰트 불러오기:GPT에게 얻은 모듈이이다.

data = pd.read_csv('stars_new.csv') #저장한 파일을 새롭게 불러왔다.
data.head()

import seaborn as sns #seaborn 모듈을 통해 시각 보조 그래프의 가시성을 강화했다.

x = data['Temp']
y = data['Mag']
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
sns.scatterplot(data=data,x='Temp',y='Mag',hue="Name") #이름 값으로 Data를 구분했다.

X = data[['Temp','Lumin','Mag']]
X_train = X  #원래 표준화 전처리를 진행했으나 오류가 빈번히 생겨 기본 데이터로 진행하게 되었다...

X_train.head()

for i in range(len(starlist)):
    data.loc[data["Name"] == starlist[i], 'label'] = i #str대신 int index를 새로 부여해 학습에 사용함.

y_train = data['label'].astype(int)
y_train

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=2)  #이곳이 K를 정하는, 즉 가중치 설정 구간이다. 가장 정확도가 높은 K로 설정한 상태이다.
model.fit(X_train, y_train)  #가공한 데이터로 학습을 진행했다.


DataForPrediction = []  #사용자가 직접 입력한 신규 Data를 저장하는 list이다.
for i in range(3):
    DataForPrediction.append(float(input("{}은/는 몇 {}인가요?".format(column_list_KOR[i],unit_list[i]))))

print("입력한 값:",DataForPrediction)

data_input = pd.DataFrame([DataForPrediction], columns=['Temp', 'Lumin', 'Mag'])
#예측 메서드는 pd의 DataFrame 형태이므로 그에 맞게 가공해주었다.


predicted_label = model.predict(data_input) #예측후 그 label 값을 새 변수에 저장한다.
predicted_star_type = starlist_KOR[predicted_label[0]]  

print(f"예측된 별의 분류: {predicted_star_type}")  #최종적인 결과물이다!