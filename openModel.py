import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import joblib

#โหลดmodel มาเก็บไว้ในตัวแปร
model = joblib.load(open('model/20-29_model.sav', 'rb'))
ii = 81                                                 #1-80 คือ ข้อมูลเรา เรื่มทำนาย 81 จะตรงวันวันที่ 11 เดือน11 2021
idxlist =[]
for i in range (25):                                     #ทำนายเป็นช่วง ปรับช่วงตรงrange
     idxlist.append(ii) 
     ii += 1
    
xx = np.array(idxlist).reshape(-1,1)                    # reshape
degTemp = 2                                             # ปรับ degree ให้ตรงกับ model
poly_features = PolynomialFeatures(degree=degTemp)      
xxpol = poly_features.fit_transform(xx)                 #เอาตัวแปรมารับค่าหลังแปลง degree

pre = model.predict(xxpol)                              #เอาตัวแปรมารับผลการทำนาย
a=0
print(pre[0][0])
for i in pre:                                             #แสดงค่า
    print(f'ผลการทำนายวันที่่ {idxlist[a]-70} พ.ย. 2021 -> {int(*i)}')
    a+=1