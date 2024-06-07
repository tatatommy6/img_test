import cv2
# 이미지를 불러옵니다.
img = cv2.imread('C:/programing/imgscaleup_test/img_test/IMG_6140.jpg')


# Super Resolution을 하기 위해 2배 확대를 위한 ESPCN 모델을 사용합니다.
sr = cv2.dnn_superres.DnnSuperResImpl_create()
# 모델 파일의 이름을 적어주면 됩니다.
sr.readModel('EDSR_x4.pb')
# 모델 파일에 적힌 숫자와 일치하도록 적어줘야 합니다.
sr.setModel("edsr",4)
# img를 입력 받아 결과로 result1를 돌려줍니다. 
result1 = sr.upsample(img)
# OpenCV에서 제공하는 보간법을 사용해봅니다.
result2 = cv2.resize(img, dsize=None, fx=2, fy=2)

# 입력 이미지와 Super Resolution, OpenCV 보간법 적용 결과를 화면에 보여줍니다.
cv2.imshow('origianl', img)
cv2.imshow('result - ESPCN', result1)
cv2.imshow('result - Opencv', result2)
cv2.waitKey(0)