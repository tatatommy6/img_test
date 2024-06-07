import cv2
from PIL import Image

# OpenCV의 DNN 모듈을 사용하여 pre-trained super-resolution 모델 로드
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "EDSR_x3.pb"  # pre-trained 모델 파일
sr.readModel(path)
sr.setModel("edsr", 3)  # EDSR 모델과 확대 비율 설정

# 이미지를 읽고 OpenCV 형식으로 변환
image = cv2.imread('C:/programing/imgscaleup_test/img_test/IMG_6140.jpg')

# 초해상도 적용
result = sr.upsample(image)

# 결과 이미지를 저장
cv2.imwrite('C:\programing\imgscaleup_test\img_test\IMG_6140_super_resolutioned.jpg', result)

# 결과 이미지를 Pillow를 사용하여 출력
result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
result_image.show()
