import cv2

def upscale_image(image_path, model_path, output_path):
    # 이미지를 불러옵니다.
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: 이미지를 읽을 수 없습니다. 경로를 확인해주세요: {image_path}")
        return

    # Super Resolution을 하기 위해 4배 확대를 위한 EDSR 모델을 사용합니다.
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    # 모델 파일을 불러옵니다.
    try:
        sr.readModel(model_path)
    except Exception as e:
        print(f"Error: 모델 파일을 읽을 수 없습니다. 경로를 확인해주세요: {model_path}")
        print(e)
        return

    # 모델을 설정합니다.
    sr.setModel("edsr", 4)

    # 이미지를 모델이 기대하는 크기로 리사이즈합니다 (가정: 모델은 1:1 비율을 기대함).
    # 가로와 세로 중 더 작은 길이를 기준으로 이미지를 크롭하여 1:1 비율로 만듭니다.
    height, width, _ = img.shape
    min_dim = min(height, width)
    cropped_img = img[:min_dim, :min_dim]

    # img를 입력 받아 결과로 result1를 돌려줍니다.
    try:
        result1 = sr.upsample(cropped_img)
    except Exception as e:
        print(f"Error: Super resolution 업샘플링에 실패했습니다.")
        print(e)
        return

    # 결과를 저장합니다.
    cv2.imwrite(output_path, result1)
    print(f"Upscaled image saved to {output_path}")

    # 결과 이미지를 화면에 보여줍니다.
    cv2.imshow('Original Image', cropped_img)
    cv2.imshow('Upscaled Image - EDSR', result1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 경로 설정
image_path = 'C:/programing/imgscaleup_test/img_test/IMG_6140.jpg'
model_path = 'C:/programing/imgscaleup_test/img_test/EDSR_x4.pb'
output_path = 'C:/programing/imgscaleup_test/img_test/IMG_6140_upscaled.jpg'

# 업스케일링 함수 호출
upscale_image(image_path, model_path, output_path)
