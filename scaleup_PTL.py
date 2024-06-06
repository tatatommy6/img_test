from PIL import Image
image= Image.open('C:/programing/imgscaleup_test/IMG_6140.jpg')

new_width=8568
new_height=11424

resized_img=image.resize((new_width,new_height),Image.LANCZOS)
resized_img.save('C:/programing/imgscaleup_test/IMG_6140_scaleuped.jpg')

resized_img.show()