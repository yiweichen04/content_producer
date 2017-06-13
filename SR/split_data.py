import glob
import shutil

path = '/home/invigen/wan/course/DL/final_project/img_align_celeba'
dst = '/home/invigen/wan/course/DL/final_project/subpixel-master/data/celebA'
image_names = glob.glob(path + '/*.jpg')
print (len(image_names))

for i, name in enumerate(image_names):
	if i < 162770:
		shutil.copy(name, dst + '/train/' + str(i) + '.jpg')
	elif i < 182637:
		shutil.copy(name, dst + '/val/' + str(i) + '.jpg')
	else:
		shutil.copy(name, dst + '/test/' + str(i) + '.jpg')
