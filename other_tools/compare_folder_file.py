import os


foldname = 'real1010'
path1 = '/media/yt/ssd/other_data/lango_menjin/data/ir_jpg/' + foldname
path2 = '/media/yt/ssd/other_data/lango_menjin/data/rgb_jpg/' + foldname

files1 = os.listdir(path1)
files2 = os.listdir(path2)


#rgb
for file in files2:
    if file not in files1:
        print(file)
print("////////////////////////////////////////////////////////////////")


#ir
for file2 in files1:
    if file2 not in files2:
        print(file2)
