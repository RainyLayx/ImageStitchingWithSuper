import os
import pdb
path = r'./testdir/imgs_0405_640_480'
txt = os.path.join(path,'annos_0405.txt')
m = ''
contrast = ''
with open(txt, 'w') as f:
    for fname in os.listdir(path):
        if '.jpg' in fname or '.png' in fname:
            print(fname)
            if not contrast:#第一个文件
                m = fname + ' '
                contrast = fname
            else:
                if fname[:-6]==contrast[:-6]:
                    m = m + fname + ' '
                    contrast = fname
                else:
                    f.write(m+'\n')
                    print('-------------')
                    print(m)
                    m = fname + ' '
                    contrast = fname
    f.write(m)
