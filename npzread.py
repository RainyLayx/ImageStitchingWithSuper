import numpy as np
import pdb
import cv2
import os

def show_Image_Cv(Image, Name="Demo"):
    cv2.namedWindow(Name,cv2.WINDOW_NORMAL)
    cv2.imshow(Name, Image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

npz_dir = r'testdir/imgs'
for i in os.listdir(npz_dir):
    if '.npz' in i:
        path = os.path.join(npz_dir,i)
        # path = 'testdir/imgs/04010701_04010702_matches.npz'
        name = os.path.basename(os.path.splitext(path)[0])
        # fname,lname = name.split('_s')[0:2]
        # fname = os.path.join('assets\scannet_sample_images',fname+'.jpg')
        # lname = os.path.join('assets\scannet_sample_images','s'+lname.split('_matches')[0]+'.jpg')
        fname = os.path.join('testdir\imgs',name.split('_')[0]+'.jpg')
        lname = os.path.join('testdir\imgs',name.split('_')[1]+'.jpg')
        img1 = cv2.imread(fname)
        img2 = cv2.imread(lname)
        w = 480
        h = 640
        img1 = cv2.resize(img1,(w,h))
        img2 = cv2.resize(img2,(w,h))
        print(fname,lname)
        npz = np.load(path)
        # print(npz.files)
        # print(npz['keypoints0'][0])
        # # print(npz['matches']>-1)
        # print(npz['matches'].shape)
        # pdb.set_trace()
        xy_F = npz['keypoints0']
        xy_L = npz['keypoints1']
        confidence = npz['match_confidence']
        Match = list(npz['matches'])

        PtsA = []
        PtsB = []
        for i in range(len(Match)):
            if confidence[i]>0:
                PtsA.append(xy_F[i])
                PtsB.append(xy_L[Match[i]])
        PtsA = np.float32(PtsA)
        PtsB = np.float32(PtsB)
        # pdb.set_trace()
        Mat, status = cv2.findHomography(PtsB, PtsA, cv2.RANSAC, 4)

        warpImg = cv2.warpPerspective(img2, Mat, (img1.shape[1],int(img1.shape[0]+img2.shape[0])))
        direct=warpImg.copy()
        direct[0:img1.shape[0], 0:img1.shape[1]] = img1

        rows,cols=img1.shape[:2]
        # drawMatches(img1, img2, PtsA, PtsB, Match, status)
        for row in range(0,rows):
            if img1[row, :].any() and warpImg[row, :].any():#开始重叠的最左端
                top = row
                break
        for row in range(rows-1, 0, -1):
            if img1[row, :].any() and warpImg[row, :].any():#重叠的最右一列
                bot = row
                break

        res = np.zeros([rows, cols, 3], np.uint8)
        for col in range(0, cols):
            for row in range(0, rows):
                if not img1[row, col].any():#如果没有原图，用旋转的填充
                    res[row, col] = warpImg[row, col]
                elif not warpImg[row, col].any():
                    res[row, col] = img1[row, col]
                else:
                    srcImgLen = float(abs(row - top))
                    testImgLen = float(abs(row - bot))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = np.clip(img1[row, col] * (1-alpha) + warpImg[row, col] * alpha, 0, 255)

        warpImg[0:img1.shape[0], 0:img1.shape[1]]=res
        cv2.imwrite(os.path.join('./testdir/',name+'.jpg'),warpImg)
        # show_Image_Cv(direct)
        # pdb.set_trace()
