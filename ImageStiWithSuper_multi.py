#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import pdb
import cv2
import os
from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)
def show_Image_Cv(Image, Name="Demo"):
    cv2.namedWindow(Name,cv2.WINDOW_NORMAL)
    cv2.imshow(Name, Image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def main(opt,name0,name1,matches_path,viz_path):
    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    stem0, stem1 = Path(name0).stem, Path(name1).stem
   
    # Handle --cache logic.
    do_match = True
    do_viz = opt.viz

    if not (do_match or do_viz):
        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))


    # If a rotation integer is provided (e.g. from EXIF data), use it:
    if len(pair) >= 5:
        rot0, rot1 = int(pair[2]), int(pair[3])
    else:
        rot0, rot1 = 0, 0
    # pdb.set_trace()
    # Load the image pair.
    image0, image0_color,inp0, scales0 = read_image(
        os.path.join(input_dir,name0), device, opt.max_length0, rot0, opt.resize_float)
    image1, image1_color,inp1, scales1 = read_image(
        os.path.join(input_dir,name1), device, opt.max_length1, rot1, opt.resize_float)
    if image0 is None or image1 is None:
        print('Problem reading image pair: {} {}'.format(
            input_dir/name0, input_dir/name1))
        exit(1)
    timer.update('load_image')

    if do_match:
        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        timer.update('matcher')

        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                        'matches': matches, 'match_confidence': conf}
        # pdb.set_trace()
        # np.savez(str(matches_path), **out_matches)
        xy_F = kpts0 #特征点坐标
        xy_L = kpts1
        confidence = conf   #置信度
        Match = matches     #匹配对，存的是特征点的索引

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
        # show_Image_Cv(image1_color)
        # pdb.set_trace()
        warpImg = cv2.warpPerspective(image1_color, Mat, (image0_color.shape[1],int(image0_color.shape[0]+image1_color.shape[0])))
        # show_Image_Cv(warpImg)
        direct=warpImg.copy()
        direct[0:image0_color.shape[0], 0:image0_color.shape[1]] = image0_color

        rows,cols=image0_color.shape[:2]
        # drawMatches(image0, image1, PtsA, PtsB, Match, status)
        for row in range(0,rows):
            if image0_color[row, :].any() and warpImg[row, :].any():#开始重叠的最左端
                top = row
                break
        for row in range(rows-1, 0, -1):
            if image0_color[row, :].any() and warpImg[row, :].any():#重叠的最右一列
                bot = row
                break    

        res = np.zeros([rows, cols, 3], np.uint8)
        for col in range(0, cols):
            for row in range(0, rows):
                if not image0_color[row, col].any():#如果没有原图，用旋转的填充
                    res[row, col] = warpImg[row, col]
                elif not warpImg[row, col].any():
                    res[row, col] = image0_color[row, col]
                else:
                    srcImgLen = float(abs(row - top))
                    testImgLen = float(abs(row - bot))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = np.clip(image0_color[row, col] * (1-alpha) + warpImg[row, col] * alpha, 0, 255)

        warpImg[0:image0_color.shape[0], 0:image0_color.shape[1]]=res
        # flag = 1
        # for i,x in enumerate(warpImg):
        #     if flag:
        #         for j,y in enumerate(x):
        #             if (y!=np.array([0,0,0])).all():
        #                 break
        #             flag = 0
        #     else:
        #         break
        flag = 1
        h_warp,w_warp = warpImg.shape[0:2]
        for col in range(h_warp-1,0,-1):
            if flag:
                for row in range(w_warp):
                    if (warpImg[col,row]!=np.array([0,0,0])).any():
                        flag = 0
                        break
            else:
                break
        # pdb.set_trace()
        warpImg_noblack = warpImg[0:col,:]
        cv2.imwrite(str(matches_path),warpImg_noblack)

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    if do_viz:
        # Visualize the matches.
        color = cm.jet(mconf)
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0)),
        ]
        if rot0 != 0 or rot1 != 0:
            text.append('Rotation: {}:{}'.format(rot0, rot1))

        # Display extra parameter info.
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {}:{}'.format(stem0, stem1),
        ]
        # 把txt里的KA KB整理起来
        make_matching_plot(
            image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
            text, viz_path, opt.show_keypoints,
            opt.fast_viz, opt.opencv_display, 'Matches', small_text)

        timer.update('viz_match')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='testdir/annos.txt',
        help='Path to the list of image pairs')

    parser.add_argument(
        '--input_dir', type=str, default='testdir/imgs',
        help='Path to the directory that contains the images')

    parser.add_argument(
        '--output_dir', type=str, default='testdir/imgs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length0', type=int, nargs='+', default=720,
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--max_length1', type=int, nargs='+', default=720,
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')

    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    
    opt = parser.parse_args()
    opt.input_dir = r'./testdir/imgs_0405_v2_640_res'
    opt.output_dir = opt.input_dir
    opt.input_pairs = r'./testdir/annos_0405.txt'
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'


    with open(opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    # if opt.max_length0 > -1:
    #     pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    timer = AverageTimer(newline=True)
    for i, pair in enumerate(pairs):
        try:
            if len(pair)==2:
                name0, name1 = pair[:2]
                stem0, stem1 = Path(name0).stem, Path(name1).stem
                matches_path = output_dir / '{}_{}_match.jpg'.format(stem0, stem1)
                viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
                main(opt,name0,name1,matches_path,viz_path)

            else:
                matches_path = ''
                for i in range(1,len(pair)):
                    if i==1:
                        opt.max_length0 = 640
                        opt.max_length1 = 640
                        name0 = pair[0]
                        stem0 = Path(name0).stem
                    else:              
                        opt.max_length0 = 640
                        opt.max_length1 = 640
                        print('matches path:',matches_path)
                        name0 = os.path.basename(matches_path)
                    name1 = pair[i]
                    stem1 = Path(name1).stem
                    matches_path = output_dir / '{}_{}_match.jpg'.format(stem0, stem1)
                    viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
                    main(opt,name0,name1,matches_path,viz_path)
        except Exception as e:
            print(str(e))
            continue
            