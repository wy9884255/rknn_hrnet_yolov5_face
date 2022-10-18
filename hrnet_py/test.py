import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = 'hrnet.onnx'
RKNN_MODEL = 'hrnet.rknn'
IMG_PATH = './face.jpg'
DATASET = './dataset.txt'
IMG_WIDTH = 256
IMG_HEIGHT = 256

QUANTIZE_ON = True
        
def _get_max_preds(heatmaps):
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals 
    
    
def transform_preds(coords, center, scale, output_size, use_udp=False):
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = np.array(scale)
    scale = scale * 200.0
    scale = scale.tolist()
    print(scale)
    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]
    print(scale_x)
    print(scale_y)
    print(center)
    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords

    
def _taylor(heatmap, coord):
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (
            heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
            heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        dyy = 0.25 * (
            heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
            heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord
    
    
def post_dark_udp(coords, batch_heatmaps, kernel=3):
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert (B == 1 or B == N)
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(
        batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode='edge').flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords
        
        
def _gaussian_blur(heatmaps, kernel=11):
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border),
                          dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def keypoints_from_heatmaps(heatmaps,
                            center,
                            scale,
                            unbiased=False,
                            post_process='default',
                            kernel=11,
                            valid_radius_factor=0.0546875,
                            use_udp=False,
                            target_type='GaussianHeatmap'):

    # Avoid being affected
    heatmaps = heatmaps.copy()

    # detect conflicts
    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert not post_process == 'megvii'

    # normalize configs
    if post_process is False:
        warnings.warn(
            'post_process=False is deprecated, '
            'please use post_process=None instead', DeprecationWarning)
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                'post_process=True, unbiased=True is deprecated,'
                " please use post_process='unbiased' instead",
                DeprecationWarning)
            post_process = 'unbiased'
        else:
            warnings.warn(
                'post_process=True, unbiased=False is deprecated, '
                "please use post_process='default' instead",
                DeprecationWarning)
            post_process = 'default'
    elif post_process == 'default':
        if unbiased is True:
            warnings.warn(
                'unbiased=True is deprecated, please use '
                "post_process='unbiased' instead", DeprecationWarning)
            post_process = 'unbiased'

    # start processing
    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        pass
        # if target_type.lower() == 'GaussianHeatMap'.lower():
            # preds, maxvals = _get_max_preds(heatmaps)
            # preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        # elif target_type.lower() == 'CombinedTarget'.lower():
            # for person_heatmaps in heatmaps:
                # for i, heatmap in enumerate(person_heatmaps):
                    # kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    # cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # # valid radius is in direct proportion to the height of heatmap.
            # valid_radius = valid_radius_factor * H
            # offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            # offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            # heatmaps = heatmaps[:, ::3, :]
            # preds, maxvals = _get_max_preds(heatmaps)
            # index = (preds[..., 0] + preds[..., 1] * W).flatten()
            # index += W * H * np.arange(0, N * K / 3)
            # index = index.astype(int).reshape(N, K // 3, 1)
            # preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        # else:
            # raise ValueError('target_type should be either '
                             # "'GaussianHeatmap' or 'CombinedTarget'")
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == 'unbiased':  # alleviate biased coordinate
            pass
            # apply Gaussian distribution modulation.
            # heatmaps = np.log(
                # np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            # for n in range(N):
                # for k in range(K):
                    # preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5
    print(preds)
    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(
            preds[i], center[i], scale[i], [W, H], use_udp=use_udp)

    if post_process == 'megvii':
        maxvals = maxvals / 255.0 + 0.5

    return preds, maxvals
      
def bbox_xywh2cs(bbox, aspect_ratio, padding=1., pixel_std=200.):

    x, y, w, h = bbox[:4]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    print(x, y, w, h)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    scale = np.array([w, h], dtype=np.float32) / pixel_std
    scale = scale * padding

    return center, scale


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt
    
    
def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt
    

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # pixel_std is 200.
    scale_tmp = scale * 200.0

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    print(src)
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    print(dst)
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans
    
    
if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(target_platform='rk3588s', mean_values=[[0.485*255, 0.456*255, 0.406*255]], std_values=[[0.229*255, 0.224*255, 0.225*255]])
    #rknn.config()
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk3588s', device_id='a867f77fbf5bb43c')
    # ret = rknn.init_runtime('rk3568')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img_origin = cv2.imread(IMG_PATH)  
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # preprocess
    bbox = [0, 0, 465, 580]
    c, s = bbox_xywh2cs(
        bbox,
        aspect_ratio=1.0,
        padding=1.25,
        pixel_std=200)
    print(c)
    print(s)
    trans = get_affine_transform(c, s, 0, [IMG_WIDTH, IMG_HEIGHT])
    img = cv2.warpAffine(
        img_origin,
        trans, (int(IMG_WIDTH), int(IMG_HEIGHT)),
        flags=cv2.INTER_LINEAR)
    #normalize you mf
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]
    # img = img[:,:, ::-1]
    # img = img .astype(np.float32, copy=False)
    # img /= 255.0
    # img -= mean
    # img /= std
    # Inference
    cv2.imwrite("2.jpg", img)
    print('--> Running model')
    img = img[np.newaxis, :]
    outputs = rknn.inference(inputs=[img])
    #np.save('./onnx_hrnet_1.npy', outputs[1])
    #np.save('./onnx_hrnet_2.npy', outputs[2])
    heatmap = outputs[0]
    c = np.array([[c[0], c[1]]])
    s = np.array([[s[0], s[1]]])
    #print(np.max(heatmap))
    #print(np.min(heatmap))
    preds, maxvals = keypoints_from_heatmaps(heatmap, c, s)
    #print(preds.shape)
    #print(maxvals.shape)
    pts = preds[0]
    for i in range(len(pts)):  
        pts[i][pts[i] < 0] = 0
        pt = pts[i]
        #print(pt)
        image = cv2.circle(img_origin, tuple(pt), 5, (255, 0, 0), 2)
    cv2.imwrite('test.png', image)
    
    rknn.eval_perf(inputs=[img], is_print=True)
    # print(preds)
    # print(maxvals)
    print('done')

    # # post process
    # input0_data = outputs[0]
    # input1_data = outputs[1]
    # input2_data = outputs[2]

    # input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
    # input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
    # input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

    # input_data = list()
    # input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    # input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    # input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    # boxes, classes, scores = yolov5_post_process(input_data)

    # img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # if boxes is not None:
        # draw(img_1, boxes, scores, classes)
    # show output
    # cv2.imshow("post process result", img_1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    rknn.release()
