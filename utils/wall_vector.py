import cv2
import numpy as np
import math

# global para
max_k = 99999  # max 斜率
# save wall-lines k
K_map = None
# save no(1,2,3...) of wall line, 0-not wall
line_map = None

# 自适应图像滑动分块


def make_grid(shape, window=1024, min_overlap=4):
    """
        return - array of grid : [number, coord(x1,x2,y1,y2)]
    """
    # logging.info('shift window size is: {}'.format(window_size))
    grid_list = []
    h, w = shape
    overlap = int(window // min_overlap)
    not_overlap = window - overlap
    window_h = window if h > window else h
    window_w = window if w > window else w
    if max(shape) <= window:
        # shape smaller than window
        grid = [0, w, 0, h]
        grid_list.append(grid)
    else:
        nh = (h - window_h) // not_overlap + 2 if (h - window_h) > 0 else 1
        y_list = [0 for i in range(nh)]
        for i in range(1, nh):
            y_list[i] = y_list[i - 1] + not_overlap
            if y_list[i] + window_h > h:
                y_list[i] = h - window_h
                break

        nw = (w - window_w) // not_overlap + 2 if (w - window_w) > 0 else 1
        x_list = [0 for i in range(nw)]
        for i in range(1, nw):
            x_list[i] = x_list[i - 1] + not_overlap
            if x_list[i] + window_w > w:
                x_list[i] = w - window_w
                break

        for xi in x_list:
            for yi in y_list:
                x1 = xi
                y1 = yi
                x2 = x1 + window_w
                y2 = y1 + window_h
                grid = [x1, x2, y1, y2]
                grid_list.append(grid)

    return np.array(grid_list)


def imgIOU(label, predict):
    """
    label:[x,y,3]-uint8(0/255)
    predict:[x,y,3]-uint8(0/255)
    return:IOU-float
    """
    label = label[:, :, 0] / 255
    predict = predict[:, :, 0] / 255
    iou_up = np.sum(label * predict)
    iou_down = np.sum(label) + np.sum(predict) - iou_up
    iou = iou_up / (iou_down + 0.0000001)
    return iou


def cnt2img(cnt, x, y):
    new_mask = np.zeros((y, x), np.uint8)
    cv2.drawContours(new_mask, [cnt], 0, 1, -1)
    return new_mask


def points2img(points, x, y):
    new_mask = np.zeros((y, x), np.uint8)
    cv2.drawContours(new_mask, [points], 0, 1, -1)
    return new_mask


def process_cnt(cnt, x, y):
    cnt_ = cnt.copy()
    for i in range(len(cnt_)):
        cnt_[i][0][0] -= x
        cnt_[i][0][1] -= y
    return cnt_


def campIOU(cnt):
    area_cnt = cv2.contourArea(cnt)
    rect = cv2.minAreaRect(cnt)
    points = cv2.boxPoints(rect)
    points = np.int0(points)[:, np.newaxis, :]
    # print('cnt',cnt)
    # print('points',points)
    area_rect = cv2.contourArea(points)
    # print('cnt shape',cnt.shape)
    # print('points shape',points.shape)
    # print('cnt type',type(cnt))
    # print('points type',type(points))
    # print('area_cnt',area_cnt)
    # print('area_rect',area_rect)
    return 1 if area_rect == 0 else area_cnt / area_rect


def find_k_line(cnt, k_line):
    rect = cv2.minAreaRect(cnt)
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    # print('points',points)
    center_x, center_y = 0, 0
    num_point = len(points)
    for point in points:
        x, y = point
        center_x += x
        center_y += y
    center_x = center_x / num_point
    center_y = center_y / num_point
    # print('center_x, center_y',center_x,center_y)
    a, b, c = None, None, None
    if k_line == 0:
        a = 0
        b = 1
        c = -center_y
    elif k_line == max_k:
        b = 0
        a = 1
        c = -center_x
    else:
        a = k_line
        b = -1
        c = center_y - k_line * center_x
    # print('a,b,c',a,b,c)
    return a, b, c


def pointsCrossLine(x1, y1, x2, y2, a, b, c):
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = x2 * y1 - x1 * y2
    cx = (c1 * b - b1 * c) / (a * b1 - b * a1 + 0.0001)
    cy = (c * a1 - a * c1) / (a * b1 - b * a1 + 0.0001)
    return cx, cy


def isin_cntflag(cnt, cnt_flag):
    """
    cnt: np.array -> (n,1,2)
    cnt_flag: np.array -> (h,w)
    """
    # print('cnt.shape',cnt.shape)
    # print('cnt_flag.shape',cnt_flag.shape)
    # area_cnt=cv2.contourArea(cnt)
    shape_flag = cnt_flag.shape
    cnt_img = np.zeros((shape_flag[0], shape_flag[1]), np.uint8)
    cv2.drawContours(cnt_img, [cnt], 0, 1, -1)
    # print('np.sum(cnt_flag)',np.sum(cnt_flag))
    area_iou = np.sum(cnt_img * cnt_flag)
    # print('area_cnt',area_cnt)
    # print('area_iou',area_iou)
    # print('np.sum(cnt_img)',np.sum(cnt_img))
    if area_iou < np.sum(cnt_img):
        return False
    return True


def split_cnt(cnt, a, b, c, cnt_flag):
    num_cnt = len(cnt)
    cnts = []
    new_cnt = []
    for num in range(num_cnt - 1):
        x1, y1 = cnt[num][0]
        x2, y2 = cnt[num + 1][0]
        f1 = a * x1 + b * y1 + c
        f2 = a * x2 + b * y2 + c
        new_cnt.append(cnt[num])
        # 点集分类
        if (f1 == 0 and f2 != 0) or f1 * f2 < 0:
            if f1 == 0 and f2 != 0:
                new_cnt = np.array(new_cnt, np.int0)
                cnts.append(new_cnt)
                new_cnt = [cnt[num]]

            else:
                # 计算交点
                cx, cy = pointsCrossLine(x1, y1, x2, y2, a, b, c)
                new_cnt.append(np.array([[cx, cy]], np.int0))
                new_cnt = np.array(new_cnt, np.int0)
                cnts.append(new_cnt)
                new_cnt = []
                new_cnt.append(np.array([[cx, cy]], np.int0))

        # 始末点合并
        if num + 1 == num_cnt - 1:
            if len(cnts) == 0:
                return [cnt]
            # 判断始末点关系
            start_point = cnt[0]
            end_point = cnt[num + 1]
            f_s = a * start_point[0][0] + b * start_point[0][1] + c
            f_e = a * end_point[0][0] + b * end_point[0][1] + c
            # print('start_point',start_point)
            # print('end_point',end_point)
            # print('f_s',f_s)
            # print('f_e',f_e)
            # print('a,b,c',[a,b,c])
            if f_s * f_e < 0:
                # print('------------f_s*f_e<0------------')
                new_cnt.append(end_point)
                cx, cy = pointsCrossLine(
                    start_point[0][0],
                    start_point[0][1],
                    end_point[0][0],
                    end_point[0][1],
                    a,
                    b,
                    c,
                )
                new_cnt.append(np.array([[cx, cy]], np.int0))
                new_cnt = np.array(new_cnt, np.int0)
                cnts.append(new_cnt)
                # print('np.array(new_cnt,np.int0) shape',np.array(new_cnt,np.int0).shape)
                # print('cnts[0] shape',cnts[0].shape)
                # print('np.array([[cx,cy]],np.int0) shape',np.array([[[cx,cy]]],np.int0).shape)
                cnts[0] = np.concatenate(
                    (cnts[0], np.array([[[cx, cy]]], np.int0)))
                return cnts
            elif f_e == 0 and f_s != 0:
                # print('------------f_e==0------------')
                new_cnt.append(end_point)
                new_cnt = np.array(new_cnt, np.int0)
                cnts.append(new_cnt)
                # print('cnts[0]',cnts[0])
                # print('cnts[0] shape',cnts[0].shape)
                # print('end_point',end_point)
                # print('end_point shape',end_point.shape)
                cnts[0] = np.concatenate((np.array([end_point]), cnts[0]))
                return cnts
            elif f_s == 0:
                # print('------------f_s==0------------')
                new_cnt.append(end_point)
                new_cnt.append(start_point)
                new_cnt = np.array(new_cnt, np.int0)
                cnts.append(new_cnt)
                if len(cnts[0]) == 1:
                    del cnts[0]
                # print('cnts[0] shape',cnts[0].shape)
                # print('new_cnt shape',new_cnt.shape)
                return cnts
            elif f_s * f_e > 0:
                new_cnt.append(end_point)
                cnts[0] = np.concatenate((new_cnt, cnts[0]))
                return cnts


def findMaxMIOU(cnt, k_line, thr_iou=0.9, max_times=20):
    now_iou = campIOU(cnt)
    # print('before_iou',now_iou)
    if now_iou >= thr_iou:
        return [cnt]

    # create cnt flag
    stand_rect = cv2.boundingRect(cnt)
    # print('stand_rect',stand_rect)
    cnt_flag = np.zeros((stand_rect[3], stand_rect[2]), np.uint8)
    cnt_ = process_cnt(cnt, stand_rect[0], stand_rect[1])
    cv2.drawContours(cnt_flag, [cnt_], 0, 1, -1)

    fa, fb, fc = None, None, None
    seg_times = 0
    min_cnt = cnt
    min_cnts = [min_cnt]
    while seg_times < max_times:
        # print('seg_times',seg_times)
        # print('area min_cnt',cv2.contourArea(min_cnt))
        if cv2.contourArea(min_cnt) < 10:
            break
        seg_times += 1
        # print('min_cnt',min_cnt)
        a, b, c = find_k_line(min_cnt, k_line)
        # print('a,b,c',a,b,c)
        cnt_location = split_cnt(min_cnt, a, b, c, cnt_flag)

        # update min_cnt
        min_iou = 1
        for cnt_j in cnt_location:
            iou_j = campIOU(cnt_j)
            if iou_j < min_iou:
                min_cnt = cnt_j
                min_iou = iou_j
                min_cnts.append(min_cnt)

        # 计算miou及更新直线
        cnts_compare = split_cnt(cnt, a, b, c, cnt_flag)
        # print('cnts_compare',cnts_compare)
        # print('len(cnts_compare)',len(cnts_compare))
        miou = 0
        for cnt_i in cnts_compare:
            iou_i = campIOU(cnt_i)
            # print('iou',iou_i)
            miou += iou_i
        miou = miou / len(cnts_compare)
        # print('miou',miou)
        # 若miou更大
        if miou > now_iou:
            # print('miou',miou)
            fa, fb, fc = a, b, c
            now_iou = miou

    # print('fa,fb,fc',[fa,fb,fc])
    # print('now_iou',now_iou)
    final_cnts = [cnt] if fa == None else split_cnt(cnt, fa, fb, fc, cnt_flag)
    return final_cnts  # , min_cnts, now_iou,cnt_flag


def is_K_same(k1, k2, same_a=10):
    # 定义斜率K的相似性
    k1 = k1 if k1 != np.inf else 999999
    k2 = k2 if k2 != np.inf else 999999
    # 夹角在10°内
    tan_a = (k1 - k2) / (1 + k1 * k2 + 0.0001)
    if tan_a ** 2 <= math.tan(same_a / 180 * math.pi) ** 2:
        return True
    return False


def find_wall_lines(
    img, img_pre, detact_length=8, min_line_length=2, exist_thr=0.001, diff_thr=0.3
):
    """
    find lines on img and filter wall lines by pre
    """
    # save wall lines , direction and k
    wall_lines = []
    direc_lines = []
    k_lines = []
    # process img
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Fast LSD
    lsd_dec = cv2.ximgproc.createFastLineDetector()
    lines_fastLSD = lsd_dec.detect(img_gray)
    # get BINARY prediction img
    img_bool = img_pre.copy()
    img_bool = cv2.cvtColor(img_bool, cv2.COLOR_BGR2GRAY)  # 灰化
    _, img_bool = cv2.threshold(img_bool, 127, 1, cv2.THRESH_BINARY)  # 二值化
    # get img size
    y, x = img_bool.shape
    # K feature map
    K_map = np.zeros((y, x), np.float16)  # save k
    # save no(1,2,3...) of wall line, 0-not wall
    line_map = np.zeros((y, x), np.uint16)
    # NO. of wall line
    no_line = 0
    # filter wall lines
    for line in lines_fastLSD:
        x1, y1, x2, y2 = line[0]
        # filter dx or dy <= min_line_length
        if (x1 - x2) ** 2 <= min_line_length ** 2 and (
            y1 - y2
        ) ** 2 <= min_line_length ** 2:
            continue
        x0, x3, x4, x5, = x1, x1, x1, x1
        y0, y3, y4, y5, = y1, y1, y1, y1
        dx = 0
        dy = detact_length
        if y1 == y2:
            dx = 0
            dy = detact_length
        else:
            k = (x1 - x2) / (y2 - y1)
            dx = ((detact_length ** 2) / (1 + k ** 2)) ** 0.5
            dy = dx * k
        x5 = x1 - dx
        y5 = y1 - dy
        x4 = x2 - dx
        y4 = y2 - dy

        x3 = x1 + dx
        y3 = y1 + dy
        x0 = x2 + dx
        y0 = y2 + dy

        x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5 = (
            int(x0),
            int(x1),
            int(x2),
            int(x3),
            int(x4),
            int(x5),
            int(y0),
            int(y1),
            int(y2),
            int(y3),
            int(y4),
            int(y5),
        )

        # filter line
        rec1 = [[x1, y1], [x2, y2], [x4, y4], [x5, y5]]
        rec2 = [[x1, y1], [x2, y2], [x0, y0], [x3, y3]]
        cnts = []
        cnts.append(rec1)
        cnts.append(rec2)
        rec1 = np.array(rec1)[:, np.newaxis, :]
        rec2 = np.array(rec2)[:, np.newaxis, :]

        # create mask
        mask1 = np.zeros((y, x), np.uint8)
        cv2.drawContours(mask1, rec1[np.newaxis, :, :, :], 0, (255), -1)
        mask2 = np.zeros((y, x), np.uint8)
        cv2.drawContours(mask2, rec2[np.newaxis, :, :, :], 0, (255), -1)
        # 二值化
        _, mask1_b = cv2.threshold(mask1, 127, 1, cv2.THRESH_BINARY)
        _, mask2_b = cv2.threshold(mask2, 127, 1, cv2.THRESH_BINARY)
        # 计算探测区域 wall_area/detect_area 百分比
        mask1_area = np.sum(mask1_b * img_bool)
        mask2_area = np.sum(mask2_b * img_bool)
        mask_area = max(mask1_area, mask2_area)
        mask1_p = mask1_area / np.sum(mask1_b)
        mask2_p = mask2_area / np.sum(mask2_b)
        mask_p = max(mask1_p, mask2_p)
        # wall lines bool
        if (
            (mask1_p + mask2_p) > (exist_thr * 2)
            and ((mask1_p - mask2_p) / mask_p) ** 2 > (diff_thr ** 2)
            and mask_area >= 20
        ):
            no_line += 1  # num of wall line
            # get k - wall line
            k_wallLine = None
            if x1 == x2:
                k_wallLine = max_k
            elif y1 == y2:
                k_wallLine = 0
            else:
                k_wallLine = (line[0][1] - line[0][3]) / (
                    line[0][0] - line[0][2]
                )  # (y1-y2)/(x1-x2)
            k_lines.append(k_wallLine)  # 压入斜率K
            # bool wall direction
            t1 = 1 if mask2_p > mask1_p else -1
            t2 = 1 if mask1_p > mask2_p else -1
            # draw line flag on line map
            # 此段wall line的轨迹
            line_flg = np.zeros((y, x), np.uint8)
            cv2.line(line_flg, (x2, y2), (x1, y1), 1, 1)
            # ROI
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            # line_map基于wall line轨迹，保存全局wall line的序号NO，与wall_line中保存的wall line坐标的序号一一对应
            # K_map 基于wall line轨迹，保存全局wall line的斜率K值，如果有交叉，以长度长的为准
            for y_line in range(y_min, y_max + 1):
                for x_line in range(x_min, x_max + 1):
                    if line_flg[y_line, x_line] == 1:
                        # 如果轨迹为空，则直接写入NO和K
                        if line_map[y_line, x_line] == 0:
                            line_map[y_line, x_line] = no_line
                            # print('k',k)
                            K_map[y_line, x_line] = k_wallLine
                        # 若有交叉，以长度长的为准
                        else:
                            # keep longer line k
                            l_now = (x1 - x2) ** 2 + (y1 - y2) ** 2
                            no_befor = line_map[y_line, x_line] - 1
                            l_befor = (
                                (wall_lines[no_befor][0] -
                                 wall_lines[no_befor][2]) ** 2
                                + (wall_lines[no_befor][1] -
                                   wall_lines[no_befor][3])
                                ** 2
                            )
                            if l_now > l_befor:
                                line_map[y_line, x_line] = no_line
                                K_map[y_line, x_line] = k_wallLine
            # add wall lines
            wall_line = [x1, y1, x2, y2]
            wall_lines.append(wall_line)
            # get direction
            direc_line = [x1, y1, x5, y5] if mask1_p > mask2_p else [
                x1, y1, x3, y3]
            direc_lines.append(direc_line)

    return wall_lines, direc_lines, k_lines


def wall_vector(img_o, img_pre):
    # get original shape
    # print("process img data")
    img_shape = img_o.shape[:2]

    # resize 0.8
    resize_shape = [int(img_shape[1] * 0.8), int(img_shape[0] * 0.8)]
    img_o = cv2.resize(img_o, resize_shape, interpolation=cv2.INTER_NEAREST)
    img_pre = cv2.resize(img_pre, resize_shape,
                         interpolation=cv2.INTER_NEAREST)

    # get resize scale
    # resize_scale = 1
    resize_scale = max(resize_shape) // 500
    if resize_scale < 1:
        resize_scale = 1

    # resize img_pre
    new_resize_shape = [
        resize_shape_size // resize_scale for resize_shape_size in resize_shape
    ]
    img_pre = cv2.resize(img_pre, new_resize_shape,
                         interpolation=cv2.INTER_NEAREST)
    # img_wallLines = cv2.resize(img_o, new_resize_shape, interpolation=cv2.INTER_NEAREST)

    # process img
    img_gray = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)

    # Fast LSD
    lsd_dec = cv2.ximgproc.createFastLineDetector()
    lines_fastLSD = lsd_dec.detect(img_gray)

    # filter lines
    # print("filter lines")
    # get BINARY prediction img
    img_bool = img_pre.copy()
    img_bool = cv2.cvtColor(img_bool, cv2.COLOR_BGR2GRAY)  # 灰化
    _, img_bool = cv2.threshold(img_bool, 127, 1, cv2.THRESH_BINARY)  # 二值化

    # get img size
    y, x = img_bool.shape

    # save wall lines , direction and k
    wall_lines = []
    direc_lines = []
    k_lines = []

    # process condition
    a1 = 2  # fusion dis
    a2 = 8  # detection dis

    # init detection position
    dx = 0
    dy = 0

    # wall area condition
    thr_mp = 0.01  # exist condition
    thr_dis = 0.3  # diff condition

    # K feature map
    K_map = np.zeros((y, x), np.float16)  # save k
    # save no(1,2,3...) of wall line, 0-not wall
    line_map = np.zeros((y, x), np.uint16)

    # NO. of wall line
    no_line = 0

    # filter wall lines
    for line in lines_fastLSD:
        x1, y1, x2, y2 = line[0]

        # 线段坐标等比例放缩
        x1 = x1 // resize_scale
        y1 = y1 // resize_scale
        x2 = x2 // resize_scale
        y2 = y2 // resize_scale

        # 过滤过短线段
        if (x1 - x2) ** 2 <= a1 ** 2 and (y1 - y2) ** 2 <= a1 ** 2:
            continue

        x0, x3, x4, x5, = x1, x1, x1, x1
        y0, y3, y4, y5, = y1, y1, y1, y1
        dx = 0
        dy = a2
        if y1 == y2:
            dx = 0
            dy = a2
        else:
            k = (x1 - x2) / (y2 - y1)
            dx = ((a2 ** 2) / (1 + k ** 2)) ** 0.5
            dy = dx * k
        x5 = x1 - dx
        y5 = y1 - dy
        x4 = x2 - dx
        y4 = y2 - dy

        x3 = x1 + dx
        y3 = y1 + dy
        x0 = x2 + dx
        y0 = y2 + dy

        x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5 = (
            int(x0),
            int(x1),
            int(x2),
            int(x3),
            int(x4),
            int(x5),
            int(y0),
            int(y1),
            int(y2),
            int(y3),
            int(y4),
            int(y5),
        )

        # filter line
        rec1 = [[x1, y1], [x2, y2], [x4, y4], [x5, y5]]
        rec2 = [[x1, y1], [x2, y2], [x0, y0], [x3, y3]]
        cnts = []
        cnts.append(rec1)
        cnts.append(rec2)
        rec1 = np.array(rec1)[:, np.newaxis, :]
        rec2 = np.array(rec2)[:, np.newaxis, :]

        # create mask
        mask1 = np.zeros((y, x), np.uint8)
        cv2.drawContours(mask1, rec1[np.newaxis, :, :, :], 0, (255), -1)
        mask2 = np.zeros((y, x), np.uint8)
        cv2.drawContours(mask2, rec2[np.newaxis, :, :, :], 0, (255), -1)
        # 二值化
        _, mask1_b = cv2.threshold(mask1, 127, 1, cv2.THRESH_BINARY)
        _, mask2_b = cv2.threshold(mask2, 127, 1, cv2.THRESH_BINARY)
        # 计算探测区域 wall_area/detect_area 百分比
        mask1_area = np.sum(mask1_b * img_bool)
        mask2_area = np.sum(mask2_b * img_bool)
        mask_area = max(mask1_area, mask2_area)
        mask1_p = mask1_area / (np.sum(mask1_b) + 0.0000001)
        mask2_p = mask2_area / (np.sum(mask2_b) + 0.0000001)
        mask_p = max(mask1_p, mask2_p)
        if (
            (mask1_p + mask2_p) > (thr_mp * 2)
            and ((mask1_p - mask2_p) / mask_p) ** 2 > (thr_dis ** 2)
            and mask_area >= 20
        ):
            no_line += 1  # num of wall line
            # get k - wall line
            k_wallLine = None
            if x1 == x2:
                k_wallLine = max_k
            elif y1 == y2:
                k_wallLine = 0
            else:
                k_wallLine = (line[0][1] - line[0][3]) / (
                    line[0][0] - line[0][2]
                )  # (y1-y2)/(x1-x2)
            k_lines.append(k_wallLine)  # 压入斜率K

            # draw line flag on line map
            # 此段wall line的轨迹
            line_flg = np.zeros((y, x), np.uint8)
            cv2.line(line_flg, (x2, y2), (x1, y1), 1, 1)
            # ROI
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2)
            y_max = max(y1, y2)
            # line_map基于wall line轨迹，保存全局wall line的序号NO，与wall_line中保存的wall line坐标的序号一一对应
            # K_map 基于wall line轨迹，保存全局wall line的斜率K值，如果有交叉，以长度长的为准
            for y_line in range(y_min, y_max + 1):
                if y_line not in range(y):
                    continue
                for x_line in range(x_min, x_max + 1):
                    if x_line not in range(x):
                        continue
                    if line_flg[y_line, x_line] == 1:
                        # 如果轨迹为空，则直接写入NO和K
                        if line_map[y_line, x_line] == 0:
                            line_map[y_line, x_line] = no_line
                            # print('k',k)
                            K_map[y_line, x_line] = k_wallLine
                        # 若有交叉，以长度长的为准
                        else:
                            # keep longer line k
                            l_now = (x1 - x2) ** 2 + (y1 - y2) ** 2
                            no_befor = line_map[y_line, x_line] - 1
                            l_befor = (
                                (wall_lines[no_befor][0] -
                                 wall_lines[no_befor][2]) ** 2
                                + (wall_lines[no_befor][1] -
                                   wall_lines[no_befor][3])
                                ** 2
                            )
                            if l_now > l_befor:
                                line_map[y_line, x_line] = no_line
                                K_map[y_line, x_line] = k_wallLine
            # add wall lines
            wall_line = [x1, y1, x2, y2]
            wall_lines.append(wall_line)
            # get direction
            direc_line = [x1, y1, x5, y5] if mask1_p > mask2_p else [
                x1, y1, x3, y3]
            direc_lines.append(direc_line)

    # # test wall lines
    # return wall_lines, img_pre

    #### prediction pix 分类阶段 ####
    # save new no(1,2,3...) of wall line, 0-not wall
    new_line_map = np.zeros((y, x), np.uint16)
    # save distance from pix point to wall lines
    dis_k_line_map = np.zeros((y, x), np.float16)
    # 保存对应wall line的长度
    len_dis_k_line_map = np.zeros((y, x), np.float16)

    ############### and this part can classify the rest of wall pix #########################################################
    # 点距探测 - 固定 / 非固定 探测范围 算法均由内向外扩张
    def point_detect(x_point, y_point, r_max):
        r_detect = 1  # 起始探测半径
        is_detected = False  # 是否已经探测到
        while is_detected == False and r_detect <= r_max:
            has_detect_position = []  # 保存遍历过的位置
            for y_detect_r in range(-r_detect, r_detect + 1):
                y_detect = y_point + y_detect_r
                if y_detect < 0:
                    continue
                if y_detect >= y:
                    break
                # 仅对外缘进行探测
                if y_detect_r == r_detect or y_detect_r == -r_detect:
                    for x_detect_r in range(-r_detect, r_detect + 1):
                        x_detect = x_point + x_detect_r
                        if x_detect < 0:
                            continue
                        if x_detect >= x:
                            break
                        # 判断此点是否已经遍历过
                        if [y_detect, x_detect] in has_detect_position:
                            continue
                        else:
                            has_detect_position.append([y_detect, x_detect])
                            # 若探测到wall line
                            if (
                                line_map[y_detect, x_detect] > 0
                                and is_detected is False
                            ):
                                # 赋值距离map(while逐步外探)
                                dis_k_line_map[y_point, x_point] = r_detect
                                no_dis_wall_line = line_map[y_detect,
                                                            x_detect] - 1
                                this_wall_line = wall_lines[no_dis_wall_line]
                                this_len = (
                                    (this_wall_line[0] -
                                     this_wall_line[2]) ** 2
                                    + (this_wall_line[1] -
                                       this_wall_line[3]) ** 2
                                ) ** 0.5
                                # 赋值新的预测像素所属wall line的长度
                                len_dis_k_line_map[y_point, x_point] = this_len
                                # 赋值新的预测像素所属wall line的序号
                                new_line_map[y_point, x_point] = line_map[
                                    y_detect, x_detect
                                ]
                                # 赋值斜率特征图
                                K_map[y_point, x_point] = k_lines[
                                    line_map[y_detect, x_detect] - 1
                                ]
                                is_detected = True
                                # if r_max == 999:
                                # print("r_detect:", r_detect)
                                return
                else:
                    for x_detect_r in [-r_detect, r_detect]:
                        x_detect = x_point + x_detect_r
                        if x_detect >= 0 and x_detect < x:
                            # 判断此点是否已经遍历过
                            if [y_detect, x_detect] in has_detect_position:
                                continue
                            else:
                                has_detect_position.append(
                                    [y_detect, x_detect])
                                # 若探测到wall line
                                if (
                                    line_map[y_detect, x_detect] > 0
                                    and is_detected is False
                                ):
                                    # 赋值距离map(while逐步外探)
                                    dis_k_line_map[y_point, x_point] = r_detect
                                    no_dis_wall_line = line_map[y_detect,
                                                                x_detect] - 1
                                    this_wall_line = wall_lines[no_dis_wall_line]
                                    this_len = (
                                        (this_wall_line[0] -
                                         this_wall_line[2]) ** 2
                                        + (this_wall_line[1] -
                                           this_wall_line[3]) ** 2
                                    ) ** 0.5
                                    # 赋值新的预测像素所属wall line的长度
                                    len_dis_k_line_map[y_point,
                                                       x_point] = this_len
                                    # 赋值新的预测像素所属wall line的序号
                                    new_line_map[y_point, x_point] = line_map[
                                        y_detect, x_detect
                                    ]
                                    # 赋值斜率特征图
                                    K_map[y_point, x_point] = k_lines[
                                        line_map[y_detect, x_detect] - 1
                                    ]
                                    is_detected = True
                                    # if r_max == 999:
                                    #     print("r_detect:", r_detect)
                                    return
            # 若未探测到则增大探测半径继续探测
            if is_detected is False:
                r_detect += 1

    # classify each prediction pix
    # print("classify each prediction pix")
    r_max = 20  # 最大探测半径
    for yi in range(y):
        for xi in range(x):
            # 若此点为prediction pix, 且不为wall line 轨迹
            if img_bool[yi, xi] == 1 and line_map[yi, xi] == 0:
                point_detect(xi, yi, r_max)

    #### k feature map充填阶段 ####
    # print("fill up k feature map")
    # wall line 方向的探测区域长度
    l_d = 8

    # save the flag that pix is or not filled by same lines
    is_fill_by_same_line = np.zeros((y, x), np.uint8)

    # create k map
    for i in range(len(wall_lines)):
        new_no = i + 1  # 此时的wall line编号
        x1, y1, x2, y2 = wall_lines[i]  # 提取wall line坐标
        k_line = k_lines[i]  # 此条wall line的斜率K
        x1, y1, x_d, y_d = direc_lines[i]  # 方向线坐标，以point1为起点
        vector_wall_i = [x_d - x1, y_d - y1]  # wall line 的方向向量

        # draw line flag on line map
        # 绘制这条wall line 的轨迹（不水平和不竖直的wall line需要以此为生长种子）
        line_flg = np.zeros((y, x), np.uint8)
        cv2.line(line_flg, (x2, y2), (x1, y1), 1, 1)

        # 分为：水平、垂直和其他情况分类处理
        # ROI区域
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        x_min = min(x1, x2)
        x_max = max(x1, x2)

        # 垂直wall line -- x水平扩展
        if x1 == x2:
            # 获取X的方向
            k_direc = 1 if x_d > x1 else -1
            for y_line in range(y_min, y_max + 1):
                for x_line in range(x_min, x_max + 1):
                    # 若存在wall line轨迹，则进行探测
                    if line_flg[y_line, x_line] == 1:
                        is_re_wall_line = False  # 探测结果是否为同向相似墙体线段
                        is_coupled_wall_line = False  # 探测结果是否为成对（斜率相似+墙体方向相反）墙体线段
                        position_coupled_wall_line = None  # 记录成对wall line的点位

                        # 分为：探测阶段+充填阶段
                        # 探测阶段：是否有相似斜率的wall line - 同向/成对做好标记返回
                        for ddx in range(1, l_d + 1):
                            x_k = x_line + k_direc * ddx  # 正向探测端点

                            if x_k < 0 or x_k >= x:
                                break  # 超出img范围，则结束探测

                            # 当探测端点是wall line时，则启动判断
                            if line_map[y_line, x_k] > 0:
                                k_detect = K_map[y_line, x_k]  # 探测点wall line斜率

                                # 如果端点直线K相似，需判断是否同向
                                if is_K_same(k_line, k_detect):
                                    # 探测端点 wall line 信息
                                    # 探测点wall line编号
                                    no_detect = line_map[y_line, x_k] - 1
                                    # 探测点wall line方向
                                    direct_detect = direc_lines[no_detect]
                                    # 探测点wall line方向线坐标，以点1为起点
                                    (
                                        direct_x1,
                                        direct_y1,
                                        direct_x_d,
                                        direct_y_d,
                                    ) = direct_detect
                                    vector_wall_i_detect = [
                                        direct_x_d - direct_x1,
                                        direct_y_d - direct_y1,
                                    ]  # wall line 的方向向量
                                    # 探测点方向是否与原wall line方向一致
                                    is_same_direct = (
                                        True
                                        if (
                                            vector_wall_i_detect[0] *
                                            vector_wall_i[0]
                                            + vector_wall_i_detect[1] * vector_wall_i[1]
                                        )
                                        >= 0
                                        else False
                                    )
                                    # 同向时
                                    if is_same_direct:
                                        is_re_wall_line = True
                                        break
                                    # 异向成对时
                                    else:
                                        is_coupled_wall_line = True
                                        position_coupled_wall_line = [
                                            y_line,
                                            x_k,
                                        ]  # 保存成对wall line点位
                                        break

                        # 如是同向相似时，不进入充填阶段，直接跳出
                        if is_re_wall_line:
                            break

                        # 充填阶段: 优先充填成对wall line / 其次尝试像素点充填
                        if is_coupled_wall_line:
                            y_fill, x_fill = position_coupled_wall_line
                            x_start = min(x_fill, x_line)
                            x_end = max(x_fill, x_line) + 1
                            # 开始充填
                            for xi_fill in range(x_start, x_end):
                                # 若此点未被相似成对wall line生长过 + 是wall line pix
                                if (
                                    is_fill_by_same_line[y_line, xi_fill] != 1
                                    and img_bool[y_line, xi_fill] == 1
                                ):
                                    # 标记此点被成对wall line充填
                                    is_fill_by_same_line[y_line, xi_fill] = 1
                                    K_map[y_line, xi_fill] = max_k  # 赋值此点斜率值
                                    # 赋值此点wall line序号
                                    new_line_map[y_line, xi_fill] = new_no
                            break  # 成对充填结束直接弹出

                        # 像素充填
                        elif is_re_wall_line is False and is_coupled_wall_line is False:
                            num_contect_black_pix = 0  # 是否继续充填的标志-连续空洞像素的数量
                            dis_pix_detect = 0  # 像素探测距离-从自身开始

                            while num_contect_black_pix < 3 and dis_pix_detect <= l_d:
                                # 探测点坐标
                                x_detect = x_line + k_direc * dis_pix_detect
                                y_detect = y_line

                                # 符合充填条件-存在wall像素+没有被成对充填
                                if (
                                    img_bool[y_detect, x_detect] == 1
                                    and is_fill_by_same_line[y_detect, x_detect] == 0
                                ):
                                    num_contect_black_pix = 0  # 重置连续空洞像素数量

                                    # 如未被生长过 或 已经被生长但距离更短，则直接充填
                                    if new_line_map[y_detect, x_detect] == 0 or (
                                        new_line_map[y_detect, x_detect] != 0
                                        and dis_k_line_map[y_detect, x_detect]
                                        > dis_pix_detect
                                    ):
                                        K_map[y_detect, x_detect] = max_k
                                        new_line_map[y_detect,
                                                     x_detect] = new_no
                                        dis_k_line_map[
                                            y_detect, x_detect
                                        ] = dis_pix_detect

                                # 不符合充填条件
                                else:
                                    num_contect_black_pix += 1  # 增加连续空洞

                                # 增加探测距离
                                dis_pix_detect += 1

        # 水平wall line -- y竖直扩展
        elif y1 == y2:
            # 获取Y的方向
            k_direc = 1 if y_d > y1 else -1
            for x_line in range(x_min, x_max + 1):
                if x_line not in range(x):
                    continue
                for y_line in range(y_min, y_max + 1):
                    if y_line not in range(y):
                        continue
                    # 若ROI区域有wall line轨迹，则进行探测
                    if line_flg[y_line, x_line] == 1:
                        is_re_wall_line = False  # 探测结果是否为同向相似墙体线段
                        is_coupled_wall_line = False  # 探测结果是否为成对（斜率相似+墙体方向相反）墙体线段
                        position_coupled_wall_line = None  # 记录成对wall line的点位

                        # 分为：探测阶段+充填阶段

                        # 探测阶段：是否有相似斜率的wall line - 同向/成对做好标记返回
                        for ddy in range(1, l_d + 1):
                            y_k = y_line + k_direc * ddy  # 探测端点

                            if y_k < 0 or y_k >= y:
                                break  # 若超出范围则停止探测

                            # # 若探测过程中出现wall像素，则标记is_exist_wall_pix
                            # if img_bool[y_k,x_line]==1 and is_exist_wall_pix==False: is_exist_wall_pix=True

                            # 若探测端点是wall line，则启动判断
                            if line_map[y_k, x_line] > 0:
                                k_detect = K_map[y_k, x_line]  # 探测点wall line斜率

                                # 如果端点直线K相似，直接充填
                                if is_K_same(k_line, k_detect):
                                    # 探测端点 wall line 信息
                                    # 探测点wall line编号
                                    no_detect = line_map[y_k, x_line] - 1
                                    # 探测点wall line方向
                                    direct_detect = direc_lines[no_detect]
                                    # 探测点wall line方向线坐标，以点1为起点
                                    (
                                        direct_x1,
                                        direct_y1,
                                        direct_x_d,
                                        direct_y_d,
                                    ) = direct_detect
                                    vector_wall_i_detect = [
                                        direct_x_d - direct_x1,
                                        direct_y_d - direct_y1,
                                    ]  # wall line 的方向向量
                                    # 探测点方向是否与原wall line方向一致
                                    is_same_direct = (
                                        True
                                        if (
                                            vector_wall_i_detect[0] *
                                            vector_wall_i[0]
                                            + vector_wall_i_detect[1] * vector_wall_i[1]
                                        )
                                        >= 0
                                        else False
                                    )
                                    # 同向时
                                    if is_same_direct:
                                        is_re_wall_line = True
                                        break
                                    # 异向成对时
                                    else:
                                        is_coupled_wall_line = True
                                        position_coupled_wall_line = [
                                            y_k,
                                            x_line,
                                        ]  # 保存成对wall line点位
                                        break

                        # 如是同向相似时，不进入充填阶段，直接跳出
                        if is_re_wall_line:
                            break

                        # 充填阶段: 优先充填成对wall line / 其次尝试像素点充填
                        if is_coupled_wall_line:
                            y_fill, x_fill = position_coupled_wall_line
                            y_start = min(y_fill, y_line)
                            y_end = max(y_fill, y_line) + 1
                            # 开始充填
                            for yi_fill in range(y_start, y_end):
                                # 若此点未被相似成对wall line生长过  + 是wall line pix
                                if (
                                    is_fill_by_same_line[yi_fill, x_line] != 1
                                    and img_bool[yi_fill, x_line] == 1
                                ):
                                    # 标记此点被成对wall line充填
                                    is_fill_by_same_line[yi_fill, x_line] = 1
                                    K_map[yi_fill, x_line] = 0  # 赋值此点斜率值
                                    # 赋值此点wall line序号
                                    new_line_map[yi_fill, x_line] = new_no
                            break  # 成对充填结束直接弹出

                        # 像素充填阶段
                        elif is_re_wall_line is False and is_coupled_wall_line is False:
                            num_contect_black_pix = 0  # 是否继续充填的标志-连续空洞像素的数量
                            dis_pix_detect = 0  # 像素探测距离-从自身开始

                            while num_contect_black_pix < 3 and dis_pix_detect <= l_d:
                                # 探测点坐标
                                x_detect = x_line
                                y_detect = y_line + k_direc * dis_pix_detect

                                # 符合充填条件-存在wall像素+没有被成对充填
                                if (
                                    img_bool[y_detect, x_detect] == 1
                                    and is_fill_by_same_line[y_detect, x_detect] == 0
                                ):
                                    num_contect_black_pix = 0  # 重置连续空洞像素数量

                                    # 如未被生长过 或 已经被生长但距离更短，则直接充填
                                    if new_line_map[y_detect, x_detect] == 0 or (
                                        new_line_map[y_detect, x_detect] != 0
                                        and dis_k_line_map[y_detect, x_detect]
                                        > dis_pix_detect
                                    ):
                                        K_map[y_detect, x_detect] = 0
                                        new_line_map[y_detect,
                                                     x_detect] = new_no
                                        dis_k_line_map[
                                            y_detect, x_detect
                                        ] = dis_pix_detect

                                # 不符合充填条件
                                else:
                                    num_contect_black_pix += 1  # 增加连续空洞

                                # 增加探测距离
                                dis_pix_detect += 1

        # 斜边扩展
        else:
            # 获取X,Y的方向
            xk_direc = 1 if x_d > x1 else -1
            dk = -1 / k_line  # 探测线的斜率
            dx = (l_d ** 2 / (1 + dk ** 2)) ** 0.5  # 探测线长度的X变量 - 始终大于等于0
            dy = dk * dx if dk > 0 else -dk * dx  # 探测线长度的Y变量 - 始终大于等于0

            for x_line in range(x_min, x_max + 1):
                if x_line not in range(x):
                    continue
                for y_line in range(y_min, y_max + 1):
                    if y_line not in range(y):
                        continue
                    # 若ROI区域有wall line轨迹
                    if line_flg[y_line, x_line] == 1:
                        is_re_wall_line = False  # 探测结果是否为同向相似墙体线段
                        is_coupled_wall_line = False  # 探测结果是否为成对（斜率相似+墙体方向相反）墙体线段
                        position_coupled_wall_line = []  # 记录成对wall line的点位

                        # 分为：探测阶段+充填阶段
                        # 探测阶段
                        # 确定过此点的探测直线方程
                        db = y_line - dk * x_line
                        # 确定探测端点
                        x_k = x_line + xk_direc * dx
                        y_k = dk * x_k + db
                        x_k, y_k = int(x_k), int(y_k)

                        # 绘制探测线段
                        detection_line = np.zeros((y, x), np.uint8)
                        cv2.line(
                            detection_line, (x_line, y_line), (int(
                                x_k), int(y_k)), 1, 1
                        )

                        # 根据探测标记进行探测:是否有相似斜率的wall line - 同向/成对做好标记返回
                        # 以（x_line, y_line）为起始点向（x_k,y_k）探测
                        # 创建遍历路径
                        has_passed = np.zeros((y, x), np.uint8)

                        # 初始化起始点
                        start_x = x_line
                        start_y = y_line

                        # 开始探测 - 是否有成对wall line
                        while (
                            has_passed[y_k, x_k] == 0
                            and (is_re_wall_line is False)
                            and (is_coupled_wall_line is False)
                        ):
                            has_passed[start_y, start_x] = 1  # 标记此点为已遍历
                            position_coupled_wall_line.append(
                                [start_y, start_x]
                            )  # 保存路径轨迹，成对wall line点位
                            # 以此点为中心，探测下一个像素点
                            is_find_next = False  # 是否找到下一个像素点
                            # 八邻域探测
                            for d_x_i in range(-1, 2):
                                if is_find_next:
                                    break
                                ner_x = start_x + d_x_i
                                if ner_x not in range(x):
                                    continue
                                for d_y_i in range(-1, 2):
                                    if d_x_i == 0 and d_y_i == 0:
                                        continue  # 不遍历自身
                                    ner_y = start_y + d_y_i
                                    if ner_y not in range(y):
                                        continue
                                    # 找到下一个点
                                    if (
                                        detection_line[ner_y, ner_x] == 1
                                        and has_passed[ner_y, ner_x] == 0
                                    ):
                                        start_x = ner_x
                                        start_y = ner_y
                                        is_find_next = True

                                        # 检测是否为wall line
                                        if line_map[start_y, start_x] > 0:
                                            # 探测点wall line斜率
                                            k_detect = K_map[start_y, start_x]
                                            # 如果端点直线K相似，需判断是否同向
                                            if is_K_same(k_line, k_detect):
                                                # 探测端点 wall line 信息
                                                # 探测点wall line编号
                                                no_detect = (
                                                    line_map[start_y,
                                                             start_x] - 1
                                                )
                                                # 探测点wall line方向
                                                direct_detect = direc_lines[no_detect]
                                                # 探测点wall line方向线坐标，以点1为起点
                                                (
                                                    direct_x1,
                                                    direct_y1,
                                                    direct_x_d,
                                                    direct_y_d,
                                                ) = direct_detect
                                                vector_wall_i_detect = [
                                                    direct_x_d - direct_x1,
                                                    direct_y_d - direct_y1,
                                                ]  # wall line 的方向向量
                                                # 探测点方向是否与原wall line方向一致
                                                is_same_direct = (
                                                    True
                                                    if (
                                                        vector_wall_i_detect[0]
                                                        * vector_wall_i[0]
                                                        + vector_wall_i_detect[1]
                                                        * vector_wall_i[1]
                                                    )
                                                    >= 0
                                                    else False
                                                )
                                                # 同向时
                                                if is_same_direct:
                                                    is_re_wall_line = True
                                                    break
                                                # 异向成对时
                                                else:
                                                    is_coupled_wall_line = True
                                                    break

                                        break  # 由于已经找到了下一个探测像素点，因此跳出此八邻域寻找循环

                        # 判断充填规则
                        # 若是同向wall line , 不进入充填阶段， 直接跳出
                        if is_re_wall_line:
                            break

                        # 充填阶段 - 优先充填成对wall line / 其次尝试像素点充填
                        if is_coupled_wall_line:
                            for fill_point in position_coupled_wall_line:
                                y_fill, x_fill = fill_point
                                if (
                                    is_fill_by_same_line[y_fill, x_fill] == 0
                                    and img_bool[y_fill, x_fill] == 1
                                ):
                                    is_fill_by_same_line[y_fill, x_fill] = 1
                                    K_map[y_fill, x_fill] = k_line
                                    new_line_map[y_fill, x_fill] = new_no
                            continue  # 完成成对wall line 充填后，继续探测下一个像素点

                        # 尝试像素点充填 - 如无相似wall line，则启动像素点充填试探
                        elif is_re_wall_line is False and is_coupled_wall_line is False:
                            num_contect_black_pix = 0  # 是否继续充填的标志-连续空洞像素的数量
                            dis_pix_detect = 0  # 记录像素探测距离-从自身开始

                            # 由于遍历位置保存在position_coupled_wall_line，且从自身开始
                            for fill_point in position_coupled_wall_line:
                                if num_contect_black_pix > 3:
                                    break
                                y_fill, x_fill = fill_point
                                # 符合充填条件-存在wall像素+没有被成对充填
                                if (
                                    img_bool[y_fill, x_fill] == 1
                                    and is_fill_by_same_line[y_fill, x_fill] == 0
                                ):
                                    num_contect_black_pix = 0  # 重置连续空洞像素数量
                                    # 如未被生长过 或 已经被生长但距离更短，则直接充填
                                    if new_line_map[y_fill, x_fill] == 0 or (
                                        new_line_map[y_fill, x_fill] != 0
                                        and dis_k_line_map[y_fill, x_fill]
                                        > dis_pix_detect
                                    ):
                                        K_map[y_fill, x_fill] = k_line
                                        new_line_map[y_fill, x_fill] = new_no
                                        dis_k_line_map[y_fill,
                                                       x_fill] = dis_pix_detect
                                else:
                                    num_contect_black_pix += 1
                                # 增加探测距离
                                dis_pix_detect += 1

    ###############this part classify the rest of wall pix #########################################################
    # classify rest each prediction pix
    # print("classify rest each prediction pix")
    r_max = 999  # 最大探测半径
    # print("img size", [y, x])
    for yi in range(y):
        for xi in range(x):
            # 若此点为prediction pix, 且没有被分类
            if img_bool[yi, xi] == 1 and new_line_map[yi, xi] == 0:
                # print("points position:", [yi, xi])
                point_detect(xi, yi, r_max)

    ######### 区域生长阶段 ############
    # print("区域生长阶段")

    def isPointIn(point_x, point_y, x_b_min=0, x_b_max=x, y_b_min=0, y_b_max=y):
        # 判断点是否在某一区域内
        if (
            point_x >= x_b_min
            and point_x < x_b_max
            and point_y >= y_b_min
            and point_y < y_b_max
        ):
            return True
        else:
            return False

    # 区域生长
    def k_RG(point_x, point_y, k_stand, k_rg_map, k_rg_flag_map, k_size=3, thr=0.5):
        # 8邻域
        # 0.5
        # 判断是否已经遍历过
        if k_rg_flag_map[point_y, point_x] != 255:
            k_rg_flag_map[point_y, point_x] = 255  # 标记已经遍历
            x_nears = []
            y_nears = []
            for i in range(k_size):
                for j in range(k_size):
                    x_near = point_x + i - 1
                    y_near = point_y + j - 1
                    if isPointIn(x_near, y_near):
                        x_nears.append(x_near)
                        y_nears.append(y_near)
            num_near = len(x_nears)
            # 是否属于k_stand区域
            num_k = 0
            num_same_k = 0
            for i in range(num_near):
                x_near, y_near = x_nears[i], y_nears[i]
                if new_line_map[y_near, x_near] > 0:
                    num_k += 1
                    if is_K_same(k_stand, K_map[y_near, x_near]):
                        num_same_k += 1
            seeds_x = []
            seeds_y = []
            if num_same_k >= int(num_k * thr) and num_k >= int(num_near * thr):
                k_rg_map[point_y, point_x] = 255
                seeds_x.append(point_x)
                seeds_y.append(point_y)
            # 四周生长
            while len(seeds_x) > 0:
                seed_x = seeds_x.pop(0)
                seed_y = seeds_y.pop(0)
                for ii in range(k_size):
                    for jj in range(k_size):
                        seed_x_near = seed_x - 1 + ii
                        seed_y_near = seed_y - 1 + jj
                        if (
                            isPointIn(seed_x_near, seed_y_near)
                            and k_rg_flag_map[seed_y_near, seed_x_near] == 0
                        ):
                            k_rg_flag_map[seed_y_near, seed_x_near] = 255
                            is_same_k_flag = False
                            num_near_seed = 0
                            num_k_seed = 0
                            num_same_k_seed = 0
                            for iii in range(k_size):
                                for jjj in range(k_size):
                                    seed_x_near_near = seed_x_near - 1 + iii
                                    seed_y_near_near = seed_y_near - 1 + jjj
                                    if isPointIn(seed_x_near_near, seed_y_near_near):
                                        num_near_seed += 1
                                        if (
                                            new_line_map[
                                                seed_y_near_near, seed_x_near_near
                                            ]
                                            > 0
                                        ):
                                            num_k_seed += 1
                                            if is_K_same(
                                                k_stand,
                                                K_map[
                                                    seed_y_near_near, seed_x_near_near
                                                ],
                                            ):
                                                num_same_k_seed += 1
                            if num_same_k_seed >= int(
                                num_k_seed * thr
                            ) and num_k_seed >= int(num_near_seed * thr):
                                is_same_k_flag = True
                            if is_same_k_flag:
                                k_rg_map[seed_y_near, seed_x_near] = 255
                                seeds_x.append(seed_x_near)
                                seeds_y.append(seed_y_near)

    # wall_vectors = []
    img_vector = np.zeros((img_shape[0], img_shape[1]), np.uint8)
    # img_feature = np.zeros((img_shape[0], img_shape[1], 3), np.uint8)

    k_rgs = [0, max_k, 1, -1, 0.4142, 2.4142, -0.4142, -2.4142]
    rg_flag_map = np.zeros((y, x), np.uint8)
    for k_rg in k_rgs:
        rg_map = np.zeros((y, x), np.uint8)
        for x_rg in range(x):
            for y_rg in range(y):
                if new_line_map[y_rg, x_rg] > 0 and is_K_same(K_map[y_rg, x_rg], k_rg):
                    k_RG(
                        x_rg,
                        y_rg,
                        k_stand=k_rg,
                        k_rg_map=rg_map,
                        k_rg_flag_map=rg_flag_map,
                    )

        # 反变换到原始尺寸
        resize_shape = (img_shape[1], img_shape[0])
        rg_map = cv2.resize(rg_map, resize_shape,
                            interpolation=cv2.INTER_NEAREST)

        # # draw faeture
        # if k_rg == 0:
        #     img_feature[:, :, 0] += rg_map
        # elif k_rg == max_k:
        #     img_feature[:, :, 1] += rg_map
        # else:
        #     img_feature[:, :, 2] += rg_map

        # 拟合轮廓成矩形
        contours, _ = cv2.findContours(
            rg_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # ***基于:轮廓面积>area_thr---过滤杂点
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5:
                continue
            cnts = findMaxMIOU(cnt, k_rg)
            for cnt_i in cnts:
                rect = cv2.minAreaRect(cnt_i)
                points = cv2.boxPoints(rect)

                # draw vecter img
                cv2.drawContours(img_vector, [np.int0(points)], 0, (255), -1)
                """
                # vector points
                l1 = (points[0][0]-points[1][0])**2 + \
                    (points[0][1]-points[1][1])**2
                l2 = (points[2][0]-points[1][0])**2 + \
                    (points[2][1]-points[1][1])**2
                min_l = l1 if l1 < l2 else l2
                width_wall = min_l**0.5
                x_e_vector = (points[0][0]+points[1][0]) / \
                    2 if min_l == l1 else (points[2][0]+points[1][0])/2
                y_e_vector = (points[0][1]+points[1][1]) / \
                    2 if min_l == l1 else (points[2][1]+points[1][1])/2
                x_s_vector = (points[2][0]+points[3][0]) / \
                    2 if min_l == l1 else (points[0][0]+points[3][0])/2
                y_s_vector = (points[2][1]+points[3][1]) / \
                    2 if min_l == l1 else (points[0][1]+points[3][1])/2

                # Coordinate system conversion - y_axis conversion
                y_e_vector = img_shape[0]-y_e_vector-1
                y_s_vector = img_shape[0]-y_s_vector-1

                wall_vector = {}
                wall_vector['sPoint'] = [x_e_vector, y_e_vector]
                wall_vector['ePoint'] = [x_s_vector, y_s_vector]
                wall_vector['width'] = width_wall
                wall_vector['height'] = 0
                wall_vector['isStructural'] = True
                wall_vectors.append(wall_vector)
                """
    return img_vector  # wall_lines, img_pre, img_feature,   img_wallLines,K_map
    # return wall_vectors
