import math
import sys

import cv2
import numpy as np
import random
import os
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

char_num = ['10', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'A', 'J', 'Q', 'K']
char_shape = ['红桃', '方片', '黑桃', '梅花']
char_other = ['other']
char_table = char_num + char_shape + char_other

cur_dir = sys.path[0]
data_dir = os.path.join(cur_dir, 'pic/screenshot')
char_model_path = os.path.join(cur_dir, "model/char_recongnize/model1.ckpt")

# 加载字符识别模型
model = tf.saved_model.load(char_model_path)

def proc_sigle_card(card, prefix=""):
    """
    该函数处理单张牌为颜色+两个字符
    输入：牌全色图，牌二值化图
    输出：颜色，字符图集合
    """
    # cv2.imshow('aaa', cardBinImg)
    color = card[0]
    card_bin_img = card[1]
    image, contours, bin = cv2.findContours(card_bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    prefix += '_'
    prefix += str(random.randint(0, 10000000))
    prefix += '_'
    i = 0
    img_list = []
    # 需要处理的是不是王
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 1500 or w < 5 or h < 5:  # 过小或者比例严重不符合
            continue
        count += 1
    if count > 3:
        # 这里依据王的图形比较复杂确定
        if color:
            return "大王"
        else:
            return "小王"

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 1500 or w < 5 or h < 5:  # 过小或者比例严重不符合
            continue
        # cv2.rectangle(cardBinImg, (x, y), (x + w, y + h), (255, 0, 0), 3)
        char_img = card_bin_img[y:y + h, x:x + w]
        max_l = max(h, w)
        size = (20, 20)
        offset_x = 0
        offset_y = 0
        if max_l == w:
            w1 = int(20 * (h * 1.0 / w))
            size = (20, w1)
            offset_y = (20 - w1) / 2
        if max_l == h:
            h1 = int(20 * (w * 1.0 / h))
            size = (h1, 20)
            offset_x = (20 - h1) / 2
        char_img = cv2.resize(char_img, size)
        blank_image = np.ones((20, 20), np.uint8) * 0
        blank_image[int(offset_y):(int(offset_y) + size[1]), int(offset_x):(int(offset_x) + size[0])] = char_img
        img_list.append(blank_image)
        # cv2.imshow('fff', cardBinImg)
        # cv2.imwrite('./pic/char/' + prefix + str(i) + '.png', blank_image)
        i += 1
    return img_list


def judgeColor(hsvImg):
    """
    判断颜色
    """
    shape = hsvImg.shape
    height = shape[0]
    weight = shape[1]
    count = 0
    for i in range(height):
        for j in range(weight):
            hsv = hsvImg[i, j]
            if (156 <= hsv[0] <= 180 or 0 <= hsv[0] <= 10 ) and 80 <= hsv[1] <= 220 and 50 <= hsv[2] <= 255:
                count +=1
    # print(count)
    if count >= 40:
        return True
    return False

def findCards(colorImg):
    """
    提取图片中所有可见牌，并分不同区域返回
    参数 彩色图片
    返回 [inHans:[牌图],outHands:[牌图],back:[牌图]]
    """
    gray_img = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(colorImg, cv2.COLOR_BGR2HSV)

    ret, binary = cv2.threshold(gray_img, 170, 255, cv2.THRESH_BINARY_INV)
    k1 = np.ones((3, 3), np.uint8)
    up = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k1)
    k2 = np.ones((5, 5), np.uint8)
    up = cv2.morphologyEx(up, cv2.MORPH_OPEN, k2)
    ret, binaryinv = cv2.threshold(up, 170, 255, cv2.THRESH_BINARY_INV)
    image, contours, bin = cv2.findContours(binaryinv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('bin_img', binary)
    count = 0
    in_hands_imgs = []
    out_hands_imgs = []
    back_imgs = []
    tmp = True
    for contour in contours:
        area = cv2.contourArea(contour)
        # 通过面积过滤非牌面区域
        if area > 1200:
            x, y, w, h = cv2.boundingRect(contour)
            print(x, y, w, h)
            cards_bin_img = binary[y:y + h, x:x + w]
            cards_hsv_img = hsv_img[y:y + h, x:x + w]
            # cv2.rectangle(colorImg, (x, y), (x + w, y + h), (255, 0, 0), 3)
            if y > 310:
                # 兼容无法出牌情况
                pass
            elif y > 235:
                # 自己的牌
                # cv2.imwrite('./' + name + '_myCardsImg.png', cardsGrayImg)
                card_count = int(round((w - 56) * 1.0 / 32))
                for i in range(0, card_count):
                    card = cards_bin_img[0:h, int(32.5 * i):int(32.5 * i) + 32]
                    hsv_card = cards_hsv_img[0:h, int(32.5 * i):int(32.5 * i) + 32]
                    is_red = judgeColor(hsv_card)
                    # cv2.imwrite('./my/' + str(i) + '.png', card)
                    in_hands_imgs.append((is_red,card))
                    # procSigleCard(card, name+'_my')
            elif y < 40:
                # 底牌
                # cv2.imwrite('./' + name + '_bottomCardsImg.png', bottomCardsImg)
                card_count = int(round(w * 1.0 / 16))
                for i in range(0, card_count):
                    card = cards_bin_img[0:h, int(16 * i):int(16 * i) + 16]
                    hsv_card = cards_hsv_img[0:h, int(16 * i):int(16 * i) + 16]
                    is_red = judgeColor(hsv_card)
                    # grayCard = cardsGrayImg[0:h, int(16 * i):int(16 * i) + 16]
                    # cv2.imwrite('./bottom/' + str(i) + '.png', card)
                    back_imgs.append((is_red,card))
                    # cv2.imwrite('./bottom/' + name + "_" + str(i) + '.png', card)
            else:
                # 出牌区
                # cv2.imwrite('./outingCardsImg_'+str(count)+'.png', cardsBinImg)
                # TODO 此处仅考虑单行情况 牌过多折行会出错
                card_count = int(round((w - 25) * 1.0 / 21))
                for i in range(0, card_count):
                    card = cards_bin_img[0:h, int(21.5 * i):int(21.5 * i) + 21]
                    hsv_card = cards_hsv_img[0:h, int(21.5 * i):int(21.5 * i) + 21]
                    is_red = judgeColor(hsv_card)
                    out_hands_imgs.append((is_red,card))
                count += 1
    return in_hands_imgs, out_hands_imgs, back_imgs


def proc_pic(filepath):
    """
    处理单张游戏截图
    """
    color_img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    color_img = cv2.resize(color_img, (int(800), int(400)))
    # cv2.imshow(filepath, color_img)

    in_hand_imgs, out_hand_imgs, back_imgs = findCards(color_img)
    name = os.path.split(filepath)[-1].split('.')[0]
    # 手牌判断
    print('----inhands------')
    chars = recg_chars(in_hand_imgs, name)
    print(chars)
    print('-----out-----')
    # 出牌判断
    chars = recg_chars(out_hand_imgs, name)
    print(chars)
    print('-----back-----')
    # 底牌判断
    chars = recg_chars(back_imgs, name)
    print(chars)


def recg_chars(in_hand_imgs, name):
    """
    批量识别牌型
    """
    img_list = []
    chars = []
    for i in range(len(in_hand_imgs)):
        card = in_hand_imgs[i]
        card_content = proc_sigle_card(card, name + "_inhand")
        if isinstance(card_content, str):
            chars.append([card_content])
        else:
            img_list.extend(card_content)
    chars.extend(cnn_reconginze_char_2(img_list))
    return chars

@tf.function(experimental_relax_shapes=True)
def call_model(img):
    return model(img)

def cnn_reconginze_char_2(img_list):
    if len(img_list) <= 0:
        return []
    img_list = np.array(img_list)
    test_X = img_list.reshape(-1, 20, 20, 1).astype('float32')
    y_pred = call_model(test_X)
    text_list = []
    tmp = []
    for y in y_pred:
        char = char_table[np.argmax(y)]
        if char in char_num:
            tmp.append(char)
            if len(tmp) > 1:
                text_list.append(tmp)
            tmp = []
        if char in char_shape:
            tmp.append(char)
    return text_list

def list_all_files(root):
    files = []
    file_list = os.listdir(root)
    for i in range(len(file_list)):
        element = os.path.join(root, file_list[i])
        if file_list[i] == '.DS_Store':
            continue
        if os.path.isfile(element):
            files.append(element)
    return files

if __name__ == "__main__":

    files = list_all_files(data_dir)
    for file in files:
        proc_pic(file)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
