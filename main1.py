import math
import os.path
import shutil
import sys
import time

import imagehash
import numpy
import pyautogui
import win32gui
from PIL import Image, ImageGrab
import pytesseract
import cv2
import numpy as np
from numpy import average, dot, linalg
from matplotlib import pyplot as plt
# from python_imagesearch.imagesearch import imagesearch
import pyautogui as pg

from os import listdir
from os.path import isfile, isdir, join

from random import choice, sample
from timeit import default_timer as timer
import win32api
from imagehash import average_hash
from array import array
from enum import Enum
from itertools import combinations
import win32file


class GameStatue(Enum):
    Fighting = 0
    Victory = 1
    Lost = 1
    WinWithExp = 2
    WinWithSLPBounds = 3
    MissionSelect = 4


class StepType(Enum):
    CONTINUE = 'continue'
    BREAK = 'break'
    STEPOVER = 'stepover'


class HealCards(Enum):
    Berry = '3-2.JPG'
    FrontHeal = '3-1.JPG'
    DOHeal = '2-2.JPG'


TESSERACT_OCR = r'C:\\Program Files\\tesseract-ORC\\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_OCR
# https://github.com/tesseract-ocr
AXIE_WINDOW_PROPERTY = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0, 'width': 0, 'height': 0}


class GalleryLibrary(Enum):
    AXIE_NORMAL_Icon = '.\\picture\\gallery\\axies\\status_library\\normal'
    AXIE_FIRE_Icon = '.\\picture\\gallery\\axies\\status_library\\fire'
    AXIE_DIE_Icon = '.\\picture\\gallery\\axies\\status_library\\die'
    AXIE_HP_Icon = '.\\picture\\gallery\\axies\\hp_icon'
    AXIE_Sequence = '.\\picture\\gallery\\axies\\sequence'
    AXIE_RUNTIME_FULL = '.\\picture\\runtime\\axies\\full'
    AXIE_RUNTIME_DEBUFF = '.\\picture\\runtime\\axies\\debuff'

    Button_Energy_0_Icon = '.\\picture\\gallery\\button\\energy_0.JPG'
    Button_Energy_1_Icon = '.\\picture\\gallery\\button\\energy_1.JPG'
    Button_Victory_Icon = '.\\picture\\gallery\\button\\victory.JPG'
    Button_Defeated_Icon = '.\\picture\\gallery\\button\\defeated_fight.JPG'
    Button_Defeated_Result_Icon = '.\\picture\\gallery\\button\\defeated_result.JPG'  # 這是在戰鬥結束後，會有三隻畫加一個defeated的過場動畫的畫面
    Button_Start_Mission_Button = '.\\picture\\gallery\\button\\start.JPG'
    Button_SLP_Bounds = '.\\picture\\gallery\\button\\bounds_1.JPG'
    Button_EndTurn_Button = '.\\picture\\gallery\\button\\end_turn.JPG'
    Button_Debuff_1_Icon = '.\\picture\\gallery\\button\\debuff_1.JPG'

    Misson_Final_Icon = '.\\picture\\gallery\\mission\\final'

    Monster_Runtime_HP = '.\\picture\\runtime\\monsters\\hp'

    Cards_Library = '.\\picture\\gallery\\cards'
    Temp_Origin = '.\\picture\\origin.JPG'
    Temp_Image = '.\\picture\\image_temp.JPG'
    Full_Image = '.\\picture\\1.png'


print(GalleryLibrary.AXIE_NORMAL_Icon.value)
print(GalleryLibrary.Cards_Library.value)
CARDS_LIBRARY_LIST = listdir(GalleryLibrary.Cards_Library.value)
AXIE_NORMAL_LIST = listdir(GalleryLibrary.AXIE_NORMAL_Icon.value)

MISSION_ID = 19
pyautogui.PAUSE = 1
MAX_WIN_COUNT = 5
#      1      4
#  0       3     6
#      2      5
# pvp mode
AXIES_FORMATION = [{'id': '1986033', 'position_id': 6},  # fish
                   {'id': '6805632', 'position_id': 3},  # pink
                   {'id': '7035133', 'position_id': 0}  # green
                   ]
AXIES_CARDS_PIC_PREFIX = {'1986033': '0', '6805632': '1', '7035133': '2'}
# class AXIE_STATUS(Enum):
#     NORMAL = 'normal'
#     FIRE = 'fire'
#     DIE = 'die'


AXIE_STATUS = ['normal', 'fire', 'die']


class POSITION_RUEL(Enum):
    SINGLE = 'single'
    DOUBLE = 'double'


AXIE_RUNTIME_SNAPSHOP_LOCATION = [
    {'id': "1986033", 'nickname': 'fish', 'left': -110, 'top': -200, 'right': 85, 'bottom': -15},
    {'id': "6805632", 'nickname': 'pink', 'left': -110, 'top': -200, 'right': 85, 'bottom': -15},
    {'id': "7035133", 'nickname': 'green', 'left': -110, 'top': -200, 'right': 85, 'bottom': -15},
    {'id': "1980381", 'nickname': 'do', 'left': -110, 'top': -200, 'right': 85, 'bottom': -15}
]

AXIE_RUNTIME_DEBUFF_SNAPSHOP_LOCATION = [
    {'id': "1986033", 'nickname': 'fish', 'left': -95, 'top': -220, 'right': 75, 'bottom': -130},
    {'id': "6805632", 'nickname': 'pink', 'left': -95, 'top': -220, 'right': 75, 'bottom': -130},
    {'id': "7035133", 'nickname': 'green', 'left': -95, 'top': -220, 'right': 75, 'bottom': -130}
]

AXIES_POSITION_ID = [
    {'id': 0, 'x': 711, 'y': 576, 'row': 0, 'col': 1, 'rule': POSITION_RUEL.DOUBLE},
    {'id': 1, 'x': 600, 'y': 474, 'row': 1, 'col': 0, 'rule': POSITION_RUEL.DOUBLE},
    {'id': 2, 'x': 600, 'y': 676, 'row': 1, 'col': 2, 'rule': POSITION_RUEL.SINGLE},
    {'id': 3, 'x': 485, 'y': 576, 'row': 2, 'col': 1, 'rule': POSITION_RUEL.SINGLE},
    {'id': 4, 'x': 377, 'y': 474, 'row': 3, 'col': 0, 'rule': POSITION_RUEL.SINGLE},
    {'id': 5, 'x': 377, 'y': 676, 'row': 4, 'col': 2, 'rule': POSITION_RUEL.DOUBLE},
    {'id': 6, 'x': 262, 'y': 576, 'row': 5, 'col': 1, 'rule': POSITION_RUEL.DOUBLE}
]
AXIE_PVP_SNAPSHOP = {'left': -90, 'top': -160, 'right': 75, 'bottom': -90}
AXIE_PVP_POSITION_ID = [
    {'id': 0, 'x': 772, 'y': 496, 'row': 0, 'col': 1, 'rule': POSITION_RUEL.SINGLE},
    {'id': 1, 'x': 867, 'y': 410, 'row': 1, 'col': 0, 'rule': POSITION_RUEL.DOUBLE},
    {'id': 2, 'x': 867, 'y': 582, 'row': 1, 'col': 2, 'rule': POSITION_RUEL.DOUBLE},
    {'id': 3, 'x': 962, 'y': 496, 'row': 2, 'col': 1, 'rule': POSITION_RUEL.SINGLE},
    {'id': 4, 'x': 1057, 'y': 410, 'row': 3, 'col': 0, 'rule': POSITION_RUEL.DOUBLE},
    {'id': 5, 'x': 1057, 'y': 582, 'row': 3, 'col': 2, 'rule': POSITION_RUEL.DOUBLE},
    {'id': 6, 'x': 1152, 'y': 496, 'row': 4, 'col': 1, 'rule': POSITION_RUEL.SINGLE}
]


class AXIE_DEBUFF(Enum):
    Nothing = 0
    DISABLE_ONE_CARD = 1


AXIES_NICKNAME = {'1986033': 'Fish', '1980381': 'do', '6805632': 'Pink', '7035133': 'Green'}
AXIE_HEAL_RELATION = [{'id': "1986033", 'healby': '1985659', 'healcard': ['3-1.JPG', '3-2.JPG']},
                      {'id': "1980381", 'healby': '1980381', 'healcard': ['2-2.JPG']},
                      {'id': "6805632", 'healby': '1985659', 'healcard': ['3-2.JPG', '3-3.JPG']},
                      {'id': "7035133", 'healby': '1985659', 'healcard': ['3-2.JPG', '3-3.JPG']}
                      ]
MY_AXIES = [
    {'id': "1986033", 'nickname': 'fish', 'health': 1315, 'speed': 136, 'skill': 88, 'morale': 80,
     'x': 0, 'y': 0, 'debuff': [], 'healByWhichAxie': '1980381',
     'cards': [{'name': '1986033-1.JPG', 'attack': 387, 'shield': 96, 'energy': 1, 'heal': 0,
                'property': ['attack', 'shield'], 'other': '', 'comboed': False, 'debuff_seq': 3},
               {'name': '1986033-2.JPG', 'attack': 161, 'shield': 258, 'energy': 1, 'heal': 0,
                'property': ['attack', 'shield'], 'other': 'add 30% shield', 'comboed': False, 'debuff_seq': 0},
               {'name': '1986033-3.JPG', 'attack': 354, 'shield': 64, 'energy': 1, 'heal': 0,
                'property': ['attack', 'shield', 'speed'], 'other': '', 'comboed': False, 'debuff_seq': 1},
               {'name': '1986033-4.JPG', 'attack': 354, 'shield': 96, 'energy': 1, 'heal': 0,
                'property': ['attack', 'shield', 'speed'], 'other': '', 'comboed': True, 'debuff_seq': 2},
               ]},
    {'id': "6805632", 'nickname': 'pink', 'health': 943, 'speed': 91, 'skill': 64, 'morale': 8591,
     'x': 0, 'y': 0, 'debuff': [], 'healByWhichAxie': '1985659',
     'cards': [{'name': '6805632-1.JPG', 'attack': 174, 'shield': 109, 'energy': 1, 'heal': 0,
                'property': ['attack', 'shield'], 'other': 'heal front', 'comboed': True, 'debuff_seq': 3},
               {'name': '6805632-2.JPG', 'attack': 91, 'shield': 0, 'energy': 0, 'heal': 0,
                'property': ['attack'], 'other': 'heal front or itself', 'comboed': False, 'debuff_seq': 1},
               {'name': '6805632-3.JPG', 'attack': 87, 'shield': 130, 'energy': 1, 'heal': 0,
                'property': ['attack', 'shield'], 'other': '', 'comboed': False, 'debuff_seq': 0},
               {'name': '6805632-4.JPG', 'attack': 240, 'shield': 68, 'energy': 1, 'heal': 0,
                'property': ['attack', 'shield'], 'other': '', 'comboed': False, 'debuff_seq': 2},
               ]},
    {'id': "7035133", 'nickname': 'green', 'health': 1182, 'speed': 71, 'skill': 88, 'morale': 85,
     'x': 0, 'y': 0, 'debuff': [], 'healByWhichAxie': '1985659',
     'cards': [{'name': '7035133-1.JPG', 'attack': 68, 'shield': 68, 'energy': 1, 'heal': 0,
                'property': ['attack', 'shield'], 'other': 'heal front', 'comboed': True, 'debuff_seq': 1},
               {'name': '7035133-2.JPG', 'attack': 0, 'shield': 91, 'energy': 1, 'heal': 275,
                'property': ['heal'], 'other': 'heal front or itself', 'comboed': False, 'debuff_seq': 2},
               {'name': '7035133-3.JPG', 'attack': 103, 'shield': 183, 'energy': 1, 'heal': 0,
                'property': ['attack', 'shield'], 'other': '', 'comboed': False, 'debuff_seq': 0},
               {'name': '7035133-4.JPG', 'attack': 160, 'shield': 91, 'energy': 1, 'heal': 0,
                'property': ['attack', 'shield'], 'other': '', 'comboed': False, 'debuff_seq': 3},
               ]},
    {'id': "1980381", 'nickname': 'do', 'health': 1393, 'speed': 130, 'skill': 88, 'morale': 88,
     'x': 0, 'y': 0, 'debuff': [], 'healByWhichAxie': '1980381',
     'cards': [{'name': '2-1.JPG', 'attack': 338, 'shield': 96, 'energy': 1, 'heal': 0,
                'property': ['attack', 'shield'], 'other': 'attack 120%', 'comboed': True, 'debuff_seq': 2},
               {'name': '2-2.JPG', 'attack': 0, 'shield': 129, 'energy': 1, 'heal': 387,
                'property': ['heal', 'shield'], 'other': '', 'comboed': False, 'debuff_seq': 3},
               {'name': '2-3.JPG', 'attack': 161, 'shield': 322, 'energy': 1, 'heal': 0,
                'property': ['attack', 'shield'], 'other': 'plant target plus', 'comboed': False, 'debuff_seq': 1},
               {'name': '2-4.JPG', 'attack': 96, 'shield': 0, 'energy': 0, 'heal': 0,
                'property': ['attack', 'get energy'], 'other': 'gain 1 energy', 'comboed': True, 'debuff_seq': 0},
               ]}
]

MONSTERS_POSITION_ID = [
    {'id': 0, 'x': 772, 'y': 496, 'row': 0, 'col': 1, 'rule': POSITION_RUEL.SINGLE},
    {'id': 1, 'x': 867, 'y': 410, 'row': 1, 'col': 0, 'rule': POSITION_RUEL.DOUBLE},
    {'id': 2, 'x': 867, 'y': 582, 'row': 1, 'col': 2, 'rule': POSITION_RUEL.DOUBLE},
    {'id': 3, 'x': 962, 'y': 496, 'row': 2, 'col': 1, 'rule': POSITION_RUEL.SINGLE},
    {'id': 4, 'x': 1057, 'y': 410, 'row': 3, 'col': 0, 'rule': POSITION_RUEL.DOUBLE},
    {'id': 5, 'x': 1057, 'y': 582, 'row': 3, 'col': 2, 'rule': POSITION_RUEL.DOUBLE},
    {'id': 6, 'x': 1152, 'y': 496, 'row': 4, 'col': 1, 'rule': POSITION_RUEL.SINGLE}
]
MONSTER_HP_INFO = {'w': 46, 'h': 22}
POSITION_COORDINATES = [{'id': 0, 'idx_x': 0, 'idx_y': 0},
                        {'id': 1, 'idx_x': 1, 'idx_y': 0},
                        {'id': 2, 'idx_x': 1, 'idx_y': 1},
                        {'id': 3, 'idx_x': 2, 'idx_y': 0},
                        {'id': 4, 'idx_x': 3, 'idx_y': 0},
                        {'id': 5, 'idx_x': 3, 'idx_y': 1},
                        {'id': 6, 'idx_x': 4, 'idx_y': 0}]
MONSTER_MISSION_INFO = \
    [{'mission_id': '20', 'mission_number': 3, 'x': 27, 'y': 118, 'w': 290, 'h': 40,
      'sub_mission':
          [{'mission_id': '20-0', 'monsters_number': 3,
            'monster': [
                {'id': 0, 'max_hp': 742, 'property': 'plant', 'hp_x': 715, 'hp_y': 400, 'cur_hp': 0, 'idx_x': 0,
                 'idx_y': 0},
                {'id': 3, 'max_hp': 742, 'property': 'plant', 'hp_x': 907, 'hp_y': 400, 'cur_hp': 0, 'idx_x': 0,
                 'idx_y': 0},
                {'id': 6, 'max_hp': 742, 'property': 'plant', 'hp_x': 1098, 'hp_y': 400, 'cur_hp': 0, 'idx_x': 0,
                 'idx_y': 0}]},
           {'mission_id': '20-1', 'monsters_number': 3,
            'monster': [{'id': 0, 'max_hp': 742, 'property': 'plant', 'hp_x': 713, 'hp_y': 400, 'cur_hp': 0, 'idx_x': 0,
                         'idx_y': 0},
                        {'id': 1, 'max_hp': 742, 'property': 'plant', 'hp_x': 810, 'hp_y': 312, 'cur_hp': 0, 'idx_x': 0,
                         'idx_y': 0},
                        {'id': 2, 'max_hp': 742, 'property': 'plant', 'hp_x': 810, 'hp_y': 485, 'cur_hp': 0, 'idx_x': 0,
                         'idx_y': 0}]},
           {'mission_id': '20-2', 'monsters_number': 3,
            'monster': [
                {'id': 0, 'max_hp': 1523, 'property': 'bigstone', 'hp_x': 713, 'hp_y': 230, 'cur_hp': 0, 'idx_x': 0,
                 'idx_y': 0},
                {'id': 3, 'max_hp': 935, 'property': 'bigstone', 'hp_x': 904, 'hp_y': 230, 'cur_hp': 0, 'idx_x': 0,
                 'idx_y': 0},
                {'id': 6, 'max_hp': 935, 'property': 'bigstone', 'hp_x': 1096, 'hp_y': 227, 'cur_hp': 0, 'idx_x': 0,
                 'idx_y': 0}]}
           ]
      }, {'mission_id': '19', 'mission_number': 3, 'x': 27, 'y': 118, 'w': 290, 'h': 40,
          'sub_mission':
              [{'mission_id': '19-0', 'monsters_number': 3,
                'monster': [
                    {'id': 1, 'max_hp': 864, 'property': 'stone', 'hp_x': 947, 'hp_y': 291, 'cur_hp': 0, 'idx_x': 0,
                     'idx_y': 0},
                    {'id': 3, 'max_hp': 864, 'property': 'plant', 'hp_x': 1061, 'hp_y': 392, 'cur_hp': 0, 'idx_x': 0,
                     'idx_y': 0},
                    {'id': 2, 'max_hp': 924, 'property': 'stone', 'hp_x': 947, 'hp_y': 492, 'cur_hp': 0, 'idx_x': 0,
                     'idx_y': 0}]},
               {'mission_id': '19-1', 'monsters_number': 3,
                'monster': [
                    {'id': 1, 'max_hp': 803, 'property': '燈籠', 'hp_x': 948, 'hp_y': 328, 'cur_hp': 0, 'idx_x': 0,
                     'idx_y': 0},
                    {'id': 0, 'max_hp': 803, 'property': '燈籠', 'hp_x': 837, 'hp_y': 428, 'cur_hp': 0, 'idx_x': 0,
                     'idx_y': 0},
                    {'id': 2, 'max_hp': 803, 'property': '燈籠', 'hp_x': 948, 'hp_y': 528, 'cur_hp': 0, 'idx_x': 0,
                     'idx_y': 0}]},
               {'mission_id': '19-2', 'monsters_number': 3,
                'monster': [
                    {'id': 1, 'max_hp': 1025, 'property': '燈籠', 'hp_x': 810, 'hp_y': 284, 'cur_hp': 0, 'idx_x': 0,
                     'idx_y': 0},
                    {'id': 0, 'max_hp': 924, 'property': 'plant', 'hp_x': 714, 'hp_y': 336, 'cur_hp': 0, 'idx_x': 0,
                     'idx_y': 0},
                    {'id': 2, 'max_hp': 864, 'property': 'stone', 'hp_x': 810, 'hp_y': 422, 'cur_hp': 0, 'idx_x': 0,
                     'idx_y': 0}]}
               ]
          }, {'mission_id': '1', 'mission_number': 1, 'x': 27, 'y': 118, 'w': 290, 'h': 40,
              'sub_mission':
                  [{'mission_id': '1-0', 'monsters_number': 3,
                    'monster': [
                        {'id': 1, 'max_hp': 375, 'property': 'stone', 'hp_x': 810, 'hp_y': 316, 'cur_hp': 0, 'idx_x': 0,
                         'idx_y': 0},
                        {'id': 0, 'max_hp': 375, 'property': 'plant', 'hp_x': 714, 'hp_y': 400, 'cur_hp': 0, 'idx_x': 0,
                         'idx_y': 0},
                        {'id': 2, 'max_hp': 375, 'property': 'stone', 'hp_x': 810, 'hp_y': 486, 'cur_hp': 0, 'idx_x': 0,
                         'idx_y': 0}]}]}

     ]


def is_used(file_name):
    try:
        if not os.path.exists(file_name):
            return False

        vHandle = win32file.CreateFile(file_name, win32file.GENERIC_READ, 0, None, win32file.OPEN_EXISTING,
                                       win32file.FILE_ATTRIBUTE_NORMAL, None)
        result = bool(int(vHandle) == win32file.INVALID_HANDLE_VALUE)
        win32file.CloseHandle(vHandle)
    except Exception:
        #檔案不存在也會跳來這
        return True
    return result


class dictmenu(object):
    def __init__(self, dicts):
        self.dicts = dicts
        self.len = len(dicts)

    def value_index(self, key=None, value=None):
        # print('dicts data:', self.dicts)
        # self.dicts.keys()
        for idx in range(self.len):
            dict = self.dicts[idx]
            if type(dict).__name__ == 'dict':
                if key and value:
                    v = dict[key]
                    if v == value:
                        return idx, dict
                elif key:
                    v = dict[key]
                    return idx, dict
                else:
                    array_dict = dict.values()
                    for i in range(len(array_dict)):
                        if value in array_dict:
                            return idx, dict
                # if value in array_dict:
                #     return array_dict.index(value)

            else:
                print(dict)
                return None, None

        # if value in self.dicts.value():
        #
        #     pass
        return None, None

    def multi_value_index(self, key=None, value=None):
        answer_idx = []
        answer = []
        for idx in range(self.len):
            dict = self.dicts[idx]
            if type(dict).__name__ == 'dict':
                if key:
                    v = dict[key]
                    if v == value:
                        answer_idx.append(idx)
                        answer.append(dict)
                        # return idx, dict
                else:
                    array_dict = dict.values()
                    for i in range(len(array_dict)):
                        if value in array_dict:
                            answer_idx.append(idx)
                            answer.append(dict)
                            # return idx, dict
            else:
                print(dict)
                return None, None
        return answer_idx, answer


def detect_number(image):
    rawImage = cv2.imread(image)
    hight, width, deep = rawImage.shape
    gray = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
    dst = np.zeros((hight, width, 1), np.uint8)
    for i in range(0, hight):
        for j in range(0, width):
            grayPixel = gray[i, j]
            dst[i, j] = 255 - grayPixel

    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    string = pytesseract.image_to_string(binary, lang='eng', config='--psm 6 --oem 3 -c '
                                                                    'tessedit_char_whitelist'
                                                                    '=0123456789')
    isdigit = any(chr.isdigit() for chr in string)
    if isdigit:
        # print('hp-ocr:', string)
        return string
    else:
        # print('hp-orc can nott detect number: return None:', string)
        return None


def phash(path):
    img = cv2.imread(path)
    img1 = cv2.resize(img, (32, 32), cv2.COLOR_RGB2GRAY)

    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)

    vis0[:h, :w] = img1

    img_dct = cv2.dct(cv2.dct(vis0))
    img_dct.resize(8, 8)
    img_list = np.array().flatten(img_dct.tolist())

    img_mean = cv2.mean(img_list)
    avg_list = ['0' if i < img_mean else '1' for i in img_list]
    return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 64, 4)])


# Press the green button in the gutter to run the script.
def classify_gray_hist(image1, image2, size=(256, 256)):
    # 先計算直方圖
    # 幾個參數必須用方括號括起來
    # 這裡直接用灰度圖計算直方圖，所以是使用第一個通道，
    # 也可以進行通道分離後，得到多個通道的直方圖
    # bins 取為16
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 可以比較下直方圖
    # plt.plot(range(256), hist1, 'r')
    # plt.plot(range(256), hist2, 'b')
    # plt.show()
    # 計算直方圖的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
    else:
        degree = degree + 1
    degree = degree / len(hist1)
    return degree

    # 計算單通道的直方圖的相似值


# 對圖片進行統一化處理
def get_thum(image, size=(64, 64), greyscale=False):
    # 利用image對影象大小重新設定, Image.ANTIALIAS為高質量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 將圖片轉換為L模式，其為灰度圖，其每個畫素用8個bit表示
        image = image.convert('L')
    return image


# 計算圖片的餘弦距離
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（線性）+algebra（代數），norm則表示範數
        # 求圖片的範數？？
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是點積，對二維陣列（矩陣）進行計算
    res = dot(a / a_norm, b / b_norm)
    return res


def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 計算直方圖的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
    else:
        degree = degree + 1
    degree = degree / len(hist1)
    return degree


def detect_card():
    card_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread('2.JPG')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # faces = card_cascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.08,
    #     minNeighbors=5,
    #     minSize=(32, 32))
    #
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    # cv2.imwrite('result.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def snapshot(handle, filepath=GalleryLibrary.Full_Image.value, sleft=0, stop=0,
             sright=0, sbot=0):
    b = os.path.dirname(filepath)
    os.makedirs(b, exist_ok=True)

    setForegroundWindows(handle)
    image = ImageGrab.grab(bbox=(sleft, stop, sright, sbot))

    while True:
        isused= is_used(filepath)
        if isused:
            continue
        else:
            image.save(filepath, optimize=1)
            break


def cmpHash(hash1, hash2):
    n = 0
    # hash長度不同反回-1，此時不能比較
    if len(hash1) != len(hash2):
        return -1
    # 如果hash長度相同遍歷長度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def search_image_position(handle, file1, file2, show, thresh_hold, click=True, shot=True, coordinate='absolute'):
    img = cv2.imread(file1)
    img_w = img.shape[1]
    img_h = img.shape[0]
    img2 = img.copy()
    template = cv2.imread(file2)

    w = template.shape[1]
    h = template.shape[0]
    # print(w, h)
    # All the 6 methods for comparison in a list
    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    methods = ['cv2.TM_CCOEFF']
    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, 255, 2)

        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        # print(top_left, bottom_right)
        if coordinate == 'absolute':
            c_left = top_left[0] + AXIE_WINDOW_PROPERTY['left']
            c_top = top_left[1] + AXIE_WINDOW_PROPERTY['top']
            c_right = bottom_right[0] + AXIE_WINDOW_PROPERTY['left']
            c_bottom = bottom_right[1] + AXIE_WINDOW_PROPERTY['top']
        else:
            c_left = top_left[0]
            c_top = top_left[1]
            c_right = bottom_right[0]
            c_bottom = bottom_right[1]
        shutil.copy2(file2, GalleryLibrary.Temp_Origin.value)
        temp_card = GalleryLibrary.Temp_Image.value
        if shot and coordinate == 'absolute':
            snapshot(axie_handle, temp_card, c_left, c_top, c_right, c_bottom)
        else:
            cropped = img[c_top:c_bottom, c_left:c_right]
            cv2.imwrite(temp_card, cropped)
        # print(c_left, c_top, c_right, c_bottom)
        c_x = int(round((top_left[0] + bottom_right[0]) / 2, 0))
        c_y = int(round((top_left[1] + bottom_right[1]) / 2, 0))
        # print("image position:", c_x, c_y)

        # img1 = cv2.imread(file2)  # 圖庫裡的
        # img2 = cv2.imread(temp_card)  # 即時抓取的圖
        # degree = classify_gray_hist(img1, img2)
        img1 = Image.open(file2)
        img2 = Image.open(temp_card)
        # degree = image_similarity_vectors_via_numpy(img1, img2)
        hash1 = imagehash.average_hash(img1)
        hash2 = imagehash.average_hash(img2)
        degree1 = hash1 - hash2
        hash1 = imagehash.dhash(img1)
        hash2 = imagehash.dhash(img2)
        degree2 = hash1 - hash2
        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)
        degree3 = hash1 - hash2
        degree = degree1 + degree2 + degree3
        # print('degree:', degree1, degree2, degree3, degree)
        if (degree <= thresh_hold) and click:
            pg.moveTo(c_x + AXIE_WINDOW_PROPERTY['left'], c_y + AXIE_WINDOW_PROPERTY['top'], duration=0.2)
            pg.click(c_x + AXIE_WINDOW_PROPERTY['left'], c_y + AXIE_WINDOW_PROPERTY['top'], duration=0.2)

        if show:
            plt.show()
        if coordinate == 'absolute':
            x = c_x + AXIE_WINDOW_PROPERTY['left']
            y = c_y + AXIE_WINDOW_PROPERTY['top']
        else:
            x = c_x
            y = c_y
        return degree, x, y


def compare_hash(file1, file2):
    img1 = Image.open(file1)
    img2 = Image.open(file2)
    # degree = image_similarity_vectors_via_numpy(img1, img2)
    hash1 = imagehash.average_hash(img1)
    hash2 = imagehash.average_hash(img2)
    degree1 = hash1 - hash2
    hash1 = imagehash.dhash(img1)
    hash2 = imagehash.dhash(img2)
    degree2 = hash1 - hash2
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)
    degree3 = hash1 - hash2
    degree = degree1 + degree2 + degree3
    # print(degree1, degree2, degree3, degree)
    return degree


def config_axie_window():
    classname = "UnityWndClass"
    titlename = "Axie Infinity"
    handle = win32gui.FindWindow(classname, titlename)
    if handle != 0:
        global AXIE_WINDOW_PROPERTY
        left, top, right, bottom = win32gui.GetWindowRect(handle)
        width, height = (right - left), (bottom - top)
        AXIE_WINDOW_PROPERTY['left'] = left
        AXIE_WINDOW_PROPERTY['top'] = top
        AXIE_WINDOW_PROPERTY['right'] = right
        AXIE_WINDOW_PROPERTY['bottom'] = bottom
        AXIE_WINDOW_PROPERTY['width'] = width
        AXIE_WINDOW_PROPERTY['height'] = height
        setForegroundWindows(handle)
        print(AXIE_WINDOW_PROPERTY)
        return handle


# 將視窗移到最上層，並且拍照
def setForegroundWindows(handle):
    text = win32gui.SetForegroundWindow(handle)


def randon_cards(cards, size=10):
    idxs = np.random.randint(0, len(cards), size=size)
    a = [cards[i] for i in idxs]
    return a


def test_all_card(handle):
    click_end_turn_button = False
    ramdon_cards = sample(CARDS_LIBRARY_LIST, 12)
    # ramdon_cards = randon_cards(CARDS_LIBRARY_LIST, 10)
    print(ramdon_cards)

    for f in CARDS_LIBRARY_LIST:
        snapshot(handle, sleft=AXIE_WINDOW_PROPERTY['left'], stop=AXIE_WINDOW_PROPERTY['top'],
                 sright=AXIE_WINDOW_PROPERTY['right'], sbot=AXIE_WINDOW_PROPERTY['bottom'])
        fullpath = join(GalleryLibrary.Cards_Library.value, f)
        print(fullpath)
        search_image_position(handle, GalleryLibrary.Full_Image.value, fullpath, show=False, click=False,
                              thresh_hold=0.65)
        time.sleep(1)


def black_cards(path, x, y):
    # print(x, y)
    # pic = '.\\picture\\temp_hp_number_' + id + '.JPG'
    img = cv2.imread(path)
    # print('img shape:', img.shape)

    left = x - 40
    top = y - 50
    right = x + 40
    bottom = y + 50
    # pg.moveTo(left, top)
    # time.sleep(3)
    # pg.moveTo(right, bottom)
    # time.sleep(3)
    # print(left, top, right, bottom)
    cropped = img[top:bottom, left:right]
    for x in range(left, right):
        for y in range(top, bottom):
            a = x - AXIE_WINDOW_PROPERTY['left']
            b = y - AXIE_WINDOW_PROPERTY['top']
            img[b, a] = (0, 0, 0)
    cv2.imwrite(path, img)
    # cv2.imshow('a',img)
    # plt.show()
    # hsv=cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
    # h, w, d = cropped.shape
    # # print(w, h)
    # cropped = cv2.resize(cropped, (w * 5, h * 5))
    # cv2.imwrite(path, cropped)


def create_empty_array(x):
    empty_array = []
    for i in range(0, x):
        empty_array.append([])  # 令其轉為1維串列形式
    return empty_array


def detect_cards():
    cards_library = []
    cards_count = 0
    snapshot(axie_handle, sleft=AXIE_WINDOW_PROPERTY['left'], stop=AXIE_WINDOW_PROPERTY['top'],
             sright=AXIE_WINDOW_PROPERTY['right'], sbot=AXIE_WINDOW_PROPERTY['bottom'])
    count = 0
    while count < 3:
        for f in CARDS_LIBRARY_LIST:
            # prefix = f[:1]
            # # 代表該角色play cards的卡權滿了，就不再挑他的卡片出來了
            # if prefix not in cae_be_selected_card_prefix:
            #     continue
            fullpath = join(GalleryLibrary.Cards_Library.value, f)
            # print(fullpath)
            # search cards
            threashold = 25
            degree, x, y = search_image_position(axie_handle, GalleryLibrary.Full_Image.value, fullpath, False,
                                                 threashold, click=False)
            # if '1-1.JPG' in f:
            # print(f,":",degree)
            if degree <= threashold:
                # f_prefix_idx=int(f[:1])-1
                cards_library.append(f)
                black_cards(GalleryLibrary.Full_Image.value, x, y)
                # cards_count = cards_count+1
        count = count + 1
    # print(degree, x, y)
    print(cards_library, len(cards_library))
    return len(cards_library), cards_library


def bounds_check(center_x, center_y):
    bounds_degree, x, y = search_image_position(axie_handle, GalleryLibrary.Full_Image.value,
                                                GalleryLibrary.Button_SLP_Bounds.value, show=False, thresh_hold=12,
                                                click=False)

    if bounds_degree <= 12:
        print('[STATUS-WIN] SLP Bounds(拿SLP畫面)，點擊離開此畫面 degree ', bounds_degree)
        pg.click(center_x, center_y, duration=0.1)
        time.sleep(10)
        return True
    return False


def start_mission_check():
    degree_start_mission, x, y = search_image_position(axie_handle, GalleryLibrary.Full_Image.value,
                                                       GalleryLibrary.Button_Start_Mission_Button.value, show=False,
                                                       thresh_hold=3, click=True)

    if degree_start_mission <= 4:
        print('[STATUS- new game] start_mission:', degree_start_mission)
        time.sleep(5)
        return True
    return False


def fight_result_check(center_x, center_y):
    degree_win, x, y = search_image_position(axie_handle, GalleryLibrary.Full_Image.value,
                                             GalleryLibrary.Button_Victory_Icon.value, show=False,
                                             thresh_hold=3, click=False)
    degree_lost, x, y = search_image_position(axie_handle, GalleryLibrary.Full_Image.value,
                                              GalleryLibrary.Button_Defeated_Icon.value, show=False,
                                              thresh_hold=3, click=False)
    # print('degree  win/lost:', degree_win, degree_lost)
    #
    if degree_win <= 3 or degree_lost <= 3:
        corner_x = AXIE_WINDOW_PROPERTY['right'] - 20
        corner_y = AXIE_WINDOW_PROPERTY['bottom'] - 20
        print('[STATUS-WIN], 點擊，離開關卡勝利畫面')
        pg.moveTo(corner_x, corner_y, duration=0.1)
        # 點擊，離開 關卡說你勝利的畫面
        pg.click(corner_x, corner_y, duration=0.1)
        time.sleep(5)
        print('[STATUS-WIN], 點擊，離開 經驗值獎勵畫面, 若沒體力則不會有這畫面')
        # 點擊，離開 恭喜勝利畫面(包含經驗值畫面)
        pg.click(center_x, center_y, duration=0.1)
        time.sleep(5)
        print('[STATUS-WIN], 點擊，離開 SLP獎勵畫面, 若沒體力則不會有這畫面')
        # 點擊，離開 恭喜勝利畫面(包含經驗值畫面)
        pg.click(center_x, center_y, duration=0.1)
        time.sleep(5)

        if degree_win <= 3:
            return True, degree_win
        else:
            return False, degree_lost

    return None, None


def snapshot_axies_runtime():
    for formation in AXIES_FORMATION:
        axie_id = formation['id']
        axie_position_id = formation['position_id']
        axie_position = AXIES_POSITION_ID[axie_position_id]
        axie_center_x = axie_position['x'] + AXIE_WINDOW_PROPERTY['left']
        axie_center_y = axie_position['y'] + AXIE_WINDOW_PROPERTY['top']
        # print(AXIE_WINDOW_PROPERTY['left'], AXIE_WINDOW_PROPERTY['top'], axie_center_x, axie_center_y)
        #
        # pg.moveTo(axie_center_x, axie_center_y, duration=0.3)
        # time.sleep(2)
        # #
        # r取得runtime axies的完整圖像
        for shape in AXIE_RUNTIME_SNAPSHOP_LOCATION:
            if axie_id in shape['id']:
                left = axie_center_x + shape['left']
                top = axie_center_y + shape['top']
                right = axie_center_x + shape['right']
                bottom = axie_center_y + shape['bottom']
        path = GalleryLibrary.AXIE_RUNTIME_FULL.value + '\\' + axie_id + '.JPG'
        snapshot(axie_handle, filepath=path, sleft=left, stop=top,
                 sright=right, sbot=bottom)

        # 取得axie頭上的debuff圖像
        for shape in AXIE_RUNTIME_DEBUFF_SNAPSHOP_LOCATION:
            if axie_id in shape['id']:
                left = axie_center_x + shape['left']
                top = axie_center_y + shape['top']
                right = axie_center_x + shape['right']
                bottom = axie_center_y + shape['bottom']
        path = GalleryLibrary.AXIE_RUNTIME_DEBUFF.value + '\\' + axie_id + '.JPG'
        snapshot(axie_handle, filepath=path, sleft=left, stop=top,
                 sright=right, sbot=bottom)


def detect_axie_hp(status):
    # find all axies's position
    # 移除沒有hp值的axie
    # for f in status:
    #     id = f['id']
    #     status = f['status']
    #     status_id = f['status_id']
    #     for h in axie_hp_status:
    #         if id in h['id']:
    #             h['status'] = status
    #             if status_id != 0:
    #                 axie_hp_status.remove(h)
    # print('which axies detect hp:', axie_hp)
    MAX = 7  # 找血量 抓5次的狀況
    retry = 3  #
    count = 0
    # len(axie_hp_status)
    for axie_stat in status:
        axie_stat.update({'cur_hp': 0})
        axie_stat.update({'debuff': AXIE_DEBUFF.Nothing})
        # 移除沒有hp值的axie, 0是normal，不是0代表沒有血量條了
        status_id = axie_stat['status_id']
        if status_id != 0:
            continue

        id = axie_stat['id']
        status = axie_stat['status']
        debuff = []
        debuff_threshold = 50
        # while retry > 0:
        #     count = 0
        hp = []
        while count < MAX:
            # snapshot_axies_runtime()
            # 檢查debuff
            axie_debuff_path = GalleryLibrary.AXIE_RUNTIME_DEBUFF.value + '\\' + id + '.JPG'
            degree, x, y = search_image_position(axie_handle, axie_debuff_path,
                                                 GalleryLibrary.Button_Debuff_1_Icon.value, show=False,
                                                 thresh_hold=debuff_threshold, click=False, coordinate='relative')
            print('debuff degree,x,y:', degree, x, y)
            debuff.append(degree)
            # 檢查hp
            normal_axie = GalleryLibrary.AXIE_RUNTIME_FULL.value + '\\' + id + '.JPG'
            # axie_pic = '.\\picture\\temp_axie.JPG'
            for f in listdir(GalleryLibrary.AXIE_HP_Icon.value):
                file = join(GalleryLibrary.AXIE_HP_Icon.value, f)
                print(file)
                degree, x, y = search_image_position(axie_handle, normal_axie, file,
                                                     show=False, thresh_hold=80, click=False, coordinate='relative')
                print('hp degree,x,y:', degree, x, y)

            pic = '.\\picture\\temp_hp_number_' + id + '.JPG'
            img = cv2.imread(normal_axie)
            # print('img shape:', img.shape)
            left = x + 13
            top = y - 10
            right = x + 65
            bottom = y + 10
            cropped = img[top:bottom, left:right]
            h, w, d = cropped.shape
            # print(w, h)
            cropped = cv2.resize(cropped, (w * 10, h * 10))
            cv2.imwrite(pic, cropped)

            # 側測數字
            temp_hp_number = detect_number(pic)
            hp_number = ''
            if temp_hp_number:
                for chr in temp_hp_number:
                    # print(chr, "is digit?", chr.isdigit())
                    if chr.isdigit():
                        hp_number = hp_number + chr
                    else:
                        break
                # print('type hp:',type(hp))
                # print('hp :',hp)
                hp.append(hp_number)

            print('hp is ', hp_number)

            # print(hp_number.rstrip('\r\n'))
            # 這段容易有處理影像的bug，先high起來
            # pic = '.\\picture\\temp_hp_len_' + id + '.JPG'
            # left = x + 55
            # top = y - 10
            # right = x + 130
            # bottom = y + 10
            # cropped = img[top:bottom, left:right]
            # cv2.imwrite(pic, cropped)
            count = count + 1
            if len(hp_number) >= 3:
                break
        if hp and len(hp):
            # print()
            axie_stat.update({'cur_hp': int(max(hp))})

        if len(debuff):
            degree = int(sum(debuff) / len(debuff))
        else:
            degree = None
        # print("final debuff degree:", degree)
        if degree and degree <= debuff_threshold:
            # f['debuff'].append(AXIE_DEBUFF.DISABLE_ONE_CARD)
            axie_stat.update({'debuff': AXIE_DEBUFF.DISABLE_ONE_CARD})

    # print(hp)

    # print(axie_hp_status)
    # return axie_hp_status
    # return status


def snapshot_axies_pvp_runtime():
    # AXIE_PVP_SNAPSHOP = {'left': -90, 'top': -160, 'right': 75, 'bottom': -20}
    # AXIE_PVP_POSITION_ID = [
    #     {'id': 0, 'x': 772, 'y': 496, 'row': 0, 'col': 1, 'rule': POSITION_RUEL.SINGLE},
    axie_pvp_hp_status = []
    for position in AXIE_PVP_POSITION_ID:
        axie_position_id = position['id']
        axie_center_x = position['x'] + AXIE_WINDOW_PROPERTY['left']
        axie_center_y = position['y'] + AXIE_WINDOW_PROPERTY['top']
        left = axie_center_x + AXIE_PVP_SNAPSHOP['left']
        top = axie_center_y + AXIE_PVP_SNAPSHOP['top']
        right = axie_center_x + AXIE_PVP_SNAPSHOP['right']
        bottom = axie_center_y + AXIE_PVP_SNAPSHOP['bottom']
        # print(AXIE_WINDOW_PROPERTY['left'], AXIE_WINDOW_PROPERTY['top'], axie_center_x, axie_center_y)

        # pg.moveTo(left, top, duration=0.3)
        # #
        # time.sleep(2)
        path = '.\\picture\\pvp_axies\\runtime\\' + str(axie_position_id) + '.JPG'
        snapshot(axie_handle, filepath=path, sleft=left, stop=top,
                 sright=right, sbot=bottom)


def detect_axie_pvp_hp():
    # axie_pvp_hp_status = [
    #     {'position_id': 0, 'hp': 0, 'cur_hp': 0, 'status': '', 'debuff': []},
    #     {'position_id': 0, 'hp': 0, 'cur_hp': 0, 'status': '', 'debuff': []},
    #     {'position_id': 0, 'hp': 0, 'cur_hp': 0, 'status': '', 'debuff': []},
    # ]
    axie_pvp_hp_status = []
    # print('which axies detect hp:', axie_hp)
    MAX = 3  # 找血量 抓5次的狀況
    retry = 3  #
    for position in AXIE_PVP_POSITION_ID:
        hp = []
        # id = f['id']
        # status = f['status']
        # debuff = []
        # debuff_threshold = 50
        while retry > 0:
            count = 0
            while count < MAX:
                count = count + 1
                # snapshot pvp axies
                axie_position_id = position['id']
                axie_center_x = position['x']
                axie_center_y = position['y']
                left = axie_center_x + AXIE_PVP_SNAPSHOP['left'] + AXIE_WINDOW_PROPERTY['left']
                top = axie_center_y + AXIE_PVP_SNAPSHOP['top'] + AXIE_WINDOW_PROPERTY['top']
                right = axie_center_x + AXIE_PVP_SNAPSHOP['right'] + AXIE_WINDOW_PROPERTY['left']
                bottom = axie_center_y + AXIE_PVP_SNAPSHOP['bottom'] + AXIE_WINDOW_PROPERTY['top']
                path = '.\\picture\\pvp_axies\\runtime\\' + str(axie_position_id) + '.JPG'
                snapshot(axie_handle, filepath=path, sleft=left, stop=top, sright=right, sbot=bottom)
                # 檢查hp
                normal_axie = '.\\picture\\pvp_axies\\runtime\\' + str(axie_position_id) + '.JPG'
                ff = '.\\picture\\pvp_axies\\icons'
                find_pvp_axie = False
                for f in listdir(ff):
                    fullpath = join(ff, f)
                    # axie_pic = '.\\picture\\temp_axie.JPG'
                    degree, x, y = search_image_position(axie_handle, normal_axie, fullpath, show=False,
                                                         thresh_hold=50, click=False, coordinate='relative')
                    print(axie_position_id, '  pvp hp degree,x,y:', degree, x, y)
                    if degree <= 50:
                        find_pvp_axie = True
                        break
                if not find_pvp_axie:
                    break
                # 抓HP的位置
                pic = '.\\picture\\temp_hp_number_' + str(axie_position_id) + '.JPG'
                img = cv2.imread(path)
                # print('img shape:', img.shape)
                left = x + 13
                top = y - 10
                right = x + 55
                bottom = y + 10
                cropped = img[top:bottom, left:right]
                h, w, d = cropped.shape
                # print(w, h)
                cropped = cv2.resize(cropped, (w * 5, h * 5))
                cv2.imwrite(pic, cropped)

                # 側測數字
                temp_hp_number = detect_number(pic)
                hp_number = ''
                if temp_hp_number:
                    for chr in temp_hp_number:
                        # print(chr, "is digit?", chr.isdigit())
                        if chr.isdigit():
                            hp_number = hp_number + chr
                        else:
                            break
                    hp.append(hp_number)

                print('hp is ', hp_number)
            find_hp = False
            # 這段為何需要重新scan axie_hp???不是最外層已經挑出 現在正在判斷的axie了嗎
            # axie_pvp_hp_status = [
            #     {'position_id': 0, 'hp': 0, 'cur_hp': 0, 'status': '', 'debuff': []},
            print('hp:', hp)
            if hp and len(hp):
                # h['cur_hp'] = int(max(hp))
                pvp_axie = dict(position_id=axie_position_id,
                                hp=int(max(hp)))
                axie_pvp_hp_status.append(pvp_axie)
                find_hp = True
            else:
                hp = 0

            if not find_pvp_axie:
                break

            retry = retry - 1
            if find_hp:
                break
    print(axie_pvp_hp_status)
    # # print('debuff:', debuff)
    # if len(debuff):
    #     degree = int(sum(debuff) / len(debuff))
    # else:
    #     degree = 0
    # # print("final debuff degree:", degree)
    # if degree <= debuff_threshold:
    #     f['debuff'].append(AXIE_DEBUFF.DISABLE_ONE_CARD)


def get_axie_waise_one_card_debuff(axie, axie_hp):
    # 取得attacker的debuff狀態
    waise_one_card = False
    a = dictmenu(axie_hp)
    axie_attacker_hpstatis_idx, axie_attacker_hpstatis = a.value_index('id', axie)
    if not axie_attacker_hpstatis:
        axie_attacker_debuff = axie_attacker_hpstatis['debuff']
        if len(axie_attacker_debuff) and AXIE_DEBUFF.DISABLE_ONE_CARD == axie_attacker_debuff[0]:
            waise_one_card = True
    return waise_one_card


def stratgy(runtime_cards_library, play_cards_array, axie_status_info, MISSION_ID):
    # 把卡片根據AXIE分類

    runtime_card_list = create_empty_array(3)
    for card in runtime_cards_library:
        card_id = card[:7]
        card_idx = int(AXIES_CARDS_PIC_PREFIX.get(card_id))
        runtime_card_list[card_idx].append(card)
    print('AXIE 可出的卡片(分類):', runtime_card_list)

    runtime_cards_library_copy = runtime_cards_library.copy()

    new_card_list = []
    sub_monsters_for_showing = []
    # 先補血 取出所有補血卡
    # 從畫面上的所有卡片中，只取出現在指定axie的卡，然後卡進axie_runtime_card裡，讓他做等一下的決策用
    # axie_hp_status = [
    #     {'id': "1986033", 'nickname': 'fish', 'hp': 0, 'cur_hp': 0, 'status': '', 'debuff': []},
    #     {'id': "1980381", 'nickname': 'do', 'hp': 0, 'cur_hp': 0, 'status': '', 'debuff': []},
    #     {'id': "1985659", 'nickname': 'berry', 'hp': 0, 'cur_hp': 0, 'status': '', 'debuff': []},
    # ]
    for axie in axie_status_info:
        # 檢查debuff，決定要犧牲的卡片
        # 如果有bug，判定axie是normal，但其實是死了，那可能抓不到血量，則跳過
        id = axie['id']
        print("正在決策", id, ":", AXIES_NICKNAME.get(id), '的補血狀況')
        cur_hp = axie['cur_hp']
        if cur_hp <= 0:
            print(AXIES_NICKNAME.get(id), ' do not have HP!????')
            continue

    #     # 檢查補血者，是否有補血卡，如果沒有，就跳過
    #     a = dictmenu(AXIE_HEAL_RELATION)
    #     axie_heal_relation_idx, axie_heal_relation = a.value_index('id', id)
    #     # 誰是補血者
    #     axie_healter_id = axie_heal_relation['healby']
    #     # 取出他那些卡牌是補血卡
    #     healter_heal_cards = axie_heal_relation['healcard']
    #
    #     axie_healter_card_idx = AXIES_CARDS_PIC_PREFIX.get(axie_healter_id)
    #
    #     # 把runtime這位補血者所有的補血卡都取出來
    #     if len(play_cards_array[int(axie_healter_card_idx) - 1]) >= 4:
    #         continue
    #
    #     axie_runtime_heal_cards = create_empty_array(3)
    #     for card in runtime_cards_library_copy:
    #         if card in healter_heal_cards:
    #             axie_runtime_heal_cards[int(axie_healter_card_idx) - 1].append(card)
    #     # 如果沒有補血卡 就跳過
    #     if len(axie_runtime_heal_cards) <= 0:
    #         continue
    #
    #     # 取出要被補血者的資料
    #     axie_healter_runtime_cards = axie_runtime_heal_cards[int(axie_healter_card_idx) - 1]
    #     a = dictmenu(MY_AXIES)
    #     axie_info_idx, axie_info = a.value_index('id', id)
    #     axie_healter_info_idx, axie_healter_info = a.value_index('id', axie_healter_id)
    #     max_hp = axie_info['health']
    #     dif_hp = max_hp - cur_hp
    #     print("axie損失的血量", dif_hp)
    #     healter_card_value = []
    #     healter_card_name = []
    #     axie_healter_card_info = axie_healter_info['cards']
    #     a = dictmenu(axie_healter_card_info)
    #     for cardname in axie_healter_runtime_cards:
    #         healcard_idx, healcard_info = a.value_index('name', cardname)
    #         healcard_value = healcard_info['heal']
    #         healter_card_value.append(healcard_value)
    #         healter_card_name.append(cardname)
    #     selected_heal_card, selected_heal_card_name = takeHealClosest(healter_card_value, healter_card_name,
    #                                                                   dif_hp)
    #     # 取得補血者的debuff狀態
    #     waise_one_card = get_axie_waise_one_card_debuff(axie_healter_id, axie_status_info)
    #
    #     print(selected_heal_card_name, selected_heal_card)
    #     if selected_heal_card_name:
    #         if waise_one_card:
    #             # 檢查healter擁有卡的 棄卡順位，先把他打出去
    #             card_grade = 999
    #             useless_card_name = ''
    #             a = dictmenu(axie_healter_card_info)
    #
    #             for card in runtime_card_list[int(axie_healter_card_idx) - 1]:
    #                 healter_card_idx, healter_card = a.value_index('name', card)
    #                 debuff_seq = healter_card['debuff_seq']
    #                 if debuff_seq < card_grade:
    #                     card_grade = debuff_seq
    #                     useless_card_name = card
    #
    #             print("要西生的卡:", useless_card_name)
    #             play_cards_array[int(axie_healter_card_idx) - 1].append(useless_card_name)
    #             new_card_list.append(useless_card_name)
    #             runtime_cards_library_copy.remove(useless_card_name)
    #             # TODO: 需要補血，卻需要先西生一張卡
    #         for card in selected_heal_card_name:
    #             play_cards_array[int(axie_healter_card_idx) - 1].append(card)
    #             new_card_list.append(card)
    #             runtime_cards_library_copy.remove(card)
    #         # 取得補血者的卡牌資料
    #     # print("剩下可出的卡片有:", sorted(runtime_cards_library_copy))
    # print("最後決定的補血卡有:", new_card_list)

    print("============== 決定攻擊的策略 =======")
    #
    #     # 取得下一回axies的攻擊順序
    # 只針對有normal狀況的axies 取出順序
    axies_sequence = get_axies_sequence(axie_status_info)

    # 取出關卡跟怪物資料 hp
    # print("MISSION_ID:",MISSION_ID)
    submission_id, sub_monsters = get_monsters(MISSION_ID)
    print("戰鬥關卡:", submission_id)

    sub_monsters = detect_monster_hp(submission_id, sub_monsters)
    print_list("關卡怪物:", sub_monsters)

    for m in sub_monsters:
        sub_monsters_for_showing.append(m)
    # sub_monsters_for_showing= sub_monsters.copy()
    # 取得怪物的位置和順序，等下用來跟axie互相交集，判斷那隻axie會打那隻怪
    monster_sequence = collect_the_monster_sequence(sub_monsters)
    print_list("怪物的順序:", monster_sequence)
    # 取得每一隻角色的卡片
    # print('a:', axies_sequence)

    #     sequ_axie是該axie的ID
    # 這一段的用途是什麼?
    # axies_runtime_cards = create_empty_array(3)
    # for card in runtime_cards_library_copy:
    #     axies_runtime_cards[int(card[:1]) - 1].append(card)
    # 如果沒有卡片 就跳過

    for sequ_axie in axies_sequence:
        # 從畫面上的所有卡片中，只取出現在指定axie的卡，然後卡進axie_runtime_card裡，讓他做等一下的決策用
        axie_card_idx = AXIES_CARDS_PIC_PREFIX.get(sequ_axie)
        axie_runtime_card = runtime_card_list[int(axie_card_idx)]
        if len(axie_runtime_card) <= 0:
            continue

        # 決定axie要打那隻monster
        axie_formation = dictmenu(AXIES_FORMATION)
        idx, myaxie = axie_formation.value_index('id', sequ_axie)
        the_pk_monster = detect_the_next_pk_monster(myaxie, sub_monsters)
        if len(the_pk_monster) <= 0:
            continue

        a = dictmenu(AXIE_RUNTIME_SNAPSHOP_LOCATION)
        idx, axie_nickname_info = a.value_index('id', sequ_axie)
        print("----------開始分析卡片:----axie-----", axie_nickname_info['nickname'], myaxie)
        # print("---:", axie_nickname_info['nickname'], myaxie)
        print("---axie card:", axie_runtime_card)
        print("---pk monster:", the_pk_monster)
        print("--已經丟出去要打的卡:", play_cards_array)

        axie_detel_info = dictmenu(MY_AXIES)
        idx, axie_attack_info = axie_detel_info.value_index('id', sequ_axie)
        axie_attacker_cards = dictmenu(axie_attack_info['cards'])
        waise_one_card = get_axie_waise_one_card_debuff(sequ_axie, axie_status_info)
        # print('是否要棄卡:',waise_one_card)
        # 檢查attacker擁有卡的 棄卡順位，先把他打出去
        if len(axie_runtime_card) >= 0 and waise_one_card:
            card_grade = 999
            useless_card_name = ''
            a = dictmenu(axie_attack_info['cards'])
            for card in runtime_card_list[int(axie_card_idx) - 1]:
                attacker_card_idx, attacker_card = a.value_index('name', card)
                debuff_seq = attacker_card['debuff_seq']
                if debuff_seq < card_grade:
                    card_grade = debuff_seq
                    useless_card_name = card

            print("Attacker 要西生的卡:", useless_card_name)
            play_cards_array[int(axie_card_idx) - 1].append(useless_card_name)
            new_card_list.append(useless_card_name)
            # print("還有的卡牌:", runtime_cards_library_copy, " 要打出去西生的卡:", useless_card_name)
            runtime_cards_library_copy.remove(useless_card_name)
            axie_runtime_card.remove(useless_card_name)

        card_len = len(axie_runtime_card)
        card_pull = []
        card_attack = []
        card_attack_idx_name = []
        print("test:", axie_runtime_card)
        for card in axie_runtime_card:
            card_idx, card_info = axie_attacker_cards.value_index('name', card)
            card_pull.append(card_info)
            if 'attack' in card_info['property']:
                card_attack.append(card_info['attack'])
                card_attack_idx_name.append(card_info['name'])

        # print('card_pull:', card_pull)
        card_attack_temp = card_attack.copy()

        m_cur_hp = the_pk_monster['cur_hp']
        m_max_hp = the_pk_monster['max_hp']

        # print('card_attack_temp:', card_attack_temp, len(card_attack_temp))
        selected_attack_value, selected_attack_card_name = \
            takeAttackClosest(card_attack_temp, card_attack_idx_name, m_cur_hp)
        print("卡片攻擊加總:", sum(selected_attack_value), ",即將出的卡牌:", selected_attack_card_name)
        for card in selected_attack_card_name:
            idx = int(axie_card_idx)
            if len(play_cards_array[idx - 1]) < 4:
                new_card_list.append(card)
                play_cards_array[idx - 1].append(card)
                print("還有的卡牌:", runtime_cards_library_copy, " 要打出去的卡:", card)
                runtime_cards_library_copy.remove(card)
            else:
                break

        m_cur_hp_temp = m_cur_hp - sum(selected_attack_value)
        id = the_pk_monster['id']
        for i in range(len(sub_monsters)):
            if id == sub_monsters[i]['id']:
                if m_cur_hp_temp <= 0:  # 代表這一次可以打死這隻怪
                    sub_monsters[i]['cur_hp'] = 0
                else:  # 代表打不死，要靠下一個axie繼續
                    sub_monsters[i]['cur_hp'] = m_cur_hp_temp

    # 計算axie可能會打那隻怪物

    # print('new cards:', new_card_list)
    print_list("怪物資料:", sub_monsters_for_showing)
    return new_card_list


def takeHealClosest(mylist, mylistIdx, mynumber):
    res = []
    res_idx = []

    # 先取出數值和清單，，並且做出所有排列組合
    for i in range(1, len(mylist) + 1):
        res += list(combinations(mylist, i))
        res_idx += list(combinations(mylistIdx, i))
    # print('res:', res)
    # print('res_idx:', res_idx)

    # 把所有組告都各自加總，然後 決定挑選的條件
    min_combine_card = []
    min_combine_card_idx = []
    idx = 0
    for j in res:
        if sum(j) <= mynumber:
            min_combine_card.append(list(j))
            min_combine_card_idx.append(list(res_idx[idx]))
        idx = idx + 1

    # 如果沒挑出來，那就把原來的卡牌 傳回去
    if len(min_combine_card) <= 0:
        # print('補血卡 takeHEALClosest return ----')
        # print('mylist:', mylist)
        # print('mylistIdx:', mylistIdx)
        return None, None

    # print('補血卡 takeHEALClosest return ----')
    # print('min_combine_card:', min_combine_card)
    # print('min_combine_card_idx:', min_combine_card_idx)
    # 如果有挑出來，那抓出 最適合的，
    min_idx = min_combine_card.index(max(min_combine_card))
    return max(min_combine_card), min_combine_card_idx[min_idx]


def takeAttackClosest(mylist, mylistIdx, mynumber):
    res = []
    res_idx = []
    for i in range(len(mylist) + 1):
        res += list(combinations(mylist, i))
        res_idx += list(combinations(mylistIdx, i))
    # print('res:', res)
    # print('res_idx:', res_idx)
    min_combine_card = []
    min_combine_card_idx = []
    idx = 0
    for j in res:
        if sum(j) >= mynumber:
            min_combine_card.append(list(j))
            min_combine_card_idx.append(list(res_idx[idx]))
        idx = idx + 1

    if len(min_combine_card) <= 0:
        # print('takeAttackClosest return ----')
        # print('mylist:', mylist)
        # print('mylistIdx:', mylistIdx)
        return mylist, mylistIdx
    # print('takeAttackClosest return ----')
    # print('min_combine_card:', min_combine_card)
    # print('min_combine_card_idx:', min_combine_card_idx)
    min_idx = min_combine_card.index(min(min_combine_card))
    return min(min_combine_card), min_combine_card_idx[min_idx]


# no used
def init_axies():
    snapshot(axie_handle, sleft=AXIE_WINDOW_PROPERTY['left'], stop=AXIE_WINDOW_PROPERTY['top'],
             sright=AXIE_WINDOW_PROPERTY['right'], sbot=AXIE_WINDOW_PROPERTY['bottom'])
    # find all axies's position
    axie_hp = [{'name': "1980381", 'nickname': 'do', 'hp': 0},
               {'name': "1985659", 'nickname': 'berry', 'hp': 0},
               {'name': "1986033", 'nickname': 'fish', 'hp': 0}]
    axie_snapshop = [{'name': "1980381", 'nickname': 'do', 'left': -90, 'top': -85, 'right': 75, 'bottom': 55},
                     {'name': "1985659", 'nickname': 'berry', 'left': -90, 'top': -80, 'right': 75, 'bottom': 55},
                     {'name': "1986033", 'nickname': 'fish', 'left': -90, 'top': -85, 'right': 75, 'bottom': 55}]
    for f in AXIE_NORMAL_LIST:
        fullpath = join(GalleryLibrary.AXIE_NORMAL_Icon.value, f)
        print('axie:', fullpath)
        for axie in MY_AXIES:
            if f[0:7] in axie['id']:
                axie_name = f[0:7]
                print('axie name:', axie_name)
                print('axie name:', axie['nickname'])
        degree, x, y = search_image_position(axie_handle, GalleryLibrary.Full_Image.value, fullpath, show=False,
                                             thresh_hold=25, click=False)

        print('axie degerr:', degree, x, y)


def detect_axie_status():
    axie_final_status = []
    answer = []
    for axies in AXIES_FORMATION:
        # print("detect axie:", axie_id)
        axie_id = axies['id']

        runtime_path = GalleryLibrary.AXIE_RUNTIME_FULL.value + '\\' + axie_id + '.JPG'
        sum_normal = 0
        sum_fire = 0
        sum_die = 0
        count = 0
        MAX = 3
        while count < MAX:
            snapshot_axies_runtime()
            # print('-----')

            degree = compare_hash(runtime_path, GalleryLibrary.AXIE_NORMAL_Icon.value + '\\' + axie_id + '.JPG')
            sum_normal = sum_normal + degree

            # degree = compare_hash(runtime_path, GalleryLibrary.AXIE_FIRE_Icon.value + '\\' + axie_id + '.JPG')
            # sum_fire = sum_fire + degree
            sum_fire = 999

            degree = compare_hash(runtime_path, GalleryLibrary.AXIE_DIE_Icon.value + '\\' + axie_id + '.JPG')
            sum_die = sum_die + degree

            count = count + 1
        axie_grade = []
        axie_grade.append(sum_normal)
        axie_grade.append(sum_fire)
        axie_grade.append(sum_die)

        print(axie_id, " sum:", sum_normal, sum_fire, sum_die)
        min_value = min(axie_grade)
        min_index = axie_grade.index(min_value)
        # print("min_index:", min_index)

        nickname = AXIES_NICKNAME.get(axie_id)
        final = nickname + " is " + AXIE_STATUS[min_index]
        axie_final_status.append(final)

        a = {'id': axie_id, 'status': AXIE_STATUS[min_index], 'status_id': min_index}
        answer.append(a)
    print(axie_final_status)
    # print(answer)
    return answer


def mission_statue(win_count, gamestatus):
    # 狀態檢查-:
    center_x = round((AXIE_WINDOW_PROPERTY['left'] + AXIE_WINDOW_PROPERTY['right']) / 2, 0)
    center_y = round((AXIE_WINDOW_PROPERTY['top'] + AXIE_WINDOW_PROPERTY['bottom']) / 2, 0)

    snapshot(axie_handle, sleft=AXIE_WINDOW_PROPERTY['left'], stop=AXIE_WINDOW_PROPERTY['top'],
             sright=AXIE_WINDOW_PROPERTY['right'], sbot=AXIE_WINDOW_PROPERTY['bottom'])

    # 狀態檢查-1:是否在關卡選擇畫面 ，點"start"按鈕，以執行打關卡任務
    if gamestatus == GameStatue.WinWithExp or \
            gamestatus == GameStatue.WinWithSLPBounds or \
            gamestatus == GameStatue.Lost or \
            gamestatus == GameStatue.Victory:
        isexit = start_mission_check()
        if isexit:
            gamestatus = GameStatue.Fighting
            return StepType.STEPOVER, win_count, gamestatus

    # if gamestatus == GameStatue.WinWithExp:
    #     # 狀態檢查-2:是否在勝利(拿多少SLP)的畫面，是的話點擊一下以讓畫面go next(勝了，可能會再這個給你多少能量的畫面)
    #     # 若當時已經沒點數了，就不會有這個能量畫面
    #     isexit = bounds_check(center_x, center_y)
    #     if isexit:
    #         gamestatus = GameStatue.WinWithSLPBounds
    #         return StepType.STEPOVER, win_count, gamestatus

    if gamestatus == GameStatue.Fighting:
        # 狀態檢查-4:是否勝利 了，如果勝的話，檢查勝幾次了，若到達勝利次數，則停止程式
        fight_win, degree = fight_result_check(center_x, center_y)
        print('fight_result_check:', fight_win, degree)
        if degree is not None:  # if degree is not NONE , it means we win or lost
            if fight_win:
                win_count = win_count + 1
                print('[WINNING~~~]', win_count, type(win_count), type(MAX_WIN_COUNT))
                gamestatus = GameStatue.Victory
                if win_count >= MAX_WIN_COUNT:
                    print('[STATUS-WIN] MISSION FINSIH!!! win ', win_count, ' times]')
                    return StepType.BREAK, win_count, gamestatus
            else:
                gamestatus = GameStatue.Lost
                print('[STATUS-LOSE] 戰鬥輸了')
            return StepType.CONTINUE, win_count, gamestatus

        # 狀態檢查-3:若沒出現 End Turn 按鈕， 代表還在打怪 ，若出現，則代表可以開始選戰鬥卡片了
        threashold = 7
        degree, x, y = search_image_position(axie_handle, GalleryLibrary.Full_Image.value,
                                             GalleryLibrary.Button_EndTurn_Button.value,
                                             show=False, thresh_hold=threashold, click=False)
        # print(degree, x, y)
        gamestatus = GameStatue.Fighting
        if degree > threashold:
            time.sleep(2)
            print('$$$$$ END　ＴＵＲＮ　還沒出現', degree)
            return StepType.CONTINUE, win_count, gamestatus
        print('$$$$$ END　ＴＵＲＮ　出現了，可以開始下一場戰鬥')
        return StepType.STEPOVER, win_count, gamestatus
    # dont match any status
    return StepType.CONTINUE, win_count, gamestatus


def remove_heal_cards(cards):
    while HealCards.FrontHeal.value in cards:
        cards.remove(HealCards.FrontHeal.value)
    while HealCards.Berry.value in cards:
        cards.remove(HealCards.Berry.value)
    while HealCards.DOHeal.value in cards:
        cards.remove(HealCards.DOHeal.value)


def snapshot_mission_level_runtime(x, y, w, h, level='cur_mission_temp.JPG'):
    left = x
    top = y
    right = x + w
    bottom = y + h

    path = '.\\picture\\' + level
    snapshot(axie_handle, filepath=path, sleft=left, stop=top,
             sright=right, sbot=bottom)
    return path


def check_current_mission(x, y, w, h, level='19', number=3):
    # 抓現在關卡的"Lunacia Runin 19的run time圖片
    runtimepath = snapshot_mission_level_runtime(x, y, w, h)

    # 跟圖褲比較，看符合那一關
    fit_mission = 0
    min_degree = 999
    degree = 0
    for f in listdir(GalleryLibrary.Misson_Final_Icon.value):
        fullpath = join(GalleryLibrary.Misson_Final_Icon.value + '\\', f)
        # print(fullpath)
        degree = compare_hash(runtimepath, fullpath)
        # print('degree:', degree)
        if degree < min_degree:
            min_degree = degree
            fit_mission = int(f.rstrip(".JPG")[-1:])

    # print("fit mission:",fit_mission)
    return fit_mission


def get_monsters(missionid=19):
    a = dictmenu(MONSTER_MISSION_INFO)
    mission_info_idx, mission_info = a.value_index('mission_id', str(missionid))
    # print("******mission_info:",mission_info)
    mission_id = mission_info['mission_id']
    mission_id_number = mission_info['mission_number']
    x = mission_info['x'] + AXIE_WINDOW_PROPERTY['left']
    y = mission_info['y'] + AXIE_WINDOW_PROPERTY['top']
    w = mission_info['w']
    h = mission_info['h']
    # print(x, y, w, h)
    current_mission_idx = missionid - 1
    if missionid != 1:
        current_mission_idx = check_current_mission(x, y, w, h, level=mission_id, number=mission_id_number)
    # 取出指定子關卡的資料
    submissions = mission_info['sub_mission'][current_mission_idx]
    submission_name = submissions['mission_id']
    sub_monster_number = submissions['monsters_number']
    sub_monster = submissions['monster']

    # print_list("現在的怪物:", sub_monster)
    return submission_name, sub_monster
    # 取出每隻怪物的血條


def detect_monster_hp(mission_id, monsters, move=False):
    # {'id': 0, 'hp': 677, 'property': 'stone', 'x': 795, 'y': 253, 'hp_left': 0, 'hp_top': 0},
    # {'id': 3, 'hp': 724, 'property': 'plant', 'x': 890, 'y': 336, 'hp_left': 0, 'hp_top': 0},
    # {'id': 5, 'hp': 677, 'property': 'stone', 'x': 795, 'y': 448, 'hp_left': 0, 'hp_top': 0}]},
    for m in monsters:
        m_id = m['id']
        m_hp = m['max_hp']
        left = m['hp_x'] + AXIE_WINDOW_PROPERTY['left']
        top = m['hp_y'] + AXIE_WINDOW_PROPERTY['top']
        if move:
            pg.moveTo(left, top)
        # time.sleep(5)
        # pg.moveTo(10, 10)
        w = MONSTER_HP_INFO['w']
        h = MONSTER_HP_INFO['h']
        right = left + w
        bottom = top + h
        pic_name = mission_id + '_' + str(m_id) + '.JPG'
        # pg.moveTo(left,top)
        # time.sleep(5)
        MAX = 5  # 找血量 抓5次的狀況
        retry = 1  #
        count = 0
        # print("detect monster:", pic_name)
        hp = []
        while retry > 0:
            count = 0
            while count < MAX:

                m_hp_pic = GalleryLibrary.Monster_Runtime_HP.value + '\\' + pic_name
                snapshot(axie_handle, filepath=m_hp_pic, sleft=left, stop=top,
                         sright=right, sbot=bottom)
                cropped = cv2.imread(m_hp_pic)
                cropped = cv2.resize(cropped, (w * 20, h * 20))

                cv2.imwrite(m_hp_pic, cropped)

                # 偵測數字
                temp_hp_number = detect_number(m_hp_pic)
                hp_number = ''
                if temp_hp_number:
                    for chr in temp_hp_number:
                        # print(chr, "is digit?", chr.isdigit())
                        if chr.isdigit():
                            hp_number = hp_number + chr
                        else:
                            break
                    hp.append(hp_number)

                # print('hp is ', hp_number)
                count = count + 1
                if len(hp_number) >= 3:
                    break

            find_hp = False
            if hp and len(hp):
                m['cur_hp'] = int(max(hp))
                find_hp = True
            retry = retry - 1
            if find_hp:
                break
    return monsters


def snapshot_monster_hp_runtime():
    w = MONSTER_HP_INFO['w']
    h = MONSTER_HP_INFO['h']

    # path = '.\\picture\\' + level
    # snapshot(axie_handle, filepath=path, sleft=left, stop=top,
    #          sright=right, sbot=bottom)


def collect_the_monster_sequence(monsters):
    monster_sequence = []
    for i in range(7):
        monster_sequence.append({})  # 令其轉為1維串列形式
    # print('monsters:', monsters)
    for m in monsters:
        old = m
        id = m['id']
        monster_sequence[id] = m
        m['idx_x'] = POSITION_COORDINATES[id]['idx_x']
        m['idx_y'] = POSITION_COORDINATES[id]['idx_y']

    return monster_sequence


def get_axies_sequence(axies):
    seq_path = GalleryLibrary.AXIE_Sequence.value
    temp = []
    answer = []
    for f in listdir(seq_path):
        id = f[:7]
        a = dictmenu(axies)
        idx, axie_id = a.value_index('id', id)
        if not axie_id:
            continue
        fullpath = join(seq_path, f)
        threashold = 30
        degree, x, y = search_image_position(axie_handle, GalleryLibrary.Full_Image.value,
                                             fullpath, show=False, thresh_hold=threashold, click=False)
        # print("axie sequence degree")
        t = str(x) + "_" + f  # 把x的位置 和axie的名字當成sort()的關鍵字
        if degree < threashold:
            temp.append(t)
    # sample: ['2534_1986033.JPG', '2654_1980381.JPG', '2778_1985659.JPG']
    nickname_dict = []
    for b in sorted(temp):
        s = b[-11:-4]
        answer.append(s)
        nickname = AXIES_NICKNAME.get(s)
        nickname_dict.append(nickname)

    print(answer, nickname_dict)
    return (answer)


# TODO: 如何決定怪物是"死亡"?
def detect_the_next_pk_monster(myaxie, sub_monsters):
    monster_sequence = collect_the_monster_sequence(sub_monsters)

    # print(len(monster_sequence), 'b:', monster_sequence)
    the_pk_monster = {}

    axie_id = myaxie['position_id']
    # print('myaxie:', myaxie, axie_id)
    for i in range(len(monster_sequence)):

        sequ_monster = monster_sequence[i]
        # for sequ_monster in monster_sequence:
        if len(sequ_monster) <= 0:
            continue
        m_cur_hp = sequ_monster['cur_hp']
        if m_cur_hp <= 0:  # monster is die
            continue
        # print('sequ_monster:',sequ_monster)
        m_id = sequ_monster['id']
        m_position = dictmenu(MONSTERS_POSITION_ID)
        m_idx, m_select_position = m_position.value_index('id', m_id)
        # print('m_select_position:', m_select_position)
        the_pk_monster = sequ_monster

        # 如何可能有並排的怪物 那就要檢查一下
        if m_select_position['rule'] != POSITION_RUEL.SINGLE:

            # 先找到AXIE位置的 行列值
            a_position = dictmenu(AXIES_POSITION_ID)
            a_idx, a_selected_p = a_position.value_index('id', axie_id)
            # print('a_selected_p:', a_selected_p)
            a_col = a_selected_p['col']

            # 現在怪物的座標資料，以行為比較方法
            m_col = m_select_position['col']

            # 找下一個怪物的資料，看是不是並排的怪物
            if i < len(monster_sequence) - 1:
                next_sequence_m = monster_sequence[i + 1]
                next_m_col = None
                next_m_rule = None

                # 如果確定有下一筆資料，那才拿出來做比對，如果沒有的話 就直接選擇現在最前面順位的怪物
                if len(next_sequence_m) != 0:
                    # print('next_m:', next_sequence_m)
                    # 找到下一個怪物的座標資料
                    next_m_idx, next_m_select_position = m_position.value_index('id', next_sequence_m['id'])
                    # print('next_m_select_position:', next_m_select_position)
                    next_m_col = next_m_select_position['col']
                    next_m_rule = next_m_select_position['rule']
                    # 如果正好是並排的，那檢查 AXIE 跟那一個怪物在同一行
                    if (next_m_rule == POSITION_RUEL.DOUBLE):
                        if (a_col != m_col):
                            the_pk_monster = next_sequence_m
                            break
            break
        else:
            break
            # TODO: 就要檢查自己是不是跟這個怪物同一排
    print('答案 axie:', myaxie, 'pk monster:', the_pk_monster)
    print('[the_pk_monster]:', the_pk_monster)
    return the_pk_monster


def test(k, n):
    # snapshot(axie_handle, sleft=AXIE_WINDOW_PROPERTY['left'], stop=AXIE_WINDOW_PROPERTY['top'],
    #          sright=AXIE_WINDOW_PROPERTY['right'], sbot=AXIE_WINDOW_PROPERTY['bottom'])
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    res = []
    for i in range(len(nums)):
        res += list(combinations(nums, i))
    print(res)
    res = [x for x in res if len(x) == k]
    a = []
    for j in res:
        if sum(j) == n:
            a.append(list(j))
    print(a)


def test3(b):
    b.update({'abc': 'aaa'})
    print(b)


def test2():
    a = '7035133-2.JPG'
    card_idx = AXIES_CARDS_PIC_PREFIX.get(a[:7])
    print(card_idx)
    exit()
    snapshot_axies_runtime()
    path = GalleryLibrary.AXIE_NORMAL_Icon.value + '\\1986033.JPG'
    # snapshot(axie_handle, filepath=path, sleft=left, stop=top, sright=right, sbot=bottom)
    exit()


def print_list(msg=None, list=None):
    for l in list:
        print(msg, l)


# phash感知哈希算法 0-64，值愈小，相似度愈高
if __name__ == '__main__':

    # can not be removed --- start
    if len(sys.argv) >= 3:
        MAX_WIN_COUNT = int(sys.argv[1])
        MISSION_ID = int(sys.argv[2])

    start_time = time.time()
    axie_handle = config_axie_window()
    snapshot(axie_handle, sleft=AXIE_WINDOW_PROPERTY['left'], stop=AXIE_WINDOW_PROPERTY['top'],
             sright=AXIE_WINDOW_PROPERTY['right'], sbot=AXIE_WINDOW_PROPERTY['bottom'])
    # can not be removed --- end
    # # LOAD AXIE 應用程式視窗資訊
    # print(AXIE_WINDOW_PROPERTY)
    # test2()
    # exit()
    win_count = 0
    # 0. Start
    game_status = GameStatue.Fighting
    while True:
        # 檢查是否有非戰鬥中的狀況
        # print('[WIN COUNT]:', win_count)
        time.sleep(1)
        step_status, win_count, game_status = mission_statue(win_count, game_status)
        print(step_status, win_count, ', game status, ', game_status.name)
        if step_status == StepType.CONTINUE:
            continue
        elif step_status == StepType.BREAK:
            break
        pg.moveTo(AXIE_WINDOW_PROPERTY['right'], AXIE_WINDOW_PROPERTY['bottom'])
        # ------ start fight------
        # 1. fighting init
        click_end_turn_button = False
        find_card_count = 0
        play_cards_array = create_empty_array(3)
        # for i in range(3):
        #     play_cards_array.append([])  # 令其轉為1維串列形式

        # 2. check axies hp
        print('##########  Next Round ####################')
        # 2.1 檢查axie狀態(normal/fire/die)
        axie_status = detect_axie_status()  # create runtime axies
        print_list("status:", axie_status)
        # exit()
        # 2.2 對normal的axie 檢查血量 和有沒有debuff
        detect_axie_hp(axie_status)
        # axie_hp_status = detect_axie_hp(axie_status)
        print_list("AXIE血量狀態:", axie_status)

        # 如果血量是0 但狀態是normal/ 血量正常，但狀態是die，代表辯識錯誤， 要去修正
        # 開始挑選卡片，一直選到1、上面被出牌的卡牌都滿了(每個角色上限4張) ，2、沒有卡牌可出 3、能量用光  則執行戰鬥
        while True:
            # ramdon_cards = sample(CARDS_LIBRARY_LIST, 12) #不重覆
            # ramdon_cards = randon_cards(CARDS_LIBRARY_LIST, 24) #重覆
            # print('play card list:', play_cards_array)

            # 3-1. 檢查畫面上，有那些卡牌可以出
            runtime_card_lib_len, runtime_cards_library = detect_cards()
            print("AXIE 可出的卡片:", runtime_cards_library)
            # print("[INLINE CARDS]:len:", str(runtime_card_lib_len), " list:", sorted(runtime_cards_library))

            # 3-2。把這些卡牌, 丟進策略池裡，挑選等下要出的牌
            ramdon_cards = sample(sorted(runtime_cards_library), runtime_card_lib_len)
            if axie_status:
                want_to_play_cards = stratgy(runtime_cards_library, play_cards_array, axie_status, MISSION_ID)
                # exit()
                # os.system('pause')
            else:  # 如果剩最後一隻，而且被打到fire狀態，那時就沒有血條的資料了，這時直接隨機出卡了
                want_to_play_cards = ramdon_cards
                remove_heal_cards(want_to_play_cards)

            print("要出的卡牌:", want_to_play_cards)
            # exit()
            # 3-3。如果沒有任何牌可以出的話，就直接去執行 End Turn Button
            if len(want_to_play_cards) <= 0:
                click_end_turn_button = True

            # 3-4 如果只剩下一張草莓卡，而且能量也只剩一點時， 就直接值行戰鬥
            elif runtime_card_lib_len == 1 and runtime_cards_library[0] == HealCards.Berry.value:
                # 這裡要重新抓圖是因為，可能出卡牌出到，接下來要出草莓卡，及但此時只剩1點能量，這時候需要重新抓圖來取得 能量_1圖案
                snapshot(axie_handle, sleft=AXIE_WINDOW_PROPERTY['left'], stop=AXIE_WINDOW_PROPERTY['top'],
                         sright=AXIE_WINDOW_PROPERTY['right'], sbot=AXIE_WINDOW_PROPERTY['bottom'])
                fullpath = join(GalleryLibrary.Cards_Library.value, runtime_cards_library[0])
                threashold = 4
                degree, x, y = search_image_position(axie_handle, GalleryLibrary.Full_Image.value,
                                                     GalleryLibrary.Button_Energy_1_Icon.value, show=False,
                                                     thresh_hold=threashold, click=False)
                # print("engery_1 degree:", degree)
                if degree <= threashold:
                    click_end_turn_button = True
            else:
                # 4 執行出卡牌
                print("[開始出卡牌，請勿移動滑鼠]")
                for f in want_to_play_cards:
                    snapshot(axie_handle, sleft=AXIE_WINDOW_PROPERTY['left'], stop=AXIE_WINDOW_PROPERTY['top'],
                             sright=AXIE_WINDOW_PROPERTY['right'], sbot=AXIE_WINDOW_PROPERTY['bottom'])
                    fullpath = join(GalleryLibrary.Cards_Library.value, f)
                    # print(fullpath)
                    # search cards
                    # print("[Search Cards]")
                    # 找指定卡片的位置，然後移動過去把 卡牌打出去
                    search_image_position(axie_handle, GalleryLibrary.Full_Image.value, fullpath, False, 25, click=True)

                    # # 取得此卡牌的prefix字，用來做index
                    # card_key = int(f[:1]) - 1
                    # for i in range(0, 4):
                    #     played_count = len(play_cards_array[card_key])
                    #     print(played_count, i, card_key, f)
                    #     # 若還沒有4張，則把現在指定的這張卡牌打出去，放進"上方已出的卡槽中"
                    #     if played_count < 4:
                    #         play_cards_array[card_key].append(f)
                    #         break
                    # 這行是用來防困用的，萬一遇到不預期的狀況，造成無法離開loop ，則用檢查卡牌上限30次，強制離開loop
                    find_card_count = find_card_count + 1

                    # 每打一張牌就檢查是否已經能量用完了
                    # search energy 0 , 如果找的到，代表現在已經沒能量了，則不需要再去找卡牌了
                    # print("[Search Energy]")
                    threashold = 3
                    degree, x, y = search_image_position(axie_handle, GalleryLibrary.Full_Image.value,
                                                         GalleryLibrary.Button_Energy_0_Icon.value, show=False,
                                                         thresh_hold=threashold, click=False)
                    # print("engery degree:", degree)
                    if degree <= threashold:
                        click_end_turn_button = True
                        break
                    click_end_turn_button = True
            # if count >30, maybe there is only 1 energy and we only have strawberry cards
            # so we should just stop find cards and click EndTurn
            if click_end_turn_button or find_card_count >= 30 or runtime_card_lib_len == 0:
                # search EndTurn
                print("[Search EndTurn]")
                degree, x, y = search_image_position(axie_handle, GalleryLibrary.Full_Image.value,
                                                     GalleryLibrary.Button_EndTurn_Button.value,
                                                     show=False, thresh_hold=7)
                print("play degree:", degree, x, y)
                break
    end_time = time.time()
    print(f'[Running Time]: %f seconds', {end_time - start_time})

# TODO: 做自動偵測AXIES的位置
