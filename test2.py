import ctypes, sys
import random
import threading
from config_loader import load_config
from PIL import Image
import numpy as np
import cv2
import pydirectinput
import keyboard
import pyautogui
import pygetwindow as gw
import mss
import win32gui
import time
import math
from ui import run_ui
import requests
import os
import sys
from update import update_program




discordwebhook = "https://discord.com/api/webhooks/1352891066893996175/jLH8uAlLlzeF7peem1OcHnePLnLOoPNU5lg3UqmTG42-70DlKYxku0_ORLqLVoZyzIEI"

def Sendmsg(msg):
    data = {
        "content" : msg,
    }

    result = requests.post(discordwebhook, json = data)


# Sendmsg("용왕 토끼 젠!!!!!!!!!!")
def Sendmsg2(msg, window_title=None):
    data = {
        "content": msg
    }

    files = None
    if window_title:
        img = capture_game_window(window_title)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        buffered.seek(0)
        files = {
            "file": ("screenshot.png", buffered, "image/png")
        }

    if files:
        response = requests.post(discordwebhook, data=data, files=files)
    else:
        response = requests.post(discordwebhook, json=data)

    if response.status_code != 204:
        print("전송 실패:", response.text)
    else:
        print("전송 성공")

# 사용 예시
# Sendmsg("게임창 캡처와 함께 보냅니다", window_title)


sel_hunt=0
pyautogui.FAILSAFE = False
pydirectinput.FAILSAFE = False
pydirectinput.PAUSE = 0

window_title = "MapleStory Worlds-빅뱅리턴즈"
templates0 = {}
templates1 = {}
templates2 = {}
templates3 = {}
buff_timers = [0] * 6

config = load_config()

def set_hunting_field(index):
    global sel_hunt
    sel_hunt = index
    print(f"[main.py] 사냥터 인덱스 설정됨: {sel_hunt}")

def press_key(key):
    if key == 'left':
        keyboard.press('left')
        keyboard.press('z')
    elif key == 'right':
        keyboard.press('right')
        keyboard.press('z')
    # elif key == 'up':
    #     keyboard.press('up')


def release_keys():
    keyboard.release('left')
    keyboard.release('right')
    # keyboard.release('z')
    keyboard.release('up')
def load_templates0(base_path, number_image_paths):
    global templates0
    templates0 = {}
    for number, paths in number_image_paths.items():
        for path in paths:
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"템플릿 이미지 {path}를 열 수 없습니다.")
                continue
            templates0[(number, path)] = template
    return templates0

def load_templates1(base_path, number_image_paths):
    global templates1
    templates1 = {}
    for number, paths in number_image_paths.items():
        for path in paths:
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"템플릿 이미지 {path}를 열 수 없습니다.")
                continue
            templates1[(number, path)] = template
    return templates1

def load_templates2(base_path, number_image_paths):
    global templates2
    templates2 = {}
    for number, paths in number_image_paths.items():
        for path in paths:
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"템플릿 이미지 {path}를 열 수 없습니다.")
                continue
            templates2[(number, path)] = template
    return templates2
def load_templates3(base_path, number_image_paths):
    global templates3
    templates3 = {}
    for number, paths in number_image_paths.items():
        for path in paths:
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"템플릿 이미지 {path}를 열 수 없습니다.")
                continue
            templates3[(number, path)] = template
    return templates3
base_path = r'C:\imgsearch\maplestory\img\\'

number_image_paths = {
    'char0': [base_path + 'char00.png',base_path + 'char11.png',base_path + 'char000.png',base_path + 'char111.png'],

}
load_templates0(base_path, number_image_paths)


base_path = r'C:\imgsearch\maplestory\img\\'

number_image_paths = {
    # if sel_hunt == 0:
    #     'dal': [base_path + 'dal1.png',base_path + 'dal2.png',base_path + 'dal3.png',base_path + 'dal4.png'],
    #if sel_hunt == 1:
    'dal': [base_path + 'dal1.png',base_path + 'dal2.png',base_path + 'dal3.png',base_path + 'dal4.png'],
    # 'pig': [base_path + 'pig1.png',base_path + 'pig2.png'],
}
load_templates1(base_path, number_image_paths)



base_path = r'C:\imgsearch\maplestory\img\\'

number_image_paths = {

    'lope': [base_path + 'lope1.png',base_path + 'lope2.png',base_path + 'lope3.png'],
}
load_templates2(base_path, number_image_paths)

number_image_paths = {

    'chac_back': [base_path + 'chac_back.png'],
}
load_templates3(base_path, number_image_paths)

def capture_game_window(window_title):
    with mss.mss() as sct:
        while True:
            try:
                game_window = gw.getWindowsWithTitle(window_title)
                if game_window:
                    game_window = game_window[0]
                    if game_window.isActive:
                        hwnd = game_window._hWnd
                        left, top, right, bottom = win32gui.GetClientRect(hwnd)
                        win32gui.ClientToScreen(hwnd, (left, top))
                        monitor = {
                            "top": top, "left": left,
                            "width": right - left, "height": bottom - top
                        }
                        screenshot = sct.grab(monitor)
                        return Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)
                    else:
                        time.sleep(0.016)
                else:
                    time.sleep(0.016)
            except Exception as e:
                print(f"오류 발생: {e}")
                time.sleep(0.016)

def get_hp_mp_exp_ratio():

    screenshot = capture_game_window(window_title)
    bars = {
        "HP": (317, 682, 488, 692),
        "MP": (530, 682, 700, 692),
        "EXP": (316, 702, 700, 714)
    }

    results = {}
    for name, (x1, y1, x2, y2) in bars.items():
        cropped = screenshot.crop((x1, y1, x2, y2))
        img = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        if name == "HP":

            lower1 = np.array([0, 50, 50])
            upper1 = np.array([15, 255, 255])
            lower2 = np.array([160, 50, 50])
            upper2 = np.array([180, 255, 255])
            mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
        elif name == "MP":
            lower = np.array([100, 80, 80])
            upper = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
        elif name == "EXP":
            lower = np.array([30, 100, 100])
            upper = np.array([90, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)

        center_line = mask[mask.shape[0] // 2: mask.shape[0] // 2 + 1, :]
        ratio = np.count_nonzero(center_line) / center_line.size
        results[name] = ratio

    return results
def get_position():
    screenshot = capture_game_window(window_title)
    if sel_hunt==0:
        mini_map_coords = (11, 76, 278, 172)
    if sel_hunt == 1:
        mini_map_coords = (11, 76, 239, 275)

    mini_map = screenshot.crop(mini_map_coords)
    img = cv2.cvtColor(np.array(mini_map), cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    lower_yellow = np.array([25, 200, 200])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

   # cv2.imwrite("mini_map.png", img)  # 잘린 미니맵 저장
    #cv2.imwrite("yellow_mask.png", mask)  # 노란색 마스크 저장

    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)

    return (0, 0)

def find_character():
    global templates0
    screenshot = capture_game_window(window_title)
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
    threshold = 0.6
    best_val = 0
    best_loc = None
    best_template = None
    for (_, _), template in templates0.items():
        result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_template = template
    if best_val >= threshold and best_template is not None:
        h, w = best_template.shape[:2]
        center_x = best_loc[0] + w // 2
        center_y = best_loc[1] + h // 2
        print(f"내 캐릭터 이름 위치: ({center_x}, {center_y}) | 신뢰도: {best_val:.2f}")
        return (center_x, center_y)
    else:
        print(f"캐릭터 이름 인식 실패 (신뢰도: {best_val:.2f})")
        char_x,char_y=get_position()
        release_keys()
        if char_x<80:
            press_key('right')
        if char_x>160:

            press_key('left')
        return None

def find_back():
    global templates3
    screenshot = capture_game_window(window_title)
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
    threshold = 0.6
    best_val = 0
    best_loc = None
    best_template = None
    for (_, _), template in templates3.items():
        result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_template = template
    if best_val >= threshold and best_template is not None:
        h, w = best_template.shape[:2]
        center_x = best_loc[0] + w // 2
        center_y = best_loc[1] + h // 2
        #print(f"내 캐릭터 이름 위치: ({center_x}, {center_y}) | 신뢰도: {best_val:.2f}")
        return (center_x, center_y)
    else:
        print(f"캐릭터 이름 인식 실패 (신뢰도: {best_val:.2f})")
        return None

def find_lope():
    global templates2
    screenshot = capture_game_window(window_title)
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
    threshold = 0.8 # 높은 신뢰도로 몬스터 찾기
    monster_positions = []

    # 템플릿 매칭을 통해 몬스터의 위치 찾기
    for (_, _), template in templates2.items():
        result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            h, w = template.shape[:2]
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            monster_positions.append((center_x, center_y))
    #print(f"몬스터 위치: ({monster_positions}")
    return monster_positions


def find_monster(min_distance=15):
    global templates1
    screenshot = capture_game_window(window_title)
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
    threshold = 0.95
    raw_positions = []
    #print("몬스터 위치")
    for (_, _), template in templates1.items():
        result = cv2.matchTemplate(screenshot_cv, template, cv2.TM_CCOEFF_NORMED)
        h, w = template.shape[:2]
        locations = np.where(result >= threshold)

        for pt in zip(*locations[::-1]):  # (x, y) 형식으로 변환
            center_x = pt[0] + w // 2
            center_y = pt[1] + h // 2
            raw_positions.append((center_x, center_y))
    #print("몬스터 위치2")
    # 중복 제거 (너무 가까운 좌표 필터링)
    # filtered_positions = []
    # for pos in raw_positions:
    #     if all(math.hypot(pos[0] - fp[0], pos[1] - fp[1]) > min_distance for fp in filtered_positions):
    #         filtered_positions.append(pos)

    print(f"몬스터 위치 ({len(raw_positions)}개): {raw_positions}")
    return raw_positions


def match_template_for_text(screenshot, templates):

    best_match = None
    best_match_score = -1  # 유사도는 0에서 1 사이로 나오므로 초기 값은 가장 낮은 값으로 설정

    # 스크린샷을 그레이스케일로 변환
    screenshot_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

    # 템플릿 매칭을 통해 가장 유사한 템플릿 찾기
    for number, template_list in templates.items():
        #for template in template_list:
            # 템플릿과 스크린샷의 크기 비교

            #print(f"넘버 템플릿리스트 템플릿 : {number}##{template_list}##{template}")

        template_height, template_width = template_list.shape
        screenshot_height, screenshot_width = screenshot_gray.shape

#        print(f"스크린샷 크기 {screenshot_width}x{screenshot_height}가 템플릿 크기 {template_width}x{template_height},templates{number[0]}")

        if screenshot_height < template_height or screenshot_width < template_width:
            # 스크린샷이 템플릿보다 작은 경우 템플릿 매칭을 수행하지 않음
            print(f"스크린샷 크기 {screenshot_width}x{screenshot_height}가 템플릿 크기 {template_width}x{template_height}보다 작습니다.")
            continue

        # 템플릿 매칭 수행
        result = cv2.matchTemplate(screenshot_gray, template_list, cv2.TM_CCOEFF_NORMED)
        max_score = result.max()  # 템플릿과 스크린샷 간의 최대 유사도 값
#        print(f"result{max_score}")

        # 유사도 비교
        if max_score > best_match_score:
            best_match_score = max_score
            best_match = number[0]

        # 유사도가 충분히 낮으면 "0"으로 처리 (기본값)
        if best_match_score < 0.95:
            best_match = "8172"

    return best_match


def extract_number_from_region_text(screenshot, region_xy=(400, 0, 600, 200)):
    global templates_detect
    region = (region_xy[0], region_xy[1], region_xy[2], region_xy[3])
    digit_image = screenshot.crop(region)

    # 템플릿 매칭을 통한 숫자 추출
    digit = match_template_for_text(digit_image, templates_detect)

    return digit


def get_detect():
    screenshot = capture_game_window(window_title)  # 게임 창 캡처

    if screenshot:
        map_data = get_info(screenshot, "detect")
    return map_data



# 게임 정보를 추출하는 함수
def get_info(screenshot, char):
    # HP, MP, XY 영역 각각 추출

    hp_start_x = 1096
    mp_start_x = 1096
    x_start_x = 1016
    y_start_x = 1084
    step_size = 13

    # 각 영역에 대해 자리수 추출

    if char == "detect":
        result = extract_number_from_region_text(screenshot,region_xy=(550, 200, 750, 232))




    # 앞의 '0'을 제거하고, int형으로 반환
    result = int(result)  # 숫자 앞의 0을 제거
    return result


def jump():
    keyboard.press('j')
    time.sleep(0.5)
    keyboard.release('j')
    time.sleep(0.7)
    # keyboard.press('up')
    # time.sleep(0.5)
    # keyboard.release('up')
    # time.sleep(0.7)


def down_jump():
    keyboard.press('down')
    keyboard.press('alt')
    # keyboard.press('left')
    time.sleep(0.2)
    # keyboard.release('left')
    keyboard.release('alt')
    keyboard.release('down')
    time.sleep(1)



def move_to_position_continuous(target_x, target_y, tolerance=15):
    max_duration =7  # 최대 이동 시간 제한
    start_time = time.time()



    while True:

        curr_x, curr_y = get_position()
        dx = target_x - curr_x
        dy = -(target_y - curr_y)
        print(f"[이동] 현재({curr_x},{curr_y}) / 목표({target_x},{target_y})")

        # 목적지 근처 도달 시 정지
        if abs(dx) <= 2 and abs(dy) <= 5:
            #print("[이동 완료] 목표 도달")
            release_keys()
            get_status()
            break
        # if dy > 4:
        #     # press_key('up')
        #     release_keys()
        #     jump()
        #
        # elif dy < -6:
        #     release_keys()
        #     down_jump()
        # 좌우 이동만 판단 (연속 이동)
        elif dx > 2:
            press_key('right')
            # if dy > 4:
            #     release_keys()
            #     # press_key('up')
            #     jump()
            #
            # elif dy < -4:
            #     release_keys()
            #     down_jump()

        elif dx < -2:
            press_key('left')
            # if dy > 4:
            #     release_keys()
            #     # press_key('up')
            #     jump()
            #
            # elif dy < -4:
            #     release_keys()
            #     down_jump()



        # elif dy > 4:
        #         press_key('up')
        #         jump()
        #
        # elif dy < -100:
        #         down_jump()

        if abs(dx) <= 2:

            if dy > 2:
                release_keys()
                # press_key('up')
                jump()

            elif dy < -2:
                release_keys()
                down_jump()
        # y축 차이 크면 점프


        # 몬스터 공격 (빠르게 비동기 처리 권장)





        if time.time() - start_time > max_duration:
            print("[이동 중단] 시간 초과")
            release_keys()
            break
    attack_nearby_monsters(100, 70)
    release_keys()






def move_to_lope(target_x, target_y, tolerance=15):
    max_duration = 30  # 최대 이동 시간 제한
    start_time = time.time()

    curr_x, curr_y = get_position()
    dx = target_x - curr_x
    dy = -(target_y - curr_y)

    if dx > 0:
        press_key('right')
        time.sleep(0.2)
        if dy > 4:
            press_key('up')
            jump()

    elif dx < 0:
        press_key('left')
        time.sleep(0.2)
        if dy > 4:
            press_key('up')
            jump()

    release_keys()

    while True:
        curr_x, curr_y = get_position()
        dx = target_x - curr_x
        dy = -(target_y - curr_y)
        print(f"[이동] lope 현재({curr_x},{curr_y}) / 목표({target_x},{target_y})")

        # 목적지 근처 도달 시 정지
        if abs(dx) == 1 and dy < -1:
            #print("[이동 완료] 목표 도달")
            break

        # 좌우 이동만 판단 (연속 이동)
        press_key('up')

        # elif dy > 4:
        #         press_key('up')
        #         jump()
        #
        # elif dy < -100:
        #         down_jump()

        # y축 차이 크면 점프


        # 몬스터 공격 (빠르게 비동기 처리 권장)


        time.sleep(0.05)

        if time.time() - start_time > max_duration:
            print("[이동 중단] 시간 초과")
            break
    attack_nearby_monsters(100, 100)
    release_keys()



def move_left():
    pyautogui.keyDown("left")

def move_right():
    pyautogui.keyDown("right")

def stop_moving():
    pyautogui.keyUp("left")
    pyautogui.keyUp("right")

def press_jump():
    pyautogui.press("alt")  # 점프 키


def face_direction(target_x, current_x):
    if target_x > current_x:
        press_key('right')
    else:
        press_key('left')


def approach_and_climb_lope():


    character_pos = find_character()     # (x, y)
    rope_positions = find_lope()         # [(x, y), (x, y), ...]
    # print(f"1: ({character_x}, {character_y}) |")
    print(f"2: ({rope_positions}) | ")
    if not character_pos or not rope_positions:
        print("[오류] 캐릭터 또는 로프 위치를 찾지 못했습니다.")
        return
    # r_p_x,r_p_y = rope_positions
    character_x, character_y = character_pos
    print(f"내 캐릭터 이름 위치: ({character_x}, {character_y}) |")
    # print(f"로프 위치: ({r_p_x}, {r_p_y}) | ")

    # 가장 가까운 로프 선택
    rope_x, rope_y = min(
        rope_positions,
        key=lambda pos: math.hypot(pos[0] - character_x, pos[1] - character_y)
    )

    distance= abs(rope_x - character_x)

    # 거리가 멀면 이동
    while distance > 150 or distance < 100:
        character_pos = find_character()  # (x, y)
        rope_positions = find_lope()  # [(x, y), (x, y), ...]

        if not character_pos or not rope_positions:
            print("[오류] 캐릭터 또는 로프 위치를 찾지 못했습니다.")
            return

        character_x, character_y = character_pos

        # y 거리 차이 100 이하인 로프만 필터링
        valid_ropes = [pos for pos in rope_positions if abs(pos[1] - character_y) < 150]

        if not valid_ropes:
            print("[오류] 유효한 로프가 없습니다. (y 차이 150 이상)")
            return main()

        # 가장 가까운 로프 선택
        rope_x, rope_y = min(
            valid_ropes,
            key=lambda pos: math.hypot(pos[0] - character_x, pos[1] - character_y)
        )

        print(f"내 캐릭터 이름 위치: ({character_x}, {character_y}) |")
        print(f"선택된 로프 위치: ({rope_x}, {rope_y}) |")

        distance = abs(rope_x - character_x)

        if rope_x > character_x:
            print("로프가 오른쪽 → 오른쪽 이동")
            press_key('right')
        else:
            print("로프가 왼쪽 → 왼쪽 이동")
            press_key('left')


    release_keys()
    time.sleep(0.1)



    # 거리가 적절하면 점프 후 위로
    print("적절한 위치 도달 → 점프 후 로프 타기")
    face_direction(rope_x, character_x)  # 로프 방향 바라보기
    time.sleep(0.5)
    press_jump()
    release_keys()
    press_key('up')
    time.sleep(1)
    while True:
        back_pos = find_back()
        if back_pos is None:
            break  # 더 이상 뒤통수가 안 보이면 멈춤

    release_keys()

    print("로프 올라타기 시도 완료")
#
# def attack_nearby_monsters(x_tolerance=5, y_tolerance=5):
#
#     while True:  # 계속해서 몬스터를 탐지하고 공격
#
#         my_pos = find_character()
#         if my_pos is None:
#             return
#
#         monsters = find_monster()  # 몬스터 위치를 다시 찾음
#         if not monsters:
#             break  # 몬스터가 없으면 루프 종료
#
#         attacking = False  # 공격 상태 추적 변수
#
#         for mx, my in monsters:
#             x_distance = abs(mx - my_pos[0])
#             y_distance = abs(my - my_pos[1])
#
#             if x_distance <= x_tolerance and y_distance <= y_tolerance:
#                 direction_key = 'right' if mx > my_pos[0] else 'left'
#                 print(f"공격할 몬스터 발견! 방향: {direction_key}, 거리: ({x_distance:.1f}, {y_distance:.1f})")
#
#                 # 방향키 눌러 방향 전환
#                 release_keys()
#                 get_status()
#
#                 pyautogui.keyDown(direction_key)
#                 time.sleep(0.1)
#                 pyautogui.keyUp(direction_key)
#
#                 # 공격키 (예: 'ctrl') 입력
#                 pyautogui.keyDown("ctrl")
#                 time.sleep(0.3)
#                 pyautogui.keyUp("ctrl")
#
#
#                 attacking = True  # 몬스터가 발견되면 공격 상태로 변경
#
#                 # 공격 후 잠시 기다린 뒤, 다음 몬스터를 공격
#                 # time.sleep(0.0)
#
#         # 몬스터가 없을 때까지 반복
#         if not attacking:
#
# #             break  # 몬스터가 없다면 루프 종료
# def attack_nearby_monsters(x_tolerance=5, y_tolerance=5):
#     my_pos = find_character()
#     if my_pos is None:
#         return
#
#     monsters = find_monster()
#     if not monsters:
#         return  # 몬스터가 없으면 아무것도 안 하고 종료
#
#     attacking = False
#     for mx, my in monsters:
#         x_distance = abs(mx - my_pos[0])
#         y_distance = abs(my - my_pos[1])
#
#         if x_distance <= 300 and y_distance <= 150:
#             direction_key = 'right' if mx > my_pos[0] else 'left'
#             print(f"공격할 몬스터 발견! 방향: {direction_key}, 거리: ({x_distance:.1f}, {y_distance:.1f})")
#             attacking = True  # 공격 조건 만족;
#     # attacking = True
#     # if attacking:
#     #     release_keys()
#     #     x_char,y_char=get_position()
#     #     if x_char<80:
#     #         press_key('right')
#     #     if x_char>160:
#     #         press_key('left')
#     #     # press_key('right')
#     #d
#     #     get_status()
#     #
#     #     # pyautogui.keyDown("ctrl")
#     #     # time.sleep(0.7)
#     #     # pyautogui.keyUp("ctrl")
#     #
#     #
#     #     pyautogui.keyDown("del")
#     #     time.sleep(3)
#     #     pyautogui.keyUp("del")
#     #
#     #     # pyautogui.keyDown("del")
#     #     # time.sleep(0.5)
#     #     # pyautogui.keyUp("del")
#     #     release_keys()
#     #     # time.sleep(0.5)d
#
#     if attacking:
#         release_keys()
#         # x_char,y_char=get_position()
#         # if x_char<80:
#         #     press_key('right')
#         # if x_char>160:
#         #     press_key('left')
#         # press_key('right')
#
#         get_status()
#
#         pyautogui.keyDown(direction_key)
#         time.sleep(0.01)
#
#
#
#         # pyautogui.keyDown("ctrl")
#         # time.sleep(3)
#         # pyautogui.keyUp("ctrl")
#
#         pyautogui.keyDown("del")
#         time.sleep(1)
#         pyautogui.keyUp("del")
#         pyautogui.keyUp(direction_key)
#         release_keys()
#             # time.sleep(0.5)d


def attack_nearby_monsters(x_tolerance=5, y_tolerance=5):
    # my_pos = find_character()
    # if my_pos is None:
    #     return
    #
    # monsters = find_monster()
    # if not monsters:
    #     return  # 몬스터가 없으면 아무것도 안 하고 종료

    attacking = 1
    # for mx, my in monsters:
    #     x_distance = abs(mx - my_pos[0])
    #     y_distance = abs(my - my_pos[1])
    #
    #     if x_distance <= 300 and y_distance <= 150:
    #         direction_key = 'right' if mx > my_pos[0] else 'left'
    #         print(f"공격할 몬스터 발견! 방향: {direction_key}, 거리: ({x_distance:.1f}, {y_distance:.1f})")
    #         attacking = True  # 공격 조건 만족;
    # attacking = True
    # if attacking:
    #     release_keys()
    #     x_char,y_char=get_position()
    #     if x_char<80:
    #         press_key('right')
    #     if x_char>160:
    #         press_key('left')
    #     # press_key('right')
    #d
    #     get_status()
    #
    #     # pyautogui.keyDown("ctrl")
    #     # time.sleep(0.7)
    #     # pyautogui.keyUp("ctrl")
    #
    #
    #     pyautogui.keyDown("del")
    #     time.sleep(3)
    #     pyautogui.keyUp("del")
    #
    #     # pyautogui.keyDown("del")
    #     # time.sleep(0.5)
    #     # pyautogui.keyUp("del")
    #     release_keys()
    #     # time.sleep(0.5)d

    if attacking:
        release_keys()
        # x_char,y_char=get_position()
        # if x_char<80:
        #     press_key('right')
        # if x_char>160:
        #     press_key('left')
        # press_key('right')

        get_status()

        # pyautogui.keyDown(direction_key)
        time.sleep(0.01)



        # pyautogui.keyDown("ctrl")
        # time.sleep(3)
        # pyautogui.keyUp("ctrl")

        pyautogui.keyDown("del")
        time.sleep(0.6)
        pyautogui.keyUp("del")
        # pyautogui.keyUp(direction_key)
        release_keys()

#
# def attack_nearby_monsters(x_tolerance=5, y_tolerance=5):
#     my_pos = find_character()
#     if my_pos is None:
#         return
#
#     monsters = find_monster()
#     if not monsters:
#         return  # 몬스터가 없으면 아무것도 안 하고 종료
#
#     attacking = 1
#     # for mx, my in monsters:
#     #     x_distance = abs(mx - my_pos[0])
#     #     y_distance = abs(my - my_pos[1])
#     #
#     #     if x_distance <= 200 and y_distance <= 100:
#     #         direction_key = 'right' if mx > my_pos[0] else 'left'
#     #         print(f"공격할 몬스터 발견! 방향: {direction_key}, 거리: ({x_distance:.1f}, {y_distance:.1f})")
#     #         attacking = True  # 공격 조건 만족
#     # attacking = True
#     # if attacking:
#     #     release_keys()
#     #     x_char,y_char=get_position()
#     #     if x_char<80:
#     #         press_key('right')
#     #     if x_char>160:
#     #         press_key('left')
#     #     # press_key('right')
#     #d
#     #     get_status()
#     #
#     #     # pyautogui.keyDown("ctrl")
#     #     # time.sleep(0.7)
#     #     # pyautogui.keyUp("ctrl")
#     #
#     #
#     #     pyautogui.keyDown("del")
#     #     time.sleep(3)
#     #     pyautogui.keyUp("del")
#     #
#     #     # pyautogui.keyDown("del")
#     #     # time.sleep(0.5)
#     #     # pyautogui.keyUp("del")
#     #     release_keys()
#     #     # time.sleep(0.5)d
#
#     if attacking:
#         release_keys()
#         x_char,y_char=get_position()
#         if x_char<80:
#             press_key('right')
#         if x_char>160:
#             press_key('left')
#         # press_key('right')
#
#         get_status()
#
#         # pyautogui.keyDown("ctrl")
#         # time.sleep(0.7)
#         # pyautogui.keyUp("ctrl")
#
#
#         pyautogui.keyDown("ctrl")
#         time.sleep(3)
#         pyautogui.keyUp("ctrl")
#         pyautogui.keyDown("del")
#         time.sleep(1)
#         pyautogui.keyUp("del")
#             # pyautogui.keyDown("del")
#             # time.sleep(0.5)
#             # pyautogui.keyUp("del")
#         release_keys()
#             # time.sleep(0.5)d

def move_by_path(path, tolerance=5):
    for target_x, target_y in path:
        print(f"\n[경로 이동] 이동할 좌표: ({target_x}, {target_y})")

        move_to_position_continuous(target_x, target_y, 5)

def lope_by_path(path, tolerance=5):
    for target_x, target_y in path:
        print(f"\n[경로 이동] 이동할 좌표: ({target_x}, {target_y})")

        move_to_lope(target_x, target_y, 5)
        # 이동 후 몬스터 감지 및 공격


def parse_percent(text):
    """'30%' -> 0.3 변환"""
    return float(text.strip('%')) / 100

def use_buff(buff_index):
    """지정된 버프 사용 함수"""
    hotkey = config["buffs"][buff_index]["hotkey"]
    time.sleep(1.2)
    keyboard.press(hotkey)
    time.sleep(0.7)
    keyboard.release(hotkey)
    time.sleep(1)
    print(f"{buff_index}버프를 사용했습니다.")


def check_and_use_buffs():
    global buff_timers
    current_time = time.time()

    for i in range(6):
        buff = config["buffs"][i]
        if buff["use"]:
            interval = buff["interval"]
            print(f"{current_time}, {buff_timers}")
            print(f"{current_time-buff_timers[i]:.2f},{interval}")
            if current_time - buff_timers[i] >= interval:
                use_buff(i)
                buff_timers[i] = current_time

def get_status():
    global config
    ratios = get_hp_mp_exp_ratio()
    print(f"HP: {ratios['HP']:.2%} | MP: {ratios['MP']:.2%} | EXP: {ratios['EXP']:.2%}")

    hp_cfg = config.get("hp", {})
    mp_cfg = config.get("mp", {})
    pet_cfg = config.get("pet", {})

    hp_threshold = int(hp_cfg.get("threshold", "30%").replace("%", ""))
    mp_threshold = int(mp_cfg.get("threshold", "30%").replace("%", ""))
    pet_threshold = int(pet_cfg.get("threshold", "30%").replace("%", ""))

    while (ratios["HP"] * 100 < hp_threshold or ratios["MP"] * 100 < mp_threshold):
        ratios = get_hp_mp_exp_ratio()
        if hp_cfg.get("use", False) and ratios["HP"] * 100 < hp_threshold:
            print("→ HP 물약 사용")
            # pyautogui.press(hp_cfg.get("hotkey", "Home"))
            keyboard.press(hp_cfg.get("hotkey", "Home"))
            time.sleep(0.5)
            keyboard.release(hp_cfg.get("hotkey", "Home"))
        if mp_cfg.get("use", False) and ratios["MP"] * 100 < mp_threshold:
            print("→ MP 물약 사용")
            keyboard.press(mp_cfg.get("hotkey", "Insert"))
            time.sleep(0.5)
            keyboard.release(mp_cfg.get("hotkey", "Insert"))
            # pyautogui.press(mp_cfg.get("hotkey", "Insert"))

        if pet_cfg.get("use", False) and ratios["HP"] * 100 < pet_threshold:
            print("→ 펫 회복 물약 사용")
            pyautogui.press(pet_cfg.get("hotkey", "Home"))
            time.sleep(0.5)
            keyboard.release(mp_cfg.get("hotkey", "Home"))


def main():

    print("main() 함수 실행됨! 자동 사냥 시작")
    while True:
        #Sendmsg2("게임창 캡처와 함께 보냅니다", window_title)
        #path = [(78,32),(80, 45),(160,45),(80, 45),(160,45),(80, 45),(160,45)]
        #path = [(80, 45), (160, 45), (80, 45)]
        if sel_hunt == 0:
            check_and_use_buffs()
            path = [(115, 72), (170, 72),(115, 72),(115, 55)]
            move_by_path(path)

            for i in range(10):
                check_and_use_buffs()
                path = [(139, 72), (144, 72), (139, 55),(144, 72)]
                move_by_path(path)
        if sel_hunt == 1:
            # check_and_use_buffs()
            path = [(136,54),(116, 62), (116,72), (110,72),(116,86), (116,93), (116,100), (116,115),(116,122), (116,133),(134,133)]
            move_by_path(path)

            # for i in range(10):
            #     check_and_use_buffs()
            #     path = [(139, 72), (144, 72), (139, 55),(144, 72)]
            #     move_by_path(path)
        #check_and_use_buffs()
        # attack_nearby_monsters(100,200)
        # path = [(87, 32)]
        # move_by_path(path)
        # attack_nearby_monsters(100, 200)
        # path = [(88, 48)]
        # move_by_path(path)
        # attack_nearby_monsters(100, 200)

        release_keys()
        # press_key('j')
        # time.sleep(1)
        # release_keys()6
        # original_path = [(80, 106), (150, 106)]
        #
        # # ±20 범위 내에서 x 좌표를 랜덤하게 조정
        # randomized_path = [(
        #     random.randint(x - 20, x + 20),  # x 좌표 랜덤화
        #     y  # y 좌표는 고정
        # ) for x, y in original_path]
        # # path = [(35,88),(52,88),(56,93),(62,98),(119,98),(126,98),(69,98)]
        # move_by_path(randomized_path)
        # approach_and_climb_lope()
        #
        # path_lope=[(69,78)]
        # lope_by_path(path_lope)
        #
        # path=[(72,78)]
        # move_by_path(path)
        # time.sleep(1)
        #
        # path_lope=[(81,58)]
        # lope_by_path(path_lope)

if __name__ == "__main__":
    #update_program()
    run_ui(main_callback=main, set_hunting_field_callback=set_hunting_field)