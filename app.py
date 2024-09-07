from flask import Flask, render_template, request, jsonify, send_from_directory
import random
import base64
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.ai.functions.Basefunctions_for_ai import *
from src.ai.functions.detection import *

app = Flask(__name__)

# 저장할 경로 설정
SAVE_FOLDER = 'src/ai/data/test'  # 원하는 저장 경로로 설정
os.makedirs(SAVE_FOLDER, exist_ok=True)  # 폴더가 없으면 생성
app.config['UPLOAD_FOLDER'] = SAVE_FOLDER
# names: [ 'can_steel', 'can_aluminium', 'paper', 'PET_transparent' ,'PET_color' ,'plastic_PE', 'plastic_PP', 'plastic_PS', 'styrofoam' ,'plastic_bag' ,'glass_brown' ,'glass_green' ,'glass_transparent' ,'battery' ,'light' ]

# waste_types = [
#     {
#         "category": "01. 금속캔",
#         "subcategories": [
#             {
#                 "name": "001. 철캔",
#                 "description": "철로 만들어진 음료 및 식품 용기",
#                 "disposal_method": "철캔 전용 수거함에 버리세요. 내용물을 비우고 가능한 압축해주세요.",
#                 "icon": "can",
#                 "icon_color": "text-gray-600"
#             },
#             {
#                 "name": "002. 알루미늄캔",
#                 "description": "알루미늄으로 만들어진 음료 용기",
#                 "disposal_method": "알루미늄캔 전용 수거함에 버리세요. 내용물을 비우고 가능한 압축해주세요.",
#                 "icon": "can",
#                 "icon_color": "text-gray-400"
#             }
#         ]
#     },
#     {
#         "category": "02. 종이",
#         "subcategories": [
#             {
#                 "name": "001. 종이",
#                 "description": "신문, 잡지, 책, 노트, 종이 상자 등",
#                 "disposal_method": "종이류 전용 수거함에 버리세요. 물기에 젖지 않도록 주의하세요.",
#                 "icon": "file-text",
#                 "icon_color": "text-yellow-700"
#             }
#         ]
#     },
#     {
#         "category": "03. 페트병",
#         "subcategories": [
#             {
#                 "name": "001. 무색단일",
#                 "description": "투명한 페트병",
#                 "disposal_method": "라벨을 제거하고 내용물을 비운 후 페트병 전용 수거함에 버리세요.",
#                 "icon": "bottle",
#                 "icon_color": "text-blue-300"
#             },
#             {
#                 "name": "002. 유색단일",
#                 "description": "색깔이 있는 페트병",
#                 "disposal_method": "라벨을 제거하고 내용물을 비운 후 페트병 전용 수거함에 버리세요.",
#                 "icon": "bottle",
#                 "icon_color": "text-green-500"
#             }
#         ]
#     },
#     {
#         "category": "04. 플라스틱",
#         "subcategories": [
#             {
#                 "name": "001. PE",
#                 "description": "포리에틸렌으로 만든 플라스틱 제품",
#                 "disposal_method": "내용물을 비우고 깨끗이 씻은 후 플라스틱 수거함에 버리세요.",
#                 "icon": "package",
#                 "icon_color": "text-blue-500"
#             },
#             {
#                 "name": "002. PP",
#                 "description": "폴리프로필렌으로 만든 플라스틱 제품",
#                 "disposal_method": "내용물을 비우고 깨끗이 씻은 후 플라스틱 수거함에 버리세요.",
#                 "icon": "package",
#                 "icon_color": "text-red-500"
#             },
#             {
#                 "name": "003. PS",
#                 "description": "폴리스티렌으로 만든 플라스틱 제품",
#                 "disposal_method": "내용물을 비우고 깨끗이 씻은 후 플라스틱 수거함에 버리세요.",
#                 "icon": "package",
#                 "icon_color": "text-purple-500"
#             }
#         ]
#     },
#     {
#         "category": "05. 스티로폼",
#         "subcategories": [
#             {
#                 "name": "001. 스티로폼",
#                 "description": "발포 폴리스티렌으로 만든 포장재",
#                 "disposal_method": "이물질을 제거하고 부피를 줄인 후 스티로폼 전용 수거함에 버리세요.",
#                 "icon": "box",
#                 "icon_color": "text-white"
#             }
#         ]
#     },
#     {
#         "category": "06. 비닐",
#         "subcategories": [
#             {
#                 "name": "001. 비닐",
#                 "description": "비닐 봉투, 포장재 등",
#                 "disposal_method": "이물질을 제거하고 깨끗이 씻은 후 비닐 전용 수거함에 버리세요.",
#                 "icon": "shopping-bag",
#                 "icon_color": "text-gray-300"
#             }
#         ]
#     },
#     {
#         "category": "07. 유리병",
#         "subcategories": [
#             {
#                 "name": "001. 갈색",
#                 "description": "갈색 유리병",
#                 "disposal_method": "내용물을 비우고 깨끗이 씻은 후 유리병 수거함에 버리세요.",
#                 "icon": "wine-bottle",
#                 "icon_color": "text-yellow-900"
#             },
#             {
#                 "name": "002. 녹색",
#                 "description": "녹색 유리병",
#                 "disposal_method": "내용물을 비우고 깨끗이 씻은 후 유리병 수거함에 버리세요.",
#                 "icon": "wine-bottle",
#                 "icon_color": "text-green-700"
#             },
#             {
#                 "name": "003. 투명",
#                 "description": "투명한 유리병",
#                 "disposal_method": "내용물을 비우고 깨끗이 씻은 후 유리병 수거함에 버리세요.",
#                 "icon": "wine-bottle",
#                 "icon_color": "text-blue-200"
#             }
#         ]
#     },
#     {
#         "category": "08. 건전지",
#         "subcategories": [
#             {
#                 "name": "001. 건전지",
#                 "description": "일회용 또는 충전식 건전지",
#                 "disposal_method": "전용 수거함이나 주민센터에 비치된 수거함에 버리세요.",
#                 "icon": "battery",
#                 "icon_color": "text-red-600"
#             }
#         ]
#     },
#     {
#         "category": "09. 형광등",
#         "subcategories": [
#             {
#                 "name": "001. 형광등",
#                 "description": "형광등, LED 전구 등",
#                 "disposal_method": "깨지지 않도록 주의하여 전용 수거함에 버리세요.",
#                 "icon": "zap",
#                 "icon_color": "text-yellow-400"
#             }
#         ]
#     }
# ]

waste_types = [
    {
        "name": "001. 철캔",
        "description": "철로 만들어진 음료 및 식품 용기",
        "disposal_method": "철캔 전용 수거함에 버리세요. 내용물을 비우고 가능한 압축해주세요.",
        "icon": "can",
        "icon_color": "text-gray-600"
    },
    {
        "name": "002. 알루미늄캔",
        "description": "알루미늄으로 만들어진 음료 용기",
        "disposal_method": "알루미늄캔 전용 수거함에 버리세요. 내용물을 비우고 가능한 압축해주세요.",
        "icon": "can",
        "icon_color": "text-gray-400"
    },
    {
        "name": "001. 종이",
        "description": "신문, 잡지, 책, 노트, 종이 상자 등",
        "disposal_method": "종이류 전용 수거함에 버리세요. 물기에 젖지 않도록 주의하세요.",
        "icon": "file-text",
        "icon_color": "text-yellow-700"
    },
    {
        "name": "001. 무색단일",
        "description": "투명한 페트병",
        "disposal_method": "라벨을 제거하고 내용물을 비운 후 페트병 전용 수거함에 버리세요.",
        "icon": "bottle",
        "icon_color": "text-blue-300"
    },
    {
        "name": "002. 유색단일",
        "description": "색깔이 있는 페트병",
        "disposal_method": "라벨을 제거하고 내용물을 비운 후 페트병 전용 수거함에 버리세요.",
        "icon": "bottle",
        "icon_color": "text-green-500"
    },
    {
        "name": "001. PE",
        "description": "포리에틸렌으로 만든 플라스틱 제품",
        "disposal_method": "내용물을 비우고 깨끗이 씻은 후 플라스틱 수거함에 버리세요.",
        "icon": "package",
        "icon_color": "text-blue-500"
    },
    {
        "name": "002. PP",
        "description": "폴리프로필렌으로 만든 플라스틱 제품",
        "disposal_method": "내용물을 비우고 깨끗이 씻은 후 플라스틱 수거함에 버리세요.",
        "icon": "package",
        "icon_color": "text-red-500"
    },
    {
        "name": "003. PS",
        "description": "폴리스티렌으로 만든 플라스틱 제품",
        "disposal_method": "내용물을 비우고 깨끗이 씻은 후 플라스틱 수거함에 버리세요.",
        "icon": "package",
        "icon_color": "text-purple-500"
    },
    {
        "name": "001. 스티로폼",
        "description": "발포 폴리스티렌으로 만든 포장재",
        "disposal_method": "이물질을 제거하고 부피를 줄인 후 스티로폼 전용 수거함에 버리세요.",
        "icon": "box",
        "icon_color": "text-white"
    },
    {
        "name": "001. 비닐",
        "description": "비닐 봉투, 포장재 등",
        "disposal_method": "이물질을 제거하고 깨끗이 씻은 후 비닐 전용 수거함에 버리세요.",
        "icon": "shopping-bag",
        "icon_color": "text-gray-300"
    },
    {
        "name": "001. 갈색",
        "description": "갈색 유리병",
        "disposal_method": "내용물을 비우고 깨끗이 씻은 후 유리병 수거함에 버리세요.",
        "icon": "wine-bottle",
        "icon_color": "text-yellow-900"
    },
    {
        "name": "002. 녹색",
        "description": "녹색 유리병",
        "disposal_method": "내용물을 비우고 깨끗이 씻은 후 유리병 수거함에 버리세요.",
        "icon": "wine-bottle",
        "icon_color": "text-green-700"
    },
    {
        "name": "003. 투명",
        "description": "투명한 유리병",
        "disposal_method": "내용물을 비우고 깨끗이 씻은 후 유리병 수거함에 버리세요.",
        "icon": "wine-bottle",
        "icon_color": "text-blue-200"
    },
    {
        "name": "001. 건전지",
        "description": "일회용 또는 충전식 건전지",
        "disposal_method": "전용 수거함이나 주민센터에 비치된 수거함에 버리세요.",
        "icon": "battery",
        "icon_color": "text-red-600"
    },
    {
        "name": "001. 형광등",
        "description": "형광등, LED 전구 등",
        "disposal_method": "깨지지 않도록 주의하여 전용 수거함에 버리세요.",
        "icon": "zap",
        "icon_color": "text-yellow-400"
    }
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_waste():
    
    # 실제 구현에서는 여기에 이미지 분류 로직을 추가해야 합니다.
    # 이 예제에서는 랜덤하게 쓰레기 유형을 선택합니다.
    # classification_result = random.choice(random.choice(waste_types)['subcategories'])
    
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ['val', 'test']:  # run normally

        tLst = detection(opt.data,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.conf_thres,
            opt.iou_thres,
            opt.save_json,
            opt.save_ans_log,
            opt.single_cls,
            opt.augment,
            opt.verbose,
            save_txt=opt.save_txt,
            save_conf=opt.save_conf,
            )
        now = datetime.now()
        print('\n')
        print("Model Test End at ", now)
        print(tLst)

        result = {}
        cnt = 0
        for i in tLst:
            result[cnt] = waste_types[i]
            result[cnt]['filename'] = cnt+'_'+i+'.jpg'
            print(result[cnt]['filename'])
            cnt += 1
        
        classification_result = waste_types[tLst[0]]
        print(result)
        return jsonify(result)
        # return jsonify(classification_result)

@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.json
    image_data = data.get('image')

    if not image_data:
        return jsonify({'message': '이미지 데이터가 없습니다.'}), 400

    try:
        # 이미지 파일명 설정
        file_name = f"captured-image.jpg"
        save_path = os.path.join(SAVE_FOLDER, file_name)

        # Base64 문자열을 바이너리 데이터로 변환하여 저장
        with open(save_path, "wb") as file:
            file.write(base64.b64decode(image_data))

        return jsonify({'message': '이미지 저장 성공', 'path': save_path}), 200
    except Exception as e:
        print(f"이미지 저장 실패: {e}")
        return jsonify({'message': '이미지 저장 실패'}), 500

@app.route('/test/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)