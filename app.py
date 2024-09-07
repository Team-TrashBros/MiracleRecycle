from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

waste_types = [
    {
        "category": "01. 금속캔",
        "subcategories": [
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
            }
        ]
    },
    {
        "category": "02. 종이",
        "subcategories": [
            {
                "name": "001. 종이",
                "description": "신문, 잡지, 책, 노트, 종이 상자 등",
                "disposal_method": "종이류 전용 수거함에 버리세요. 물기에 젖지 않도록 주의하세요.",
                "icon": "file-text",
                "icon_color": "text-yellow-700"
            }
        ]
    },
    {
        "category": "03. 페트병",
        "subcategories": [
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
            }
        ]
    },
    {
        "category": "04. 플라스틱",
        "subcategories": [
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
            }
        ]
    },
    {
        "category": "05. 스티로폼",
        "subcategories": [
            {
                "name": "001. 스티로폼",
                "description": "발포 폴리스티렌으로 만든 포장재",
                "disposal_method": "이물질을 제거하고 부피를 줄인 후 스티로폼 전용 수거함에 버리세요.",
                "icon": "box",
                "icon_color": "text-white"
            }
        ]
    },
    {
        "category": "06. 비닐",
        "subcategories": [
            {
                "name": "001. 비닐",
                "description": "비닐 봉투, 포장재 등",
                "disposal_method": "이물질을 제거하고 깨끗이 씻은 후 비닐 전용 수거함에 버리세요.",
                "icon": "shopping-bag",
                "icon_color": "text-gray-300"
            }
        ]
    },
    {
        "category": "07. 유리병",
        "subcategories": [
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
            }
        ]
    },
    {
        "category": "08. 건전지",
        "subcategories": [
            {
                "name": "001. 건전지",
                "description": "일회용 또는 충전식 건전지",
                "disposal_method": "전용 수거함이나 주민센터에 비치된 수거함에 버리세요.",
                "icon": "battery",
                "icon_color": "text-red-600"
            }
        ]
    },
    {
        "category": "09. 형광등",
        "subcategories": [
            {
                "name": "001. 형광등",
                "description": "형광등, LED 전구 등",
                "disposal_method": "깨지지 않도록 주의하여 전용 수거함에 버리세요.",
                "icon": "zap",
                "icon_color": "text-yellow-400"
            }
        ]
    }
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_waste():
    # 실제 구현에서는 여기에 이미지 분류 로직을 추가해야 합니다.
    # 이 예제에서는 랜덤하게 쓰레기 유형을 선택합니다.
    classification_result = random.choice(random.choice(waste_types)['subcategories'])

    return jsonify(classification_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)