from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

waste_types = [
    {
        "name": "플라스틱",
        "description": "병, 용기, 포장재 등 플라스틱으로 만든 물품.",
        "disposal_method": "파란색 재활용 통에 버리세요. 재활용 전 용기를 깨끗이 씻어주세요.",
        "icon": "trash-2",
        "icon_color": "text-blue-500"
    },
    {
        "name": "종이",
        "description": "신문, 잡지, 골판지 및 기타 종이 제품.",
        "disposal_method": "파란색 재활용 통에 버리세요. 종이를 건조하고 깨끗하게 유지하세요.",
        "icon": "trash-2",
        "icon_color": "text-green-500"
    },
    {
        "name": "음식물 쓰레기",
        "description": "남은 음식, 과일 및 채소 조각, 기타 유기물.",
        "disposal_method": "녹색 통에 퇴비로 버리거나 가정용 퇴비 시스템을 사용하세요.",
        "icon": "trash-2",
        "icon_color": "text-green-700"
    },
    {
        "name": "일반 쓰레기",
        "description": "재활용이나 퇴비화할 수 없는 비재활용 품목.",
        "disposal_method": "일반 쓰레기통(보통 검정색 또는 회색)에 버리세요.",
        "icon": "trash-2",
        "icon_color": "text-gray-500"
    }
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_waste():
    # 실제 구현에서는 여기에 이미지 분류 로직을 추가해야 합니다.
    # 이 예제에서는 랜덤하게 쓰레기 유형을 선택합니다.
    classification_result = random.choice(waste_types)
    return jsonify(classification_result)

if __name__ == '__main__':
    app.run(debug=True)