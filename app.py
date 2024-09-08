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
UPLOAD_FOLDER = 'src/ai/test/result'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

RESOURCE_FOLDER = 'static/image'
app.config['RESOURCE_FOLDER'] = RESOURCE_FOLDER

waste_types = [
    {
        "name": "001. Can_steel",
        "description": "Containers made of steel for beverages and food.",
        "disposal_method": "Dispose of in the steel can recycling bin. Empty the contents and compress if possible.",
        "icon": "trash-2",
        "icon_color": "text-gray-600",
        "guide": "1.png"
    },
    {
        "name": "002. Can_aluminium",
        "description": "Containers made of aluminium for beverages.",
        "disposal_method": "Dispose of in the aluminium can recycling bin. Empty the contents and compress if possible.",
        "icon": "trash-2",
        "icon_color": "text-gray-400",
        "guide": "1.png"
    },
    {
        "name": "003. Paper",
        "description": "Newspapers, magazines, books, notebooks, paper boxes, etc.",
        "disposal_method": "Dispose of in the paper recycling bin. Avoid getting it wet.",
        "icon": "file-text",
        "icon_color": "text-yellow-700",
        "guide": "2.png"
    },
    {
        "name": "004. PET_transparent",
        "description": "Transparent PET bottles.",
        "disposal_method": "Remove labels, empty the contents, and dispose of in the PET bottle recycling bin.",
        "icon": "droplet",
        "icon_color": "text-blue-300",
        "guide": "3.png"
    },
    {
        "name": "005. PET_color",
        "description": "Colored PET bottles.",
        "disposal_method": "Remove labels, empty the contents, and dispose of in the PET bottle recycling bin.",
        "icon": "droplet",
        "icon_color": "text-green-500",
        "guide": "3.png"
    },
    {
        "name": "006. Plastic_PE",
        "description": "Plastic products made from polyethylene.",
        "disposal_method": "Empty, clean, and dispose of in the plastic recycling bin.",
        "icon": "package",
        "icon_color": "text-blue-500",
        "guide": "3.png"
    },
    {
        "name": "007. Plastic_PP",
        "description": "Plastic products made from polypropylene.",
        "disposal_method": "Empty, clean, and dispose of in the plastic recycling bin.",
        "icon": "package",
        "icon_color": "text-red-500",
        "guide": "3.png"
    },
    {
        "name": "008. Plastic_PS",
        "description": "Plastic products made from polystyrene.",
        "disposal_method": "Empty, clean, and dispose of in the plastic recycling bin.",
        "icon": "package",
        "icon_color": "text-purple-500",
        "guide": "3.png"
    },
    {
        "name": "009. Styrofoam",
        "description": "Packaging materials made from expanded polystyrene.",
        "disposal_method": "Remove contaminants, compress, and dispose of in the Styrofoam recycling bin.",
        "icon": "box",
        "icon_color": "text-white",
        "guide": "4.png"
    },
    {
        "name": "010. Plastic_bag",
        "description": "Plastic bags, packaging materials, etc.",
        "disposal_method": "Remove contaminants, clean, and dispose of in the plastic bag recycling bin.",
        "icon": "shopping-bag",
        "icon_color": "text-gray-300",
        "guide": "3.png"
    },
    {
        "name": "011. Glass_brown",
        "description": "Brown glass bottles.",
        "disposal_method": "Empty, clean, and dispose of in the glass bottle recycling bin.",
        "icon": "droplet",
        "icon_color": "text-yellow-900",
        "guide": "5.png"
    },
    {
        "name": "012. Glass_green",
        "description": "Green glass bottles.",
        "disposal_method": "Empty, clean, and dispose of in the glass bottle recycling bin.",
        "icon": "droplet",
        "icon_color": "text-green-700",
        "guide": "5.png"
    },
    {
        "name": "013. Glass_transparent",
        "description": "Transparent glass bottles.",
        "disposal_method": "Empty, clean, and dispose of in the glass bottle recycling bin.",
        "icon": "droplet",
        "icon_color": "text-blue-200",
        "guide": "5.png"
    },
    {
        "name": "014. Battery",
        "description": "Disposable or rechargeable batteries.",
        "disposal_method": "Dispose of in the designated recycling bin or at the collection point in community centers.",
        "icon": "battery",
        "icon_color": "text-red-600",
        "guide": "6.png"
    },
    {
        "name": "015. Light",
        "description": "Fluorescent lights, LED bulbs, etc.",
        "disposal_method": "Dispose of carefully in the designated recycling bin to avoid breaking.",
        "icon": "zap",
        "icon_color": "text-yellow-400",
        "guide": "6.png"
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
            result[cnt]['filename'] = str(cnt+1)+'_'+str(i)+'.jpg'
            print(result[cnt]['filename'])
            cnt += 1
        
        print(result)
        return jsonify(result)

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

@app.route('/img/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/image/<filename>')
def serve_img(filename):
    return send_from_directory(app.config['RESOURCE_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)