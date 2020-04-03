# -*- encoding: utf-8 -*-
from flask import *
import base64
from PIL import Image
from io import BytesIO
from src.inpainting import InpaintNet, EdgeNet, InputImages
import numpy as np
import torch


app = Flask(__name__)

#NN作成
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#edge_netはマスクされた画像とマスクされた画像のエッジとマスクのみの画像から、マスクされた部分を含んだエッジを作成
edge_net = EdgeNet().to(device)
edge_net.load_state_dict(edge_net.weight['generator'])

#inpait_netはマスクされた画像とedge_netで生成されたエッジ画像から、修復された画像を作成
inpaint_net = InpaintNet().to(device)
inpaint_net.load_state_dict(inpaint_net.weight['generator'])

@app.route('/')
def index():
    return render_template("content.html")


@app.route('/send_img', methods=['POST'])
def inpaiting():

    # 画像データを受け取ってデコード
    enc_original = request.get_json()['img1']
    enc_mask = request.get_json()['img2']
    dec_original = base64.b64decode(enc_original.split(',')[1])
    dec_mask = base64.b64decode(enc_mask.split(',')[1])

    #送信容量削減のためJSでリサイズしたかったが、上手く動作しないため、一旦サーバ側でリサイズ
    img_original = Image.open(BytesIO(dec_original))
    img_original = img_original.resize((512, 512))
    img_mask = Image.open(BytesIO(dec_mask))
    img_mask = img_mask.resize((512, 512))

    #画像にマスクをかける。(paint画像)
    c = Image.new('RGBA', img_original.size, (255, 255, 255, 0))
    c.paste(img_mask, (0, 0), img_mask)
    img_masked = Image.alpha_composite(img_original, c)
    img_masked = img_masked.convert("RGB")
    img_mask = img_mask.convert("RGB")

    #inpaining処理
    img_setting = InputImages(np.array(img_masked)/255, np.array(img_mask)/255)
    output_edge = edge_net(img_setting.input_edgenet().float())
    output_inpaint = inpaint_net(img_setting.input_inpaintnet(output_edge).float()).detach().numpy()
    output_inpaint = (output_inpaint * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    result_img = Image.fromarray(output_inpaint.reshape(512, 512, 3))


    #生成画像とオリジナル画像のデコード処理
    buffer = BytesIO()
    img_original.save(buffer, format="PNG")
    base64Img_original = base64.b64encode(buffer.getvalue()).decode("utf-8").replace("'", "")
    base64Img_original = "data:image/png;base64,{}".format(base64Img_original)

    buffer = BytesIO()
    result_img.save(buffer, format="PNG")
    base64Img_result = base64.b64encode(buffer.getvalue()).decode("utf-8").replace("'", "")
    base64Img_result = "data:image/png;base64,{}".format(base64Img_result)

    res = {
        'ip_type': 'inpaint_success',
        'result': base64Img_result,
        'original': base64Img_original
    }

    return jsonify(ResultSet=res)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
