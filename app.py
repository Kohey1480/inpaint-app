# -*- encoding: utf-8 -*-
from flask import *
import base64
from PIL import Image
from io import BytesIO
# import datetime
from src.imgcombine import Imgcombine4
from src.inpainting import InpaintNet, EdgeNet, Preprocessing
import numpy as np
import torch

app = Flask(__name__)


#NN作成

#edge_netはマスクされた画像とマスクされた画像のエッジとマスクのみの画像から、マスクされた部分を含んだエッジを作成
edge_net = EdgeNet().to("cuda:0" if torch.cuda.is_available() else "cpu")
edge_net.load_state_dict(edge_net.weight['generator'])

#inpait_netはマスクされた画像とedge_netで生成されたエッジ画像から、修復された画像を作成
inpaint_net = InpaintNet().to("cuda:0" if torch.cuda.is_available() else "cpu")
inpaint_net.load_state_dict(inpaint_net.weight['generator'])

@app.route('/')
def hello():
    return render_template("content.html")


@app.route('/send_img', methods=['POST'])
def inpaiting():

    # 画像データを受け取ってデコード
    enc_data1 = request.get_json()['img1']
    enc_data2 = request.get_json()['img2']
    dec_data1 = base64.b64decode(enc_data1.split(',')[1])
    dec_img1 = Image.open(BytesIO(dec_data1))
    dec_img1 = dec_img1.resize((512, 512))
    dec_data2 = base64.b64decode(enc_data2.split(',')[1])
    dec_img2 = Image.open(BytesIO(dec_data2))
    dec_img2 = dec_img2.resize((512, 512))

    #画像にマスクをかける。(paint画像)
    c = Image.new('RGBA', dec_img1.size, (255, 255, 255, 0))
    c.paste(dec_img2, (0, 0), dec_img2)
    dec_img3 = Image.alpha_composite(dec_img1, c)
    dec_img3 = dec_img3.convert("RGB")
    dec_img2 = dec_img2.convert("RGB")

    #inpaining処理
    preprocess = Preprocessing(np.array(dec_img3)/255, np.array(dec_img2)/255)
    output_edge = edge_net(preprocess.input_edge.float())
    input_inpaint = torch.cat((torch.from_numpy(preprocess.image_masked).float(), output_edge), dim=1)
    output_inpaint = inpaint_net(input_inpaint.float()).detach().numpy()
    output_inpaint = (output_inpaint * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    result_img = Image.fromarray(output_inpaint.reshape(512, 512, 3))
    # result_img.save('./output_inpait.png')


    #生成画像とオリジナル画像のデコード処理
    buffer = BytesIO()
    dec_img1.save(buffer, format="PNG")
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







