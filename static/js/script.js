$(function(){

    $('#loading').hide();
    $('#canvasButton').show();
    var canvas1 = new fabric.Canvas('cnvs1', {preserveObjectStacking: true}); //キャンバスのインスタンス作成
    var img; //読み込まれた画像
    var imgUrl; //画像のURL
    var imgSlength; //画像の縦と横の長さうち短い方

    var drawingMode; //0:画像の位置を決める 1:ペンでお絵かき
    var burshWidth = 20; //ペンの太さ
    var width = 1;

    $('#inImg').on('change', function (e) {
        $('#uploader').hide();
        $('.canvas').show();
        $('.mode1').show();
        $('.mode2').hide();
        $('#cnvs2').hide();
        img = new Image();
        var reader = new FileReader();
        //読み込んだ画像をcanvasに入れる
        reader.onload = function (e) {
            canvas1.clear(); //前の画像を消去
            img.src = reader.result;
            fabric.Image.fromURL(this.result, function(oImg) {
                imgSlength = Math.min(img.naturalWidth, img.naturalHeight) / 512; //読み込んだ写真の長い方をキャンバスいっぱいに拡大し固定　もう一方はスライドできる
                if(img.naturalWidth > img.naturalHeight){
                    oImg.set({
                        lockMovementY: true,
                    });
                }else if(img.naturalWidth < img.naturalHeight){
                    oImg.set({
                        lockMovementX: true,
                    });
                }else{
                    oImg.set({
                        lockMovementX: true,
                        lockMovementY: true
                    });
                }
                oImg.scale(1 / imgSlength);
                oImg.set({
                    hasRotatingPoint: false
                });
                oImg.setControlsVisibility({                                                            //ガイド線を非表示
                     mt: false,    // middle top
                     mb: false,    // middle bottom
                     ml: false,    // middle left
                     mr: false,    // middle right
                     bl: false,    // bottom left
                     br: false,    // bottom right
                     tl: false,    // top left
                     tr: false,    // top right
                     mtr: false,    // middle top rotete
                });
                canvas1.add(oImg);
            });
        };
        reader.readAsDataURL(e.target.files[0]);
    });

    $('#paint').on('click', function(){
        canvas1.isDrawingMode = false;

        $('#cnvs2').show();

        var canvas2 = new fabric.Canvas('cnvs2', {preserveObjectStacking: true});
            canvas2.forEachObject(function(object){
            object.selectable = false;
        });
        $('.mode1').hide();
        $('.mode2').show();
        canvas2.isDrawingMode = true;
        canvas2.freeDrawingBrush.width = burshWidth;
        canvas2.freeDrawingBrush.color = "white";
    });

    $('#move').on('click', function(){
        canvas1.isDrawingMode = false;
        $('#cnvs2').hide();
        $('.mode1').show();
        $('.mode2').hide();
        // canvas2.clear();
        canvas2.isDrawingMode = false;
    });
    $('#changeImg').on('click', function(){
        // $('.uploader').show();
        // $('.canvas').hide();
        location.reload();
        canvas1.clear();
        canvas2.clear();

    });




    $('#sendImg').on('click', function(){

        $('#loading').show();
        $('#canvasButton').hide();
        $('#canvasDescription').hide();
        //canvas elementを取得
        var canvas1 = document.getElementById('cnvs1');
        // canvas1.width = 512;
        // canvas1.height = 512;
        var canvas2 = document.getElementById('cnvs2');
        // canvas2.width = 512;
        // canvas2.height = 512;



        //base64データを取得（エンコード）
        var base64_1 = canvas1.toDataURL('image/png');
        var base64_2 = canvas2.toDataURL('image/png');

        var fData = new FormData();

        fData.append('img1', base64_1);
        fData.append('img2', base64_2);

        var JSONdata = {
            img1: base64_1,
            img2: base64_2
        };

        $.ajax({
            url: 'http://127.0.0.1:5000/send_img',
            type: 'POST',
            data : JSON.stringify(JSONdata),
            contentType: 'application/JSON',
            dataType : 'JSON',
            processData: false,

            success: function(data, dataType) {
                if (data.ResultSet.ip_type == 'inpaint_success') {
                     console.log('Success', data);
                     var result = document.getElementById("result-img");
                     result.src = data.ResultSet.result;
                     var original = document.getElementById("original-img");
                     original.src = data.ResultSet.original;

                    $('#content-before').hide();
                    $('#loading').hide();
                    $('#content-after').show();

                 }
            },
            error: function(XMLHttpRequest, textStatus, errorThrown) {
                console.log('Error : ' + errorThrown);
            }
        })
    });
});