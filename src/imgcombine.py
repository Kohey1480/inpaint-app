from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def Imgcombine4(time):
    time = time
    print(time)
    im00 = Image.open(f'./img/after/{time}-0-0.png')
    im01 = Image.open(f'./img/after/{time}-0-1.png')
    im10 = Image.open(f'./img/after/{time}-1-0.png')
    im11 = Image.open(f'./img/after/{time}-1-1.png')

    img = get_concat_v(get_concat_h(im00,im01),get_concat_h(im10,im11))
    img.save(f"./img/after/test.png")

    return img



def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst