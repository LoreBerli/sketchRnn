from PIL import Image
from PIL import ImageDraw
import json
import math
import copy
import numpy as np
import time

import os

def json_gen(path):
    with open(path) as f:
        for line in f:
            ln = f.readline()
            yield ln

def draw_from_points(pts):

    im_out = Image.new("RGB", (256, 256), (255, 255, 255))
    im_dra = ImageDraw.ImageDraw(im_out)
    for s in pts:
        points = list(zip(s[0], s[1]))
        im_dra.line(points, fill=(0, 0, 0))
    return im_out

def draw_from_net(pts):

    im_out = Image.new("RGB", (256, 256), (255, 255, 255))
    im_dra = ImageDraw.ImageDraw(im_out)
    # x=pts[0:int(len(pts)/2)]
    # y=pts[1+int(len(pts)/2):-1]
    x=[pts[i] for i in range(0,len(pts)) if i%2==0]
    y=[pts[i] for i in range(0,len(pts)) if i%2==1]
    points = list(zip(x,y))
    im_dra.line(points, fill=(0, 0, 0))
    return im_out

def draw_coords(pts):

    im_out = Image.new("RGB", (256, 256), (255, 255, 255))
    im_dra = ImageDraw.ImageDraw(im_out)
    # x=pts[0:int(len(pts)/2)]
    # y=pts[1+int(len(pts)/2):-1]

    points = [ tuple(couple) for couple in pts]
    im_dra.line(points, fill=(0, 0, 0))
    return im_out




def draw_both_coords(net,gt):
    im_out = Image.new("RGB", (256, 256), (255, 255, 255))
    im_dra = ImageDraw.ImageDraw(im_out)
    # x=pts[0:int(len(pts)/2)]
    # y=pts[1+int(len(pts)/2):-1]

    points = [ tuple(couple) for couple in net]
    gt_points=[ tuple(couple) for couple in gt]
    im_dra.line(points, fill=(0, 0, 0))
    im_dra.line(gt_points, fill=(255, 128, 128))
    return im_out

def draw_with_z_axis(pts):
    im_out = Image.new("RGB", (256, 256), (255, 255, 255))
    im_dra = ImageDraw.ImageDraw(im_out)

    points = [ tuple(couple) for couple in pts]
    for p in range(0,len(points)-1):
        print(points[p][2])
        if(points[p][2]>128):
            im_dra.line((points[p][0:2],points[p+1][0:2]),fill=(0, 0, 0))

    return im_out

def draw_both_with_z_axis(pts,gt):
    im_out = Image.new("RGB", (256, 256), (255, 255, 255))
    im_dra = ImageDraw.ImageDraw(im_out)
    
    points = [ tuple(couple) for couple in pts]
    gt_points=[tuple(couple) for couple in gt]
    for p in range(0,len(points)-1):

        if(points[p][2]>128):
            im_dra.line((points[p][0:2],points[p+1][0:2]),fill=(0, 0, 0))
    for p in range(0,len(gt_points)-1):
        if(gt_points[p][2]>128):
            im_dra.line((gt_points[p][0:2],gt_points[p+1][0:2]),fill=(255, 128, 128))

    return im_out

def draw_one_hot(pts,got,scale):
    gt=copy.copy(got)
    im_out = Image.new("RGB", (scale,scale), (255, 255, 255))
    im_dra = ImageDraw.ImageDraw(im_out)
    pts[:,0:2]=((1.0+pts[:,0:2])*128)
    gt[:,0:2]=((1.0+gt[:,0:2])*128)

    points = [tuple(couple) for couple in pts]
    gt_points = [tuple(couple) for couple in gt]
    direct = 0

    for p in range(0, len(points) - 1):

        #im_dra.ellipse([points[p][0:2]+prev[0],(points[p][0:2][0]+1,points[p][0:2][1]+1)],fill=(30, 120, 30))
        if (points[p][3] > points[p][2]):
            im_dra.line((points[p][0:2], points[p + 1][0:2]), fill=(0, 0,max(0,(130+direct))))
            direct+=10

    direct = 0


    for p in range(0, len(gt_points) - 1):
        #im_dra.point(gt_points[p][0:2], fill=(70, 70, 70))
        #im_dra.ellipse([gt_points[p][0:2],(gt_points[p][0:2][0]+1,gt_points[p][0:2][1]+1)],fill=(70, 70, 70))
        if (gt_points[p][3] > gt_points[p][2]):

            im_dra.line((gt_points[p][0:2], gt_points[p + 1][0:2]), fill=(min(130+direct,255), 120, 120))
            direct+=10
            #im_dra.point(gt_points[p][0:2], fill=(50, 120, 50))



    return im_out

def save_tested(pts,name,id):

    im = draw_coords(pts)
    im.save("out/"+name+"/"+id+".png")

def save_batch(batch,name,id):
    total=Image.new("RGB",(256*int(math.sqrt(len(batch))),256*int(math.sqrt(len(batch)))))
    ims=[]
    for j,i in enumerate(batch):
        img=draw_coords(i)
        total.paste(img,(j%int(math.sqrt(len(batch)))*256,j//int(math.sqrt(len(batch)))*256))
    total.save(name+"/"+id+".png")
    return total

def save_batch_z_axis(batch,name,id):
    total=Image.new("RGB",(256*int(math.sqrt(len(batch))),256*int(math.sqrt(len(batch)))))
    ims=[]
    for j,i in enumerate(batch):
        img=draw_with_z_axis(i)
        total.paste(img,(j%int(math.sqrt(len(batch)))*256,j//int(math.sqrt(len(batch)))*256))
    total.save(name+"/"+id+".png")
    return total

def save_batch_diff(batch,gt,name,id):
    total=Image.new("RGB",(256*int(math.sqrt(len(batch))),256*int(math.sqrt(len(batch)))))
    ims=[]
    for j,i in enumerate(batch):
        img=draw_both_coords(batch[j],gt[j])
        total.paste(img,(j%int(math.sqrt(len(batch)))*256,j//int(math.sqrt(len(batch)))*256))
    total.save(name+"/"+id+".png")
    return total

def save_batch_diff_z_axis(batch,gt,name,id):
    scale=256#broken
    total=Image.new("RGB",(scale*int(math.sqrt(len(batch))),scale*int(math.sqrt(len(batch)))))
    ims=[]
    #print(batch[0])
    for j,i in enumerate(batch):

        img=draw_one_hot(batch[j],gt[j],scale)
        total.paste(img,(j%int(math.sqrt(len(batch)))*scale,j//int(math.sqrt(len(batch)))*scale))
    total.save(name+"/"+id+".png")
    return total


def save_stupid_batch(batch,name,id):
    total=Image.new("RGB",(256*int(math.sqrt(len(batch))),256*int(math.sqrt(len(batch)))))
    ims=[]
    for j,i in enumerate(batch):
        img=draw_from_net(i)
        total.paste(img,(j%int(math.sqrt(len(batch)))*256,j//int(math.sqrt(len(batch)))*256))
    total.save("out/"+name+"/"+id+".png")

def draw_from_json(ln):
    jso=json.loads(ln)
    strokes=jso['drawing']
    return draw_from_points(strokes)

def test_drawing():
    import utils
    gen=utils.get_slightly_less_simplified_data("dataset/car/car",70)
    for i in range(0,10):
        print(i)
        dta=gen.__next__()
        print(dta)
        img=draw_with_z_axis(dta)
        img.save("out/test"+str(i)+".png")

def test():
    gene = json_gen("ambulance")
    for g in gene:
        print(g)
        im = draw_from_json(g)
        im.show()
