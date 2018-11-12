import os
import json
import numpy as np
import math
import random
import time

def json_gen(path):
    with open(path) as f:
        for ln in f:
            yield ln

def get_data(MAX):
    gen = json_gen("ambulance")
    for v in gen:
        o=json.loads(v)
        points=[]
        strokes = o["drawing"]
        for s in strokes:
            points.extend(list(zip(s[0],s[1])))
        while(len(points)<MAX):
            points.append((0,0))
        yield (np.array(points[0:MAX])/128.0)-1.0

def get_simplified_data(dt,MAX):

    gen = create_shuffled_mixed_dataset("",200)
    for v in gen:

        o=json.loads(v)
        if(o["recognized"]==True):
            points=[]
            strokes = o["drawing"]
            for s in strokes:
                points.extend(list(zip(s[0],s[1])))
            while(len(points)<MAX):

                points.append(points[-1])
            yield (np.array(points[0:MAX])/128.0)-1.0

def create_shuffled_mixed_dataset(categories,per_cat):
    path="dataset"
    dts=[d for d in os.listdir(path)]
    random.shuffle(dts)
    dts_gens=[json_gen(path+"/"+d) for d in dts]
    random.shuffle(dts_gens)
    while(True):
        for i in range(0,per_cat):
            for g in dts_gens:
                yield next(g)

def get_slightly_less_simplified_data(dt,MAX):
    """
    Terribile ma funziona
    :param dt: json quickDraw dataset
    :param MAX: MAX # of points
    :return: normalized data , [x,y,pen_down]
    """
    gen = create_shuffled_mixed_dataset("",50000)
    for v in gen:

        o=json.loads(v)

        if(o["recognized"]==True):
            points=[]
            strokes = o["drawing"]
            for s in strokes:
                strok = list(zip(s[0], s[1]))
                pen_stroks=[]
                for pts in strok:
                    norm_pts=[(pts[0]/128.0)-1.0,(pts[1]/128.0)-1.0,1]
                    pen_stroks.append(np.array(norm_pts))

                pen_stroks[-1][2]=0
                points.extend(np.array(pen_stroks))
            while(len(points)<MAX):
                last_p_token=[0,0,-1]
                points.append(last_p_token)
            points=np.asarray(points)

            yield points[0:MAX,:]

def get_one_hot_data(dt,MAX):
    """
    Terribile ma funziona
    :param dt: json quickDraw dataset
    :param MAX: MAX # of points
    :return: normalized data , [x,y,pen_down]
    """
    UP=  [1.0, 0.0, 0.0]
    DOWN=[0.0, 1.0, 0.0]
    LAST=[0.0, 0.0, 1.0]

    gen = create_shuffled_mixed_dataset("",2000)

    for v in gen:

        o=json.loads(v)

        if(o["recognized"]==True):
            points=[]
            strokes = o["drawing"]
            for s in strokes:
                strok = list(zip(s[0], s[1]))
                pen_stroks=[]
                for i,pts in enumerate(strok):

                    norm_pts=[(pts[0]/128.0)-1.0,(pts[1]/128.0)-1.0]
                    if i<len(strok)-1:
                        norm_pts.extend(DOWN)
                    else:
                        norm_pts.extend(UP)
                    pen_stroks.append(np.array(norm_pts))


                points.extend(np.array(pen_stroks))
            while(len(points)<MAX):
                last_p_token=[0,0]
                last_p_token.extend(LAST)
                points.append(last_p_token)
            points=np.asarray(points)


            yield points[0:MAX]


def get_info(dt):
    gen = json_gen(dt)
    avg=0
    avgSqr=0
    max=0
    min=9999
    tot=0
    for v in gen:


        o=json.loads(v)
        if(o["recognized"]==True):
            tot+=1
            points=[]
            strokes = o["drawing"]
            for s in strokes:
                for p in zip(s[0],s[1]):
                    assert p[0]>=0 and p[0]<=256
                    assert p[1]>=0 and p[1]<=256
                points.extend(list(zip(s[0],s[1])))
            pts=len(points)
            avg+=pts
            avgSqr+=pts*pts
            min=pts if pts<min else min
            max=pts if pts>max else max

    avg=(float(avg/float(tot)))
    avgSqr=avgSqr/float(tot)
    print(max)
    print(min)
    std=math.sqrt(avgSqr-(avg*avg))
    print(avg,std)

def get_coord_drawings_z_axis(BATCH,leng):
    s=time.time()
    gg = get_one_hot_data("",leng)
    print("one_:hot",time.time()-s)
    while True:
        x_batched = np.zeros([BATCH,leng,5])

        for b in range(BATCH):
            x = next(gg)
            ll = len(x)
            x_batched[b,:,:]=x


        yield x_batched,x_batched

#print("Hi")

#get_slightly_less_simplified_data("dataset",10)
#get_slightly_less_simplified_data("dataset/car/car",10)
#get_info()
