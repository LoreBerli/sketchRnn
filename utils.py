import os
import json
import numpy as np
import math

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

    gen = json_gen(dt)
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


#get_info()
