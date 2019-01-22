import os
import json
import numpy as np
import math
import random
import time
import multiprocessing as mp

def json_gen(path):
    with open(path) as f:
        lines=f.readlines()
        random.shuffle(lines)
        for ln in lines:
            yield ln,path

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
    keys=[i for i in range(0,len(dts))]
    one_hot=np.eye(len(dts))[keys]
    #random.shuffle(dts)
    dts_gens=[json_gen(path+"/"+d) for d in dts]

    while(True):
        for i in range(0,per_cat):
            for g in dts_gens:
                h=next(g)

                yield h[0],one_hot[dts.index(h[1].split("/")[-1])]



def get_slightly_less_simplified_data(dt,MAX):
    """
    Terribile ma funziona
    :param dt: json quickDraw dataset
    :param MAX: MAX # of points
    :return: normalized data , [x,y,pen_down]
    """
    gen = create_shuffled_mixed_dataset("",100000)
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

    gen = create_shuffled_mixed_dataset("",10)

    for v in gen:

        o=json.loads(v[0])

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
            # if(len(points)>MAX):
            #     points=equidistant_subsampling(points,MAX)


            yield points[0:MAX],v[1]

def equidistant_subsampling(points,MAX):
    #pts=points[:,3]>0.0
    indexes = [i * len(points) // MAX + 100 // (2 * MAX) for i in range(MAX)]

    return points[indexes]
            
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
        cat_bat = np.zeros([BATCH, 17])
        for b in range(BATCH):
            x,cat = next(gg)
            ll = len(x)
            x_batched[b,:,:]=x
            cat_bat[b,:]=cat
        yield x_batched,cat_bat

def get_train_test_gens(BATCH,leng):
    train_path="splits_2/mini_train"
    test_path = "splits_2/test"
    spls={train_path:[],test_path:[]}
    paths=[train_path,test_path]
    for p in paths:
        spls[p]=[]
        for split in os.listdir(p):
            shuf=False if p==test_path else True
            print(split)
            spls[p].append([open(p + "/" + split), leng,BATCH,shuf])
    random.shuffle(spls[train_path])


    return Splits_handler(spls[train_path],"train"),Splits_handler(spls[test_path],"test")


def one_hot():
    classes=[c.replace("full_simplified_","").replace(".ndjson","") for c in os.listdir("good_dataset/")]
    print(classes)
    keys=[i for i in range(0,len(classes))]
    one_hot=np.eye(len(classes))[keys]
    return dict(zip(classes,one_hot))



def get_split_data(split, MAX,BATCH,shuffle):
    """
    Terribile ma funziona
    :param dt: json quickDraw dataset
    :param MAX: MAX # of points
    :return: normalized data , [x,y,pen_down]
    """
    UP=  [1.0, 0.0, 0.0]
    DOWN=[0.0, 1.0, 0.0]
    LAST=[0.0, 0.0, 1.0]
    f=split
    lines = f.readlines()
    print("#",str(len(lines)))
    if(shuffle):
        random.shuffle(lines)
    one_hot_m=one_hot()

    idx = 0
    while(True):
        data = np.zeros([BATCH, MAX, 5])
        classes = np.zeros([BATCH, len(one_hot_m.items())])
        if(idx>=len(lines)):
            idx=0
            if (shuffle):
                print("shuffle")
                random.shuffle(lines)

        for b in range(BATCH):
            v=lines[idx%len(lines)-1]
            o=json.loads(v)
            idx+=1
            while(o["recognized"]==False):
                v = lines[idx%(len(lines)-1)]
                o = json.loads(v)
                idx+=1

            cls=np.asarray(one_hot_m[o["word"]])
            points=[]
            strokes = o["drawing"]
            for s in strokes:

                strok = list(zip(s[0], s[1]))
                pen_stroks=[]
                for i,pts in enumerate(strok):
                    # rx = (np.random.rand()-0.5) / 200.0
                    # ry = (np.random.rand()-0.5) / 200.0
                    #norm_pts=[rx+(pts[0]/128.0)-1.0,ry+(pts[1]/128.0)-1.0]
                    norm_pts=[pts[0]/128.0-1.0,(pts[1]/128.0)-1.0]
                    if i<len(strok)-1:
                        norm_pts.extend(DOWN)
                    else:
                        norm_pts.extend(UP)
                    pen_stroks.append(np.array(norm_pts))


                points.extend(np.array(pen_stroks))
            while(len(points)<MAX):
                last_p_token=[points[-1][0],points[-1][1]]
                last_p_token.extend(LAST)
                points.append(last_p_token)
            points = np.asarray(points)
            if(len(points)>MAX):
                #points=equidistant_subsampling(points,MAX)
                points=points[0:MAX]
            # points=np.expand_dims(points)
            # points[:,-1]=cls
            data[b,:]=points
            classes[b,:]=cls


        yield data,classes

class Splits_handler():
    def __init__(self,jsons,n):
        self.jsons=jsons
        self.name=n
        self.last=0
        self.q=mp.Queue(maxsize=32)
        #open(p + "/" + split), leng,BATCH,shuf
        for j in self.jsons:
            Data_producer(j[0],j[1],j[2],j[3],self.q)

    def __next__(self):
        return self.q.get()

class Data_producer():
    def __init__(self,split, MAX,BATCH,shuffle,q):
        self.split=split
        self.MAX=MAX
        self.BATCH=BATCH
        self.shuffle=shuffle
        self.q=q
        random.seed(int(str(time.time()).replace(".", "")[-5:]))
        thread=mp.Process(target=self.produce)
        thread.daemon=True
        thread.start()

    def produce(self):
        """
        Terribile ma funziona
        :param dt: json quickDraw dataset
        :param MAX: MAX # of points
        :return: normalized data , [x,y,pen_down]
        """
        UP=  [1.0, 0.0, 0.0]
        DOWN=[0.0, 1.0, 0.0]
        LAST=[0.0, 0.0, 1.0]
        f=self.split
        lines = f.readlines()
        print("#",str(len(lines)))
        if(self.shuffle):
            random.shuffle(lines)
        one_hot_m=one_hot()

        idx = 0
        while(True):
            data = np.zeros([self.BATCH, self.MAX, 5])
            classes = np.zeros([self.BATCH, len(one_hot_m.items())])
            if(idx>=len(lines)):
                idx=0
                if (self.shuffle):
                    print("shuffle")
                    random.shuffle(lines)

            for b in range(self.BATCH):
                v=lines[idx%len(lines)-1]
                o=json.loads(v)
                idx+=1
                # while(o["recognized"]==False):
                #     v = lines[idx%(len(lines)-1)]
                #     o = json.loads(v)
                #     idx+=1

                cls=np.asarray(one_hot_m[o["word"]])
                points=[]
                strokes = o["drawing"]

                for s in strokes:
                    strok = list(zip(s[0], s[1]))
                    pen_stroks=[]

                    for i,pts in enumerate(strok):
                        rx = (np.random.rand()-0.5) / 128.0
                        ry = (np.random.rand()-0.5) / 128.0
                        # norm_pts=[rx+(pts[0]/128.0)-1.0,ry+(pts[1]/128.0)-1.0]
                        norm_pts=[rx+(pts[0])/128.0-1.0,(ry+(pts[1])/128.0)-1.0]



                        if i<len(strok)-1:
                            norm_pts.extend(DOWN)
                        else:
                            norm_pts.extend(UP)
                        pen_stroks.append(np.array(norm_pts))


                    points.extend(np.array(pen_stroks))

                while(len(points)<self.MAX):
                    last_p_token=[points[-1][0],points[-1][1]]
                    last_p_token.extend(LAST)
                    points.append(last_p_token)
                points = np.asarray(points)
                # if(len(points)>self.MAX):
                #     #points=equidistant_subsampling(points,MAX)
                #     points=points[0:self.MAX]
                # points=np.expand_dims(points)
                # points[:,-1]=cls
                data[b,:]=points[0:self.MAX]
                classes[b,:]=cls
            self.q.put([data,classes])