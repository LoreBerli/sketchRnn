import json
import os
import random

path="good_dataset"
def genr(fil):
    with open(fil) as f:
        lines=f.readlines()
        print(len(lines))
        for ln in lines:
            yield ln

def check(g):
    try:
        return next(g)
    except StopIteration:
        pass

def main():
    gens=[]
    splits=12
    SPLIT_SIZE=10000
    for fe in os.listdir(path):
        gens.append(genr(path+"/"+fe))

    # for s in range(splits):
    #
    #     for i in range(0,SPLIT_SIZE):
    #         for g in gens:
                # try:
                #     tmp.append(next(g))
                # except StopIteration:
                #     pass
    #     with open(str(s)+'split.json', 'w') as outfile:
    #         for l in tmp:
    #             outfile.write(l)
    s=0
    has_next=True
    while(has_next):
        s+=1
        tmp=[]
        for i in range(0,SPLIT_SIZE):
            for k,g in enumerate(gens):
                try:
                    tmp.append(next(g))
                except StopIteration:
                    has_next=False
        with open("splits_2/"+str(s)+'split.json', 'w') as outfile:
            for l in tmp:
                outfile.write(l)


main()
