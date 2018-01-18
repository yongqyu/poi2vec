from multiprocessing import Process
import numpy as np
import pandas as pd
import tqdm
import os
from models import Node, Rec

checkin_file = "../dataset/loc-gowalla_totalCheckins.txt"
df = pd.read_csv(checkin_file, sep='\t', header=None)
df.columns = ["user", "time", "latitude", "longitude", "poi"]
print "total visit :", len(df),
df = df.drop_duplicates(subset=['poi'])
print "/ total poi :", len(df)
poi2pos = df.loc[:, ['latitude', 'longitude', 'poi']].set_index('poi').T.to_dict('list')

proc_n = 20

poi2id = {'unk':0}
id2poi = ['unk']
for poi in df['poi']:
    if poi2id.get(poi) == None:
        poi2id[poi] = len(id2poi)
        id2poi.append(poi)
np.save("./npy/poi2id.npy", poi2id)
np.save("./npy/id2poi.npy", id2poi)

# build a tree of area
tree = Node(df['latitude'].min(), df['latitude'].max(),df['longitude'].max(), df['longitude'].min(), 0)
tree.build()
print "total node of tree :", Node.count
theta = Node.theta

def main(id2poi_batch, proc_i):
    id2route = []
    id2lr = []
    id2prob = []

    # make route/left_right_choice/probability list of each poi
    for poi in tqdm.tqdm(id2poi_batch):
        # each poi, they have a area. p_n is each corner
        p_n = [(poi2pos[poi][0] - 0.5*theta, poi2pos[poi][1] - 0.5*theta)\
                ,(poi2pos[poi][0] - 0.5*theta, poi2pos[poi][1] + 0.5*theta)\
                ,(poi2pos[poi][0] + 0.5*theta, poi2pos[poi][1] - 0.5*theta)\
                ,(poi2pos[poi][0] + 0.5*theta, poi2pos[poi][1] + 0.5*theta)]
        # that area
        poi_area = Rec((poi2pos[poi][1]+0.5*theta, poi2pos[poi][1]-0.5*theta\
                        ,poi2pos[poi][0]-0.5*theta, poi2pos[poi][0]+0.5*theta))

        route_list = []
        lr_list = []
        area_list = []
        # each corner, where they are contained in
        for p in p_n:
            route, lr = tree.find_route(p)
            route_list.append(route)
            lr_list.append(lr)

        # remove duplicate
        route_set = []
        for route in route_list:
            if route not in route_set:
                route_set.append(route)
        lr_set = []
        for lr in lr_list:
            if lr not in lr_set:
                lr_set.append(lr)

        # each leaf, how much they are overlaped
        for route in route_set:
            leaf_area = Rec(tree.find_idx(route[0]))
            area_list.append(leaf_area.overlap(poi_area))
        area_list = np.divide(area_list, sum(area_list))

        id2route.append(route_set)
        id2lr.append(lr_set)
        id2prob.append(area_list)

    np.save("./npy/splited_file/id2route_%02d.npy" % proc_i, id2route)
    np.save("./npy/splited_file/id2lr_%02d.npy" % proc_i, id2lr)
    np.save("./npy/splited_file/id2prob_%02d.npy" % proc_i, id2prob)
    
if __name__ == '__main__':
    procs = []
    batch_size = len(id2poi)/proc_n
    for i in xrange(proc_n+1):
        print "process #%02d running..."%(i+1)
        proc = Process(target=main, args=(id2poi[i*batch_size+1:(i+1)*batch_size+1], i+1))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
