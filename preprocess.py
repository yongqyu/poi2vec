from multiprocessing import Pool
import numpy as np
import pandas as pd
import tqdm
import os
from model import Node, Rec

checkin_file = "../dataset/loc-gowalla_totalCheckins.txt"
df = pd.read_csv(checkin_file, sep='\t', header=None)
df.columns = ["user", "time", "latitude", "longitude", "poi"]
print "total visit :", len(df),
df = df.drop_duplicates(subset=['poi'])
print "/ total poi :", len(df)
poi2pos = df.loc[:, ['latitude', 'longitude', 'poi']].set_index('poi').T.to_dict('list')

proc_n = 20

poi2id = {}
id2poi = []
for poi in df['poi']:
    if poi2id.get(poi) == None:
        poi2id[poi] = len(id2poi)
        id2poi.append(poi)
np.save("./poi2id.npy", poi2id)
np.save("./id2poi.npy", id2poi)

# build a tree of area
tree = Node(df['latitude'].min(), df['latitude'].max(),df['longitude'].max(), df['longitude'].min(), 0)
tree.build()
print "total node of tree :", Node.count
theta = Node.theta

def main(id2poi_batch):
    id2route = []
    id2lr = []
    id2prob = []
    # for find max len of route
    max_len = 0

    # make route/left_right_choice/probability list of each poi
    for idx, poi in enumerate(tqdm.tqdm(id2poi_batch)):
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

        # calculate max len of route. to be removed
        for lr in lr_set:
            if len(lr) > max_len:
                max_len = len(lr)

        # each leaf, how much they are overlaped
        for route in route_set:
            leaf_area = Rec(tree.find_idx(route[0]))
            area_list.append(leaf_area.overlap(poi_area))
        area_list = np.divide(area_list, sum(area_list))

        id2route.append(route_set)
        id2lr.append(lr_set)
        id2prob.append(area_list)

    print "max_len: ", max_len
    np.save("./id2route_%d.npy" % os.getpid(), id2route)
    np.save("./id2lr_%d.npy" % os.getpid(), id2lr)
    np.save("./id2prob_%d.npy" % os.getpid(), id2prob)
    
if __name__ == '__main__':
    pool = Pool(processes=proc_n)
    batch_size = len(id2poi)/proc_n
    id2poi_list = []
    for i in xrange(proc_n):
       id2poi_list.append(id2poi[i*batch_size:(i+1)*batch_size]) 
    pool.map(main, id2poi_list)
