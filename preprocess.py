import numpy as np
import pandas as pd
import tqdm

checkin_file = "../dataset/loc-gowalla_totalCheckins.txt"
df = pd.read_csv(checkin_file, sep='\t', header=None)
df.columns = ["user", "time", "latitude", "longitude", "poi"]
print "total visit :", len(df),
df = df.drop_duplicates(subset=['poi'])
print "/ total poi :", len(df)
poi2pos = df.loc[:, ['latitude', 'longitude', 'poi']].set_index('poi').T.to_dict('list')

poi2id = {}
id2poi = []
for poi in df['poi']:
    if poi2id.get(poi) == None:
        poi2id[poi] = len(id2poi)
        id2poi.append(poi)

class Rec:
    def __init__(self, (top, down, left, right)):
        self.top = top
        self.down = down
        self.left = left
        self.right = right

    def overlap(self, a):
        dx = min(self.top, a.top) - max(self.down, a.down)
        dy = min(self.right, a.right) - max(self.left, a.left)
        if (dx>=0) and (dy>=0):
            return dx*dy
        else:
            # error
            return -1

class Node:
    theta = 0.1
    count = 0
    leaves = []

    def find_idx(self, idx):
        # find in leaves
        for leaf in Node.leaves:
            if leaf.count == idx:
                return leaf.north, leaf.south, leaf.west, leaf.east

    def __init__(self, west, east, north, south, level):
        self.left = None
        self.right = None
        self.west = west
        self.east = east
        self.north = north
        self.south = south
        self.level = level
        self.count = Node.count
        Node.count += 1

    def build(self):
        # even : horizen, odd : vertical
        if self.level%2 == 0:
            if (self.east - (self.west+self.east)/2) > 2*Node.theta:
                self.left = Node(self.west, (self.west+self.east)/2, self.north, self.south, self.level+1)
                self.right = Node((self.west+self.east)/2, self.east, self.north, self.south, self.level+1)
                self.left.build()
                self.right.build()
            else:
                Node.leaves.append(self)
        else:
            if (self.north - (self.north+self.south)/2) > 2*Node.theta:
                self.left = Node(self.west, self.east, self.north, (self.north+self.south)/2, self.level+1)
                self.right = Node(self.west, self.east, (self.north+self.south)/2, self.south, self.level+1)
                self.left.build()
                self.right.build()
            else:
                Node.leaves.append(self)

    def find_route(self, (latitude, longitude)):
        if self.left == None:
            prev_route = [self.count]
            prev_lr = []
            return prev_route, prev_lr

        # left : 0, right : 1
        if self.level%2 == 0:
            if self.left.east < latitude:
                prev_route, prev_lr = self.right.find_route((latitude, longitude))
                prev_lr.append(1)
            else:
                prev_route, prev_lr = self.left.find_route((latitude, longitude))
                prev_lr.append(0)
        else:
            if self.left.south < longitude:
                prev_route, prev_lr = self.left.find_route((latitude, longitude))
                prev_lr.append(0)
            else:
                prev_route, prev_lr = self.right.find_route((latitude, longitude))
                prev_lr.append(1)
        prev_route.append(self.count)
        return prev_route, prev_lr

if __name__ == '__main__':
    # build a tree of area
    tree = Node(df['latitude'].min(), df['latitude'].max(),df['longitude'].max(), df['longitude'].min(), 0)
    tree.build()
    print "total node of tree :", Node.count
    # for find max len of route
    max_len = 0

    id2route = []
    id2lr = []
    id2prob = []
    # make route/left_right_choice/probability list of each poi
    for idx, poi in enumerate(tqdm.tqdm(id2poi)):
        # each poi, they have a area. p_n is each corner
        p_n = [(poi2pos[poi][0] - 0.5*Node.theta, poi2pos[poi][1] - 0.5*Node.theta)\
                ,(poi2pos[poi][0] - 0.5*Node.theta, poi2pos[poi][1] + 0.5*Node.theta)\
                ,(poi2pos[poi][0] + 0.5*Node.theta, poi2pos[poi][1] - 0.5*Node.theta)\
                ,(poi2pos[poi][0] + 0.5*Node.theta, poi2pos[poi][1] + 0.5*Node.theta)]
        # that area
        poi_area = Rec((poi2pos[poi][1]+0.5*Node.theta, poi2pos[poi][1]-0.5*Node.theta\
                        ,poi2pos[poi][0]-0.5*Node.theta, poi2pos[poi][0]+0.5*Node.theta))

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
    print max_len
