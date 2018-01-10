
class Rec:
    # Rectangle for calculate overlaped area
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
# Tree Node
    theta = 0.1 
    count = 0 
    leaves = []

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

    def find_idx(self, idx):
        # find in leaves
        for leaf in Node.leaves:
            if leaf.count == idx:
                return leaf.north, leaf.south, leaf.west, leaf.east
