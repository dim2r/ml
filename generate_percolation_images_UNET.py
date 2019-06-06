import numpy as np
from PIL import Image
import  sys

image_size = 100
import random
import string


def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list


def genetate_percolation_system2d(size=40, prob=0.4):
    rnd_arr = np.random.rand(size, size)
    arr_mark_clusters = np.zeros((size, size)).astype(int)


    for x in range(size):
        for y in range(size):
            if rnd_arr[x][y] < prob:
                rnd_arr[x][y] = 1.0
            else:
                rnd_arr[x][y] = 0.0

    cluster_cnt = 0
    def mark(x, y, cluster_num, recursion_level):
        if recursion_level > 100000:
            print('recursion limit')
            return
        arr_mark_clusters[x][y] = cluster_num
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                xx = x + dx
                yy = y + dy
                if xx < size and yy < size and xx >= 0 and yy >= 0:
                    if arr_mark_clusters[xx][yy] == 0 and rnd_arr[xx][yy] == 1:
                        mark(xx, yy, cluster_num, recursion_level + 1)


    cluster_cnt = 0
    for x in range(size):
        for y in range(size):
            if rnd_arr[x][y] == 1 and arr_mark_clusters[x][y] == 0:
                cluster_cnt = cluster_cnt + 1
                #			print((x,y,cluster_cnt))
                mark(x, y, cluster_cnt, 0)

    # print(arr_mark_clusters)

    perc_list = []
    # start_point = []
    # end_point = []
    perc_cluster_id=None
    def find_percolation_cluster_id():
        for x in range(size):
            if arr_mark_clusters[x][0] > 0:
                cluster_id = arr_mark_clusters[x][0]
                for xx in range(size):
                    if arr_mark_clusters[xx][size - 1] == cluster_id and not (cluster_id in perc_list):
                        perc_list.append(cluster_id)
                        # perc_cluster_id=cluster_id
                        # start_point=[x,0]
                        # end_point = [xx,size-1]
                        return cluster_id
        return None

    def find_longest_from_top_cluster_id():
        for button in range(size-1):
            for x in range(size):
                if arr_mark_clusters[x][0] > 0:
                    cluster_id = arr_mark_clusters[x][0]
                    for xx in range(size):
                        if arr_mark_clusters[xx][size - 1 - button] == cluster_id:
                            return cluster_id
        return None

    # perc_cluster_id=find_percolation_cluster_id()
    perc_cluster_id = find_longest_from_top_cluster_id()

    if False:
        for x in range(size):
            s = ""
            for y in range(size):
                m = arr_mark_clusters[x][y]
                if m == 0:
                    s = s + "  "
                else:
                    sm = str(m).ljust(2)
                    if m in perc_list:
                        s = s + "~ "
                    else:
                        s = s + sm
            print(s)
    return [flatten(rnd_arr), arr_mark_clusters ,perc_cluster_id]


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))




# resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
# sys.setrecursionlimit(10**6)
sys.setrecursionlimit(100000)
for i in range(10000):

    # while True:
    #     (arr_fatten,  arr_mark_clusters,perc_cluster_id)=genetate_percolation_system2d(image_size,0.37)
    #     if(perc_cluster_id is not None):
    #         break
    perc_cluster_id=None
    (arr_fatten,  arr_mark_clusters,perc_cluster_id)=genetate_percolation_system2d(image_size,0.41)

    # PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]
    img1 = Image.new( 'RGB', (image_size ,image_size ), "white") # create a new black image
    pixels1 = img1.load() # create the pixel map


    for i in range(img1.size[0]):    # for every col:
        for j in range(img1.size[1]):    # For every row
            if(arr_mark_clusters[i][j]>0):
                pixels1[i,j] = (0, 0, 0) # set the colour accordingly

    # img1.show()
    s=randomString()

    img1.save(f"perc_train/{s}.jpg","JPEG")


    img2 = Image.new( 'RGB', (image_size ,image_size ), "black") # create a new black image
    pixels2 = img2.load() # create the pixel map

    if perc_cluster_id is not None:
        for i in range(img2.size[0]):    # for every col:
            for j in range(img2.size[1]):    # For every row
                if(arr_mark_clusters[i][j]==perc_cluster_id):
                    pixels2[i,j] = (255, 255, 255) # set the colour accordingly


    img2.save(f"perc_train_mask/{s}_mask.gif","GIF")
    # img2.show()