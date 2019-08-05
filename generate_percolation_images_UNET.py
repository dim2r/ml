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
                            start_x = x
                            start_y = 0
                            end_x   = xx
                            end_y   = size - 1 - button
                            return cluster_id,start_x,start_y,end_x,end_y
        return None

    # perc_cluster_id=find_percolation_cluster_id()
    (perc_cluster_id, start_x, start_y, end_x, end_y) = find_longest_from_top_cluster_id()
    # print(perc_cluster_id, start_x, start_y, end_x, end_y)

    arr_mark_wave = np.zeros((size, size)).astype(int)
    def find_perc_path_forward_pass(perc_cluster_id, start_x, start_y, end_x, end_y):
        iteration=1
        end_reached = False
        frontier=[ (start_x, start_y) ]

        while not end_reached and len(frontier)>0:
            new_frontier=[]
            for (x,y) in frontier:
                arr_mark_wave[x][y]=iteration


            for (x,y) in frontier:
                if not end_reached:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            xx = x + dx
                            yy = y + dy
                            if(         xx>=0 and yy>=0 and xx<size and yy<size and (not(dx==0 and dy==0))
                                    and arr_mark_wave[xx][yy]==0
                                    and arr_mark_clusters[xx][yy]==perc_cluster_id
                                    and (not (xx,yy) in new_frontier)
                                    and not end_reached
                            ):

                                new_frontier.append( (xx,yy) )
                                if xx==end_x and yy==end_y:
                                    end_reached=True
                                    arr_mark_wave[xx][yy]=iteration+1

            frontier=new_frontier
            iteration+=1

    arr_mark_path = np.zeros((size, size)).astype(int)

    def find_perc_path_back_pass(perc_cluster_id, start_x, start_y, end_x, end_y):
        x = end_x
        y = end_y
        start_found=False
        interation = arr_mark_wave[x][y]
        arr_mark_path[x][y]=1
        def scan_neibours():
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nonlocal x,y,interation,start_found
                    xx = x + dx
                    yy = y + dy
                    if(  xx>=0 and yy>=0 and xx<size and yy<size and (not(dx==0 and dy==0))
                        and arr_mark_wave[xx][yy]==interation-1
                    ):
                        arr_mark_path[xx][yy]=1
                        interation -= 1
                        if(xx==start_x and yy==start_y):
                            return False
                        else:
                            x=xx
                            y=yy
                            return True
            return True
        while scan_neibours():
            pass

    def printit():
        for x in range(size):
            s = ""
            for y in range(size):
                m = arr_mark_clusters[x][y]
                if m == 0:
                    s +="  "
                else:
                    sm = str(m).ljust(2)
                    add=""
                    if m in perc_list:
                        add= "~ "
                    else:
                        add= sm
                    if arr_mark_path[x][y]>0:
                        add= "$ "
                    s += add
            print(s)
        print('-------------------------')
        for x in range(size):
            s = ""
            for y in range(size):
                m = arr_mark_wave[x][y]
                if m == 0:
                    s +="  "
                else:
                    sm = str(m).ljust(2)
                    s += sm
            print(s)
        print('-------------------------')
        for x in range(size):
            s = ""
            for y in range(size):
                m = arr_mark_path[x][y]
                if m == 0:
                    s +="  "
                else:
                    sm = str(m).ljust(2)
                    s += sm
            print(s)

    find_perc_path_forward_pass(perc_cluster_id, start_x, start_y, end_x, end_y)
    find_perc_path_back_pass(perc_cluster_id, start_x, start_y, end_x, end_y)
    # printit()



    return [flatten(rnd_arr), arr_mark_clusters ,perc_cluster_id, arr_mark_path]


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
    (arr_fatten,  arr_mark_clusters,perc_cluster_id, arr_mark_path)=genetate_percolation_system2d(image_size,0.41)

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


    img2.save(f"perc_train/{s}_cluster.gif","GIF")

    img3 = Image.new( 'RGB', (image_size ,image_size ), "black") # create a new black image
    pixels3 = img3.load() # create the pixel map

    if perc_cluster_id is not None:
        for i in range(img3.size[0]):    # for every col:
            for j in range(img3.size[1]):    # For every row
                if(arr_mark_path[i][j]>0):
                    pixels3[i,j] = (255, 255, 255) # set the colour accordingly


    img3.save(f"perc_train/{s}_path.gif","GIF")
    # img2.show()