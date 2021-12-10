from networkx.algorithms import mis
import numpy as np
import os
import random
import pandas as pd
import networkx
import itertools
random.seed(1234)


data_root = '/media/group1/data/tianhang/sorted_samples_pn_2048/'
train_root = '/media/group1/data/tianhang/MD/'
test_root = '/media/group1/data/tianhang/MD/'


def proprecess_data(data_root, train_root, test_root):
    if not os.path.exists(train_root):
        os.makedirs(train_root)
    if not os.path.exists(test_root):
        os.makedirs(test_root)

    all_file_list = os.listdir(data_root)
    random.shuffle(all_file_list)
    train_number = int(len(all_file_list) * 0.7)
    train_file_list = all_file_list[:train_number]
    val_file_list = all_file_list[train_number:]
    generate_corresponding_txt(train_file_list, train_root, module="train")
    generate_corresponding_txt(val_file_list, test_root, module="valid")
    print("Start to process training set!")        
    graph_edges = generate_graph_from_list(range(19))

    for number, i in enumerate(train_file_list):
        filled_before = np.load(os.path.join(data_root, i), allow_pickle=True).item()
        frames = filled_before["location"].shape[0]
        maskers = []

        filled_before_location = filled_before['location']
        for t in range(frames):
            mask_t = filled_before_location[t,:,1:5].sum(-1) != 0.
            maskers.append(mask_t)                                                  
        maskers_np = np.array(maskers)
        graph_weight = generate_weight(filled_before_location, graph_edges, maskers_np)
        filled_before["graph_index"] = graph_edges   
        filled_before["graph_weight"] = graph_weight 
        filled_before["maskers"] = maskers_np
        np.save(os.path.join(train_root, i), filled_before)
        print("The Training process has been {:.2f} % done!".format((number + 1) / len(train_file_list) * 100))
    
    print("#" * 30)
    print("Start to process Testing set!")
    for val_number, j in enumerate(val_file_list):
 
        val_infor = np.load(os.path.join(data_root, j), allow_pickle=True).item()
        frames = val_infor["location"].shape[0]
        val_maskers = []
        val_location = val_infor["location"] 
        for t in range(frames):
            val_mask_t = val_location[t,:,1:5].sum(-1) != 0.
            val_maskers.append(val_mask_t)

        val_maskers_np = np.array(val_maskers)
        val_infor["maskers"] = val_maskers_np
        np.save(os.path.join(test_root, j), val_infor)
        print("The Testing process has been {:.2f} % done!".format((val_number + 1) / len(val_file_list) * 100))

def generate_weight(location, graph_edges, mask):
    # location: frames x 19 x 16
    # graph_weight frames x num_edge

    weights = np.zeros((location.shape[0], len(graph_edges), ), dtype=np.float32)
    # normolize
    location_n = location.copy()
    location_n[:, :, [1,3]] /= 1280
    location_n[:, :, [2,4]] /= 720

    for i, edge in enumerate(graph_edges):

        c1 = [0.5 * (location_n[:, edge[0], 1] + location_n[:, edge[0], 3]),
              0.5 * (location_n[:, edge[0], 2] + location_n[:, edge[0], 4])]
        c2 = [0.5 * (location_n[:, edge[1], 1] + location_n[:, edge[1], 3]),
              0.5 * (location_n[:, edge[1], 2] + location_n[:, edge[1], 4])]
            
        d = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2

        md = get_motion_distance(edge[0], edge[1], location)

        weights[:, i] = np.exp(-(0.7*d + 0.3*md))
        frame_index = ((mask[:, edge[0]]==True) & (mask[:, edge[1]] == True))
        weights[frame_index==False, i] = 0

    # normalize
    weights_nor = (weights - np.min(weights, axis=1, keepdims=True)) / (np.max(weights, axis=1, keepdims=True) - np.min(weights, axis=1, keepdims=True))
    # account for NaN
    weights_nor = np.nan_to_num(weights_nor)

    return weights_nor

def get_motion_distance(id_0, id_1, location_all):
    md = []
    center_point = location_all[:,:,3:-1] - location_all[:,:,1:3]
    for t in range(100):
        if t <= 4:
            d = get_normolize_distance(id_0, id_1, 0, t, center_point)
        else:
            d = get_normolize_distance(id_0, id_1, t-4, t, center_point)
        md.append(d)

    md = np.array(md)
    return md

def get_normolize_distance(id_0, id_1, t1, t2, center_point):

    v_id_0 = normalize(((center_point[t1, id_0, 0], center_point[t1, id_0, 1]),(center_point[t2, id_0, 0], center_point[t2, id_0, 1])))
    v_id_1 = normalize(((center_point[t1, id_1, 0], center_point[t1, id_1, 1]),(center_point[t2, id_1, 0], center_point[t2, id_1, 1])))

    # Eud_distance
    d_1 = ((v_id_0[1][0] - v_id_1[1][0])**2 + (v_id_0[1][1] - v_id_1[1][1])**2)**0.5 #箭头距离
    d_2 = ((v_id_0[0][0] - v_id_1[0][0])**2 + (v_id_0[0][1] - v_id_1[0][1])**2)**0.5 #箭尾距离

    md = d_1 - d_2

    return md

def normalize(v):
    delta_x = v[1][0] - v[0][0]
    delta_y = v[1][1] - v[0][1]
    length = (delta_x**2 + delta_y**2)**0.5
    if length != 0:
        nor_delta_x = delta_x/length
        nor_delta_y = delta_y/length
        return ((v[0][0], v[0][1]),(v[0][0]+nor_delta_x, v[0][1]+nor_delta_y))
    else:
        return v  

def generate_graph_from_list(L, create_using=None):
    G = networkx.empty_graph(len(L), create_using)
    if len(L) > 1:
        if G.is_directed():
            edges = itertools.permutations(L,2)
        else:
            edges = itertools.combinations(L,2)
        G.add_edges_from(edges)
    graph_edges = list(G.edges())

    return graph_edges

def generate_corresponding_txt(file_list, root, module="train"):
    """
    all valid module is train or valid
    """
    str = '\n'
    f = open(os.path.join(root, module + "_video_list.txt"), "w")
    f.write(str.join(file_list))
    f.close()

proprecess_data(data_root, train_root, test_root)