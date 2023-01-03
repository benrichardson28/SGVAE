#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pdb

def property_df(class_type='cluster',scale=True):
    ballname = {0:'empty',1:'baseball_1',
              2:'red_squishy',3:'blue_green_squishy',4:'golfball_1',
              5:'pingpong_1',6:'purple_spiky',7:'tennis_1',
              8:'globe_squishy',9:'foam_1',10:'purple_squash_1',
              11:'tennis_2',12:'yellow_hockey',13:'golfball_2',
              14:'purple_squash_2',15:'tennis_3',16:'baseball_2',
              17:'foam_2',18:'lacrosse',19:'foam_3',
              20:'foam_4',21:'purple_squash_3',
              #new balls
              31:'inflate_globe',32:'yellow_soft_l',33:'yellow_soft_m',
              34:'hard_plastic_l',35:'beige_1',
              36:'smooth_foam_m',37:'smooth_foam_xl',38:'purple_resistance',
              39:'orange_foam_stf',40:'purple_foam_stf',
              41:'yellow_soft_s',42:'smooth_foam_l',43:'smooth_foam_s',
              44:'blue_resistance',45:'yellow_foam_stf',46:'yellow_soft_xs',#47:'hard_plastic_l',
              #new new
              51:'salt_s',52:'salt_m',53:'salt_l',54:'salt_xl',
              55:'popcrn_s',56:'popcrn_m',57:'popcrn_l',58:'popcrn_xl',
              59:'stone_s',60:'stone_m',61:'stone_l',62:'stone_xl',
              63:'cork_miner',64:'yellow_soft_popcrn',
              #
              71:'wt_fm_cb',72:'beige_fm_cb',73:'yellow_fm_cb',74:'owl',
              75:'duck',76:'squirrel',77:'lego_tire',78:'blue_star',
              79:'wood_box_s',80:'wood_box_m',81:'pencil_sharp',82:'gum',
              83:'eco_cb',84:'popcorn_bag',85:'wood_box_l',86:'dense_wood_block',
              87:'paper_clip_box',88:'taurus_seg',89:'lg_cardboard_box',90:'tape_roll',
              95:'wood_box_s_bear',96:'wood_box_m_pop',97:'wood_box_l_pop',
              }

    #size in m
    sq_size = {0: 0.000, 1:0.07209, 2:0.101, 3:0.103, 4:0.04275,
        5:0.0396, 6:0.063, 7:0.0653, 8:0.0643, 9:0.0618,
        10:0.05565, 11:0.0653, 12:0.0498, 13:0.04275, 14:0.05565,
        15:0.0653, 16:0.07209, 17:0.0618, 18:0.0633, 19:0.0618,
        20:0.0618, 21:0.05565, 31:0.068, 32:0.090, 33:0.080, 34:0.100,
        35:0.0348, 36:0.057, 37:0.098, 38:0.050, 39:0.041, 40:0.04015,
        41:0.064, 42:0.078, 43:0.0387, 44:0.050, 45:0.041, 46:0.055,
        51:.0502,52:.060,53:.070,54:.080,
        55:.0502,56:.060,57:.070,58:.080,
        59:.0502,60:.060,61:.070,62:.080,
        63:.060,64:.070,
        # for these next ones, size has to be where the hand positions to slide.
        # But actual size in classification should be based on dimension perpendicular to grip.
        71:.080, 72:.060, 73:.060, 74:.075, 75:.065,
        76:.065, 77:.05, 78:.065, 79:0.03, 80:0.05,
        81:.060, 82:.078, 83:.03, 84:.075, 85:0.07,
        86:.035, 87:.045, 88:.06, 89:.085, 90:.03,
        95:.03,96:.05,97:.07,}

    pr_size = {0: 0.000, 1:0.07209, 2:0.101, 3:0.103, 4:0.04275,
        5:0.0396, 6:0.063, 7:0.0653, 8:0.0643, 9:0.0618,
        10:0.05565, 11:0.0653, 12:0.0498, 13:0.04275, 14:0.05565,
        15:0.0653, 16:0.07209, 17:0.0618, 18:0.0633, 19:0.0618,
        20:0.0618, 21:0.05565, 31:0.068, 32:0.090, 33:0.080, 34:0.100,
        35:0.0348, 36:0.057, 37:0.098, 38:0.050, 39:0.041, 40:0.04015,
        41:0.064, 42:0.078, 43:0.0387, 44:0.050, 45:0.041, 46:0.055,
        51:.0502,52:.060,53:.070,54:.080,
        55:.0502,56:.060,57:.070,58:.080,
        59:.0502,60:.060,61:.070,62:.080,
        63:.060,64:.070,
        # for these next ones, size has to be where the hand positions to slide.
        # But actual size in classification should be based on dimension perpendicular to grip.
        71:.080, 72:.05 ,73:.065, 74:.06, 75:.065,
        76:.085, 77:.03, 78:.032, 79:.030, 80:.050,
        81:.052, 82:.05, 83:.03,84:.075, 85:.070,
        86:.050, 87:.040, 88:.050, 89:.085, 90:.050,
        95:.03,96:.05,97:.07,
        }

    #stiffnes mean and variance
    stiffness = {0:0, 1:20654, 2:527, 3:1463, 4:37888, 5:17579,
          6:1486, 7:6035, 8:1592, 9:21155, 10:5681,
          11:7066, 12:7123, 13:37099, 14:5426, 15:5540,
          16:23024, 17:23694, 18:16330, 19:22026, 20:21952, 21:5674,
          31:911, 32:1049, 33:763, 34:22129, 35:4585, 36:15455,
          37:11165, 38:829, 39:5459, 40:6085, 41:853, 42:9304,
          43:11697, 44:1088, 45:4805, 46:601,
          51:33637, 52:33238, 53:33108, 54:30926,
          55:34424, 56:32942, 57:33182, 58:30360,
          59:34139, 60:33166, 61:32810, 62:31460,
          63:20496, 64:390,
          71:18745, 72:924, 73:1038, 74:526 ,75:2998,
          76:391, 77:3025, 78:4041, 79:39289, 80:40750,
          81:32312, 82:14340, 83:2195, 84:3917, 85:36638,
          86:38851, 87:8987, 88:22729, 89:4244, 90:16809,
          95:39289, 96:40750, 97:36638,
          }
    #extras: 75 alternate direction - 3558+-54

    st_var = {0:0, 1:904, 2:19, 3:33, 4:2239, 5:920,
          6:29, 7:281, 8:21, 9:5256, 10:101,
          11:425, 12:585, 13:1964, 14:122, 15:268,
          16:1472, 17:4646, 18:624, 19:5611, 20:4275, 21:125,
          31:8.3, 32:14, 33:13, 34:1243, 35:327, 36:2262,
          37:1312, 38:15, 39:337, 40:335, 41:11, 42:955,
          43:1725, 44:13, 45:323, 46:4,
          51:1676, 52:2288, 53:1945, 54:1303,
          55:2224, 56:1497 ,57:1420, 58:1764,
          59:1300, 60:1767, 61:1369 ,62:1813,
          63:884, 64:4,
          71:2818, 72:26, 73:20, 74:33, 75:67,
          83:42, 77:80, 78:112, 79:1823, 80:2666,
          81:1688, 82:965, 76:23, 84:1062, 85:2166,
          86:2112, 87:838, 88:2389, 89:312, 90:622,
          95:1823, 96:2666, 97:2166,
          }


    #grams
    mass = {0:0, 1:144.9, 2:11.5, 3:58.0, 4:45.8, 5:3.1,
          6:39.6, 7:56.6, 8:21.0, 9:3.7, 10:41.2,
          11:55.1, 12:7.3, 13:45.7, 14:41.0, 15:55.6,
          16:145.2, 17:3.6, 18:143.6, 19:3.7, 20:3.8,
          21:40.4,
          31:32.8, 32:23.7, 33:16.3, 34:29.6, 35:1.7,
          36:2.3, 37:9.9, 38:57.7, 39:2.8, 40:3.0,
          41:8.4, 42:4.6, 43:0.6, 44:57.7, 45:2.5,
          46:5.2,
          51:36.9,52:36.9,53:36.9,54:36.9,
          55:37.0,56:36.9,57:36.9,58:36.9,
          59:36.6,60:36.8,61:37.0,62:36.9,
          63:28.5,64:37.0,
          71:9.2, 72:3.3, 73:13.3, 74:26.9, 75:34.4,
          83:32.4, 77:26.3, 78:21.7, 79:7.3, 80:26,
          81:41.7, 82:84.8, 76:28.6, 84:162.4, 85:54.2,
          86:61.7, 87:31.3, 88:6.6, 89:38.9, 90:29.4,
          95:17.9, 96:53.4, 97:75.6,
          }


    #contents
    contents = {0:None,1:None,2:None,3:None,4:None,5:None,
           6:'loose',7:None,8:None,9:None,10:None,11:None,
           12:None,13:None,14:None,15:None,16:None,17:None,
           18:None,19:None,20:None,21:None,
           31:None,32:None,33:None,34:None,35:None,
           36:None,37:None,38:None,39:None,40:None,
           41:None,42:None,43:None,44:None,45:None,46:None,
           # lower number is less mass of contents
           51:'salt_4',52:'salt_3',53:'salt_2',54:'salt_1',
           55:'popcorn_4',56:'popcorn_3',57:'popcorn_2',58:'popcorn_1',
           59:'rocks_4',60:'rocks_3',61:'rocks_2',62:'rocks_1',
           63:None,64:'popcorn_4',
           71:None,72:None,73:None,74:None,75:None,
           76:None,77:None,78:None,79:None,80:None,
           81:None,82:'gum',83:None,84:'popcorn',85:None,
           86:None,87:'paper_clips',88:None,89:None,90:None,
           95:'bearings', 96:'popcorn', 97:'popcorn'
           }

    # def friction(obj_id):
    #     fr = {0:0,1:,2:,3:,4:,5:,6:,7:,8:,9:,10:,11:,
    #           12:,13:,14:,15:,16:,17:,18:,19:,20:,21:,
    #           31:,32:,33:,34:,35:,36:,37:,38:,39:,40:,
    #           41:,42:,43:,44:,45:,46:,
    #           51:,52:,53:,54:,
    #           55:,56:,57:,58:,
    #           59:,60:,61:,62:,
    #           63:,64:}
    #     return fr[obj_id]

    # def hardness(obj_id):  #shore hardness
    #     hd = {0:0,1:,2:,3:,4:,5:,6:,7:,8:,9:,10:,11:,
    #           12:,13:,14:,15:,16:,17:,18:,19:,20:,21:,
    #           31:,32:,33:,34:,35:,36:,37:,38:,39:,40:,
    #           41:,42:,43:,44:,45:,46:,
    #           51:,52:,53:,54:,
    #           55:,56:,57:,58:,
    #           59:,60:,61:,62:,
    #           63:,64:}
    #     return hd[obj_id]

    obj_prop = pd.DataFrame(columns = ['ball_id','ball_name','sq_size','pr_size','stiffness','mass','contents'])
    for key in ballname.keys():
        obj_prop = obj_prop.append({'ball_id':key,
                                    'ball_name':ballname[key],
                                    'sq_size':sq_size[key],
                                    'pr_size':pr_size[key],
                                    'stiffness':stiffness[key],
                                    'mass':mass[key],
                                    'contents':contents[key]},ignore_index=True)
    cnt_tps = obj_prop['contents'].unique()
    obj_prop['contents_fine_label'] = [np.where(obj_prop['contents'].iloc[i]==cnt_tps)[0].item() \
                                     for i in range(len(obj_prop))]
    # cnt_tps = pd.unique([i[:4] for i in cnt_tps])
    # obj_prop['contents_rough_label'] = [np.where(cnt_tps==obj_prop['contents'].iloc[i][:4])[0].item() \
    #                                     for i in range(len(obj_prop))]
    obj_prop['contents_binary_label'] = [(i is not None) for i in obj_prop['contents']]
    if scale:
        for prop in ['sq_size','pr_size','stiffness','mass']:
            obj_prop[prop] /= obj_prop[prop].max()

    if class_type == 'cluster':
        # do some sort of cluster
        kmeans = KMeans(
                init="random",
                n_clusters=10,
                n_init=20,
                max_iter=300,
                random_state=0
            )
        for att in ['sq_size','pr_size','mass','stiffness']:
            kmeans.fit(obj_prop[att].values.reshape(-1,1))
            obj_prop[f'{att}_cluster_label'] = kmeans.labels_
            obj_prop[f'{att}_cluster_mean']=kmeans.cluster_centers_[kmeans.labels_]
        obj_prop['index_ball']=obj_prop.index
    return obj_prop.set_index('ball_id')



def get_num_classes(label,properties):
    if 'contents' in label:
        cnt = properties[f'{label}_label'].unique().shape[0]
    elif label in ['sq_size','pr_size','mass','stiffness']:
        cnt = properties[f'{label}_cluster_label'].unique().shape[0]
    else:
        cnt = properties['ball_id'].unique().shape[0]
    return cnt
