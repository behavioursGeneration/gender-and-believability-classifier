from collections import Counter
import pickle
import os
from os.path import join
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
from utils.utils import get_maj_label, categories_pad, categories_val_dom, categories_val_dom_cert, categories_mini_dialog_act, label_to_one_hot
import sys


CLUSTER="jean-zay"
visual_features = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry",
                "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]


def create_set(ipu_path, video_path, segment_length, overlap_test, timestep, data_details):
    overlap = overlap_test
    for key in data_details.keys():
        if("scene3_confrontation4_prise2" in key):
            continue
        if(data_details[key]["set"] == "test"):
            final_dict = {"key": key, 
                        "wav_array": [], "hubert_array": [], "hubert_french_array":[], 
                        "time_array": [], "details_time": [], 
                        "behaviour_array": [], 
                        "speak_or_not": [], "dialog_act": [], "valence": [], "arousal": [], "certainty": [], "dominance": []}

            df_ipu = pd.read_csv(ipu_path + key + ".csv")
            df_ipu.timestamp = df_ipu.timestamp.astype(float)
            end_time_annotations = df_ipu["timestamp"].iloc[-1]

            df_video = pd.read_csv(video_path + key + ".csv")
            end_time_video = df_video["timestamp"].iloc[-1]


            df_result = df_video.merge(df_ipu, on='timestamp', how='inner')
            end_time_result = df_result["timestamp"].iloc[-1]

            del df_video, df_ipu

            t1, t2 = 0, segment_length
            final_dict["gender"] = data_details[key]["genre"]
            final_dict["role"] = data_details[key]["role"]
            final_dict["set"] = data_details[key]["set"]
            final_dict["attitude"] = data_details[key]["attitude_harceuleur"]
            
            while t2 <= end_time_result:
                first_cut = df_result[df_result["timestamp"] < t2]
                second_cut = first_cut[first_cut["timestamp"] >= t1]

                #speak_or_not
                final_dict["speak_or_not"].append(second_cut["bool_speak"].values)

                #tag
                final_dict["dialog_act"].append(second_cut["dialogAct"].values)
                final_dict["valence"].append(second_cut["valence"].values)
                final_dict["arousal"].append(second_cut["arousal"].values)
                final_dict["certainty"].append(second_cut["certainty"].values)
                final_dict["dominance"].append(second_cut["dominance"].values)

                #behaviour (openface)
                final_dict["behaviour_array"].append(second_cut[visual_features].values)

                #time
                final_dict["time_array"].append((t1,t2))
                final_dict["details_time"].append(second_cut["timestamp"].values)

                t1, t2 = round(t1 + segment_length - overlap,2), round(t2 + segment_length - overlap,2)

            with open(video_path + key + ".p", 'wb') as f:
                pickle.dump(final_dict, f)
            del final_dict


def getPath(link_to_data):
    if(CLUSTER=="jean-zay"):
        dataset_path = "C:/Users/alice/Projets/Data/trueness/trueness_1/"
        video_path = "generated_data/"+link_to_data+"/"
        ipu_path = dataset_path + "annotations/processed/from_jean_zay/ipu_with_tag/align/"
        details_file = dataset_path + "details.xlsx"
        details_df = pd.read_excel(details_file)
        data_details = details_df.set_index("nom").to_dict(orient='index')
    else:
        sys.exit("Error in the cluster name")
    return ipu_path, video_path, data_details

def load_generated_data(link_to_data, create_init_files=False):
    if(create_init_files):
        segment_length = 4 #secondes
        timestep = 0.04
        overlap_test = round(0.1 * segment_length,2) 
        ipu_path, video_path, data_details = getPath(link_to_data)
        create_set(ipu_path, video_path, segment_length, overlap_test, timestep, data_details)

    dict = {}
    dict = {}
    dict["Y_behaviour"]  = []
    dict["speak_or_not"]  = []
    dict["gender"]  = []
    dict["keys"]  = []
    dict["raw_keys"]  = []
    dict["raw_dialog_act"] = []
    dict["raw_valence"]  = []
    dict["raw_arousal"]  = []
    dict["raw_certainty"]  = []
    dict["raw_dominance"]  = []
    dict["dialog_act"]  = []
    dict["valence"]  = []
    dict["arousal"]  = []
    dict["certainty"]  = []
    dict["dominance"]  = []
    dict["interval"] = []
    dict["details_time"]  = []

    for file in os.listdir(video_path):
        if(".p" in file):
            with open(join(video_path, file), 'rb') as f:
                data = pickle.load(f)
            dict["Y_behaviour"].extend(data["behaviour_array"])
            dict["details_time"].extend(data["details_time"])
            dict["interval"].extend(data["time_array"])
            dict["speak_or_not"].extend(data["speak_or_not"])
            dict["gender"].extend([ele for ele in [data["gender"]] for _ in range(len(data["behaviour_array"]))])
            dict["keys"].extend([ele for ele in [data["key"]] for _ in range(len(data["behaviour_array"]))])
            dict["raw_keys"].append(data["key"])
            dict["dialog_act"].extend([get_maj_label(ele) for ele in data["dialog_act"]])
            dict["raw_dialog_act"].extend(data["dialog_act"])
            dict["valence"].extend([get_maj_label(ele) for ele in data["valence"]])
            dict["raw_valence"].extend(data["valence"])
            dict["arousal"].extend([get_maj_label(ele) for ele in data["arousal"]])
            dict["raw_arousal"].extend(data["arousal"])
            dict["certainty"].extend([get_maj_label(ele) for ele in data["certainty"]])
            dict["raw_certainty"].extend(data["certainty"])
            dict["dominance"].extend([get_maj_label(ele) for ele in data["dominance"]])
            dict["raw_dominance"].extend(data["dominance"])
  
    dict["speak_or_not"] = torch.as_tensor(list(map(list, dict["speak_or_not"])))
    dict["one_hot_tensor_dialog_act"] = torch.stack([label_to_one_hot(label, "dialog_act") for label in dict["dialog_act"]])
    dict["one_hot_tensor_valence"] = torch.stack([label_to_one_hot(label, "valence") for label in dict["valence"]])
    dict["one_hot_tensor_arousal"] = torch.stack([label_to_one_hot(label, "arousal") for label in dict["arousal"]])
    dict["one_hot_tensor_certainty"] = torch.stack([label_to_one_hot(label, "certainty") for label in dict["certainty"]])
    dict["one_hot_tensor_dominance"] = torch.stack([label_to_one_hot(label, "dominance") for label in dict["dominance"]])
    dict["gender"] = [dict["gender"][i] if value != "silence" else value for i, value in enumerate(dict["valence"])]
    dict["one_hot_tensor_gender"] = torch.stack([label_to_one_hot(label, "gender") for label in dict["gender"]])
    dict["speak_or_not_repeat"] = torch.repeat_interleave(dict["speak_or_not"], 2, dim=1)
    
    dict["Y_behaviour"] = torch.as_tensor(np.array(dict["Y_behaviour"]))

    return dict
