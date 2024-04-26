from collections import Counter
from genericpath import isfile
import pickle
import os
from os.path import join
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
from utils.utils import get_maj_label, categories_pad, categories_val_dom, categories_val_dom_cert, categories_mini_dialog_act, label_to_one_hot

def combine_labels(label_row):
    return '_'.join(label_row)

def load_data(hubert_latent_dim=24):
    dict = {}
    for set in ["train", "test"]:
        dict[set] = {}
        dict[set]["X_audio"] = []
        dict[set]["X_audio_hubert"] = []
        dict[set]["Y_behaviour"]  = []
        dict[set]["interval"]  = []
        dict[set]["speak_or_not"]  = []
        dict[set]["gender"]  = []
        dict[set]["previous_behaviour"]  = []
        dict[set]["final_Y"]  = []
        dict[set]["final_Y_init"]  = []
        dict[set]["time"]  = []
        dict[set]["keys"]  = []
        dict[set]["raw_keys"]  = []
        dict[set]["raw_dialog_act"] = []
        dict[set]["raw_valence"]  = []
        dict[set]["raw_arousal"]  = []
        dict[set]["raw_certainty"]  = []
        dict[set]["raw_dominance"]  = []
        dict[set]["dialog_act"]  = []
        dict[set]["valence"]  = []
        dict[set]["arousal"]  = []
        dict[set]["certainty"]  = []
        dict[set]["dominance"]  = []
        dict[set]["details_time"]  = []
    
    path = "data/final_data/"
    details_file = "data/details.xlsx"
    details = pd.read_excel(details_file)
    train = details[["nom", "set"]].where(details["set"] == "train").dropna()["nom"].values
    test = details[["nom", "set"]].where(details["set"] == "test").dropna()["nom"].values

    for set in ["train", "test"]:
        if(set == "train"):
            files = train
        else:
            files = test
        for file in files:
            if(isfile(join(path, file + ".p"))):
                with open(join(path, file + ".p"), 'rb') as f:
                    data = pickle.load(f)
                for value_hubert_seq in data["hubert_array"]:
                    dict[set]["X_audio_hubert"].extend(value_hubert_seq[hubert_latent_dim])  
                dict[set]["X_audio"].extend(data["prosody_array"])
                dict[set]["Y_behaviour"].extend(data["behaviour_array"])
                dict[set]["details_time"].extend(data["details_time"])
                dict[set]["interval"].extend(data["time_array"])
                dict[set]["speak_or_not"].extend(data["speak_or_not"])
                dict[set]["gender"].extend([ele for ele in [data["gender"]] for _ in range(len(data["behaviour_array"]))])
                dict[set]["previous_behaviour"].extend(data["previous_behaviour"])
                dict[set]["final_Y_init"].append(data["final_behaviour_init"])
                dict[set]["final_Y"].append(data["final_behaviour"])
                dict[set]["time"].extend(data["details_time"])
                dict[set]["keys"].extend([ele for ele in [data["key"]] for _ in range(len(data["behaviour_array"]))])
                dict[set]["raw_keys"].append(data["key"])
                dict[set]["dialog_act"].extend([get_maj_label(ele) for ele in data["dialog_act"]])
                dict[set]["raw_dialog_act"].extend(data["dialog_act"])
                dict[set]["valence"].extend([get_maj_label(ele) for ele in data["valence"]])
                dict[set]["raw_valence"].extend(data["valence"])
                dict[set]["arousal"].extend([get_maj_label(ele) for ele in data["arousal"]])
                dict[set]["raw_arousal"].extend(data["arousal"])
                dict[set]["certainty"].extend([get_maj_label(ele) for ele in data["certainty"]])
                dict[set]["raw_certainty"].extend(data["certainty"])
                dict[set]["dominance"].extend([get_maj_label(ele) for ele in data["dominance"]])
                dict[set]["raw_dominance"].extend(data["dominance"])
                del data

        dict[set]["speak_or_not"] = torch.as_tensor(list(map(list, dict[set]["speak_or_not"])))
        dict[set]["one_hot_tensor_dialog_act"] = torch.stack([label_to_one_hot(label, "dialog_act") for label in dict[set]["dialog_act"]])
        dict[set]["one_hot_tensor_valence"] = torch.stack([label_to_one_hot(label, "valence") for label in dict[set]["valence"]])
        dict[set]["one_hot_tensor_arousal"] = torch.stack([label_to_one_hot(label, "arousal") for label in dict[set]["arousal"]])
        dict[set]["one_hot_tensor_certainty"] = torch.stack([label_to_one_hot(label, "certainty") for label in dict[set]["certainty"]])
        dict[set]["one_hot_tensor_dominance"] = torch.stack([label_to_one_hot(label, "dominance") for label in dict[set]["dominance"]])
        dict[set]["gender"] = [dict[set]["gender"][i] if value != "silence" else value for i, value in enumerate(dict[set]["valence"])]
        dict[set]["one_hot_tensor_gender"] = torch.stack([label_to_one_hot(label, "gender") for label in dict[set]["gender"]])
        dict[set]["speak_or_not_repeat"] = torch.repeat_interleave(dict[set]["speak_or_not"], 2, dim=1)
        dict[set]["X_audio"] = torch.as_tensor(np.array(dict[set]["X_audio"])[:,:,np.r_[1,2]])
        dict[set]["X_audio"] = torch.concatenate((dict[set]["speak_or_not"].unsqueeze(2), dict[set]["X_audio"]), dim=2)
        dict[set]["X_audio_hubert"] = torch.stack(dict[set]["X_audio_hubert"])
        dict[set]["X_audio_hubert"] = torch.concatenate((dict[set]["speak_or_not_repeat"].unsqueeze(2), dict[set]["X_audio_hubert"]), dim=2)
        
        dict[set]["Y_behaviour"] = torch.as_tensor(np.array(dict[set]["Y_behaviour"]))

        label_pad = np.stack((dict[set]["valence"], dict[set]["arousal"], dict[set]["dominance"]), axis=1)
        combined_labels_array_ref = []
        for value in label_pad:
            combined_labels_array_ref.append(combine_labels(value))
        dict[set]["pad"] = [value if value in categories_pad else "Other" for value in combined_labels_array_ref]
        dict[set]["one_hot_tensor_pad"] = torch.stack([label_to_one_hot(label, "pad") for label in dict[set]["pad"]])

        label_val_dom = np.stack((dict[set]["valence"], dict[set]["dominance"]), axis=1)
        combined_labels_array_ref = []
        for value in label_val_dom:
            combined_labels_array_ref.append(combine_labels(value))
        dict[set]["val_dom"] = [value if value in categories_val_dom else "Other" for value in combined_labels_array_ref]
        dict[set]["one_hot_tensor_val_dom"] = torch.stack([label_to_one_hot(label, "val_dom") for label in dict[set]["val_dom"]])

        label_val_dom_cert = np.stack((dict[set]["valence"], dict[set]["dominance"], dict[set]["certainty"]), axis=1)
        combined_labels_array_ref = []
        for value in label_val_dom_cert:
            combined_labels_array_ref.append(combine_labels(value))
        dict[set]["val_dom_cert"] = [value if value in categories_val_dom_cert else "Other" for value in combined_labels_array_ref]
        dict[set]["one_hot_tensor_val_dom_cert"] = torch.stack([label_to_one_hot(label, "val_dom_cert") for label in dict[set]["val_dom_cert"]])

        dict[set]["mini_dialog_act"] = [value if value in categories_mini_dialog_act else "Other" for value in dict[set]["dialog_act"]]
        dict[set]["one_hot_tensor_mini_dialog_act"] = torch.stack([label_to_one_hot(label, "mini_dialog_act") for label in dict[set]["mini_dialog_act"]])
        
    return dict