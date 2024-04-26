from collections import Counter
import torch


# Tableau de labels
categories_dialog_act = ["silence", "Declaration", "Backchannel", "Agree/accept" , "Disagree/disaccept", "Question", "Directive" , "Non-understanding", "Opening", "Apology", "Thanking"]
categories_certainty = ["silence", "Certain", "Neutral", "Uncertainty"]
categories_valence = ["silence", "Positive", "Negative", "Neutral"]
categories_arousal = ["silence", "Active", "Passive", "Neutral"]
categories_dominance = ["silence", "Strong", "Weak", "Neutral"]
categories_gender = ["H", "F", "silence"]

categories_pad = ["silence_silence_silence","Negative_Active_Strong", "Positive_Active_Strong", "Other"]
categories_val_dom = ['silence_silence', 'Negative_Strong', 'Positive_Strong', "Other"]
categories_val_dom_cert = ['silence_silence_silence', 'Negative_Strong_Certain', 'Positive_Strong_Certain', "Other"]
#categories_mini_dialog_act = ["silence", "Declaration", "Agree/accept" , "Disagree/disaccept", "Question", "Directive" , "Non-understanding", "Other"]
categories_mini_dialog_act = ["silence", "Non-understanding", "Disagree/disaccept", "Declaration", "Directive", "Agree/accept", "Question", "Other"]

# Créer les dictionnaires de mapping
label_to_index_dialog_act  = {label: i for i, label in enumerate(sorted(categories_dialog_act))}
index_to_label_dialog_act  = {i: label for label, i in label_to_index_dialog_act.items()}

label_to_index_valence  = {label: i for i, label in enumerate(sorted(categories_valence))}
index_to_label_valence  = {i: label for label, i in label_to_index_valence.items()}

label_to_index_arousal  = {label: i for i, label in enumerate(sorted(categories_arousal))}
index_to_label_arousal  = {i: label for label, i in label_to_index_arousal.items()}

label_to_index_dominance  = {label: i for i, label in enumerate(sorted(categories_dominance))}
index_to_label_dominance  = {i: label for label, i in label_to_index_dominance.items()}

label_to_index_certainty  = {label: i for i, label in enumerate(sorted(categories_certainty))}
index_to_label_certainty  = {i: label for label, i in label_to_index_certainty.items()}

label_to_index_gender  = {label: i for i, label in enumerate(sorted(categories_gender))}
index_to_label_gender  = {i: label for label, i in label_to_index_gender.items()}

label_to_index_pad  = {label: i for i, label in enumerate(sorted(categories_pad))}
index_to_label_pad = {i: label for label, i in label_to_index_pad.items()}

label_to_index_val_dom  = {label: i for i, label in enumerate(sorted(categories_val_dom))}
index_to_label_val_dom = {i: label for label, i in label_to_index_val_dom.items()}

label_to_index_val_dom_cert  = {label: i for i, label in enumerate(sorted(categories_val_dom_cert))}
index_to_label_val_dom_cert = {i: label for label, i in label_to_index_val_dom_cert.items()}

label_to_index_mini_dialog_act  = {label: i for i, label in enumerate(sorted(categories_mini_dialog_act))}
index_to_label_mini_dialog_act  = {i: label for label, i in label_to_index_mini_dialog_act.items()}

# Fonction pour convertir les labels en représentation one-hot
def label_to_one_hot(label, type):
    if type == "dialog_act":
        label_to_index = label_to_index_dialog_act
    elif type == "valence":
        label_to_index = label_to_index_valence
    elif type == "arousal":
        label_to_index = label_to_index_arousal
    elif type == "dominance":
        label_to_index = label_to_index_dominance
    elif type == "certainty":
        label_to_index = label_to_index_certainty
    elif type == "gender":
        label_to_index = label_to_index_gender
    elif type == "pad":
        label_to_index = label_to_index_pad
    elif type == "val_dom":
        label_to_index = label_to_index_val_dom
    elif type == "val_dom_cert":
        label_to_index = label_to_index_val_dom_cert
    elif type == "mini_dialog_act":
        label_to_index = label_to_index_mini_dialog_act
    num_classes = len(label_to_index)
    one_hot = torch.zeros(num_classes)
    one_hot[label_to_index[label]] = 1
    return one_hot

# Fonction pour récupérer le label à partir de la représentation one-hot
def one_hot_to_label(one_hot, type):
    if type == "dialog_act":
        index_to_label = index_to_label_dialog_act
    elif type == "valence":
        index_to_label = index_to_label_valence
    elif type == "arousal":
        index_to_label = index_to_label_arousal
    elif type == "dominance":
        index_to_label = index_to_label_dominance
    elif type == "certainty":
        index_to_label = index_to_label_certainty
    elif type == "gender":
        index_to_label = index_to_label_gender
    elif type == "pad":
        index_to_label = index_to_label_pad
    elif type == "val_dom":
        index_to_label = index_to_label_val_dom
    elif type == "val_dom_cert":
        index_to_label = index_to_label_val_dom_cert
    elif type == "mini_dialog_act":
        index_to_label = index_to_label_mini_dialog_act
    index = torch.argmax(one_hot)
    return index_to_label[index.item()]

def one_hot_to_index(one_hot, type):
    index = torch.argmax(one_hot)
    if type == "dialog_act":
        index_to_label = index_to_label_dialog_act
        return label_to_index_dialog_act[index_to_label[index.item()]]
    elif type == "valence":
        index_to_label = index_to_label_valence
        return label_to_index_valence[index_to_label[index.item()]]
    elif type == "arousal":
        index_to_label = index_to_label_arousal
        return label_to_index_arousal[index_to_label[index.item()]]
    elif type == "dominance":
        index_to_label = index_to_label_dominance
        return label_to_index_dominance[index_to_label[index.item()]]
    elif type == "certainty":
        index_to_label = index_to_label_certainty
        return label_to_index_certainty[index_to_label[index.item()]]
    elif type == "gender":
        index_to_label = index_to_label_gender
        return label_to_index_gender[index_to_label[index.item()]]
    elif type == "pad":
        index_to_label = index_to_label_pad
        return label_to_index_pad[index_to_label[index.item()]]
    elif type == "val_dom":
        index_to_label = index_to_label_val_dom
        return label_to_index_val_dom[index_to_label[index.item()]]
    elif type == "val_dom_cert":
        index_to_label = index_to_label_val_dom_cert
        return label_to_index_val_dom_cert[index_to_label[index.item()]]
    elif type == "mini_dialog_act":
        index_to_label = index_to_label_mini_dialog_act
        return label_to_index_mini_dialog_act[index_to_label[index.item()]]
    
    
def get_labels(type):
    if type == "dialog_act":
        labels = categories_dialog_act
    elif type == "valence":
        labels = categories_valence
    elif type == "arousal":
        labels = categories_arousal
    elif type == "dominance":
        labels = categories_dominance
    elif type == "certainty":
        labels = categories_certainty
    elif type == "gender":
        labels = categories_gender
    elif type == "pad":
        labels = categories_pad
    elif type == "val_dom":
        labels = categories_val_dom
    elif type == "val_dom_cert":
        labels = categories_val_dom_cert
    elif type == "mini_dialog_act":
        labels = categories_mini_dialog_act
    return labels

def get_labels_to_index(type):
    if type == "dialog_act":
        labels = label_to_index_dialog_act
    elif type == "valence":
        labels = label_to_index_valence
    elif type == "arousal":
        labels = label_to_index_arousal
    elif type == "dominance":
        labels = label_to_index_dominance
    elif type == "certainty":
        labels = label_to_index_certainty
    elif type == "gender":
        labels = label_to_index_gender
    elif type == "pad":
        labels = label_to_index_pad
    elif type == "val_dom":
        labels = label_to_index_val_dom
    elif type == "val_dom_cert":
        labels = label_to_index_val_dom_cert
    elif type == "mini_dialog_act":
        labels = label_to_index_mini_dialog_act
    return labels


def get_maj_label(labels):
    # Compter le nombre d'occurrences de chaque label
    label_counts = Counter(labels)
    # Trouver le label majoritaire
    majority_label = max(label_counts, key=label_counts.get)
    # Calculer le pourcentage de la présence du label majoritaire
    percentage_majority = label_counts[majority_label] / len(labels) * 100
    # Vérifier si le label majoritaire est "silence" et s'il représente plus de 75% des labels
    if "silence" in majority_label :
        if percentage_majority < 100:
            majority_label = Counter(labels).most_common(2)[1][0]
            #second_percentage_majority = label_counts[second_majority_label] / len(labels) * 100
    return majority_label