import os

# setup path of the directories
pos_path = os.path.join("C:/Users/Boran/Desktop/Semester 5/Practical Work in AI/data/", "positive")
neg_path = os.path.join("C:/Users/Boran/Desktop/Semester 5/Practical Work in AI/data/", "negative")
anc_path = os.path.join("C:/Users/Boran/Desktop/Semester 5/Practical Work in AI/data/", "anchor")

# create directories
if os.path.exists(pos_path) is False:
    os.makedirs(pos_path)
if os.path.exists(neg_path) is False:
    os.makedirs(neg_path)
if os.path.exists(anc_path) is False:
    os.makedirs(anc_path)

test = os.listdir(neg_path)
# copying Labelled Faces in the Wild Dataset to negative example folder
if len(os.listdir(neg_path)) == 0:
    for all_dirs in os.listdir("lfw"):
        for file in os.listdir(f"lfw/{all_dirs}"):
            existing_path = os.path.join("lfw", all_dirs, file)
            new_path = os.path.join(neg_path, file)
            os.replace(existing_path, new_path)