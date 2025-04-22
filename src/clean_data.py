# This file contains functions and methods that are used to clean and preprocess your raw data
import pandas as pd
# face detection
import cv2
# path
from pathlib import Path
import os
# Three Forehead and Five Eyes
import dlib
import numpy as np
import csv
from PIL import Image



# create new folder
output_dir = "../data/processed/District_Demographics"
os.makedirs(output_dir, exist_ok = True)

## District Demographics

# poverty
poverty_2018 = pd.read_csv("../data/raw/District_Demographics/poverty/ACSST1Y2018.S1701-Data.csv", skiprows = [1], encoding = "utf-8")
poverty_2020 = pd.read_csv("../data/raw/District_Demographics/poverty/ACSST5Y2020.S1701-Data.csv", skiprows = [1], encoding = "utf-8")
new_columns = ["District", "Poverty (%)"]

poverty_2018 = poverty_2018[["NAME", "S1701_C03_001E"]].head(53).reset_index(drop = True)
poverty_2018.columns = new_columns
poverty_2018["District"] = range(1, 54)
poverty_2020 = poverty_2020[["NAME", "S1701_C03_001E"]].head(53).reset_index(drop = True)
poverty_2020.columns = new_columns
poverty_2020["District"] = range(1, 54)

# income
income_2018 = pd.read_csv("../data/raw/District_Demographics/income/ACSDT1Y2018.B29004-Data.csv", skiprows = [1], encoding = "utf-8")
income_2020 = pd.read_csv("../data/raw/District_Demographics/income/ACSDT5Y2020.B29004-Data.csv", skiprows = [1], encoding = "utf-8")
new_columns = ["District", "Median_Household_Income"]

income_2018 = income_2018[["NAME", "B29004_001E"]].head(53).reset_index(drop = True)
income_2018.columns = new_columns
income_2018["District"] = range(1, 54)
income_2020 = income_2020[["NAME", "B29004_001E"]].head(53).reset_index(drop = True)
income_2020.columns = new_columns
income_2020["District"] = range(1, 54)

# population
population_2018 = pd.read_csv("../data/raw/District_Demographics/population/ACSDT1Y2018.B01001-Data.csv", skiprows = [1], encoding = "utf-8")
population_2020 = pd.read_csv("../data/raw/District_Demographics/population/ACSDT5Y2020.B01001-Data.csv", skiprows = [1], encoding = "utf-8")
new_columns = ["District", "Pop_Total", "Pop_M", "Pop_F"]

population_2018 = population_2018[["NAME", "B01001_001E", "B01001_002E", "B01001_026E"]].head(53).reset_index(drop = True)
population_2018.columns = new_columns
population_2018["Sex Ratio"] = population_2018["Pop_M"] / population_2018["Pop_F"]
population_2018["District"] = range(1, 54)
population_2020 = population_2020[["NAME", "B01001_001E", "B01001_002E", "B01001_026E"]].head(53).reset_index(drop = True)
population_2020.columns = new_columns
population_2020["Sex Ratio"] = population_2020["Pop_M"] / population_2020["Pop_F"]
population_2020["District"] = range(1, 54)

## merge & save the results to csv file
# merge
CA_District_Demographics_2018 = poverty_2018.merge(income_2018, on = "District", how = "outer").merge(population_2018, on = "District", how = "outer")
CA_District_Demographics_2020 = poverty_2020.merge(income_2020, on = "District", how = "outer").merge(population_2020, on = "District", how = "outer")
# create new folder
output_dir = "../data/processed/District_Demographics"
os.makedirs(output_dir, exist_ok = True)
# file path & save
file_path_2018 = os.path.join(output_dir, "CA_District_Demographics_2018.csv")
file_path_2020 = os.path.join(output_dir, "CA_District_Demographics_2020.csv")
CA_District_Demographics_2018.to_csv(file_path_2018, index = False, encoding = "utf-8")
CA_District_Demographics_2020.to_csv(file_path_2020, index = False, encoding = "utf-8")



## facial attractiveness score

# HOG + Linear SVM face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../data/raw/dlib_model/shape_predictor_68_face_landmarks.dat")
# MMOD CNN face detector
detector_CNN = dlib.cnn_face_detection_model_v1("../data/raw/dlib_model/mmod_human_face_detector.dat")
predictor_CNN = dlib.shape_predictor("../data/raw/dlib_model/shape_predictor_68_face_landmarks.dat")

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# two courts
def calculate_two_courts(landmarks):    
    # eyebow_center to lowest_nose
    eyebrow_center = ((landmarks[21] + landmarks[22]) / 2).astype(int)
    midcourt = calculate_distance(eyebrow_center, landmarks[33])
    # lowest_nose to jaw
    lowcourt = calculate_distance(landmarks[33], landmarks[8])
    
    # two courts height
    total_height = midcourt + lowcourt
    
    return [midcourt/total_height, lowcourt/total_height]
    
# five eyes
def calculate_five_eyes(landmarks):   
    left_eye = calculate_distance(landmarks[36], landmarks[39])
    right_eye = calculate_distance(landmarks[42], landmarks[45])
    eye_distance = calculate_distance(landmarks[39], landmarks[42])
    left_face = calculate_distance(landmarks[0], landmarks[36])
    right_face = calculate_distance(landmarks[45], landmarks[16])
    
    # five eyes width    
    face_width = calculate_distance(landmarks[0], landmarks[16])
    
    return [left_face/face_width, left_eye/face_width, eye_distance/face_width, right_eye/face_width, right_face/face_width]

# calculate attractiveness_score
def calculate_attractiveness_score(two_courts, five_eyes):
    full_score = 100
    minus_score = 0

    # ideal ratio
    ideal_two_courts = [0.5, 0.5]
    ideal_five_eyes = [0.2, 0.2, 0.2, 0.2, 0.2]
    # minus score
    minus_score += sum([abs(a-b) * 100 for a, b in zip(two_courts, ideal_two_courts)])    
    minus_score += sum([abs(a-b) * 100 for a, b in zip(five_eyes, ideal_five_eyes)])

    # final score
    final_score = max(0, full_score - int(minus_score))
    
    return final_score

# check image's file extension and convert it to .jpeg
def check_and_convert_image(image_path):
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    file_extension = os.path.splitext(image_path)[1].lower()
    
    if file_extension not in valid_extensions:
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                img = img.convert("RGB")
            new_image_path = os.path.splitext(image_path)[0] + ".jpg"
            img.save(new_image_path, "JPEG")
            return new_image_path
    return image_path

# calculate attractiveness_score for each image: HOG + Linear SVM face detector
def analyze_face(image_path):
    image_path = check_and_convert_image(image_path)
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    landmarks = predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
    
    two_courts = calculate_two_courts(landmarks)
    five_eyes = calculate_five_eyes(landmarks)
    attractiveness_score = calculate_attractiveness_score(two_courts, five_eyes)
    
    return attractiveness_score

# calculate attractiveness_score for each image: MMOD CNN face detector
def analyze_face_CNN(image_path):
    image_path = check_and_convert_image(image_path)
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector_CNN(gray)

    # the first face
    face_rect = dets[0].rect
    
    landmarks = predictor_CNN(gray, face_rect)
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
    
    two_courts = calculate_two_courts(landmarks)
    five_eyes = calculate_five_eyes(landmarks)
    attractiveness_score = calculate_attractiveness_score(two_courts, five_eyes)
    
    return attractiveness_score

# root directory
os.makedirs("../data/processed/candidate_images/CA", exist_ok = True)
root_image_folder = Path("../data/processed/candidate_images/CA")
# get all folders' path in root_directory
folders = [folder for folder in root_image_folder.iterdir() if folder.is_dir()]
# get all images' path in all folders
for folder in folders: 
    # current folder name
    folder_name = folder.name
    
    results = []
    # use iterdir() to iterate over images in this folder
    for image in folder.iterdir():
                
        # current image name = ID (do not include file extension)
        image_name = image.stem
        # current image path
        image_path = str(image)
        # calculate attractiveness_score for each image
        try:
            score = analyze_face(image_path)
            score_CNN = analyze_face_CNN(image_path)
            if score is not None and score_CNN is not None:
                results.append({"ID": image_name, "score": score, "score_CNN": score_CNN}) 
        except Exception as e:
            print(f"Error - {image_path}: {str(e)}")
    
    # save in .csv
    csv_dir = Path("../data/processed/candidate_images/CA")
    csv_path = csv_dir / f"{folder_name}_scores.csv"
    with open(csv_path, "w", newline = "") as csvfile:
        fieldnames = ["ID", "score", "score_CNN"]
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)



## merge & save all data to csv file

# election results
CA_house_election_2018 = pd.read_csv("../data/raw/election_result/CA_house_election_2018.csv", encoding = "utf-8")
CA_house_election_2020 = pd.read_csv("../data/raw/election_result/CA_house_election_2020.csv", encoding = "utf-8")
# candidate data
CA_candidate_data_2018 = pd.read_excel("../data/processed/candidate_data/CA_candidate_data_2018.xlsx")
CA_candidate_data_2020 = pd.read_excel("../data/processed/candidate_data/CA_candidate_data_2020.xlsx")
# attractiveness score
score_2018 = pd.read_csv("../data/processed/candidate_images/CA/2018_scores.csv", encoding = "utf-8")
score_2020 = pd.read_csv("../data/processed/candidate_images/CA/2020_scores.csv", encoding = "utf-8")
# district demographics
CA_District_Demographics_2018 = pd.read_csv("../data/processed/District_Demographics/CA_District_Demographics_2018.csv", encoding = "utf-8")
CA_District_Demographics_2020 = pd.read_csv("../data/processed/District_Demographics/CA_District_Demographics_2020.csv", encoding = "utf-8")
# merge all data
CA_2018 = CA_house_election_2018.merge(CA_candidate_data_2018[["ID", "Gender", "Year_of_Birth", "Edu"]], on = "ID", how = "outer").merge(score_2018, on = "ID", how = "outer").merge(CA_District_Demographics_2018, on = "District", how = "outer")
CA_2020 = CA_house_election_2020.merge(CA_candidate_data_2020[["ID", "Gender", "Year_of_Birth", "Edu"]], on = "ID", how = "outer").merge(score_2020, on = "ID", how = "outer").merge(CA_District_Demographics_2020, on = "District", how = "outer")

# calculate ln vote share
CA_2018["Votes"] = CA_2018["Votes"].str.replace(",", "").astype(float)
CA_2020["Votes"] = CA_2020["Votes"].str.replace(",", "").astype(float)
CA_2018["total_votes"] = CA_2018.groupby("District")["Votes"].transform("sum")
CA_2020["total_votes"] = CA_2020.groupby("District")["Votes"].transform("sum")
CA_2018["vote_share"] = CA_2018["Votes"] / CA_2018["total_votes"]
CA_2020["vote_share"] = CA_2020["Votes"] / CA_2020["total_votes"]
CA_2018["ln_vote_share"] = np.log(CA_2018["vote_share"])
CA_2020["ln_vote_share"] = np.log(CA_2020["vote_share"])
# calculate age
CA_2018["Age"] = CA_2018["Year"] - CA_2018["Year_of_Birth"]
CA_2020["Age"] = CA_2020["Year"] - CA_2020["Year_of_Birth"]

# save
CA_final = pd.concat([CA_2018, CA_2020], ignore_index = True)
CA_final.to_csv("../data/processed/CA_final.csv", index = False, encoding = "utf-8")
