# The Impact of Facial Attractiveness on Candidates’ Vote Share: Case Study of U.S. House Elections

This is the final project for DSCI-510 course.

This project examines the influence of facial attractiveness on voting outcomes in California’s 2018 and 2020 U.S. House elections.  

Please install all required librearies and read the following detailed instuctions about how to run this project.  

## Requirements

Please install the following Python libraries:  

requests, beautifulsoup4, pandas, opencv-python (cv2), dlib, numpy, Pillow (PIL), matplotlib, seaborn, statsmodels, stargazer  

## Instructions

This project inludes 4 parts: get_data, clean_data, analyze_data, and visualize_results.  

### 1. get_data.py

(1) The project involves web-scraping two Wikipedia pages to obtain U.S. House of Representatives election results for California in 2018 and 2020.  

&emsp;[U.S. House of Representatives election results for California in 2018](https://en.wikipedia.org/wiki/2018_United_States_House_of_Representatives_elections_in_California)  
&emsp;[U.S. House of Representatives election results for California in 2020](https://en.wikipedia.org/wiki/2020_United_States_House_of_Representatives_elections_in_California)  

(2) After running get_data.py, the collected data is stored as follows:  

&emsp;A. Election results: "final-project-Sylvie515/data/raw/election_result"  

&emsp;&emsp;(A) Two CSV files containing election results for California's House elections in 2018 and 2020  

&emsp;&emsp;(B) Information includes: state, district, candidate name, party, votes, photo URL, incumbent status, and year  
    
&emsp;B. Candidate photos: "final-project-Sylvie515/data/raw/candidate_images/CA"  

&emsp;&emsp;(A) Photos downloaded using URLs from the CSV files  

&emsp;&emsp;(B) Stored in separate folders for 2018 and 2020  

&emsp;&emsp;(C) Thumbnail URLs have been corrected to download original photos  

### 2. clean_data.py

Before running clean_data.py, ensure the following data is correctly processed and placed in folders.  

(1) District Demographics  

&emsp;A. Download raw .csv files from U.S. Census Bureau Website and save in "final-project-Sylvie515/data/raw/District_Demographics"  

&emsp;&emsp;(A) Includes poverty, income, and population data  

&emsp;&emsp;&emsp;  - Poverty (% below poverty level): ACS S1701 Table  

&emsp;&emsp;&emsp;  - Income (median household income): ACS B29004 Table  

&emsp;&emsp;&emsp;  - Population (total population & gender): ACS B01001 Table  

&emsp;&emsp;(B) Use ACS 1-year estimates for 2018 and 5-year estimates for 2020, as 1-year estimates are not available for 2020.  

&emsp;&emsp;&emsp;  While 5-year estimates are more reliable, they may smooth out some of the pandemic's impact.  
        
&emsp;B. After running clean_data.py,   

&emsp;&emsp;(A) Two csv files: CA_District_Demographics_2018.csv & CA_District_Demographics_2020.csv  

&emsp;&emsp;(B) Processed data will be in "final-project-Sylvie515/data/processed/District_Demographics"  

(2) Facial Attractiveness Score  

In order to perform face detection with the dlib library using both HOG + Linear SVM face detector and MMOD CNN face detector,  

&emsp;A. Download pre-trained face detector files and save in "final-project-Sylvie515/data/raw/dlib_model"  

&emsp;&emsp;(A) shape_predictor_68_face_landmarks.dat  

&emsp;&emsp;(B) mmod_human_face_detector.dat  

&emsp;B. Updated the candidate photos and saved in "final-project-Sylvie515/data/processed/candidate_images/CA"  

&emsp;&emsp;The 68 points Face landmark Detection is primarily suitable for near frontal photos, so for photos with face angle deviating significantly 
&emsp;&emsp;from frontal view, poor image resolution, or partially obscured, I manually put alternative photos as replacements.  

Calculated facial attractiveness scores are saved as 2018_scores.csv and 2020_scores.csv in "final-project-Sylvie515/data/processed/candidate_images/CA"  

(3) Candidate Data  

Candidate data is scattered across various sources, such as Wikipedia, Ballotpedia, Vote Smart, JoinCalifornia, BallotReady, and other news websites.  

So I manually orginized the data and saved as two .csv files in "final-project-Sylvie515/data/processed/candidate_data"  

(4) Combined Data  

&emsp;A. All processed data combined with election results would be saved as CA_final.csv in "final-project-Sylvie515/data/processed"  

&emsp;B. Final file: CA_final.csv in "final-project-Sylvie515/data/processed"  

Ensure all folders are correctly placed for clean_data.py to run properly.  

### 3. analyze_data.py

(1) Retained only the necessary variables required for analysis and saved as "final-project-Sylvie515/results/alldata_CA.csv"  

&emsp;A. Removing irrelevant columns  

&emsp;B. Generating dummy variables  

&emsp;C. Removed samples with missing values in personal characteristic data  

(2) Descriptive statistics saved in "final-project-Sylvie515/results/descriptive_statistics" folder  

(3) OLS regression  

&emsp;A. Use alldata_CA.csv  

&emsp;B. Results saved as HTML in "final-project-Sylvie515/results/table" folder  

### 4. visualize_results.py

(1) Use alldata_CA.csv  

(2) Generated figures saved in "final-project-Sylvie515/results/figures/lnVoteShare_Score" folder  
