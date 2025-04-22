# This file contains functions and methods that are used to get the raw data
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os



# get original image url
def get_original_image_url(thumb_url):
    # remove /thumb/
    original_url = thumb_url.replace("/thumb/", "/")    
    # remove anything after the last / (thumbnail information)
    original_url = original_url.rsplit("/", 1)[0]

    return original_url

# get California House Election Result
def CA_house_election(url, year):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    CA_house_data = []

    for district in range(1, 54):
        suffix = ("st" if district in [1, 21, 31, 41, 51] else
                  "nd" if district in [2, 22, 32, 42, 52] else
                  "rd" if district in [3, 23, 33, 43, 53] else "th")
    
        # table header
        header = soup.find("caption", string = f"{year} California's {district}{suffix} congressional district election")
    
        if header:
            table = header.find_parent("table")
            if table:
                rows = table.find_all("tr")[1:]

                # candidate  name
                name_columns = rows[3].find_all("td")
                # candidate photo
                photo_columns = rows[2].find_all("img")
                # candidate party
                party_columns = rows[4].find_all("td")
                # candidate vote
                vote_columns = rows[5].find_all("td")
                # current representative
                current_rep_columns = rows[8].find_all("td")

                for i in range(len(name_columns)):
                    # candidate  name
                    name = name_columns[i].text.strip()
                    # candidate photo                
                    img_src = photo_columns[i]["src"] if i < len(photo_columns) else ""
                    photo_url = f"https:{img_src}" if img_src.startswith("//") else img_src
                    original_photo_url = get_original_image_url(photo_url)
                    # photo is not downloadable (original_photo_url end with ".svg") = False, else = True
                    downloadable = "F" if original_photo_url.lower().endswith(".svg") else "T"
                    # candidate party
                    party = party_columns[i].text.strip() if i < len(party_columns) else ""
                    # candidate vote
                    votes = vote_columns[i].text.strip() if i < len(vote_columns) else ""
                    # current representative (yes = 1, no = 0)
                    if len(current_rep_columns) >= 2:
                        current_rep = current_rep_columns[0].find("p").find("a").text.strip()
                        re_election = 1 if current_rep == name else 0
                    else:
                        re_election = 0

                    CA_house_data.append({"Year": year,
                                          "State": "CA",
                                          "District": district,
                                          "Name": name,
                                          "Photo": original_photo_url,
                                          "Photo_Downloadable": downloadable,
                                          "Party": party,
                                          "Votes": votes,
                                          "Incumbent": re_election})

    CA_house_election_df = pd.DataFrame(CA_house_data)
    CA_house_election_df = CA_house_election_df[CA_house_election_df["Name"] != ""]

    return CA_house_election_df

# call CA 2018 house election results
url_2018 = "https://en.wikipedia.org/wiki/2018_United_States_House_of_Representatives_elections_in_California"
CA_house_election_2018 = CA_house_election(url_2018, 2018)
CA_house_election_2018["ID"] = CA_house_election_2018.groupby(["State", "District"]).cumcount() + 1
CA_house_election_2018["ID"] = CA_house_election_2018.apply(lambda row: f"CA_{row["District"]}_{row["ID"]}", axis = 1)
# call CA 2020 house election results
url_2020 = "https://en.wikipedia.org/wiki/2020_United_States_House_of_Representatives_elections_in_California"
CA_house_election_2020 = CA_house_election(url_2020, 2020)
CA_house_election_2020["ID"] = CA_house_election_2020.groupby(["State", "District"]).cumcount() + 1
CA_house_election_2020["ID"] = CA_house_election_2020.apply(lambda row: f"CA_{row["District"]}_{row["ID"]}", axis = 1)

## save the results to csv file
# create new folder
output_dir = "../data/raw/election_result"
os.makedirs(output_dir, exist_ok = True)
# file path & save
file_path_2018 = os.path.join(output_dir, "CA_house_election_2018.csv")
file_path_2020 = os.path.join(output_dir, "CA_house_election_2020.csv")
CA_house_election_2018.to_csv(file_path_2018, index = False, encoding = "utf-8")
CA_house_election_2020.to_csv(file_path_2020, index = False, encoding = "utf-8")



## save the photos
def download_images(election_data, output_dir):
    # Set User-Agent
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
    
    os.makedirs(output_dir, exist_ok = True)

    for index, row in election_data.iterrows():
        photo_url = row["Photo"]
        image_id = row["ID"]
        # download images if Photo_Downloadable == T
        if row["Photo_Downloadable"] == "T":
            try:
                response = requests.get(photo_url, headers = headers, stream = True)
                if response.status_code == 200:
                    # get filename extension
                    image_extension = os.path.splitext(photo_url)[1]
                    # image name
                    image_name = f"{image_id}{image_extension}"
                    output_image_path = os.path.join(output_dir, image_name)

                    # download images
                    with open(output_image_path, "wb") as file:
                        for chunk in response.iter_content(1024):
                            file.write(chunk)
                else:
                    print(f"Failed to download {image_id}: {response.status_code}")
            except Exception as e:
                print(f"Error - {image_id}: {str(e)}")

# download CA 2018 Candidate Photos
download_images(CA_house_election_2018, "../data/raw/candidate_images/CA/2018")
# download CA 2020 Candidate Photos
download_images(CA_house_election_2020, "../data/raw/candidate_images/CA/2020")
