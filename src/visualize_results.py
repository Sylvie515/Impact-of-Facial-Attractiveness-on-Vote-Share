# This file contains functions and methods that are used to visualize the results
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# read the data
alldata_CA = pd.read_csv("../results/alldata_CA.csv", encoding = "utf-8")



## Scatter Plot: ln Vote Share V.S. Score & Score_CNN

# party
os.makedirs("../results/figures/lnVoteShare_Score/byParty", exist_ok = True)

def scatterplot_lnVoteShare_Score_byParty(data, score_type, num_groups):

    # score groups
    if num_groups == 3:
        bins = [0, 75, 80, 100]
        labels = ["below 74.99", "75-79.99", "above 80"]
    else:
        bins = [0, 70, 75, 80, 100]
        labels = ["below 69.99", "70-74.99", "75-79.99", "above 80"]
    data[f"{score_type}_group"] = pd.cut(data[score_type], bins = bins, labels = labels, include_lowest = True)

    grouped_data = data.groupby(f"{score_type}_group", observed = False).agg({"ln_vote_share": "mean",
                                                                              "Democratic": "sum",
                                                                              score_type: "count"}).reset_index()

    grouped_data.columns = [f"{score_type}_group", "ln_vote_share (mean)", "democratic_count", "count"]
    # each score group: non-democratic sample = total sample - democratic sample
    grouped_data["nondemocratic_count"] = grouped_data["count"] - grouped_data["democratic_count"]
    
    # each score group: democratic ratio = democratic sample / total sample (for circle color)
    grouped_data["democratic_ratio"] = grouped_data["democratic_count"] / grouped_data["count"]

    # scatter plot
    plt.figure(figsize = (8, 5))
    scatter = plt.scatter(grouped_data[f"{score_type}_group"], 
                          grouped_data["ln_vote_share (mean)"], 
                          # circle size
                          s = grouped_data["count"]*5, 
                          # circle color
                          c = grouped_data["democratic_ratio"], 
                          cmap = "coolwarm", 
                          alpha = 0.7)
    
    for i, row in grouped_data.iterrows():
        x_pos = row[f"{score_type}_group"]
        y_pos = row["ln_vote_share (mean)"]
        if x_pos == grouped_data[f"{score_type}_group"].max():
            # the highest score group
            xytext = (-150, -5)
        else:
            xytext = (-1, 5)        
        plt.annotate(f"Total: {int(row["count"])}\n(Democratic: {int(row["democratic_count"])}, Non-democratic: {int(row["nondemocratic_count"])})", 
                     (row[f"{score_type}_group"], row["ln_vote_share (mean)"]),
                     xytext = xytext, textcoords = "offset points", fontsize = 8)
    
    plt.title(f"ln Vote Share by {score_type} ({num_groups} groups)")
    plt.xlabel("Score Groups")
    plt.ylabel("ln Vote Share (mean)")
    plt.colorbar(scatter, label = "Proportion of Democratic Candidates")
    plt.tight_layout()
    plt.savefig(f"../results/figures/lnVoteShare_Score/byParty/ln_vote_share_{score_type}_{num_groups}_groups.png")
    plt.close()

scatterplot_lnVoteShare_Score_byParty(alldata_CA, "score", 3)
scatterplot_lnVoteShare_Score_byParty(alldata_CA, "score", 4)
scatterplot_lnVoteShare_Score_byParty(alldata_CA, "score_CNN", 3)
scatterplot_lnVoteShare_Score_byParty(alldata_CA, "score_CNN", 4)

# incumbent
os.makedirs("../results/figures/lnVoteShare_Score/byIncumbent", exist_ok = True)

def scatterplot_lnVoteShare_Score_byIncumbent(data, score_type, num_groups):

    # score groups
    if num_groups == 3:
        bins = [0, 75, 80, 100]
        labels = ["below 74.99", "75-79.99", "above 80"]
    else:
        bins = [0, 70, 75, 80, 100]
        labels = ["below 69.99", "70-74.99", "75-79.99", "above 80"]
    data[f"{score_type}_group"] = pd.cut(data[score_type], bins = bins, labels = labels, include_lowest = True)

    grouped_data = data.groupby(f"{score_type}_group", observed = False).agg({"ln_vote_share": "mean",
                                                                              "Incumbent": "sum",
                                                                              score_type: "count"}).reset_index()

    grouped_data.columns = [f"{score_type}_group", "ln_vote_share (mean)", "Incumbent_count", "count"]
    # each score group: non-Incumbent sample = total sample - Incumbent sample
    grouped_data["nonIncumbent_count"] = grouped_data["count"] - grouped_data["Incumbent_count"]

    # each score group: Incumbent ratio = Incumbent sample / total sample (for circle color)
    grouped_data["Incumbent_ratio"] = grouped_data["Incumbent_count"] / grouped_data["count"]
    
    # scatter plot
    plt.figure(figsize = (8, 5))
    scatter = plt.scatter(grouped_data[f"{score_type}_group"], 
                          grouped_data["ln_vote_share (mean)"], 
                          # circle size
                          s = grouped_data["count"]*5, 
                          # circle color
                          c = grouped_data["Incumbent_ratio"], 
                          cmap = "coolwarm", 
                          alpha = 0.7)
    
    for i, row in grouped_data.iterrows():
        x_pos = row[f"{score_type}_group"]
        y_pos = row["ln_vote_share (mean)"]
        if x_pos == grouped_data[f"{score_type}_group"].max():
            # the highest score group
            xytext = (-150, -5)
        else:
            xytext = (-1, 5)        
        plt.annotate(f"Total: {int(row["count"])}\n(Incumbent: {int(row["Incumbent_count"])}, Not Incumbent: {int(row["nonIncumbent_count"])})", 
                     (row[f"{score_type}_group"], row["ln_vote_share (mean)"]),
                     xytext = xytext, textcoords = "offset points", fontsize = 8)
    
    plt.title(f"ln Vote Share by {score_type} ({num_groups} groups)")
    plt.xlabel("Score Groups")
    plt.ylabel("ln Vote Share (mean)")
    plt.colorbar(scatter, label = "Proportion of Incumbent Candidates")
    plt.tight_layout()
    plt.savefig(f"../results/figures/lnVoteShare_Score/byIncumbent/ln_vote_share_{score_type}_{num_groups}_groups.png")
    plt.close()

scatterplot_lnVoteShare_Score_byIncumbent(alldata_CA, "score", 3)
scatterplot_lnVoteShare_Score_byIncumbent(alldata_CA, "score", 4)
scatterplot_lnVoteShare_Score_byIncumbent(alldata_CA, "score_CNN", 3)
scatterplot_lnVoteShare_Score_byIncumbent(alldata_CA, "score_CNN", 4)
