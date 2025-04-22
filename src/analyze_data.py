# This file contains functions and methods that are used to analyze data
import os
import pandas as pd
import numpy as np
# OLS
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer



## descriptive statistics
os.makedirs("../results/descriptive_statistics", exist_ok = True)

# read the data
CA_final = pd.read_csv("../data/processed/CA_final.csv", encoding = "utf-8")
CA_final = CA_final[["Year", "District", "Party", "Incumbent", "Gender", "Age", "Edu", "score", "score_CNN", "vote_share", "ln_vote_share", "Poverty (%)", "Median_Household_Income", "Pop_Total", "Sex Ratio"]]

# drop sample with missing value
CA_final = CA_final.dropna()
# age square
CA_final["Agesq"] = CA_final["Age"] ** 2
# Education Level
CA_final["Edu_Level"] = CA_final["Edu_Level"] = CA_final["Edu"].map({"Master": 3,
                                                                     "Doctor": 3,
                                                                     "Bachelor": 2,
                                                                     "High_school": 1})
CA_final["Master & Above"] = ((CA_final["Edu"] == "Master") | (CA_final["Edu"] == "Doctor")).astype(int)
CA_final["College"] = (CA_final["Edu"] == "Bachelor").astype(int)
CA_final["High School"] = (CA_final["Edu"] == "High_school").astype(int)
# Party
CA_final["Republican"] = (CA_final["Party"] == "Republican").astype(int)
CA_final["Democratic"] = (CA_final["Party"] == "Democratic").astype(int)
CA_final["Other"] = ((CA_final["Party"] == "Green") | (CA_final["Party"] == "No party preference")).astype(int)
# city_type
def city_type(row):
    if row["Year"] == 2018:
        if row["Median_Household_Income"] >= 100000 and row["Pop_Total"] >= 760000:
            return "urban"
        elif row["Median_Household_Income"] < 75000 and row["Pop_Total"] < 720000:
            return "rural"
        else:
            return "transition"
    else:
        if row["Median_Household_Income"] >= 100000 and row["Pop_Total"] >= 760000:
            return "urban"
        elif row["Median_Household_Income"] < 75000 and row["Pop_Total"] < 730000:
            return "rural"
        else:
            return "transition"
CA_final["city_type"] = CA_final.apply(city_type, axis = 1)
# dummy
categorical_vars = ["Incumbent", "Gender", "city_type"]
for var in categorical_vars:
    dummies = pd.get_dummies(CA_final[var], prefix = var, prefix_sep = "_").astype(int)
    CA_final = pd.concat([CA_final, dummies], axis = 1)
# save
CA_final.to_csv("../results/alldata_CA.csv")

def calculate_statistics(data, variables):
    stats = data[variables].agg(["mean", "std", "min", "max"]).transpose()
    stats = stats.round(4)
    stats.columns = ["Mean", "Std", "Min", "Max"]
    stats.loc["N", :] = len(data)
    return stats

# descriptive statistics (by gender)
variables = ["vote_share", "ln_vote_share", "score", "score_CNN", 
             "Republican", "Democratic", "Other",
             "Incumbent_0", "Incumbent_1",
             "Age", "Master & Above", "College", "High School", 
             "city_type_rural", "city_type_transition", "city_type_urban",
             "Poverty (%)", "Median_Household_Income", "Pop_Total", "Sex Ratio"]
# all data
stats_total = calculate_statistics(CA_final, variables)
# Gender_M == 1
stats_male = calculate_statistics(CA_final[CA_final["Gender_M"] == 1], variables)
# Gender_F == 1
stats_female = calculate_statistics(CA_final[CA_final["Gender_F"] == 1], variables)
# table
all_stats = pd.concat([stats_total, stats_male, stats_female], keys = ["Total", "Male", "Female"], axis = 1)
# save
all_stats.to_csv("../results/descriptive_statistics/descriptive_statistics_CA.csv")

# descriptive statistics (by party)
variables = ["vote_share", "ln_vote_share", "score", "score_CNN", 
             "Gender_M", "Gender_F",
             "Incumbent_0", "Incumbent_1",
             "Age", "Master & Above", "College", "High School", 
             "city_type_rural", "city_type_transition", "city_type_urban",
             "Poverty (%)", "Median_Household_Income", "Pop_Total", "Sex Ratio"]
# all data
stats_total = calculate_statistics(CA_final, variables)
# Republican == 1
stats_Rep = calculate_statistics(CA_final[CA_final["Republican"] == 1], variables)
# Democratic == 1
stats_Democ = calculate_statistics(CA_final[CA_final["Democratic"] == 1], variables)
# Other == 1
stats_Other = calculate_statistics(CA_final[CA_final["Other"] == 1], variables)
# table
all_stats = pd.concat([stats_total, stats_Rep, stats_Democ, stats_Other], keys = ["Total", "Republican", "Democratic", "Other"], axis = 1)
# save
all_stats.to_csv("../results/descriptive_statistics/descriptive_stats_CA.csv")



## score in different groups

# candidate features
candidate_features = ["Gender_M", "Gender_F",
                      "Master & Above", "College", "High School",
                      "Republican", "Democratic", "Other",
                      "Incumbent_0", "Incumbent_1", 
                      "city_type_rural", "city_type_transition", "city_type_urban"]

# age groups
CA_final["Age_Group"] = pd.cut(CA_final["Age"], bins = [0, 40, 50, 60, 70, 100], labels = ["<40", "40-50", "50-60", "60-70", ">70"])
candidate_features += ["Age_Group"]

def calculate_group_statistics(data, feature):
    if feature == "Age_Group":
        grouped = data.groupby("Age_Group", observed = False)["score"]
        grouped_CNN = data.groupby("Age_Group", observed = False)["score_CNN"]
    else:
        grouped = data[data[feature] == 1]["score"]
        grouped_CNN = data[data[feature] == 1]["score_CNN"]
    
    # mean, std, N
    mean = round(grouped.mean(), 4)
    std = round(grouped.std(), 4)
    count = grouped.count()
    mean_CNN = round(grouped_CNN.mean(), 4)
    std_CNN = round(grouped_CNN.std(), 4)
    count_CNN = grouped_CNN.count()
    
    return {"Mean": mean, "Std": std, "N": count, "Mean_CNN": mean_CNN, "Std_CNN": std_CNN, "N_CNN": count_CNN,}

# table
CA_results = []
for feature in candidate_features:
    stats = calculate_group_statistics(CA_final, feature)
    if feature == "Age_Group":
        for group in stats["Mean"].index:
            CA_results.append([f"Age: {group}", stats["Mean"][group], stats["Std"][group], stats["N"][group], stats["Mean_CNN"][group], stats["Std_CNN"][group], stats["N_CNN"][group]])
    else:
        CA_results.append([feature, stats["Mean"], stats["Std"], stats["N"], stats["Mean_CNN"], stats["Std_CNN"], stats["N_CNN"]])
# convert to DataFrame
CA_results_df = pd.DataFrame(CA_results, columns = ["Feature", "Score (Mean)", "Score (Std)", "N", "Score_CNN (Mean)", "Score_CNN (Std)", "N_CNN"])

# save
CA_results_df.to_csv("../results/descriptive_statistics/grouped_score_statistics_CA.csv", index = False)



## Candidate Attractiveness Levels and Voting Share

# score group 1
CA_final["score_group"] = pd.cut(CA_final["score"], bins = [0, 70, 75, 80, 100], labels = ["below 69.99", "70-74.99", "75-79.99", "above 80"], include_lowest = True)
score_stats = CA_final.groupby("score_group").agg({"vote_share": ["mean", "std"],
                                                   "ln_vote_share": ["mean", "std"],
                                                   "score": "count"}).round(4).reset_index()
score_stats.columns = ["score_group", "vote_share (mean)", "vote_share (std)", "ln_vote_share (mean)", "ln_vote_share (std)", "count"]
# save
score_stats.to_csv("../results/descriptive_statistics/score_group_statistics_1.csv", index = False)
# score group 2
CA_final["score_group"] = pd.cut(CA_final["score"], bins = [0, 75, 80, 100], labels = ["below 74.99", "75-79.99", "above 80"], include_lowest = True)
score_stats = CA_final.groupby("score_group").agg({"vote_share": ["mean", "std"],
                                                   "ln_vote_share": ["mean", "std"],
                                                   "score": "count"}).round(4).reset_index()
score_stats.columns = ["score_group", "vote_share (mean)", "vote_share (std)", "ln_vote_share (mean)", "ln_vote_share (std)", "count"]
# save
score_stats.to_csv("../results/descriptive_statistics/score_group_statistics_2.csv", index = False)

# score group (CNN) 1
CA_final["score_group"] = pd.cut(CA_final["score_CNN"], bins = [0, 70, 75, 80, 100], labels = ["below 69.99", "70-74.99", "75-79.99", "above 80"], include_lowest = True)
score_stats = CA_final.groupby("score_group").agg({"vote_share": ["mean", "std"],
                                                   "ln_vote_share": ["mean", "std"],
                                                   "score_CNN": "count"}).round(4).reset_index()
score_stats.columns = ["score_group", "vote_share (mean)", "vote_share (std)", "ln_vote_share (mean)", "ln_vote_share (std)", "count"]
# save
score_stats.to_csv("../results/descriptive_statistics/score_group_statistics_CNN_1.csv", index = False)
# score group (CNN) 2
CA_final["score_group"] = pd.cut(CA_final["score_CNN"], bins = [0, 75, 80, 100], labels = ["below 74.99", "75-79.99", "above 80"], include_lowest = True)
score_stats = CA_final.groupby("score_group").agg({"vote_share": ["mean", "std"],
                                                   "ln_vote_share": ["mean", "std"],
                                                   "score_CNN": "count"}).round(4).reset_index()
score_stats.columns = ["score_group", "vote_share (mean)", "vote_share (std)", "ln_vote_share (mean)", "ln_vote_share (std)", "count"]
# save
score_stats.to_csv("../results/descriptive_statistics/score_group_statistics_CNN_2.csv", index = False)



## OLS

def OLS_regressions(data, bins, labels):
    data["score_group"] = pd.cut(data["score"], bins = bins, labels = labels, include_lowest = True).astype(int)
    data["score_group_CNN"] = pd.cut(data["score_CNN"], bins = bins, labels = labels, include_lowest = True).astype(int)
    
    models = {"Total": smf.ols("ln_vote_share ~ score_group + Age + Agesq + C(Gender) + Democratic + C(Incumbent) + C(Year)", data = data).fit(cov_type = "HC1"),
              "Total (CNN)": smf.ols("ln_vote_share ~ score_group_CNN + Age + Agesq + C(Gender) + Democratic + C(Incumbent) + C(Year)", data = data).fit(cov_type = "HC1"),
              "Total_edu": smf.ols("ln_vote_share ~ score_group + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(Incumbent) + C(Year)", data = data).fit(cov_type = "HC1"),
              "Total_edu (CNN)": smf.ols("ln_vote_share ~ score_group_CNN + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(Incumbent) + C(Year)", data = data).fit(cov_type = "HC1"),
              "Total_edu_city": smf.ols("ln_vote_share ~ score_group + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(Incumbent) + C(city_type) + C(Year)", data = data).fit(cov_type = "HC1"),
              "Total_edu_city (CNN)": smf.ols("ln_vote_share ~ score_group_CNN + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(Incumbent) + C(city_type) + C(Year)", data = data).fit(cov_type = "HC1"),
              "Not Democratic": smf.ols("ln_vote_share ~ score_group + Age + Agesq + C(Gender) + C(Edu_Level) + C(Incumbent) + C(city_type) + C(Year)", data = data[data["Party"] != "Democratic"]).fit(cov_type = "HC1"),
              "Democratic": smf.ols("ln_vote_share ~ score_group + Age + Agesq + C(Gender) + C(Edu_Level) + C(Incumbent) + C(city_type) + C(Year)", data = data[data["Party"] == "Democratic"]).fit(cov_type = "HC1"),
              "Not Democratic (CNN)": smf.ols("ln_vote_share ~ score_group_CNN + Age + Agesq + C(Gender) + C(Edu_Level) + C(Incumbent) + C(city_type) + C(Year)", data = data[data["Party"] != "Democratic"]).fit(cov_type = "HC1"),
              "Democratic (CNN)": smf.ols("ln_vote_share ~ score_group_CNN + Age + Agesq + C(Gender) + C(Edu_Level) + C(Incumbent) + C(city_type) + C(Year)", data = data[data["Party"] == "Democratic"]).fit(cov_type = "HC1"),
              "Not Incumbent": smf.ols("ln_vote_share ~ score_group + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(city_type) + C(Year)", data = data[data["Incumbent"] == 0]).fit(cov_type = "HC1"),
              "Incumbent": smf.ols("ln_vote_share ~ score_group + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(city_type) + C(Year)", data = data[data["Incumbent"] == 1]).fit(cov_type = "HC1"),
              "Not Incumbent (CNN)": smf.ols("ln_vote_share ~ score_group_CNN + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(city_type) + C(Year)", data = data[data["Incumbent"] == 0]).fit(cov_type = "HC1"),
              "Incumbent (CNN)": smf.ols("ln_vote_share ~ score_group_CNN + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(city_type) + C(Year)", data = data[data["Incumbent"] == 1]).fit(cov_type = "HC1")}
    
    return models

def stargazer_table(models, filename):
    stargazer = Stargazer([models["Total"], models["Total (CNN)"], models["Total_edu"], models["Total_edu (CNN)"], models["Total_edu_city"], models["Total_edu_city (CNN)"], models["Not Democratic"], models["Democratic"], models["Not Democratic (CNN)"], models["Democratic (CNN)"], models["Not Incumbent"], models["Incumbent"], models["Not Incumbent (CNN)"], models["Incumbent (CNN)"]])
    stargazer.title("Table: Effects of Facial Attractiveness Score on ln Vote Share")
    stargazer.custom_columns(["Total", "Total (CNN)", "Total", "Total (CNN)", "Total", "Total (CNN)", "Not Democratic", "Democratic", "Not Democratic (CNN)", "Democratic (CNN)", "Not Incumbent", "Incumbent", "Not Incumbent (CNN)", "Incumbent (CNN)"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    stargazer.show_model_numbers(False)
    stargazer.covariate_order(["score_group", "score_group_CNN", "Democratic", "C(Incumbent)[T.1]", "Intercept"])
    stargazer.add_line("Age FE", ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V"])
    stargazer.add_line("Gender FE", ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V"])
    stargazer.add_line("Education FE", ["", "", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V"])
    stargazer.add_line("Party FE", ["V", "V", "V", "V", "V", "V", "", "", "", "", "V", "V", "V", "V"])
    stargazer.add_line("Incumbent FE", ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "", "", "", ""])
    stargazer.add_line("City FE", ["", "", "", "", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V"])
    stargazer.add_line("Year FE", ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V"])
    
    with open(filename, "w") as f:
        f.write(stargazer.render_html())

alldata_CA = pd.read_csv("../results/alldata_CA.csv", encoding="utf-8")
os.makedirs("../results/table", exist_ok = True)

# For 3 score groups
models_3 = OLS_regressions(alldata_CA, [0, 75, 80, 100], [1, 2, 3])
stargazer_table(models_3, "../results/table/lnVoteShare_3.html")
# For 4 score groups
models_4 = OLS_regressions(alldata_CA, [0, 70, 75, 80, 100], [1, 2, 3, 4])
stargazer_table(models_4, "../results/table/lnVoteShare_4.html")



## interaction term

# interaction term (Incumbent)

# For 3 score groups
alldata_CA["score_group"] = pd.cut(alldata_CA["score"], bins = [0, 75, 80, 100], labels = [1, 2, 3], include_lowest = True).astype(int)
alldata_CA["score_group_CNN"] = pd.cut(alldata_CA["score_CNN"], bins = [0, 75, 80, 100], labels = [1, 2, 3], include_lowest = True).astype(int)
m_01_total = smf.ols("ln_vote_share ~ score_group + C(Incumbent) + score_group:C(Incumbent) + Age + Agesq + C(Gender) + Democratic + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_02_total = smf.ols("ln_vote_share ~ score_group_CNN + C(Incumbent) + score_group_CNN:C(Incumbent) + Age + Agesq + C(Gender) + Democratic + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_03_total = smf.ols("ln_vote_share ~ score_group + C(Incumbent) + score_group:C(Incumbent) + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_04_total = smf.ols("ln_vote_share ~ score_group_CNN + C(Incumbent) + score_group_CNN:C(Incumbent) + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_05_total = smf.ols("ln_vote_share ~ score_group + C(Incumbent) + score_group:C(Incumbent) + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(city_type) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_06_total = smf.ols("ln_vote_share ~ score_group_CNN + C(Incumbent) + score_group_CNN:C(Incumbent) + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(city_type) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
# For 4 score groups
alldata_CA["score_group"] = pd.cut(alldata_CA["score"], bins = [0, 70, 75, 80, 100], labels = [1, 2, 3, 4], include_lowest = True).astype(int)
alldata_CA["score_group_CNN"] = pd.cut(alldata_CA["score_CNN"], bins = [0, 70, 75, 80, 100], labels = [1, 2, 3, 4], include_lowest = True).astype(int)
m_07_total = smf.ols("ln_vote_share ~ score_group + C(Incumbent) + score_group:C(Incumbent) + Age + Agesq + C(Gender) + Democratic + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_08_total = smf.ols("ln_vote_share ~ score_group_CNN + C(Incumbent) + score_group_CNN:C(Incumbent) + Age + Agesq + C(Gender) + Democratic + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_09_total = smf.ols("ln_vote_share ~ score_group + C(Incumbent) + score_group:C(Incumbent) + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_10_total = smf.ols("ln_vote_share ~ score_group_CNN + C(Incumbent) + score_group_CNN:C(Incumbent) + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_11_total = smf.ols("ln_vote_share ~ score_group + C(Incumbent) + score_group:C(Incumbent) + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(city_type) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_12_total = smf.ols("ln_vote_share ~ score_group_CNN + C(Incumbent) + score_group_CNN:C(Incumbent) + Age + Agesq + C(Gender) + C(Edu_Level) + Democratic + C(city_type) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
# table
stargazer = Stargazer([m_01_total, m_02_total, m_03_total, m_04_total, m_05_total, m_06_total, m_07_total, m_08_total, m_09_total, m_10_total, m_11_total, m_12_total])
stargazer.title("Table: Effects of Facial Attractiveness Score on ln Vote Share")
stargazer.custom_columns(["3 Score Groups", "4 Score Groups"], [6, 6])
stargazer.custom_columns(["Total", "Total (CNN)", "Total", "Total (CNN)", "Total", "Total (CNN)", "Total", "Total (CNN)", "Total", "Total (CNN)", "Total", "Total (CNN)"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
stargazer.show_model_numbers(False)
stargazer.covariate_order(["score_group", "score_group:C(Incumbent)[T.1]", "score_group_CNN", "score_group_CNN:C(Incumbent)[T.1]", "C(Incumbent)[T.1]", "Intercept"])
stargazer.add_line("Age FE", ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V"])
stargazer.add_line("Gender FE", ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V"])
stargazer.add_line("Education FE", ["", "", "V", "V", "V", "V", "", "", "V", "V", "V", "V"])
stargazer.add_line("Party FE", ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V"])
stargazer.add_line("City FE", ["", "", "", "", "V", "V", "", "", "", "", "V", "V"])
stargazer.add_line("Year FE", ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V"])
# save
with open("../results/table/lnVoteShare_Incumbent.html", "w") as f:
    f.write(stargazer.render_html())

# interaction term (Party)

# For 3 score groups
alldata_CA["score_group"] = pd.cut(alldata_CA["score"], bins = [0, 75, 80, 100], labels = [1, 2, 3], include_lowest = True).astype(int)
alldata_CA["score_group_CNN"] = pd.cut(alldata_CA["score_CNN"], bins = [0, 75, 80, 100], labels = [1, 2, 3], include_lowest = True).astype(int)
m_01_total = smf.ols("ln_vote_share ~ score_group + Democratic + score_group:Democratic + Age + Agesq + C(Gender) + C(Incumbent) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_02_total = smf.ols("ln_vote_share ~ score_group_CNN + Democratic + score_group_CNN:Democratic + Age + Agesq + C(Gender) + C(Incumbent) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_03_total = smf.ols("ln_vote_share ~ score_group + Democratic + score_group:Democratic + Age + Agesq + C(Gender) + C(Edu_Level) + C(Incumbent) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_04_total = smf.ols("ln_vote_share ~ score_group_CNN + Democratic + score_group_CNN:Democratic + Age + Agesq + C(Gender) + C(Edu_Level) + C(Incumbent) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_05_total = smf.ols("ln_vote_share ~ score_group + Democratic + score_group:Democratic + Age + Agesq + C(Gender) + C(Edu_Level) + C(Incumbent) + C(city_type) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_06_total = smf.ols("ln_vote_share ~ score_group_CNN + Democratic + score_group_CNN:Democratic + Age + Agesq + C(Gender) + C(Edu_Level) + C(Incumbent) + C(city_type) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
# For 4 score groups
alldata_CA["score_group"] = pd.cut(alldata_CA["score"], bins = [0, 70, 75, 80, 100], labels = [1, 2, 3, 4], include_lowest = True).astype(int)
alldata_CA["score_group_CNN"] = pd.cut(alldata_CA["score_CNN"], bins = [0, 70, 75, 80, 100], labels = [1, 2, 3, 4], include_lowest = True).astype(int)
m_07_total = smf.ols("ln_vote_share ~ score_group + Democratic + score_group:Democratic + Age + Agesq + C(Gender) + C(Incumbent) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_08_total = smf.ols("ln_vote_share ~ score_group_CNN + Democratic + score_group_CNN:Democratic + Age + Agesq + C(Gender) + C(Incumbent) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_09_total = smf.ols("ln_vote_share ~ score_group + Democratic + score_group:Democratic + Age + Agesq + C(Gender) + C(Edu_Level) + C(Incumbent) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_10_total = smf.ols("ln_vote_share ~ score_group_CNN + Democratic + score_group_CNN:Democratic + Age + Agesq + C(Gender) + C(Edu_Level) + C(Incumbent) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_11_total = smf.ols("ln_vote_share ~ score_group + Democratic + score_group:Democratic + Age + Agesq + C(Gender) + C(Edu_Level) + C(Incumbent) + C(city_type) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
m_12_total = smf.ols("ln_vote_share ~ score_group_CNN + Democratic + score_group_CNN:Democratic + Age + Agesq + C(Gender) + C(Edu_Level) + C(Incumbent) + C(city_type) + C(Year)", data = alldata_CA).fit(cov_type = "HC1")
# table
stargazer = Stargazer([m_01_total, m_02_total, m_03_total, m_04_total, m_05_total, m_06_total, m_07_total, m_08_total, m_09_total, m_10_total, m_11_total, m_12_total])
stargazer.title("Table: Effects of Facial Attractiveness Score on ln Vote Share")
stargazer.custom_columns(["3 Score Groups", "4 Score Groups"], [6, 6])
stargazer.custom_columns(["Total", "Total (CNN)", "Total", "Total (CNN)", "Total", "Total (CNN)", "Total", "Total (CNN)", "Total", "Total (CNN)", "Total", "Total (CNN)"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
stargazer.show_model_numbers(False)
stargazer.covariate_order(["score_group", "score_group:Democratic", "score_group_CNN", "score_group_CNN:Democratic", "Democratic", "Intercept"])
stargazer.add_line("Age FE", ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V"])
stargazer.add_line("Gender FE", ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V"])
stargazer.add_line("Education FE", ["", "", "V", "V", "V", "V", "", "", "V", "V", "V", "V"])
stargazer.add_line("Incumbent FE", ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V"])
stargazer.add_line("City FE", ["", "", "", "", "V", "V", "", "", "", "", "V", "V"])
stargazer.add_line("Year FE", ["V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V", "V"])
# save
with open("../results/table/lnVoteShare_Party.html", "w") as f:
    f.write(stargazer.render_html())
