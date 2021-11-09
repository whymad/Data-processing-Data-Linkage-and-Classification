import pandas as pd
import textdistance as td
import re
#read the csv
abt = pd.read_csv("abt.csv", encoding = "ISO-8859-1")
buy = pd.read_csv("buy.csv", encoding = "ISO-8859-1")
# match_table = pd.read_csv("abt_buy_truth.csv", encoding = "ISO-8859-1")

#create the list for the true match
# truth_match = []
# for b in match_table["idBuy"]:
#    a = match_table.loc[match_table['idBuy'] == b, "idAbt"].iloc[0]
#    truth_match.append((a, b))
# print(truth_match)
# count = 0

idabt = []
idbuy = []
# result_match = []

#iterate over the two data sets to find the match pairs
pattern = ' - [0-9A-Za-z]+$'
for row in abt.iterrows():
    x = re.findall(pattern, row[1][1])
    code = x[0][3:]
    names = (row[1][1]).lower()
    for r in buy.iterrows():
        name = str(r[1][3]).lower()
        string = re.split('\-|\/|\.|\,', r[1][1].lower())
        str1 = ''.join(string)
        if((code.lower() in str1) or (td.tversky(row[1][1], r[1][1])>0.7)):
            a_id = abt.loc[abt['name'] == row[1][1], "idABT"].iloc[0]
            b_id = buy.loc[buy['name'] == r[1][1], "idBuy"].iloc[0]
            # result_match.append((a_id, b_id))
            #put the match pair into lists
            idabt.append(a_id)
            idbuy.append(b_id)

            # count += 1
#save the pairs into a csv
match = pd.DataFrame({"idAbt": idabt, "idBuy": idbuy})
match.to_csv("task1a.csv", index=False)

#to calculate the precision and recall
# my_match = count
# tp = 0
# for i in result_match:
#    if i in truth_match:
#        tp += 1
# total_match = len(match_table)
# recall = tp/total_match
# precision = tp/my_match
# print(precision)
# print(recall)