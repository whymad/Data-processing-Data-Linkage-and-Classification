import pandas as pd

# read the csv
abt = pd.read_csv("abt.csv", encoding = "ISO-8859-1")
buy = pd.read_csv("buy.csv", encoding = "ISO-8859-1")
# for check RR and PC
#match_table = pd.read_csv("abt_buy_truth.csv", encoding = "ISO-8859-1")
# put the ture pairs into a list(for check RR and PC)
# true_match = []
# for b in match_table["idBuy"]:
#    a = match_table.loc[match_table['idBuy'] == b, "idAbt"].iloc[0]
#    true_match.append((a, b))

newabt = abt.copy()
newbuy = buy.copy()
# selest the columns i need
newabt = newabt[['idABT', 'name']]
newbuy = newbuy[['idBuy', 'manufacturer']]

# preprocess the strings in data
newabt['name'] = newabt['name'].map(lambda x: str(x).split(" ")[0])
newbuy['manufacturer'] = newbuy['manufacturer'].map(lambda x: str(x).split(" ")[0])

# create lists and put blocks in
aset = []
bset = []
for row in newabt.iterrows():
    aset.append([row[1][1].lower(),row[1][0]])
for row in newbuy.iterrows():
    bset.append([row[1][1].lower(),row[1][0]])
# put out matches in to one list(for check RR and PC)
# match = []
# for i in list(aset):
#    for j in list(bset):
#        if i[0] == j[0]:
#            match.append((i[1], j[1]))

# rearrange the id information and save in two csv blocks
a1 = []
a2 = []
b1 = []
b2 = []
for item in aset:
    a1.append(item[0])
    a2.append(item[1])
for item in bset:
    b1.append(item[0])
    b2.append(item[1])
# save datasets into two csv files
ablock = pd.DataFrame({"block_key": a1, "product_id": a2})
ablock.to_csv("abt_blocks.csv", index=False)
bblock = pd.DataFrame({"block_key": b1, "product_id": b2})
bblock.to_csv("buy_blocks.csv", index=False)

# calculate and print out the RR and PC for check
# tp = 0
# for item in match:
#     if item in true_match:
#         tp += 1
# n = len(newabt) * len(newbuy)
# print(1- (len(match)/n))
# print(tp/len(true_match))
