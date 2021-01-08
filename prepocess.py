import os
import pandas as pd

df = pd.read_csv("youtube_done_reduce2_no_veryhigh_dislike.csv")

df_train = df.sample(frac=0.7, random_state=9527)
df_test = df.drop(df_train.index)

df_train = df_train.reset_index()
df_test = df_test.reset_index()

df_train = df_train.loc[:, ['Title', 'Category', 'View']]
df_train.columns = ['Title', 'Category', 'View']
df_test = df_test.loc[:, ['Title', 'Category', 'View']]
df_test.columns = ['Title', 'Category', 'View']

df_train.to_csv("train_no_veryhigh_dislike.tsv", sep="\t", index=False)
df_test.to_csv("test_no_veryhigh_dislike.tsv", sep="\t", index=False)

print("number of training data:", len(df_train))
df_train.head()
print("number of testing data:", len(df_test))
df_test.head()