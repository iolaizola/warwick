"""
Streamlit app for Vicomtech Department Resource Planning
2024 iolaizola@vicomtech.org 
"""

import pandas as pd
import streamlit as st
#import plotly.graph_objects as go
#import plotly.figure_factory as ff
#import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#import os
from PIL import Image # just to open the favicon (OMG)
#from utils import *  # our own library for the VicomPlanner
#import base64
#from plotly.offline import iplot
#import extra_streamlit_components as stx

favicon = Image.open("data/vicomtech.ico")
st.set_page_config(page_title='Dashboard', page_icon=favicon, layout="wide", initial_sidebar_state="collapsed",
     menu_items={
         'Get Help': 'https://www.vicomtech.org',
         'Report a bug': "https://www.vicomtech.org",
         'About': "Dashboard for Warwick Data v0.08(<iolaizola@vicomtech.org>)"
    }
)
#apply_layout_settings()

#logger.info("Init-------------------------")


#######Functions and Init

def calculate_new_score(x):
    points = 0
    #w = [0.5,1,2,3,4,4,3,2,1,.5]
    #w = [0.5,3,4,7,10,1,2,5,6,8]
    w = [1,3,5,7,9,2,4,6,8,10]

    if x['Sweet -5'] == 10: points += w[0]
    if x['Sweet -4'] == 10: points += w[1]
    if x['Sweet -3'] == 10: points += w[2] 
    if x['Sweet -2'] == 10: points += w[3] 
    if x['Sweet -1'] == 10: points += w[4] 
    if x['Sweet +1'] == 10: points += w[5] 
    if x['Sweet +2'] == 10: points += w[6] 
    if x['Sweet +3'] == 10: points += w[7] 
    if x['Sweet +4'] == 10: points += w[8] 
    if x['Sweet +5'] == 10: points += w[9]
    return points/sum(w)*10

k4 = 2.2
k3 = 1.6
k2 = 1.2
k1 = 1.0
#####################
#####################

uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx", "xls"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, sheet_name="Data")
    # Pivot the dataframe to transform the 'Score 1' values into columns named 'Score_iter_n'
    grouped_score = df.pivot(index='ID', columns='Iteration', values='Score 1').add_prefix('S_it_')
    # Pivot the dataframe to transform the 'Delta It' values into columns named 'Delta_iter_n'
    grouped_delta = df.pivot(index='ID', columns='Iteration', values='Delta It').add_prefix('Delta_iter_')
    combined_df = pd.concat([grouped_score, grouped_delta], axis=1)
    combined_df.reset_index(inplace=True)
    columns_to_select = ['S_it_1','S_it_2','S_it_3','S_it_4','S_it_5','S_it_10','S_it_9','S_it_8','S_it_7','S_it_6']

    uniques = result = df.groupby('ID').agg({
        'Age': 'first',
        'Gender':'first',
        "Condition":'first',
        "Tag":"first",
        "Type":"first"}
         ).reset_index()
    uniques['Condition'].fillna('Healthy', inplace=True)
    C = combined_df[columns_to_select]
    C2 = C.copy() # C.applymap(lambda x: 0 if x == 10 else 1)

    C2.columns =   sweetness_tags = ['Sweet -5','Sweet -4','Sweet -3','Sweet -2','Sweet -1','Sweet +1','Sweet +2','Sweet +3','Sweet +4','Sweet +5']
    C3 =  C2.merge(uniques[['ID','Age',"Gender","Condition", "Tag", "Type"]], left_index=True, right_index=True)
    C3['Condition'] = C3['Condition'].replace('None', 'Healthy')
    C3.loc[C3['ID'] == 115, 'Condition'] = 'Alzheimers'
    C3.loc[C3['ID'] == 100, 'Condition'] = 'Alzheimers'
    C3['version'] = 2
    D = C3.copy()

    #Remove Outliers, "Alan"
    D = D[D['ID'] != 112]
    D["score"] = D.apply(lambda x: calculate_new_score(x),axis=1)  # Extract the numeric value using regex

    # First scatter plot
    sns.set_theme()
    sns.set(font_scale=3)
    binsize = 25
    mypalette =  {
        'Healthy': 'blue',
        'Alzheimers': 'red',
        'Others':'magenta',
        'After smoking':'cyan',
        'Before smoking':'#0099FF',
        'Individual':'lightgreen'
    }
    dft = D[D['version']==2].copy()
    dft['Age Group'] = np.floor(dft['Age'] / binsize) * binsize + binsize/2
    dft.loc[dft['Age Group']==dft['Age Group'].min(),'Age Group'] = dft['Age'].min()
    dft.loc[dft['Age Group']==dft['Age Group'].max(),'Age Group'] = dft['Age'].max()
    dft.loc[dft['score']==0,'Condition'] = "Others"
    plt.figure(figsize=(24,10))
    sns.lineplot(data=dft, x="Age Group", y="score", errorbar=('pi',50))
    #sns.scatterplot(data=dft, x='Age', y='Perception Performance', hue='Gender',legend=True, palette=["#AB00AB", "blue"],s=300) #.set(title="Age vs. Correct Answers")
    sns.scatterplot(data=dft, x='Age', y='score', hue='Condition', palette=mypalette,legend=True, s=300) #.set(title="Age vs. Correct Answers")
    plt.ylim(-1, 10.5)
    # Add lines for repeating tests
    repeated_test_IDs = filtered_df = dft.loc[dft['ID'] > 1000,'ID'].values
    for i in repeated_test_IDs:
        plt.plot([ dft.loc[dft['ID'] == i,'Age'].values[0],dft.loc[dft['ID'] == i/1000,'Age'].values[0] ], 
             [ dft.loc[dft['ID'] == i,'score'].values[0],dft.loc[dft['ID'] == i/1000,'score'].values[0] ])

    for index, row in dft.iterrows():
        plt.text(row['Age'], row['score'], str(row['ID']), fontsize=10, ha='center', va='center')


    plt.legend(loc='upper right', fontsize='14', bbox_to_anchor=(1, 1.2))

    st.pyplot(plt)



    # Using Tags
    sns.set_theme()
    sns.set(font_scale=3)
    binsize = 25
    typalette =  {
        'A': 'cyan',
        'CH Group': 'red',
        '2nd test': 'green'
    }
    dft = D[D['version']==2].copy()
    dft['Age Group'] = np.floor(dft['Age'] / binsize) * binsize + binsize/2
    dft.loc[dft['Age Group']==dft['Age Group'].min(),'Age Group'] = dft['Age'].min()
    dft.loc[dft['Age Group']==dft['Age Group'].max(),'Age Group'] = dft['Age'].max()
    dft.loc[dft['score']==0,'Condition'] = "Others"
    plt.figure(figsize=(24,10))
    sns.lineplot(data=dft, x="Age Group", y="score", errorbar=('pi',50))
    #sns.scatterplot(data=dft, x='Age', y='Perception Performance', hue='Gender',legend=True, palette=["#AB00AB", "blue"],s=300) #.set(title="Age vs. Correct Answers")
    sns.scatterplot(data=dft, x='Age', y='score', hue='Type', palette=typalette,legend=True, s=300) #.set(title="Age vs. Correct Answers")
    plt.ylim(-1, 10.5)
    # Add lines for repeating tests
    repeated_test_IDs = filtered_df = dft.loc[dft['ID'] > 1000,'ID'].values
    for i in repeated_test_IDs:
        plt.plot([ dft.loc[dft['ID'] == i,'Age'].values[0],dft.loc[dft['ID'] == i/1000,'Age'].values[0] ], 
             [ dft.loc[dft['ID'] == i,'score'].values[0],dft.loc[dft['ID'] == i/1000,'score'].values[0] ])
    dft['Tag'] = dft['Tag'].fillna('')
    for index, row in dft.iterrows():
        plt.text(row['Age'], row['score'], str(row['Tag']), fontsize=18, ha='right', va='center')


    plt.legend(loc='upper right', fontsize='x-small', bbox_to_anchor=(1, 1.2))
    st.pyplot(plt)



    # By gender
    sns.set_theme()
    sns.set(font_scale=3)
    binsize = 25
    typalette =  {
        'Female': 'green',
        'Male': 'blue'
    }
    dft = D[D['version']==2].copy()
    dft['Age Group'] = np.floor(dft['Age'] / binsize) * binsize + binsize/2
    dft.loc[dft['Age Group']==dft['Age Group'].min(),'Age Group'] = dft['Age'].min()
    dft.loc[dft['Age Group']==dft['Age Group'].max(),'Age Group'] = dft['Age'].max()
    dft.loc[dft['score']==0,'Condition'] = "Others"
    plt.figure(figsize=(24,10))
    sns.lineplot(data=dft, x="Age Group", y="score", errorbar=('pi',50))
    #sns.scatterplot(data=dft, x='Age', y='Perception Performance', hue='Gender',legend=True, palette=["#AB00AB", "blue"],s=300) #.set(title="Age vs. Correct Answers")
    sns.scatterplot(data=dft, x='Age', y='score', hue='Gender', palette=typalette,legend=True, s=300) #.set(title="Age vs. Correct Answers")
    plt.ylim(-1, 10.5)
    # Add lines for repeating tests
    repeated_test_IDs = filtered_df = dft.loc[dft['ID'] > 1000,'ID'].values
    for i in repeated_test_IDs:
        plt.plot([ dft.loc[dft['ID'] == i,'Age'].values[0],dft.loc[dft['ID'] == i/1000,'Age'].values[0] ], 
             [ dft.loc[dft['ID'] == i,'score'].values[0],dft.loc[dft['ID'] == i/1000,'score'].values[0] ])
    dft['Tag'] = dft['Tag'].fillna('')
    for index, row in dft.iterrows():
        plt.text(row['Age'], row['score'], str(row['Tag']), fontsize=18, ha='right', va='center')


    plt.legend(loc='upper right', fontsize='x-small', bbox_to_anchor=(1, 1.2))
    st.pyplot(plt)