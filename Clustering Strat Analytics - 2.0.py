#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Importing basic packages
import os
import warnings
import requests
import numpy as np
import pandas as pd
import pandas as pd
import calendar
import datetime

#Visualisations Libraries
import matplotlib.pyplot as plt
import plotly.express as px 
import squarify 
import seaborn as sns 
from pprint import pprint as pp 
from plotly.subplots import make_subplots 
import plotly.graph_objects as go
from Levenshtein import distance

### Data Standardization and Modeling with K-Means and PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer


# In[4]:


# !/usr/bin/env/ python
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import urllib
import pyodbc

# import tqdm as tqdm
import snowflake.connector
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
import pandas as pd
import numpy as n
import os
import json
from datetime import date  


# In[5]:


# SQL and snow flake connection
os.chdir("C:\\Users\\prapa001\\OneDrive - Corporate\\Desktop\\python_trials") 

credentials= json.load(open("credentials.json"))

cnxn_str = ("Driver={ODBC Driver 17 for SQL Server};"
            "Server=WINMPNDBp02;"
            "Database=ANALYSIS_PROJECTS;"
            "UID="+credentials['SQL']['user'] + ";" 
            +"pwd=" + credentials['SQL']['password'] +";" +
            "Trusted_Connection=Yes;"
           ) 
sql_connection = pyodbc.connect(cnxn_str)

sf_connection = snowflake.connector.connect( 
    user =credentials['SF']['user'], 
    password=credentials['SF']['password'] ,
    role='SF_SCM_ANALYTICS_DBRL',
    account='staples.east-us-2.azure', 
    warehouse='CAP_PRD_SC_WH',
    database='DATALAB_SANDBOX',
    schema='SCM_ANALYTICS',
    authenticator='externalbrowser' 
    ) 

engine = create_engine(URL(
    user =credentials['SF']['user'], 
    password=credentials['SF']['password'],
    role='SF_SCM_ANALYTICS_DBRL',
    account='staples.east-us-2.azure', 
    warehouse='CAP_PRD_SC_WH',
    database='DATALAB_SANDBOX',
    schema='SCM_ANALYTICS',
    authenticator='externalbrowser' 
)) 


# In[19]:


Active_SKU_FC = '''SELECT FC_Number, 
       FC_Name, 
       COUNT(DISTINCT Staples_SKU) AS Count_of_SKUs,
       COUNT(DISTINCT CASE WHEN PURCHASE_STATUS = 'A' THEN STAPLES_SKU ELSE '0' END) AS Active_Staples_SKU  
FROM LINKED.SCCDATA.FC_MASTER_V    
GROUP BY FC_Number, FC_Name'''
 # SKU Level DF
Active_SKU_FC_01 =  pd.read_sql(Active_SKU_FC,sql_connection)


# In[22]:


# Sorting Active SKUs by descending order
Active_SKU_FC_01 = Active_SKU_FC_01.sort_values(by = 'Active_Staples_SKU', ascending = False).head(20)
# Create the bar chart
fig = px.bar(Active_SKU_FC_01, x='FC_Name', y='Active_Staples_SKU', text='Active_Staples_SKU',text_auto='.3s',color='FC_Name')
# Add labels inside the bars
fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')

# Set the title of the chart
fig.update_layout(title_text='No of Active SKUs at FC level')
# Set the label color to white

# Show the chart
fig.show()


# In[29]:


# P1 to P3 Pick Type level cartons and count of SKU's
Pick_Type = '''SELECT  PICK_TYPE, 
        COUNT(DISTINCT CTN_ID) AS Count_of_Cartons,
      --  COUNT(Distinct CASE WHEN pick_type = 'CONVEY BREAKPACK' THEN CTN_ID END) As Total_BP_Cartons, 
      --  COUNT(Distinct CASE WHEN pick_type = 'FULL CASE' THEN  CTN_ID END) As Total_FC_Cartons, 
        COUNT(DISTINCT SHPD_SKU) AS Count_of_SKUs        
FROM [COST_TO_SERVE_ARCHIVE].[SC_Cost].[Carton_Pick_List_SC_Costs_Archive]  
WHERE     STAT_IND <> '99' 
      AND PICK_CTL_CHAR NOT IN ('#','T') 
      AND PICK_TYPE NOT IN ('DUMMY WRAP AND LABEL', 'RSI', 'DNR') 
      AND [Unit_Zero_Valid] IS NULL 
      AND [Valid_Period_Record] = 'Y' 
      AND  YEAR IN  ('2023')
      AND BU = 'CONTRACT'
      AND TimePeriod IN ('1_CTS','2_CTS','3_CTS')
      AND FC = '00974'
      AND Hub_nm = 'LaMirada Fleet'
GROUP BY Pick_TYPE'''
 # SKU Level DF
Pick_Type_01 =  pd.read_sql(Pick_Type,sql_connection)


# In[30]:


Pick_Type_01 = Pick_Type_01.style.format({
'Count_of_Cartons': '{:,.0f}',
'Count_of_SKUs': '{:,.0f}'})
# P1 - P3 Pick Type Level Aggregation Summary 
Pick_Type_01 


# In[8]:


FC_SKUs = '''SELECT  shpd_sku, 
          COUNT(DISTINCT CTN_ID) AS Count_of_Cartons
      --  COUNT(DISTINCT CASE WHEN pick_type = 'CONVEY BREAKPACK' THEN CTN_ID END) As Total_BP_Cartons
FROM [COST_TO_SERVE_ARCHIVE].[SC_Cost].[Carton_Pick_List_SC_Costs_Archive]  
WHERE     STAT_IND <> '99' 
      AND PICK_CTL_CHAR NOT IN ('#','T') 
      AND PICK_TYPE NOT IN ('DUMMY WRAP AND LABEL', 'RSI', 'DNR') 
      AND [Unit_Zero_Valid] IS NULL 
      AND [Valid_Period_Record] = 'Y' 
      AND  YEAR IN  ('2023')
      AND TimePeriod = '4_CTS'
      AND BU = 'CONTRACT'
      AND FC = '00974'
      AND pick_type = 'FULL CASE'
GROUP BY shpd_sku '''
 # SKU Level DF
FC_SKUs_01 =  pd.read_sql(FC_SKUs,sql_connection)


# In[13]:


MQ_001 = '''SELECT * FROM (
SELECT DISTINCT A.shpd_sku,
       A.Count_of_Cartons,
       B.SKU_Dept,
       B.SKU_Class, 
       B.SKU_Dept_Name, 
       B.SKU_Class_Name, 
       B.SKU_Description, 
       B.FC_DIMs_Length, 
       B.FC_DIMs_Width,
       B.FC_DIMs_Height,
       B.FC_DIMs_Weight_lbs,
       B.FC_DIMs_Volume,
       C.SKU, 
       C.FC, 
       C.Seasonal_4_WK_FCST,
       C.VENDOR_NAME,
       C.VENDOR,
       C.VENDOR_LEAD_TIME_QUOTED, 
       C.VENDOR_LEAD_TIME_FCST, 
       C.VENDOR_FIXED_ORDER_CYCLE_WEEK
FROM (SELECT shpd_sku, 
             COUNT(DISTINCT CTN_ID) AS Count_of_Cartons
      -- COUNT(DISTINCT CASE WHEN pick_type = 'CONVEY BREAKPACK' THEN CTN_ID END) AS Total_BP_Cartons
      FROM [COST_TO_SERVE_ARCHIVE].[SC_Cost].[Carton_Pick_List_SC_Costs_Archive]  
      WHERE STAT_IND <> '99' 
        AND PICK_CTL_CHAR NOT IN ('#','T') 
        AND PICK_TYPE NOT IN ('DUMMY WRAP AND LABEL', 'RSI', 'DNR') 
        AND [Unit_Zero_Valid] IS NULL 
        AND [Valid_Period_Record] = 'Y' 
        AND YEAR IN ('2023')
        AND BU = 'CONTRACT'
        AND TimePeriod IN ('1_CTS','2_CTS','3_CTS')
        AND FC = '00974'
        AND pick_type = 'FULL CASE'
        AND Hub_nm = 'LaMirada Fleet'
      GROUP BY shpd_sku) A 
LEFT JOIN (
  SELECT DISTINCT SKU_NUM,
         SKU_Dept,
         SKU_Class,
         SKU_Dept_Name,
         SKU_Class_Name, 
         SKU_Description,
         FC_DIMs_Length,
         FC_DIMs_Width,
         FC_DIMs_Height,
         FC_DIMs_Weight_lbs,
         FC_DIMs_Volume
  FROM LINKED.PRISM.MASTER_DETAIL_CURRENT_V
  WHERE FC_Num = '974'
) B ON A.shpd_sku = B.SKU_NUM 
LEFT JOIN (
  SELECT * 
  FROM OPENQUERY(SNWFLK_SC_ANALYTICS_FA,
    'SELECT SKU, 
            FC, 
            UNIT_OH,
            Seasonal_4_WK_FCST,
            VENDOR_NAME,
            VENDOR,
            VENDOR_LEAD_TIME_QUOTED, 
            VENDOR_LEAD_TIME_FCST, 
            VENDOR_FIXED_ORDER_CYCLE_WEEK 
     FROM DATALAB_SANDBOX.SCM_REPLEN.NADTRIM
     WHERE FC = ''974'' AND RUN_DATE = ''2023-05-02'' '
  )
) C ON A.shpd_sku = C.SKU) AS filtered_results '''

 # SKU Level DF 
Master_query_02 =  pd.read_sql(MQ_001,sql_connection)


# In[28]:


Master_query_02['Count_of_Cartons'].sum()


# In[131]:


# Data Prepation for 85 Percent scenario or 233 SKUs 
Master_File_01_Emb_SDO = Per_85_Scenario_Embe_SDO[["shpd_sku","Total_Space","2-Week-Forecast"]]
Master_File_01_Emb_SDO.set_index('shpd_sku', inplace=True)
Master_File_01_Emb_SDO.head()
# Standard Scaling for the 233 sku and make shpd sku

# Define StandardScaler object
scaler = StandardScaler()

# Fit the StandardScaler object to the Master_File_01 DataFrame
Master_file_02_std_copy_Embd_SDO = scaler.fit_transform(Master_File_01_Emb_SDO)

# Create a DataFrame from the standardized data
Master_file_05_std_copy_Embd_SDO = pd.DataFrame(Master_file_02_std_copy_Embd_SDO, columns=Master_File_01_Emb_SDO.columns, index=Master_File_01_Emb_SDO.index)

# Reset the index of the DataFrame
Master_file_05_std_copy_Embd_SDO = Master_file_05_std_copy_Embd_SDO.reset_index()

# Replacing NAN with Mean 
Master_file_05_std_copy_Embd_SDO[['Total_Space','2-Week-Forecast']] = Master_file_05_std_copy_Embd_SDO[['Total_Space','2-Week-Forecast']].fillna(value = Master_file_05_std_copy_Embd_SDO[['Total_Space','2-Week-Forecast']].mean())
# Performing K-means within 6 Clusters as it 6 clusters was threshold where we saw maximum decline
kmeans = KMeans(n_clusters=6, init='k-means++', random_state=42)

# Fitting our model to the dataset
kmeans.fit(Master_file_05_std_copy_Embd_SDO)

# Adding Cluster labels
Master_file_05_std_copy_Embd_SDO["Cluster"] = kmeans.labels_

# Join the original df to scaled df to tie up the cluster associated with each SKU
Merged_DF_SDO = Per_85_Scenario_Embe_SDO.merge(Master_file_05_std_copy_Embd_SDO, on='shpd_sku', how='left', indicator=True)
Merged_DF_SDO


# In[129]:


# Count the number of SKUs at cluster level
Cluster_groupby = Merged_DF_SDO.groupby(['Cluster']).agg({
    'shpd_sku': ['count'],
    'Total_Space_x': ['sum'],
    '2-Week-Forecast_x': ['sum']
}).reset_index()

# Reset the column index
Cluster_groupby.columns = Cluster_groupby.columns.get_level_values(0) + '_' + Cluster_groupby.columns.get_level_values(1)

# Calculate Usage % and Volume %
SKU_Usage_sum = Cluster_groupby['2-Week-Forecast_x_sum'].sum()
SKU_Volume = Cluster_groupby['Total_Space_x_sum'].sum()
Cluster_groupby['Usage %'] = (Cluster_groupby['2-Week-Forecast_x_sum'] / SKU_Usage_sum) * 100
Cluster_groupby['Volume %'] = (Cluster_groupby['Total_Space_x_sum'] / SKU_Volume) * 100

# Format the DataFrame
Cluster_groupby_formatted = Cluster_groupby.copy()

# Apply desired formatting to the columns
Cluster_groupby_formatted['Usage %'] = Cluster_groupby_formatted['Usage %'].map('{:.2f}%'.format)
Cluster_groupby_formatted['Volume %'] = Cluster_groupby_formatted['Volume %'].map('{:.2f}%'.format)

# Print the formatted DataFrame
Cluster_groupby_formatted 


# In[13]:


# CLustering model for all the 1922 Full Case SKUs
# Total Space calculation 
Master_file['Total_Space'] = Master_file['FC_DIMs_Volume'] * Master_file['2-Week-Forecast']

# Data Prepation for 85 Percent scenario or 212 SKUs 
Master_File_03 = Master_file[["SHPD SKU","Total_Space","2-Week-Forecast"]]
Master_File_03.set_index('SHPD SKU', inplace=True)
Master_File_03.head()
# Standard Scaling for the 1680 sku and make shpd sku

# Define StandardScaler object
scaler = StandardScaler()

# Fit the StandardScaler object to the Master_File_01 DataFrame
MF_07_std_copy = scaler.fit_transform(Master_File_03)

# Create a DataFrame from the standardized data
MF_06_std_copy = pd.DataFrame(MF_07_std_copy, columns=Master_File_03.columns, index=Master_File_03.index)

# Reset the index of the DataFrame
MF_06_std_copy = MF_06_std_copy.reset_index()

# Replacing NAN with Mean 
MF_06_std_copy[['Total_Space','2-Week-Forecast']] = MF_06_std_copy[['Total_Space','2-Week-Forecast']].fillna(value = MF_06_std_copy[['Total_Space','2-Week-Forecast']].mean())
# Performing K-means within 6 Clusters as it 6 clusters was threshold where we saw maximum decline
kmeans = KMeans(n_clusters=6, init='k-means++', random_state=42)

# Fitting our model to the dataset
kmeans.fit(MF_06_std_copy)

# Adding Cluster labels
MF_06_std_copy["Cluster"] = kmeans.labels_


# In[ ]:


MF_06_std_copy 
Master_file_05_std_copy 
# Join the original df to scaled df to tie up the cluster associated with each SKU
MF_2k_SKUs = Master_file.merge(MF_06_std_copy, on='SHPD SKU', how='left', indicator=True)
# Move this file to excel 
# Transfer the SKU level file to excel
with pd.ExcelWriter('C:\\Users\\prapa001\\OneDrive - Corporate\\Documents\\SC Strategic Projects\\LA - MIRADA CONS\\5123_SDOProject\\Stra_cluster_2K_SKUs.xlsx') as writer : 
    MF_2k_SKUs.to_excel(writer, index=False)


# In[16]:


# Join the original df to scaled df to tie up the cluster associated with each SKU
Master_file_05 = Master_File_01.merge(Master_file_05_std_copy, on='SHPD SKU', how='left', indicator=True)
Master_file_05.nunique()


# In[17]:


# Count the number of SKUs at cluster level
Cluster_groupby = Master_file_05.groupby(['Cluster']).agg({
    'SHPD SKU': ['count'],
    'Total_Space_x': ['sum'],
    '2-Week-Forecast_x': ['sum']
}).reset_index()

# Reset the column index
Cluster_groupby.columns = Cluster_groupby.columns.get_level_values(0) + '_' + Cluster_groupby.columns.get_level_values(1)

# Calculate Usage % and Volume %
SKU_Usage_sum = Cluster_groupby['2-Week-Forecast_x_sum'].sum()
SKU_Volume = Cluster_groupby['Total_Space_x_sum'].sum()
Cluster_groupby['Usage %'] = (Cluster_groupby['2-Week-Forecast_x_sum'] / SKU_Usage_sum) * 100
Cluster_groupby['Volume %'] = (Cluster_groupby['Total_Space_x_sum'] / SKU_Volume) * 100

# Format the DataFrame
Cluster_groupby_formatted = Cluster_groupby.copy()

# Apply desired formatting to the columns
Cluster_groupby_formatted['Usage %'] = Cluster_groupby_formatted['Usage %'].map('{:.2f}%'.format)
Cluster_groupby_formatted['Volume %'] = Cluster_groupby_formatted['Volume %'].map('{:.2f}%'.format)

# Print the formatted DataFrame
Cluster_groupby_formatted 


# In[86]:


Master_file_05


# In[ ]:


# Transfer the SKU level file to excel
with pd.ExcelWriter('C:\\Users\\prapa001\\OneDrive - Corporate\\Documents\\SC Strategic Projects\\LA - MIRADA CONS\\5123_SDOProject\\Stra_cluster_method_85_sample_8_Clusters.xlsx') as writer : 
    Master_file_05.to_excel(writer, index=False)

