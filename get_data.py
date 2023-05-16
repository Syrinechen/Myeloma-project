# %% imports

import pandas as pd
import numpy as np
from pathlib import Path    
from tqdm import tqdm
import csv

# %%
#get all genes names from one patient to create header of all_data file
header=['Patient_id']
def get_header():
    csv_file='/home/irit/Documents/Myeloma/data/900-04_TPM.csv'
    example=pd.read_csv(csv_file)
    patient_id=csv_file.split('_')[0][-6:] #get patient id from csv file name
    example[['GeneID', 'Patient '+patient_id]]=example['GeneID\t'+patient_id].str.split("\t", expand = True)
    example=example.drop('GeneID\t'+patient_id, axis=1) #separate the two columns
    genes=example['GeneID'].values.tolist()
    res=header+genes
    res.append('MRD Response')
    return res

# %%
#Some patients in the csv files do not have their id but their admission number. 
#This dictionary contains the correspondance between the two in order to change the ones with admission number

def Correspondances_interne_id():
    d={}
    with open("nums_internes_cassioppe.tsv") as f:
        for line in f:
            (value, key)=line.split()
            d[key]=value[2:8]
    return d


dict=Correspondances_interne_id()
# %%

#Get MRD response from patient_id
reponses=pd.read_excel('Cassiopee Reponse-MRD.xls')
def get_patient_response(patient_id):
    for i in range(len(reponses)):
        if (patient_id==reponses['USUBJID'].values[i][-6:]):
            if (reponses['post_consolidation_MRD at 10-5'].values[i]=='POSITIVE'):
                return 1
            elif (reponses['post_consolidation_MRD at 10-5'].values[i]=='NEGATIVE') :
                return 0
            else :
                return None

# %%
#Create one big matrix: lines are patients, columns are genes, last column is the label (MRD+ or MRD-)
directory = '/home/irit/Documents/Myeloma/data'
pathlist = Path(directory).glob('*.csv')
headerList=get_header()
not_found=0
patient_list=[]
def get_all_data():
    file = open("all_data.csv", "w")
    writer = csv.writer(file)
    writer.writerow(headerList)
    patient_number=0
    for path in tqdm(pathlist):
        res=[]
        csv_file=str(path)
        data=pd.read_csv(csv_file)
        patient_id=csv_file.split('_')[0][34:]
        genes_count=data['GeneID\t'+patient_id].str.split("\t", expand = True)[1].values
        if patient_id[0]=='T':
            if (patient_id not in dict.keys()):
                print(patient_id, ' not found')
            else :
                patient_id=dict[patient_id]
        patient_list.append(patient_id)
        y=get_patient_response(patient_id)
        res.append(patient_id)
        res=res+genes_count.tolist()
        res.append(y)
        writer.writerow(res)    
    file.close()


# %%
get_all_data()


# %%
