import xml.etree.ElementTree as ET
import pandas as pd
import os
from pathlib import Path

tree = ET.parse('word.xml')
root = tree.getroot()
datafile=[]
for child in root:
    datafile.append(child.attrib)        

image = []
for i in range(len(datafile)):
    w = Path(datafile[i]['file'])
    
    image.append(os.path.join(os.getcwd(),w))
tag=[]
for i in range(len(datafile)) :
    tag.append(datafile[i]['tag'])

df = pd.DataFrame(data={'imagefile':image,'data':tag})
df.to_csv('datafile.csv',sep=',',index=False)