---
layout: post
title: How to parse XML file and save the data as CSV
categories: [tutorial]
featured-img: 2021-12-24-xml-files
mathjax: true
description: A quick tutorial to parse XML files using XML Element Tree
---

Hi there, let's say we have an XML file like this:

### songs.xml
{% include codeHeader.html %}
```xml
<?xml version="1.0"?>
<genre catalogue="Pop">
<song title="Song 1">
<artist>Artist 1</artist>
<year>2000</year>
<album>Album 1</album>
</song>
<song title="Song 2">
<artist>Artist 2</artist>
<year>2010</year>
<album>Album 2</album>
</song>
<song title="Song 3">
<artist>Artist 3</artist>
<year>2015</year>
<album>Album 3</album>
</song>
</genre>

```


So all we need to provide the name of the xml file in the Python script below:

### main.py
{% include codeHeader.html %}
```python
import pandas as pd
import xml.etree.ElementTree as ET


filename = 'songs'
tree = ET.parse(filename+'.xml')


root = tree.getroot()
element_dict =  {}

for elem in root.iter():
    element_dict[elem.tag]=[]

for elem in root.iter():
    if elem.text=='\n':
        element_dict[elem.tag].append(elem.attrib)
    else:        
        element_dict[elem.tag].append(elem.text)
      
    
def make_list(dict_list, placeholder):
    
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if  ll < lmax:
            dict_list[lname] += [placeholder] * (lmax - ll)
    
    return dict_list


ans = make_list(element_dict,-1)
df = pd.DataFrame(ans)
print(df)
df.to_csv(filename+".csv")

```

The above make_list function will take input dictionary as:

```
{'genre': [{'catalogue': 'Pop'}], 'song': [{'title': 'Song 1'}, {'title': 'Song 2'}, {'title': 'Song 3'}], 'artist': ['Artist 1', 'Artist 2', 'Artist 3'], 'year': ['2000', '2010', '2015'], 'album': ['Album 1', 'Album 2', 'Album 3']}
```

And output a dictionary with unknown information replaced by the placeholder like this:

```
{'genre': [{'catalogue': 'Pop'}, -1, -1], 'song': [{'title': 'Song 1'}, {'title': 'Song 2'}, {'title': 'Song 3'}], 'artist': ['Artist 1', 'Artist 2', 'Artist 3'], 'year': ['2000', '2010', '2015'], 'album': ['Album 1', 'Album 2', 'Album 3']}
```

### output of main.py

Run the `main.py` as 

``` python3 main.py```

### output

A csv file will be generated (songs.csv) at the same location or folder as main.py file and you will see this output:

```csv

                  genre                 song    artist  year    album
0  {'catalogue': 'Pop'}  {'title': 'Song 1'}  Artist 1  2000  Album 1
1                    -1  {'title': 'Song 2'}  Artist 2  2010  Album 2
2                    -1  {'title': 'Song 3'}  Artist 3  2015  Album 3
```


So, next time no matter how complex is your XML file, you can still get the data out of it, and use it for the applications of Artificial Intelligence.
