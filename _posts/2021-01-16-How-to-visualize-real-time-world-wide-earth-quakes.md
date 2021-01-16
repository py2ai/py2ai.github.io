---
layout: post
title: How to visualize Earthquakes in Python
categories: [Python learning series]
mathjax: true
featured-img: earthquakes
summary: This tutorial is about visualization of the most recent earthquakes on the planet Earth.
---
Hi friends! today we will use two libraries in Python 3 to plot the balloons to represent the earthquakes. We will use a Red balloon for an earthquake 
above 5 magnitude and Green balloon for less than 5. You can tweak the icon_anchor=(9,25) for (horizontal, vertical) adjustments of the text over balloon. 
No more wait and here is the code:

### main.py

```python
# Welcome to Pyshine Earthquake visualization tutorial
""" Essentials
	pip install quakefeeds
	pip install folium
"""
from quakefeeds import QuakeFeed
import folium
from datetime import datetime
from folium.features import DivIcon

feed = QuakeFeed("4.5", "day")

my_map = folium.Map(tiles='Stamen Terrain', zoom_start = 13)

for i in range(len(feed)):

	""" Convert the timestamp to UTC time """
	timestamp = round (feed[i]['properties']['time']/1000)
	dt_object = str(datetime.utcfromtimestamp(timestamp))+' (UTC)'
	print(i,feed.event_title(i)+ ' at '+ dt_object,[feed.location(i)[1],feed.location(i)[0]],feed.magnitude(i),feed.depth(i))

	""" Marks in terms of Pixels units """
	folium.CircleMarker(
	radius=1.5**feed.magnitude(i),
	location=[feed.location(i)[1],feed.location(i)[0]],
	popup=feed.event_title(i)+' at '+ dt_object,
	color='#3186cc',
	fill=True,
	).add_to(my_map)

	""" Marks in terms of Meters units"""
	folium.Circle(
	radius=feed.magnitude(i),
	location=[feed.location(i)[1],feed.location(i)[0]],
	popup=feed.event_title(i)+' at '+ dt_object,
	color='#3186cc',
	fill=False,
	).add_to(my_map)

	""" Add some text to the marker """
	tooltip = feed.event_title(i)+' at '+ dt_object

	if (feed.magnitude(i))>5:
		color='red'
	else:
		color='green'
		
	""" A balloon marker """	
	folium.Marker([feed.location(i)[1],feed.location(i)[0]] ,
	popup=feed.event_title(i)+' at '+ dt_object,
	tooltip=tooltip,
	icon=folium.Icon(color=(color), icon='circle' )#icon='info-sign',prefix='fa',icon='circle'
	).add_to(my_map)

	""" Add text on the map """
	folium.map.Marker(
	[feed.location(i)[1],feed.location(i)[0]] ,
	icon=DivIcon(
		icon_size=(18,70),
		icon_anchor=(9,25),
		html='<div style="font-size: 10pt; color: white;">%s</div>' % float(feed.magnitude(i)),
		)
	).add_to(my_map)


""" Display the map """
my_map.save('index.html')


```
Simply save this main.py to a directory and run the code as:

```
python main.py
```
The terminal/console will show the recent earthquakes and a separate index.html file will be generated for the visualization purpose in the same directory as this code. Simply open
this index.html in a browser to visualize and thats all for today!




