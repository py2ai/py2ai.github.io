---
layout: post
title: Basics about SQL database in Python
categories: [GUI tutorial series]
mathjax: true
featured-img: sqlite3
description:  This tutorial is about using sqlite3 to create, insert data and update SQL table
tags: [Python, SQLite, SQL, Database, Tutorial]
keywords: [SQLite, Python SQL, Database Tutorial, sqlite3, SQL Basics]
---

Hi friends! SQLite is basically a C library which gives us a lightweight disk-based database. It does not require a separate server process. On the other hand it allows us to access the database using a nonstandard variant of the SQL query 
language. We can use SQLite in various applications for internal storage. Prototyping an application using SQLite is possible so that we can port the code to a
much larger database such as PostgreSQL or even Oracle.

In Python3 the sqlite3 builtin library provides an SQL interface compliant with the DB-API 2.0 specification as described by PEP249. After import sqlite3 we need to
create a connection object which will represent the database and the all data will be stored in a ```.db``` file. The code below is self explanatory it will first 
try to create a ```test.db``` database for a company. It will then insert the data into the table. After that it will print the table and then we can add more data
to update it. Finally we will print the whole database. 

### sql.py
{% include codeHeader.html %}
```python
import sqlite3

""" CREATE TABLE """

try:
	
	conn = sqlite3.connect('test.db')
	conn.execute('''CREATE TABLE COMPANY
         (ID INT PRIMARY KEY     NOT NULL,
         NAME           TEXT    NOT NULL,
         STUDENTID            INT     NOT NULL,
         ADDRESS        CHAR(50),
         SCORE         REAL);''')
	print ("Table created successfully")
	conn.close()
except:
	pass

""" INSERT DATA INTO TABLE """
try:
 
	conn = sqlite3.connect('test.db')
	conn.execute("INSERT INTO COMPANY (ID,NAME,STUDENTID,ADDRESS,SCORE) VALUES (1, 'AAA', 32, 'UK', 20000.00 )");
	conn.execute("INSERT INTO COMPANY (ID,NAME,STUDENTID,ADDRESS,SCORE) VALUES (2, 'BBB', 25, 'Canada', 15000.00 )");
	conn.execute("INSERT INTO COMPANY (ID,NAME,STUDENTID,ADDRESS,SCORE) VALUES (3, 'CCC', 23, 'China', 20000.00 )");
	conn.execute("INSERT INTO COMPANY (ID,NAME,STUDENTID,ADDRESS,SCORE) VALUES (9, 'DDD', 25, 'Mont Blanc ', 65000.00 )");
	conn.commit()
	conn.close()
except:
	pass

""" PRINT TABLE """

conn = sqlite3.connect('test.db')
cursor = conn.execute("SELECT ID, NAME, STUDENTID, ADDRESS, SCORE from COMPANY")
print(cursor)
for row in cursor:
   print ("ID = ", row[0])
   print ("NAME = ", row[1])
   print ("STUDENT ID = ", row[2])
   print ("ADDRESS = ", row[3])
   print ("SCORE = ", row[4], "\n")
print("Records created successfully")
conn.close()
# visit pyshine.com for more detail
""" UPDATE TABLE """

conn = sqlite3.connect('test.db')
conn.execute("UPDATE COMPANY set SCORE = 25000.00 where ID = 1")
conn.execute("UPDATE COMPANY set SCORE = 77777.00 where STUDENTID = 25")
conn.commit()
conn.close()

""" PRINT TABLE """

conn = sqlite3.connect('test.db')
cursor = conn.execute("SELECT ID, NAME, STUDENTID,  ADDRESS, SCORE from COMPANY")
print(cursor)
for row in cursor:
   print ("ID = ", row[0])
   print ("NAME = ", row[1])
   print ("STUDENT ID = ", row[2])
   print ("ADDRESS = ", row[3])
   print ("SCORE = ", row[4], "\n")
print("Records created successfully")
conn.close()


```
You can run the above code as:

``` python3 sql.py```
