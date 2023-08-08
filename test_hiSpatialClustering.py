# -*- coding: utf-8 -*-
"""
Example for classification with postgis and standalone python.

Created on Tue May  9 11:47:05 2017

@author: cheny
"""

'''
note: GPU is used in this example. so packages list after is required: numba, cudatoolkit with CUDA7.5 or later.
          you can install in anaconda using 'conda install numba' and 'conda install -c numba cudatoolkit=8.0'
                                      or using Unofficial Binaries for Python Extension Packages
      if CPU is required, the arcpy import and using in file 'fs_c_calcdens.py' and 'fs_c_findnrstdist.py' should be removed.
          arcpy in these files is only for enable the progressor bar and is not refered by calculation.
'''

import psycopg2
from section_gpu import calc_density_gpu, calc_nrst_dist_gpu
import numpy as np

'''
define the class num.
'''
CLS_NUM=100

# import psycopg2
# from psycopg2 import extensions

# # Connect to the PostgreSQL server
# conn = psycopg2.connect(
#     host="localhost",
#     port="5432",
#     database="postgres",
#     user="postgres",
#     password="1234"
# )

# # Disable transactions
# conn.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)

# # Create a cursor object to interact with the database
# cursor = conn.cursor()

# # Create a new database
# new_database = "temp_geometry"
# cursor.execute(f"CREATE DATABASE {new_database}")

# # Close the cursor and the connection to the default database
# cursor.close()
# conn.close()

# # Connect to the newly created database
# conn = psycopg2.connect(
#     host="localhost",
#     port="5432",
#     database=new_database,
#     user="postgres",
#     password="1234"
# )

# # Create a cursor object to interact with the new database
# cursor = conn.cursor()

# # Perform further operations on the new database as needed

# # Close the cursor and the connection to the new database
# cursor.close()
# conn.close()


# Create table named flickr_sh_wt in the database temp_geometry with the following schema:
# gid: integer
# st_x: double precision
# st_y: double precision
# weight: integer

# import psycopg2
# from psycopg2 import extensions

# # Connect to the PostgreSQL server
# conn = psycopg2.connect(
#     host="localhost",
#     port="5432",
#     database="postgres",
#     user="postgres",
#     password="1234"
# )

# # Disable transactions
# conn.set_isolation_level(extensions.ISOLATION_LEVEL_AUTOCOMMIT)

# # Create a cursor object to interact with the database
# cursor = conn.cursor()

# # CReate a new table
# cursor.execute('''
#     CREATE TABLE flickr_sh_wt (
#         gid integer,    
#         st_x double precision,
#         st_y double precision,
#         weight integer
#     )
# ''')
               
# # Close the cursor and the connection to the default database
# cursor.close()
# conn.close()




'''
connect to postgresql/postgis.
'''
conn = psycopg2.connect(database="temp_geometry", user="postgres", password="1234", host="localhost", port="5432")
conn.set_session(autocommit=True)
cur=conn.cursor()



'''
fetch data with schema:(id,x,y,weight).
note: the coordinates must be in projected coordinate system. EPSG:4528 is CGCS2000 / 3-degree Gauss-Kruger zone 40
'''
cur.execute('''
            SELECT gid,st_x(st_transform(geom,4528)),st_y(st_transform(geom,4528)),weight
            from flickr_sh_wt
            ''')
result=cur.fetchall()

# If the table flickr_sh_wt does not exist, create it
try:
    cur.execute('''
        CREATE TABLE flickr_sh_wt (
            gid integer,    
            st_x double precision,
            st_y double precision,
            weight integer
        )
    ''')
except:
    pass





'''
transform data to array.
'''
ids=[]
xs=[]
ys=[]
weights=[]

for row in result:
    ids.append(int(row[0]))
    xs.append(row[1])
    ys.append(row[2])
    weights.append(row[3])
    
'''
main calculation.
note: params can be modified.
'''
density=calc_density_gpu(np.array(xs,np.float64),np.array(ys,np.float64),np.array(weights,np.int64),'GAUSS',sigma=15)
#ndresult is an tuple with (nrst_dist, parent_gid)
ndresult=calc_nrst_dist_gpu(np.array(ids,np.int64),np.array(xs,np.float64),np.array(ys,np.float64),density)

'''
find the class center.
'''
cls_cntr={ids[i]:[] for i in [i for i in (density*np.sqrt(ndresult[0])).argsort()][-CLS_NUM:]}

'''
classify all the points.
'''
parent_gid=ndresult[1]

cls_tree={}

for i in range(len(ids)):
    if ids[i] not in cls_cntr.keys():
        pgid=parent_gid[i]    
        if pgid in cls_tree.keys():
            cls_tree[pgid].append(ids[i])
        else:
            cls_tree[pgid]=[ids[i]]
        
def appendallchild(cls_tree,cur_list,cur_gid):
    cur_list.append(cur_gid)
    if cur_gid in cls_tree.keys():
        for c_gid in cls_tree[cur_gid]:
            appendallchild(cls_tree,cur_list,c_gid)
for cntr_gid in cls_cntr.keys():
    appendallchild(cls_tree,cls_cntr[cntr_gid],cntr_gid)
    
'''
now classification is done.
'''

'''
insert results to postgresql/postgis.
the results only contain (gid,cls), so for further processing, it's required that joining the results' table with the original table using the condition of equal of gid.
'''
#try to overwrite the results table. so drop first, and then create back.
try:
    cur.execute('''DROP TABLE public.classification_result_%d'''%CLS_NUM)    
except:
    pass
  
cur.execute('''
            CREATE TABLE public.classification_result_%d 
            (
                gid int primary key,
                cls int
            )
            '''%CLS_NUM)
#insert results.            
for cntr_gid in cls_cntr.keys():
    for c_gid in cls_cntr[cntr_gid]:
        cur.execute('''
                    INSERT INTO public.classification_result_%d (gid,cls) 
                    VALUEs (%d,%d)
        '''%(CLS_NUM,c_gid,cntr_gid))  