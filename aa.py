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

# import psycopg2

# # Connect to the PostGIS database
# conn = psycopg2.connect(database="temp_geometry", user="postgres", password="******", host="localhost", port="5432")
# conn.set_session(autocommit=True)
# cur = conn.cursor()

# # Create the flickr_sh_wt table
# cur.execute('''
#     CREATE TABLE flickr_sh_wt (
#         gid serial PRIMARY KEY,
#         geom geometry(Point),
#         weight numeric
#     )
# ''')

# # Close the database connection
# cur.close()
# conn.close()


import psycopg2
import random

# Connect to the PostGIS database
conn = psycopg2.connect(database="temp_geometry", user="postgres", password="******", host="localhost", port="5432")
conn.set_session(autocommit=True)
cur = conn.cursor()

# Generate and insert random data into the flickr_sh_wt table
data = []
for i in range(50):
    gid = i + 1
    x = random.uniform(0, 100)
    y = random.uniform(0, 100)
    weight = random.uniform(1, 10)
    point = 'POINT({} {})'.format(x, y)
    data.append((gid, point, weight))

# Insert data into the flickr_sh_wt table
for row in data:
    cur.execute('''
        INSERT INTO flickr_sh_wt (gid, geom, weight)
        VALUES (%s, ST_GeomFromText(%s, 4528), %s)
    ''', row)

# Close the database connection
cur.close()
conn.close()

