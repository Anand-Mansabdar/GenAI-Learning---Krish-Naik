import sqlite3

connection = sqlite3.connect("student.db")

# Object to insert record/create table
cursor = connection.cursor()

# Create table
table_info = """
  create table Student(Name varchar(25), Class varchar(25), Section varchar(25), Marks int)
"""

cursor.execute(table_info) # Executes the above query

# Insert Records
cursor.execute("insert into Student values('Anand', 'CSE', 'A', 76)")
cursor.execute("insert into Student values('Atharva', 'Cyber Security', 'B', 88)")
cursor.execute("insert into Student values('Sarang', 'AI/ML', 'A', 55)")
cursor.execute("insert into Student values('Krishna', 'Data Science', 'B', 72)")

# Display the above records
print("---------The Inserted Records are:---------")
data = cursor.execute("select * from Student")

# The rows will be in the form of a list
for row in data:
  print(row)
  

connection.commit()
connection.close()