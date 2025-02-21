import pandas as pd
import numpy as nm


##Variables Notes

## python is dynamical types - its variables are determined at runtime
## to check type use type(variable)
##datatypes
##list,tuples,sets,dictionary

##conditional statements


for i in range(1,5):
    print(f"Hello, AI,{i}")


##list comprehension
words = ["aa","awe123","dsed","qweere"]
lengths = [len(l) for l in words]
print(lengths)

##tuples
##tuples are ordered collection of items that are immutable. similar to list but immutable in nature

##creating tuples
empty_tuple = (1,"hi",2)
lst = [1,"hi",2]
print(empty_tuple)
print(lst)
# lst = lst[:0] + 1
empty_tuple= empty_tuple * 2
print(lst)
print(empty_tuple)



##dictionary
## data is stored in key-value pairs, and immutable

empty_dict = dict()
student = {'id':101,'name':'shiva','sub':'ai'}
print(student['id'])

##merge dict
dict1 = {'id':101,'age':20}
dict2 = {'id':101,'name':'shiva'}
merge_dict = {**dict1,**dict2}
print(merge_dict)


##functions
##regular user defined function, lambda functions/map functions/filter funtions

##lambda example
##syntax - lambda arguments: expression
addition = lambda a,b:a+b
type(addition)
print(addition(1,2))
##map function - applies a given fucntion to all items in an input list and returns a map object
##this is particularly useful for transforming data in a list comprehension


##class and objects







