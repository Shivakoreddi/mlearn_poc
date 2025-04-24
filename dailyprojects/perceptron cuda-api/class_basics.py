
class Person:
    def __init__(self,name,age):
        self.name = name
        self.age = age

    def greet(self):
        return f"Hi {self.name}"

##__init__: special method ,that runs when object is created
##self - current object
##greet - method tied to object

##creating object
p1 = Person('shiva',36)
print(p1.greet())


class Dog:
    breed = "family type" ##class variable
    def __init__(self,name):
        self.name = name ##instance variables


d1 = Dog("buddy")
print(d1.name)
print(d1.breed)