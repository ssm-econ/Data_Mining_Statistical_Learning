#!/usr/bin/env python
# coding: utf-8

# Classes help us define data structures with some shared properties and methods.

# In[43]:


class MyClass:
    """A simple example class"""
    def __init__(self,number):
        self.i=number

    def f(self):
        return 'hello world'
    
    def g(self,another_member):
        return MyClass(self.i+another_member.i)
        


# In[44]:


thisguy = MyClass(1.0)


# In[45]:


thisguy.i


# In[46]:


thisguy.f()


# In[51]:


thatguy=MyClass(17.0)


# In[52]:


totallynewguy=thisguy.g(thatguy)


# In[53]:


totallynewguy.i


# In[29]:


type(1.0)


# In[54]:


type(thisguy)


# In[22]:


from math import gcd

class Fraction:
    def __init__(self, numerator, denominator):
        if type(numerator)==float or type(denominator)==float:
            raise ValueError("Numerator and denominator must be integers")
        if denominator == 0:
            raise ValueError("Denominator cannot be zero")
        common = gcd(numerator, denominator)
        self.numerator = numerator // common
        self.denominator = denominator // common
        if self.denominator < 0:  # Ensuring denominator is always positive
            self.numerator = -self.numerator
            self.denominator = -self.denominator

    def show(self):
        return f"{self.numerator}/{self.denominator}"

    def add(self, other):
        num = self.numerator * other.denominator + other.numerator * self.denominator
        den = self.denominator * other.denominator
        return Fraction(num, den)

    def sub(self, other):
        num = self.numerator * other.denominator - other.numerator * self.denominator
        den = self.denominator * other.denominator
        return Fraction(num, den)

    def mul(self, other):
        num = self.numerator * other.numerator
        den = self.denominator * other.denominator
        return Fraction(num, den)

    def div(self, other):
        if other.numerator == 0:
            raise ZeroDivisionError("Cannot divide by zero fraction")
        num = self.numerator * other.denominator
        den = self.denominator * other.numerator
        return Fraction(num, den)

# Example usage
f1 = Fraction(3, 4)
f2 = Fraction(2, 5)

print(f"Addition: {f1.show()} + {f2.show()} = {(f1.add(f2)).show()}")
print(f"Subtraction: {f1.show()} - {f2.show()} = {(f1.sub(f2)).show()}")
print(f"Multiplication: {f1.show()} * {f2.show()} = {(f1.mul(f2)).show()}")
print(f"Division: {f1.show()} / {f2.show()} = {(f1.mul(f2)).show()}")


# In[23]:


f1.show()


# In[26]:


f1=Fraction(1,10)


# ## Inheritance

# In[55]:


class Person:
    def __init__(self, fname, lname):
        self.firstname = fname
        self.lastname = lname

    def printname(self):
        print(self.firstname, self.lastname)

class Student(Person):
    def __init__(self, fname, lname, eid):
        Person.__init__(self, fname, lname)
        self.EID=eid;

    def showEID(self):
        print(self.EID)

    def talk(self,other_person):
        print(f"{self.firstname} {self.lastname} says hi to {other_person.firstname} {other_person.lastname}")

x = Student("Mike", "Olsen",1234)
y = Person("George","Maxwell")
x.printname()
x.showEID()
x.talk(y)


# In[56]:


x.EID=123


# In[58]:


x.showEID()


# In[59]:


class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def move(self):
        print("Drive!")

class Boat:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def move(self):
        print("Sail!")

class Plane:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def move(self):
        print("Fly!")

car1 = Car("Ford", "Mustang")       #Create a Car object
boat1 = Boat("Ibiza", "Touring 20") #Create a Boat object
plane1 = Plane("Boeing", "747")     #Create a Plane object

car1.move()
boat1.move()
plane1.move()


# In[ ]:




