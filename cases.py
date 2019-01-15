class Case:
    def __init__(self, name: str, text_code: str):
        self.name = name
        self.text_code = text_code


TEST_CASES = [
    Case(
        name="constant",
        text_code=r"""
print(17)
"""),
    Case(
        name="test102",
        text_code=r"""
d = {1: 2, 3: 4}
print(d[1])
print(d)
"""),
    Case(
        name="test103",
        text_code=r"""
d = "lol"
print(d)
print(5 > 7)
"""),
Case(
        name="test104",
        text_code=r"""
print("lol")
print("5"*5)
"""),
Case(
        name="test105",
        text_code=r"""
print(lol)
"""),
    Case(
        name="test104",
        text_code=r"""
a = [1, 2, 3, 4, 5] +[5]
print(a[3])
print(a)
a[4] = 4
print(a)
print(a.pop())
a.append(8)
print(a)
a.push(7)
print(a)
"""),
    Case(
        name="Class",
        text_code=r"""
class T(object):
    def __init__(self):
        self._a = 1

t = T()
print(t._a)
"""),
    Case(
        name="catching_IndexError",
        text_code=r"""
try:
    [][1]
    print("Shouldn't be here...")
except IndexError:
    print("caught it!")
"""),
    Case(
        name="function namespace",
        text_code=r"""
a = 5
def func(x, y =70):
    f = 6
    print(x)
    print(f)
    print(a)
    a = 3
    print(a)
    print(y)
print(a)
func(4, 1000)
print(a)
"""),
 Case(
        name="for",
        text_code=r"""
for i in range(10):
    print(i)
"""),
 Case(
        name="for",
        text_code=r"""
x = 10
if x % 2 == 0:
    print("lol")
else:
    print("kek")

if x % 2 != 0:
    print("lol")
else:
    print("kek")
"""),
    Case(
        name="for",
        text_code=r"""
x = 5
y = 4
x += y
y+=3
print(x)
print(y)
"""),
    Case(
        name="list comprehension",
        text_code=r"""
print([i for i in range(10)])
"""),
    Case(
        name="set comprehension",
        text_code=r"""
print({i//2 for i in range(10)})
"""),
    Case(
        name="dict comprehension",
        text_code=r"""
print({i//2 : i for i in range(10)})
"""),
    Case(
        name="list slices",
        text_code=r"""
a = [1,2, 3,4,5,6,7]
print(a[:2])
print(a[2:])
print(a[1:2])
print(a[-2:])
"""),
    Case(
        name="list comprehension",
        text_code=r"""
def add_2(x):
    return x + 2

def multiply_and_add_2(x):
    return add_2(2*x)


x = multiply_and_add_2(9)
print(x)
print(2**2)
print(100**100)

if x % 2 == 0:
    print(x-1)
elif x == 1:
    print(-1)
else:
    print("kek")
"""),
    Case(
        name="func_unsual_args_kwargs",
        text_code=r"""
def funcc(*x, **kwargs):
    print(*x)

funcc(*[1,2,3,4,5,6,7,8], lol = "kek")
"""),
    Case(
        name="func_kwargs",
        text_code=r"""
def func(y, z=3, o = 0, **kwargs):
    print(y, z, o)
    print(kwargs)
    kek = 0
    jkfldsa = 10

func(1)
func(2, 11)
func(3, 11, 12)
func(5, z = 40, k = 90)
func(5, z = 40, k = 90, lol = 101)
"""),
    Case(
        name="func_args_kwargs",
        text_code=r"""
def func(y, *args, z=3, x = 10, **kwargs):
    print(y, x)
    print(kwargs)
    print()

func(1)
func(2, 11)
func(3, 11, 12)
func(5, 6, z = 40, k = 90)
func(5, 6, z = 40, k = 90, lol = 101)
"""),
    Case(
        name="strings sum",
        text_code=r"""
print("lol" + " " + "kek")
a = "b"
b = "a"*3
c = a
c += a
b += b
c += "lolololol"
print(a+b+c)
"""),
    Case(
        name="kurva",
        text_code=r"""
def func(y, z=3, x = 10, **kwargs):
    func.__name__ = "kurva"
    print(y, z, x)
    print(kwargs)
    print()

func(1)
func(2, 11)
func(3, 11, 12)
func(4,5,6,7)
func(5, 6, k = 90)
func(5, 6, 0, k = 90, lol = 101)
"""),
    Case(
        name="list_tuple",
        text_code=r"""
a = [[1,2, 3, 4], [5, 6,7 ,8 ], [9 ,10]]
tuple(x for x in a)
"""),

    Case(
        name="list_tuple1",
        text_code=r"""
a = [1,2, 3, 4]
tuple(x for x in a)
"""),
    Case(
        name="list_set",
        text_code=r"""
a = [1,2, 3, 4]
set(x for x in a)
"""),

    Case(
        name="unary",
        text_code=r"""
a = [1,2, 3, 4]
x -= a[3] if a[2] else a[1]
print(x)
x = -a[2]
print(x)
#kurva
a.append(False)
print(x)
a.append(True)
x = +a[2]
print(x)
x = not a[5]
print(x)
x = ~a[4]
print(x)
"""),
    Case(
        name="generetor",
        text_code=r"""
def func():
    array = [1, 2, 3, 4, 5, 6,7 ]
    for element in array:
        yield element

for element in func():
    print(element)
"""),
    Case(
        name="EXTENDED_ARG",
        text_code=r"""
x = false
if x:
    x += 1
    x ** 2
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    x += 1
    x -= 1
    print(x)
else:
    print(hello)    

"""
    ),
    Case(
        name="function with atribute",
        text_code=r"""
def func(*x, **kwargs):
    print(*x)
    func.printed = True


func.printed = False
func(*[1, 2, 3, 4, 5, 6, 7, 8], lol=65)
if not func.printed:
    func(10000000)
"""),
]
