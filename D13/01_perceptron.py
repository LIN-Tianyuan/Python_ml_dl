"""
Realization of perceptual machines
"""


def AND(x1, x2):
    w1, w2 = 0.5, 0.5
    theta = 0.7
    temp = x1 * w1 + x2 * w2
    if temp <= theta:
        return 0
    else:
        return 1


print(AND(1, 1))    # 1
print(AND(1, 0))    # 0
print(AND(0, 1))    # 0
print(AND(0, 0))    # 0


def OR(x1, x2):
    w1, w2 = 0.5, 0.5
    theta = 0.3
    temp = x1 * w1 + x2 * w2
    if temp <= theta:
        return 0
    else:
        return 1


print(OR(1, 1))    # 1
print(OR(1, 0))    # 1
print(OR(0, 1))    # 1
print(OR(0, 0))    # 0


# logical exception or argument
def XOR(x1, x2):
    s1 = not AND(x1, x2)
    s2 = OR(x1, x2)
    res = AND(s1, s2)
    return res


print(XOR(1, 1))    # 0
print(XOR(1, 0))    # 1
print(XOR(0, 1))    # 1
print(XOR(0, 0))    # 0
