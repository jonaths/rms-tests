from tools.history import History


def is_bad(number, num_rows=6):
    test = number % num_rows
    print(number, test)
    if test == 0:
        return True
    elif test == num_rows - 1:
        return True
    else:
        return False


bad = []
for i in range(6*12):
    if is_bad(i):
        bad.append(i)

print(bad)