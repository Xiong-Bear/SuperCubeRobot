import requests
import json


def solve3(c1, c2, c3):
    t = [c1, c2, c3]
    if 'B' in t and 'W' in t and 'O' in t:
        return {'W': '2', 'O': '20', 'B': '44'}
    elif 'B' in t and 'W' in t and 'R' in t:
        return {'W': '8', 'R': '35', 'B': '38'}
    elif 'O' in t and 'W' in t and 'G' in t:
        return {'W': '0', 'O': '26', 'G': '47'}
    elif 'G' in t and 'W' in t and 'R' in t:
        return {'W': '6', 'R': '29', 'G': '53'}
    elif 'B' in t and 'O' in t and 'Y' in t:
        return {'Y': '9', 'O': '18', 'B': '42'}
    elif 'B' in t and 'R' in t and 'Y' in t:
        return {'Y': '15', 'R': '33', 'B': '36'}
    elif 'G' in t and 'O' in t and 'Y' in t:
        return {'Y': '11', 'O': '24', 'G': '45'}
    elif 'G' in t and 'R' in t and 'Y' in t:
        return {'Y': '17', 'G': '51', 'R': '27'}
    return False


def solve2(c1, c2):
    t = [c1, c2]
    if 'W' in t and 'B' in t:
        return {'W': '5', 'B': '41'}
    elif 'W' in t and 'O' in t:
        return {'W': '1', 'O': '23'}
    elif 'W' in t and 'G' in t:
        return {'W': '3', 'G': '50'}
    elif 'W' in t and 'R' in t:
        return {'W': '7', 'R': '32'}
    elif 'O' in t and 'B' in t:
        return {'O': '19', 'B': '43'}
    elif 'O' in t and 'G' in t:
        return {'O': '25', 'G': '46'}
    elif 'R' in t and 'B' in t:
        return {'R': '34', 'B': '37'}
    elif 'R' in t and 'G' in t:
        return {'R': '28', 'G': '52'}
    elif 'Y' in t and 'B' in t:
        return {'Y': '12', 'B': '39'}
    elif 'Y' in t and 'O' in t:
        return {'Y': '10', 'O': '21'}
    elif 'Y' in t and 'G' in t:
        return {'Y': '14', 'G': '48'}
    elif 'Y' in t and 'R' in t:
        return {'Y': '16', 'R': '30'}
    return False


def solve1(c):
    if c == 'W':
        return '4'
    elif c == 'Y':
        return '13'
    elif c == 'B':
        return '40'
    elif c == 'O':
        return '22'
    elif c == 'R':
        return '31'
    elif c == 'G':
        return '49'
    return False


def solveAll(n, f):
    try:
        t = solve3(n.get('G')[0], n.get('W')[6], n.get('O')[2])
        if t == False:
            print('输入有误')
            return False
        f.get('G')[0] = t.get(n.get('G')[0])
        f.get('W')[6] = t.get(n.get('W')[6])
        f.get('O')[2] = t.get(n.get('O')[2])
        t = solve3(n.get('G')[2], n.get('R')[0], n.get('W')[8])
        if t == False:
            print('输入有误')
            return False
        f.get('G')[2] = t.get(n.get('G')[2])
        f.get('R')[0] = t.get(n.get('R')[0])
        f.get('W')[8] = t.get(n.get('W')[8])
        t = solve3(n.get('G')[6], n.get('O')[8], n.get('Y')[0])
        if not t:
            print('输入有误')
            return False
        f.get('G')[6] = t.get(n.get('G')[6])
        f.get('O')[8] = t.get(n.get('O')[8])
        f.get('Y')[0] = t.get(n.get('Y')[0])
        t = solve3(n.get('G')[8], n.get('Y')[2], n.get('R')[6])
        if t == False:
            print('输入有误')
            return False
        f.get('G')[8] = t.get(n.get('G')[8])
        f.get('Y')[2] = t.get(n.get('Y')[2])
        f.get('R')[6] = t.get(n.get('R')[6])
        t = solve3(n.get('B')[8], n.get('R')[2], n.get('W')[2])
        if t == False:
            print('输入有误')
            return False
        f.get('B')[8] = t.get(n.get('B')[8])
        f.get('R')[2] = t.get(n.get('R')[2])
        f.get('W')[2] = t.get(n.get('W')[2])
        t = solve3(n.get('B')[6], n.get('W')[0], n.get('O')[0])
        if t == False:
            print('输入有误')
            return False
        f.get('B')[6] = t.get(n.get('B')[6])
        f.get('W')[0] = t.get(n.get('W')[0])
        f.get('O')[0] = t.get(n.get('O')[0])
        t = solve3(n.get('B')[2], n.get('R')[8], n.get('Y')[8])
        if t == False:
            print('输入有误')
            return False
        f.get('B')[2] = t.get(n.get('B')[2])
        f.get('R')[8] = t.get(n.get('R')[8])
        f.get('Y')[8] = t.get(n.get('Y')[8])
        t = solve3(n.get('B')[0], n.get('O')[6], n.get('Y')[6])
        if t == False:
            print('输入有误')
            return False
        f.get('B')[0] = t.get(n.get('B')[0])
        f.get('O')[6] = t.get(n.get('O')[6])
        f.get('Y')[6] = t.get(n.get('Y')[6])
        t = solve2(n.get('G')[1], n.get('W')[7])
        if t == False:
            print('输入有误')
            return False
        f.get('G')[1] = t.get(n.get('G')[1])
        f.get('W')[7] = t.get(n.get('W')[7])
        t = solve2(n.get('G')[3], n.get('O')[5])
        if t == False:
            print('输入有误')
            return False
        f.get('G')[3] = t.get(n.get('G')[3])
        f.get('O')[5] = t.get(n.get('O')[5])
        t = solve2(n.get('G')[5], n.get('R')[3])
        if t == False:
            print('输入有误')
            return False
        f.get('G')[5] = t.get(n.get('G')[5])
        f.get('R')[3] = t.get(n.get('R')[3])
        t = solve2(n.get('G')[7], n.get('Y')[1])
        if t == False:
            print('输入有误')
            return False
        f.get('G')[7] = t.get(n.get('G')[7])
        f.get('Y')[1] = t.get(n.get('Y')[1])
        t = solve2(n.get('W')[3], n.get('O')[1])
        if t == False:
            print('输入有误')
            return False
        f.get('W')[3] = t.get(n.get('W')[3])
        f.get('O')[1] = t.get(n.get('O')[1])
        t = solve2(n.get('W')[5], n.get('R')[1])
        if t == False:
            print('输入有误')
            return False
        f.get('W')[5] = t.get(n.get('W')[5])
        f.get('R')[1] = t.get(n.get('R')[1])
        t = solve2(n.get('Y')[3], n.get('O')[7])
        if t == False:
            print('输入有误')
            return False
        f.get('Y')[3] = t.get(n.get('Y')[3])
        f.get('O')[7] = t.get(n.get('O')[7])
        t = solve2(n.get('Y')[5], n.get('R')[7])
        if t == False:
            print('输入有误')
            return False
        f.get('Y')[5] = t.get(n.get('Y')[5])
        f.get('R')[7] = t.get(n.get('R')[7])
        t = solve2(n.get('B')[7], n.get('W')[1])
        if t == False:
            print('输入有误')
            return False
        f.get('B')[7] = t.get(n.get('B')[7])
        f.get('W')[1] = t.get(n.get('W')[1])
        t = solve2(n.get('B')[5], n.get('R')[5])
        if t == False:
            print('输入有误')
            return False
        f.get('B')[5] = t.get(n.get('B')[5])
        f.get('R')[5] = t.get(n.get('R')[5])
        t = solve2(n.get('B')[3], n.get('O')[3])
        if t == False:
            print('输入有误')
            return False
        f.get('B')[3] = t.get(n.get('B')[3])
        f.get('O')[3] = t.get(n.get('O')[3])
        t = solve2(n.get('B')[1], n.get('Y')[7])
        if t == False:
            print('输入有误')
            return False
        f.get('B')[1] = t.get(n.get('B')[1])
        f.get('Y')[7] = t.get(n.get('Y')[7])
        f.get('G')[4] = solve1(n.get('G')[4])
        f.get('O')[4] = solve1(n.get('O')[4])
        f.get('R')[4] = solve1(n.get('R')[4])
        f.get('W')[4] = solve1(n.get('W')[4])
        f.get('Y')[4] = solve1(n.get('Y')[4])
        f.get('B')[4] = solve1(n.get('B')[4])
        return True
    except:
        raise UserWarning("Invalid Input!")


def output(f):
    t = []
    for i in f.get('W'):
        t.append(int(i))
    for i in f.get('Y'):
        t.append(int(i))
    for i in f.get('O'):
        t.append(int(i))
    for i in f.get('R'):
        t.append(int(i))
    for i in f.get('B'):
        t.append(int(i))
    for i in f.get('G'):
        t.append(int(i))
    return t


def color2array(date):
    try:
        f = {'W': ['#' for i in range(9)], 'Y': ['#' for i in range(9)], 'B': ['#' for i in range(9)],
             'G': ['#' for i in range(9)], 'O': ['#' for i in range(9)], 'R': ['#' for i in range(9)]}
        solveAll(date, f)
        result = output(f)
        return result
    except:
        raise UserWarning("Invalid Input!")


def post(state):
    url = 'http://39.97.212.230:5000/solve'
    myobj = {'state': str(state)}
    print(state)
    x = requests.post(url, data=myobj)
    finalData = ''
    for i in x.json()['solve_text']:
        finalData += i + ' '
    finalData = finalData.strip()
    print(finalData)
    return finalData


def parse(result):
    originData = {'W': ['#' for i in range(9)], 'Y': ['#' for i in range(9)], 'B': ['#' for i in range(9)],
                  'G': ['#' for i in range(9)], 'O': ['#' for i in range(9)], 'R': ['#' for i in range(9)]}
    if result == 'q':
        print('用户退出，程序结束')
        exit(0)
    else:
        if solveAll(result, originData):
            # url='http://czx.ac.cn:8080/solve'
            r = output(originData)
            # re=json.dumps(r)
            # myobj={'state':re}
            # x=requests.post(url,data=myobj)
            # # print(x.json()['solve_text'])
            # finalData=''
            # for i in x.json()['solve_text']:
            #     finalData+=i+' '
            # finalData=finalData.strip()
            # print(finalData)
            return post(r)
        else:
            raise UserWarning("Solve failes!")
