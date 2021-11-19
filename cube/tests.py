from cube.solver import CubeSolver

data = {
    'state': [38, 16, 18, 39, 4, 19, 15, 48, 0, 2, 3, 27, 46, 13, 7, 24, 23, 6, 35, 12, 36, 52, 22, 34, 11, 25, 20, 47,
              43, 42, 21, 31, 5, 51, 32, 29, 45, 1, 53, 28, 40, 41, 8, 30, 9, 33, 14, 26, 37, 49, 10, 44, 50, 17]}
# data = json.dumps(data)
# print(data['state'])
state = []
# data['state'] = ast.literal_eval(data['state'])
for i in data['state']:
    state.append(int(i))
# solver=CubeSolver('../formula/F2L.txt','../formula/OLL.txt','../formula/PLL.txt')
print("input state:", state)
result = CubeSolver.getResults(state)
print("complete!")
print("result form:", result)
