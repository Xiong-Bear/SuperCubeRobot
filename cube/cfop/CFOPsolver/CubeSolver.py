from cube.cfop.utils import data_utils, nnet_utils, env_utils
from typing import Dict, List, Tuple, Any

from cube.cfop.environments.environment_abstract import Environment
from cube.cfop.environments import cube3_full
import numpy as np
from cube.cfop.search_methods.gbfs import GBFS
from cube.cfop.search_methods.astar import AStar
from cube.cfop.search_methods import astar
import torch


class CubeSolver:
    def __init__(self, Fformula, Oformula, Pformula):
        self.env: Environment = cube3_full.Cube3()
        self.cube_len = 3
        self.dtype = np.uint8
        self.goal_colors: np.ndarray = np.arange(0, (self.cube_len ** 2) * 6, 1, dtype=self.dtype)

        self.s2i = {}
        for id, s in enumerate(self.env.moves):
            self.s2i[s] = id

        self.F2L_furmula = []
        self.OLL_furmula = []
        self.PLL_furmula = []

        with open(Fformula, 'r') as file_object:
            for line in file_object:
                strings = line.rstrip().split(' ')
                if len(strings) < 2: continue
                action_list = []
                for s in strings:
                    operator = s[0]
                    if "'" in s[1:]:
                        operator += '-1'
                    else:
                        operator += '1'
                    action = self.s2i[operator]
                    if '2' in s:
                        action_list.append(action)
                        action_list.append(action)
                    else:
                        action_list.append(action)
                self.F2L_furmula.append(action_list)
                if action_list[0] == self.s2i['y1'] or action_list[0] == self.s2i['y-1']:
                    self.F2L_furmula.append([action_list[0]] + [1] + action_list[1:])
                    self.F2L_furmula.append([action_list[0]] + [1, 1] + action_list[1:])
                    self.F2L_furmula.append([action_list[0]] + [0] + action_list[1:])
                else:
                    self.F2L_furmula.append([1] + action_list)
                    self.F2L_furmula.append([1, 1] + action_list)
                    self.F2L_furmula.append([0] + action_list)

        with open(Oformula, 'r') as file_object:
            for line in file_object:
                strings = line.rstrip().split(' ')
                if len(strings) < 2: continue
                action_list = []
                for s in strings:
                    operator = s[0]
                    if "'" in s[1:]:
                        operator += '-1'
                    else:
                        operator += '1'
                    action = self.s2i[operator]
                    if '2' in s:
                        action_list.append(action)
                        action_list.append(action)
                    else:
                        action_list.append(action)
                self.OLL_furmula.append(action_list)
                self.OLL_furmula.append([1] + action_list)
                self.OLL_furmula.append([1, 1] + action_list)
                self.OLL_furmula.append([0] + action_list)

        with open(Pformula, 'r') as file_object:
            for line in file_object:
                strings = line.rstrip().split(' ')
                if len(strings) < 2: continue
                action_list = []
                for s in strings:
                    operator = s[0]
                    if "'" in s[1:]:
                        operator += '-1'
                    else:
                        operator += '1'
                    action = self.s2i[operator]
                    if '2' in s:
                        action_list.append(action)
                        action_list.append(action)
                    else:
                        action_list.append(action)
                self.PLL_furmula.append(action_list)
                self.PLL_furmula.append([1] + action_list)
                self.PLL_furmula.append([1, 1] + action_list)
                self.PLL_furmula.append([0] + action_list)

        self.F2L_state = {}
        self.OLL_state = {}
        self.PLL_state = {}

        self.oll_important_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53]
        self.pll_important_id = [20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53]

        for index, f in enumerate(self.F2L_furmula):
            s = self.f2l_formula2code(f)
            if s in self.F2L_state:
                continue
            self.F2L_state[s] = index
        for index, f in enumerate(self.OLL_furmula):
            s = self.oll_formula2code(f)
            if s in self.OLL_state:
                continue
            self.OLL_state[s] = index
        for index, f in enumerate(self.PLL_furmula):
            self.PLL_state[self.pll_formula2code(f)] = index

        self.corner_id = self.get_corner_id()
        self.edge_id = self.get_edge_id()
        # self.device = torch.device('cuda')
        self.device = torch.device('cpu')
        print('cuda available', torch.cuda.is_available())
        self.nnet = nnet_utils.load_nnet('cube/cfop/saved_models/cube3_cross_blind/target/model_state_dict.pt',
                                         self.env.get_nnet_model(), device=self.device)
        self.nnet.to(self.device)

    def CROSS(self, state):
        states = [cube3_full.Cube3State(state.copy())]

        # gbfs = GBFS(states, self.env, eps=None)
        heuristic_fn = nnet_utils.get_heuristic_fn(self.nnet, self.device, self.env, batch_size=10)
        # for _ in range(30):
        #     gbfs.step(heuristic_fn)
        # trajs=gbfs.get_trajs()
        # print(gbfs.get_is_solved())
        # print(gbfs.get_trajs())
        # print(gbfs.get_num_steps())
        astar_solver = AStar(states, self.env, heuristic_fn, [1])
        while not min(astar_solver.has_found_goal()):
            astar_solver.step(heuristic_fn, 10)
        goal_node = astar_solver.get_goal_node_smallest_path_cost(0)
        path, soln, path_cost = astar.get_path(goal_node)
        return soln

    def f2l_formula2code(self, action_sequence):
        current_state = self.goal_colors.copy()
        current_state = current_state[None, :]
        for action in action_sequence[::-1]:
            inv_action = action - 1 if action % 2 else action + 1
            current_state, _ = self.env._move_np(current_state, inv_action)
        current_state = current_state[0]
        f_c_color = current_state[49]
        if f_c_color == 49:
            e = 52
            c = 51
        elif f_c_color == 40:
            e = 43
            c = 42
        elif f_c_color == 31:
            e = 34
            c = 33
        elif f_c_color == 22:
            e = 25
            c = 24
        edgepos = np.where(current_state == e)[0][0]
        cornerpos = np.where(current_state == c)[0][0]
        code = str(edgepos) + '-' + str(cornerpos)
        return code

    def oll_formula2code(self, action_sequence):
        current_state = self.goal_colors.copy()
        current_state = current_state[None, :]
        for action in action_sequence[::-1]:
            inv_action = action - 1 if action % 2 else action + 1
            current_state, _ = self.env._move_np(current_state, inv_action)
        current_state = current_state[0]

        up_state = current_state[self.oll_important_id] < 9

        code = ''
        for b in up_state:
            code = code + '1' if b else code + '0'
        return code

    def pll_formula2code(self, action_sequence):
        current_state = self.goal_colors.copy()
        current_state = current_state[None, :]
        for action in action_sequence[::-1]:
            inv_action = action - 1 if action % 2 else action + 1
            current_state, _ = self.env._move_np(current_state, inv_action)
        current_state = current_state[0]

        up_state = (current_state[self.pll_important_id] / 9).astype(np.uint8)
        code = ''
        for b in up_state:
            code = code + str(b)
        return code

    def PLL(self, color_state):
        re_id0 = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
        re_id1 = np.array([0, 1, 5, 4, 2, 3], dtype=np.uint8)
        re_id2 = np.array([0, 1, 4, 5, 3, 2], dtype=np.uint8)
        re_id3 = np.array([0, 1, 3, 2, 5, 4], dtype=np.uint8)

        up_color = color_state[self.pll_important_id]
        judge_color0 = re_id0[up_color]
        judge_color1 = re_id1[up_color]
        judge_color2 = re_id2[up_color]
        judge_color3 = re_id3[up_color]
        code0 = ''
        code1 = ''
        code2 = ''
        code3 = ''
        for b in judge_color0:
            code0 = code0 + str(b)
        for b in judge_color1:
            code1 = code1 + str(b)
        for b in judge_color2:
            code2 = code2 + str(b)
        for b in judge_color3:
            code3 = code3 + str(b)
        if code0 in self.PLL_state:
            return self.PLL_furmula[self.PLL_state[code0]].copy()
        elif code1 in self.PLL_state:
            return self.PLL_furmula[self.PLL_state[code1]].copy()
        elif code2 in self.PLL_state:
            return self.PLL_furmula[self.PLL_state[code2]].copy()
        elif code3 in self.PLL_state:
            return self.PLL_furmula[self.PLL_state[code3]].copy()
        return []

    def OLL(self, color_state):
        up_state = color_state[self.oll_important_id]
        code = ''
        for b in up_state:
            code = code + '1' if b == 0 else code + '0'
        if code in self.OLL_state:
            return self.OLL_furmula[self.OLL_state[code]].copy()
        else:
            return []

    def get_corner_id(self):
        start_id = np.array([6, 29, 53], dtype=np.uint8)
        initial_state = np.zeros([54], dtype=self.dtype)
        initial_state[start_id] = 1
        action_list = [1, 1, 1, 7, 3, 3, 3]
        current_state = initial_state[None, :]
        corner_id = [start_id]
        for action in action_list:
            current_state, _ = self.env._move_np(current_state, action)
            current_id = np.where(current_state[0] == 1)[0]
            corner_id.append(current_id)
        return np.array(corner_id)

    def get_edge_id(self):
        start_id = np.array([50, 3], dtype=np.uint8)
        initial_state = np.zeros([54], dtype=self.dtype)
        initial_state[start_id] = 1
        action_list = [1, 1, 1, 7, 15, 15, 15, 6, 3, 3, 3]
        current_state = initial_state[None, :]
        edge_id = [start_id]
        for action in action_list:
            current_state, _ = self.env._move_np(current_state, action)
            current_id = np.where(current_state[0] == 1)[0]
            edge_id.append(current_id)
        return np.array(edge_id)

    def F2L_completed_check(self, state):
        idx = [2, 3, 4, 5, 6]
        for start_idx, end_idx in zip(idx[:-1], idx[1:]):
            face_state = state[start_idx * 9:end_idx * 9]
            face_state = face_state[[0, 1, 3, 4, 6, 7]]
            if not (face_state == face_state[3]).all():
                return False
        return True

    def F2L(self, color_state):
        action_y = self.s2i['y1']
        action_y_inv = self.s2i['y-1']
        finished_color = []
        current_state = color_state.copy()
        solver_list = []
        # while not self.F2L_completed_check(current_state):
        while not len(finished_color) == 4:
            simplest_length = 100
            step_solution = None
            simulated_state = current_state.copy()
            step_face = -1
            for i in range(4):
                corner_color = simulated_state[self.corner_id]
                edge_color = simulated_state[self.edge_id]
                target_color_f = simulated_state[49]
                target_color_r = simulated_state[31]
                target_color_d = simulated_state[13]
                if target_color_f in finished_color:
                    if i < 3:
                        simulated_state, _ = self.env._move_np(simulated_state[None, :], action_y)
                        simulated_state = simulated_state[0]
                    continue

                corner_index = np.where(np.any(corner_color == target_color_f, axis=1) * \
                                        np.any(corner_color == target_color_r, axis=1) * \
                                        np.any(corner_color == target_color_d, axis=1))[0][0]
                select_corner = self.corner_id[corner_index, :]
                select_color = corner_color[corner_index, :]
                corner_target_pos = select_corner[select_color == target_color_f][0]

                edge_index = np.where(np.any(edge_color == target_color_f, axis=1) * \
                                      np.any(edge_color == target_color_r, axis=1))[0][0]
                edge = self.edge_id[edge_index, :]
                edge_target_pos = edge[edge_color[edge_index] == target_color_f][0]

                if corner_target_pos == 51 and edge_target_pos == 52:
                    finished_color.append(target_color_f)
                    if i < 3:
                        simulated_state, _ = self.env._move_np(simulated_state[None, :], action_y)
                        simulated_state = simulated_state[0]
                    continue
                code = str(edge_target_pos) + '-' + str(corner_target_pos)
                if code in self.F2L_state:
                    solution = self.F2L_furmula[self.F2L_state[code]].copy()
                    rotation_num = i
                    for bais in range(len(solution)):
                        if solution[bais] == action_y:
                            rotation_num += 1
                        elif solution[bais] == action_y_inv:
                            rotation_num -= 1
                        else:
                            break
                    rotation_num = rotation_num % 4
                    if rotation_num < 3:
                        solution = [action_y] * rotation_num + solution[bais:]
                    else:
                        solution = [action_y_inv] + solution[bais:]
                    length = len(solution)
                else:
                    solution = [action_y] * i + [self.s2i['R1'], self.s2i['U1'], self.s2i['R-1']]
                    length = 99

                if length < simplest_length:
                    simplest_length = length
                    step_solution = solution
                    step_face = target_color_f
                if i < 3:
                    simulated_state, _ = self.env._move_np(simulated_state[None, :], action_y)
                    simulated_state = simulated_state[0]

            current_state = current_state[None, :]
            for action in step_solution:
                current_state, _ = self.env._move_np(current_state, action)
            current_state = current_state[0]
            if simplest_length < 99:
                finished_color.append(step_face)
            solver_list.extend(step_solution)
        return solver_list

    def test_FOP(self, color_state):
        f2l_action = self.F2L(color_state)
        current_state = color_state[None, :].copy()
        for action in f2l_action:
            current_state, _ = solver.env._move_np(current_state, action)
        current_state = current_state[0]
        oll_actions = self.OLL(current_state)
        current_state = current_state[None, :].copy()
        for action in oll_actions:
            current_state, _ = solver.env._move_np(current_state, action)
        current_state = current_state[0]
        pll_actions = self.PLL(current_state)
        print(np.array(solver.env.moves)[f2l_action])
        print(np.array(solver.env.moves)[oll_actions])
        print(np.array(solver.env.moves)[pll_actions])
        return f2l_action + oll_actions + pll_actions

    def __call__(self, color_state):
        cross_action = self.CROSS(color_state)
        current_state = color_state[None, :].copy()
        for action in cross_action:
            current_state, _ = self.env._move_np(current_state, action)
        current_state = current_state[0]
        f2l_action = self.F2L(current_state)
        current_state = current_state[None, :]
        for action in f2l_action:
            current_state, _ = self.env._move_np(current_state, action)
        current_state = current_state[0]
        oll_actions = self.OLL(current_state)
        current_state = current_state[None, :]
        for action in oll_actions:
            current_state, _ = self.env._move_np(current_state, action)
        current_state = current_state[0]
        pll_actions = self.PLL(current_state)
        current_state = current_state[None, :]
        for action in pll_actions:
            current_state, _ = self.env._move_np(current_state, action)
        current_state = current_state[0]
        # append final up-layer-rotation for registration
        if current_state[49] == current_state[32]:
            pll_actions.append(self.s2i['U1'])
        elif current_state[49] == current_state[23]:
            pll_actions.append(self.s2i['U-1'])
        elif current_state[49] == current_state[41]:
            pll_actions.extend([self.s2i['U1'], self.s2i['U1']])

        cat_solution = cross_action + f2l_action + oll_actions + pll_actions
        step_indx = [0] * len(cross_action) + [1] * len(f2l_action) + [2] * len(oll_actions) + [3] * len(pll_actions)
        cat_solution.append(-1)
        step_indx.append(-1)
        final_solution = []
        final_step = []
        current_action = []
        current_step = []
        # print(cat_solution)
        for a, s in zip(cat_solution, step_indx):
            if len(current_action) == 0:
                current_action.append(a)
                current_step.append(s)
                continue
            if current_action[-1] == a:
                current_action.append(a)
                current_step.append(s)
                continue
            inv_a = a - 1 if a % 2 else a + 1
            if current_action[-1] == inv_a:
                current_action.pop()
                current_step.pop()
                continue
            repeat_num = len(current_action)
            repeat_num = repeat_num % 4
            if repeat_num < 3:
                final_solution.extend(current_action[:(repeat_num)])
                final_step.extend(current_step[:(repeat_num)])
            else:
                final_solution.append(current_action[0] - 1 if current_action[0] % 2 else current_action[0] + 1)
                final_step.append(current_step[0])
            # print(final_solution)
            current_action = [a]
            current_step = [s]

        return final_solution, final_step


def get_color(state, useful_idx=np.arange(6 * 9)):
    faces_idx = np.arange(7) * 9
    output_color = np.empty([54, 3], dtype=np.float32)
    color = [[255, 255, 255], [0, 255, 255], [51, 153, 255], [0, 0, 255], [255, 51, 0], [51, 255, 51]]
    for i in range(6):
        output_color[np.where(state == i)] = color[i]
    for id in useful_idx:
        output_color[np.where(state == id)] *= 0.5
    output_color = output_color.astype(np.uint8)
    return output_color


def getResults(state):
    cubesolver = CubeSolver('cube/cfop/formula/F2L.txt', 'cube/cfop/formula/OLL.txt', 'cube/cfop/formula/PLL.txt')

    state = np.array(state)
    FE2STATE = np.array(
        [6, 3, 0, 7, 4, 1, 8, 5, 2, 15, 12, 9, 16, 13, 10, 17, 14, 11, 24, 21, 18, 25, 22, 19, 26, 23, 20, 33, 30, 27,
         34, 31, 28, 35, 32, 29, 38, 41, 44, 37, 40, 43, 36, 39, 42, 51, 48, 45, 52, 49, 46, 53, 50, 47])

    state = state[FE2STATE]
    state = (state / 9).astype('uint8')

    moves = []
    moves_rev = []
    solve_text = []

    print("final_state:", state)
    p, s = cubesolver(state)
    output = np.array(cubesolver.env.moves)[p]
    print("output:", output)

    # p=solver.test_FOP(current_state)
    # p,s=solver(current_state)
    # output=np.array(solver.env.moves)[p]
    method = ['C', 'F', 'O', 'P']
    # print('scramble:',scramble)
    for i, m in enumerate(method):
        solution = output[np.where(np.array(s) == i)]
        b = solution[0]
        b_1 = b[0]
        b_dir = int(b[1:])
        moves.append(str(b_1) + '_' + str(b_dir))
        moves_rev.append(str(b_1) + '_' + str(-b_dir))
        if b_dir == 1:
            solve_text.append(m + ': ' + str(b_1))
        else:
            solve_text.append(m + ': ' + str(b_1) + "'")
        for action in solution[1:]:
            a_1 = action[0]
            dir = int(action[1:])
            moves.append(str(a_1) + '_' + str(dir))
            moves_rev.append(str(a_1) + '_' + str(-dir))
            if dir == 1:
                solve_text.append(str(a_1))
            else:
                solve_text.append(str(a_1) + "'")
        solve_text.append("<br>")

    results = {"moves": moves, "moves_rev": moves_rev, "solve_text": solve_text}

    return results


def get_color(state, useful_idx=np.arange(6 * 9)):
    faces_idx = np.arange(7) * 9
    output_color = np.empty([54, 3], dtype=np.float32)
    color = [[255, 255, 255], [0, 255, 255], [51, 153, 255], [0, 0, 255], [255, 51, 0], [51, 255, 51]]
    for i in range(6):
        output_color[np.where(state == i)] = color[i]
    for id in useful_idx:
        output_color[np.where(state == id)] *= 0.5
    output_color = output_color.astype(np.uint8)
    return output_color


if __name__ == '__main__':
    # solver=CubeSolver('formula/F2L.txt','formula/OLL.txt','formula/PLL.txt')
    # current_state = [51, 32, 26, 30, 4, 3, 2, 19, 36, 9, 39, 29, 28, 13, 14, 38, 5, 45, 27, 16, 44, 21, 22, 46, 8, 52, 42, 15, 50, 47, 23, 31, 34, 6, 48, 11, 35, 41, 24, 10, 40, 37, 17, 7, 0, 20, 43, 33, 25, 49, 1, 18, 12, 53]
    current_state = [47, 12, 20, 19, 4, 52, 11, 5, 44, 24, 34, 26, 37, 13, 30, 35, 7, 0, 8, 3, 33, 1, 22, 41, 17, 16,
                     15, 29, 10, 45, 39, 31, 23, 9, 50, 18, 38, 46, 42, 43, 40, 48, 27, 25, 6, 53, 28, 2, 32, 49, 21,
                     36, 14, 51]
    print(current_state)
    print(getResults(current_state))

    '''
    #import cv2
    solver=CubeSolver('formula/F2L.txt','formula/OLL.txt','formula/PLL.txt')
    #scramble="L D2 R D2 B2 R' B2 L U2 L F2 U R' F U2 R2 B' U2 R U'"
    scramble="U' B R2 U' R F' B' U' R' F' R2 B D2 F' U2 L2 D2 F R2 L2 F2"


    strings=scramble.strip().split(' ')
    action_list=[]
    s2i={}
    for id,s in enumerate(solver.env.moves):
        s2i[s]=id
    for s in strings:
        operator=s[0]
        if "'" in s[1:]:
            operator+='-1'
        else:
            operator+='1'
        action=s2i[operator]
        if '2' in s:
            action_list.append(action)
            action_list.append(action)
        else:
            action_list.append(action)
    current_state=np.arange(6).repeat(9)
    current_state=np.array([0]*9+[1]*9+[2]*9+[3]*9+[4]*9+[5]*9)
    #current_state=np.arange(0,54)
    for action in action_list:
        current_state,_=solver.env._move_np(current_state[None,:],action)
        current_state=current_state[0]

    print("current_state:",current_state)
    #p=solver.test_FOP(current_state)
    p,s=solver(current_state)
    output=np.array(solver.env.moves)[p]
    method=['C','F','O','P']
    print("output:",output)
    print('scramble:',scramble)
    for i,m in enumerate(method):
        print(m,end=':')
        solution=output[np.where(np.array(s)==i)]
        for action in solution:
            a_1=action[0]
            dir=int(action[1:])
            if dir==1:
                print(a_1,end=' ')
            else:
                print(a_1+"'",end=' ')
        print()
    '''

    # print(output[np.where(np.array(s)==0)])
    # print(output[np.where(np.array(s)==1)])
    # print(output[np.where(np.array(s)==2)])
    # print(output[np.where(np.array(s)==3)])
    # print(np.array(solver.env.moves)[p])
    # print(s)

    # faces_idx=np.arange(7)*9
    # face_size=80
    # useful_idx=np.array([4,13,22,31,40,49,10,12,14,16,21,30,39,48],dtype=int)
    # current_color=get_color(current_state,useful_idx)
    # face_name={0:'U',1:'D',2:'L',3:'R',4:'B',5:'F'}
    # for start_idx,end_idx in zip(faces_idx[:-1],faces_idx[1:]):
    #     face_color=current_color[start_idx:end_idx].copy()
    #     face_color=face_color.reshape(3,3,3)
    #     face_color=face_color.transpose(1,0,2)
    #     face_color=face_color[::-1,:,:]
    #     face_color=face_color.repeat(face_size,1)
    #     face_color=face_color.repeat(face_size,0)
    #     for i in range(1,3):
    #         cv2.line(face_color,(i*face_size,0),(i*face_size,3*face_size),[0,0,0],3)
    #         cv2.line(face_color,(0,i*face_size),(3*face_size,i*face_size),[0,0,0],3)
    #     cv2.imshow(face_name[start_idx//9],face_color)
    # cv2.waitKey()
    # for action in p:
    #     current_state,_=solver.env._move_np(current_state[None,:],action)
    #     current_state=current_state[0]
    #     faces_idx=np.arange(7)*9
    #     useful_idx=np.array([4,13,22,31,40,49,10,12,14,16,21,30,39,48],dtype=int)
    #     current_color=get_color(current_state,useful_idx)
    #     face_name={0:'U',1:'D',2:'L',3:'R',4:'B',5:'F'}
    #     for start_idx,end_idx in zip(faces_idx[:-1],faces_idx[1:]):
    #         face_color=current_color[start_idx:end_idx].copy()
    #         face_color=face_color.reshape(3,3,3)
    #         face_color=face_color.transpose(1,0,2)
    #         face_color=face_color[::-1,:,:]
    #         face_color=face_color.repeat(face_size,1)
    #         face_color=face_color.repeat(face_size,0)
    #         for i in range(1,3):
    #             cv2.line(face_color,(i*face_size,0),(i*face_size,3*face_size),[0,0,0],3)
    #             cv2.line(face_color,(0,i*face_size),(3*face_size,i*face_size),[0,0,0],3)
    #         cv2.imshow(face_name[start_idx//9],face_color)

    # cv2.waitKey()
