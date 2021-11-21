from typing import List, Dict, Tuple, Union
import numpy as np
from torch import nn
from random import randrange

from cube.cfop.utils.pytorch_models import ResnetModel
from .environment_abstract import Environment, State


class Cube3State(State):
    __slots__ = ['colors', 'hash']

    def __init__(self, colors: np.ndarray):
        self.colors: np.ndarray = colors
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.colors.tostring())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.colors, other.colors)


class Cube3(Environment):
    moves: List[str] = ["%s%i" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F','M','E','S','u','d','l','r','b','f','x','y','z'] for n in [-1, 1]]
    moves_rev: List[str] = ["%s%i" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F','M','E','S','u','d','l','r','b','f','x','y','z'] for n in [1, -1]]

    def __init__(self):
        super().__init__()
        self.dtype = np.uint8
        self.cube_len = 3

        # solved state
        self.goal_colors: np.ndarray = np.arange(0, (self.cube_len ** 2) * 6, 1, dtype=self.dtype)
        self.useful_idx=np.array([4,13,22,31,40,49,10,12,14,16,21,30,39,48],dtype=int)
        # get idxs changed for moves
        self.rotate_idxs_new: Dict[str, np.ndarray]
        self.rotate_idxs_old: Dict[str, np.ndarray]

        self.adj_faces: Dict[int, np.ndarray]
        self._get_adj()

        self.rotate_idxs_new, self.rotate_idxs_old = self._compute_rotation_idxs(self.cube_len, self.moves)

        self.edge_id=self.get_edge_id()

    def next_state(self, states: List[Cube3State], action: int) -> Tuple[List[Cube3State], List[float]]:
        states_np = np.stack([x.colors for x in states], axis=0)
        states_next_np, transition_costs = self._move_np(states_np, action)

        states_next: List[Cube3State] = [Cube3State(x) for x in list(states_next_np)]

        return states_next, transition_costs

    def prev_state(self, states: List[Cube3State], action: int) -> List[Cube3State]:
        move: str = self.moves[action]
        move_rev_idx: int = np.where(np.array(self.moves_rev) == np.array(move))[0][0]

        return self.next_state(states, move_rev_idx)[0]

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[Cube3State], np.ndarray]:
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_colors.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states: List[Cube3State] = [Cube3State(self.goal_colors.copy()) for _ in range(num_states)]

        return solved_states

    def is_solved(self, states: List[Cube3State]) -> np.ndarray:
        states_np = np.stack([state.colors for state in states], axis=0)
        goal_color=np.arange(6).repeat(9)
        is_equal = np.equal(states_np[:,self.useful_idx], np.expand_dims(goal_color[self.useful_idx], 0))

        return np.all(is_equal, axis=1)

    def get_edge_id(self):
        start_id=np.array([50,3],dtype=np.uint8)
        initial_state=np.zeros([54],dtype=self.dtype)
        initial_state[start_id]=1
        action_list=[1,1,1,7,15,15,15,6,3,3,3]
        current_state=initial_state[None,:]
        edge_id=[start_id]
        for action in action_list:
            current_state,_=self._move_np(current_state,action)
            current_id=np.where(current_state[0]==1)[0]
            edge_id.append(current_id)
        return np.array(edge_id)

    # def state_to_nnet_input(self, states: List[Cube3State]) -> List[np.ndarray]:
    #     states_np = np.stack([state.colors for state in states], axis=0)
    #     states_np_select = np.full(states_np.shape,6,states_np.dtype)
    #     for idx in self.useful_idx:
    #         input_color=idx/(self.cube_len ** 2)
    #         states_np_select[np.where(states_np==idx)]=input_color.astype(self.dtype)

    #     representation_np: np.ndarray = states_np_select.astype(self.dtype)

    #     representation: List[np.ndarray] = [representation_np]

    #     return representation
    def state_to_nnet_input(self, states: List[Cube3State]) -> List[np.ndarray]:
        states_np = np.stack([state.colors for state in states], axis=0)
        states_np_select = np.full(states_np.shape,6,states_np.dtype)
        center_id=[4,13,22,31,40,49]
        states_np_select[:,center_id]=states_np[:,center_id]
        edge_color=states_np[:,self.edge_id]
        select_edge=np.any(edge_color==1,axis=2)
        for i in range(len(states)):
            edge_id=self.edge_id[select_edge[i]].reshape(-1)
            states_np_select[i,edge_id]=states_np[i,edge_id]
        
        representation_np: np.ndarray = states_np_select.astype(self.dtype)

        representation: List[np.ndarray] = [representation_np]
        return representation


    def get_num_moves(self) -> int:
        return len(self.moves)

    def get_nnet_model(self) -> nn.Module:
        state_dim: int = (self.cube_len ** 2) * 6
        nnet = ResnetModel(state_dim, 7, 5000, 1000, 4, 1, True)

        return nnet

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[Cube3State], List[int]]:
        assert (num_states > 0)
        assert (backwards_range[0] >= 0)
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = self.get_num_moves()

        # Get goal states
        states_np: np.ndarray = self.generate_goal_states(num_states, np_format=True)

        # Scrambles
        scramble_nums: np.array = np.random.choice(scrambs, num_states)
        num_back_moves: np.array = np.zeros(num_states)

        # Go backward from goal state
        moves_lt = num_back_moves < scramble_nums
        while np.any(moves_lt):
            idxs: np.ndarray = np.where(moves_lt)[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)
            states_np[idxs], _ = self._move_np(states_np[idxs], move)

            num_back_moves[idxs] = num_back_moves[idxs] + 1
            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

        states: List[Cube3State] = [Cube3State(x) for x in list(states_np)]

        return states, scramble_nums.tolist()

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # initialize
        num_states: int = len(states)
        num_env_moves: int = self.get_num_moves()

        states_exp: List[List[State]] = [[] for _ in range(len(states))]

        tc: np.ndarray = np.empty([num_states, num_env_moves])

        # numpy states
        states_np: np.ndarray = np.stack([state.colors for state in states])

        # for each move, get next states, transition costs, and if solved
        move_idx: int
        move: int
        for move_idx in range(num_env_moves):
            # next state
            states_next_np: np.ndarray
            tc_move: List[float]
            states_next_np, tc_move = self._move_np(states_np, move_idx)

            # transition cost
            tc[:, move_idx] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(Cube3State(states_next_np[idx]))

        # make lists
        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        return states_exp, tc_l

    def _move_np(self, states_np: np.ndarray, action: int):
        action_str: str = self.moves[action]

        states_next_np: np.ndarray = states_np.copy()
        states_next_np[:, self.rotate_idxs_new[action_str]] = states_np[:, self.rotate_idxs_old[action_str]]

        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs

    def _get_adj(self) -> None:
        # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5
        self.adj_faces: Dict[int, np.ndarray] = {0: np.array([2, 5, 3, 4]),
                                                 1: np.array([2, 4, 3, 5]),
                                                 2: np.array([0, 4, 1, 5]),
                                                 3: np.array([0, 5, 1, 4]),
                                                 4: np.array([0, 3, 1, 2]),
                                                 5: np.array([0, 2, 1, 3])
                                                 }

    def _compute_rotation_idxs(self, cube_len: int,
                               moves: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        rotate_idxs_new: Dict[str, np.ndarray] = dict()
        rotate_idxs_old: Dict[str, np.ndarray] = dict()

        for move in moves:
            f: str = move[0]
            if f in ['M','E','S','u','d','l','r','b','f','x','y','z']:continue
            sign: int = int(move[1:])

            rotate_idxs_new[move] = np.array([], dtype=int)
            rotate_idxs_old[move] = np.array([], dtype=int)

            colors = np.zeros((6, cube_len, cube_len), dtype=np.int64)
            colors_new = np.copy(colors)

            # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5

            adj_idxs = {0: {2: [range(0, cube_len), cube_len - 1], 3: [range(0, cube_len), cube_len - 1],
                            4: [range(0, cube_len), cube_len - 1], 5: [range(0, cube_len), cube_len - 1]},
                        1: {2: [range(0, cube_len), 0], 3: [range(0, cube_len), 0], 4: [range(0, cube_len), 0],
                            5: [range(0, cube_len), 0]},
                        2: {0: [0, range(0, cube_len)], 1: [0, range(0, cube_len)],
                            4: [cube_len - 1, range(cube_len - 1, -1, -1)], 5: [0, range(0, cube_len)]},
                        3: {0: [cube_len - 1, range(0, cube_len)], 1: [cube_len - 1, range(0, cube_len)],
                            4: [0, range(cube_len - 1, -1, -1)], 5: [cube_len - 1, range(0, cube_len)]},
                        4: {0: [range(0, cube_len), cube_len - 1], 1: [range(cube_len - 1, -1, -1), 0],
                            2: [0, range(0, cube_len)], 3: [cube_len - 1, range(cube_len - 1, -1, -1)]},
                        5: {0: [range(0, cube_len), 0], 1: [range(cube_len - 1, -1, -1), cube_len - 1],
                            2: [cube_len - 1, range(0, cube_len)], 3: [0, range(cube_len - 1, -1, -1)]}
                        }
            face_dict = {'U': 0, 'D': 1, 'L': 2, 'R': 3, 'B': 4, 'F': 5}
            face = face_dict[f]

            faces_to = self.adj_faces[face]
            if sign == 1:
                faces_from = faces_to[(np.arange(0, len(faces_to)) + 1) % len(faces_to)]
            else:
                faces_from = faces_to[(np.arange(len(faces_to) - 1, len(faces_to) - 1 + len(faces_to))) % len(faces_to)]

            cubes_idxs = [[0, range(0, cube_len)], [range(0, cube_len), cube_len - 1],
                          [cube_len - 1, range(cube_len - 1, -1, -1)], [range(cube_len - 1, -1, -1), 0]]
            cubes_to = np.array([0, 1, 2, 3])
            if sign == 1:
                cubes_from = cubes_to[(np.arange(len(cubes_to) - 1, len(cubes_to) - 1 + len(cubes_to))) % len(cubes_to)]
            else:
                cubes_from = cubes_to[(np.arange(0, len(cubes_to)) + 1) % len(cubes_to)]

            for i in range(4):
                idxs_new = [[idx1, idx2] for idx1 in np.array([cubes_idxs[cubes_to[i]][0]]).flatten() for idx2 in
                            np.array([cubes_idxs[cubes_to[i]][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([cubes_idxs[cubes_from[i]][0]]).flatten() for idx2 in
                            np.array([cubes_idxs[cubes_from[i]][1]]).flatten()]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index((face, idxNew[0], idxNew[1]), colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face, idxOld[0], idxOld[1]), colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

            # Rotate adjacent faces
            face_idxs = adj_idxs[face]
            for i in range(0, len(faces_to)):
                face_to = faces_to[i]
                face_from = faces_from[i]
                idxs_new = [[idx1, idx2] for idx1 in np.array([face_idxs[face_to][0]]).flatten() for idx2 in
                            np.array([face_idxs[face_to][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([face_idxs[face_from][0]]).flatten() for idx2 in
                            np.array([face_idxs[face_from][1]]).flatten()]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index((face_to, idxNew[0], idxNew[1]), colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face_from, idxOld[0], idxOld[1]), colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))
        rotate_idxs_new['M1']=np.array([3,4,5,48,49,50,12,13,14,41,40,39],dtype='uint8')
        rotate_idxs_old['M1']=np.array([41,40,39,3,4,5,48,49,50,12,13,14],dtype='uint8')

        rotate_idxs_new['E1']=np.array([19,22,25,46,49,52,28,31,34,37,40,43],dtype='uint8')
        rotate_idxs_old['E1']=np.array([37,40,43,19,22,25,46,49,52,28,31,34],dtype='uint8')

        rotate_idxs_new['S1']=np.array([1,4,7,32,31,30,16,13,10,21,22,23],dtype='uint8')
        rotate_idxs_old['S1']=np.array([21,22,23,1,4,7,32,31,30,16,13,10],dtype='uint8')

        rotate_idxs_new['M-1']=rotate_idxs_old['M1'].copy()
        rotate_idxs_old['M-1']=rotate_idxs_new['M1'].copy()

        rotate_idxs_new['E-1']=rotate_idxs_old['E1'].copy()
        rotate_idxs_old['E-1']=rotate_idxs_new['E1'].copy()

        rotate_idxs_new['S-1']=rotate_idxs_old['S1'].copy()
        rotate_idxs_old['S-1']=rotate_idxs_new['S1'].copy()

        rotate_idxs_new['u1']=np.concatenate((rotate_idxs_new['U1'], rotate_idxs_new['E-1']))
        rotate_idxs_old['u1']=np.concatenate((rotate_idxs_old['U1'], rotate_idxs_old['E-1']))

        rotate_idxs_new['d1']=np.concatenate((rotate_idxs_new['D1'], rotate_idxs_new['E1']))
        rotate_idxs_old['d1']=np.concatenate((rotate_idxs_old['D1'], rotate_idxs_old['E1']))

        rotate_idxs_new['l1']=np.concatenate((rotate_idxs_new['L1'], rotate_idxs_new['M1']))
        rotate_idxs_old['l1']=np.concatenate((rotate_idxs_old['L1'], rotate_idxs_old['M1']))

        rotate_idxs_new['r1']=np.concatenate((rotate_idxs_new['R1'], rotate_idxs_new['M-1']))
        rotate_idxs_old['r1']=np.concatenate((rotate_idxs_old['R1'], rotate_idxs_old['M-1']))

        rotate_idxs_new['b1']=np.concatenate((rotate_idxs_new['B1'], rotate_idxs_new['S-1']))
        rotate_idxs_old['b1']=np.concatenate((rotate_idxs_old['B1'], rotate_idxs_old['S-1']))

        rotate_idxs_new['f1']=np.concatenate((rotate_idxs_new['F1'], rotate_idxs_new['S1']))
        rotate_idxs_old['f1']=np.concatenate((rotate_idxs_old['F1'], rotate_idxs_old['S1']))

        rotate_idxs_new['u-1']=np.concatenate((rotate_idxs_new['U-1'], rotate_idxs_new['E1']))
        rotate_idxs_old['u-1']=np.concatenate((rotate_idxs_old['U-1'], rotate_idxs_old['E1']))

        rotate_idxs_new['d-1']=np.concatenate((rotate_idxs_new['D-1'], rotate_idxs_new['E-1']))
        rotate_idxs_old['d-1']=np.concatenate((rotate_idxs_old['D-1'], rotate_idxs_old['E-1']))

        rotate_idxs_new['l-1']=np.concatenate((rotate_idxs_new['L-1'], rotate_idxs_new['M-1']))
        rotate_idxs_old['l-1']=np.concatenate((rotate_idxs_old['L-1'], rotate_idxs_old['M-1']))

        rotate_idxs_new['r-1']=np.concatenate((rotate_idxs_new['R-1'], rotate_idxs_new['M1']))
        rotate_idxs_old['r-1']=np.concatenate((rotate_idxs_old['R-1'], rotate_idxs_old['M1']))

        rotate_idxs_new['b-1']=np.concatenate((rotate_idxs_new['B-1'], rotate_idxs_new['S1']))
        rotate_idxs_old['b-1']=np.concatenate((rotate_idxs_old['B-1'], rotate_idxs_old['S1']))

        rotate_idxs_new['f-1']=np.concatenate((rotate_idxs_new['F-1'], rotate_idxs_new['S-1']))
        rotate_idxs_old['f-1']=np.concatenate((rotate_idxs_old['F-1'], rotate_idxs_old['S-1']))

        rotate_idxs_new['x1']=np.concatenate((rotate_idxs_new['R1'], rotate_idxs_new['M-1'], rotate_idxs_new['L-1']))
        rotate_idxs_old['x1']=np.concatenate((rotate_idxs_old['R1'], rotate_idxs_old['M-1'], rotate_idxs_old['L-1']))

        rotate_idxs_new['y1']=np.concatenate((rotate_idxs_new['U1'], rotate_idxs_new['E-1'], rotate_idxs_new['D-1']))
        rotate_idxs_old['y1']=np.concatenate((rotate_idxs_old['U1'], rotate_idxs_old['E-1'], rotate_idxs_old['D-1']))

        rotate_idxs_new['z1']=np.concatenate((rotate_idxs_new['F1'], rotate_idxs_new['S1'], rotate_idxs_new['B-1']))
        rotate_idxs_old['z1']=np.concatenate((rotate_idxs_old['F1'], rotate_idxs_old['S1'], rotate_idxs_old['B-1']))

        rotate_idxs_new['x-1']=np.concatenate((rotate_idxs_new['R-1'], rotate_idxs_new['M1'], rotate_idxs_new['L1']))
        rotate_idxs_old['x-1']=np.concatenate((rotate_idxs_old['R-1'], rotate_idxs_old['M1'], rotate_idxs_old['L1']))

        rotate_idxs_new['y-1']=np.concatenate((rotate_idxs_new['U-1'], rotate_idxs_new['E1'], rotate_idxs_new['D1']))
        rotate_idxs_old['y-1']=np.concatenate((rotate_idxs_old['U-1'], rotate_idxs_old['E1'], rotate_idxs_old['D1']))

        rotate_idxs_new['z-1']=np.concatenate((rotate_idxs_new['F-1'], rotate_idxs_new['S-1'], rotate_idxs_new['B1']))
        rotate_idxs_old['z-1']=np.concatenate((rotate_idxs_old['F-1'], rotate_idxs_old['S-1'], rotate_idxs_old['B1']))

        return rotate_idxs_new, rotate_idxs_old
