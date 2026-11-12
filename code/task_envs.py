"""Deterministic task adapters. FrozenLake retains the vendored generator unchanged."""
import copy
import hashlib
from collections import deque
import numpy as np
DIRECTIONS={1:(0,-1),2:(1,0),3:(0,1),4:(-1,0)}
WORDS={1:'Left',2:'Down',3:'Right',4:'Up'}

def map_hash(rows):
    return hashlib.sha256('\n'.join(rows).encode()).hexdigest()

def lake_metadata(env):
    rows=[''.join(x.decode() for x in row) for row in env.desc]
    start=next((r,c) for r,row in enumerate(rows) for c,x in enumerate(row) if x=='S')
    goal=next((r,c) for r,row in enumerate(rows) for c,x in enumerate(row) if x=='G')
    q=deque([(start,[])]);seen={start}
    while q:
        (r,c),path=q.popleft()
        if (r,c)==goal:
            direct=abs(start[0]-goal[0])+abs(start[1]-goal[1])
            return dict(shortest_path_length=len(path),requires_detour=len(path)>direct,map_sha256=map_hash(rows),solution=path)
        for a,(dr,dc) in DIRECTIONS.items():
            p=(r+dr,c+dc)
            if 0<=p[0]<len(rows) and 0<=p[1]<len(rows[0]) and rows[p[0]][p[1]]!='H' and p not in seen:
                seen.add(p);q.append((p,path+[a]))
    raise ValueError('Unsolvable map')

class SokobanEnv:
    """6x6, one box, deterministic push-only moves; sparse terminal success reward.

    Static walls use #, floor _, goal G; dynamic player P and box B. A corner
    deadlock is reported for UCF but does not add a reward or terminate early.
    """
    environment='sokoban'
    def __init__(self,task,max_steps=10):
        self.task=copy.deepcopy(task);self.nrow=self.ncol=6;self.max_steps=max_steps
        self.walls=frozenset(tuple(x) for x in task['walls']);self.goal=tuple(task['goal'])
        self.initial_player=tuple(task['player']);self.initial_box=tuple(task['box'])
        self.initial_distance=self.manhattan(self.initial_box,self.goal)
        assert self.initial_distance>0
        self.desc=np.full((6,6),b'F',dtype='S1')
        for p in self.walls:self.desc[p]=b'H'
        self.desc[self.goal]=b'G';self.desc[self.initial_player]=b'S'
        self.reset()
    @staticmethod
    def manhattan(a,b):return abs(a[0]-b[0])+abs(a[1]-b[1])
    def reset(self):
        self.player=self.initial_player;self.box=self.initial_box;self.turns=0
        return self.render(),{}
    @property
    def s(self):return self.player[0]*6+self.player[1]
    def anchor(self):return (self.s,self.box[0]*6+self.box[1])
    def success(self):return self.box==self.goal
    def blocked(self,p):return not (0<=p[0]<6 and 0<=p[1]<6) or p in self.walls
    def deadlock(self):
        if self.success():return False
        r,c=self.box
        return (self.blocked((r-1,c)) or self.blocked((r+1,c))) and (self.blocked((r,c-1)) or self.blocked((r,c+1)))
    def transition(self,state,action):
        player,box=state
        dr,dc=DIRECTIONS.get(int(action),(0,0));p=(player[0]+dr,player[1]+dc)
        if self.blocked(p):return state
        if p==box:
            b=(box[0]+dr,box[1]+dc)
            if self.blocked(b):return state
            return p,b
        return p,box
    def step(self,action):
        before=(self.player,self.box)
        self.player,self.box=self.transition(before,action);self.turns+=1
        done=self.success() or self.turns>=self.max_steps
        return self.render(),float(self.success()),done,dict(action_is_effective=before!=(self.player,self.box),corner_deadlock=self.deadlock())
    def render(self):
        rows=[['#' if (r,c) in self.walls else '_' for c in range(6)] for r in range(6)]
        rows[self.goal[0]][self.goal[1]]='G'
        rows[self.box[0]][self.box[1]]='*' if self.success() else 'B'
        rows[self.player[0]][self.player[1]]='+' if self.player==self.goal else 'P'
        return '\n'.join(' '.join(row) for row in rows)
    def solution(self,limit=10):
        start=(self.player,self.box);q=deque([(start,[])]);seen={start}
        while q:
            state,path=q.popleft()
            if state[1]==self.goal:return path
            if len(path)>=limit:continue
            for a in DIRECTIONS:
                nxt=self.transition(state,a)
                if nxt not in seen:seen.add(nxt);q.append((nxt,path+[a]))
        return None
    def consequence_classes(self):
        result={0:'STAY'};state=(self.player,self.box)
        for a in DIRECTIONS:
            other=copy.deepcopy(self);other.player,other.box=self.transition(state,a)
            if (other.player,other.box)==state:label='STAY'
            elif other.success():label='GOAL'
            elif other.deadlock():label='FAIL'
            else:label='STATE:'+hashlib.sha256(repr((other.player,other.box)).encode()).hexdigest()
            result[a]=label
        return result

def sokoban_agent(max_steps):
    from patr_core import FrozenLakeAgent
    class Agent(FrozenLakeAgent):
        SYSTEM_PROMPT='''You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You are solving Sokoban.
Goal: Push the single box (B) onto the goal (G). You can push but cannot pull.
Symbols: # Wall | _ Floor | G Goal | P Player | B Box | * Box on goal | + Player on goal.
Only one orthogonal move is performed per turn. Walls block movement and boxes cannot be pushed into walls.
Reward: solved +1, otherwise 0. Plan ahead and avoid pushing the box into non-goal corners.
You should show your thought process and then input the final action in triple backticks.
In every turn, output only the NEXT ACTION in the fence, one of Up, Down, Left, Right. For example: ```Up```.
'''
        def update_from_env(self,observation,reward,done,info,**kwargs):
            super().update_from_env(observation,reward,done,info,**kwargs)
            self.messages[-1]['content']=self.messages[-1]['content'].replace('P has not reached G yet','the box has not reached G yet')
    return Agent(max_steps=max_steps)

def generate_flh_map(size,p,seed):
    """Original draw order/random endpoints; acceptance replaced with exact BFS.

    Never calls or changes vendored is_valid/MAX_STEPS. Each (size,p,seed) replays
    its own RNG from scratch, so the task tuple identifies the accepted map.
    """
    from types import SimpleNamespace
    rng=np.random.default_rng(seed)
    assert 9<=size<=12 and .60<=p<=.75
    while True:
        board=rng.choice(['F','H'],(size,size),p=[p,1-p]).tolist()
        while True:
            sr,sc,gr,gc=[int(rng.integers(0,size)) for _ in range(4)]
            if (sr,sc)!=(gr,gc):break
        board[sr][sc]='S';board[gr][gc]='G'
        rows=[''.join(row) for row in board]
        try:meta=lake_metadata(SimpleNamespace(desc=np.asarray(board,dtype='S1')))
        except ValueError:continue
        if 7<=meta['shortest_path_length']<=16:return rows,meta
