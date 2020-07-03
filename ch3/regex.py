class State:
    next = {}
    idCnt = 0

    def __init__(self):
        self.id = self.idCnt
        self.idCnt = self.idCnt + 1

    def add_next(self, path, state):
        self.next.setdefault(path, []).append(state)


class NFAGraph:

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def add_parallel_graph(self,nfa_graph):
        new_start = State()
        new_end = State()
        
