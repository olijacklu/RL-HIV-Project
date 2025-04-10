import numpy as np
from numba import int32, float32


spec_SumSegmentTree = [
    ('capacity', int32),
    ('tree', float32[:]),
]
class SumSegmentTree(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)

    def sum_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        return _sum_helper(self.tree, start, end, node, node_start, node_end)

    def sum(self, start: int = 0, end: int = 0) -> float:
        if end <= 0:
            end += self.capacity
        end -= 1
        return self.sum_helper(start, end, 1, 0, self.capacity - 1)

    def retrieve(self, upperbound: float) -> int:
        return _sum_retrieve_helper(self.tree, 1, self.capacity, upperbound)

    def __setitem__(self, idx: int, val: float) -> None:
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        _sum_setter_helper(self.tree, idx)

    def __getitem__(self, idx: int) -> float:
        return self.tree[self.capacity + idx]


spec_MinSegmentTree = [
    ('capacity', int32),
    ('tree', float32[:]),
]
INF = float('inf')
class MinSegmentTree(object):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.array([INF for _ in range(2 * capacity)], dtype=np.float32)

    def min_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        return _min_helper(self.tree, start, end, node, node_start, node_end)

    def min(self, start: int = 0, end: int = 0) -> float:
        if end <= 0:
            end += self.capacity
        end -= 1
        return self.min_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float) -> None:
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        _min_setter_helper(self.tree, idx)

    def __getitem__(self, idx: int) -> float:
        return self.tree[self.capacity + idx]


def _sum_helper(tree: np.ndarray, start: int, end: int, node: int, node_start: int, node_end: int) -> np.float32:
    result = 0
    stack = [(start, end, node, node_start, node_end)]
    
    while stack:
        start, end, node, node_start, node_end = stack.pop()
        if start == node_start and end == node_end:
            result += tree[node]
            continue
            
        mid = (node_start + node_end) // 2
        if end <= mid:
            stack.append((start, end, 2 * node, node_start, mid))
        else:
            if mid + 1 <= start:
                stack.append((start, end, 2 * node + 1, mid + 1, node_end))
            else:
                stack.append((start, mid, 2 * node, node_start, mid))
                stack.append((mid + 1, end, 2 * node + 1, mid + 1, node_end))
                
    return result


def _min_helper(tree: np.ndarray, start: int, end: int, node: int, node_start: int, node_end: int) -> np.float32:
    if start == node_start and end == node_end:
        return tree[node]
    mid = (node_start + node_end) // 2
    if end <= mid:
        return _min_helper(tree, start, end, 2 * node, node_start, mid)
    else:
        if mid + 1 <= start:
            return _min_helper(tree, start, end, 2 * node + 1, mid + 1, node_end)
        else:
            a = _min_helper(tree, start, mid, 2 * node, node_start, mid)
            b = _min_helper(tree, mid + 1, end, 2 * node + 1, mid + 1, node_end)
            if a < b: 
                return a
            else:
                return b


def _sum_setter_helper(tree: np.ndarray, idx: int) -> None:
    while idx >= 1:
        tree[idx] = tree[2 * idx] + tree[2 * idx + 1]
        idx = idx // 2


def _min_setter_helper(tree: np.ndarray, idx: int) -> None:
    while idx >= 1:
        a = tree[2 * idx]
        b = tree[2 * idx + 1]
        if a < b:
            tree[idx] = a
        else:
            tree[idx] = b
        idx = idx // 2


def _sum_retrieve_helper(tree: np.ndarray, idx: int, capacity: int, upperbound: float) -> int:
    while idx < capacity:  # while non-leaf
        left = 2 * idx
        right = left + 1
        if tree[left] > upperbound:
            idx = 2 * idx
        else:
            upperbound -= tree[left]
            idx = right
    return idx - capacity
