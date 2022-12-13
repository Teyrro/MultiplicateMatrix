
from itertools import product
import string

import numpy as np

class Node(object):
	def __init__(self, size = 0, ind = []) -> None:
		self.size = size
		self.ind = ind



	def __eq__(self, __o: object) -> bool:
		return (self.size == __o.size)
	def __lt__(self, other) -> bool:
		return self.size < other.size

def split_to_2x2_blocks(matrix):
	return list(map(
		lambda row: np.hsplit(row, 2),
		np.vsplit(matrix, 2)
	))

def strassen_mul_2x2(lb, rb):
	d = strassen_mul(lb[0][0] + lb[1][1], rb[0][0] + rb[1][1])
	d_1 = strassen_mul(lb[0][1] - lb[1][1], rb[1][0] + rb[1][1])
	d_2 = strassen_mul(lb[1][0] - lb[0][0], rb[0][0] + rb[0][1])

	left = strassen_mul(lb[1][1], rb[1][0] - rb[0][0])
	right = strassen_mul(lb[0][0], rb[0][1] - rb[1][1])
	top = strassen_mul(lb[0][0] + lb[0][1], rb[1][1])
	bottom = strassen_mul(lb[1][0] + lb[1][1], rb[0][0])

	return [[d + d_1 + left - top, right + top],
			[left + bottom, d + d_2 + right - bottom]]

def trivial_mul(left, right):
	height, mid_size = left.shape
	mid_size, width = right.shape

	result = np.zeros((height, width))
	for row, col, mid in product(*map(range, [height, width, mid_size])):
		result[row][col] += left[row][mid] * right[mid][col]

	return result

TRIVIAL_MULTIPLICATION_BOUND = 1

def strassen_mul(left, right):
	assert(left.shape == right.shape)
	assert(left.shape[0] == left.shape[1])

	if left.shape[0] <= TRIVIAL_MULTIPLICATION_BOUND:
		return trivial_mul(left, right)

	assert(left.shape[0] % 2 == 0)
	return np.block(
		strassen_mul_2x2(*map(split_to_2x2_blocks, [left, right]))
	)



def RandAndMultMatrix(rows_columns):
	low = 0
	high = 50
	rows = columns = rows_columns


	matrix_1 = np.random.randint (low, high, (rows, columns))
	matrix_2 = np.random.randint (low, high, (rows, columns))
	m3 = strassen_mul(matrix_1, matrix_2)
	print(matrix_1, "\n\n", matrix_2)
	print()
	print(m3)

def FindBetterWayFMM(listOfSizeMatrix: list):
	data = [[None]] * len(listOfSizeMatrix)
	for i in range(len(data)):
		data[i] = [None] * (len(data))
		data[i][i] = Node(0, [i, i])
	
	for t in range(1, len(data)):
		for k in range(1, len(data) - t):
			ansCell = []
			rightBorder = k+t
			for j in range (k, rightBorder):
				ans = listOfSizeMatrix[k-1] * listOfSizeMatrix[j] * listOfSizeMatrix[rightBorder]
				ansCell.append(Node(data[k][j].size + data[j+1][rightBorder].size + ans, [[k, j], [j+1, rightBorder]]))
			data[k][rightBorder] = min(ansCell)
	return data

def ReverseWay(data : list, string, startD : list):
	node : Node = data[startD[0]][startD[1]]
	if (node.ind[0][0] == node.ind[0][1]):
		string += "M" + str(node.ind[0][0])
	else:
		string += '('
		string = ReverseWay(data, string, node.ind[0])
		string += ")"

	string += " * "

	if (node.ind[1][0] == node.ind[1][1]):
		string += "M" + str(node.ind[1][0])
	else:
		string += "("
		string = ReverseWay(data, string, node.ind[1])
		string += ")"

	return string


def FinalFunc():
	data = FindBetterWayFMM([10,20,50,1,100])
	string = ""
	print(ReverseWay(data, string, [1, len(data)-1]))

FinalFunc()
