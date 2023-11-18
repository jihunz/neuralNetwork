from neural_network import neural_network

# 노드의 개수가 각 계층마다 3개 -> fully connected
input = 3
hidden = 3
output = 3
# 학습률은 보통 0.001 정도 -> 0.3이면 학습률이 높은 편 / 초기 가중치는 보통 낮게 할당
learning_rate = 0.3

n = neural_network(input, hidden, output, learning_rate)
query_result = n.query([1.0, 0.5, -1.5])
print(query_result)