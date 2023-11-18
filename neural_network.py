import inline
import matplotlib
import numpy  # 행렬 수학 연산
import scipy # 활성화 함수 expit()


# 각 계층이 1개이므로 얕은 신경망
class neural_network:
    # 신경망 초기화
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # normal(평균, 표준편차, 행렬(행, 열)) -> 정규분포 -> 평균, 표준편차(퍼진 정도)
        # 가중치는 음수와 양수 모두 설정하는 것이 좋음 -> 평균은 보통 0.0으로 설정
        self.wih = numpy.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        self.lr = learning_rate

        self.activation_func = lambda x: scipy.special.expit(x)

    # 신경망 질의 -> 순전파
    def query(self, input_list):
        # 입력 리스트를 2차원 행렬로 변환 -> T: 전치 행렬
        inputs = numpy.array(input_list, ndmin=2).T

        # 입력 계층 - 은닉 계층
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        # 은닉 계층 - 출력 계층
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs

    # 신경망 학습 -> 역전파
    def train(self, input_list, target_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        # 출력 계층 오차 = 실제값(정답) - 계산값
        output_errors = targets - final_outputs
        # 은닉 계층 오차 = 출력 계층 오차를 역전파
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 가중치 업데이트
        self.who = self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                       numpy.transpose(hidden_outputs))
        self.wih = self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                       numpy.transpose(inputs))
