import numpy as np
import matplotlib.pyplot as plt

# Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Classe da Rede Neural Multicamadas
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.1, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Define o tamanho de cada camada
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Inicializa os pesos de cada camada
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i+1]))
            self.weights.append(weight_matrix)
    
    def feedforward(self, x):
        activations = [x]
        input = x
        # Propagação para frente
        for weight in self.weights:
            net_input = np.dot(input, weight)
            activation = sigmoid(net_input)
            activations.append(activation)
            input = activation
        return activations
    
    def backpropagation(self, activations, y_true):
        error = y_true - activations[-1]
        deltas = [error * sigmoid_derivative(activations[-1])]
        
        # Propaga o erro para trás
        for i in reversed(range(len(self.weights)-1)):
            delta = deltas[-1].dot(self.weights[i+1].T) * sigmoid_derivative(activations[i+1])
            deltas.append(delta)
        
        deltas.reverse()
        
        # Atualiza os pesos
        for i in range(len(self.weights)):
            layer_input = np.atleast_2d(activations[i])
            delta = np.atleast_2d(deltas[i])
            self.weights[i] += self.learning_rate * layer_input.T.dot(delta)
    
    def train(self, X, y):
        self.errors = []  # Lista para armazenar o erro em cada época
        for epoch in range(self.epochs):
            for xi, yi in zip(X, y):
                activations = self.feedforward(xi)
                self.backpropagation(activations, yi)
            loss = np.mean(np.square(y - self.predict(X)))
            self.errors.append(loss)
            if epoch % 1000 == 0:
                print(f"Época {epoch}, Erro: {loss}")

    def predict(self, X):
        y_pred = []
        for xi in X:
            activations = self.feedforward(xi)
            y_pred.append(activations[-1])
        return np.array(y_pred)

# Função para rodar os experimentos
def rodar_experimentos():
    # Dados de entrada (função XOR)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    
    y = np.array([
        [0],
        [1],
        [1],
        [0],
    ])
    
    # Definindo diferentes combinações de parâmetros para os experimentos
    experimentos = [
        {'hidden_sizes': [2], 'output_size': 1, 'learning_rate': 0.1, 'epochs': 10000},
        {'hidden_sizes': [4], 'output_size': 1, 'learning_rate': 0.1, 'epochs': 10000},
        {'hidden_sizes': [2], 'output_size': 1, 'learning_rate': 0.01, 'epochs': 10000},
        {'hidden_sizes': [2], 'output_size': 1, 'learning_rate': 0.1, 'epochs': 5000},
        {'hidden_sizes': [6], 'output_size': 1, 'learning_rate': 0.1, 'epochs': 10000},
    ]
    
    # Executando os experimentos
    for i, params in enumerate(experimentos, 1):
        print(f"\nExecutando Experimento {i}...")
        
        # Passando todos os parâmetros necessários para a rede neural
        nn = NeuralNetwork(input_size=2, **params)
        nn.train(X, y)
        
        # Plotando o gráfico de erro durante o treinamento
        plt.plot(nn.errors)
        plt.title(f"Curva de Convergência - Experimento {i}")
        plt.xlabel("Épocas")
        plt.ylabel("Erro Quadrático Médio")
        plt.grid(True)
        plt.tight_layout()
        
        # Salvando o gráfico como imagem
        plt.savefig(f"grafico_experimento_{i}.png")
        plt.close()  # Fecha o gráfico atual para o próximo experimento

        # Exibindo os resultados
        outputs = nn.predict(X)
        print("\nResultados do Experimento", i)
        for xi, yi_pred in zip(X, outputs):
            print(f"Entrada: {xi}, Saída Prevista: {yi_pred.round()}")
    
# Rodando os experimentos
rodar_experimentos()
