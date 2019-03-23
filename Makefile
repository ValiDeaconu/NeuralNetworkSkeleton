OBJ=NeuralNetwork
ATR=-Wall -Wextra

make:
	g++ $(ATR) src/*.cpp -o $(OBJ)