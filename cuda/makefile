CXX = g++
CXXFLAGS = -fopenmp -std=c++17 -O3
TARGET = a.out 


TARGET = ./raytrace
INC_DIR = ./includes
SRC_DIR = ./src
OBJ_DIR = ./obj
SRC = $(wildcard $(SRC_DIR)/*.cpp)
INCS = $(wildcard $(INC_DIR)/*.hpp)

vpath %.hpp $(LIB_DIR)
vpath %.cpp $(SRC_DIR)
vpath %.o $(OBJ_DIR)


$(TARGET): $(SRC_DIR)/main.cu
	nvcc -I. -rdc=true -use_fast_math -O3 $(SRC_DIR)/main.cu -o $(TARGET)

run: $(TARGET)
	$(TARGET)


.PHONY: clean c objs srcs
clean:
	rm -rf $(TARGET) $(OBJ_DIR)/*.o $(IMG) *.o
c: 
	make clean

