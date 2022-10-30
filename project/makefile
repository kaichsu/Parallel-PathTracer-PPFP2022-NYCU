CXX = g++

TARGET = a.out


TARGET = ./raytrace
SRC_DIR = ./src
OBJ_DIR = ./obj
SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, %.o, $(SRC))

vpath %.cpp $(SRC_DIR)
vpath %.o $(OBJ_DIR)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $(addprefix $(OBJ_DIR)/,  $(OBJS))

%.o: %.cpp
	$(CXX) -c $< -o $(OBJ_DIR)/$@

.PHONY: clean
clean:
	rm -rf $(TARGET) $(OBJ_DIR)/*.o 



