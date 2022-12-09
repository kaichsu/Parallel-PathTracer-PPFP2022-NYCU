CXX = g++
CXXFLAGS = -std=c++17 -O3
TARGET = a.out


TARGET = ./raytrace
INC_DIR = ./includes
SRC_DIR = ./src
OBJ_DIR = ./obj
SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, %.o, $(SRC))
INCS = $(wildcard $(INC_DIR)/*.hpp)

vpath %.hpp $(LIB_DIR)
vpath %.cpp $(SRC_DIR)
vpath %.o $(OBJ_DIR)


$(TARGET): $(OBJS)
	$(CXX) -o $@ $(addprefix $(OBJ_DIR)/,  $(OBJS))

run: $(TARGET)
	$(TARGET)

%.o: %.cpp $(INCS)
	$(CXX) -c $(CXXFLAGS) $< -o $(OBJ_DIR)/$@ 


.PHONY: clean c objs srcs
clean:
	rm -rf $(TARGET) $(OBJ_DIR)/*.o $(IMG) *.o
c: 
	make clean

