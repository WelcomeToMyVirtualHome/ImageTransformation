CXX=g++
CXXFLAGS=-std=c++17 -Wall -pedantic -O3
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)
all:    main
main:    main.o  
	$(CXX) -o main main.o $(CXXFLAGS) $(LIBS)
main.o: main.cpp
	$(CXX) -o main.o -c main.cpp $(CXXFLAGS)
clean:
	rm -f *.o
