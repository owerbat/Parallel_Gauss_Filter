CC              := g++
STD             := -std=c++17
OPEN_CV         := -I /usr/local/include/opencv4 -L /usr/local/lib
TBB             := -I /home/owerbat/Libraries/tbb/tbb2019_20190206oss/include -L /home/owerbat/Libraries/tbb/tbb2019_20190206oss/lib/intel64/gcc4.7
CFLAGS          := -c -Wall
LIBRARIES       := $(OPEN_CV) $(TBB) -lopencv_core -lopencv_highgui -lopencv_imgcodecs -ltbb -fopenmp
SOURCES         := main.cpp gauss.cpp
OBJECTS         := $(SOURCES:.cpp=.o)
TARGET          := main

.PHONY: all clean bar

all: main tbb

tbb: tbb_presentation.o gauss.o
	$(CC) $(STD) tbb_presentation.o gauss.o $(LIBRARIES) -o tbb

tbb_presentation.o: tbb_presentation.cpp
	$(CC) $(STD) $(CFLAGS) tbb_presentation.cpp $(LIBRARIES)

main: main.o gauss.o
	$(CC) $(STD) main.o gauss.o $(LIBRARIES) -o main

main.o: main.cpp
	$(CC) $(STD) $(CFLAGS) main.cpp $(LIBRARIES)

gauss.o: gauss.cpp
	$(CC) $(STD) $(CFLAGS) gauss.cpp $(LIBRARIES)
        
clean:
	rm -rf main *.o
