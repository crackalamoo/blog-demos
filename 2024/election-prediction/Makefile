SOURCES := election.cpp
OBJS := $(SOURCES:.cpp=.o)

election: $(OBJS)
	g++ -o election $(OBJS)

.cpp.o:
	g++ -c -std=c++11 $<

clean:
	rm -f election $(OBJS)

all: election