TARGET = tm_worker.so
SOURCES = tm_worker.c
OBJECTS = tm_worker.o
LIBS = -lm
CC = gcc
CFLAGS = -fopenmp -g -O3 -Wall -Wextra -Wpedantic -fPIC
LFLAGS = -shared
# -fPIC   Generate position-independent code (PIC) suitable for use in a shared library...

default: $(TARGET)
all: default

#%.o: $(OBJECTS) $(HEADERS)
#	$(CC) -c $(CFLAGS) $< -o $@

$(TARGET): $(SOURCES)
	$(CC) $(SOURCES) $(CFLAGS) $(LFLAGS) $(LIBS) -o $@

clean:
	-rm -f $(OBJECTS)
	-rm -f $(TARGET)
