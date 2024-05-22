# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra -Werror -std=c99

# Source files
SRCS = .\tester.c .\model\model.c .\util\track_memory.c .\src\model_fc.c .\util\forward_prop.c .\data\eqcheck_data.c .\data\true_data.c .\data\ft_data.c .\util\back_prop.c .\util\loss_functions.c .\util\activation_functions.c .\util\model_binding.c .\src\partial_model_fc.c .\util\model_gradients.c

# Object files
OBJS = $(SRCS:.c=.o)

# Executable name
TARGET = out

# Default target
all: $(TARGET)

# Rule for compiling source files into executable
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET).exe


# Clean rule
clean:
	del /Q $(TARGET).exe



