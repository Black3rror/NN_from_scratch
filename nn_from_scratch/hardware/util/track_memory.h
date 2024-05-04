// Define macros to override default memory allocation functions

#ifndef TRACK_MEMORY_H
#define TRACK_MEMORY_H
#include "config.h"
#include <stddef.h>
size_t total_allocated;
size_t total_freed;
size_t peak_allocated;
void tracked_free(void *ptr);
void *tracked_calloc(size_t num, size_t size);
void *tracked_malloc(size_t size);
void print_memory();

#define malloc(size) tracked_malloc(size)
#define calloc(num, size) tracked_calloc(num, size)
#define free(ptr) tracked_free(ptr)

struct MemoryBlock
{
    void *ptr;
    size_t size;
    int freed;
};
struct MemoryBlock memoryBlocks[MAX_BLOCKS];

#endif
