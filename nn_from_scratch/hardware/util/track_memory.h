#ifndef TRACK_MEMORY_H
#define TRACK_MEMORY_H
#include "config.h"
#include <stddef.h>
#include <stdint.h>
void tracked_free(void *ptr);
void *tracked_calloc(size_t num, size_t size);
void *tracked_malloc(size_t size);
void print_memory();
void reset_memory_tracking();
#ifdef ENABLE_TRACK_MEMORY
#define malloc(size) tracked_malloc(size)
#define calloc(num, size) tracked_calloc(num, size)
#define free(ptr) tracked_free(ptr)
#endif
typedef struct MemoryBlock
{
    void *ptr;
    size_t size;
    struct MemoryBlock *next;
} MemoryBlock;
#endif
