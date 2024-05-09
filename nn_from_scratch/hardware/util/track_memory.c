#include <stdio.h>
#include <stdlib.h>
#include "track_memory.h"
// undef to avoid recursive loop-call, when macro is defined from header
#undef malloc
#undef free
#undef calloc

// Define global variables to track memory allocation
size_t total_allocated = 0;
size_t total_freed = 0;
size_t peak_allocated;

int num_blocks = 0;
int occupied_blocks = 0;
struct MemoryBlock memoryBlocks[MAX_BLOCKS];
// Custom memory allocation functions

/*
    TODO: Decide to maybe reuse blocks?
*/
void *tracked_malloc(size_t size)
{
    void *ptr = malloc(size);
    if (ptr != NULL)
    {
        if (num_blocks == MAX_BLOCKS)
        {
            printf("Error: Maximum blocks has been used\n");
        }
        else
        {
            memoryBlocks[num_blocks].ptr = ptr;
            memoryBlocks[num_blocks].size = size;
            memoryBlocks[num_blocks].freed = 0;
            occupied_blocks++;
            num_blocks++;
            total_allocated += size;

            if (total_allocated - total_freed > peak_allocated)
            {
                peak_allocated = total_allocated - total_freed;
            }
        }
    }
    return ptr;
}

void *tracked_calloc(size_t num, size_t size)
{
    void *ptr = calloc(num, size);
    if (ptr != NULL)
    {

        if (num_blocks == MAX_BLOCKS)
        {
            printf("Error: Maximum blocks has been used\n");
        }
        else
        {

            memoryBlocks[num_blocks].ptr = ptr;
            memoryBlocks[num_blocks].size = size * num;
            memoryBlocks[num_blocks].freed = 0;
            occupied_blocks++;
            num_blocks++;

            total_allocated += size * num;

            if (total_allocated - total_freed > peak_allocated)
            {
                peak_allocated = total_allocated - total_freed;
            }
        }
    }
    return ptr;
}
void tracked_free(void *ptr)
{

    for (int i = 0; i < num_blocks; i++)
    {
        if (memoryBlocks[i].ptr == ptr && memoryBlocks[i].freed == 0)
        {
            free(ptr);
            memoryBlocks[i].freed = 1;
            total_freed += memoryBlocks[i].size;
            occupied_blocks--;

            if (total_allocated - total_freed > peak_allocated)
            {
                peak_allocated = total_allocated - total_freed;
            }

            return;
        }
    }
    printf("Error: Attempted to free unallocated memory at address %p\n", ptr);
}

/* Will reset memory tracking numbers, if everything is freed*/
void reset_memory_tracking()
{
    if (occupied_blocks != 0)
    {
        printf("Error: Could not reset memory tracking, blocks still being used! \n");
        return;
    }
    total_allocated = 0;
    total_freed = 0;
    num_blocks = 0;
    occupied_blocks = 0;
    peak_allocated = 0;
}
void print_memory()
{
    // Print memory usage
    printf("Peak allocated memory: %zu bytes\n", peak_allocated);
    printf("Total allocated memory: %zu bytes\n", total_allocated);
    printf("Total freed memory: %zu bytes\n", total_freed);
    printf("Peak blocks used: %d\n", num_blocks);
    printf("Total blocks still being used: %d\n", occupied_blocks);
}
