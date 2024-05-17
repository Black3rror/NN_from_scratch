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
size_t peak_allocated = 0;

int num_blocks = 0;
int occupied_blocks = 0;
MemoryBlock *headBlock = NULL;
MemoryBlock *currBlock = NULL;
void *tracked_malloc(size_t size)
{
    void *ptr = malloc(size);
    if (ptr != NULL)
    {
        MemoryBlock *block = (MemoryBlock *)malloc(sizeof(MemoryBlock));
        block->ptr = ptr;
        block->size = size;
        block->next = NULL;

        if (headBlock == NULL)
        {
            headBlock = block;
            currBlock = headBlock;
        }
        else
        {
            currBlock->next = block;
            currBlock = block;
        }
        occupied_blocks++;
        num_blocks++;
        total_allocated += size;

        if (total_allocated - total_freed > peak_allocated)
        {
            peak_allocated = total_allocated - total_freed;
        }
    }
    return ptr;
}

void *tracked_calloc(size_t num, size_t size)
{
    void *ptr = calloc(num, size);
    if (ptr != NULL)
    {
        MemoryBlock *block = (MemoryBlock *)malloc(sizeof(MemoryBlock));
        block->ptr = ptr;
        block->size = size * num;
        block->next = NULL;

        if (headBlock == NULL)
        {
            headBlock = block;
            currBlock = headBlock;
        }
        else
        {
            currBlock->next = block;
            currBlock = block;
        }

        occupied_blocks++;
        num_blocks++;
        total_allocated += size * num;

        if (total_allocated - total_freed > peak_allocated)
        {
            peak_allocated = total_allocated - total_freed;
        }
    }
    return ptr;
}
void tracked_free(void *ptr)
{
    MemoryBlock *block = headBlock;
    MemoryBlock *prevBlock = NULL;

    if (block == NULL)
    {
        printf("Error: Attempted to free unallocated memory at address %p\n", ptr);
        return;
    }

    // If the block to be freed is the first block
    if (block->ptr == ptr)
    {
        headBlock = block->next;
        total_freed += block->size;
        occupied_blocks--;
        if (total_allocated - total_freed > peak_allocated)
        {
            peak_allocated = total_allocated - total_freed;
        }
        free(ptr);
        free(block);
        return;
    }

    // Traverse the linked list to find the block to free
    while (block != NULL)
    {
        if (block->ptr == ptr)
        {
            // Update bookkeeping information
            total_freed += block->size;
            occupied_blocks--;

            if (total_allocated - total_freed > peak_allocated)
            {
                peak_allocated = total_allocated - total_freed;
            }

            // Update the previous block's next pointer
            prevBlock->next = block->next;

            // If the current block is the current allocation block,
            // update the currBlock pointer
            if (block == currBlock)
            {
                currBlock = prevBlock;
            }

            // Free the memory block and the allocated memory
            free(ptr);
            free(block);
            return;
        }
        prevBlock = block;
        block = block->next;
    }

    // If the pointer does not match any tracked memory block
    printf("Error: Attempted to free untracked memory at address %p\n", ptr);
}

/* Will reset memory tracking numbers, if everything is freed*/
void reset_memory_tracking()
{
    if (headBlock != NULL)
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
