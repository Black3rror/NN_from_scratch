#include "inlines.h"

int num(int x)
{
    return x;
}

#define num_v(macro)  \
    int num_##macro() \
    {                 \
        return macro; \
    }

num_v(m1);