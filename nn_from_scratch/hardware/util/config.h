
/* Memory tracking configs*/

#include "../settings/user_settings.h"
#ifdef ENABLE_TRACK_MEMORY
#include "track_memory.h"
#endif

#ifndef MAX_BLOCKS
#define MAX_BLOCKS 2500
#endif

/*                         */

/*    PARAMETERS           */
#ifndef BATCH_SIZE
#define BATCH_SIZE 64
#endif

#ifndef N_BATCHES
#define N_BATCHES 1
#endif

#ifndef LEARNING_RATE
#define LEARNING_RATE 0.001
#endif

/*                         */
