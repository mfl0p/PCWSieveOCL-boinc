/* util.h -- (C) Geoffrey Reynolds, March 2009.


   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
*/

#ifndef _UTIL_H
#define _UTIL_H 1

#ifdef __cplusplus
extern "C" {
#endif

int parse_uint(unsigned int *result, const char *str,
               unsigned int lo, unsigned int hi);
int parse_uint64(uint64_t *result, const char *str,
                 uint64_t lo, uint64_t hi);

#ifdef __cplusplus
}
#endif

#endif /* _UTIL_H */
