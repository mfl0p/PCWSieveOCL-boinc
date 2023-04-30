/* util.c -- (C) Geoffrey Reynolds, March 2009.


   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
*/

#include <errno.h>
#include <stdlib.h>
#include "stdint.h"
#include "putil.h"


/* Returns 0 if successful, -1 if cannot parse, -2 if out of range.
 */
int parse_uint(uint32_t *result, const char *str,
               uint32_t lo, uint32_t hi)
{
  uint64_t result64;
  int status;

  status = parse_uint64(&result64,str,lo,hi);

  if (status == 0)
    *result = (uint32_t)result64;

  return status;
}

/* Returns 0 if successful, -1 if cannot parse, -2 if out of range.
 */
int parse_uint64(uint64_t *result, const char *str,
                 uint64_t lo, uint64_t hi)
{
  uint64_t num;
  uint32_t expt;
  char *tail;

  expt = 0;
  errno = 0;
  num = strtoull(str,&tail,0);

  if (errno != 0 || num > hi)
    return -2;

  switch (*tail)
  {
    case 'P': expt += 3;
    case 'T': expt += 3;
    case 'G': expt += 3;
    case 'M': expt += 3;
    case 'K': expt += 3;
      if (tail[1] != '\0')
        return -1;
      for ( ; expt > 0; expt -= 3)
        if (num > hi/1000)
          return -2;
        else
          num *= 1000;
      break;

    case 'e':
    case 'E':
      expt = strtoul(tail+1,&tail,0);
      if (errno != 0)
        return -2;
      if (*tail != '\0')
        return -1;
      while (expt-- > 0)
        if (num > hi/10)
          return -2;
        else
          num *= 10;
      break;

    case 'p': expt += 10;
    case 't': expt += 10;
    case 'g': expt += 10;
    case 'm': expt += 10;
    case 'k': expt += 10;
      if (tail[1] != '\0')
        return -1;
      if (num > (hi>>expt))
        return -2;
      num <<= expt;
      break;

    case 'b':
    case 'B':
      expt = strtoul(tail+1,&tail,0);
      if (errno != 0)
        return -2;
      if (*tail != '\0')
        return -1;
      while (expt-- > 0)
        if (num > (hi>>1))
          return -2;
        else
          num <<= 1;
      break;

    case '\0':
      break;

    default:
      return -1;
  }

  if (num < lo)
    return -2;

  *result = num;
  return 0;
}


