/* TweetNaCl declares randombytes() extern because its key generation and
 * secret-box paths need entropy. gtec_attest verifies only: it never
 * generates a key and never produces a signature. Providing a stub that
 * aborts satisfies the linker and makes accidental use of those paths a
 * crash rather than a silent weak key.
 */
#include <stdlib.h>

void randombytes(unsigned char *x, unsigned long long xlen)
{
    (void)x;
    (void)xlen;
    abort();
}
