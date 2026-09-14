/* gtec_attest - Ed25519 attestation verification (pure function).
 *
 * This header plus gtec_attest_verify.c, tweetnacl.c and a key table are the
 * complete verifier. They are intended to be compiled directly into the
 * enforcing product's own compiled unit, so that the check cannot be replaced
 * by shadowing an importable Python module on sys.path.
 *
 * Contains no secrets: public keys only. See README.
 */
#ifndef GTEC_ATTEST_VERIFY_H
#define GTEC_ATTEST_VERIFY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GTEC_ATTEST_VERSION_BYTE     0x01
#define GTEC_ATTEST_ATTESTATION_LEN    67
#define GTEC_ATTEST_NONCE_LEN          32
#define GTEC_ATTEST_SIGNATURE_LEN      64
#define GTEC_ATTEST_PUBLICKEY_LEN      32
#define GTEC_ATTEST_DOMAIN_LEN         16
#define GTEC_ATTEST_SERIAL_MAX        256

typedef enum {
    GTEC_ATTEST_OK              = 0,
    GTEC_ATTEST_BAD_SIGNATURE   = 1,
    GTEC_ATTEST_UNKNOWN_KEY_ID  = 2,
    GTEC_ATTEST_MALFORMED       = 3,
    GTEC_ATTEST_SERIAL_MISMATCH = 4,  /* see note in README: not distinguishable
                                       * from BAD_SIGNATURE with the v1 layout */
    GTEC_ATTEST_RETIRED_KEY     = 5
} gtec_attest_reason_t;

typedef struct {
    unsigned short       key_id;
    const unsigned char *public_key;  /* GTEC_ATTEST_PUBLICKEY_LEN bytes */
    int                  retired;     /* non-zero: known, no longer accepted */
} gtec_attest_key_t;

/* Verify a 67-byte attestation against an expected serial and nonce.
 *
 * Pure: no I/O, no clock, no network, no allocation, no global state.
 * Returns 1 if verified, 0 otherwise. *out_reason is always set.
 */
int gtec_attest_verify(const unsigned char *attestation, size_t attestation_len,
                       const unsigned char *nonce, size_t nonce_len,
                       const char *serial, size_t serial_len,
                       const gtec_attest_key_t *keys, size_t key_count,
                       unsigned short *out_key_id,
                       gtec_attest_reason_t *out_reason);

#ifdef __cplusplus
}
#endif
#endif /* GTEC_ATTEST_VERIFY_H */
