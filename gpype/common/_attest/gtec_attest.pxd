# Declarations for statically linking the verifier into another compiled unit.
#
# Enforcing products should compile core/gtec_attest_verify.c, core/tweetnacl.c
# and core/randombytes_stub.c into their own extension and call
# gtec_attest_verify() directly, rather than importing gtec_attest at runtime.
# An importable module can be shadowed on sys.path; a statically linked
# function cannot.
#
# In the consumer's .pyx:
#
#     from gtec_attest cimport gtec_attest_verify, gtec_attest_key_t
#
# and add this package's core/ directory to the extension include_dirs and its
# .c files to the extension sources.

cdef extern from "gtec_attest_verify.h":

    int GTEC_ATTEST_ATTESTATION_LEN
    int GTEC_ATTEST_NONCE_LEN
    int GTEC_ATTEST_SIGNATURE_LEN
    int GTEC_ATTEST_PUBLICKEY_LEN
    int GTEC_ATTEST_SERIAL_MAX

    ctypedef enum gtec_attest_reason_t:
        GTEC_ATTEST_OK
        GTEC_ATTEST_BAD_SIGNATURE
        GTEC_ATTEST_UNKNOWN_KEY_ID
        GTEC_ATTEST_MALFORMED
        GTEC_ATTEST_SERIAL_MISMATCH
        GTEC_ATTEST_RETIRED_KEY

    ctypedef struct gtec_attest_key_t:
        unsigned short key_id
        const unsigned char *public_key
        int retired

    int gtec_attest_verify(const unsigned char *attestation,
                           size_t attestation_len,
                           const unsigned char *nonce, size_t nonce_len,
                           const char *serial, size_t serial_len,
                           const gtec_attest_key_t *keys, size_t key_count,
                           unsigned short *out_key_id,
                           gtec_attest_reason_t *out_reason)
