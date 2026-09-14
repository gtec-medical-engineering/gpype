#include "gtec_attest_verify.h"
#include "tweetnacl.h"
#include <string.h>

/* 16 bytes, fixed, domain separation: "g.tec-attest-v1\0" */
static const unsigned char GTEC_ATTEST_DOMAIN[GTEC_ATTEST_DOMAIN_LEN] = {
    'g', '.', 't', 'e', 'c', '-', 'a', 't',
    't', 'e', 's', 't', '-', 'v', '1', '\0'
};

#define GTEC_ATTEST_PAYLOAD_MAX \
    (GTEC_ATTEST_DOMAIN_LEN + 2 + GTEC_ATTEST_NONCE_LEN + GTEC_ATTEST_SERIAL_MAX)

int gtec_attest_verify(const unsigned char *attestation, size_t attestation_len,
                       const unsigned char *nonce, size_t nonce_len,
                       const char *serial, size_t serial_len,
                       const gtec_attest_key_t *keys, size_t key_count,
                       unsigned short *out_key_id,
                       gtec_attest_reason_t *out_reason)
{
    unsigned char sm[GTEC_ATTEST_SIGNATURE_LEN + GTEC_ATTEST_PAYLOAD_MAX];
    unsigned char m[GTEC_ATTEST_SIGNATURE_LEN + GTEC_ATTEST_PAYLOAD_MAX];
    unsigned long long mlen = 0;
    const gtec_attest_key_t *key = NULL;
    unsigned short key_id = 0;
    size_t off = 0;
    size_t i;

    if (out_reason) *out_reason = GTEC_ATTEST_MALFORMED;
    if (out_key_id) *out_key_id = 0;

    if (!attestation || !nonce || !serial || !keys) return 0;
    if (attestation_len != GTEC_ATTEST_ATTESTATION_LEN) return 0;
    if (nonce_len != GTEC_ATTEST_NONCE_LEN) return 0;
    if (serial_len == 0 || serial_len > GTEC_ATTEST_SERIAL_MAX) return 0;
    if (attestation[0] != GTEC_ATTEST_VERSION_BYTE) return 0;

    key_id = (unsigned short)(((unsigned)attestation[1] << 8) | attestation[2]);
    if (out_key_id) *out_key_id = key_id;

    for (i = 0; i < key_count; i++) {
        if (keys[i].key_id == key_id) { key = &keys[i]; break; }
    }
    if (!key || !key->public_key) {
        if (out_reason) *out_reason = GTEC_ATTEST_UNKNOWN_KEY_ID;
        return 0;
    }
    if (key->retired) {
        if (out_reason) *out_reason = GTEC_ATTEST_RETIRED_KEY;
        return 0;
    }

    /* TweetNaCl verifies a combined buffer: signature || message */
    memcpy(sm, attestation + 3, GTEC_ATTEST_SIGNATURE_LEN);
    off = GTEC_ATTEST_SIGNATURE_LEN;
    memcpy(sm + off, GTEC_ATTEST_DOMAIN, GTEC_ATTEST_DOMAIN_LEN);
    off += GTEC_ATTEST_DOMAIN_LEN;
    sm[off++] = attestation[1];              /* key_id, big-endian, as sent */
    sm[off++] = attestation[2];
    memcpy(sm + off, nonce, GTEC_ATTEST_NONCE_LEN);
    off += GTEC_ATTEST_NONCE_LEN;
    memcpy(sm + off, serial, serial_len);
    off += serial_len;

    if (crypto_sign_open(m, &mlen, sm, (unsigned long long)off,
                         key->public_key) != 0) {
        if (out_reason) *out_reason = GTEC_ATTEST_BAD_SIGNATURE;
        return 0;
    }

    if (out_reason) *out_reason = GTEC_ATTEST_OK;
    return 1;
}
