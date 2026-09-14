/* GENERATED FILE - do not edit by hand.
 *
 * Rendered from keys/registry.json by tools/keys.py.
 * Contains PUBLIC keys only. Signing keys live in the amplifier
 * packages and must never appear here.
 */
#ifndef GTEC_ATTEST_KEYS_H
#define GTEC_ATTEST_KEYS_H

#include "gtec_attest_verify.h"

/* key_id 0x0001  created 2026-08-18  active  first production key */
static const unsigned char GTEC_ATTEST_PK_0001[GTEC_ATTEST_PUBLICKEY_LEN] = {
    0xEFu, 0x4Du, 0x6Cu, 0x33u, 0x79u, 0x06u, 0xACu, 0x8Fu,
    0xD4u, 0x40u, 0x63u, 0xFDu, 0x20u, 0x26u, 0xE7u, 0x0Du,
    0x18u, 0x30u, 0x42u, 0x74u, 0xCCu, 0xF9u, 0x6Eu, 0x3Cu,
    0x5Fu, 0x83u, 0x7Du, 0xECu, 0xE7u, 0x79u, 0x3Du, 0x0Eu
};

static const gtec_attest_key_t GTEC_ATTEST_PRODUCTION_KEYS[] = {
    { 0x0001u, GTEC_ATTEST_PK_0001, 0 },
};

#define GTEC_ATTEST_PRODUCTION_KEY_COUNT \
    (sizeof(GTEC_ATTEST_PRODUCTION_KEYS) / \
     sizeof(GTEC_ATTEST_PRODUCTION_KEYS[0]))

#endif /* GTEC_ATTEST_KEYS_H */
