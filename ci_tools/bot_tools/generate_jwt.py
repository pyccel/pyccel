#!/usr/bin/env python3
import jwt
import time
import sys

def get_jwt(pem):
    # Get PEM
    signing_key = jwt.jwk_from_pem(pem)

    payload = {
        # Issued at time
        'iat': int(time.time()),
        # JWT expiration time (10 minutes maximum)
        'exp': int(time.time()) + 60,
        # GitHub App's identifier
        'iss': 337566
    }

    # Create JWT
    jwt_instance = jwt.JWT()
    encoded_jwt = jwt_instance.encode(payload, signing_key, alg='RS256')

    return encoded_jwt
