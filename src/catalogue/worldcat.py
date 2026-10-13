"""WorldCat Search API 2.0 client (STUB, optional).

Requires institutional OCLC credentials (OCLC_CLIENT_ID / OCLC_CLIENT_SECRET in .env) and OAuth 2.0.
Endpoint: /worldcat/search/v2/brief-bibs, max 50 records per request (limit/offset).
"""


def get_access_token():
    raise NotImplementedError


def search_brief_bibs(query, offset=1, limit=50):
    raise NotImplementedError
