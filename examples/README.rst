Examples
========

Runnable examples. The ones that use Jev read its answers from the answer
cache committed under ``benchmarks/cache``, so they run without an API key and
cost nothing. They set ``max_cost_usd=0`` so that a cache miss raises instead
of calling the API; raise the cap, and set ``TYPESAFE_API_KEY``, to run them
against Jev on your own data.
