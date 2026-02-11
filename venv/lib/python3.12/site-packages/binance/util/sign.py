from dataclasses import dataclass
from urllib.parse import urlencode
from .client import ClientMixin

def sign(query_string: str, *, secret: str) -> str:
  import hmac
  import hashlib
  return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def encode_query(obj) -> str:
  import json
  return (json.dumps(obj, separators=(',', ':'))) # binance can't cope with spaces, it seems

@dataclass
class UserMixin(ClientMixin):
  api_key: str
  api_secret: str

  def sign(self, query_string: str) -> str:
    return sign(query_string, secret=self.api_secret)
  
  def signed_query(self, params: dict) -> str:
    # fix bools, which would show otherwise as "hello=True" instead of "hello=true"
    fixed_params = [(k, str(v).lower() if isinstance(v, bool) else v) for k, v in params.items()]
    query = urlencode(fixed_params)
    return query + '&signature=' + self.sign(query)
  
  @classmethod
  def env(cls):
    import os
    from dotenv import load_dotenv
    load_dotenv()
    errs = []
    if (api_key := os.getenv('API_KEY')) is None:
      errs.append('API_KEY is not set')
    if (api_secret := os.getenv('API_SECRET')) is None:
      errs.append('API_SECRET is not set')
    if errs:
      raise RuntimeError(', '.join(errs))
    return cls(api_key, api_secret) # type: ignore