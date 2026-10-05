from hashlib import sha256

expected = b"base A, adapter B"
received = b"base A, adapter C"
print("期待內容指紋", sha256(expected).hexdigest()[:12])
print("收到內容指紋", sha256(received).hexdigest()[:12])
print("內容符合清單", sha256(expected).digest() == sha256(received).digest())
