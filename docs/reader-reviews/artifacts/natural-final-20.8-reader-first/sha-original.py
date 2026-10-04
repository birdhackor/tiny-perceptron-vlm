from hashlib import sha256

original = b"base revision A, adapter revision B"
received = b"base revision A, adapter revision C"
expected_fingerprint = sha256(original).hexdigest()
received_fingerprint = sha256(received).hexdigest()
print("期待指紋", expected_fingerprint[:12])
print("收到指紋", received_fingerprint[:12])
print("內容符合交付清單", expected_fingerprint == received_fingerprint)
