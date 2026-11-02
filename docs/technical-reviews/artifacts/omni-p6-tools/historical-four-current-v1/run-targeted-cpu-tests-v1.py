"""Offline existing CPU author-patch tests; no model execution."""

import platform
import socket
import sys

import pytest


def blocked(*args, **kwargs):
    raise RuntimeError('network prohibited in independent CPU historical test batch')


socket.create_connection = blocked
socket.getaddrinfo = blocked
socket.socket.connect = blocked
print({'reviewer_task': '/root/omni_p6_technical_tools', 'python': sys.version,
       'platform': platform.platform(), 'network': 'blocked'})
raise SystemExit(pytest.main([
    '-q', '-p', 'no:cacheprovider', 'tests/test_incremental_revision_review.py', '-k',
    'plain_fenced_block or fences_are_not_plain or ending_inside_a_fence or reviewed_version_must_have_plain or closed_plain_block',
]))
