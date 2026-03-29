#!/usr/bin/env python

import sys
import struct

n = [10.75, 11.11, 22.22, 33.33]
ba = bytearray(struct.pack("d", n[0]))

for i in range(1, len(n)):
    ba.extend(bytearray(struct.pack("d", n[i])))

sys.stdout.buffer.write(ba)
