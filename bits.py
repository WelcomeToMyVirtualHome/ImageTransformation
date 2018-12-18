import numpy as np

def bitfield(n):
    return [bool(int(digit)) for digit in bin(n)[2:]]

def int_to_bitfield(integer,integer_max=8):
    if integer < integer_max:
        n = int(np.log2(integer_max) + 1)
        integer_bits = bitfield(integer)
        length = len(integer_bits)
        if n > length:
            integer_bits.reverse()
            integer_bits.extend(np.zeros(n - length,dtype=np.bool_))
            integer_bits.reverse()
        return integer_bits
    else:
        n = int(np.log2(integer) + 1)
        integer_bits = bitfield(integer)
        return integer_bits

def bitfield_to_int(bitfield):
    return sum([(2*int(bitfield[len(bitfield)-1-i]))**i for i in range(0,len(bitfield))])