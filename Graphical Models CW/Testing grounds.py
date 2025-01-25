from string import ascii_letters
chrs = ascii_letters + " " + "0123456789"
ciphertext = [
   11, 193, 15, 47, 203, 207, 227, 159, 18, 49, 195,
   21, 36, 208, 24, 47, 189, 23, 44, 203, 17, 54,
   124, 9, 50, 206, 195, 54, 203, 15, 57, 197, 17,
   42, 124, 23, 43, 193, 195, 10, 189, 16, 37, 197,
   23, 227, 191, 11, 36, 200, 15, 40, 202, 10, 40,
   138, 195, 19, 200, 8, 36, 207, 8, 227, 207, 8,
   49, 192, 195, 60, 203, 24, 53, 124, 22, 50, 200,
   24, 55, 197, 18, 49, 124, 4, 49, 192, 195, 6,
   178, 195, 55, 203, 195, 44, 191, 4, 49, 191, 18,
   39, 193, 227, 42, 189, 16, 37, 197, 23, 53, 193,
   22, 40, 189, 21, 38, 196, 209, 38, 203, 16, 227,
   205, 24, 50, 208, 12, 49, 195, 195, 53, 193, 9,
   40, 206, 8, 49, 191, 8, 253, 124, 4, 249, 146,
   216, 251, 190, 218, 41, 194, 211
]


def decrypt(ciphertext, a, b, c):
    """
    dec
    """
    out = []
    for i, val in enumerate(ciphertext):
        if i % 3 == 0:
            original = (val - a) % 256
        elif i % 3 == 1:
            original = (val - b) % 256
        else:
            original = (val - c) % 256
        out.append(original)
    return bytes(out)

def valid(val):
    """
    iv?
    """
    if chr(val) in chrs:
        return True
    return False

def bestShift(values):
    """
    bsv
    """
    bestShift = 0
    bestScore = -1
    for sc in range(256):
        score = 0
        for v in values:
            pt = (v - sc) % 256
            if valid(pt):
                score += 1
        if score > bestScore:
            bestScore = score
            bestShift = sc
    return bestShift, bestScore

group0 = ciphertext[0::3]
group1 = ciphertext[1::3]
group2 = ciphertext[2::3]


a, score0 = bestShift(group0)
b, score1 = bestShift(group1)
c, score2 = bestShift(group2)

print(f"[+] best a={a} score={score0}")
print(f"[+] best b={b} score={score1}")
print(f"[+] best c={c} score={score2}")

bytes = decrypt(ciphertext, a, b, c)

try:
    text = bytes.decode('ascii', errors='replace')
except:
    text = ''.join(chr(b) for b in bytes)

print("\n[+] Decrypted text guess:\n")
print(text)
