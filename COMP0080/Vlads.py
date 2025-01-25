import numpy as np
from math import log, exp
from scipy.special import logsumexp

def intToSpinArray(s, n=10):
    arr = []
    for i in range(n):
        bitI = (s >> i) & 1
        spn = +1 if bitI == 1 else -1
        arr.append(spn)
    return np.array(arr, dtype=int)

def countVerticalAlignments(s, n=10):
    cnt = 0
    for i in range(n - 1):
        bitI = (s >> i) & 1
        bitIp1 = (s >> (i + 1)) & 1
        if bitI == bitIp1:
            cnt += 1
    return cnt

def countHorizontalAlignments(s1, s2, n=10):
    c = 0
    for i in range(n):
        bit1 = (s1 >> i) & 1
        bit2 = (s2 >> i) & 1
        if bit1 == bit2:
            c += 1
    return c

def buildLogPotentials(n=10, beta=1.0):
    nstates = 1 << n
    logPhi = np.zeros(nstates, dtype=np.float64)
    logPsi = np.zeros((nstates, nstates), dtype=np.float64)
    for s in range(nstates):
        vcount = countVerticalAlignments(s, n)
        logPhi[s] = beta * vcount
    for s1 in range(nstates):
        for s2 in range(nstates):
            hcount = countHorizontalAlignments(s1, s2, n)
            logPsi[s1, s2] = beta * hcount
    return logPhi, logPsi

def forwardPassChainInference(n=10, beta=1.0):
    logPhi, logPsi = buildLogPotentials(n, beta)
    nstates = len(logPhi)
    nCols = n
    alphaPrev = np.zeros(nstates, dtype=np.float64)
    alphaCurr = np.zeros(nstates, dtype=np.float64)
    alphaPrev[:] = logPhi[:]
    for j in range(1, nCols):
        M = alphaPrev.reshape((nstates, 1)) + logPsi
        alphaCurr = logsumexp(M, axis=0) + logPhi
        alphaPrev, alphaCurr = alphaCurr, alphaPrev
    logZ = logsumexp(alphaPrev)
    logPosteriorColN = alphaPrev - logZ
    return logPosteriorColN

def marginalTopBottom(logPosteriorColN, n=10):
    nstates = 1 << n
    probs = np.exp(logPosteriorColN)
    joint = {(+1, +1): 0.0, (+1, -1): 0.0, (-1, +1): 0.0, (-1, -1): 0.0}
    for s in range(nstates):
        pS = probs[s]
        topBit = (s >> 0) & 1
        botBit = (s >> (n - 1)) & 1
        topSpin = +1 if topBit == 1 else -1
        botSpin = +1 if botBit == 1 else -1
        joint[(topSpin, botSpin)] += pS
    return joint

def runInferenceForBetas(betas=[4.0, 1.0, 0.01], n=10):
    for beta in betas:
        print(f"=== beta = {beta} ===")
        logPostColN = forwardPassChainInference(n=n, beta=beta)
        joint = marginalTopBottom(logPostColN, n=n)
        spinsOrder = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
        total = sum(joint.values())
        for tb in spinsOrder:
            print(f"P(x1,10={tb[0]}, x10,10={tb[1]}) = {joint[tb]:.6f}")
        print(f"Check sum = {total:.6f}\n")

if __name__ == "__main__":
    runInferenceForBetas([4.0, 1.0, 0.01], n=10)
