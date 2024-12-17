# !pip install sympy
# !pip install galois
import sympy
import galois
import numpy as np


#################### Input to Integer Matrix ####################

# The values q and n will be global.

q = input("In the finite field F_q such that q is a prime.\nq = ")
q = int(q)
N = input("N, the length of the codes.\nN = ")
N = int(N)

while(not sympy.isprime(q)):
    q = input(f"{q} is not a prime, enter a prime number.\nq = ")
    q = int(q)


# The following algorithm converts string data
# (which consists of codes in F_q^n) to matrix
# in which each row stores the code, and the columns are the coordinates.

def getMatrix(q, n):
    print(f"Detected: q = {q} and N = {n}")
    print(f"Enter the code A1A2A3...AN such that Ai an element of F_{q} by typing them separately by space.")
    print(f"For example, if N=3 and q=11, and A1 = 1, A2 = 10, A3 = 3, then type \"1 10 3\".")
    print("Please enter each of the Ai only with integers.")

    k = input("\nHow many strings? Enter a natural number: ")
    k = int(k)
    M = [[0 for i in range(N)] for j in range(k)]
    for i in range(k):
        b = input(f"Code {i+1}: ")
        numlist = b.split()
        while(len(numlist) != N):
            print(f"The length is not {N}, try again.")
            b = input(f"Code {i+1}: ")
            numlist = b.split()
        numlist = np.array(numlist)
        numlist = numlist.astype('int64')
        M[i] = np.asarray(numlist)
    return M

M = getMatrix(q,N) # Basically, M is just the inputted code in matrix format


#################### Finite Field and Matrix Initiation ####################

# C is the k x n matrix with elements in the GF(q) field.
print("Generating the matrix C...")

GF = galois.GF(q)
C = M
# C = np.transpose(C)
k = len(C)
n = len(C[0])
C = np.matrix(C)
C = GF(C%q)

print(C)


#################### Matrix to GF(q) & Creating Elementary Matrices ####################

# The following three procedures is for creating the three elementary matrices.
# To do the elementary row operation, do:
#                              row_______(k, parameters) @ C
# where C is the matrix that we needed to do elementary row operations with.

# -----------------------------------------------------------------------
# To swap the a+1 -th and b+1 -th row.
def rowSwap(k,a,b,ffield):
    N = [[0 for i in range(k)] for j in range(k)]
    for i in range(k): N[i][i] = 1
    N[a][a] = 0
    N[b][b] = 0
    N[a][b] = 1
    N[b][a] = 1
    N = ffield(N)
    return N
# -----------------------------------------------------------------------
# To multiply the a+1 -th row with x.
def rowMult(k,a,x,ffield):
    N = [[0 for i in range(k)] for j in range(k)]
    N = np.matrix(N)
    N = ffield(N)
    for i in range(k): N[i,i] = 1
    N[a] = N[a] * x
    return N
# -----------------------------------------------------------------------
# To add the b+1 -th row with x times of the a+1 -th row.
def rowMAdd(k,a,b,x,ffield):
    N = [[0 for i in range(k)] for j in range(k)]
    N = np.matrix(N)
    for i in range(k): N[i,i] = 1
    N[a,b] = x
    N = ffield(N)
    return N
# -----------------------------------------------------------------------


#################### Gaussian Elimination Routines ####################

# swapNonZero(C, k, r, c)
# swaps the (r,c)-th element (technically rows) in C for a non-zero element at (i,c), 
# with r ≤ i ≤ k.
# Essentially, this creates so that the pivot is non-zero, making the elimination possible.

def swapNonZero(C, Kc, row, col, ffield):
    for i in range(row, Kc):
        if (C[i][col] != 0):
            C = rowSwap(Kc,row,i,ffield) @ C
            break
    return C
# -----------------------------------------------------------------------

# searchNonZero(C, k, n, r, c)
# searches for the non-zero pivot next available 
# after eliminating all the (i,c)-th elements 
# with r+1 ≤ i ≤ k such that r < k and c < n.

def searchNonZero(C, Kc, Nc, row, col):
    if (row == Kc-1) or (col == Nc):
        return [row+1, col+1]
    else:
        Row = row + 1
        Col = col + 1
        while(Col < Nc):
            if C[Row][Col]!=0:
                return [Row, Col]
            Col = Col + 1
        return [Row, Col]
# -----------------------------------------------------------------------

# gaussNonZero(C, k, n, r, c)
# exhibit the same behaviour as searchNonZero
# but instead it searches the first non-zero element
# that is encountered on the rectangular block spanned by (r+1, c+1) and (k,n) 
# and swaps the first non-zero element that is found, column-wise.

def gaussNonZero(C, Kc, Nc, row, col, ffield):
    if (row == Kc-1) or (col == Nc):
        return C
    else:
        Row = row + 1
        Col = col + 1
        while(Col < Nc):
            C = swapNonZero(C, Kc, Row, Col, ffield)
            if C[Row][Col]!=0:
                break
            Col = Col + 1
        return C


#################### Gaussian Elimination: Row Echelon Form and Reduced Row Echelon Form ####################

# RowEchelonForm(C)
# makes the matrix C into row echelon form.

def RowEchelonForm(C, ffield):
    pivot = [0, 0]
    k = len(C)
    n = len(C[0])
    C = gaussNonZero(C, k, n, -1, -1, ffield)
    pivot = searchNonZero(C, k, n, -1, -1)
    while(pivot[0]<k-1) and (pivot[1]<n):
        for j in range(pivot[0]+1, k):
            m = (-1)*C[j][pivot[1]] // C[pivot[0]][pivot[1]]
            C = rowMAdd(k, j, pivot[0], m, ffield) @ C
        C = gaussNonZero(C, k, n, pivot[0], pivot[1], ffield)
        pivot = searchNonZero(C, k, n, pivot[0], pivot[1])
    return C

print("REF of C = ")
print(RowEchelonForm(C, GF))
# -----------------------------------------------------------------------

# pivotREF(C)
# stores all the pivot after REF.

def pivotREF(C, ffield):
    pivot = [0, 0]
    pivots = []
    k = len(C)
    n = len(C[0])
    C = gaussNonZero(C, k, n, -1, -1, ffield)
    pivot = searchNonZero(C, k, n, -1, -1)
    pvt = np.array(pivot)
    pivots.append(pvt)
    while(pivot[0]<k-1) and (pivot[1]<n):
        for j in range(pivot[0]+1, k):
            m = (-1)*C[j][pivot[1]] // C[pivot[0]][pivot[1]]
            C = rowMAdd(k, j, pivot[0], m, ffield) @ C
        C = gaussNonZero(C, k, n, pivot[0], pivot[1], ffield)
        pivot = searchNonZero(C, k, n, pivot[0], pivot[1])
        if pivot[1]<n:
            pvt = np.array(pivot)
            pivots.append(pvt)
    pivots = np.asarray(pivots)
    return pivots.astype('int64')

# print(pivotREF(C, GF))
# -----------------------------------------------------------------------

# RREchelonForm(C, p)
# makes the matrix C into reduced row echelon form,
# given C is in REF and p is the information regarding the pivots.

def RREchelonForm(C, pivots, ffield):
    L = len(pivots)
    k = len(C)
    n = len(C[0])
    for i in range(L):
        s = L-i-1
        if pivots[s][1]==n:
            break
        m = C[pivots[s][0]][pivots[s][1]] ** -1
        C = rowMult(k, pivots[s][0], m, ffield) @ C
        for j in range(pivots[s][0]):
            m = (-1)*C[j][pivots[s][1]]
            C = rowMAdd(k, j, pivots[s][0], m, ffield) @ C
    return C

D = RowEchelonForm(C, GF)
pivots = pivotREF(C, GF)

print("RREF of C = ")
print(RREchelonForm(D,pivots,GF))


#################### Base-finding algorithm ####################

# Algorithm 1: to find a basis B ⊆ F_q^N for C = span(S), with S ⊆ F_q^N
def generalBase(C, ffield):
    pivots = pivotREF(C, ffield)
    D = RowEchelonForm(C, ffield)
    nzrows = len(pivots)
    return D[:nzrows,:]

print("B1 basis for C = ")
print(generalBase(C, GF))
# -----------------------------------------------------------------------
# Algorithm 2: to find a basis B ⊆ S for C
def subsetedBase(C, ffield):
    D = np.transpose(C)
    pivots = pivotREF(D, ffield)
    nzrows = len(pivots)
    return C[pivots[:,1],:]

print("B2 subsetted basis for C = ")
print(subsetedBase(C, GF))
# -----------------------------------------------------------------------
# Algorithm 3: to find a basis B for C^⊥ (the dual of C)
def dualBase(C, ffield):
    k = len(C)
    n = len(C[0])
    D = RowEchelonForm(C, ffield)
    pivots = pivotREF(C, ffield)
    nzrows = len(pivots)
    C = RREchelonForm(D, pivots, ffield)

    G = C[0:nzrows,:]
    Gprime = np.transpose(G)
    for i in range(nzrows):
        Gprime = rowSwap(n, i, pivots[i][1], ffield) @ Gprime
    Gprime = np.transpose(Gprime)
    X = Gprime[:,nzrows:]

    if nzrows < n:
        Xprime = (-1)*np.transpose(X)
        In_k = rowSwap(n-nzrows, 0, 0, ffield)
        H = np.concatenate((Xprime, In_k), axis=1)

        H = np.transpose(H)
        for i in range(nzrows):
            j = nzrows-i-1
            H = rowSwap(n, j, pivots[j][1], ffield) @ H
        H = np.transpose(H)
        return H
    else:
        return np.array([])

print("B3 basis for C dual = ")
print(dualBase(C, GF))


#################### Syndrome Look-Up Table ####################

def nextSequence(array):
    if array[0]==0:
        chain=0
        for i in range(len(array)-1):
            chain=i
            if array[i]+1 != array[i+1]:
                break
            elif i==len(array)-2 and array[i]+1 == array[i+1]:
                chain=i+1
                break

        if chain==len(array)-1:
            return []
        else:
            translation = array[chain+1]-array[chain]-2
            array[chain+1] = array[chain+1] - 1
            for i in range(chain+1): array[i] += translation
            return array
    else:
        array[0] = array[0] - 1
        return array
    
def generateNext(w, ffield):
    L = len(w)-1
    q = ffield.order
    temp = ffield(1)
    w = ffield(w)

    nonZeros = 0
    indZeros = []
    for i in range(len(w)):
        if w[i]!=0:
            nonZeros +=1
            indZeros.append(i)

    if nonZeros==0:
        w[L] = w[L] + ffield(1)
    else:
        isAllq_1 = True
        for i in range(len(indZeros)):
            if w[indZeros[i]] != q-1:
                isAllq_1 = False
                break

        isAllLeft = False
        indLeft = [i for i in range(nonZeros)]
        if indLeft == indZeros:
            isAllLeft = True

        if not isAllq_1:
            w[indZeros[nonZeros-1]] = w[indZeros[nonZeros-1]]+ffield(1)
            temp = ffield(1)
            for i in range(nonZeros-1):
                s = nonZeros-i-2
                if w[indZeros[s+1]] == 0:
                    w[indZeros[s]] = w[indZeros[s]]+temp
                else:
                    temp=ffield(0)

            for i in range(nonZeros):
                if w[indZeros[i]] == ffield(0):
                    w[indZeros[i]] = ffield(1)
        else:
            if not isAllLeft:
                nextloc = nextSequence(indZeros)
                w = ffield.Zeros(len(w))
                for i in range(len(nextloc)):
                    w[nextloc[i]] = ffield(1)
            else:
                if nonZeros==len(w):
                    w = ffield.Zeros(len(w))
                else:
                    w = ffield.Zeros(len(w))
                    nextloc = [i for i in range(len(w)-nonZeros-1,len(w))]
                    for i in range(len(nextloc)):
                        w[nextloc[i]] = ffield(1)
    return w


def isArrInArrays(arrays, arr):
    isIn = True
    lenarr = len(arr)
    for i in range(len(arrays)):
        isIn = True
        for j in range(lenarr):
            if arrays[i,j] != arr[j]:
                isIn = False
        if isIn==True:
            break
    return isIn

def generateLUT(H, ffield):
    n = len(H[0])
    n_k = len(H)
    cLeader = np.array([0 for i in range(n)])
    cLeader = ffield(cLeader)
    synd = cLeader @ np.transpose(H)

    syndromes = []
    cosLeds = []
    syndromes.append(synd)
    cosLeds.append(cLeader)
    p = np.asarray(syndromes)

    syndMaxSize = ffield.order**n_k
    while(len(p) < syndMaxSize):
        cLeader = generateNext(cLeader, ffield)
        synd = cLeader @ np.transpose(H)
        isIn = isArrInArrays(p, synd)
        if not isIn:
            syndromes.append(synd)
            cosLeds.append(cLeader)
            p = np.asarray(syndromes)
    return np.asarray(cosLeds), p

cLeaders, synd = generateLUT(dualBase(C, GF), GF)

print("coset leader | syndrome")
for i in range(len(synd)):
    print(cLeaders[i], " | ", synd[i])


#################### Syndrome Decoding ####################
# The decoding scheme that making use of the syndrome 
# to identify the coset to which the received word belongs.

def findSyndrome(sy, s):
    lens = len(s)
    for i in range(len(sy)):
        isIn = True
        for j in range(lens):
            if sy[i,j] != s[j]:
                isIn = False
        if isIn==True:
            return i
        
def decodeSyndrome(C, ffield):
    N = len(C[0])
    q = ffield.order
    D = dualBase(C, ffield)
    cLeaders, synd = generateLUT(D, ffield)

    print(f"Enter the code A1A2A3...AN such that Ai an element of F_{q} by typing them separately by space.")
    print(f"For example, if N=3 and q=11, and A1 = 1, A2 = 10, A3 = 3, then type \"1 10 3\".")
    print("Please enter each of the Ai only with integers.")

    string = input(F"\nEnter code of length {N}: ")
    code = string.split()
    while(len(code) != N):
        print(f"The length is not {N}, try again.")
        string = input(f"\nEnter code of length {N} again: ")
        code = string.split()
    code = np.array(code)
    code = code.astype('int64')
    code = ffield(np.array(code)%q)

    s = code @ np.transpose(D)
    print(f"Your code is {code}.")
    print(f"{s} is the syndrome.")
    cosetCode = cLeaders[findSyndrome(synd, s)]
    print(f"{cosetCode} is the coset leader.")
    print(f"Your decoded code is {code - ffield(cosetCode)}.")

decodeSyndrome(C, GF)

input("Press Enter to exit...")
