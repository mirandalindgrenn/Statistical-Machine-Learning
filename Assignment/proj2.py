import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#mage=mpimg.imread('TrainDigits.npy')

TrainMat = np.load('HandwrittenDigits/TrainDigits.npy')
TrainLabel = np.load('HandwrittenDigits/TrainLabels.npy')
TestMat = np.load('HandwrittenDigits/TestDigits.npy')
TestLabel = np.load('HandwrittenDigits/TestLabels.npy')

#print('type TrainMat', type(TrainMat))
#print('Shape of TrainMat', TrainMat.shape)
#print('\n')
#print('type TrainLabel', type(TrainLabel))
#print('Shape of TrainLabel', TrainLabel.shape)
#print('\n')
#print('type TestMat', type(TestMat))
#print('Shape of TestMat', TestMat.shape)
#print('\n')
#print('type TestLabel', type(TestLabel))
#print('Shape of TestLabel', TestLabel.shape)


#Create subplots: 2 rows and 5 columns (for a total of 10 images)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))  # 2 rows, 5 columns
fig.suptitle('First 10 Digits', fontsize=16)  # Title for the entire figure

# Loop through the first 10 digits
for i in range(10):
    digit_id = i  # The digit index
    d = TrainMat[:, digit_id]  # The image in vector form
    D = np.reshape(d, (28, 28)).T  # Reshaping the vector to a 28x28 matrix
    
    # Determine the row and column of the subplot
    row = i // 5
    col = i % 5
    
    # Plot the digit in the appropriate subplot
    axes[row, col].imshow(D, cmap='gray')
    axes[row, col].set_title(f'Label: {TrainLabel[0, digit_id]}')
    axes[row, col].axis('off')  # Hide the axis for better visualization

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
#plt.show()

def extract_indices(searchDigit,DataLabel,nSearch):
    '''Extract indices of elements in DataLabel that match searchDigit. only consider the first nSearch elements in DataLabel.
the number of matches is stored in numInd the extracted indices are stored in the vector vecInd'''
    indices = (DataLabel.flatten()[:nSearch] == searchDigit)
    indVec = np.argwhere(indices).flatten()
    numInd = len(indVec)
    return numInd,indVec

#numInd, indVec = extract_indices(6, TrainLabel, 1000 )

#print(numInd)
#print(extract_indices(2, TrainLabel, 1000 ))

#print(np.shape(TrainLabel))
#limDataLabel = TrainLabel[:1000]  # Ingen flattening behövs
#print(len(limDataLabel))  # Detta ger 1000

#x = np.array([1, 3, 5, 7, 2])
#nyx = x[:3].flatten()
#print(len(nyx))

def set_up_testmatrix(searchDigit,DataSet,DataLabel,numData):
    '''General comment: use the function extract_indices(searchDigit,DataLabel,nSearch),
    and choose nSearch big enough so that you get more than numData indices back, and then 
    take the first numData indices. call extract_indices'''
    DataLabel= DataLabel.flatten()
    nSearch = 100000
    numInd,indVec = extract_indices(searchDigit,DataLabel,nSearch)
# throw error
    if numInd < numData:
        raise ValueError("Did not find enough indices")
    '''extract the first numData columns that corresponds to searchDigit. Let these columns form the matrix A
    convert the matrix A to single precision floats, using np.float32'''
    A = DataSet[:,indVec[:numData]]
    A = np.float32(A)
    return A

#Task 2
M3 = set_up_testmatrix(3, TestMat, TestLabel, numData=2000)
M8 = set_up_testmatrix(8, TestMat, TestLabel, numData=2000)
U3, S3, VT3 = np.linalg.svd(M3, full_matrices=False)
U8, S8, VT8 = np.linalg.svd(M8, full_matrices=False)

# Steg 2: Plotta singulära värden (linjär på x, logaritmisk på y)

# Plot singulära värden
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Task 2: Graph of singular values for number 3 and 8", fontsize=16)
for i, (S, title) in enumerate(zip([S3, S8], ['siffra 3', 'siffra 8'])):
    axes[i].plot(S)
    axes[i].grid()
    axes[i].set_xlim(0, 30)  # Begränsar x-axeln till [0, 30]
    axes[i].set_ylim(1e3, 1e5)
    axes[i].set_yscale('log')
    axes[i].set_title(f'Singulära värden för {title}')
    axes[i].set_xlabel('Index')
    axes[i].set_ylabel('Singulärt värde')

plt.tight_layout()
#plt.show()

# Plot singulära vektorer (u1, u2, u3) för siffrorna 3 och 8
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Task 2: U1, U2 and U3 for number 3 and 8", fontsize=16)
for j, U in enumerate([U3, U8]):
    for i in range(3):
        axes[j, i].imshow(U[:, i].reshape(28, 28), cmap='gray')
        axes[j, i].set_title(f'U{i+1} för siffra {3 if j == 0 else 8}')
        axes[j, i].axis('off')
plt.tight_layout()
#plt.show()


#Task 3
U_matrices = {} # Dictionary för att lagra Uk-matriserna
min_k= 5
max_k= 15

for digit in range(10): # Loopa över siffrorna 0 till 9
    M_digit = set_up_testmatrix(digit, TestMat, TestLabel, numData=2000) # Skapa matrisen för varje siffra med set_up_testmatrix
    U_digit, S_digit, VT_digit = np.linalg.svd(M_digit, full_matrices=False) # Beräkna SVD för matrisen
    U_matrices[digit] = U_digit[:, :max_k]
    
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle("Task 3: First Column of Uk for Digits 0-9", fontsize=16)
for digit in range(10): # Verifiera genom att plotta första kolumnen i Uk för varje siffra
    row = digit // 5  # Plotta första kolumnen av Uk-matrisen för varje siffra
    col = digit % 5
    axes[row, col].imshow(U_matrices[digit][:, 0].reshape(28, 28), cmap='gray')
    axes[row, col].set_title(f'Siffra {digit}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

#Task 4 and 5

numd_test_digit = TestLabel.shape[1] #Number of test digits, that is number of colums in TestLabel
print('Number test data', numd_test_digit)
numd_test_digit_d = np.zeros((1, 10)) #lägger in alla nummer på rätt plats i arrayen som består av
for digit in range(10):
    numInd, ind_wished_digit = extract_indices(digit, TestLabel, numd_test_digit)
    print('digit:',digit,'ind', numInd)
    numd_test_digit_d[:, digit] = numInd

def residual(Uk, delta):
    I = np.eye(784)
    U = np.dot(Uk, Uk.T)
    I_U_delta = np.dot(I - U, delta)
    return np.linalg.norm(I_U_delta, axis=0)

numd_test_digit = TestLabel.shape[1] #sätter numd_test_digit till kolumnvärdet av TestLasbels storlek
numd_test_digit_d = int(numd_test_digit / 10) #delar kolumnvärdet på testlabel med 10 pga vi vet sedan innan att det är jämt fördelat mellan alla siffror (10st)
percentage = np.zeros(max_k - min_k + 1)
percentage_d = np.zeros([10, max_k - min_k + 1])

for k in range(min_k, max_k + 1):
    norm2res = np.zeros([10, numd_test_digit])
    
    for digit in range(10):
        Uk = U_matrices[digit][:, :k]
        norm2res[digit, :] = residual(Uk, TestMat)
    
    classification = np.argmin(norm2res, axis=0)
    success = np.where((classification - TestLabel.flatten()) == 0)
    percentage[k - min_k] = (len(success[0]) / numd_test_digit) * 100
    
    for d in range(10):
        indx = TestLabel == d
        success = np.where((classification[indx[0]] - TestLabel[:, indx[0]].flatten()) == 0)
        percentage_d[d, k - min_k] = (len(success[0]) / numd_test_digit_d) * 100

print('overall percentage =\n', percentage)

plt.figure(figsize=(6, 3))
plt.plot(range(min_k, max_k + 1), percentage, linestyle='-', marker='o', color='blue')
plt.show()



def classification(d):
    pass