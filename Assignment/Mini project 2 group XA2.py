import numpy as np
import matplotlib.pyplot as plt

#Preparation
### Step 1
TrainMat = np.load('HandwrittenDigits/TrainDigits.npy')   #loading all files
TrainLabel = np.load('HandwrittenDigits/TrainLabels.npy')
TestMat = np.load('HandwrittenDigits/TestDigits.npy')
TestLabel = np.load('HandwrittenDigits/TestLabels.npy')

print('type TrainMat', type(TrainMat)) #checking the type of TrainMat
print('Shape of TrainMat', TrainMat.shape) #checking the shape of TrainMat

digit_id = 0 # The first digit (arrays start at 0)
d = TrainMat[:,digit_id] # The image in vector form
D = np.reshape(d, (28, 28)).T # Reshaping a vector to a matrix of size 28x28
plt.imshow(D, cmap ='gray') 
plt.title("Preparation 1: first digit in TrainMat")
plt.show() 
print('Label of digit',TrainLabel[0,digit_id]) # Print what digit it is.

#Create subplots: 2 rows and 5 columns (for a total of 10 images)
fig, axes = plt.subplots(2, 5, figsize=(10, 5)) 
fig.suptitle('Preparation 1: first 10 Digits', fontsize=16)  

for i in range(10): # Loop through the first 10 digits
    digit_id = i  
    d = TrainMat[:, digit_id]  # The image in vector form
    D = np.reshape(d, (28, 28)).T  # Reshaping the vector to a 28x28 matrix
    row = i // 5 # Determine the row and column of the subplot
    col = i % 5
    axes[row, col].imshow(D, cmap='gray')
    axes[row, col].set_title(f'Label: {TrainLabel[0, digit_id]}')
    axes[row, col].axis('off') 
plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()

### Step 2
def extract_indices(searchDigit, DataLabel, nSearch):
    '''Extract indices of elements in DataLabel that match searchDigit. Only consider the first nSearch elements in DataLabel.
       The number of matches is stored in numInd, the extracted indices are stored in the vector vecInd'''
    indices = (DataLabel.flatten()[:nSearch] == searchDigit)
    indVec = np.argwhere(indices).flatten()
    numInd = len(indVec)
    return numInd,indVec

a = np.array([4,5,6,2,4,8,0,0,9,7])
i = 0
index = (a==i)
print('index =',index) #Should give output index = [False False False False False False True True False False]
indices = np.argwhere(index)
indices_flatten= indices.flatten()
print('indices =',indices) #Should give output indices = [[6][7]]
print('indices_flatten = ',indices_flatten) #Should give output indices_flatten = [6 7]
print('shape: indices = ',indices.shape) #Should give output shape: indices = (2, 1)
print('shape: indices_flatten = ',indices_flatten.shape) #Should give output shape: indices_flatten = (2,)

searchDigit = 2
nSearch = 1000
DataLabel = TrainLabel
numInd,indVec = extract_indices(searchDigit,DataLabel,nSearch)
print('Number of indices found:', numInd)  # Output will tell you how many matches were found
print('Indices with digit', searchDigit, 'are: ',indVec)
print('Print from labels to confirm', TrainLabel[:,indVec])

### Step 3
def set_up_testmatrix(searchDigit,DataSet,DataLabel,numData):
    '''General comment: Use the function extract_indices(searchDigit,DataLabel,nSearch),
  and choose nSearch big enough so that you get more than numData indices back, and then take the first numData indices'''
    nSearch = 100000
    numInd,indVec = extract_indices(searchDigit,DataLabel,nSearch) #Get the number of columns with given search digit and their indices  
    if numInd < numData:    # throw error if numInd is less than numData
        raise ValueError("Did not find enough indices")
  # extract the first numData columns that corresponds to searchDigit. Let these columns form the matrix A
    A = DataSet[:,indVec[:numData]] #From data set, get the numData number of columns at the indices where our digit is located
    A = np.float32(A)   # convert the matrix A to single precision floats, using np.float32
    return A

#Testing that the matrix A has the right shape and type
A = set_up_testmatrix(searchDigit,TestMat,TestLabel,numData=1000)
print(A.shape)
print(type(A[0,0]))

A = set_up_testmatrix(6,TestMat,TestLabel,numData=1000)
U, S, VT = np.linalg.svd(A, full_matrices=False)
plt.imshow(U[:, 0].reshape(28,28).T, cmap='gray')
plt.title(f'First column of reduced U in SVD for search digit 6')
plt.show()

# Verify the dimensions of U, S, VT
print("Shape of U:", U.shape)    # Should print (784, 784)
print("Shape of s:", S.shape)    # Should print (784,)
print("Shape of Vt:", VT.shape)  # Should print (784, 1000)

# Tasks
### Task 2
M3 = set_up_testmatrix(3,TestMat,TestLabel,numData=1000)
M8 = set_up_testmatrix(8,TestMat,TestLabel,numData=1000)
U3, S3, VT3 = np.linalg.svd(M3, full_matrices=False)
U8, S8, VT8 = np.linalg.svd(M8, full_matrices=False)

digits = [3, 8]
colors = ['thistle', 'pink']  # Blue for 3, Red for 8
labels = ['Digit 3', 'Digit 8']

singular_values = [] # List to store singualr values for each number

fig, axes = plt.subplots(1, 3, figsize=(13, 7)) # Creates 3 subplots, one for the singular values of 3 and 8 and one for the normalized values
fig.suptitle('Singular and normalized values for 3 and 8, numData=2000', fontsize=16) 
for i, digit in enumerate(digits): # Loops through the numbers and then plotting them with numData=2000
    M = set_up_testmatrix(digit, TestMat, TestLabel, numData=2000) #set_up_testmatrix returns the matrix A which we use 
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    singular_values.append(S)
    axes[i].plot(S, colors[i], marker='o', label=f'Digit {digit}')  # Plots the singular values
    axes[i].set_yscale('log')
    axes[i].set_title(f'Singular Values for Digit {digit}')
    axes[i].set_xlabel('Index')
    axes[i].set_ylabel('Singular Value (log)', labelpad=0)
    axes[i].legend()
    axes[i].grid(True)

for i, S in enumerate(singular_values): # Plots the normalized singualar values for 3 and 8 in the same graph
    axes[2].plot(S / S[0],  colors[i], marker='o', label=labels[i])
axes[2].set_title('Normalized Singular Values for Digit 3 and 8')
axes[2].set_xlabel('Index')
axes[2].set_ylabel('Normalized Singular Value (log)', labelpad=0)
axes[2].legend()
axes[2].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at the top for the suptitle
plt.show()

#plotting singular vaues with numData=30
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Singular values numData=30', fontsize=16)
for i, digit in enumerate(digits):
    M = set_up_testmatrix(digit, TestMat, TestLabel, numData=30)
    U, S, VT = np.linalg.svd(M, full_matrices=False)
    axes[i].plot(S, colors[i], marker='o', label=f'Digit {digit}')
    axes[i].set_yscale('log')
    axes[i].set_title(f'Singular Values for Digit {digit}')
    axes[i].set_xlabel('Index')
    axes[i].set_ylabel('Singular Value (log)', labelpad=-15)
    axes[i].legend()
    axes[i].grid(True)
plt.tight_layout()
plt.show()

# Plot singular vectors (u1, u2, u3) for numbers 3 and 8
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("U1, U2 and U3 for number 3 and 8", fontsize=16)
for j, U in enumerate([U3, U8]):
    for i in range(3): #plots 3 images for each number
        axes[j, i].imshow(U[:, i].reshape(28, 28).T, cmap='gray')
        axes[j, i].set_title(f'U{i+1} för siffra {3 if j == 0 else 8}')
        axes[j, i].axis('off')
plt.tight_layout()
plt.show()

### Task 3
U_matrices = {} # Dictionary to store the Uk-matrices
min_k = 5 
max_k = 15 

#Precompute all ten SVDs and store Uk matrices for all ten digits.
for digit in range(10): 
    M_digit = set_up_testmatrix(digit, TrainMat, TrainLabel, 2000) # creating a matrix for each digit with set_up_testmatrix which returns a matrix
    U_digit, S_digit, VT_digit = np.linalg.svd(M_digit, full_matrices=False) # calculating SVD for the matrix
    U_matrices[digit] = U_digit[:, :max_k]

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle("Task 3: First Column of Uk for Digits 0-9", fontsize=16)
for digit in range(10): # Verifying trough plotting the first column in Uk for each digit
    row = digit // 5  
    col = digit % 5
    reshapeMatrix = U_matrices[digit][:, 0].reshape(28, 28)
    axes[row, col].imshow(reshapeMatrix.T, cmap='gray')
    axes[row, col].set_title(f'Siffra {digit}')   
    axes[row, col].axis('off')
plt.tight_layout()
plt.show()

### Task 4 & 5 

num_test_images = TestLabel.shape[1]  # Total number of columns in TestLabel (i.e. number of test images)
print('Total number of test images:', num_test_images)

# Initialize an array to store the number of images for each digit (0-9)
num_images_per_digit = np.zeros((1, 10))  # A 1x10 array for storing the count of images for each digit
for digit in range(10):  # Loop over each digit (0-9)
    digit_indices, _ = extract_indices(digit, TestLabel, num_test_images)  # Get the number of images for each digit
    num_images_per_digit[:, digit] = digit_indices  # Store the count in the corresponding index
    print(f'\nDigit: {digit}, Number of images: {digit_indices}')  # Print the number of images for the current digit

# Define a residual function to calculate the difference between the projection and the original image
def residual(Uk_submatrix, test_image_vector):
    identity_matrix = np.eye(784)  
    projection = np.dot(Uk_submatrix, Uk_submatrix.T)  # Project the test data vector onto the Uk subspace
    difference = np.dot(identity_matrix - projection, test_image_vector)  # Calculate the difference between the original and projection
    return np.linalg.norm(difference, axis=0) 

# Initialize arrays to store success percentages
overall_success_percentage = np.zeros(max_k - min_k + 1)  # To store overall success percentage for each k
success_percentage_per_digit = np.zeros([10, max_k - min_k + 1])  # To store success percentage for each digit and k

# Loop through different values of k (the number of columns in U)
for num_columns in range(min_k, max_k + 1):
    residuals_matrix = np.zeros([10, num_test_images])  # Initialize a matrix to store residuals for all digits and test images
    # For each digit, compute the residuals for test images using the first num_columns columns of the corresponding U matrix
    for digit in range(10):
        Uk_digit = U_matrices[digit][:, :num_columns]  # Select the first num_columns of the U matrix for the current digit
        residuals_matrix[digit, :] = residual(Uk_digit, TestMat)  # Compute the residuals for all test images
    # Classify each test image as the digit with the smallest residual
    predicted_labels = np.argmin(residuals_matrix, axis=0)  # Find the index (digit) with the smallest residual for each test image
    overall_success_percentage[num_columns - min_k] = np.mean(predicted_labels == TestLabel) * 100  # Calculate overall accuracy for this k
    # Calculate success percentage for each individual digit
    for digit in range(10):
        digit_indices = (TestLabel == digit)[0]  # Create a mask where True corresponds to images of the current digit
        success_percentage_per_digit[digit, num_columns - min_k] = np.mean(predicted_labels[digit_indices] == TestLabel[0, digit_indices]) * 100  # Success percentage for this digit

k_values = np.arange(min_k, max_k + 1) # creates array with desired k-values

# Success percentage for digit=1 (index 1 in success_percentage_per_digit)
success_digit_1 = success_percentage_per_digit[1, :]  # Extract the success percentages for digit 1
print(f"{'k Value':<10} {'Success Percentage (%)':<30}") 
print("-" * 40)
for k, percentage in zip(k_values, success_digit_1):  # Print the success percentage for each k
    print(f"{k:<10} {percentage:.3f}")

# Total success percentage over all digits 
print(f"{'k Value':<10} {'Total Success Percentage (%)':<30}")
print("-" * 40)
for k, percentage in zip(k_values, overall_success_percentage):  # Print the overall success percentage for digits 0-9 for each k
   print(f"{k:<10} {percentage:.3f}")

plt.plot(k_values, overall_success_percentage, marker='o', color='pink')  # Create a line plot with markers for overall success percentages
plt.title('Overall Success Percentage')  
plt.xlabel('Number of k (columns)') 
plt.ylabel('Success Percentage (%)')  
plt.show()  

for digit in range(10): # Success percentage for each digit (0-9)
    plt.plot(k_values, success_percentage_per_digit[digit, :], label=f'Digit {digit}')  # Plot success percentages for each digit
plt.title('Success Percentage for Digits 0-9')  
plt.xlabel('Number of k (columns)')  
plt.ylabel('Success Percentage (%)')  
plt.legend()
plt.show()  


