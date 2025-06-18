# NeuralNetworkExample

step-1:
difference between np.random.randint and np.random.rand 
np.random.randint(low,high,size)
np.random.rand(shape mean it takes no.of rows and cols)Generates random floating-point numbers in the range [0.0, 1.0)
eg:np.random.rand(2, 3) =>
[[0.5488, 0.7152, 0.6028],
 [0.5449, 0.4237, 0.6459]]
features = np.random.rand(data_size, 2)  here row will be = 200 and cols = 2 so totalli 400 values

labels = (features[:, 0] + features[:, 1] > 1).astype(int)
Creates a binary label:
If the sum of VisitDuration + PagesVisited > 1, it assigns 1 (Purchase).
Otherwise, 0 (No Purchase).
features[:, 0] means:all rows of column 0 
features[:, 1] → all rows of column 1 
Checks if each sum > 1:
Creates a Boolean array  =>   [True, True, True, ..., False]
Converts True to 1 and False to 0 using .astype(int)     => [1, 1, 1, ..., 0]

df = pd.DataFrame(features, columns=['VisitDuration', 'PagesVisited']) here we are creating table and giving col names
df['Purchase'] = labels here we are adding one more col to table



step-2:
splitting the data into test adn train



step-3:
tensorflow: A powerful library used for building and training neural networks.
Sequential: A linear stack of layers — each layer feeds its output to the next one.
Dense: A fully connected layer — every neuron connects to every neuron in the next layer.

model = Sequential([
    Dense(10, activation='relu', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])
sequential is the model we are using to build feed forward neural network
Layer-1: Dense(10, activation='relu', input_shape=(2,))=>This layer has 10 neurons,Each of the 10 neurons connects to both inputfeatures
why we need activation function?
Without activation functions, your neural network becomes just a stack of linear equations — no matter how many layers you add.That means:It would only be able to learn straight-line relationships (linear functions) — which is not enough for most real-world problems
so we need activation function to introduce non linearity
activation='relu':
ReLU (Rectified Linear Unit) activation function:
relu(x) = max(0, x)
input_shape=(2,)=>This tells the model that each input sample has 2 values/features:VisitDuration and PagesVisited.

Layer-2: Dense(1, activation='sigmoid') => 1 neuron ,here this one neuron will be connected all 10 neurons of layer 1
sigmoid(x) = 1 / (1 + e^(-x))

optimizer='adam':
Adam = Adaptive Moment Estimation.Smart algorithm that adjusts learning rate during training.Combines momentum and RMSprop for faster convergence.
loss='binary_crossentropy':The loss function measures how far off predictions are from actual labels.Binary cross-entropy is ideal for binary classification problems.
metrics=['accuracy']:You want to track accuracy (i.e., how many predictions are correct) during training.

model.fit(X_train, y_train, epochs=10, batch_size=10)
epochs=10:The model will see the entire training data 10 times.1 epoch = 1 full pass through all training samples.
batch_size=10:During training, data is broken into mini-batches of 10 samples

loss, accuracy = model.evaluate(X_test, y_test) uses loss function and evalution metrics specified in .compile
print(f"Test Accuracy: {accuracy}")







