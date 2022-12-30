import tensorflow as tf
from tensorflow.keras import (models, layers)
from nonconformist.base import ClassifierAdapter
import matplotlib.pyplot as plt


class FFNN():
    def __init__(self, input_shape, n_class, optimizer='adam', a1='relu', a2='sigmoid', loss=None):
        
        self.input_shape, self.n_class, self.optimizer = input_shape, n_class, optimizer
        self.epochs, self.batch_size, self.validation_split, self.verbose, self.history = [None]*5
        self.a1, self.a2 = a1, a2
        
        if not loss:
            self.from_logits = not(self.a2=='softmax' or self.a2=='sigmoid')
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=self.from_logits)
        else:
            self.loss = loss

        self.model = models.Sequential()
        self.model.add(layers.Flatten(input_shape=self.input_shape))
        self.model.add(layers.Dense(24, activation=self.a1))
        self.model.add(layers.Dense(self.n_class, activation=self.a2))
        self.model.compile(optimizer=self.optimizer,
                  loss = self.loss,
                  metrics=['accuracy'])
        
        
    def fit(self, X_train, y_train, epochs=3, batch_size=32, validation_split=0.2, verbose=0, is_graph=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose
        
        self.history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                                      validation_split=self.validation_split, verbose=self.verbose)
        return self
    
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    
    def eval_graph(self, X_test=None, y_test=None):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
        axs[0].plot(self.history.history['accuracy'], label="train")
        axs[0].plot(self.history.history['val_accuracy'], label="validation")
        axs[0].set(title='accuracy', ylabel='accuracy', xlabel='epoch')
        axs[0].legend(loc='lower right')

        axs[1].plot(self.history.history['loss'], label="train")
        axs[1].plot(self.history.history['val_loss'], label="validation")
        axs[1].set(title='loss', ylabel='loss', xlabel='epoch')
        axs[1].legend(loc='upper right')
        
        if not X_test is None:
            test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=self.verbose)
            axs[0].hlines(test_acc, 0, self.epochs-1, 'r', label="test")
            axs[0].legend(loc='lower right')
            axs[1].hlines(test_loss, 0, self.epochs-1, 'r', label="test")
            axs[1].legend(loc='upper right')
            return test_loss, test_acc
        
        
        
class FFNNAdapter(ClassifierAdapter):
    def __init__(self, model, fit_params=None):
        super(FFNNAdapter, self).__init__(model, fit_params)
        
        
    def fit(self, x, y):
        '''
            x is a numpy.array of shape (n_train, n_features)
            y is a numpy.array of shape (n_train)
            
            Here, do what is necessary to train the underlying model
            using the supplied training data
        '''
        self.model.fit(x, y, epochs=3, batch_size=128)
    
    def predict(self, x):
        '''
            Obtain predictions from the underlying model
            
            Make sure this function returns an output that is compatible with
            the nonconformity function used. For default nonconformity functions,
            output from this function should be class probability estimates in
            a numpy.array of shape (n_test, n_classes)
        '''
        return self.model.predict(x)
