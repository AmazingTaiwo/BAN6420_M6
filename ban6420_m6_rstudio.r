# Load necessary libraries
library(keras)
library(tensorflow)
library(ggplot2)

# Load the Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
x_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
x_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y

# Normalize the images to be between 0 and 1
x_train <- x_train / 255
x_test <- x_test / 255

# Reshape the images to be 28x28x1 (for grayscale)
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

# Check the shape of the training data
cat("Shape of training data:", dim(x_train), "\n")

# Data Augmentation
datagen <- image_data_generator(
  rotation_range = 20,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

datagen %>% fit_image_data_generator(x_train)

# Initialize the model
model <- keras_model_sequential()

# Layer 1: Convolutional Layer
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_batch_normalization() %>%
  
  # Layer 2: Max Pooling Layer
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Layer 3: Convolutional Layer
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_batch_normalization() %>%
  
  # Layer 4: Max Pooling Layer
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Layer 5: Convolutional Layer
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_batch_normalization() %>%
  
  # Layer 6: Dropout to reduce overfitting
  layer_dropout(rate = 0.4) %>%
  
  # Layer 7: Flatten and Fully Connected Layer
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_batch_normalization() %>%
  
  # Output Layer
  layer_dense(units = 10, activation = 'softmax')  # 10 classes for Fashion MNIST

# Compile the model with a lower learning rate for better convergence
model %>% compile(
  optimizer = optimizer_adam(lr = 0.0005),
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Define early stopping and model checkpoint callbacks
early_stop <- callback_early_stopping(monitor = 'val_loss', patience = 3, restore_best_weights = TRUE)
checkpoint <- callback_model_checkpoint('best_fashion_mnist_model.h5', monitor = 'val_accuracy', save_best_only = TRUE)

# Train the model using data augmentation
history <- model %>% fit_generator(
  datagen %>% flow_images_from_data(x_train, y_train, batch_size = 64),
  steps_per_epoch = nrow(x_train) %/% 64,
  epochs = 20,
  validation_data = list(x_test, y_test),
  callbacks = list(early_stop, checkpoint)
)

# Evaluate the model on the test data
test_result <- model %>% evaluate(x_test, y_test, verbose = 2)
cat("Test accuracy:", test_result$accuracy, "\n")

# Make predictions on the test set
predictions <- model %>% predict(x_test)

# Display the first five test images and their predictions
for (i in 1:5) {
  img <- x_test[i,,]
  actual_label <- y_test[i]
  predicted_label <- which.max(predictions[i,]) - 1  # Convert to zero-based index
  
  # Plot the image
  img_matrix <- array_reshape(img, c(28, 28))
  ggplot(data.frame(x = rep(1:28, each = 28), y = rep(1:28, 28), z = as.vector(img_matrix))) + 
    geom_tile(aes(x = x, y = y, fill = z)) +
    scale_fill_gradient(low = "white", high = "black") + 
    theme_minimal() +
    ggtitle(paste("Predicted:", predicted_label, ", Actual:", actual_label))
}

# Save the trained model
model %>% save_model_hdf5('fashion_mnist_cnn_optimized.h5')

# Plotting training & validation accuracy and loss for better visualization
history_df <- as.data.frame(history)

# Plot accuracy
ggplot(history_df, aes(x = 1:nrow(history_df))) +
  geom_line(aes(y = history_df$accuracy), color = 'blue', linetype = 'solid') +
  geom_line(aes(y = history_df$val_accuracy), color = 'red', linetype = 'dashed') +
  ggtitle('Training and Validation Accuracy') +
  xlab('Epoch') +
  ylab('Accuracy') +
  theme_minimal()

# Plot loss
ggplot(history_df, aes(x = 1:nrow(history_df))) +
  geom_line(aes(y = history_df$loss), color = 'blue', linetype = 'solid') +
  geom_line(aes(y = history_df$val_loss), color = 'red', linetype = 'dashed') +
  ggtitle('Training and Validation Loss') +
  xlab('Epoch') +
  ylab('Loss') +
  theme_minimal()

# Predicting on additional test images for more insight
additional_images <- x_test[1:5, , ]
additional_predictions <- model %>% predict(additional_images)

for (i in 1:5) {
  img <- additional_images[i,,]
  actual_label <- y_test[i]
  predicted_label <- which.max(additional_predictions[i,]) - 1  # Convert to zero-based index
  
  # Plot the image
  img_matrix <- array_reshape(img, c(28, 28))
  ggplot(data.frame(x = rep(1:28, each = 28), y = rep(1:28, 28), z = as.vector(img_matrix))) + 
    geom_tile(aes(x = x, y = y, fill = z)) +
    scale_fill_gradient(low = "white", high = "black") + 
    theme_minimal() +
    ggtitle(paste("Predicted:", predicted_label, ", Actual:", actual_label))
}
