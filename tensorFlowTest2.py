import tensorflow as tf
import tkinter as tk

# Load the model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model= tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation= "relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer= "adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.fit(x_train,y_train,epochs=5)

# Create the main window
window = tk.Tk()
window.title("Handwritten Digit Classifier")

# Create a label to display the prediction
label = tk.Label(text="")
label.pack()

# Create a function to handle the user's input
def classify_digit():
  # Get the user's input
  digit = input_field.get()

  while True:
      # kullanıcıdan el ile yazılan sayıyı alın
      digit = input("Please enter a handwritten digit: ")

      # sayıyı model için uygun hale getirin
      digit = digit.replace("\n", "")
      digit = digit.split(" ")
      digit = [int(x) for x in digit]
      digit = np.array(digit).reshape(1, 28, 28, 1)

      # modeli kullanarak tahmini alın
      prediction = model.predict(digit)

      # tahmini etikete dönüştürün
      label = np.argmax(prediction, axis=1)[0]

      # tahmini kullanıcıya gösterin
      print(f"Prediction: {label}")

  # Use the model to make a prediction
  prediction = model.predict(processed_input)
  predicted_digit = np.argmax(prediction)

  # Update the label with the prediction
  label.configure(text=f"Predicted digit: {predicted_digit}")

# Create an input field for the user to enter a digit
input_field = tk.Entry()
input_field.pack()

# Create a button to classify the digit
button = tk.Button(text="Classify", command=classify_digit)
button.pack()

# Run the main loop
window.mainloop()
