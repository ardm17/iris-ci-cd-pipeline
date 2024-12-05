import gradio as gr
import skops.io as sio
import multipart

# Attempt to load the model, catching the TypeError
try:
    model = sio.load("Model/iris_rf_pipeline.skops", trusted=[])
except TypeError as e:
    # If there are untrusted types, retrieve them from the error message
    untrusted_types = sio.get_untrusted_types()
    model = sio.load("Model/iris_rf_pipeline.skops", trusted=untrusted_types)

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)[0]
    return f"Predicted Iris Species: {prediction}"

# Define the Gradio interface
gr.Interface(
    fn=predict_iris,
    inputs=[gr.Slider(4.0, 8.0, label="Sepal Length"),
            gr.Slider(2.0, 4.5, label="Sepal Width"),
            gr.Slider(1.0, 6.9, label="Petal Length"),
            gr.Slider(0.1, 2.5, label="Petal Width")],
    outputs="text",
    title="Iris Species Predictor"
).launch()
