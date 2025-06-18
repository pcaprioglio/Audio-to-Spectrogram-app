# AI-audio-visual-generative
This is currently a work in progress and the first step of a bigger audio-visual project.

1) For now, I added to this repository a simple Streamlit app to preview your audio tracks from a specific folder and convert them into their Mel Spectrogram. The spectrogram gets displayed on the webpage, and you can play around with the different Mel bands. You can run the app.py file with the Streamlit command: streamlit run app.py

2) I also added a program to convert all audio files from one folder into Mel Spectrogram and save them into a new directory as images. Additionally, the program does an image pre-processing step, which resizes the image, enforcing a 224x224 dimension in order to match the input dimension of some common CNNs. 


