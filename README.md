Emotion-Aware Speech Recognition
This project uses a combination of audio and facial emotion recognition to analyze the mood or emotion of a speaker in an audio or video clip. The project uses Whisper for speech-to-text transcription, while also using a pre-trained emotion recognition model (Mini-XCEPTION) for facial expressions and audio features like MFCC and pitch for emotion classification.

Features
Speech Transcription: Converts speech in audio/video files into text using OpenAI's Whisper model.
Emotion Recognition: Classifies the emotion of the speaker based on audio features and facial expressions.
Multi-modal Analysis: Combines both speech and facial emotion recognition for a more accurate understanding of the speaker's mood.
MP4 to WAV Conversion: Converts video files into audio files for processing.
Requirements
Python 3.x
Libraries:
whisper
librosa
numpy
scikit-learn
moviepy
keras
opencv-python
tensorflow
You can install the required libraries using the following:

pip install whisper librosa numpy scikit-learn moviepy keras opencv-python tensorflow Setup Clone the repository or download the script files.

Install dependencies using the command above.

Download the pre-trained emotion model:

The model is a Mini-XCEPTION model trained on the FER2013 dataset. Download it using:

wget https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5 -O emotion_model.h5 Upload video or image files to analyze the emotion. Supported file types:

Audio: .mp3, .wav

Video: .mp4

Image: .jpg, .png

How to Use Emotion-Aware Speech Recognition: Run the following command to analyze the emotions in a video or audio file:

emotion_aware_speech_recognition("path_to_file.mp4") # or .mp3 for audio Facial Expression Recognition: Upload an image of a face and run the following script to detect emotions based on facial expressions:

Example for facial emotion detection
process_uploaded_image("path_to_face_image.jpg") Process Flow Audio Processing:

The audio is first converted to text using Whisper's speech-to-text model.

Then, audio features like MFCC and pitch are extracted to detect emotions.

Facial Emotion Detection:

Detect faces in images using OpenCV.

Extract the facial region and resize it to 64x64.

Classify emotions using the Mini-XCEPTION model.

Emotion Classification:

For audio, use the extracted features (MFCC + pitch) to classify emotions such as happy, sad, angry, neutral, etc.

For facial expressions, classify emotions from the pre-trained emotion model.

Model Performance The project uses two different models:

Whisper for transcribing speech to text.

Mini-XCEPTION model for facial emotion recognition.

The emotion classification uses a simple Support Vector Machine (SVM) for audio emotion recognition. The model is trained using a small dataset of audio features (MFCC + pitch).

Example Output Audio transcription: "Hello, how are you?"

Detected Emotion: Happy

Language Detected: English

Future Enhancements Integration with more sophisticated emotion models for better accuracy.

Real-time emotion recognition using live video feeds.

Extended support for more languages in speech transcription.

License This project is licensed under the MIT License - see the LICENSE file for details.



