# Public Speaking Trainer

Public Speaking Trainer is a Streamlit-based application that uses AI to analyze speeches and provide feedback to help users improve their public speaking skills. It uses speech recognition to transcribe uploaded audio files and then leverages advanced language models to offer constructive feedback on various aspects of public speaking.

## Features

- Speech audio upload and transcription
- AI-powered analysis of speech content and delivery
- Customizable feedback aspects (e.g., Content, Structure, Delivery, Body Language)
- Interactive chat interface for follow-up questions and advice
- Support for multiple AI models (OpenAI and Ollama)
- Conversation saving and loading functionality
- Token usage tracking
- Dark/Light theme options

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/public-speaking-trainer.git
   cd public-speaking-trainer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`

3. Enter your name and configure your feedback preferences in the sidebar

4. Upload an audio file of your speech (supported formats: WAV, MP3, OGG)

5. Click "Analyze Speech" to get AI-generated feedback

6. Interact with the AI to ask follow-up questions or get more public speaking advice

## Customization

- Modify the `SPEECH_ASPECTS` list in `app.py` to add or remove aspects of speech analysis
- Adjust the `custom_instructions` in the sidebar to change the AI's behavior and focus

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
