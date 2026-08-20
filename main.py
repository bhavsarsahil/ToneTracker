from fastapi import FastAPI
import re
from pydantic import BaseModel, Field
from keras.models import load_model
import pickle

app =  FastAPI()

@app.get('/')
def greet():
    return {'Hello sahil'}

"""
1. We are going to make some constants like:
A. Model Path (BiGRU)
B. Tokenizer Path
C. Max Sequence Length
D. Emotion Labels
E. Emotion emojis
"""
#A. Model Path (BiGRU)
model_path = "Artifacts/BiGRU_Model.keras"

#B. Tokenizer Path
tokenizer_path = "Artifacts/tokenizer.pkl"

#C. Max Sequence Length
max_sequence_length = 50

#D. Emotion Labels
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

#E. Emotion emojis
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

"""
2. Preprocess the upcoming text
Cleans raw text so it matches the format used while training.
A. Convert the text to lowercase. -done
B. Remove apostrophes (e.g can't -> cant). -done
C. Remove Special Characters and Punctuation. -done
D. Remove extra spaces -done
"""

def preprocess_text(text:str)->str:
    text = text.lower()
    text = re.sub(r"'","",text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ",text).strip()
    return text
    
"""
3. Request and Response Schemas
A. Text Input -> Input schema the text sent by user. -done
B. Prediciton Response -> Output schema the emotion to predict. -done
C. Health Response (Server health check)
"""

class TextInput(BaseModel):
    text : str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"}
    )
    
class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence : float
    all_probabilites: dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

"""
4. Model Loading and LifeSpan Management
Load the model and toknizer once the server starts up.
"""
dl_model = {} #{1. BiGRU, 2. Tokenizer}-> True , {} -> False

async def lifespan(app:FastAPI):
    print('Loading the model and tokenizer...')
    dl_model["BiGRU"] = load_model(model_path)                      #BiGRU Model
    with open(tokenizer_path, 'rb') as file:
        dl_model["Tokenizer"] = pickle.load(file)
    print('Model are loaded successfully...')   

    yield #Pause, model is laoded and server is running and at this point model wait karega for request

    dl_model.clear() #Ek baar server band ho gaya uske baad model ko memory se hata do.
               