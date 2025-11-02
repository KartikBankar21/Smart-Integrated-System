# Bio2id mapping (as provided)
bio2id = {'O': 0,
 'B-account_balance': 1,
 'I-account_balance': 2,
 'B-actors': 3,
 'I-actors': 4,
 'B-address': 5,
 'I-address': 6,
 'B-address_of_location': 7,
 'I-address_of_location': 8,
 'B-aggregate_rating': 9,
 'I-aggregate_rating': 10,
 'B-alarm_name': 11,
 'I-alarm_name': 12,
 'B-alarm_time': 13,
 'I-alarm_time': 14,
 'B-album': 15,
 'I-album': 16,
 'B-amount': 17,
 'I-amount': 18,
 'B-appointment_date': 19,
 'I-appointment_date': 20,
 'B-appointment_time': 21,
 'I-appointment_time': 22,
 'B-approximate_ride_duration': 23,
 'I-approximate_ride_duration': 24,
 'B-area': 25,
 'I-area': 26,
 'B-artist': 27,
 'I-artist': 28,
 'B-attraction_name': 29,
 'I-attraction_name': 30,
 'B-available_end_time': 31,
 'I-available_end_time': 32,
 'B-available_start_time': 33,
 'I-available_start_time': 34,
 'B-average_rating': 35,
 'I-average_rating': 36,
 'B-balance': 37,
 'I-balance': 38,
 'B-car_name': 39,
 'I-car_name': 40,
 'B-cast': 41,
 'I-cast': 42,
 'B-category': 43,
 'I-category': 44,
 'B-check_in_date': 45,
 'I-check_in_date': 46,
 'B-check_out_date': 47,
 'I-check_out_date': 48,
 'B-city': 49,
 'I-city': 50,
 'B-city_of_event': 51,
 'I-city_of_event': 52,
 'B-contact_name': 53,
 'I-contact_name': 54,
 'B-cuisine': 55,
 'I-cuisine': 56,
 'B-date': 57,
 'I-date': 58,
 'B-date_of_journey': 59,
 'I-date_of_journey': 60,
 'B-dentist_name': 61,
 'I-dentist_name': 62,
 'B-departure_date': 63,
 'I-departure_date': 64,
 'B-departure_time': 65,
 'I-departure_time': 66,
 'B-destination': 67,
 'I-destination': 68,
 'B-destination_airport': 69,
 'I-destination_airport': 70,
 'B-destination_airport_name': 71,
 'I-destination_airport_name': 72,
 'B-destination_city': 73,
 'I-destination_city': 74,
 'B-destination_station_name': 75,
 'I-destination_station_name': 76,
 'B-directed_by': 77,
 'I-directed_by': 78,
 'B-director': 79,
 'I-director': 80,
 'B-doctor_name': 81,
 'I-doctor_name': 82,
 'B-dropoff_date': 83,
 'I-dropoff_date': 84,
 'B-end_date': 85,
 'I-end_date': 86,
 'B-event_date': 87,
 'I-event_date': 88,
 'B-event_location': 89,
 'I-event_location': 90,
 'B-event_name': 91,
 'I-event_name': 92,
 'B-event_time': 93,
 'I-event_time': 94,
 'B-fare': 95,
 'I-fare': 96,
 'B-from': 97,
 'I-from': 98,
 'B-from_city': 99,
 'I-from_city': 100,
 'B-from_location': 101,
 'I-from_location': 102,
 'B-from_station': 103,
 'I-from_station': 104,
 'B-genre': 105,
 'I-genre': 106,
 'B-hotel_name': 107,
 'I-hotel_name': 108,
 'B-humidity': 109,
 'I-humidity': 110,
 'B-inbound_arrival_time': 111,
 'I-inbound_arrival_time': 112,
 'B-inbound_departure_time': 113,
 'I-inbound_departure_time': 114,
 'B-journey_start_time': 115,
 'I-journey_start_time': 116,
 'B-leaving_date': 117,
 'I-leaving_date': 118,
 'B-leaving_time': 119,
 'I-leaving_time': 120,
 'B-location': 121,
 'I-location': 122,
 'B-movie_name': 123,
 'I-movie_name': 124,
 'B-movie_title': 125,
 'I-movie_title': 126,
 'B-new_alarm_name': 127,
 'I-new_alarm_name': 128,
 'B-new_alarm_time': 129,
 'I-new_alarm_time': 130,
 'B-number_of_days': 131,
 'I-number_of_days': 132,
 'B-origin': 133,
 'I-origin': 134,
 'B-origin_airport': 135,
 'I-origin_airport': 136,
 'B-origin_airport_name': 137,
 'I-origin_airport_name': 138,
 'B-origin_city': 139,
 'I-origin_city': 140,
 'B-origin_station_name': 141,
 'I-origin_station_name': 142,
 'B-outbound_arrival_time': 143,
 'I-outbound_arrival_time': 144,
 'B-outbound_departure_time': 145,
 'I-outbound_departure_time': 146,
 'B-percent_rating': 147,
 'I-percent_rating': 148,
 'B-phone_number': 149,
 'I-phone_number': 150,
 'B-pickup_city': 151,
 'I-pickup_city': 152,
 'B-pickup_date': 153,
 'I-pickup_date': 154,
 'B-pickup_location': 155,
 'I-pickup_location': 156,
 'B-pickup_time': 157,
 'I-pickup_time': 158,
 'B-place_name': 159,
 'I-place_name': 160,
 'B-precipitation': 161,
 'I-precipitation': 162,
 'B-price': 163,
 'I-price': 164,
 'B-price_per_day': 165,
 'I-price_per_day': 166,
 'B-price_per_night': 167,
 'I-price_per_night': 168,
 'B-price_per_ticket': 169,
 'I-price_per_ticket': 170,
 'B-property_name': 171,
 'I-property_name': 172,
 'B-rating': 173,
 'I-rating': 174,
 'B-receiver': 175,
 'I-receiver': 176,
 'B-recipient_account_name': 177,
 'I-recipient_account_name': 178,
 'B-recipient_name': 179,
 'I-recipient_name': 180,
 'B-rent': 181,
 'I-rent': 182,
 'B-restaurant_name': 183,
 'I-restaurant_name': 184,
 'B-return_date': 185,
 'I-return_date': 186,
 'B-ride_fare': 187,
 'I-ride_fare': 188,
 'B-show_date': 189,
 'I-show_date': 190,
 'B-show_time': 191,
 'I-show_time': 192,
 'B-song_name': 193,
 'I-song_name': 194,
 'B-starring': 195,
 'I-starring': 196,
 'B-start_date': 197,
 'I-start_date': 198,
 'B-stay_length': 199,
 'I-stay_length': 200,
 'B-street_address': 201,
 'I-street_address': 202,
 'B-stylist_name': 203,
 'I-stylist_name': 204,
 'B-subcategory': 205,
 'I-subcategory': 206,
 'B-temperature': 207,
 'I-temperature': 208,
 'B-theater_name': 209,
 'I-theater_name': 210,
 'B-therapist_name': 211,
 'I-therapist_name': 212,
 'B-time': 213,
 'I-time': 214,
 'B-title': 215,
 'I-title': 216,
 'B-to': 217,
 'I-to': 218,
 'B-to_city': 219,
 'I-to_city': 220,
 'B-to_location': 221,
 'I-to_location': 222,
 'B-to_station': 223,
 'I-to_station': 224,
 'B-total': 225,
 'I-total': 226,
 'B-total_price': 227,
 'I-total_price': 228,
 'B-track': 229,
 'I-track': 230,
 'B-transfer_amount': 231,
 'I-transfer_amount': 232,
 'B-transfer_time': 233,
 'I-transfer_time': 234,
 'B-venue': 235,
 'I-venue': 236,
 'B-venue_address': 237,
 'I-venue_address': 238,
 'B-visit_date': 239,
 'I-visit_date': 240,
 'B-wait_time': 241,
 'I-wait_time': 242,
 'B-where_to': 243,
 'I-where_to': 244,
 'B-wind': 245,
 'I-wind': 246}

# Intent2id mapping (as provided)
intent2id = {
    'AddAlarm': 0, 'AddEvent': 1, 'BookAppointment': 2, 'BookHouse': 3,
    'BuyBusTicket': 4, 'BuyEventTickets': 5, 'BuyMovieTickets': 6,
    'CheckBalance': 7, 'FindApartment': 8, 'FindAttractions': 9,
    'FindBus': 10, 'FindEvents': 11, 'FindHomeByArea': 12, 'FindMovies': 13,
    'FindProvider': 14, 'FindRestaurants': 15, 'FindTrains': 16,
    'GetAlarms': 17, 'GetAvailableTime': 18, 'GetCarsAvailable': 19,
    'GetEventDates': 20, 'GetEvents': 21, 'GetRide': 22,
    'GetTimesForMovie': 23, 'GetTrainTickets': 24, 'GetWeather': 25,
    'LookupMusic': 26, 'LookupSong': 27, 'MakePayment': 28, 'NONE': 29,
    'PlayMedia': 30, 'PlayMovie': 31, 'PlaySong': 32, 'RentMovie': 33,
    'RequestPayment': 34, 'ReserveCar': 35, 'ReserveHotel': 36,
    'ReserveOnewayFlight': 37, 'ReserveRestaurant': 38,
    'ReserveRoundtripFlights': 39, 'ScheduleVisit': 40, 'SearchHotel': 41,
    'SearchHouse': 42, 'SearchOnewayFlight': 43, 'SearchRoundtripFlights': 44,
    'ShareLocation': 45, 'TransferMoney': 46
}

# Create reverse mappings
id2bio = {v: k for k, v in bio2id.items()}
id2intent = {v: k for k, v in intent2id.items()}


import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from tokenizers import Tokenizer

# Import your model components
from disan import DiSAN
from context_fusion import ContextFusion
from casa_nlu import CASA_NLU

# Import predictor and response generator (production versions)
from predictor import CASA_NLU_Predictor
from response_generator import ResponseGenerator, create_chat_response

# ============================================
# Configuration
# ============================================


# ============================================
# Request/Response Models
# ============================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    return_probabilities: Optional[bool] = False

class SlotInfo(BaseModel):
    token: str
    label: str

class ChatResponse(BaseModel):
    message: str
    intent: str
    secondary_intent: str
    slots: List[SlotInfo]
    response: str
    turn_number: int
    session_id: str
    probabilities: Optional[Dict[str, Any]] = None

class ResetRequest(BaseModel):
    session_id: Optional[str] = "default"

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool

# ============================================
# FastAPI App Initialization
# ============================================

app = FastAPI(
    title="CASA NLU Chatbot API",
    description="Context-Aware Self-Attention NLU API for conversational understanding",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Global Variables
# ============================================

model = None
predictor = None
generator = None
device = None
predictors_cache = {}  # Cache for multiple sessions

# ============================================
# Startup Event
# ============================================

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, predictor, generator, device
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = Tokenizer.from_file("whitespace_tokenizer.json")
        print("✅ Tokenizer loaded")
        
        # Model hyperparameters
        VOCAB_SIZE = tokenizer.get_vocab_size()
        INTENT_SIZE = len(intent2id)
        SLOT_SIZE = len(bio2id)
        HIDDEN = 56
        EMBED = 56
        CTX_WINDOW = 3
        SLIDE_W = 3
        
        # Initialize model
        model = CASA_NLU(
            vocab_size=VOCAB_SIZE,
            intent_size=INTENT_SIZE,
            slot_size=SLOT_SIZE,
            hidden_dim=HIDDEN,
            embed_dim=EMBED,
            context_window=CTX_WINDOW,
            sliding_window=SLIDE_W,
            dropout=0.4
        )
        
        # Load weights
        model.load_state_dict(
            torch.load('model.pth', map_location=device)
        )
        model.eval()
        model.to(device)
        print("✅ Model loaded and moved to device")
        
        # Initialize predictor (default session)
        predictor = CASA_NLU_Predictor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            context_window=3,
            hidden_dim=56
        )
        predictors_cache["default"] = predictor
        print("✅ Predictor initialized")
        
        # Initialize response generator
        generator = ResponseGenerator()
        print("✅ Response generator initialized")
        
        print("🚀 All components loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

# ============================================
# Helper Functions
# ============================================

def get_or_create_predictor(session_id: str) -> CASA_NLU_Predictor:
    """Get existing predictor for session or create new one"""
    if session_id not in predictors_cache:
        tokenizer = Tokenizer.from_file("whitespace_tokenizer.json")
        predictors_cache[session_id] = CASA_NLU_Predictor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            context_window=3,
            hidden_dim=56
        )
    return predictors_cache[session_id]

# ============================================
# API Endpoints
# ============================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "device": str(device),
        "model_loaded": model is not None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    
    Process user message and return intent, slots, and generated response
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Get or create predictor for this session
        session_predictor = get_or_create_predictor(request.session_id)
        
        # Get prediction
        result = session_predictor.predict(
            request.message,
            return_probabilities=request.return_probabilities,
            update_history=True
        )
        
        # Generate response
        response_text = generator.generate_response(
            result['intent'],
            result['slots']
        )
        
        # Format slots
        slots_formatted = [
            SlotInfo(token=token, label=label)
            for token, label in result['slots']
        ]
        
        # Prepare response
        chat_response = ChatResponse(
            message=request.message,
            intent=result['intent'],
            secondary_intent=result['secondary_intent'],
            slots=slots_formatted,
            response=response_text,
            turn_number=result['turn_number'],
            session_id=request.session_id
        )
        
        # Add probabilities if requested
        if request.return_probabilities:
            chat_response.probabilities = {
                "intent": result['intent_probabilities'].tolist(),
                "secondary_intent": result['secondary_intent_probabilities'].tolist()
            }
        
        return chat_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/reset")
async def reset_history(request: ResetRequest):
    """Reset conversation history for a session"""
    try:
        session_id = request.session_id
        
        if session_id in predictors_cache:
            predictors_cache[session_id].reset_history()
            return {
                "status": "success",
                "message": f"History reset for session: {session_id}"
            }
        else:
            return {
                "status": "info",
                "message": f"No active session found: {session_id}"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting history: {str(e)}")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    try:
        if session_id in predictors_cache:
            del predictors_cache[session_id]
            return {
                "status": "success",
                "message": f"Session deleted: {session_id}"
            }
        else:
            return {
                "status": "info",
                "message": f"Session not found: {session_id}"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "active_sessions": list(predictors_cache.keys()),
        "count": len(predictors_cache)
    }

# ============================================
# Run with: uvicorn main:app --reload
# ============================================