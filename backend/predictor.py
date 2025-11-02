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
import numpy as np
from tokenizers import Tokenizer
from disan import DiSAN
from context_fusion import ContextFusion
from casa_nlu import CASA_NLU


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

class CASA_NLU_Predictor:
    """
    Wrapper class for CASA_NLU model inference with turn history management.
    Handles dialogue context across multiple turns.
    """

    def __init__(self, model, tokenizer, device='cpu', max_length=128,
                 context_window=3, hidden_dim=56):
        """
        Initialize the predictor.

        Args:
            model: Trained CASA_NLU model
            tokenizer: Tokenizer (from tokenizers library)
            device: Device to run inference on
            max_length: Maximum sequence length
            context_window: Number of turns to maintain in history
            hidden_dim: Hidden dimension size (must match model)
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.context_window = context_window
        self.hidden_dim = hidden_dim

        # Turn history buffer - stores encoded representations of past turns
        self.turn_history_buffer = []

        # Get special token IDs
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")

    def reset_history(self):
        """Reset the turn history buffer. Call this at the start of a new dialogue."""
        self.turn_history_buffer = []
        print("🔄 Turn history reset")

    def _encode_utterance(self, utterance):
        """
        Tokenize and encode a single utterance.

        Args:
            utterance: String input (e.g., "hi find me hotels")

        Returns:
            dict with input_ids, attention_mask, and tokens
        """
        # Encode without special tokens first
        encoding = self.tokenizer.encode(utterance, add_special_tokens=False)
        token_ids = encoding.ids
        tokens = encoding.tokens

        # Add special tokens
        token_ids = [self.cls_id] + token_ids + [self.sep_id]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            tokens = tokens[:self.max_length]

        # Create attention mask
        attention_mask = [1] * len(token_ids)

        # Pad to max_length
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            tokens = tokens + ["[PAD]"] * padding_length

        return {
            'input_ids': torch.tensor([token_ids], dtype=torch.long),  # Add batch dim
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long),
            'tokens': tokens
        }

    def _encode_utterance_vector(self, input_ids):
        """
        Extract the utterance vector from the model's DiSAN encoder.
        This is the vector that should be stored in turn history.

        Args:
            input_ids: Tensor of token ids, shape (1, seq_len)

        Returns:
            Utterance vector of shape (1, hidden_dim)
        """
        with torch.no_grad():
            # Pass through embedding and DiSAN encoder
            x = self.model.embedding(input_ids)  # (1, seq_len, embed_dim)
            utt_vec, _ = self.model.disan(x)     # (1, hidden_dim), (1, seq_len, hidden_dim)
        return utt_vec

    def _update_turn_history(self, input_ids):
        """
        Update the turn history buffer with the current utterance encoding.
        Extracts the actual DiSAN encoding from the model.

        Args:
            input_ids: Tensor of token ids for the current utterance
        """
        # Extract the actual utterance vector from DiSAN encoder
        utt_vec = self._encode_utterance_vector(input_ids)

        # Add to buffer
        self.turn_history_buffer.append(utt_vec)

        # Keep only last K turns (sliding window)
        if len(self.turn_history_buffer) > self.context_window:
            self.turn_history_buffer.pop(0)

    def _get_turn_history_tensor(self):
        """
        Convert turn history buffer to model input tensor.
        Pads with the most recent encoding if buffer is not full.

        Returns:
            Tensor of shape (1, context_window, hidden_dim)
        """
        history_tensors = []

        # If we have fewer turns than context_window, pad with the first available encoding
        # or zeros if no history exists yet
        for i in range(self.context_window):
            if i < len(self.turn_history_buffer):
                history_tensors.append(self.turn_history_buffer[i])
            else:
                # Pad with zeros for turns that haven't happened yet
                history_tensors.append(
                    torch.zeros(1, self.hidden_dim, device=self.device)
                )

        # Stack: [(1, H), (1, H), ...] -> (context_window, 1, H) -> (1, context_window, H)
        turn_history = torch.stack(history_tensors, dim=0).squeeze(1).unsqueeze(0)
        return turn_history

    def predict(self, utterance, return_probabilities=False, update_history=True):
        """
        Predict intent and slots for a single utterance.

        Args:
            utterance: String input (e.g., "hi find me hotels")
            return_probabilities: If True, return prediction probabilities
            update_history: If True, update turn history with this utterance

        Returns:
            dict containing predictions and optionally probabilities
        """
        # Encode utterance
        encoded = self._encode_utterance(utterance)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        tokens = encoded['tokens']

        # Get turn history tensor
        turn_history = self._get_turn_history_tensor()

        # Run inference
        with torch.no_grad():
            intent_logits, slot_logits, sec_intent_logits = self.model(
                input_ids, turn_history
            )

            # Get predictions
            intent_probs = F.softmax(intent_logits, dim=-1)
            intent_pred_id = torch.argmax(intent_logits, dim=-1).item()

            sec_intent_probs = F.softmax(sec_intent_logits, dim=-1)
            sec_intent_pred_id = torch.argmax(sec_intent_logits, dim=-1).item()

            slot_probs = F.softmax(slot_logits, dim=-1)
            slot_pred_ids = torch.argmax(slot_logits, dim=-1)[0].cpu().numpy()

        # Convert IDs to labels
        intent_pred = id2intent[intent_pred_id]
        sec_intent_pred = id2intent[sec_intent_pred_id]

        # Extract slot predictions (only for non-padded tokens)
        slot_predictions = []
        mask = attention_mask[0].cpu().numpy()
        for j, token in enumerate(tokens):
            if mask[j] == 1 and token not in ["[PAD]", "[CLS]", "[SEP]"]:
                slot_label = id2bio[int(slot_pred_ids[j])]
                slot_predictions.append((token, slot_label))

        # Update turn history if requested
        if update_history:
            self._update_turn_history(input_ids)

        # Prepare result
        result = {
            'utterance': utterance,
            'intent': intent_pred,
            'secondary_intent': sec_intent_pred,
            'slots': slot_predictions,
            'turn_number': len(self.turn_history_buffer)
        }

        if return_probabilities:
            result['intent_probabilities'] = intent_probs[0].cpu().numpy()
            result['secondary_intent_probabilities'] = sec_intent_probs[0].cpu().numpy()
            result['slot_probabilities'] = slot_probs[0].cpu().numpy()

        return result

    def predict_batch(self, utterances, return_probabilities=False):
        """
        Predict for a batch of utterances (without turn history).
        Use this for batch processing where turn history is not needed.

        Args:
            utterances: List of strings
            return_probabilities: If True, return prediction probabilities

        Returns:
            List of prediction dictionaries
        """
        batch_size = len(utterances)

        # Encode all utterances
        encoded_batch = [self._encode_utterance(utt) for utt in utterances]

        # Stack into batch tensors
        input_ids = torch.cat([e['input_ids'] for e in encoded_batch], dim=0).to(self.device)
        attention_mask = torch.cat([e['attention_mask'] for e in encoded_batch], dim=0).to(self.device)
        tokens_list = [e['tokens'] for e in encoded_batch]

        # Create zero turn history for batch
        turn_history = torch.zeros(
            batch_size, self.context_window, self.hidden_dim,
            device=self.device
        )

        # Run inference
        with torch.no_grad():
            intent_logits, slot_logits, sec_intent_logits = self.model(
                input_ids, turn_history
            )

            intent_probs = F.softmax(intent_logits, dim=-1)
            intent_pred_ids = torch.argmax(intent_logits, dim=-1).cpu().numpy()

            sec_intent_probs = F.softmax(sec_intent_logits, dim=-1)
            sec_intent_pred_ids = torch.argmax(sec_intent_logits, dim=-1).cpu().numpy()

            slot_probs = F.softmax(slot_logits, dim=-1)
            slot_pred_ids = torch.argmax(slot_logits, dim=-1).cpu().numpy()

        # Convert to results
        results = []
        for i in range(batch_size):
            intent_pred = id2intent[int(intent_pred_ids[i])]
            sec_intent_pred = id2intent[int(sec_intent_pred_ids[i])]

            slot_predictions = []
            mask = attention_mask[i].cpu().numpy()
            for j, token in enumerate(tokens_list[i]):
                if mask[j] == 1 and token not in ["[PAD]", "[CLS]", "[SEP]"]:
                    slot_label = id2bio[int(slot_pred_ids[i, j])]
                    slot_predictions.append((token, slot_label))

            result = {
                'utterance': utterances[i],
                'intent': intent_pred,
                'secondary_intent': sec_intent_pred,
                'slots': slot_predictions
            }

            if return_probabilities:
                result['intent_probabilities'] = intent_probs[i].cpu().numpy()
                result['secondary_intent_probabilities'] = sec_intent_probs[i].cpu().numpy()
                result['slot_probabilities'] = slot_probs[i].cpu().numpy()

            results.append(result)

        return results


# ============================================
# USAGE EXAMPLES
# ============================================

def example_single_turn():
    """Example: Single utterance prediction"""
    # Load model and tokenizer
    # model = CASA_NLU(...)  # Your trained model
    # tokenizer = Tokenizer.from_file("whitespace_tokenizer.json")

    # Initialize predictor
    predictor = CASA_NLU_Predictor(
        model=model,
        tokenizer=tokenizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_length=128,
        context_window=3,
        hidden_dim=56
    )

    # Single prediction
    result = predictor.predict("hi find me hotels")

    print("🎯 Prediction Result:")
    print(f"  Utterance: {result['utterance']}")
    print(f"  Intent: {result['intent']}")
    print(f"  Secondary Intent: {result['secondary_intent']}")
    print(f"  Slots:")
    for token, slot in result['slots']:
        if slot != 'O':
            print(f"    {token}: {slot}")


def example_understanding_turn_history():
    """Example: Understanding how turn history works"""
    predictor = CASA_NLU_Predictor(
        model=model,
        tokenizer=tokenizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        context_window=3,
        hidden_dim=56
    )

    print("🔍 Understanding Turn History:\n")
    print("Context Window = 3 (keeps last 3 turn encodings)\n")

    # Turn 1: No history yet
    print("Turn 1: 'find hotels in new york'")
    result1 = predictor.predict("find hotels in new york")
    print(f"  History buffer size: {len(predictor.turn_history_buffer)}")
    print(f"  Intent: {result1['intent']}\n")

    # Turn 2: Has 1 previous turn
    print("Turn 2: 'what about ones near times square'")
    result2 = predictor.predict("what about ones near times square")
    print(f"  History buffer size: {len(predictor.turn_history_buffer)}")
    print(f"  Intent: {result2['intent']}")
    print("  Context: Model now knows about the previous 'hotel' query\n")

    # Turn 3: Has 2 previous turns
    print("Turn 3: 'book the cheapest one'")
    result3 = predictor.predict("book the cheapest one")
    print(f"  History buffer size: {len(predictor.turn_history_buffer)}")
    print(f"  Intent: {result3['intent']}")
    print("  Context: Model has full context from last 3 turns\n")

import torch
import torch.nn.functional as F
import numpy as np
from tokenizers import Tokenizer


class CASA_NLU_Predictor:
    """
    Wrapper class for CASA_NLU model inference with turn history management.
    Handles dialogue context across multiple turns.
    """

    def __init__(self, model, tokenizer, device='cpu', max_length=128,
                 context_window=3, hidden_dim=56):
        """
        Initialize the predictor.

        Args:
            model: Trained CASA_NLU model
            tokenizer: Tokenizer (from tokenizers library)
            device: Device to run inference on
            max_length: Maximum sequence length
            context_window: Number of turns to maintain in history
            hidden_dim: Hidden dimension size (must match model)
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.context_window = context_window
        self.hidden_dim = hidden_dim

        # Turn history buffer - stores encoded representations of past turns
        self.turn_history_buffer = []

        # Get special token IDs
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")

    def reset_history(self):
        """Reset the turn history buffer. Call this at the start of a new dialogue."""
        self.turn_history_buffer = []
        print("🔄 Turn history reset")

    def _encode_utterance(self, utterance):
        """
        Tokenize and encode a single utterance.

        Args:
            utterance: String input (e.g., "hi find me hotels")

        Returns:
            dict with input_ids, attention_mask, and tokens
        """
        # Encode without special tokens first
        encoding = self.tokenizer.encode(utterance, add_special_tokens=False)
        token_ids = encoding.ids
        tokens = encoding.tokens

        # Add special tokens
        token_ids = [self.cls_id] + token_ids + [self.sep_id]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            tokens = tokens[:self.max_length]

        # Create attention mask
        attention_mask = [1] * len(token_ids)

        # Pad to max_length
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.pad_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            tokens = tokens + ["[PAD]"] * padding_length

        return {
            'input_ids': torch.tensor([token_ids], dtype=torch.long),  # Add batch dim
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long),
            'tokens': tokens
        }

    def _encode_utterance_vector(self, input_ids):
        """
        Extract the utterance vector from the model's DiSAN encoder.
        This is the vector that should be stored in turn history.

        Args:
            input_ids: Tensor of token ids, shape (1, seq_len)

        Returns:
            Utterance vector of shape (1, hidden_dim)
        """
        with torch.no_grad():
            # Pass through embedding and DiSAN encoder
            x = self.model.embedding(input_ids)  # (1, seq_len, embed_dim)
            utt_vec, _ = self.model.disan(x)     # (1, hidden_dim), (1, seq_len, hidden_dim)
        return utt_vec

    def _update_turn_history(self, input_ids):
        """
        Update the turn history buffer with the current utterance encoding.
        Extracts the actual DiSAN encoding from the model.

        Args:
            input_ids: Tensor of token ids for the current utterance
        """
        # Extract the actual utterance vector from DiSAN encoder
        utt_vec = self._encode_utterance_vector(input_ids)

        # Add to buffer
        self.turn_history_buffer.append(utt_vec)

        # Keep only last K turns (sliding window)
        if len(self.turn_history_buffer) > self.context_window:
            self.turn_history_buffer.pop(0)

    def _get_turn_history_tensor(self):
        """
        Convert turn history buffer to model input tensor.
        Pads with the most recent encoding if buffer is not full.

        Returns:
            Tensor of shape (1, context_window, hidden_dim)
        """
        history_tensors = []

        # If we have fewer turns than context_window, pad with the first available encoding
        # or zeros if no history exists yet
        for i in range(self.context_window):
            if i < len(self.turn_history_buffer):
                history_tensors.append(self.turn_history_buffer[i])
            else:
                # Pad with zeros for turns that haven't happened yet
                history_tensors.append(
                    torch.zeros(1, self.hidden_dim, device=self.device)
                )

        # Stack: [(1, H), (1, H), ...] -> (context_window, 1, H) -> (1, context_window, H)
        turn_history = torch.stack(history_tensors, dim=0).squeeze(1).unsqueeze(0)
        return turn_history

    def predict(self, utterance, return_probabilities=False, update_history=True):
        """
        Predict intent and slots for a single utterance.

        Args:
            utterance: String input (e.g., "hi find me hotels")
            return_probabilities: If True, return prediction probabilities
            update_history: If True, update turn history with this utterance

        Returns:
            dict containing predictions and optionally probabilities
        """
        # Encode utterance
        encoded = self._encode_utterance(utterance)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        tokens = encoded['tokens']

        # Get turn history tensor
        turn_history = self._get_turn_history_tensor()

        # Run inference
        with torch.no_grad():
            intent_logits, slot_logits, sec_intent_logits = self.model(
                input_ids, turn_history
            )

            # Get predictions
            intent_probs = F.softmax(intent_logits, dim=-1)
            intent_pred_id = torch.argmax(intent_logits, dim=-1).item()

            sec_intent_probs = F.softmax(sec_intent_logits, dim=-1)
            sec_intent_pred_id = torch.argmax(sec_intent_logits, dim=-1).item()

            slot_probs = F.softmax(slot_logits, dim=-1)
            slot_pred_ids = torch.argmax(slot_logits, dim=-1)[0].cpu().numpy()

        # Convert IDs to labels
        intent_pred = id2intent[intent_pred_id]
        sec_intent_pred = id2intent[sec_intent_pred_id]

        # Extract slot predictions (only for non-padded tokens)
        slot_predictions = []
        mask = attention_mask[0].cpu().numpy()
        for j, token in enumerate(tokens):
            if mask[j] == 1 and token not in ["[PAD]", "[CLS]", "[SEP]"]:
                slot_label = id2bio[int(slot_pred_ids[j])]
                slot_predictions.append((token, slot_label))

        # Update turn history if requested
        if update_history:
            self._update_turn_history(input_ids)

        # Prepare result
        result = {
            'utterance': utterance,
            'intent': intent_pred,
            'secondary_intent': sec_intent_pred,
            'slots': slot_predictions,
            'turn_number': len(self.turn_history_buffer)
        }

        if return_probabilities:
            result['intent_probabilities'] = intent_probs[0].cpu().numpy()
            result['secondary_intent_probabilities'] = sec_intent_probs[0].cpu().numpy()
            result['slot_probabilities'] = slot_probs[0].cpu().numpy()

        return result

    def predict_batch(self, utterances, return_probabilities=False):
        """
        Predict for a batch of utterances (without turn history).
        Use this for batch processing where turn history is not needed.

        Args:
            utterances: List of strings
            return_probabilities: If True, return prediction probabilities

        Returns:
            List of prediction dictionaries
        """
        batch_size = len(utterances)

        # Encode all utterances
        encoded_batch = [self._encode_utterance(utt) for utt in utterances]

        # Stack into batch tensors
        input_ids = torch.cat([e['input_ids'] for e in encoded_batch], dim=0).to(self.device)
        attention_mask = torch.cat([e['attention_mask'] for e in encoded_batch], dim=0).to(self.device)
        tokens_list = [e['tokens'] for e in encoded_batch]

        # Create zero turn history for batch
        turn_history = torch.zeros(
            batch_size, self.context_window, self.hidden_dim,
            device=self.device
        )

        # Run inference
        with torch.no_grad():
            intent_logits, slot_logits, sec_intent_logits = self.model(
                input_ids, turn_history
            )

            intent_probs = F.softmax(intent_logits, dim=-1)
            intent_pred_ids = torch.argmax(intent_logits, dim=-1).cpu().numpy()

            sec_intent_probs = F.softmax(sec_intent_logits, dim=-1)
            sec_intent_pred_ids = torch.argmax(sec_intent_logits, dim=-1).cpu().numpy()

            slot_probs = F.softmax(slot_logits, dim=-1)
            slot_pred_ids = torch.argmax(slot_logits, dim=-1).cpu().numpy()

        # Convert to results
        results = []
        for i in range(batch_size):
            intent_pred = id2intent[int(intent_pred_ids[i])]
            sec_intent_pred = id2intent[int(sec_intent_pred_ids[i])]

            slot_predictions = []
            mask = attention_mask[i].cpu().numpy()
            for j, token in enumerate(tokens_list[i]):
                if mask[j] == 1 and token not in ["[PAD]", "[CLS]", "[SEP]"]:
                    slot_label = id2bio[int(slot_pred_ids[i, j])]
                    slot_predictions.append((token, slot_label))

            result = {
                'utterance': utterances[i],
                'intent': intent_pred,
                'secondary_intent': sec_intent_pred,
                'slots': slot_predictions
            }

            if return_probabilities:
                result['intent_probabilities'] = intent_probs[i].cpu().numpy()
                result['secondary_intent_probabilities'] = sec_intent_probs[i].cpu().numpy()
                result['slot_probabilities'] = slot_probs[i].cpu().numpy()

            results.append(result)

        return results


# ============================================
# USAGE EXAMPLES
# ============================================

def example_single_turn():
    """Example: Single utterance prediction"""
    # Load model and tokenizer
    # model = CASA_NLU(...)  # Your trained model
    # tokenizer = Tokenizer.from_file("whitespace_tokenizer.json")

    # Initialize predictor
    predictor = CASA_NLU_Predictor(
        model=model,
        tokenizer=tokenizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_length=128,
        context_window=3,
        hidden_dim=56
    )

    # Single prediction
    result = predictor.predict("hi find me hotels")

    print("🎯 Prediction Result:")
    print(f"  Utterance: {result['utterance']}")
    print(f"  Intent: {result['intent']}")
    print(f"  Secondary Intent: {result['secondary_intent']}")
    print(f"  Slots:")
    for token, slot in result['slots']:
        if slot != 'O':
            print(f"    {token}: {slot}")


def example_understanding_turn_history():
    """Example: Understanding how turn history works"""
    predictor = CASA_NLU_Predictor(
        model=model,
        tokenizer=tokenizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        context_window=3,
        hidden_dim=56
    )

    print("🔍 Understanding Turn History:\n")
    print("Context Window = 3 (keeps last 3 turn encodings)\n")

    # Turn 1: No history yet
    print("Turn 1: 'find hotels in new york'")
    result1 = predictor.predict("find hotels in new york")
    print(f"  History buffer size: {len(predictor.turn_history_buffer)}")
    print(f"  Intent: {result1['intent']}\n")

    # Turn 2: Has 1 previous turn
    print("Turn 2: 'what about ones near times square'")
    result2 = predictor.predict("what about ones near times square")
    print(f"  History buffer size: {len(predictor.turn_history_buffer)}")
    print(f"  Intent: {result2['intent']}")
    print("  Context: Model now knows about the previous 'hotel' query\n")

    # Turn 3: Has 2 previous turns
    print("Turn 3: 'book the cheapest one'")
    result3 = predictor.predict("book the cheapest one")
    print(f"  History buffer size: {len(predictor.turn_history_buffer)}")
    print(f"  Intent: {result3['intent']}")
    print("  Context: Model has full context from last 3 turns\n")

    # Turn 4: Buffer is full, will drop oldest
    print("Turn 4: 'for 2 nights'")
    result4 = predictor.predict("for 2 nights")
    print(f"  History buffer size: {len(predictor.turn_history_buffer)}")
    print(f"  Intent: {result4['intent']}")
    print("  Context: Oldest turn (Turn 1) is dropped, keeps last 3\n")

    # Reset for new conversation
    print("🔄 Starting new conversation...")
    predictor.reset_history()
    print(f"  History buffer size: {len(predictor.turn_history_buffer)}\n")


def example_real_world_usage():
    """Example: Real-world chatbot usage"""
    # Initialize predictor
    predictor = CASA_NLU_Predictor(
        model=model,
        tokenizer=tokenizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        context_window=3,
        hidden_dim=56
    )

    print("🤖 Chatbot Interaction:\n")

    # Simulate user conversation
    conversation = [
        "hi there",
        "I want to book a flight",
        "to Paris from New York",
        "on December 25th",
        "make it business class"
    ]

    for turn_num, user_input in enumerate(conversation, 1):
        # Get prediction with context
        result = predictor.predict(user_input, update_history=True)

        print(f"👤 User (Turn {turn_num}): {user_input}")
        print(f"🤖 Bot Understanding:")
        print(f"   Intent: {result['intent']}")
        print(f"   Entities: {[(tok, slot) for tok, slot in result['slots'] if slot != 'O']}")
        print(f"   Context turns: {result['turn_number']}")
        print()

    # Start a completely new conversation
    print("="*60)
    print("New conversation started\n")
    predictor.reset_history()

    result = predictor.predict("what's the weather like")
    print(f"👤 User: what's the weather like")
    print(f"🤖 Bot: {result['intent']}")
    print(f"   Context turns: {result['turn_number']} (fresh start)")


def example_comparing_with_without_history():
    """Example: Compare predictions with and without turn history"""
    predictor = CASA_NLU_Predictor(
        model=model,
        tokenizer=tokenizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        context_window=3,
        hidden_dim=56
    )

    print("📊 Comparing: With vs Without Turn History\n")

    # First utterance
    utterance1 = "find hotels in san francisco"
    result1 = predictor.predict(utterance1)
    print(f"Turn 1: '{utterance1}'")
    print(f"  Intent: {result1['intent']}\n")

    # Ambiguous follow-up that needs context
    utterance2 = "what about the cheap ones"

    # WITH history (normal flow)
    result2_with = predictor.predict(utterance2, update_history=True)
    print(f"Turn 2 WITH history: '{utterance2}'")
    print(f"  Intent: {result2_with['intent']}")
    print(f"  (Model knows we're still talking about hotels)\n")

    # WITHOUT history (reset first)
    predictor.reset_history()
    result2_without = predictor.predict(utterance2, update_history=False)
    print(f"Turn 2 WITHOUT history: '{utterance2}'")
    print(f"  Intent: {result2_without['intent']}")
    print(f"  (Model has no context, may misunderstand)\n")


def example_batch_processing():
    """Example: Batch processing without turn history"""
    predictor = CASA_NLU_Predictor(
        model=model,
        tokenizer=tokenizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        context_window=3,
        hidden_dim=56
    )

    utterances = [
        "hi find me hotels",
        "book a flight to paris",
        "what's the weather tomorrow"
    ]

    results = predictor.predict_batch(utterances)

    print("📦 Batch Predictions:\n")
    for result in results:
        print(f"Utterance: {result['utterance']}")
        print(f"  Intent: {result['intent']}")
        print(f"  Slots: {[(t, s) for t, s in result['slots'] if s != 'O']}")
        print()


# Quick start usage:
"""
# 1. Load your trained model and tokenizer
model = CASA_NLU(...)  # Load your trained model
tokenizer = Tokenizer.from_file("whitespace_tokenizer.json")

# 2. Create predictor
predictor = CASA_NLU_Predictor(
    model=model,
    tokenizer=tokenizer,
    device='cuda',
    context_window=3,
    hidden_dim=56
)

# 3. Make predictions
result = predictor.predict("hi find me hotels")
print(f"Intent: {result['intent']}")
print(f"Slots: {result['slots']}")
"""
   