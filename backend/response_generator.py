"""
Production-Ready Response Generator for CASA_NLU
Generates natural language responses based on predicted intents and slots
Optimized for FastAPI deployment
"""

from typing import List, Tuple, Dict, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates contextual responses based on intent and slot predictions.
    Thread-safe and optimized for production use.
    """
    
    def __init__(self, fallback_enabled: bool = True):
        """
        Initialize the response generator.
        
        Args:
            fallback_enabled: If True, return generic responses for unknown intents
        """
        self.fallback_enabled = fallback_enabled
        
        # Response templates mapping - each intent maps to a handler function
        self.response_templates = {
            # Alarm intents
            'AddAlarm': self._handle_add_alarm,
            'GetAlarms': self._handle_get_alarms,
            
            # Event intents
            'AddEvent': self._handle_add_event,
            'FindEvents': self._handle_find_events,
            'GetEvents': self._handle_get_events,
            'GetEventDates': self._handle_get_event_dates,
            'BuyEventTickets': self._handle_buy_event_tickets,
            
            # Appointment intents
            'BookAppointment': self._handle_book_appointment,
            'ScheduleVisit': self._handle_schedule_visit,
            'GetAvailableTime': self._handle_get_available_time,
            
            # Transportation intents
            'FindBus': self._handle_find_bus,
            'BuyBusTicket': self._handle_buy_bus_ticket,
            'FindTrains': self._handle_find_trains,
            'GetTrainTickets': self._handle_get_train_tickets,
            'GetRide': self._handle_get_ride,
            'GetCarsAvailable': self._handle_get_cars_available,
            'ReserveCar': self._handle_reserve_car,
            
            # Flight intents
            'SearchOnewayFlight': self._handle_search_oneway_flight,
            'SearchRoundtripFlights': self._handle_search_roundtrip_flights,
            'ReserveOnewayFlight': self._handle_reserve_oneway_flight,
            'ReserveRoundtripFlights': self._handle_reserve_roundtrip_flights,
            
            # Accommodation intents
            'SearchHotel': self._handle_search_hotel,
            'ReserveHotel': self._handle_reserve_hotel,
            'FindApartment': self._handle_find_apartment,
            'SearchHouse': self._handle_search_house,
            'BookHouse': self._handle_book_house,
            'FindHomeByArea': self._handle_find_home_by_area,
            
            # Restaurant intents
            'FindRestaurants': self._handle_find_restaurants,
            'ReserveRestaurant': self._handle_reserve_restaurant,
            
            # Entertainment intents
            'FindMovies': self._handle_find_movies,
            'GetTimesForMovie': self._handle_get_times_for_movie,
            'BuyMovieTickets': self._handle_buy_movie_tickets,
            'PlayMovie': self._handle_play_movie,
            'RentMovie': self._handle_rent_movie,
            
            # Music intents
            'LookupMusic': self._handle_lookup_music,
            'LookupSong': self._handle_lookup_song,
            'PlaySong': self._handle_play_song,
            'PlayMedia': self._handle_play_media,
            
            # Attractions
            'FindAttractions': self._handle_find_attractions,
            
            # Financial intents
            'CheckBalance': self._handle_check_balance,
            'TransferMoney': self._handle_transfer_money,
            'MakePayment': self._handle_make_payment,
            'RequestPayment': self._handle_request_payment,
            
            # Service provider intents
            'FindProvider': self._handle_find_provider,
            
            # Weather
            'GetWeather': self._handle_get_weather,
            
            # Location
            'ShareLocation': self._handle_share_location,
            
            # Default
            'NONE': self._handle_none,
        }
        
        logger.info(f"ResponseGenerator initialized with {len(self.response_templates)} intent handlers")
    
    def extract_slot_values(self, slots: List[Tuple[str, str]]) -> Dict[str, str]:
        """
        Extract slot values from (token, bio_label) pairs.
        Groups multi-token entities together using BIO tagging scheme.
        
        Args:
            slots: List of (token, bio_label) tuples from model prediction
            
        Returns:
            Dictionary mapping slot names to their extracted values
            
        Example:
            Input: [('taylor', 'B-artist'), ('swift', 'I-artist')]
            Output: {'artist': 'taylor swift'}
        """
        slot_values = {}
        current_slot = None
        current_value = []
        
        # Special tokens to ignore
        ignore_tokens = {'[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]'}
        
        for token, label in slots:
            # Skip special tokens
            if token in ignore_tokens:
                continue
                
            if label == 'O':
                # End current slot if exists
                if current_slot and current_value:
                    slot_values[current_slot] = ' '.join(current_value)
                    logger.debug(f"Completed slot: {current_slot} = {slot_values[current_slot]}")
                current_slot = None
                current_value = []
                
            elif label.startswith('B-'):
                # Begin new slot - first save previous slot if exists
                if current_slot and current_value:
                    slot_values[current_slot] = ' '.join(current_value)
                    logger.debug(f"Completed slot: {current_slot} = {slot_values[current_slot]}")
                
                # Start new slot
                current_slot = label[2:]  # Remove 'B-' prefix
                current_value = [token]
                
            elif label.startswith('I-'):
                # Continue current slot
                slot_name = label[2:]  # Remove 'I-' prefix
                
                # Handle case where I- tag appears without B- tag
                if current_slot != slot_name:
                    if current_slot and current_value:
                        slot_values[current_slot] = ' '.join(current_value)
                    current_slot = slot_name
                    current_value = [token]
                else:
                    current_value.append(token)
        
        # Don't forget to save the last slot
        if current_slot and current_value:
            slot_values[current_slot] = ' '.join(current_value)
            logger.debug(f"Completed slot: {current_slot} = {slot_values[current_slot]}")
        
        return slot_values
    
    def generate_response(
        self, 
        intent: str, 
        slots: List[Tuple[str, str]],
        include_metadata: bool = False
    ) -> str:
        """
        Generate a response based on intent and slots.
        
        Args:
            intent: Predicted intent string
            slots: List of (token, bio_label) tuples
            include_metadata: If True, include debug info in response
            
        Returns:
            Generated response string
        """
        try:
            # Extract slot values from BIO tags
            slot_values = self.extract_slot_values(slots)
            
            logger.info(f"Generating response for intent: {intent}")
            logger.debug(f"Extracted slots: {slot_values}")
            
            # Get appropriate handler function
            handler = self.response_templates.get(intent, self._handle_unknown)
            
            # Generate response
            response = handler(slot_values)
            
            # Add metadata if requested (useful for debugging)
            if include_metadata:
                response += f"\n[Intent: {intent}, Slots: {len(slot_values)}]"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return self._handle_error(intent)
    
    # ==================== Utility Methods ====================
    
    def _safe_get(self, slots: Dict[str, str], *keys: str) -> str:
        """
        Safely get a slot value, trying multiple possible keys.
        Returns first non-empty value found.
        
        Args:
            slots: Dictionary of slot values
            *keys: Variable number of keys to try
            
        Returns:
            First non-empty value found, or empty string
        """
        for key in keys:
            value = slots.get(key, '').strip()
            if value:
                return value
        return ''
    
    # ==================== Handler Methods ====================
    
    def _handle_add_alarm(self, slots: Dict[str, str]) -> str:
        alarm_name = self._safe_get(slots, 'alarm_name') or 'your alarm'
        alarm_time = self._safe_get(slots, 'alarm_time', 'time') or 'the specified time'
        return f"✅ I've set {alarm_name} for {alarm_time}."
    
    def _handle_get_alarms(self, slots: Dict[str, str]) -> str:
        return "📋 Here are your current alarms. Would you like to see them?"
    
    def _handle_add_event(self, slots: Dict[str, str]) -> str:
        event_name = self._safe_get(slots, 'event_name', 'event') or 'the event'
        event_date = self._safe_get(slots, 'event_date', 'date')
        event_time = self._safe_get(slots, 'event_time', 'time')
        
        response = f"✅ I've added {event_name} to your calendar"
        if event_date:
            response += f" on {event_date}"
        if event_time:
            response += f" at {event_time}"
        return response + "."
    
    def _handle_find_events(self, slots: Dict[str, str]) -> str:
        city = self._safe_get(slots, 'city_of_event', 'city', 'location')
        event_name = self._safe_get(slots, 'event_name', 'event')
        date = self._safe_get(slots, 'event_date', 'date')
        
        response = "🎭 Looking for events"
        if event_name:
            response += f" related to {event_name}"
        if city:
            response += f" in {city}"
        if date:
            response += f" on {date}"
        return response + "..."
    
    def _handle_get_events(self, slots: Dict[str, str]) -> str:
        city = self._safe_get(slots, 'city_of_event', 'city')
        if city:
            return f"📅 Here are the upcoming events in {city}."
        return "📅 Here are the upcoming events."
    
    def _handle_get_event_dates(self, slots: Dict[str, str]) -> str:
        event_name = self._safe_get(slots, 'event_name', 'event') or 'the event'
        return f"📆 Let me find available dates for {event_name}."
    
    def _handle_buy_event_tickets(self, slots: Dict[str, str]) -> str:
        event_name = self._safe_get(slots, 'event_name', 'event') or 'the event'
        city = self._safe_get(slots, 'city_of_event', 'city')
        date = self._safe_get(slots, 'event_date', 'date')
        
        response = f"🎫 Processing ticket purchase for {event_name}"
        if city:
            response += f" in {city}"
        if date:
            response += f" on {date}"
        return response + "."
    
    def _handle_book_appointment(self, slots: Dict[str, str]) -> str:
        provider = self._safe_get(
            slots, 'doctor_name', 'dentist_name', 'therapist_name', 
            'stylist_name', 'provider_name'
        ) or 'the provider'
        date = self._safe_get(slots, 'appointment_date', 'date')
        time = self._safe_get(slots, 'appointment_time', 'time')
        
        response = f"📅 Booking an appointment with {provider}"
        if date:
            response += f" on {date}"
        if time:
            response += f" at {time}"
        return response + "."
    
    def _handle_schedule_visit(self, slots: Dict[str, str]) -> str:
        place = self._safe_get(
            slots, 'place_name', 'attraction_name', 'location'
        ) or 'the location'
        date = self._safe_get(slots, 'visit_date', 'date')
        
        response = f"📍 Scheduling a visit to {place}"
        if date:
            response += f" on {date}"
        return response + "."
    
    def _handle_get_available_time(self, slots: Dict[str, str]) -> str:
        provider = self._safe_get(
            slots, 'doctor_name', 'dentist_name', 'provider_name'
        ) or 'the provider'
        return f"🕐 Checking available times with {provider}..."
    
    def _handle_find_bus(self, slots: Dict[str, str]) -> str:
        from_loc = self._safe_get(
            slots, 'from_location', 'from_city', 'origin_city', 'origin'
        )
        to_loc = self._safe_get(
            slots, 'to_location', 'to_city', 'destination_city', 'destination'
        )
        date = self._safe_get(slots, 'date_of_journey', 'leaving_date', 'date')
        
        response = "🚌 Searching for buses"
        if from_loc:
            response += f" from {from_loc}"
        if to_loc:
            response += f" to {to_loc}"
        if date:
            response += f" on {date}"
        return response + "..."
    
    def _handle_buy_bus_ticket(self, slots: Dict[str, str]) -> str:
        from_loc = self._safe_get(slots, 'from_location', 'from_city', 'origin')
        to_loc = self._safe_get(slots, 'to_location', 'to_city', 'destination')
        
        response = "🎫 Purchasing bus ticket"
        if from_loc and to_loc:
            response += f" from {from_loc} to {to_loc}"
        return response + "."
    
    def _handle_find_trains(self, slots: Dict[str, str]) -> str:
        from_station = self._safe_get(
            slots, 'from_station', 'origin_station_name', 'origin'
        )
        to_station = self._safe_get(
            slots, 'to_station', 'destination_station_name', 'destination'
        )
        date = self._safe_get(slots, 'date_of_journey', 'leaving_date', 'date')
        
        response = "🚆 Searching for trains"
        if from_station:
            response += f" from {from_station}"
        if to_station:
            response += f" to {to_station}"
        if date:
            response += f" on {date}"
        return response + "..."
    
    def _handle_get_train_tickets(self, slots: Dict[str, str]) -> str:
        from_station = self._safe_get(slots, 'from_station', 'origin')
        to_station = self._safe_get(slots, 'to_station', 'destination')
        
        if from_station and to_station:
            return f"🎫 Getting train tickets from {from_station} to {to_station}."
        return "🎫 Getting train tickets for your journey."
    
    def _handle_get_ride(self, slots: Dict[str, str]) -> str:
        destination = self._safe_get(slots, 'destination', 'to_location', 'to')
        pickup = self._safe_get(slots, 'pickup_location', 'from_location', 'from')
        ride_type = self._safe_get(slots, 'ride_type', 'service_type')
        
        response = f"🚗 Requesting a {ride_type + ' ' if ride_type else ''}ride"
        if pickup:
            response += f" from {pickup}"
        if destination:
            response += f" to {destination}"
        return response + "..."
    
    def _handle_get_cars_available(self, slots: Dict[str, str]) -> str:
        city = self._safe_get(slots, 'city', 'pickup_city', 'location')
        date = self._safe_get(slots, 'pickup_date', 'date')
        
        response = "🚗 Checking available cars"
        if city:
            response += f" in {city}"
        if date:
            response += f" for {date}"
        return response + "..."
    
    def _handle_reserve_car(self, slots: Dict[str, str]) -> str:
        car_name = self._safe_get(slots, 'car_name', 'car_type') or 'a car'
        pickup_date = self._safe_get(slots, 'pickup_date', 'date')
        
        response = f"🚗 Reserving {car_name}"
        if pickup_date:
            response += f" for pickup on {pickup_date}"
        return response + "."
    
    def _handle_search_oneway_flight(self, slots: Dict[str, str]) -> str:
        origin = self._safe_get(
            slots, 'origin_city', 'origin_airport', 'origin_airport_name', 'from'
        )
        dest = self._safe_get(
            slots, 'destination_city', 'destination_airport', 
            'destination_airport_name', 'to'
        )
        date = self._safe_get(slots, 'departure_date', 'leaving_date', 'date')
        
        response = "✈️ Searching for one-way flights"
        if origin:
            response += f" from {origin}"
        if dest:
            response += f" to {dest}"
        if date:
            response += f" on {date}"
        return response + "..."
    
    def _handle_search_roundtrip_flights(self, slots: Dict[str, str]) -> str:
        origin = self._safe_get(slots, 'origin_city', 'origin_airport', 'from')
        dest = self._safe_get(slots, 'destination_city', 'destination_airport', 'to')
        dep_date = self._safe_get(slots, 'departure_date', 'leaving_date')
        ret_date = self._safe_get(slots, 'return_date')
        
        response = "✈️ Searching for round-trip flights"
        if origin and dest:
            response += f" from {origin} to {dest}"
        if dep_date:
            response += f", departing {dep_date}"
        if ret_date:
            response += f", returning {ret_date}"
        return response + "..."
    
    def _handle_reserve_oneway_flight(self, slots: Dict[str, str]) -> str:
        dest = self._safe_get(
            slots, 'destination_city', 'destination_airport', 'to'
        ) or 'your destination'
        return f"✈️ Reserving one-way flight to {dest}."
    
    def _handle_reserve_roundtrip_flights(self, slots: Dict[str, str]) -> str:
        dest = self._safe_get(slots, 'destination_city', 'to') or 'your destination'
        return f"✈️ Reserving round-trip flights to {dest}."
    
    def _handle_search_hotel(self, slots: Dict[str, str]) -> str:
        city = self._safe_get(slots, 'city', 'location')
        check_in = self._safe_get(slots, 'check_in_date', 'checkin_date')
        check_out = self._safe_get(slots, 'check_out_date', 'checkout_date')
        
        response = "🏨 Searching for hotels"
        if city:
            response += f" in {city}"
        if check_in:
            response += f", check-in {check_in}"
        if check_out:
            response += f", check-out {check_out}"
        return response + "..."
    
    def _handle_reserve_hotel(self, slots: Dict[str, str]) -> str:
        hotel_name = self._safe_get(
            slots, 'hotel_name', 'property_name'
        ) or 'a hotel'
        city = self._safe_get(slots, 'city', 'location')
        check_in = self._safe_get(slots, 'check_in_date', 'date')
        
        response = f"🏨 Reserving {hotel_name}"
        if city:
            response += f" in {city}"
        if check_in:
            response += f" for {check_in}"
        return response + "."
    
    def _handle_find_apartment(self, slots: Dict[str, str]) -> str:
        city = self._safe_get(slots, 'city', 'location')
        area = self._safe_get(slots, 'area', 'neighborhood')
        
        response = "🏠 Searching for apartments"
        if city:
            response += f" in {city}"
        if area:
            response += f", {area} area"
        return response + "..."
    
    def _handle_search_house(self, slots: Dict[str, str]) -> str:
        city = self._safe_get(slots, 'city', 'location')
        area = self._safe_get(slots, 'area', 'neighborhood')
        
        response = "🏡 Searching for houses"
        if city:
            response += f" in {city}"
        if area:
            response += f", {area} area"
        return response + "..."
    
    def _handle_book_house(self, slots: Dict[str, str]) -> str:
        address = self._safe_get(
            slots, 'address', 'street_address', 'property_name'
        ) or 'the property'
        return f"🏡 Booking {address}."
    
    def _handle_find_home_by_area(self, slots: Dict[str, str]) -> str:
        area = self._safe_get(slots, 'area', 'neighborhood')
        city = self._safe_get(slots, 'city', 'location')
        
        response = "🏘️ Finding homes"
        if area:
            response += f" in the {area} area"
        if city:
            response += f" of {city}"
        return response + "..."
    
    def _handle_find_restaurants(self, slots: Dict[str, str]) -> str:
        cuisine = self._safe_get(slots, 'cuisine', 'food_type')
        city = self._safe_get(slots, 'city')
        location = self._safe_get(slots, 'location', 'area')
        
        response = "🍽️ Finding restaurants"
        if cuisine:
            response += f" serving {cuisine} cuisine"
        if city:
            response += f" in {city}"
        elif location:
            response += f" near {location}"
        return response + "..."
    
    def _handle_reserve_restaurant(self, slots: Dict[str, str]) -> str:
        restaurant = self._safe_get(slots, 'restaurant_name') or 'a restaurant'
        date = self._safe_get(slots, 'date', 'reservation_date')
        time = self._safe_get(slots, 'time', 'reservation_time')
        party_size = self._safe_get(slots, 'party_size', 'number_of_people')
        
        response = f"🍽️ Reserving a table at {restaurant}"
        if party_size:
            response += f" for {party_size} people"
        if date:
            response += f" on {date}"
        if time:
            response += f" at {time}"
        return response + "."
    
    def _handle_find_movies(self, slots: Dict[str, str]) -> str:
        genre = self._safe_get(slots, 'genre', 'movie_genre')
        movie_name = self._safe_get(slots, 'movie_name', 'movie_title', 'title')
        location = self._safe_get(slots, 'location', 'city')
        
        response = "🎬 Finding movies"
        if movie_name:
            response += f" like {movie_name}"
        elif genre:
            response += f" in the {genre} genre"
        if location:
            response += f" near {location}"
        return response + "..."
    
    def _handle_get_times_for_movie(self, slots: Dict[str, str]) -> str:
        movie = self._safe_get(
            slots, 'movie_name', 'movie_title', 'title'
        ) or 'the movie'
        theater = self._safe_get(slots, 'theater_name', 'cinema')
        
        response = f"🎬 Getting showtimes for {movie}"
        if theater:
            response += f" at {theater}"
        return response + "..."
    
    def _handle_buy_movie_tickets(self, slots: Dict[str, str]) -> str:
        movie = self._safe_get(
            slots, 'movie_name', 'movie_title', 'title'
        ) or 'the movie'
        theater = self._safe_get(slots, 'theater_name', 'cinema')
        time = self._safe_get(slots, 'show_time', 'time')
        
        response = f"🎫 Purchasing tickets for {movie}"
        if theater:
            response += f" at {theater}"
        if time:
            response += f" for the {time} show"
        return response + "."
    
    def _handle_play_movie(self, slots: Dict[str, str]) -> str:
        movie = self._safe_get(
            slots, 'movie_name', 'movie_title', 'title'
        ) or 'the movie'
        return f"▶️ Playing {movie}."
    
    def _handle_rent_movie(self, slots: Dict[str, str]) -> str:
        movie = self._safe_get(
            slots, 'movie_name', 'movie_title', 'title'
        ) or 'the movie'
        return f"🎬 Renting {movie}."
    
    def _handle_lookup_music(self, slots: Dict[str, str]) -> str:
        """
        Handle music lookup - prioritize artist > song > album > genre
        This fixes the issue where artist names were being confused with albums
        """
        artist = self._safe_get(slots, 'artist', 'artist_name')
        song = self._safe_get(slots, 'song_name', 'track', 'song', 'title')
        album = self._safe_get(slots, 'album', 'album_name')
        genre = self._safe_get(slots, 'genre', 'music_genre')
        
        response = "🎵 Looking up music"
        
        # Prioritize more specific entities
        if song and artist:
            response += f" '{song}' by {artist}"
        elif artist:
            response += f" by {artist}"
        elif song:
            response += f" '{song}'"
        elif album:
            response += f" from album {album}"
        elif genre:
            response += f" in {genre}"
        
        return response + "..."
    
    def _handle_lookup_song(self, slots: Dict[str, str]) -> str:
        song = self._safe_get(slots, 'song_name', 'track', 'title', 'song')
        artist = self._safe_get(slots, 'artist', 'artist_name')
        
        response = "🎵 Looking up"
        if song:
            response += f" '{song}'"
        if artist:
            response += f" by {artist}"
        
        if not song and not artist:
            response = "🎵 Looking up that song"
            
        return response + "..."
    
    def _handle_play_song(self, slots: Dict[str, str]) -> str:
        song = self._safe_get(slots, 'song_name', 'track', 'title') or 'the song'
        artist = self._safe_get(slots, 'artist', 'artist_name')
        
        response = f"▶️ Playing {song}"
        if artist:
            response += f" by {artist}"
        return response + "."
    
    def _handle_play_media(self, slots: Dict[str, str]) -> str:
        title = self._safe_get(
            slots, 'title', 'song_name', 'movie_name', 'media_name'
        ) or 'media'
        return f"▶️ Playing {title}."
    
    def _handle_find_attractions(self, slots: Dict[str, str]) -> str:
        city = self._safe_get(slots, 'city')
        location = self._safe_get(slots, 'location', 'area')
        attraction = self._safe_get(slots, 'attraction_name', 'attraction_type')
        
        response = "🎡 Finding attractions"
        if attraction:
            response += f" like {attraction}"
        if city:
            response += f" in {city}"
        elif location:
            response += f" near {location}"
        return response + "..."
    
    def _handle_check_balance(self, slots: Dict[str, str]) -> str:
        account = self._safe_get(slots, 'account_balance', 'balance', 'account_type')
        if account:
            return f"💰 Checking balance for {account} account."
        return "💰 Checking your account balance..."
    
    def _handle_transfer_money(self, slots: Dict[str, str]) -> str:
        amount = self._safe_get(slots, 'transfer_amount', 'amount')
        recipient = self._safe_get(
            slots, 'recipient_name', 'recipient_account_name', 'recipient'
        )
        
        response = "💸 Transferring"
        if amount:
            response += f" ${amount}"
        if recipient:
            response += f" to {recipient}"
        return response + "."
    
    def _handle_make_payment(self, slots: Dict[str, str]) -> str:
        amount = self._safe_get(slots, 'amount', 'payment_amount')
        receiver = self._safe_get(slots, 'receiver', 'recipient', 'payee')
        
        response = "💳 Processing payment"
        if amount:
            response += f" of ${amount}"
        if receiver:
            response += f" to {receiver}"
        return response + "."
    
    def _handle_request_payment(self, slots: Dict[str, str]) -> str:
        amount = self._safe_get(slots, 'amount', 'payment_amount')
        contact = self._safe_get(slots, 'contact_name', 'contact', 'payer')
        
        response = "💰 Requesting payment"
        if amount:
            response += f" of ${amount}"
        if contact:
            response += f" from {contact}"
        return response + "."
    
    def _handle_find_provider(self, slots: Dict[str, str]) -> str:
        category = self._safe_get(
            slots, 'category', 'subcategory', 'service_type'
        ) or 'service'
        city = self._safe_get(slots, 'city', 'location')
        
        response = f"🔍 Finding {category} providers"
        if city:
            response += f" in {city}"
        return response + "..."
    
    def _handle_get_weather(self, slots: Dict[str, str]) -> str:
        city = self._safe_get(slots, 'city')
        location = self._safe_get(slots, 'location', 'place')
        date = self._safe_get(slots, 'date', 'day')
        
        response = "🌤️ Getting weather"
        if city:
            response += f" for {city}"
        elif location:
            response += f" for {location}"
        if date:
            response += f" on {date}"
        else:
            response += " forecast"
        return response + "..."
    
    def _handle_share_location(self, slots: Dict[str, str]) -> str:
        contact = self._safe_get(slots, 'contact_name', 'contact', 'recipient')
        location = self._safe_get(slots, 'location', 'address', 'place')
        
        response = "📍 Sharing location"
        if location:
            response += f" ({location})"
        if contact:
            response += f" with {contact}"
        return response + "."
    
    def _handle_none(self, slots: Dict[str, str]) -> str:
        """Handle NONE intent - when no clear intent is detected"""
        return "I'm not sure what you'd like me to do. Could you please rephrase that?"
    
    def _handle_unknown(self, slots: Dict[str, str]) -> str:
        """Handle unknown/unsupported intents"""
        if self.fallback_enabled:
            return "I understand, but I'm not quite sure how to help with that yet. Can you try asking in a different way?"
        else:
            return "This intent is not currently supported."
    
    def _handle_error(self, intent: str) -> str:
        """Handle errors gracefully"""
        logger.error(f"Error occurred while processing intent: {intent}")
        return "I apologize, but I encountered an error processing your request. Please try again."


# ==================== Standalone Testing/Demo ====================

def demo_response_generator():
    """
    Demonstrate the response generator with various examples.
    Useful for testing before deployment.
    """
    generator = ResponseGenerator()
    
    # Test cases covering different scenarios
    test_cases = [
        {
            'name': 'Restaurant Search',
            'intent': 'FindRestaurants',
            'slots': [
                ('i', 'O'), ('love', 'O'), ('eating', 'O'), 
                ('italian', 'B-cuisine'), ('food', 'I-cuisine'), 
                ('in', 'O'), ('manhattan', 'B-city')
            ]
        },
        {
            'name': 'Hotel Booking',
            'intent': 'SearchHotel',
            'slots': [
                ('book', 'O'), ('hotel', 'O'), ('in', 'O'), 
                ('paris', 'B-city'), ('from', 'O'), 
                ('december', 'B-check_in_date'), ('25', 'I-check_in_date')
            ]
        },
        {
            'name': 'Ride Request',
            'intent': 'GetRide',
            'slots': [
                ('get', 'O'), ('me', 'O'), ('uber', 'O'), 
                ('to', 'O'), ('airport', 'B-destination')
            ]
        },
        {
            'name': 'Music Lookup - Fixed',
            'intent': 'LookupMusic',
            'slots': [
                ('look', 'O'), ('up', 'O'), ('for', 'O'), 
                ('a', 'O'), ('song', 'O'), ('named', 'O'), 
                ('taylor', 'B-artist'), ('swift', 'I-artist')
            ]
        },
        {
            'name': 'Play Song',
            'intent': 'PlaySong',
            'slots': [
                ('play', 'O'), ('bohemian', 'B-song_name'), 
                ('rhapsody', 'I-song_name'), ('by', 'O'), 
                ('queen', 'B-artist')
            ]
        },
        {
            'name': 'Money Transfer',
            'intent': 'TransferMoney',
            'slots': [
                ('send', 'O'), ('100', 'B-transfer_amount'), 
                ('dollars', 'I-transfer_amount'), ('to', 'O'), 
                ('john', 'B-recipient_name')
            ]
        },
        {
            'name': 'Weather Check',
            'intent': 'GetWeather',
            'slots': [
                ('what', 'O'), ('is', 'O'), ('the', 'O'), 
                ('weather', 'O'), ('in', 'O'), 
                ('new', 'B-city'), ('york', 'I-city')
            ]
        },
        {
            'name': 'Unknown Intent',
            'intent': 'SomeRandomIntent',
            'slots': [('random', 'O'), ('text', 'O')]
        }
    ]
    
    print("🤖 Response Generator Demo\n")
    print("=" * 70)
    
    for i, test_case in enumerate(test_cases, 1):
        name = test_case['name']
        intent = test_case['intent']
        slots = test_case['slots']
        
        # Generate response
        response = generator.generate_response(intent, slots)
        
        # Extract slot values for display
        slot_values = generator.extract_slot_values(slots)
        
        # Display results
        print(f"\n{i}. {name}")
        print(f"   Intent: {intent}")
        print(f"   Extracted Slots: {slot_values}")
        print(f"   Response: {response}")
        print("-" * 70)
    
    print("\n✅ Demo completed!\n")


def test_integration_with_predictor():
    """
    Example showing integration with CASA_NLU_Predictor.
    This is what you'll use in your FastAPI endpoint.
    """
    print("\n📝 Integration Example:\n")
    
    # Simulated predictor output (replace with actual predictor in production)
    mock_prediction = {
        'intent': 'LookupMusic',
        'slots': [
            ('look', 'O'), ('up', 'O'), ('taylor', 'B-artist'), ('swift', 'I-artist')
        ],
        'secondary_intent': 'NONE',
        'turn_number': 1
    }
    
    # Initialize generator
    generator = ResponseGenerator()
    
    # Generate response
    response = generator.generate_response(
        mock_prediction['intent'],
        mock_prediction['slots']
    )
    
    print(f"User: look up taylor swift")
    print(f"Intent: {mock_prediction['intent']}")
    print(f"Slots: {generator.extract_slot_values(mock_prediction['slots'])}")
    print(f"Bot: {response}")
    print()


# ==================== FastAPI Integration Helper ====================

def create_chat_response(predictor_result: dict, generator: ResponseGenerator) -> dict:
    """
    Helper function to integrate with FastAPI endpoint.
    
    Args:
        predictor_result: Output from CASA_NLU_Predictor.predict()
        generator: ResponseGenerator instance
        
    Returns:
        Dictionary with complete chat response
    """
    # Generate natural language response
    nl_response = generator.generate_response(
        predictor_result['intent'],
        predictor_result['slots']
    )
    
    # Extract structured slot information
    slot_values = generator.extract_slot_values(predictor_result['slots'])
    
    return {
        'intent': predictor_result['intent'],
        'secondary_intent': predictor_result.get('secondary_intent', 'NONE'),
        'slots': predictor_result['slots'],
        'slot_values': slot_values,  # Human-readable slot extraction
        'response': nl_response,
        'turn_number': predictor_result.get('turn_number', 0)
    }


# ==================== Main (for standalone testing) ====================

if __name__ == "__main__":
    print("Starting Response Generator Tests...\n")
    
    # Run demo
    demo_response_generator()
    
    # Test integration
    test_integration_with_predictor()
    
    print("✅ All tests completed!")