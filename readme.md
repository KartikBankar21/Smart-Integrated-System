# Smart-Integrated-System

Check out the improved Chatbot Model demonstrated at: https://smart-integrated-system.vercel.app/chatbot 

Try out with example: Book a flight from london (Check out the response slot allocation and intent)

An **AI-driven retail automation platform** combining computer vision, NLP, and a chatbot interface for streamlined billing and customer support. The system integrates three main components:

- **Vision Module (YOLOv11)** – Detects products on store shelves from camera images. We use state-of-the-art YOLO object detection models (from Ultralytics) that are fast and accurate for real-time inference【54†L311-L319】. Detected products (names, bounding boxes, confidences) are passed to the billing engine.  
- **Document Analysis Module (LayoutLMv3)** – Processes scanned or photographed invoices. Using Microsoft’s LayoutLMv3 (a multimodal transformer) it extracts line-item details (descriptions, quantities, prices) from invoice images. The extracted text is matched against the product database to compute the bill total.  
- **Billing Engine** – A backend service that takes product detections and invoice data, looks up pricing in the **Product Database**, and calculates totals. The engine merges vision-based detections with invoice information into unified billing records (e.g. receipts or expense reports). A common API layer unifies these components for end-to-end processing.  

Additionally, an **NLU-powered Chatbot** provides a conversational interface. The chatbot runs independently, handling text-based customer queries (e.g. product search, price lookup, order history). It uses a context-aware intent-and-slot model (CASA-NLU Improved) for multi-turn dialogue. Both billing and chatbot services share the same product inventory/database, so users can query the system naturally. 

## Key Features

- **Product Detection:** Real-time object detection of retail items on shelves using YOLO models.  
- **Invoice Extraction:** OCR and field extraction from invoices with LayoutLMv3.  
- **Automated Billing:** Matches detected and extracted items to prices in a central database, outputs detailed bills/invoices.  
- **Conversational Chatbot:** Multi-turn customer chatbot with intent recognition and dialogue management. FastAPI backend provides a **/chat** endpoint; frontend (Next.js/React) hosts the chat UI.  
- **Web Dashboard:** Administrative UI with metrics, model status, and history (see screenshots below).  

## System Architecture

The architecture consists of a **Billing Pipeline** and a **Chatbot Service**, as shown in the diagram. Images (shelf or invoice) are fed into the Vision and Document modules in parallel; both feed into the Billing Engine along with the Product DB. The Chatbot module interacts with the user separately, querying the same database for information or triggering billing actions as needed.

![High Level Design of Project](https://github.com/KartikBankar21/Smart-Integrated-System/blob/main/screenshots/high_level_design.png)

## Installation & Usage

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/KartikBankar21/Smart-Integrated-System.git
   cd Smart-Integrated-System
   ```

2. **Install dependencies:**  
   Set up a Python environment (Python ≥3.8) and install required packages (PyTorch, Transformers, FastAPI, etc.) using `pip install -r requirements.txt`. The vision module uses the `ultralytics` YOLO package, and the document module uses Hugging Face Transformers for LayoutLMv3.  

3. **Database:**  
   Initialize or connect to the product database (e.g. PostgreSQL or SQLite). The `product_db.sql` script (if provided) sets up the inventory table.

4. **Run services:**  
   - Start the **Billing API** (FastAPI) and **Chatbot API** (FastAPI) services (e.g. `uvicorn backend.billing:app` and `uvicorn backend.chatbot:app`).  
   - Launch the frontend (Next.js/React) with `npm start` or `yarn dev`. This will provide web pages for the dashboard and chatbot UI.  

5. **Access the app:**  
   Open a browser at `http://localhost:3000` (or your configured host) to use the integrated dashboard. The chatbot interface is typically at `http://localhost:3000/chat`.  

*Example:* Upload a shelf photo to see detected products, or upload an invoice image to see extracted line items and totals. Ask the chatbot questions like “What is the price of Product123?” or “Show me my last invoice,” and it will respond based on the latest data.

## Technologies Used

- **Computer Vision:** YOLOv8 (Ultralytics) for object detection.  
- **NLP & OCR:** Microsoft LayoutLMv3 Transformer for invoice understanding.  
- **Backend API:** FastAPI (Python) for RESTful services.  
- **Frontend:** Next.js and Tailwind CSS for the web dashboard and chatbot UI.  
- **Database:** (e.g.) PostgreSQL/MySQL for product and invoice data.  
- **Deployment:** Dockerized services, deployed on cloud (e.g. Render.com).

## Screenshots

![Dashboard view of the integrated billing and chatbot system](https://github.com/KartikBankar21/Smart-Integrated-System/blob/main/screenshots/home_dashboard.png)

*Home dashboard with links to billing history and chatbot.*

![Invoice processing view](https://github.com/KartikBankar21/Smart-Integrated-System/blob/main/screenshots/invoice_extraction_viewer.png)

*Invoice viewer showing extracted line items and totals (LayoutLMv3 output).*

![Chatbot interaction](https://github.com/KartikBankar21/Smart-Integrated-System/blob/main/screenshots/chatbot_viewer.png)

*Chatbot UI demonstrating multi-turn conversation with context tracking (CASA-NLU).*

---

# chatbot-backend

This repository contains the **FastAPI backend** service for the context-aware chatbot used in the Smart-Integrated-System. It provides a REST API that handles user queries (intents and slots) and returns natural language responses. 

Key aspects of the chatbot backend:

- **FastAPI Framework:** A modern, high-performance Python API framework. FastAPI is based on Python type hints and provides automatic interactive documentation (Swagger UI).  
- **NLU Model:** Implements the CASA-NLU model (Context-Aware Self-attentive NLU) for intent classification and slot filling. The model maintains context over multiple dialogue turns.  
- **Endpoints:** The main endpoint is `/chat`, which accepts POST requests with JSON payloads (`{"message": "..."}`) and returns chatbot responses. (Details in `app/main.py`.)  
- **Session Management:** Uses in-memory or Redis store to track dialogue state per user session.  
- **Integration:** The chatbot can query the product/invoice database to answer user questions (e.g. price lookup, invoice retrieval).  

## Installation

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/KartikBankar21/chatbot-backend.git
   cd chatbot-backend
   ```

2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```  
   Key packages include `fastapi`, `uvicorn`, and NLP libraries (e.g. PyTorch, Transformers).

3. **Run the server:**  
   ```bash
   uvicorn main:app --reload
   ```  
   The API will be available at `http://localhost:8000`. FastAPI’s interactive docs are at `http://localhost:8000/docs`.

4. **Deploy:**  
   This service can be containerized with Docker. (A `Dockerfile` is provided.) It’s already deployed on Render.com at `https://<your-service>.onrender.com`.

## Usage

- Send a POST request to `/chat` with JSON body:  
  ```json
  {"user_id": "user123", "message": "When does Item A expire?"}
  ```  
- The response will be a JSON object with the chatbot’s reply:  
  ```json
  {"reply": "Item A expires on December 31, 2026.", "intent": "query_expiry", "slots": {...}}
  ```

The chatbot backend logs each turn of the conversation and handles multi-turn context. It returns structured data (intent/slot) as well as the final user-facing text.

## Tech Stack & Credits

- **FastAPI** (Python) – high-performance API framework.  
- **Uvicorn** – ASGI server for running FastAPI.  
- **PyTorch/Transformers** – for loading the CASA-NLU model.  
- **Render.com** – Cloud platform used for deployment (free tier).  

## Challenges

Due to the significant computational requirements of models like LayoutLMv3 and the lack of free-tier cloud services capable of hosting them, the billing system was not deployed to a live environment. The system currently remains a local prototype due to these hardware and infrastructure constraints.
